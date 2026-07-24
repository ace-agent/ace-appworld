"""
adaptation_react_parallel.py

Parallel batch version of the SimplifiedReActStarAgent.

Uses AppWorld's **decoupled mode**: manually starts N remote environment
server subprocesses (``appworld serve environment --port ...``), then
uses ThreadPoolExecutor so each task in a batch runs in its own thread
against a dedicated server.

This avoids the SQLite / freezegun conflicts of running multiple worlds
in the same process (unified mode) while being lighter than full
multiprocessing (no pickling, no agent recreation overhead).

After all threads finish, the main thread collects results and runs
batch curation.
"""

import atexit
import copy
import json
import random
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import requests
from appworld import AppWorld
from appworld.common.constants import DEFAULT_EXPERIMENT_NAME
from appworld.common.path_store import path_store
from appworld.common.utils import chunk_and_return

from appworld_experiments.code.ace.adaptation_agent import StarAgent, ExecutionIO
from appworld_experiments.code.ace.adaptation_react import SimplifiedReActStarAgent
from appworld_experiments.code.ace.cost_tracker import CostTracker
from appworld_experiments.code.ace.logger import Logger
from .bulletpoint_analyzer import BulletpointAnalyzer, DEDUP_AVAILABLE
from .playbook import apply_curator_operations, extract_json_from_text
from .utils_compat import count_tokens, is_context_length_api_exception


@dataclass
class BatchTaskResult:
    """Stores the result of a single task within a batch for deferred curation."""
    task_id: str
    task_success: bool = False
    should_curate: bool = False
    trimmed_messages: list[dict] = field(default_factory=list)
    gt_code: str | None = None
    test_report: str | None = None
    question_context: str = ""


# ──────────────────────────────────────────────────────────────────────
#  Server management helpers
# ──────────────────────────────────────────────────────────────────────

def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _start_environment_servers(
    num_servers: int,
    root_dir: str | None = None,
    startup_timeout: float = 120.0,
    poll_interval: float = 2.0,
) -> tuple[list[subprocess.Popen], list[str]]:
    """
    Start ``num_servers`` AppWorld environment server subprocesses.

    Returns:
        (processes, urls) — list of Popen objects and their base URLs.
    """
    if root_dir is None:
        root_dir = os.path.abspath(path_store.root)

    processes: list[subprocess.Popen] = []
    urls: list[str] = []
    log_files: list[Any] = []

    for idx in range(num_servers):
        port = _find_free_port()
        url = f"http://localhost:{port}"

        # Capture stderr to a log file for diagnostics
        log_path = f"/tmp/appworld_server_{idx}_{port}.log"
        log_f = open(log_path, "w")
        log_files.append((log_f, log_path))

        # Use the appworld console script from the same venv as the
        # running interpreter (avoids "No module named appworld.__main__")
        appworld_bin = os.path.join(
            os.path.dirname(sys.executable), "appworld"
        )
        proc = subprocess.Popen(
            [
                appworld_bin, "serve", "environment",
                "--port", str(port),
                "--no-show-usage",
                "--root", root_dir,
            ],
            stdout=log_f,
            stderr=log_f,
        )
        processes.append(proc)
        urls.append(url)

    # Wait for all servers to become healthy
    deadline = time.time() + startup_timeout
    for i, url in enumerate(urls):
        while time.time() < deadline:
            # Check if process has died
            if processes[i].poll() is not None:
                log_f, log_path = log_files[i]
                log_f.flush()
                try:
                    with open(log_path, "r") as f:
                        err_output = f.read()
                except Exception:
                    err_output = "(could not read log)"
                _stop_servers(processes)
                for lf, _ in log_files:
                    lf.close()
                raise RuntimeError(
                    f"Server {i} at {url} exited with code "
                    f"{processes[i].returncode}.\n"
                    f"Server log ({log_path}):\n{err_output}"
                )
            try:
                resp = requests.get(f"{url}/", timeout=2)
                if resp.status_code == 200:
                    break
            except (requests.ConnectionError, requests.Timeout):
                pass
            time.sleep(poll_interval)
        else:
            _stop_servers(processes)
            for lf, _ in log_files:
                lf.close()
            raise TimeoutError(
                f"Server {i} at {url} did not become ready within "
                f"{startup_timeout}s"
            )

    # Close log file handles (servers are running fine now)
    for lf, _ in log_files:
        lf.close()

    return processes, urls


def _stop_servers(processes: list[subprocess.Popen]) -> None:
    """Gracefully stop all server subprocesses."""
    for proc in processes:
        if proc.poll() is None:  # still running
            proc.terminate()
    # Give them a moment, then force-kill stragglers
    for proc in processes:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# ──────────────────────────────────────────────────────────────────────
#  Thread-level worker functions (decoupled mode — each thread talks
#  to its own remote AppWorld server via HTTP)
# ──────────────────────────────────────────────────────────────────────

def _run_task_in_thread(
    task_id: str,
    frozen_playbook: str,
    agent_init_kwargs: dict,
    use_gt_code: bool,
    appworld_kwargs: dict,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> BatchTaskResult:
    """
    Worker function executed in a thread.

    Creates a fresh agent, runs one task against a *remote* AppWorld
    server (URL is baked into ``appworld_kwargs``), and returns the
    BatchTaskResult.
    """
    agent = SimplifiedReActStarAgent(**agent_init_kwargs)
    agent.playbook = frozen_playbook

    # Initialize the logger so complete_task() doesn't hit None counters.
    # We set num_tasks=1 since each worker handles exactly one task.
    agent.logger.initialize(
        experiment_name=experiment_name,
        num_tasks=1,
        num_processes=1,
        process_index=0,
    )

    result = BatchTaskResult(task_id=task_id)

    if use_gt_code:
        _worker_run_with_gt(agent, task_id, appworld_kwargs, result)
    else:
        _worker_run_wo_gt(agent, task_id, appworld_kwargs, result)

    return result


def _worker_run_with_gt(agent, task_id, appworld_kwargs, result):
    """Run a single task with GT code inside a worker thread."""
    reasoning_text = ""

    for retry_id in range(agent.num_retries):
        with AppWorld(task_id=task_id, **appworld_kwargs) as world:
            execution_outputs: list[ExecutionIO] = []
            agent.initialize(world)

            try:
                gt_code = (
                    world.task.ground_truth
                    .load(task_id, mode="full")
                    .compiled_solution_code
                )
            except Exception:
                raise ValueError(f"GT code not found for task: {task_id}")

            result.gt_code = gt_code
            result.question_context = getattr(world.task, "instruction", "")

            print(f"[{task_id}] retry={retry_id}  max_steps={agent.max_steps}")

            agent.step_number = 0
            for _ in range(agent.max_steps):
                agent.step_number += 1

                if agent.step_number == 1:
                    execution_inputs, cost, _ = (
                        agent.next_execution_inputs_and_cost(
                            execution_outputs, gt_code, reasoning_text
                        )
                    )
                else:
                    execution_inputs, cost, _ = (
                        agent.next_execution_inputs_and_cost(
                            execution_outputs, gt_code, ""
                        )
                    )

                if len(execution_inputs) != 0:
                    execution_outputs = [
                        ExecutionIO(
                            content=world.execute(ei.content),
                            metadata=ei.metadata,
                        )
                        for ei in execution_inputs
                    ]
                    for output in execution_outputs:
                        if output.content.strip():
                            agent.logger.show_message(
                                role="environment",
                                message=output.content,
                                step_number=agent.step_number,
                            )

                agent.cost_tracker.add(task_id, cost)
                agent.log_cost()

                if world.task_completed() or agent.cost_tracker.exceeded():
                    # Use world.evaluate() instead of evaluate_task() to
                    # avoid SQLite thread-safety issues — in decoupled mode
                    # this delegates evaluation to the remote server via HTTP.
                    test_tracker = world.evaluate()
                    agent.test_report = test_tracker.report(
                        print_it=False, colorize=False
                    )
                    result.test_report = (
                        str(agent.test_report) if agent.test_report else None
                    )
                    result.should_curate = True

                    if len(test_tracker.failures) > 0:
                        reasoning_text = agent.reflector_call()
                    else:
                        result.task_success = True
                        print(
                            f"  {task_id} passed in retry {retry_id}, "
                            f"step {agent.step_number}"
                        )
                    break

            # When we exhaust steps without task_completed/cost_exceeded,
            # still curate: evaluate and set should_curate so every task contributes.
            if not result.should_curate:
                try:
                    test_tracker = world.evaluate()
                    agent.test_report = test_tracker.report(
                        print_it=False, colorize=False
                    )
                    result.test_report = (
                        str(agent.test_report) if agent.test_report else None
                    )
                except Exception:
                    result.test_report = None
                result.should_curate = True

            # Capture conversation for later curation
            try:
                result.trimmed_messages = copy.deepcopy(agent.trimmed_messages)
            except Exception:
                result.trimmed_messages = copy.deepcopy(agent.messages)

        if result.task_success:
            break

    agent.logger.complete_task()


def _worker_run_wo_gt(agent, task_id, appworld_kwargs, result):
    """Run a single task without GT code inside a worker thread."""
    with AppWorld(task_id=task_id, **appworld_kwargs) as world:
        execution_outputs: list[ExecutionIO] = []
        agent.initialize(world)

        result.question_context = getattr(world.task, "instruction", "")

        print(f"[{task_id}] max_steps={agent.max_steps}")

        for _ in range(agent.max_steps):
            agent.step_number += 1
            execution_inputs, cost, _ = (
                agent.next_execution_inputs_and_cost(execution_outputs, None)
            )

            if len(execution_inputs) != 0:
                execution_outputs = [
                    ExecutionIO(
                        content=world.execute(ei.content),
                        metadata=ei.metadata,
                    )
                    for ei in execution_inputs
                ]
                for output in execution_outputs:
                    if output.content.strip():
                        agent.logger.show_message(
                            role="environment",
                            message=output.content,
                            step_number=agent.step_number,
                        )

            agent.cost_tracker.add(task_id, cost)
            agent.log_cost()

            if world.task_completed() or agent.cost_tracker.exceeded():
                # Use world.evaluate() instead of evaluate_task() to
                # avoid SQLite thread-safety issues — in decoupled mode
                # this delegates evaluation to the remote server via HTTP.
                test_tracker = world.evaluate()
                agent.test_report = test_tracker.report(
                    print_it=False, colorize=False
                )
                result.test_report = (
                    str(agent.test_report) if agent.test_report else None
                )
                result.should_curate = True
                result.task_success = len(test_tracker.failures) == 0
                break

        # When we exhaust steps without task_completed/cost_exceeded,
        # still curate: evaluate and set should_curate so every task contributes.
        if not result.should_curate:
            try:
                test_tracker = world.evaluate()
                agent.test_report = test_tracker.report(
                    print_it=False, colorize=False
                )
                result.test_report = (
                    str(agent.test_report) if agent.test_report else None
                )
            except Exception:
                result.test_report = None
            result.should_curate = True

        # Capture conversation for later curation
        try:
            result.trimmed_messages = copy.deepcopy(agent.trimmed_messages)
        except Exception:
            result.trimmed_messages = copy.deepcopy(agent.messages)

    agent.logger.complete_task()


# ──────────────────────────────────────────────────────────────────────
#  Main agent class (runs in the parent process / main thread)
# ──────────────────────────────────────────────────────────────────────

@StarAgent.register("ace_adaptation_react_parallel")
class ParallelReActStarAgent(SimplifiedReActStarAgent):
    """
    Parallel batch version of SimplifiedReActStarAgent.

    Uses AppWorld's **decoupled mode**: manually starts B remote
    environment server subprocesses, then uses ``ThreadPoolExecutor``
    so each task in a batch runs in its own thread against a dedicated
    server.

    Workflow with batch_size B (same curator logic as ace_batch.py):
      1. Start B ``appworld serve environment`` subprocesses on free ports.
      2. For each batch of B tasks:
         a. Freeze the current playbook.
         b. Spawn B worker threads, each creating its own agent and
            talking to a dedicated remote AppWorld server.
         c. After all B threads finish, collect ``BatchTaskResult``s.
         d. Chunk reflections by curator_batch_size (default: same as batch_size).
         e. For each chunk: combine reflections and run curator once.
            - curator_parallel=True (default): all chunk curator LLM calls run
              against a frozen playbook snapshot, then ADD ops are merged and
              applied once (ace_batch curator_parallel parity).
            - curator_parallel=False: serial curator; each chunk updates the
              live playbook so later chunks see earlier ADDs.
            - Optional use_bulletpoint_analyzer: after applying curator ops,
              embedding+LLM-merge similar playbook bullets.
      3. Stop all server subprocesses on exit.

    Falls back to the sequential parent when batch_size <= 1.
    """

    def __init__(
        self,
        curator_batch_size: int | None = None,
        augmented_shuffling_factor: int = 1,
        curator_parallel: bool = True,
        curator_max_workers: int | None = None,
        use_bulletpoint_analyzer: bool = False,
        bulletpoint_analyzer_threshold: float = 0.90,
        **kwargs: Any,
    ):
        # Store the raw init kwargs so we can recreate agents in worker threads
        kwargs = dict(kwargs)
        self.curator_batch_size = kwargs.pop("curator_batch_size", curator_batch_size)
        self.augmented_shuffling_factor = kwargs.pop(
            "augmented_shuffling_factor", augmented_shuffling_factor
        )
        self.curator_parallel = bool(
            kwargs.pop("curator_parallel", curator_parallel)
        )
        max_workers = kwargs.pop("curator_max_workers", curator_max_workers)
        self.curator_max_workers = int(max_workers) if max_workers is not None else None
        self.use_bulletpoint_analyzer = bool(
            kwargs.pop("use_bulletpoint_analyzer", use_bulletpoint_analyzer)
        )
        self.bulletpoint_analyzer_threshold = float(
            kwargs.pop(
                "bulletpoint_analyzer_threshold", bulletpoint_analyzer_threshold
            )
        )
        self._init_kwargs = dict(kwargs)
        super().__init__(**kwargs)
        self.bulletpoint_analyzer: BulletpointAnalyzer | None = None
        if self.use_bulletpoint_analyzer:
            if DEDUP_AVAILABLE:
                self.bulletpoint_analyzer = BulletpointAnalyzer(self.curator_model)
                print(
                    f"✓ BulletpointAnalyzer initialized "
                    f"(threshold={self.bulletpoint_analyzer_threshold})"
                )
            else:
                print(
                    "⚠️  use_bulletpoint_analyzer=True but sentence-transformers/faiss "
                    "unavailable; LLM playbook merge disabled"
                )

    def solve_tasks(
        self,
        task_ids: list[str],
        experiment_name: str | None = None,
        num_processes: int = 1,
        process_index: int = 0,
        batch_size: int | None = None,
    ):
        # Inject batch size into the playbook filename
        # e.g. "playbook.txt" → "playbook_bs5.txt"
        if batch_size is not None and self.trained_playbook_file_path:
            base, ext = os.path.splitext(self.trained_playbook_file_path)
            self.trained_playbook_file_path = f"{base}_bs{batch_size}{ext}"
            print(f"Playbook will be saved to: {self.trained_playbook_file_path}")

        # Fall back to sequential processing when batch_size is not set or <= 1
        if batch_size is None or batch_size <= 1:
            return super().solve_tasks(
                task_ids, experiment_name, num_processes, process_index, batch_size
            )

        experiment_name = experiment_name or DEFAULT_EXPERIMENT_NAME
        num_tasks = len(task_ids)
        num_processes = min(num_processes, num_tasks)
        task_ids = chunk_and_return(
            task_ids, num_chunks=num_processes, chunk_index=process_index
        )

        self.logger.initialize(
            experiment_name=experiment_name,
            num_tasks=num_tasks,
            num_processes=num_processes,
            process_index=process_index,
        )

        if self.log_lm_calls:
            phase2_lm_path = os.path.join(
                path_store.experiment_outputs,
                experiment_name,
                "phase2_llm_logs",
                "lm_calls.jsonl",
            )
            os.makedirs(os.path.dirname(phase2_lm_path), exist_ok=True)
            self.reflector_model.log_calls_to(file_path=phase2_lm_path)
            self.curator_model.log_calls_to(file_path=phase2_lm_path)

        total_batches = (len(task_ids) + batch_size - 1) // batch_size
        _cbs = self.curator_batch_size if self.curator_batch_size is not None else batch_size
        _workers = (
            f", max_workers={self.curator_max_workers}"
            if self.curator_parallel and self.curator_max_workers is not None
            else ""
        )
        print(
            f"Gen batch size: {batch_size} | Curator batch size: {_cbs} "
            f"(None -> same as gen batch) | curator_parallel={self.curator_parallel}{_workers}"
        )

        # ── Start decoupled AppWorld environment servers ──────────────
        print(f"\nStarting {batch_size} remote AppWorld environment servers...")
        server_procs, server_urls = _start_environment_servers(
            num_servers=batch_size,
        )
        # Ensure servers are cleaned up on unexpected exit
        atexit.register(_stop_servers, server_procs)

        print(f"  {len(server_urls)} servers ready: {server_urls}")

        # Build per-server config dicts (same shape as what AppWorld() expects)
        server_configs = []
        for url in server_urls:
            cfg = {
                "experiment_name": experiment_name,
                "remote_environment_url": url,
                **{k: v for k, v in self.appworld_config.items()
                   if k not in ("remote_environment_url",)},
            }
            server_configs.append(cfg)

        try:
            for batch_start in range(0, len(task_ids), batch_size):
                batch = task_ids[batch_start : batch_start + batch_size]
                batch_num = batch_start // batch_size + 1

                print(f"\n{'=' * 60}")
                print(
                    f"Batch {batch_num}/{total_batches}: "
                    f"tasks {batch_start + 1}-{batch_start + len(batch)} "
                    f"of {len(task_ids)}"
                )
                print(f"Current playbook size: {len(self.playbook)} chars")
                print(f"{'=' * 60}")

                # Freeze the playbook so every task in this batch sees the
                # same version
                frozen_playbook = self.playbook

                # ── Phase 1: Solve tasks in parallel threads ──────────
                batch_results: list[BatchTaskResult] = []

                with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                    future_to_task: dict = {}
                    for i, task_id in enumerate(batch):
                        # Round-robin assignment of tasks to servers
                        worker_config = server_configs[i % len(server_configs)]
                        future = executor.submit(
                            _run_task_in_thread,
                            task_id=task_id,
                            frozen_playbook=frozen_playbook,
                            agent_init_kwargs=self._init_kwargs,
                            use_gt_code=self.use_gt_code,
                            appworld_kwargs=worker_config,
                            experiment_name=experiment_name,
                        )
                        future_to_task[future] = task_id

                    for future in as_completed(future_to_task):
                        task_id = future_to_task[future]
                        try:
                            result = future.result()
                            batch_results.append(result)
                            status = "PASS" if result.task_success else "FAIL"
                            print(f"  [{status}] {task_id}")
                        except Exception as exc:
                            print(f"  [ERROR] {task_id}: {exc}")
                            traceback.print_exc()
                            batch_results.append(
                                BatchTaskResult(task_id=task_id)
                            )

                # ── Phase 2: Batch curation (main thread) ─────────────
                self._batch_curator_call(batch_results, generator_batch_size=batch_size)

                # House-keeping
                self.current_task_index = batch_start + len(batch) - 1
                if (self.current_task_index + 1) % 30 == 0:
                    self.save_playbook_snapshot()

        finally:
            # Always stop servers, even on error
            print("\nStopping AppWorld environment servers...")
            _stop_servers(server_procs)
            print("  Servers stopped")

    # ──────────────────────────────────────────────────────────────────
    #  Batch curation
    #  Chunk reflections by curator_batch_size.
    #  Serial: each chunk curator updates the live playbook.
    #  Parallel (curator_parallel): snapshot + concurrent chunks + merge ops once.
    # ──────────────────────────────────────────────────────────────────

    def _phase2_maybe_merge_playbook(self) -> None:
        """Optional embedding + LLM merge of similar bullets (ace_batch parity)."""
        if not self.use_bulletpoint_analyzer or self.bulletpoint_analyzer is None:
            return
        print(
            f"  Running BulletpointAnalyzer "
            f"(threshold={self.bulletpoint_analyzer_threshold})..."
        )
        try:
            self.playbook = self.bulletpoint_analyzer.analyze(
                playbook=self.playbook,
                threshold=self.bulletpoint_analyzer_threshold,
                merge=True,
            )
        except Exception as e:
            print(f"  ⚠️  BulletpointAnalyzer failed: {e}; keeping unmerged playbook")
            traceback.print_exc()

    def _phase2_save_playbook(self) -> None:
        if self.trained_playbook_file_path:
            try:
                with open(self.trained_playbook_file_path, "w") as f:
                    f.write(self.playbook)
                print(
                    f"  Playbook saved to {self.trained_playbook_file_path} "
                    f"({len(self.playbook)} chars)"
                )
            except Exception as e:
                print(f"  ERROR writing playbook: {e}")
                traceback.print_exc()

    def _batch_curator_call(
        self, batch_results: list[BatchTaskResult], *, generator_batch_size: int
    ) -> None:
        """Phase 2: chunk by curator_batch_size; serial or parallel curator (ace_batch parity)."""
        results_to_curate = [
            r for r in batch_results
            if r.should_curate and r.trimmed_messages
        ]
        if not results_to_curate:
            print("\nNo tasks with curation context to process")
            return

        augmented_factor = self.augmented_shuffling_factor
        if augmented_factor > 1:
            orig_count = len(results_to_curate)
            augmented = [r for r in results_to_curate for _ in range(augmented_factor)]
            random.shuffle(augmented)
            results_to_curate = augmented
            print(
                f"  [Augmented Shuffling] factor={augmented_factor} | "
                f"{orig_count} -> {len(results_to_curate)} after augmentation"
            )

        curator_batch_size = self.curator_batch_size
        if curator_batch_size is None:
            curator_batch_size = generator_batch_size
        curator_batch_size = max(1, int(curator_batch_size))
        num_chunks = (len(results_to_curate) + curator_batch_size - 1) // curator_batch_size

        chunk_specs: list[tuple[int, int, int, list[BatchTaskResult]]] = []
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * curator_batch_size
            end_idx = min(start_idx + curator_batch_size, len(results_to_curate))
            chunk_specs.append(
                (chunk_idx, start_idx, end_idx, results_to_curate[start_idx:end_idx])
            )

        print(
            f"\nPHASE 2: Aggregation + Curator (curator_batch_size={curator_batch_size})"
        )

        if self.curator_parallel and num_chunks > 1:
            # ===========================================================
            # PARALLEL Curator: snapshot + concurrent chunks + central merge
            # ===========================================================
            snapshot_playbook = self.playbook
            snapshot_next_id = self.next_global_id
            max_workers_cfg = self.curator_max_workers or num_chunks
            max_workers_cfg = max(1, min(int(max_workers_cfg), num_chunks, 32))
            print(
                f"  Standard (PARALLEL): chunk by curator_batch_size={curator_batch_size} "
                f"({len(results_to_curate)} samples, {num_chunks} chunks, "
                f"max_workers={max_workers_cfg})"
            )

            def _run_curator_chunk(
                chunk_idx: int,
                start_idx: int,
                end_idx: int,
                chunk_results: list[BatchTaskResult],
            ) -> tuple[int, int, int, list[dict]]:
                ops = self._generate_curator_operations_for_chunk(
                    chunk_results,
                    playbook_snapshot=snapshot_playbook,
                    chunk_idx=chunk_idx,
                    num_chunks=num_chunks,
                )
                return chunk_idx, start_idx, end_idx, ops or []

            chunk_ops_results: list[list[dict]] = [[] for _ in range(num_chunks)]
            with ThreadPoolExecutor(max_workers=max_workers_cfg) as executor:
                futures = {
                    executor.submit(_run_curator_chunk, *spec): spec[0]
                    for spec in chunk_specs
                }
                for fut in as_completed(futures):
                    chunk_idx = futures[fut]
                    try:
                        idx, start_idx, end_idx, ops = fut.result()
                        chunk_ops_results[idx] = ops
                        print(
                            f"  Curator chunk {idx + 1}/{num_chunks} complete "
                            f"(samples {start_idx + 1}-{end_idx}, {len(ops)} operations)"
                        )
                    except Exception as e:
                        print(
                            f"  [Phase2] Curator chunk {chunk_idx + 1}/{num_chunks} "
                            f"failed: {e} - treating as 0 operations, continuing"
                        )
                        traceback.print_exc()
                        chunk_ops_results[chunk_idx] = []

            merged_ops: list[dict] = []
            for ops in chunk_ops_results:
                merged_ops.extend(ops)
            self.playbook, self.next_global_id = apply_curator_operations(
                snapshot_playbook, merged_ops, snapshot_next_id
            )
            print(
                f"  Applied merged curator operations once: "
                f"{len(merged_ops)} operations total"
            )
            self._phase2_maybe_merge_playbook()
            print(
                f"\n  Playbook updated after {num_chunks} parallel Curator calls"
            )
        else:
            # ===========================================================
            # SERIAL Curator (each chunk sees prior playbook updates)
            # ===========================================================
            print(
                f"  Standard: chunk by curator_batch_size={curator_batch_size} "
                f"({len(results_to_curate)} samples)"
            )
            print(
                f"  Running Curator {num_chunks} times (each with up to "
                f"{curator_batch_size} samples)"
            )
            for chunk_idx, start_idx, end_idx, chunk_results in chunk_specs:
                print(
                    f"\n--- Curator chunk {chunk_idx + 1}/{num_chunks} "
                    f"(samples {start_idx + 1}-{end_idx}) ---"
                )
                try:
                    ops = self._generate_curator_operations_for_chunk(
                        chunk_results,
                        playbook_snapshot=None,
                        chunk_idx=chunk_idx,
                        num_chunks=num_chunks,
                    )
                except Exception as e:
                    print(
                        f"  [Phase2] Curator chunk {chunk_idx + 1}/{num_chunks} "
                        f"failed: {e} - skipping chunk, playbook unchanged, continuing"
                    )
                    traceback.print_exc()
                    continue
                if ops:
                    print(f"  Chunk {chunk_idx + 1}: {len(ops)} curator operations")
                    self.playbook, self.next_global_id = apply_curator_operations(
                        self.playbook, ops, self.next_global_id
                    )
                else:
                    print(f"  Chunk {chunk_idx + 1}: 0 curator operations")
            self._phase2_maybe_merge_playbook()
            print(f"\n  Playbook updated after {num_chunks} Curator calls")

        self._phase2_save_playbook()

    def _generate_curator_operations_for_chunk(
        self,
        chunk_results: list[BatchTaskResult],
        *,
        playbook_snapshot: str | None = None,
        guidebook_override: str | None = None,
        chunk_idx: int | None = None,
        num_chunks: int | None = None,
    ) -> list[dict]:
        """
        Generate curator ADD operations for a chunk of tasks (ace_batch style).
        Combines reflections from multiple tasks into one curator call.
        If playbook_snapshot is set, curator/reflector prompts use it (for parallel
        chunk calls that share one Phase-2 playbook snapshot).
        If guidebook_override is set, skip per-task reflector calls and use it as guidebook.
        """
        if not chunk_results:
            return []

        pb = self.playbook if playbook_snapshot is None else playbook_snapshot

        # Build combined content from all tasks in chunk
        combined_question_contexts = []
        combined_conversations = []
        combined_guidebooks: list[str] = []

        for i, result in enumerate(chunk_results):
            sample_label = f"[Sample {i + 1}]"
            combined_question_contexts.append(
                f"{sample_label} {result.question_context or '(no context)'}"
            )

            conversation_history = "\n".join(
                f"{msg.get('role', 'unknown').upper()}: {msg.get('content', '')}"
                for msg in result.trimmed_messages
            )
            combined_conversations.append(
                f"{sample_label}\n{conversation_history}"
            )

            if guidebook_override is None:
                reasoning_text = None
                if self.use_reflector:
                    reasoning_text = self._generate_reflection_for_task(
                        result,
                        playbook_override=pb,
                        parallel_chunk_idx=chunk_idx,
                        parallel_num_chunks=num_chunks,
                    )
                combined_guidebooks.append(
                    f"{sample_label} {reasoning_text or 'N/A'}"
                )

        combined_question_context = "\n\n---\n\n".join(combined_question_contexts)
        combined_conversation = "\n\n---\n\n".join(combined_conversations)
        if guidebook_override is not None:
            combined_guidebook = guidebook_override
        else:
            combined_guidebook = "\n\n---\n\n".join(combined_guidebooks)

        # Build curator prompt with combined multi-sample content
        content = self.curator_prompt.format(
            initial_generated_code="See full conversation history below",
            final_generated_code="See full conversation history below",
            guidebook=combined_guidebook,
            current_playbook=pb,
            question_context=combined_question_context,
            gt="(Multiple tasks - see conversation history)",
        )
        content += "\n\n=== FULL CONVERSATION HISTORY (multiple tasks) ===\n"
        content += combined_conversation

        # [DIAG] Log Curator input size per chunk (ace-batch style)
        try:
            cb_tokens = count_tokens(combined_guidebook)
            cq_tokens = count_tokens(combined_question_context)
            cc_tokens = count_tokens(combined_conversation)
            pb_tokens = count_tokens(pb)
            total_tokens = cb_tokens + cq_tokens + cc_tokens + pb_tokens
            print(
                f"  [DIAG] curator_chunk_size={len(chunk_results)} | "
                f"guidebook={cb_tokens} tok | context={cq_tokens} tok | "
                f"conversation={cc_tokens} tok | playbook={pb_tokens} tok | "
                f"total~{total_tokens} tok"
            )
        except Exception:
            pass

        # Call curator LLM once for the entire chunk
        try:
            curator_raw = self.curator_model.generate(
                messages=[{"role": "user", "content": content}],
                extra_log_fields={
                    "ace_role": "curator",
                    "ace_stage": "phase2_batch",
                    "chunk_idx": chunk_idx,
                    "chunks_total": num_chunks,
                },
            )
        except Exception as e:
            if is_context_length_api_exception(e):
                print(
                    f"  [Curator] Context length exceeded; skipping playbook update for this chunk "
                    f"({len(chunk_results)} samples): {e}"
                )
                return []
            raise
        curator_response = curator_raw.get("content", "")

        # Parse and filter operations (same logic as _generate_curator_operations_for_task)
        return self._parse_curator_operations(curator_response, chunk_results[0].task_id)

    def _parse_curator_operations(
        self, curator_response: str, task_id_for_log: str = ""
    ) -> list[dict]:
        """Parse curator JSON response and return filtered ADD operations."""
        operations_info = extract_json_from_text(curator_response, "operations")

        try:
            if not operations_info:
                raise ValueError("Failed to extract valid JSON from curator response")
            if "operations" not in operations_info:
                raise ValueError("JSON missing required 'operations' field")
            if not isinstance(operations_info["operations"], list):
                raise ValueError("'operations' field must be a list")

            allowed_sections = {
                "strategies_and_hard_rules",
                "apis_to_use_for_specific_information",
                "useful_code_snippets_and_templates",
                "common_mistakes_and_correct_strategies",
                "problem_solving_heuristics_and_workflows",
                "verification_checklist",
                "troubleshooting_and_pitfalls",
                "others",
            }

            filtered: list[dict] = []
            for i, op in enumerate(operations_info["operations"]):
                if not isinstance(op, dict):
                    continue
                if op.get("type") != "ADD":
                    continue
                required_fields = {"type", "section", "content"}
                if not required_fields.issubset(op.keys()):
                    continue
                section_name = (
                    str(op.get("section", ""))
                    .strip()
                    .lower()
                    .replace(" ", "_")
                    .replace("&", "and")
                    .rstrip(":")
                )
                if section_name not in allowed_sections:
                    print(
                        f"    Skipping op: disallowed section '{op.get('section')}'"
                    )
                    continue
                filtered.append(op)

            return filtered

        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            print(f"  Curator JSON parse failed: {e}")
            return []
        except Exception as e:
            print(f"  Curator operation failed: {e}")
            return []

    def _generate_curator_operations_for_task(
        self, result: BatchTaskResult
    ) -> list[dict]:
        """
        Generate curator ADD operations for a single task.
        Used when curator_batch_size=1 or for backward compatibility.
        """
        return self._generate_curator_operations_for_chunk([result])

    def _generate_reflection_for_task(
        self,
        result: BatchTaskResult,
        playbook_override: str | None = None,
        *,
        parallel_chunk_idx: int | None = None,
        parallel_num_chunks: int | None = None,
    ) -> str:
        """
        Generate reflector reasoning for a single task result.
        Used as the 'guidebook' input for the curator.
        """
        pb = playbook_override if playbook_override is not None else self.playbook
        filled_prompt = (
            self.reflector_prompt
            .replace("{{ground_truth_code}}", result.gt_code or "")
            .replace("{{test_report}}", result.test_report or "")
            .replace("{{generated_code}}", "See full conversation history below")
            .replace("{{generated_rationale}}", "See full conversation history below")
            .replace("{{spec_or_api_docs}}", "See full conversation history below")
            .replace("{{execution_error}}", "See full conversation history below")
            .replace("{{playbook}}", pb or "N/A")
            .replace("{{previous_reflection}}", "N/A")
        )

        conversation_history = "\n\n=== FULL CONVERSATION HISTORY ===\n"
        for i, msg in enumerate(result.trimmed_messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            conversation_history += f"[{i}] {role.upper()}: {content}\n\n"
        filled_prompt += conversation_history

        message_ = self.reflector_model.generate(
            messages=[{"role": "user", "content": filled_prompt}],
            extra_log_fields={
                "ace_role": "reflector",
                "ace_stage": "phase2_batch",
                "chunk_idx": parallel_chunk_idx,
                "chunks_total": parallel_num_chunks,
            },
        )
        return message_.get("content", "")
