"""
Adaptive Training Logger

Comprehensive logging system for adaptive training experiments.
Logs playbook versions, task selections, rollout trajectories, pruning decisions,
reflections, and all metadata needed for analysis.

Output structure:
    <log_dir>/
        metadata.json                  # Experiment metadata and configs
        summary.json                   # Final summary statistics

        playbooks/
            playbook_v000_iter0.txt    # Starting playbook
            playbook_v001_iter1.txt    # After iteration 1
            playbook_v002_iter2.txt    # After iteration 2
            ...

        playbook_analysis/
            rule_history.json          # Complete history of all rules and their contributors
                                       # Maps each rule → [contributing tasks], num_contributions
                                       # Use this to see which tasks contributed to which rules

            rule_contributions.jsonl   # Task-centric view (JSONL - one line per task)
                                       # Each line: task_id → [rules it contributed to]
                                       # Use this for streaming/incremental analysis

        iterations/
            iter_001.json              # Iteration 1: tasks, rollouts, pruning
            iter_002.json              # Iteration 2: tasks, rollouts, pruning
            ...

        rollouts/
            iter_001/
                rollout_001_task_A_attempt_1.json
                rollout_002_task_A_attempt_2.json
                ...

        reflections/
            iter_001_reflections.json  # All reflections from iteration 1
            iter_002_reflections.json  # All reflections from iteration 2
            ...

How to read rule contribution files:

    rule_history.json (rule-centric):
        - Shows all rules in the playbook
        - For each rule: which tasks contributed to it, when it was added, how many times modified
        - Best for: "Which tasks contributed to rule X?"

    rule_contributions.jsonl (task-centric):
        - JSONL format - one JSON object per line (can be read incrementally)
        - For each task: which rules it contributed to
        - Best for: "What rules did task Y contribute to?"
        - Read with: for line in open('file.jsonl'): data = json.loads(line)
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, asdict

from experiments.code.ace.rollout_pruner import RolloutInfo


@dataclass
class IterationLog:
    """Log entry for one iteration"""
    iteration_number: int
    timestamp: str
    selected_tasks: List[str]
    num_rollouts_generated: int
    num_rollouts_after_pruning: int
    pruning_strategy: Optional[str]
    rollout_results: List[Dict[str, Any]]
    pruned_rollout_ids: List[str]
    discarded_rollout_ids: List[str]
    num_reflections: int
    playbook_version_before: int
    playbook_version_after: int
    metadata: Dict[str, Any]


@dataclass
class ReflectionLog:
    """Log entry for reflections"""
    iteration_number: int
    reflection_id: str
    task_id: str
    rollout_id: str
    reflection_text: str
    success: bool
    test_failures: int
    cost: float
    timestamp: str


@dataclass
class PlaybookRuleContribution:
    """Track which tasks contributed to a playbook rule"""
    rule_text: str
    rule_index: int
    first_added_iteration: int
    contributing_tasks: List[str]
    num_contributions: int


class AdaptiveTrainingLogger:
    """
    Comprehensive logger for adaptive training experiments.

    Creates a structured directory with all experiment data:
    - Playbook versions at each iteration
    - Task selection and rollout details
    - Pruning decisions
    - Reflections and their contributing tasks
    - Rule contribution tracking
    """

    def __init__(
        self,
        experiment_name: str,
        log_base_dir: str = "experiments/logs",
        create_timestamp: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            experiment_name: Name of the experiment
            log_base_dir: Base directory for all logs
            create_timestamp: If True, append timestamp to experiment name
            config: Optional experiment configuration to log
        """
        self.experiment_name = experiment_name
        self.config = config or {}

        # Create experiment directory with timestamp
        if create_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dir_name = f"{experiment_name}_{timestamp}"
        else:
            dir_name = experiment_name

        self.log_dir = Path(log_base_dir) / dir_name

        # Create directory structure
        self.playbooks_dir = self.log_dir / "playbooks"
        self.playbook_analysis_dir = self.log_dir / "playbook_analysis"
        self.iterations_dir = self.log_dir / "iterations"
        self.rollouts_dir = self.log_dir / "rollouts"
        self.reflections_dir = self.log_dir / "reflections"

        for dir_path in [
            self.log_dir,
            self.playbooks_dir,
            self.playbook_analysis_dir,
            self.iterations_dir,
            self.rollouts_dir,
            self.reflections_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # State tracking
        self.playbook_version = 0
        self.iteration_count = 0
        self.rule_contributions: Dict[str, PlaybookRuleContribution] = {}
        self.iteration_logs: List[IterationLog] = []
        self.reflection_logs: List[ReflectionLog] = []
        self.current_iteration_tasks: List[str] = []  # Track tasks in current iteration
        self.previous_playbook_rules: Set[str] = set()  # Track previous playbook rules to detect modifications

        # Start time
        self.start_time = datetime.now()

        # Save metadata
        self._save_metadata()

        print(f"Adaptive Training Logger initialized at: {self.log_dir}")

    def _save_metadata(self):
        """Save experiment metadata"""
        metadata = {
            "experiment_name": self.experiment_name,
            "start_time": self.start_time.isoformat(),
            "log_directory": str(self.log_dir),
            "config": self.config,
        }

        metadata_path = self.log_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def log_playbook(
        self,
        playbook_text: str,
        iteration: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Log a playbook version.

        Args:
            playbook_text: The playbook content
            iteration: Iteration number (0 = initial)
            metadata: Optional metadata about this version
        """
        # Save playbook file
        filename = f"playbook_v{self.playbook_version:03d}_iter{iteration}.txt"
        playbook_path = self.playbooks_dir / filename

        with open(playbook_path, "w") as f:
            f.write(playbook_text)

        # Analyze rules and track contributions
        self._analyze_playbook_rules(playbook_text, iteration)

        # Log metadata
        version_metadata = {
            "version": self.playbook_version,
            "iteration": iteration,
            "num_rules": len(playbook_text.split("\n")) if playbook_text else 0,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        metadata_path = self.playbooks_dir / f"playbook_v{self.playbook_version:03d}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(version_metadata, f, indent=2)

        self.playbook_version += 1

        print(f"Logged playbook version {self.playbook_version - 1} (iteration {iteration})")

    def _analyze_playbook_rules(self, playbook_text: str, iteration: int):
        """
        Analyze playbook rules and track which tasks contribute to each.

        Args:
            playbook_text: The playbook content
            iteration: Current iteration number
        """
        if not playbook_text:
            return

        rules = [line.strip() for line in playbook_text.split("\n") if line.strip()]
        current_rules = set(rules)

        # Detect changes compared to previous playbook
        if iteration > 0:
            # Find new or modified rules (rules that weren't in the previous version)
            new_or_modified_rules = current_rules - self.previous_playbook_rules
        else:
            # Iteration 0 - initial playbook, no new/modified rules
            new_or_modified_rules = set()

        # Track rules and their contributors
        for idx, rule in enumerate(rules):
            if rule not in self.rule_contributions:
                # Brand new rule that never existed before
                self.rule_contributions[rule] = PlaybookRuleContribution(
                    rule_text=rule,
                    rule_index=idx,
                    first_added_iteration=iteration,
                    contributing_tasks=list(self.current_iteration_tasks) if iteration > 0 else [],
                    num_contributions=1 if iteration > 0 else 0,
                )
            elif rule in new_or_modified_rules:
                # Rule existed before but was modified/re-added in this iteration
                # Add current iteration's tasks as contributors
                contrib = self.rule_contributions[rule]
                for task in self.current_iteration_tasks:
                    if task not in contrib.contributing_tasks:
                        contrib.contributing_tasks.append(task)
                contrib.num_contributions += 1

        # Update previous playbook state for next comparison
        self.previous_playbook_rules = current_rules

        # Save rule history
        self._save_rule_history()

    def log_iteration_start(
        self,
        iteration: int,
        selected_tasks: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Log the start of an iteration.

        Args:
            iteration: Iteration number
            selected_tasks: List of task IDs selected for this iteration
            metadata: Optional metadata
        """
        self.iteration_count = iteration
        self.current_iteration_tasks = selected_tasks  # Track for rule contribution

        iteration_metadata = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "selected_tasks": selected_tasks,
            "num_tasks": len(selected_tasks),
            "metadata": metadata or {},
        }

        # Create iteration directory for rollouts
        iter_rollouts_dir = self.rollouts_dir / f"iter_{iteration:03d}"
        iter_rollouts_dir.mkdir(exist_ok=True)

        print(f"\n{'='*80}")
        print(f"ITERATION {iteration} STARTED")
        print(f"Selected {len(selected_tasks)} tasks: {selected_tasks}")
        print(f"{'='*80}")

    def log_rollout(
        self,
        iteration: int,
        rollout: RolloutInfo,
        rollout_number: int,
        trajectory: Optional[str] = None,
    ):
        """
        Log a single rollout with full trajectory.

        Args:
            iteration: Iteration number
            rollout: RolloutInfo object
            rollout_number: Sequential rollout number
            trajectory: Optional trajectory/execution log
        """
        # Use 1-based rollout_index for consistency with attempt_N naming
        rollout_index_display = rollout.metadata.get("rollout_index", 0) + 1

        rollout_data = {
            "rollout_number": rollout_number,
            "iteration": iteration,
            "task_id": rollout.task_id,
            "rollout_index": rollout_index_display,  # 1-based
            "success": rollout.success,
            "cost": rollout.cost,
            "num_steps": rollout.num_steps,
            "test_failures": rollout.test_failures,
            "has_reflection": rollout.reflection is not None,
            "trajectory": trajectory,
            "metadata": rollout.metadata,
            "timestamp": datetime.now().isoformat(),
        }

        # Save individual rollout file
        filename = f"rollout_{rollout_number:03d}_task_{rollout.task_id}_attempt_{rollout_index_display}.json"
        rollout_path = self.rollouts_dir / f"iter_{iteration:03d}" / filename

        with open(rollout_path, "w") as f:
            json.dump(rollout_data, f, indent=2)

    def log_iteration_complete(
        self,
        iteration: int,
        selected_tasks: List[str],
        all_rollouts: List[RolloutInfo],
        pruned_rollouts: List[RolloutInfo],
        pruning_strategy: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Log iteration completion with pruning decisions.

        Args:
            iteration: Iteration number
            selected_tasks: Tasks selected for this iteration
            all_rollouts: All generated rollouts
            pruned_rollouts: Rollouts after pruning
            pruning_strategy: Strategy used for pruning
            metadata: Optional metadata
        """
        # Get IDs of pruned and discarded rollouts
        pruned_ids = [
            f"{r.task_id}_attempt_{r.metadata.get('rollout_index', 0) + 1}"
            for r in pruned_rollouts
        ]

        all_ids = [
            f"{r.task_id}_attempt_{r.metadata.get('rollout_index', 0) + 1}"
            for r in all_rollouts
        ]

        discarded_ids = [rid for rid in all_ids if rid not in pruned_ids]

        # Create iteration log
        iteration_log = IterationLog(
            iteration_number=iteration,
            timestamp=datetime.now().isoformat(),
            selected_tasks=selected_tasks,
            num_rollouts_generated=len(all_rollouts),
            num_rollouts_after_pruning=len(pruned_rollouts),
            pruning_strategy=pruning_strategy,
            rollout_results=[
                {
                    "task_id": r.task_id,
                    "rollout_index": r.metadata.get("rollout_index", 0) + 1,  # 1-based to match attempt_N naming
                    "success": r.success,
                    "cost": r.cost,
                    "test_failures": r.test_failures,
                }
                for r in all_rollouts
            ],
            pruned_rollout_ids=pruned_ids,
            discarded_rollout_ids=discarded_ids,
            num_reflections=sum(1 for r in pruned_rollouts if r.reflection),
            playbook_version_before=self.playbook_version - 1,
            playbook_version_after=self.playbook_version,
            metadata=metadata or {},
        )

        self.iteration_logs.append(iteration_log)

        # Save iteration summary
        iter_path = self.iterations_dir / f"iter_{iteration:03d}.json"
        with open(iter_path, "w") as f:
            json.dump(asdict(iteration_log), f, indent=2)

        print(f"\nIteration {iteration} complete:")
        print(f"  - Generated {len(all_rollouts)} rollouts")
        print(f"  - Pruned to {len(pruned_rollouts)} rollouts")
        print(f"  - Discarded {len(discarded_ids)} rollouts")
        print(f"  - Pruning strategy: {pruning_strategy or 'none'}")

    def log_reflections(
        self,
        iteration: int,
        rollouts_with_reflections: List[RolloutInfo],
    ):
        """
        Log all reflections from an iteration.
        Also updates the rollout files to mark which ones have reflections.

        Args:
            iteration: Iteration number
            rollouts_with_reflections: List of pruned rollouts with their reflections
        """
        reflections_data = []

        for idx, rollout in enumerate(rollouts_with_reflections):
            if rollout.reflection:
                reflection_id = f"iter{iteration:03d}_refl{idx:03d}"

                reflection_log = ReflectionLog(
                    iteration_number=iteration,
                    reflection_id=reflection_id,
                    task_id=rollout.task_id,
                    rollout_id=f"{rollout.task_id}_attempt_{rollout.metadata.get('rollout_index', 0) + 1}",
                    reflection_text=rollout.reflection,
                    success=rollout.success,
                    test_failures=rollout.test_failures,
                    cost=rollout.cost,
                    timestamp=datetime.now().isoformat(),
                )

                self.reflection_logs.append(reflection_log)
                reflections_data.append(asdict(reflection_log))

                # Update the rollout file to mark it has reflection
                self._update_rollout_reflection_status(iteration, rollout)

                # Track task contribution to rules
                # (Will be updated when playbook changes)
                self._track_task_contribution(rollout.task_id, iteration)

        # Save reflections for this iteration
        if reflections_data:
            refl_path = self.reflections_dir / f"iter_{iteration:03d}_reflections.json"
            with open(refl_path, "w") as f:
                json.dump(reflections_data, f, indent=2)

            print(f"  - Logged {len(reflections_data)} reflections")

    def _update_rollout_reflection_status(self, iteration: int, rollout: RolloutInfo):
        """
        Update a rollout file to mark that it has a reflection.

        Args:
            iteration: Iteration number
            rollout: Rollout info with reflection added
        """
        # Find the rollout file
        rollout_number = rollout.metadata.get("rollout_number", 0)
        filename = f"rollout_{rollout_number:03d}_task_{rollout.task_id}_attempt_{rollout.metadata.get('rollout_index', 0) + 1}.json"
        rollout_path = self.rollouts_dir / f"iter_{iteration:03d}" / filename

        # Read existing data
        if rollout_path.exists():
            with open(rollout_path, "r") as f:
                rollout_data = json.load(f)

            # Update has_reflection flag
            rollout_data["has_reflection"] = True

            # Write back
            with open(rollout_path, "w") as f:
                json.dump(rollout_data, f, indent=2)

    def _track_task_contribution(self, task_id: str, iteration: int):
        """
        Track which task contributed to the playbook at this iteration.

        This logs task contributions to the JSONL file for easy streaming analysis.
        The detailed rule-to-task mapping is in rule_history.json.

        Args:
            task_id: Task that generated a reflection
            iteration: Current iteration
        """
        # Find which rules this task contributed to
        contributed_rules = []
        for rule_text, contrib in self.rule_contributions.items():
            if task_id in contrib.contributing_tasks:
                contributed_rules.append({
                    "rule_text": rule_text[:100] + "..." if len(rule_text) > 100 else rule_text,
                    "first_added_iteration": contrib.first_added_iteration,
                })

        # Log to JSONL for easy streaming/incremental analysis
        contrib_path = self.playbook_analysis_dir / "rule_contributions.jsonl"

        contribution = {
            "task_id": task_id,
            "iteration": iteration,
            "num_rules_contributed": len(contributed_rules),
            "contributed_rules": contributed_rules,
            "timestamp": datetime.now().isoformat(),
        }

        with open(contrib_path, "a") as f:
            f.write(json.dumps(contribution) + "\n")

    def _save_rule_history(self):
        """Save the history of rule additions and contributions"""
        history = {
            "rules": [
                {
                    "rule_text": contrib.rule_text,
                    "rule_index": contrib.rule_index,
                    "first_added_iteration": contrib.first_added_iteration,
                    "contributing_tasks": contrib.contributing_tasks,
                    "num_contributions": contrib.num_contributions,
                }
                for contrib in self.rule_contributions.values()
            ],
            "total_rules": len(self.rule_contributions),
            "last_updated": datetime.now().isoformat(),
        }

        history_path = self.playbook_analysis_dir / "rule_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    def save_summary(self, final_metadata: Optional[Dict[str, Any]] = None):
        """
        Save final experiment summary.

        Args:
            final_metadata: Optional final metadata
        """
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        summary = {
            "experiment_name": self.experiment_name,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "duration_human": f"{duration / 60:.2f} minutes",
            "total_iterations": self.iteration_count,
            "total_playbook_versions": self.playbook_version,
            "total_reflections": len(self.reflection_logs),
            "total_rules": len(self.rule_contributions),
            "final_metadata": final_metadata or {},
        }

        summary_path = self.log_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*80}")
        print(f"EXPERIMENT COMPLETE")
        print(f"{'='*80}")
        print(f"Duration: {duration / 60:.2f} minutes")
        print(f"Iterations: {self.iteration_count}")
        print(f"Playbook versions: {self.playbook_version}")
        print(f"Reflections: {len(self.reflection_logs)}")
        print(f"Rules: {len(self.rule_contributions)}")
        print(f"\nLogs saved to: {self.log_dir}")
        print(f"{'='*80}\n")
