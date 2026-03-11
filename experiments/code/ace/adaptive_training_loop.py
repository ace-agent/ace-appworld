"""
Adaptive Training Loop for ACE Framework

This module provides an adaptive training loop that integrates AdaptiveQuestionSelector
with the StarAgent (adaptation agent) to enable curriculum learning.

The workflow:
    1. Selector chooses N tasks per iteration (e.g., 5)
    2. All tasks share the SAME starting playbook state
    3. Generate N rollouts (generator + execute, NO reflection yet)
    4. Prune rollouts to top K based on heuristic (e.g., K=3)
    5. Run reflector on K pruned rollouts → generates K reflections
    6. Curator receives ALL K reflections → ONE batch update to playbook
    7. Loop continues with updated playbook

Key difference from standard adaptation:
- Standard: train task 1 → reflect → curate → update playbook → train task 2 → ...
- Adaptive: train N tasks → prune to K → reflect on K → batch curate K reflections → update playbook once

This enables more efficient learning through curriculum selection and selective curation.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from appworld import AppWorld
from appworld.common.constants import DEFAULT_EXPERIMENT_NAME
from appworld.evaluator import evaluate_task
from appworld_experiments.code.ace.adaptation_agent import StarAgent
from appworld_experiments.code.ace.rollout_pruner import BasePruner, RolloutInfo, create_pruner
from appworld_experiments.code.ace.adaptive_training_logger import AdaptiveTrainingLogger
from experiments.curriculum.adaptive_selector import AdaptiveQuestionSelector, create_selector


class AdaptiveTrainingLoop:
    """
    Training loop that uses adaptive question selection for curriculum learning.

    Wraps StarAgent and coordinates iteration-based training with adaptive selection and rollout pruning.

    Key architecture:
    - All tasks in an iteration share the SAME starting playbook
    - Generate rollouts for all tasks
    - Prune rollouts before curator (select top K from N)
    - Update playbook once per iteration
    """

    def __init__(
        self,
        agent: StarAgent,
        selector: AdaptiveQuestionSelector,
        pruner: Optional[BasePruner] = None,
        num_rollouts_per_task: int = 1,
        experiment_name: Optional[str] = None,
        save_playbook_every_iteration: bool = True,
        playbook_save_dir: Optional[str] = None,
        enable_logging: bool = True,
        log_base_dir: str = "experiments/logs",
    ):
        """
        Args:
            agent: Configured StarAgent for adaptation
            selector: AdaptiveQuestionSelector instance
            pruner: Optional rollout pruner (if None, all rollouts sent to curator)
            num_rollouts_per_task: Number of rollouts to generate per task (m)
            experiment_name: Name for AppWorld experiment
            save_playbook_every_iteration: Whether to save playbook after each iteration
            playbook_save_dir: Directory to save intermediate playbooks
            enable_logging: Whether to enable comprehensive logging
            log_base_dir: Base directory for logs
        """
        self.agent = agent
        self.selector = selector
        self.pruner = pruner
        self.num_rollouts_per_task = num_rollouts_per_task
        self.experiment_name = experiment_name or DEFAULT_EXPERIMENT_NAME
        self.save_playbook_every_iteration = save_playbook_every_iteration
        self.iteration_count = 0

        # Initialize logger
        self.enable_logging = enable_logging
        if self.enable_logging:
            # Build config dict for logging
            config_dict = {
                "num_rollouts_per_task": num_rollouts_per_task,
                "selector": {
                    "algorithm": getattr(selector, 'algorithm', "unknown"),
                    "num_tasks_per_iteration": getattr(selector, 'num_tasks_per_iteration', 5),
                    "no_repeat_tasks": getattr(selector, 'no_repeat_tasks', True),
                    "dataset": getattr(selector, 'dataset_path', None),
                },
                "pruner": {
                    "strategy": pruner.strategy if pruner else None,
                    "num_rollouts_for_reflection": pruner.num_rollouts_for_reflection if pruner else None,
                } if pruner else None,
            }

            self.logger = AdaptiveTrainingLogger(
                experiment_name=self.experiment_name,
                log_base_dir=log_base_dir,
                create_timestamp=True,
                config=config_dict,
            )
            # Log initial playbook
            if hasattr(self.agent, 'playbook') and self.agent.playbook:
                self.logger.log_playbook(
                    self.agent.playbook,
                    iteration=0,
                    metadata={"type": "initial_playbook"},
                )
        else:
            self.logger = None
            # Fall back to provided dir or default if no logger
            self.playbook_save_dir = playbook_save_dir or "experiments/playbooks"
            # Create save directory if needed
            if self.save_playbook_every_iteration:
                Path(self.playbook_save_dir).mkdir(parents=True, exist_ok=True)

    def train(
        self,
        max_iterations: Optional[int] = None,
    ):
        """
        Run adaptive training loop until all tasks completed or max_iterations reached.

        Args:
            max_iterations: Maximum number of iterations to process (None = unlimited)
        """
        self.agent.logger.initialize(
            experiment_name=self.experiment_name,
            num_tasks=len(self.selector.all_tasks),
        )

        while not self.selector.is_complete():
            # Check iteration limit
            if max_iterations is not None and self.iteration_count >= max_iterations:
                print(f"Reached max_iterations limit: {max_iterations}")
                break

            # Select next iteration
            iteration_result = self.selector.select_next_batch(
                playbook_text=self.agent.playbook,
            )

            if not iteration_result.task_ids:
                print("No more tasks to select")
                break

            self.iteration_count += 1
            print(f"\n{'='*80}")
            print(f"ITERATION {self.iteration_count}: Selected {len(iteration_result.task_ids)} tasks")
            print(f"Selection metadata: {iteration_result.metadata}")
            print(f"Tasks: {iteration_result.task_ids}")
            print(f"{'='*80}\n")

            # Log iteration start
            if self.logger:
                self.logger.log_iteration_start(
                    iteration=self.iteration_count,
                    selected_tasks=iteration_result.task_ids,
                    metadata=iteration_result.metadata,
                )

            # Save starting playbook for this iteration
            starting_playbook = self.agent.playbook

            # Generate rollouts for all tasks (they all share same starting playbook)
            print(f"Generating {self.num_rollouts_per_task} rollout(s) for {len(iteration_result.task_ids)} tasks...")
            print(f"Total rollouts to generate: {len(iteration_result.task_ids)} tasks × {self.num_rollouts_per_task} rollouts/task = {len(iteration_result.task_ids) * self.num_rollouts_per_task} rollouts")
            rollouts = self._generate_rollouts(iteration_result.task_ids, starting_playbook)

            print(f"\nGenerated {len(rollouts)} rollouts")
            print(f"  - Successful: {sum(1 for r in rollouts if r.success and r.test_failures == 0)}")
            print(f"  - Failed: {sum(1 for r in rollouts if not r.success or r.test_failures > 0)}")

            # Prune rollouts if pruner is configured
            if self.pruner:
                pruned_rollouts = self.pruner.prune(rollouts)
                print(f"Pruned to {len(pruned_rollouts)} rollouts for reflection")
                print(f"  - Successful: {sum(1 for r in pruned_rollouts if r.success and r.test_failures == 0)}")
                print(f"  - Failed: {sum(1 for r in pruned_rollouts if not r.success or r.test_failures > 0)}")
            else:
                pruned_rollouts = rollouts
                print("No pruning - all rollouts sent to reflector")

            # Process pruned rollouts through curator to update playbook
            if pruned_rollouts:
                print(f"\nProcessing {len(pruned_rollouts)} rollouts through curator...")
                self._update_playbook_from_rollouts(pruned_rollouts)
            else:
                print("\nNo rollouts to process - skipping curator")

            # Log iteration completion
            if self.logger:
                pruning_strategy = self.pruner.strategy if self.pruner else None
                self.logger.log_iteration_complete(
                    iteration=self.iteration_count,
                    selected_tasks=iteration_result.task_ids,
                    all_rollouts=rollouts,
                    pruned_rollouts=pruned_rollouts,
                    pruning_strategy=pruning_strategy,
                    metadata=iteration_result.metadata,
                )

                # Log playbook after update
                self.logger.log_playbook(
                    self.agent.playbook,
                    iteration=self.iteration_count,
                    metadata={"type": "after_iteration"},
                )

            # Mark iteration complete
            self.selector.mark_batch_complete(
                batch=iteration_result,
                playbook_text=self.agent.playbook,
            )

            # Save playbook snapshot (only if logger is disabled, otherwise logger already saved it)
            if self.save_playbook_every_iteration and not self.logger:
                self._save_playbook_snapshot()

            # Show progress
            progress = self.selector.get_progress()
            print(f"\nProgress: {progress['tried_tasks']}/{progress['total_tasks']} tasks - "
                  f"Iteration {self.iteration_count} complete\n")

        print(f"\nTraining complete! Processed {self.iteration_count} iterations")
        self._save_final_playbook()

        # Save final summary
        if self.logger:
            progress = self.selector.get_progress()
            self.logger.save_summary(final_metadata=progress)

    def _generate_rollouts(
        self,
        task_ids: List[str],
        starting_playbook: str,
    ) -> List[RolloutInfo]:
        """
        Generate multiple rollouts for all tasks using the same starting playbook.

        For each task, generates num_rollouts_per_task rollouts.
        All rollouts (across all tasks) start with the same playbook state.

        For each rollout:
        1. Run generator to create code
        2. Execute and evaluate
        3. Store rollout info (NO reflection yet - that comes after pruning)

        Args:
            task_ids: List of task IDs to generate rollouts for
            starting_playbook: Playbook state to use for all tasks

        Returns:
            List of RolloutInfo (length = len(task_ids) × num_rollouts_per_task)
        """
        rollouts = []
        total_rollouts = len(task_ids) * self.num_rollouts_per_task
        rollout_counter = 0

        for task_index, task_id in enumerate(task_ids):
            print(f"\n{'='*80}")
            print(f"Task {task_index + 1}/{len(task_ids)}: {task_id}")
            print(f"Generating {self.num_rollouts_per_task} rollout(s) for this task")
            print(f"{'='*80}")

            # Generate multiple rollouts for this task
            for rollout_index in range(self.num_rollouts_per_task):
                rollout_counter += 1
                print(f"\n--- Rollout {rollout_counter}/{total_rollouts} (Task: {task_id}, Attempt: {rollout_index + 1}/{self.num_rollouts_per_task}) ---")

                # Reset agent's playbook to starting state
                self.agent.playbook = starting_playbook

                # Track cost before
                initial_cost = self.agent.cost_tracker.overall_cost

                self.agent.current_task_index = (
                    self.selector.get_progress()['tried_tasks'] + task_index
                )

                rollout_info = {"success": False, "test_failures": 0}

                # Disable reflector and curator during rollout generation
                # We only want generator + execution
                original_use_reflector = self.agent.use_reflector
                original_curator_call = self.agent.curator_call

                # Disable reflector and curator
                self.agent.use_reflector = False
                def mock_curator_call(*args, **kwargs):
                    """Mock curator that does nothing during rollout generation"""
                    pass  # Silent - no need to print for every rollout
                self.agent.curator_call = mock_curator_call

                # Save execution context for later reflection generation
                execution_context = {}

                try:
                    # Solve task (generator + execution only)
                    self.agent.solve_task(task_id, self.experiment_name)

                    # Evaluate to get results
                    test_tracker, test_report = evaluate_task(task_id, self.experiment_name)

                    rollout_info["success"] = len(test_tracker.failures) == 0
                    rollout_info["test_failures"] = len(test_tracker.failures)

                    # Capture execution context for reflection
                    # Note: trimmed_messages is a property, so we save the underlying messages
                    execution_context = {
                        "test_report": getattr(self.agent, 'test_report', None),
                        "world_gt_code": getattr(self.agent, 'world_gt_code', None),
                        "messages": list(getattr(self.agent, 'messages', [])),  # Save messages, not trimmed_messages
                        "step_number": getattr(self.agent, 'step_number', 0),
                    }

                except Exception as e:
                    print(f"  Error generating rollout: {e}")
                    rollout_info["success"] = False
                    # test_failures=999 is a sentinel value indicating an exception occurred
                    # during rollout generation (not an actual test failure count)
                    rollout_info["test_failures"] = 999

                finally:
                    # Restore original settings
                    self.agent.use_reflector = original_use_reflector
                    self.agent.curator_call = original_curator_call

                # Calculate cost for this rollout
                final_cost = self.agent.cost_tracker.overall_cost
                rollout_cost = final_cost - initial_cost

                # Create rollout info (no reflection yet)
                rollout = RolloutInfo(
                    task_id=task_id,
                    success=rollout_info["success"],
                    cost=rollout_cost,
                    num_steps=getattr(self.agent, 'step_number', 0),
                    test_failures=rollout_info["test_failures"],
                    reflection=None,  # Will be added after pruning
                    metadata={
                        "task_index": task_index,
                        "rollout_index": rollout_index,
                        "rollout_number": rollout_counter,
                        "execution_context": execution_context,  # Store for later reflection
                    }
                )

                rollouts.append(rollout)

                # Log rollout
                if self.logger:
                    self.logger.log_rollout(
                        iteration=self.iteration_count,
                        rollout=rollout,
                        rollout_number=rollout_counter,
                        trajectory=None,  # Could add trajectory logging here if needed
                    )

                print(f"  Rollout complete: success={rollout.success}, cost={rollout.cost:.2f}, "
                      f"steps={rollout.num_steps}, test_failures={rollout.test_failures}")

        # Reset playbook to starting state after all rollouts
        self.agent.playbook = starting_playbook

        print(f"\nCompleted all rollouts: {len(task_ids)} tasks × {self.num_rollouts_per_task} rollouts/task = {len(rollouts)} total rollouts")

        return rollouts

    def _update_playbook_from_rollouts(self, rollouts: List[RolloutInfo]):
        """
        Update playbook by processing selected rollouts through reflector → curator.

        Workflow:
        1. Run reflector on each of the K pruned rollouts → generates K reflections
        2. Curator receives ALL K reflections at once → ONE batch update to playbook

        Args:
            rollouts: Pruned list of rollouts to learn from (K rollouts)
        """
        # Step 1: Generate reflections for all pruned rollouts
        print(f"\nStep 1: Generating reflections for {len(rollouts)} pruned rollouts...")
        reflections = []

        for idx, rollout in enumerate(rollouts):
            print(f"  Generating reflection {idx + 1}/{len(rollouts)} for {rollout.task_id} "
                  f"(success={rollout.success}, test_failures={rollout.test_failures})")

            try:
                # Restore execution context from rollout metadata
                execution_context = rollout.metadata.get("execution_context", {})

                if execution_context and self.agent.use_reflector:
                    # Temporarily restore agent state for reflection
                    original_test_report = self.agent.test_report
                    original_world_gt_code = self.agent.world_gt_code
                    original_messages = list(self.agent.messages)  # Save messages (trimmed_messages is a property)
                    original_step_number = self.agent.step_number
                    original_file_console = self.agent.logger.file_console

                    try:
                        # Restore context
                        self.agent.test_report = execution_context.get("test_report")
                        self.agent.world_gt_code = execution_context.get("world_gt_code")
                        self.agent.messages = list(execution_context.get("messages", []))  # Restore messages
                        self.agent.step_number = execution_context.get("step_number", 0)

                        # Disable file logging to avoid I/O errors on closed files
                        self.agent.logger.file_console = None

                        # Call reflector with restored context
                        reflection = self.agent.reflector_call()

                        if reflection:
                            reflections.append(reflection)
                            rollout.reflection = reflection
                            print(f"    Generated reflection from reflector (length: {len(reflection)} chars)")
                        else:
                            # Fallback to simple reflection
                            reflection = f"Task {rollout.task_id} - No reflection generated by reflector."
                            reflections.append(reflection)
                            rollout.reflection = reflection
                            print(f"    Reflector returned empty, using fallback")

                    finally:
                        # Restore original agent state
                        self.agent.test_report = original_test_report
                        self.agent.world_gt_code = original_world_gt_code
                        self.agent.messages = original_messages  # Restore messages
                        self.agent.step_number = original_step_number
                        self.agent.logger.file_console = original_file_console  # Restore file console
                else:
                    # Fallback: Simple text-based reflection when no context or reflector disabled
                    if rollout.success and rollout.test_failures == 0:
                        reflection = f"Task {rollout.task_id} completed successfully."
                    elif rollout.test_failures == 999:
                        reflection = f"Task {rollout.task_id} encountered an error during execution."
                    else:
                        reflection = f"Task {rollout.task_id} failed with {rollout.test_failures} test failures."

                    reflections.append(reflection)
                    rollout.reflection = reflection
                    print(f"    Simple fallback reflection: {reflection}")

            except Exception as e:
                print(f"    Error generating reflection: {e}")
                import traceback
                traceback.print_exc()
                # Use fallback on error
                reflection = f"Task {rollout.task_id} - Error during reflection generation."
                reflections.append(reflection)
                rollout.reflection = reflection

        print(f"\nGenerated {len(reflections)} reflections from {len(rollouts)} rollouts")

        # Log reflections
        if self.logger:
            self.logger.log_reflections(
                iteration=self.iteration_count,
                rollouts_with_reflections=rollouts,
            )

        # Step 2: Batch curator update with ALL reflections
        if reflections:
            print(f"\nStep 2: Curator batch update with {len(reflections)} reflections...")

            try:
                # Combine all reflections into a single batch for curator
                # The curator should process all reflections together and do ONE update

                # Option 1: Call curator once per reflection (curator handles merge internally)
                # This is the standard approach - curator is designed to handle sequential updates
                for idx, reflection in enumerate(reflections):
                    print(f"  Processing reflection {idx + 1}/{len(reflections)}...")

                    # Create a mock reflector that returns the pre-generated reflection
                    original_reflector_call = self.agent.reflector_call
                    def mock_reflector_call():
                        return reflection
                    self.agent.reflector_call = mock_reflector_call

                    # Call curator (it will use mock reflector)
                    self.agent.curator_call()

                    # Restore reflector
                    self.agent.reflector_call = original_reflector_call

                print(f"Curator completed batch update from {len(reflections)} reflections")

            except Exception as e:
                print(f"Error in curator batch update: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\nNo reflections to process - skipping curator")

        print("Playbook update complete")

    def _save_playbook_snapshot(self):
        """Save playbook snapshot after iteration completion"""
        if not self.agent.playbook:
            return

        progress = self.selector.get_progress()
        snapshot_name = f"playbook_iter{self.iteration_count:03d}_tasks{progress['tried_tasks']}.txt"
        snapshot_path = os.path.join(self.playbook_save_dir, snapshot_name)

        with open(snapshot_path, "w") as f:
            f.write(self.agent.playbook)

        print(f"Saved playbook snapshot: {snapshot_path}")

    def _save_final_playbook(self):
        """Save final trained playbook"""
        if not self.agent.playbook:
            return

        if hasattr(self.agent, 'trained_playbook_file_path') and self.agent.trained_playbook_file_path:
            final_path = self.agent.trained_playbook_file_path
        else:
            final_path = os.path.join(self.playbook_save_dir, "playbook_final.txt")

        with open(final_path, "w") as f:
            f.write(self.agent.playbook)

        print(f"Saved final playbook: {final_path}")


def run_adaptive_training(
    agent_config: Dict[str, Any],
    selector_config: Dict[str, Any],
    pruner_config: Optional[Dict[str, Any]] = None,
    num_rollouts_per_task: int = 1,
    experiment_name: Optional[str] = None,
    max_iterations: Optional[int] = None,
    save_playbook_every_iteration: bool = True,
    playbook_save_dir: Optional[str] = None,
    enable_logging: bool = True,
    log_base_dir: str = "experiments/logs",
) -> AdaptiveTrainingLoop:
    """
    High-level function to run adaptive training.

    Args:
        agent_config: Configuration dict for StarAgent.from_dict()
        selector_config: Configuration dict for create_selector()
        pruner_config: Optional configuration dict for create_pruner()
        num_rollouts_per_task: Number of rollouts to generate per task (m)
        experiment_name: AppWorld experiment name
        max_iterations: Maximum number of iterations to process
        save_playbook_every_iteration: Whether to save playbook after each iteration
        playbook_save_dir: Directory for playbook snapshots
        enable_logging: Whether to enable comprehensive logging
        log_base_dir: Base directory for logs

    Returns:
        AdaptiveTrainingLoop instance (after training completes)

    Example:
        agent_config = {
            "type": "ace_adaptation_react",
            "generator_model_config": {...},
            "reflector_model_config": {...},
            "curator_model_config": {...},
            ...
        }

        selector_config = {
            "algorithm": "difficulty_progressive",
            "num_tasks_per_iteration": 5,
            "dataset_path": "experiments/data/train.txt",
            "difficulty_schedule": "progressive",
        }

        pruner_config = {
            "strategy": "failure_first",
            "num_rollouts_for_reflection": 3,
        }

        loop = run_adaptive_training(agent_config, selector_config, pruner_config)
    """
    # Create agent
    agent = StarAgent.from_dict(agent_config)

    # Create selector
    selector = create_selector(**selector_config)

    # Create pruner if configured
    pruner = None
    if pruner_config:
        pruner = create_pruner(**pruner_config)

    # Create training loop
    loop = AdaptiveTrainingLoop(
        agent=agent,
        selector=selector,
        pruner=pruner,
        num_rollouts_per_task=num_rollouts_per_task,
        experiment_name=experiment_name,
        save_playbook_every_iteration=save_playbook_every_iteration,
        playbook_save_dir=playbook_save_dir,
        enable_logging=enable_logging,
        log_base_dir=log_base_dir,
    )

    # Run training
    loop.train(
        max_iterations=max_iterations,
    )

    return loop
