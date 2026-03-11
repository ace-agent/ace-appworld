"""
Adaptive Question Selector for Curriculum Learning

This module provides adaptive selection algorithms that choose the next batch of
problems to train based on the current playbook state and questions tried so far.

The selector integrates with the ACE framework's adaptation workflow:
    Selector → Generator (parallel rollouts) → Reflector → Curator → Updated Playbook → Selector (next batch)
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from experiments.curriculum.data_selector import (
    get_task_difficulty,
    load_dataset,
)
from experiments.code.ace.playbook import parse_playbook_line


@dataclass
class SelectionResult:
    """Result of a selection operation"""
    task_ids: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseSelector(ABC):
    """Base class for adaptive question selectors"""

    def __init__(
        self,
        num_tasks_per_iteration: int = 5,
        random_seed: Optional[int] = None,
    ):
        """
        Args:
            num_tasks_per_iteration: Number of problems to select per iteration
            random_seed: Random seed for reproducibility
        """
        self.num_tasks_per_iteration = num_tasks_per_iteration
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)

    @abstractmethod
    def select_next_batch(
        self,
        available_tasks: List[str],
        tried_tasks: Set[str],
        playbook_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SelectionResult:
        """
        Select next batch of tasks to train on.

        Args:
            available_tasks: All available task IDs
            tried_tasks: Set of task IDs already attempted
            playbook_text: Current playbook content (optional)
            metadata: Additional metadata (e.g., task difficulties, costs, results)

        Returns:
            SelectionResult containing selected task IDs and metadata
        """
        pass

    def _filter_available(
        self,
        available_tasks: List[str],
        tried_tasks: Set[str],
    ) -> List[str]:
        """Filter out already-tried tasks"""
        return [t for t in available_tasks if t not in tried_tasks]


class RandomSelector(BaseSelector):
    """
    Random selection without replacement.

    Baseline algorithm: randomly select batch_size tasks from untried tasks.
    """

    def select_next_batch(
        self,
        available_tasks: List[str],
        tried_tasks: Set[str],
        playbook_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SelectionResult:
        """Randomly select next batch of tasks"""
        untried = self._filter_available(available_tasks, tried_tasks)

        if not untried:
            return SelectionResult(task_ids=[], metadata={"reason": "no_untried_tasks"})

        # Sample without replacement
        sample_size = min(self.num_tasks_per_iteration, len(untried))
        selected = random.sample(untried, sample_size)

        return SelectionResult(
            task_ids=selected,
            metadata={
                "algorithm": "random",
                "total_available": len(untried),
                "num_selected": sample_size,
            }
        )


class DifficultyProgressiveSelector(BaseSelector):
    """
    Difficulty-based progressive selection.

    Start with easy tasks, gradually increase difficulty as training progresses.
    Uses curriculum learning principle: learn simple concepts before complex ones.
    """

    DIFFICULTY_ORDER = ["easy", "medium", "hard"]

    def __init__(
        self,
        num_tasks_per_iteration: int = 5,
        random_seed: Optional[int] = None,
        difficulty_schedule: str = "progressive",  # "progressive", "mixed", "reverse"
        mix_ratio: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            num_tasks_per_iteration: Number of problems to select per iteration
            random_seed: Random seed for reproducibility
            difficulty_schedule: How to schedule difficulty progression
                - "progressive": easy -> medium -> hard
                - "mixed": sample from all difficulties with configurable ratio
                - "reverse": hard -> medium -> easy (for testing)
            mix_ratio: For "mixed" schedule, ratio of easy:medium:hard (e.g., {"easy": 0.5, "medium": 0.3, "hard": 0.2})
        """
        super().__init__(num_tasks_per_iteration, random_seed)
        self.difficulty_schedule = difficulty_schedule
        self.mix_ratio = mix_ratio or {"easy": 0.5, "medium": 0.3, "hard": 0.2}

    def select_next_batch(
        self,
        available_tasks: List[str],
        tried_tasks: Set[str],
        playbook_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SelectionResult:
        """Select next batch based on difficulty progression"""
        untried = self._filter_available(available_tasks, tried_tasks)

        if not untried:
            return SelectionResult(task_ids=[], metadata={"reason": "no_untried_tasks"})

        # Get difficulty for each untried task
        task_difficulties = self._get_task_difficulties(untried)

        if self.difficulty_schedule == "progressive":
            selected = self._select_progressive(task_difficulties, tried_tasks)
        elif self.difficulty_schedule == "mixed":
            selected = self._select_mixed(task_difficulties)
        elif self.difficulty_schedule == "reverse":
            selected = self._select_reverse(task_difficulties, tried_tasks)
        else:
            raise ValueError(f"Unknown difficulty_schedule: {self.difficulty_schedule}")

        return SelectionResult(
            task_ids=selected,
            metadata={
                "algorithm": "difficulty_progressive",
                "schedule": self.difficulty_schedule,
                "total_available": len(untried),
                "num_selected": len(selected),
                "difficulty_distribution": self._get_difficulty_distribution(selected, task_difficulties),
            }
        )

    def _get_task_difficulties(self, task_ids: List[str]) -> Dict[str, str]:
        """Get difficulty level for each task"""
        difficulties = {}
        for task_id in task_ids:
            try:
                difficulties[task_id] = get_task_difficulty(task_id)
            except Exception:
                # If difficulty not found, assume medium
                difficulties[task_id] = "medium"
        return difficulties

    def _select_progressive(
        self,
        task_difficulties: Dict[str, str],
        tried_tasks: Set[str],
    ) -> List[str]:
        """Progressive selection: prioritize easier tasks when fewer tasks completed"""
        # Group by difficulty
        by_difficulty = self._group_by_difficulty(task_difficulties)

        # Calculate training progress (0.0 to 1.0)
        total_tasks = len(task_difficulties) + len(tried_tasks)
        progress = len(tried_tasks) / total_tasks if total_tasks > 0 else 0.0

        selected = []

        # Early stage (0-33%): focus on easy tasks
        if progress < 0.33:
            selected.extend(self._sample_from_difficulty(by_difficulty, "easy", self.num_tasks_per_iteration))
            remaining = self.num_tasks_per_iteration - len(selected)
            if remaining > 0:
                selected.extend(self._sample_from_difficulty(by_difficulty, "medium", remaining))

        # Middle stage (33-66%): focus on medium tasks
        elif progress < 0.66:
            selected.extend(self._sample_from_difficulty(by_difficulty, "medium", self.num_tasks_per_iteration))
            remaining = self.num_tasks_per_iteration - len(selected)
            if remaining > 0:
                selected.extend(self._sample_from_difficulty(by_difficulty, "easy", remaining // 2))
                selected.extend(self._sample_from_difficulty(by_difficulty, "hard", remaining - remaining // 2))

        # Late stage (66-100%): focus on hard tasks
        else:
            selected.extend(self._sample_from_difficulty(by_difficulty, "hard", self.num_tasks_per_iteration))
            remaining = self.num_tasks_per_iteration - len(selected)
            if remaining > 0:
                selected.extend(self._sample_from_difficulty(by_difficulty, "medium", remaining))

        # If still not enough, sample from any remaining
        if len(selected) < self.num_tasks_per_iteration:
            all_remaining = [t for t in task_difficulties.keys() if t not in selected]
            remaining = self.num_tasks_per_iteration - len(selected)
            selected.extend(random.sample(all_remaining, min(remaining, len(all_remaining))))

        return selected[:self.num_tasks_per_iteration]

    def _select_mixed(self, task_difficulties: Dict[str, str]) -> List[str]:
        """Mixed selection: sample from all difficulties with specified ratio"""
        by_difficulty = self._group_by_difficulty(task_difficulties)

        selected = []
        for difficulty in self.DIFFICULTY_ORDER:
            count = int(self.num_tasks_per_iteration * self.mix_ratio.get(difficulty, 0))
            selected.extend(self._sample_from_difficulty(by_difficulty, difficulty, count))

        # Fill remaining slots if ratio doesn't sum to exactly num_tasks_per_iteration
        if len(selected) < self.num_tasks_per_iteration:
            all_remaining = [t for t in task_difficulties.keys() if t not in selected]
            remaining = self.num_tasks_per_iteration - len(selected)
            selected.extend(random.sample(all_remaining, min(remaining, len(all_remaining))))

        return selected[:self.num_tasks_per_iteration]

    def _select_reverse(
        self,
        task_difficulties: Dict[str, str],
        tried_tasks: Set[str],
    ) -> List[str]:
        """Reverse progressive: start with hard tasks"""
        by_difficulty = self._group_by_difficulty(task_difficulties)

        total_tasks = len(task_difficulties) + len(tried_tasks)
        progress = len(tried_tasks) / total_tasks if total_tasks > 0 else 0.0

        selected = []

        # Early: hard tasks
        if progress < 0.33:
            selected.extend(self._sample_from_difficulty(by_difficulty, "hard", self.num_tasks_per_iteration))
            remaining = self.num_tasks_per_iteration - len(selected)
            if remaining > 0:
                selected.extend(self._sample_from_difficulty(by_difficulty, "medium", remaining))

        # Middle: medium tasks
        elif progress < 0.66:
            selected.extend(self._sample_from_difficulty(by_difficulty, "medium", self.num_tasks_per_iteration))
            remaining = self.num_tasks_per_iteration - len(selected)
            if remaining > 0:
                selected.extend(self._sample_from_difficulty(by_difficulty, "hard", remaining // 2))
                selected.extend(self._sample_from_difficulty(by_difficulty, "easy", remaining - remaining // 2))

        # Late: easy tasks
        else:
            selected.extend(self._sample_from_difficulty(by_difficulty, "easy", self.num_tasks_per_iteration))
            remaining = self.num_tasks_per_iteration - len(selected)
            if remaining > 0:
                selected.extend(self._sample_from_difficulty(by_difficulty, "medium", remaining))

        # Fill remaining
        if len(selected) < self.num_tasks_per_iteration:
            all_remaining = [t for t in task_difficulties.keys() if t not in selected]
            remaining = self.num_tasks_per_iteration - len(selected)
            selected.extend(random.sample(all_remaining, min(remaining, len(all_remaining))))

        return selected[:self.num_tasks_per_iteration]

    def _group_by_difficulty(self, task_difficulties: Dict[str, str]) -> Dict[str, List[str]]:
        """Group tasks by difficulty level"""
        grouped = {d: [] for d in self.DIFFICULTY_ORDER}
        for task_id, difficulty in task_difficulties.items():
            if difficulty in grouped:
                grouped[difficulty].append(task_id)
        return grouped

    def _sample_from_difficulty(
        self,
        by_difficulty: Dict[str, List[str]],
        difficulty: str,
        count: int,
    ) -> List[str]:
        """Sample count tasks from given difficulty level"""
        available = by_difficulty.get(difficulty, [])
        if not available:
            return []
        sample_size = min(count, len(available))
        return random.sample(available, sample_size)

    def _get_difficulty_distribution(
        self,
        selected: List[str],
        task_difficulties: Dict[str, str],
    ) -> Dict[str, int]:
        """Get count of each difficulty in selected tasks"""
        distribution = {d: 0 for d in self.DIFFICULTY_ORDER}
        for task_id in selected:
            difficulty = task_difficulties.get(task_id, "medium")
            if difficulty in distribution:
                distribution[difficulty] += 1
        return distribution


class UncertaintyBasedSelector(BaseSelector):
    """
    Uncertainty-based selection using playbook analysis.

    Select tasks where the current playbook is least confident or has limited coverage.
    Uses playbook bullet statistics (helpful/harmful counts) as a proxy for confidence.
    """

    def __init__(
        self,
        num_tasks_per_iteration: int = 5,
        random_seed: Optional[int] = None,
        selection_strategy: str = "low_confidence",  # "low_confidence", "high_variance", "least_covered"
        difficulty_filter: Optional[str] = None,  # Optional: only select from specific difficulty
    ):
        """
        Args:
            num_tasks_per_iteration: Number of problems to select per iteration
            random_seed: Random seed for reproducibility
            selection_strategy: How to measure uncertainty
                - "low_confidence": Select tasks related to bullets with low helpful/(helpful+harmful) ratio
                - "high_variance": Select diverse tasks to maximize playbook coverage
                - "least_covered": Select tasks for which playbook has few relevant bullets
            difficulty_filter: If set, only select from this difficulty level
        """
        super().__init__(num_tasks_per_iteration, random_seed)
        self.selection_strategy = selection_strategy
        self.difficulty_filter = difficulty_filter

    def select_next_batch(
        self,
        available_tasks: List[str],
        tried_tasks: Set[str],
        playbook_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SelectionResult:
        """Select next batch based on playbook uncertainty"""
        untried = self._filter_available(available_tasks, tried_tasks)

        if not untried:
            return SelectionResult(task_ids=[], metadata={"reason": "no_untried_tasks"})

        # Filter by difficulty if specified
        if self.difficulty_filter:
            task_difficulties = {t: get_task_difficulty(t) for t in untried}
            untried = [t for t in untried if task_difficulties.get(t) == self.difficulty_filter]

        if not untried:
            return SelectionResult(
                task_ids=[],
                metadata={"reason": "no_tasks_matching_difficulty_filter"}
            )

        # If no playbook provided, fall back to random selection
        if not playbook_text:
            sample_size = min(self.num_tasks_per_iteration, len(untried))
            selected = random.sample(untried, sample_size)
            return SelectionResult(
                task_ids=selected,
                metadata={
                    "algorithm": "uncertainty_based",
                    "strategy": self.selection_strategy,
                    "note": "no_playbook_provided_fallback_to_random",
                }
            )

        # Parse playbook to extract bullet statistics
        playbook_stats = self._analyze_playbook(playbook_text)

        # Select based on strategy
        if self.selection_strategy == "low_confidence":
            selected = self._select_low_confidence(untried, playbook_stats)
        elif self.selection_strategy == "high_variance":
            selected = self._select_high_variance(untried, playbook_stats)
        elif self.selection_strategy == "least_covered":
            selected = self._select_least_covered(untried, playbook_stats)
        else:
            raise ValueError(f"Unknown selection_strategy: {self.selection_strategy}")

        return SelectionResult(
            task_ids=selected,
            metadata={
                "algorithm": "uncertainty_based",
                "strategy": self.selection_strategy,
                "total_available": len(untried),
                "num_selected": len(selected),
                "playbook_stats": playbook_stats,
            }
        )

    def _analyze_playbook(self, playbook_text: str) -> Dict[str, Any]:
        """Analyze playbook to compute confidence metrics"""
        lines = playbook_text.strip().split('\n')
        bullets = []

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            try:
                parsed = parse_playbook_line(line)
                if parsed:
                    bullets.append(parsed)
            except Exception:
                continue

        if not bullets:
            return {
                "total_bullets": 0,
                "avg_confidence": 0.0,
                "confidence_variance": 0.0,
            }

        # Calculate confidence for each bullet: helpful / (helpful + harmful)
        confidences = []
        for bullet in bullets:
            helpful = bullet.get("helpful", 0)
            harmful = bullet.get("harmful", 0)
            total = helpful + harmful
            confidence = helpful / total if total > 0 else 0.5
            confidences.append(confidence)

        avg_confidence = sum(confidences) / len(confidences)
        variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)

        return {
            "total_bullets": len(bullets),
            "avg_confidence": avg_confidence,
            "confidence_variance": variance,
            "confidences": confidences,
            "bullets": bullets,
        }

    def _select_low_confidence(
        self,
        untried: List[str],
        playbook_stats: Dict[str, Any],
    ) -> List[str]:
        """
        Select tasks randomly, with bias towards learning from low-confidence areas.

        Since we don't have task-to-bullet mappings, we use playbook stats as a signal:
        - If avg confidence is low (<0.5), sample more diverse tasks
        - If avg confidence is high (>0.7), can sample randomly
        """
        avg_confidence = playbook_stats.get("avg_confidence", 0.5)

        # Low confidence → sample more broadly (use difficulty spread)
        if avg_confidence < 0.5:
            # Try to get diverse difficulties
            task_difficulties = {t: get_task_difficulty(t) for t in untried}
            by_difficulty = {}
            for task_id, diff in task_difficulties.items():
                by_difficulty.setdefault(diff, []).append(task_id)

            selected = []
            per_difficulty = max(1, self.num_tasks_per_iteration // len(by_difficulty))

            for diff, tasks in by_difficulty.items():
                sample_size = min(per_difficulty, len(tasks))
                selected.extend(random.sample(tasks, sample_size))

            # Fill remaining
            if len(selected) < self.num_tasks_per_iteration:
                remaining_tasks = [t for t in untried if t not in selected]
                remaining = self.num_tasks_per_iteration - len(selected)
                selected.extend(random.sample(remaining_tasks, min(remaining, len(remaining_tasks))))

            return selected[:self.num_tasks_per_iteration]

        # High confidence → random sampling is fine
        else:
            sample_size = min(self.num_tasks_per_iteration, len(untried))
            return random.sample(untried, sample_size)

    def _select_high_variance(
        self,
        untried: List[str],
        playbook_stats: Dict[str, Any],
    ) -> List[str]:
        """
        Select diverse tasks to maximize playbook coverage.

        Strategy: Sample from all available difficulties evenly.
        """
        task_difficulties = {t: get_task_difficulty(t) for t in untried}
        by_difficulty = {}
        for task_id, diff in task_difficulties.items():
            by_difficulty.setdefault(diff, []).append(task_id)

        selected = []
        difficulties = list(by_difficulty.keys())

        # Round-robin selection across difficulties
        idx = 0
        while len(selected) < self.num_tasks_per_iteration and any(by_difficulty.values()):
            diff = difficulties[idx % len(difficulties)]
            if by_difficulty[diff]:
                task = random.choice(by_difficulty[diff])
                selected.append(task)
                by_difficulty[diff].remove(task)
            idx += 1

        return selected[:self.num_tasks_per_iteration]

    def _select_least_covered(
        self,
        untried: List[str],
        playbook_stats: Dict[str, Any],
    ) -> List[str]:
        """
        Select tasks where playbook has least coverage.

        Strategy: If playbook is small (<10 bullets), focus on hard tasks to expand coverage.
        Otherwise, sample evenly.
        """
        total_bullets = playbook_stats.get("total_bullets", 0)

        # Small playbook → focus on challenging tasks
        if total_bullets < 10:
            task_difficulties = {t: get_task_difficulty(t) for t in untried}

            # Prioritize hard and medium tasks
            hard_tasks = [t for t, d in task_difficulties.items() if d == "hard"]
            medium_tasks = [t for t, d in task_difficulties.items() if d == "medium"]
            easy_tasks = [t for t, d in task_difficulties.items() if d == "easy"]

            selected = []

            # 60% hard, 30% medium, 10% easy
            hard_count = int(self.num_tasks_per_iteration * 0.6)
            medium_count = int(self.num_tasks_per_iteration * 0.3)
            easy_count = self.num_tasks_per_iteration - hard_count - medium_count

            selected.extend(random.sample(hard_tasks, min(hard_count, len(hard_tasks))))
            selected.extend(random.sample(medium_tasks, min(medium_count, len(medium_tasks))))
            selected.extend(random.sample(easy_tasks, min(easy_count, len(easy_tasks))))

            # Fill remaining
            if len(selected) < self.num_tasks_per_iteration:
                remaining_tasks = [t for t in untried if t not in selected]
                remaining = self.num_tasks_per_iteration - len(selected)
                selected.extend(random.sample(remaining_tasks, min(remaining, len(remaining_tasks))))

            return selected[:self.num_tasks_per_iteration]

        # Large playbook → sample evenly
        else:
            sample_size = min(self.num_tasks_per_iteration, len(untried))
            return random.sample(untried, sample_size)


class AdaptiveQuestionSelector:
    """
    Main adaptive question selector class.

    Manages the adaptive selection workflow:
        1. Select N tasks per iteration using configured algorithm
        2. Train on selected tasks sequentially (generator → reflector → curator)
        3. Update playbook after iteration completes
        4. Repeat with updated playbook

    Usage:
        selector = AdaptiveQuestionSelector(
            algorithm="difficulty_progressive",
            num_tasks_per_iteration=5,
            dataset_path="experiments/data/train.txt",
        )

        while not selector.is_complete():
            iteration = selector.select_next_batch()
            # Train on tasks in iteration...
            # Update playbook...
            selector.mark_batch_complete(iteration)
    """

    ALGORITHMS = {
        "random": RandomSelector,
        "difficulty_progressive": DifficultyProgressiveSelector,
        "uncertainty_based": UncertaintyBasedSelector,
    }

    def __init__(
        self,
        algorithm: str = "random",
        num_tasks_per_iteration: int = 5,
        dataset_path: Optional[str] = None,
        task_ids: Optional[List[str]] = None,
        no_repeat_tasks: bool = True,
        random_seed: Optional[int] = None,
        **algorithm_kwargs,
    ):
        """
        Args:
            algorithm: Selection algorithm to use ("random", "difficulty_progressive", "uncertainty_based")
            num_tasks_per_iteration: Number of tasks to select per iteration
            dataset_path: Path to dataset file containing task IDs
            task_ids: Explicit list of task IDs (alternative to dataset_path)
            no_repeat_tasks: If True, each task can only be used once across all iterations.
                           If False, tasks can be selected multiple times (useful for fine-tuning).
            random_seed: Random seed for reproducibility
            **algorithm_kwargs: Additional kwargs passed to selector algorithm
        """
        if algorithm not in self.ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. Choose from: {list(self.ALGORITHMS.keys())}"
            )

        # Load task IDs
        if dataset_path:
            self.all_tasks = load_dataset(dataset_path)
        elif task_ids:
            self.all_tasks = task_ids
        else:
            raise ValueError("Must provide either dataset_path or task_ids")

        # Initialize selector
        selector_class = self.ALGORITHMS[algorithm]
        self.selector = selector_class(
            num_tasks_per_iteration=num_tasks_per_iteration,
            random_seed=random_seed,
            **algorithm_kwargs,
        )

        # Track state
        self.no_repeat_tasks = no_repeat_tasks
        self.tried_tasks: Set[str] = set()
        self.batch_history: List[SelectionResult] = []
        self.current_playbook: Optional[str] = None
        self.metadata: Dict[str, Any] = {}

    def select_next_batch(
        self,
        playbook_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SelectionResult:
        """
        Select next batch of tasks.

        Args:
            playbook_text: Current playbook content (optional, used by some algorithms)
            metadata: Additional metadata for selection (optional)

        Returns:
            SelectionResult containing selected task IDs and metadata
        """
        # If no_repeat_tasks is True, pass tried_tasks to filter them out
        # If False, pass empty set to allow repeats
        tried_tasks_for_selection = self.tried_tasks if self.no_repeat_tasks else set()

        result = self.selector.select_next_batch(
            available_tasks=self.all_tasks,
            tried_tasks=tried_tasks_for_selection,
            playbook_text=playbook_text or self.current_playbook,
            metadata=metadata or self.metadata,
        )

        self.batch_history.append(result)
        return result

    def mark_batch_complete(
        self,
        batch: SelectionResult,
        playbook_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Mark a batch as complete and update selector state.

        Args:
            batch: The SelectionResult that was completed
            playbook_text: Updated playbook text after training on this batch
            metadata: Additional metadata from training (e.g., costs, success rates)
        """
        # Only track tried tasks if no_repeat_tasks is True
        # This ensures tasks can be selected again if repeats are allowed
        if self.no_repeat_tasks:
            self.tried_tasks.update(batch.task_ids)

        if playbook_text:
            self.current_playbook = playbook_text

        if metadata:
            self.metadata.update(metadata)

    def is_complete(self) -> bool:
        """
        Check if all tasks have been tried.

        If no_repeat_tasks is True, completes when all tasks tried once.
        If no_repeat_tasks is False, never completes (can run indefinitely).
        """
        if not self.no_repeat_tasks:
            # When repeats are allowed, training never auto-completes
            # User must set max_iterations to control stopping
            return False

        return len(self.tried_tasks) >= len(self.all_tasks)

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress statistics"""
        return {
            "total_tasks": len(self.all_tasks),
            "tried_tasks": len(self.tried_tasks),
            "remaining_tasks": len(self.all_tasks) - len(self.tried_tasks),
            "batches_completed": len(self.batch_history),
            "progress_percent": len(self.tried_tasks) / len(self.all_tasks) * 100,
        }

    def reset(self):
        """Reset selector state"""
        self.tried_tasks.clear()
        self.batch_history.clear()
        self.current_playbook = None
        self.metadata.clear()


def create_selector(
    algorithm: str,
    num_tasks_per_iteration: int = 5,
    dataset_path: Optional[str] = None,
    task_ids: Optional[List[str]] = None,
    no_repeat_tasks: bool = True,
    random_seed: Optional[int] = None,
    **kwargs,
) -> AdaptiveQuestionSelector:
    """
    Factory function to create an AdaptiveQuestionSelector.

    Args:
        algorithm: Selection algorithm ("random", "difficulty_progressive", "uncertainty_based")
        num_tasks_per_iteration: Number of tasks to select per iteration
        dataset_path: Path to dataset file
        task_ids: Explicit task ID list
        no_repeat_tasks: If True, each task used only once. If False, tasks can repeat.
        random_seed: Random seed
        **kwargs: Algorithm-specific parameters

    Returns:
        Configured AdaptiveQuestionSelector instance

    Examples:
        # Random selection
        selector = create_selector("random", num_tasks_per_iteration=5, dataset_path="train.txt")

        # Progressive difficulty
        selector = create_selector(
            "difficulty_progressive",
            num_tasks_per_iteration=5,
            dataset_path="train.txt",
            difficulty_schedule="progressive",
        )

        # Uncertainty-based
        selector = create_selector(
            "uncertainty_based",
            num_tasks_per_iteration=5,
            dataset_path="train.txt",
            selection_strategy="low_confidence",
        )
    """
    return AdaptiveQuestionSelector(
        algorithm=algorithm,
        num_tasks_per_iteration=num_tasks_per_iteration,
        dataset_path=dataset_path,
        task_ids=task_ids,
        no_repeat_tasks=no_repeat_tasks,
        random_seed=random_seed,
        **kwargs,
    )
