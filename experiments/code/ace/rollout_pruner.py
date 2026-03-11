"""
Rollout Pruning Strategies

This module provides strategies for pruning rollouts before sending to curator.
After generating N rollouts, we select the top K rollouts based on various heuristics.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import random


@dataclass
class RolloutInfo:
    """Information about a single rollout"""
    task_id: str
    success: bool  # Whether task completed successfully
    cost: float  # Token cost
    num_steps: int  # Number of ReAct steps taken
    test_failures: int  # Number of test failures
    reflection: Optional[str] = None  # Reflection text (if any)
    metadata: Dict[str, Any] = None  # Additional metadata

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BasePruner(ABC):
    """Base class for rollout pruning strategies"""

    def __init__(self, num_rollouts_for_reflection: int):
        """
        Args:
            num_rollouts_for_reflection: Number of rollouts to send to reflector
                                         (prune N rollouts down to K for reflection)
        """
        self.num_rollouts_for_reflection = num_rollouts_for_reflection

    @abstractmethod
    def prune(self, rollouts: List[RolloutInfo]) -> List[RolloutInfo]:
        """
        Select top K rollouts from N rollouts to send to reflector.

        Args:
            rollouts: List of all rollouts from iteration

        Returns:
            Pruned list of rollouts (length <= num_rollouts_for_reflection)
        """
        pass


class RandomPruner(BasePruner):
    """
    Random pruning (baseline).

    Randomly selects K rollouts from N.
    """

    def __init__(self, num_rollouts_for_reflection: int, random_seed: Optional[int] = None):
        super().__init__(num_rollouts_for_reflection)
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)

    def prune(self, rollouts: List[RolloutInfo]) -> List[RolloutInfo]:
        """Randomly select K rollouts"""
        if len(rollouts) <= self.num_rollouts_for_reflection:
            return rollouts

        return random.sample(rollouts, self.num_rollouts_for_reflection)


class FailureFirstPruner(BasePruner):
    """
    Prioritize failed tasks for reflection.

    Rationale: Learning from failures is more valuable than successes.
    """

    def prune(self, rollouts: List[RolloutInfo]) -> List[RolloutInfo]:
        """Select failed tasks first, then fill with successes if needed"""
        if len(rollouts) <= self.num_rollouts_for_reflection:
            return rollouts

        # Split into failures and successes
        failures = [r for r in rollouts if not r.success or r.test_failures > 0]
        successes = [r for r in rollouts if r.success and r.test_failures == 0]

        # Prioritize failures
        selected = []

        # Take all failures (up to limit)
        selected.extend(failures[:self.num_rollouts_for_reflection])

        # Fill remaining slots with successes
        remaining = self.num_rollouts_for_reflection - len(selected)
        if remaining > 0:
            selected.extend(successes[:remaining])

        return selected


class HighCostPruner(BasePruner):
    """
    Prioritize high-cost (complex) tasks for reflection.

    Rationale: Tasks that required more tokens/steps are more complex
    and may provide more valuable learning signals.
    """

    def prune(self, rollouts: List[RolloutInfo]) -> List[RolloutInfo]:
        """Select tasks with highest cost"""
        if len(rollouts) <= self.num_rollouts_for_reflection:
            return rollouts

        # Sort by cost (descending)
        sorted_rollouts = sorted(rollouts, key=lambda r: r.cost, reverse=True)

        return sorted_rollouts[:self.num_rollouts_for_reflection]


class DiversePruner(BasePruner):
    """
    Maximize diversity in selected rollouts.

    Strategy: Select rollouts with different outcomes
    - Some failures, some successes
    - Mix of high and low cost
    """

    def prune(self, rollouts: List[RolloutInfo]) -> List[RolloutInfo]:
        """Select diverse rollouts"""
        if len(rollouts) <= self.num_rollouts_for_reflection:
            return rollouts

        # Split into categories
        failures = [r for r in rollouts if not r.success or r.test_failures > 0]
        successes = [r for r in rollouts if r.success and r.test_failures == 0]

        # Aim for 60% failures, 40% successes (if possible)
        num_failures = min(
            int(self.num_rollouts_for_reflection * 0.6),
            len(failures)
        )
        num_successes = self.num_rollouts_for_reflection - num_failures

        # If not enough failures, take more successes
        if num_failures < int(self.num_rollouts_for_reflection * 0.6):
            num_successes = min(
                self.num_rollouts_for_reflection - num_failures,
                len(successes)
            )

        selected = []

        # Select failures (prioritize high cost)
        if failures:
            sorted_failures = sorted(failures, key=lambda r: r.cost, reverse=True)
            selected.extend(sorted_failures[:num_failures])

        # Select successes (mix of high and low cost)
        if successes and num_successes > 0:
            sorted_successes = sorted(successes, key=lambda r: r.cost, reverse=True)
            # Take half high-cost, half low-cost
            high_cost_count = num_successes // 2
            low_cost_count = num_successes - high_cost_count

            selected.extend(sorted_successes[:high_cost_count])
            selected.extend(sorted_successes[-low_cost_count:] if low_cost_count > 0 else [])

        return selected[:self.num_rollouts_for_reflection]


class MostInformativePruner(BasePruner):
    """
    Select most informative rollouts.

    Heuristic:
    - Failures (highest priority) - most learning potential
    - High-cost successes - complex tasks
    - Other successes
    """

    def prune(self, rollouts: List[RolloutInfo]) -> List[RolloutInfo]:
        """Select most informative rollouts"""
        if len(rollouts) <= self.num_rollouts_for_reflection:
            return rollouts

        # Categorize rollouts
        failures = [
            r for r in rollouts
            if not r.success or r.test_failures > 0
        ]
        successes = [r for r in rollouts if r.success and r.test_failures == 0]

        # Sort by cost within each category
        sorted_failures = sorted(failures, key=lambda r: r.cost, reverse=True)
        sorted_successes = sorted(successes, key=lambda r: r.cost, reverse=True)

        # Select in priority order
        selected = []

        # Priority 1: All failures
        selected.extend(sorted_failures[:self.num_rollouts_for_reflection])

        # Priority 2: High-cost successes
        if len(selected) < self.num_rollouts_for_reflection:
            remaining = self.num_rollouts_for_reflection - len(selected)
            selected.extend(sorted_successes[:remaining])

        return selected


# Registry of available pruners
PRUNER_REGISTRY = {
    "random": RandomPruner,
    "failure_first": FailureFirstPruner,
    "high_cost": HighCostPruner,
    "diverse": DiversePruner,
    "most_informative": MostInformativePruner,
}


def create_pruner(
    strategy: str,
    num_rollouts_for_reflection: int,
    **kwargs
) -> BasePruner:
    """
    Factory function to create a rollout pruner.

    Args:
        strategy: Pruning strategy name
        num_rollouts_for_reflection: Number of rollouts to send to reflector (prune N → K)
        **kwargs: Additional kwargs for specific pruner

    Returns:
        Configured pruner instance

    Example:
        pruner = create_pruner("failure_first", num_rollouts_for_reflection=3)
        pruned = pruner.prune(all_rollouts)  # Select 3 rollouts to send to reflector
    """
    if strategy not in PRUNER_REGISTRY:
        raise ValueError(
            f"Unknown pruning strategy '{strategy}'. "
            f"Choose from: {list(PRUNER_REGISTRY.keys())}"
        )

    pruner_class = PRUNER_REGISTRY[strategy]
    return pruner_class(num_rollouts_for_reflection=num_rollouts_for_reflection, **kwargs)
