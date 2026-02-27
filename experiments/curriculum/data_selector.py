#!/usr/bin/env python3
"""
Select and reorder tasks from a dataset file based on difficulty and size.

Usage (run from repository root):
    # Standard selection
    python3 experiments/curriculum/data_selector.py --dataset train.txt --output train_subset.txt --size 50 --difficulty balanced --order easy-to-hard

    # Cluster-based selection (ignores --size, respects --difficulty):
    # No filter (all difficulties)
    python3 experiments/curriculum/data_selector.py --dataset train.txt --output train_subset.txt --cluster cosine:0.8 --examples-per-cluster 2 --order original

    # Easy-only (strict - errors if not enough easy samples)
    python3 experiments/curriculum/data_selector.py --dataset train.txt --output train_subset.txt --cluster oracle --examples-per-cluster 3 --difficulty easy-only

    # Easy-preferred (prefers easy, uses closest as fallback)
    python3 experiments/curriculum/data_selector.py --dataset train.txt --output train_subset.txt --cluster cosine:0.8 --examples-per-cluster 2 --difficulty easy-preferred

    # LLM evaluation-based selection (selects incorrectly answered tasks)
    python3 experiments/curriculum/data_selector.py --dataset train.txt --output train_failed.txt --llm-eval --model-name "deepseek-ai/DeepSeek-V3.1" --provider together
"""

import argparse
import json
import random
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import datetime

from similarity_metrics import calculate_similarity, compute_embeddings_batch, compute_idf_scores

# Add parent directories to path to import ace modules
# experiments directory contains appworld_experiments package
sys.path.insert(0, str(Path(__file__).parent.parent))
# src directory contains appworld package
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def load_dataset(dataset_name: str) -> List[str]:
    """Load task IDs from dataset file."""
    dataset_file = Path("data/datasets") / dataset_name

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    with open(dataset_file, 'r') as f:
        task_ids = [line.strip() for line in f if line.strip()]
    return task_ids


def get_task_difficulty(task_id: str) -> int:
    """Get difficulty level for a task from its metadata.json file."""
    metadata_path = Path("data/tasks") / task_id / "ground_truth" / "metadata.json"

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            return metadata.get('difficulty', -1)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not read metadata for task {task_id}: {e}")
        return -1


def get_task_instruction(task_id: str) -> str:
    """Get instruction text for a task from its specs.json file."""
    specs_path = Path("data/tasks") / task_id / "specs.json"

    try:
        with open(specs_path, 'r') as f:
            specs = json.load(f)
            return specs.get('instruction', '')
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not read specs for task {task_id}: {e}")
        return ""


def get_tasks_with_difficulty(task_ids: List[str]) -> List[Tuple[str, int]]:
    """Get list of (task_id, difficulty) tuples."""
    task_difficulty_pairs = []
    for task_id in task_ids:
        difficulty = get_task_difficulty(task_id)
        task_difficulty_pairs.append((task_id, difficulty))
    return task_difficulty_pairs


def filter_by_difficulty(
    task_pairs: List[Tuple[str, int]],
    difficulty_mode: str
) -> List[Tuple[str, int]]:
    """Filter tasks based on difficulty mode."""
    if difficulty_mode is None:
        # No filter specified
        return [p for p in task_pairs if p[1] >= 0]
    elif difficulty_mode in ["easy-only", "easy-preferred"]:
        return [p for p in task_pairs if p[1] == 1]
    elif difficulty_mode in ["medium-only", "medium-preferred"]:
        return [p for p in task_pairs if p[1] == 2]
    elif difficulty_mode in ["hard-only", "hard-preferred"]:
        return [p for p in task_pairs if p[1] == 3]
    elif difficulty_mode in ["balanced", "custom"]:
        # Keep all difficulties for balanced/custom selection
        return [p for p in task_pairs if p[1] >= 0]
    else:
        raise ValueError(f"Unknown difficulty mode: {difficulty_mode}")


def select_tasks(
    task_pairs: List[Tuple[str, int]],
    size: int,
    difficulty_mode: str,
    ratio: str = None
) -> List[Tuple[str, int]]:
    """Select tasks based on size and difficulty distribution."""
    if difficulty_mode in ["balanced", "custom"]:
        # Group by difficulty
        by_difficulty = {1: [], 2: [], 3: []}
        for task_id, diff in task_pairs:
            if diff in by_difficulty:
                by_difficulty[diff].append((task_id, diff))

        # Determine ratio based on mode
        if difficulty_mode == "balanced":
            # Strictly equal distribution: 1:1:1
            ratio_easy = ratio_medium = ratio_hard = 1
        else:  # difficulty_mode == "custom"
            # Custom ratio from --ratio argument
            if ratio:
                try:
                    parts = [int(x) for x in ratio.split(':')]
                    if len(parts) != 3:
                        raise ValueError("Ratio must have 3 parts (easy:medium:hard)")
                    ratio_easy, ratio_medium, ratio_hard = parts
                except ValueError as e:
                    print(f"Warning: Invalid ratio '{ratio}', using equal distribution. Error: {e}")
                    ratio_easy = ratio_medium = ratio_hard = 1
            else:
                # Default for custom mode if no ratio specified
                ratio_easy = ratio_medium = ratio_hard = 1

        # Calculate how many tasks per difficulty level based on ratio
        total_ratio = ratio_easy + ratio_medium + ratio_hard
        count_easy = int(size * ratio_easy / total_ratio)
        count_medium = int(size * ratio_medium / total_ratio)
        count_hard = size - count_easy - count_medium  # Ensure we hit exact size

        selected = []

        # Select tasks according to ratio
        selected.extend(by_difficulty[1][:min(count_easy, len(by_difficulty[1]))])
        selected.extend(by_difficulty[2][:min(count_medium, len(by_difficulty[2]))])
        selected.extend(by_difficulty[3][:min(count_hard, len(by_difficulty[3]))])

        return selected[:size]  # Ensure we don't exceed size
    else:
        # For non-balanced modes, just take first N tasks
        return task_pairs[:size]


def cluster_tasks_by_similarity(
    task_ids: List[str],
    similarity_metric: str,
    threshold: float,
    embedding_model: str = "text-embedding-3-large"
) -> List[List[Dict]]:
    """Cluster tasks by instruction similarity, preserving original order within clusters.

    Returns:
        List of clusters, where each cluster is a list of dicts with 'task_id', 'difficulty',
        'instruction', and 'original_index' keys.
    """
    print(f"Reading task instructions and metadata...")

    # Build task data with original indices
    task_data = []
    for idx, task_id in enumerate(task_ids):
        difficulty = get_task_difficulty(task_id)
        instruction = get_task_instruction(task_id)
        if instruction and difficulty >= 0:  # Only include tasks with valid instruction and difficulty
            task_data.append({
                'task_id': task_id,
                'difficulty': difficulty,
                'instruction': instruction,
                'original_index': idx
            })

    if not task_data:
        print("Warning: No valid tasks found for clustering")
        return []

    print(f"Found {len(task_data)} valid tasks for clustering")

    # Handle oracle clustering (by task family ID)
    if similarity_metric == "oracle":
        from similarity_metrics import extract_task_family_id

        family_to_tasks = {}
        for task in task_data:
            family_id = extract_task_family_id(task['task_id'])
            if family_id not in family_to_tasks:
                family_to_tasks[family_id] = []
            family_to_tasks[family_id].append(task)

        # Sort each cluster by original index to preserve order
        clusters = []
        for family_id in sorted(family_to_tasks.keys()):
            cluster = sorted(family_to_tasks[family_id], key=lambda x: x['original_index'])
            clusters.append(cluster)

        return clusters

    # For other metrics, cluster by instruction similarity
    # Build instruction to tasks mapping
    instruction_to_tasks = {}
    for task in task_data:
        instruction = task['instruction']
        if instruction not in instruction_to_tasks:
            instruction_to_tasks[instruction] = []
        instruction_to_tasks[instruction].append(task)

    instructions = list(instruction_to_tasks.keys())

    # Prepare metric-specific data
    idf_scores = None
    embeddings_cache = None
    api_key = None

    if similarity_metric == "cosine":
        print("Computing IDF scores...")
        idf_scores = compute_idf_scores(instructions)
    elif similarity_metric == "embedding":
        print(f"Computing embeddings for {len(instructions)} unique instructions using model '{embedding_model}'...")
        import os
        api_key = os.environ.get("OPENAI_API_KEY")
        embeddings_cache = compute_embeddings_batch(instructions, api_key, model=embedding_model)
        print("Embeddings computed successfully")

    # Clustering based on similarity
    clusters = []
    clustered = set()

    for i, inst1 in enumerate(instructions):
        if inst1 in clustered:
            continue

        cluster_instructions = [inst1]
        clustered.add(inst1)

        for inst2 in instructions[i+1:]:
            if inst2 in clustered:
                continue

            # Check similarity with any instruction in current cluster
            is_similar = False
            for cluster_inst in cluster_instructions:
                sim_score = calculate_similarity(
                    cluster_inst, inst2, similarity_metric,
                    idf_scores=idf_scores,
                    embeddings_cache=embeddings_cache,
                    api_key=api_key
                )
                if sim_score >= threshold:
                    is_similar = True
                    break

            if is_similar:
                cluster_instructions.append(inst2)
                clustered.add(inst2)

        # Collect all tasks for this cluster and sort by original index
        cluster_tasks = []
        for instruction in cluster_instructions:
            cluster_tasks.extend(instruction_to_tasks[instruction])

        cluster_tasks.sort(key=lambda x: x['original_index'])
        clusters.append(cluster_tasks)

    # Sort clusters by the minimum original index in each cluster
    clusters.sort(key=lambda cluster: min(task['original_index'] for task in cluster))

    return clusters


def select_from_clusters(
    clusters: List[List[Dict]],
    examples_per_cluster: int,
    difficulty_filter: str = None,
    ratio: str = None
) -> Tuple[List[Dict], Dict]:
    """Select examples from each cluster with difficulty filtering, preserving original relative order.

    Args:
        clusters: List of clusters (each cluster is a list of task dicts)
        examples_per_cluster: Number of examples to select from each cluster
        difficulty_filter: Difficulty mode (easy-only, easy-preferred, medium-only, medium-preferred,
                          hard-only, hard-preferred, balanced, custom, or None for no filter)
        ratio: Custom ratio for balanced/custom mode (e.g., "1:2:1")

    Returns:
        Tuple of (selected task dicts sorted by original_index, statistics dict)

    Raises:
        ValueError: If using -only mode and not enough samples of that difficulty are available
    """
    # Parse difficulty filter mode
    if difficulty_filter is None:
        # No difficulty filter - select first N from each cluster
        selected = []
        for cluster in clusters:
            selected.extend(cluster[:examples_per_cluster])
        selected.sort(key=lambda x: x['original_index'])
        stats = {
            'exact_matches': len(selected),
            'approximate_matches': {},
            'total_selected': len(selected)
        }
        return selected, stats

    # Determine target difficulties and strictness based on filter mode
    strict_mode = False
    target_difficulties = []

    if difficulty_filter == "easy-only":
        target_difficulties = [1]
        strict_mode = True
    elif difficulty_filter == "easy-preferred":
        target_difficulties = [1]
        strict_mode = False
    elif difficulty_filter == "medium-only":
        target_difficulties = [2]
        strict_mode = True
    elif difficulty_filter == "medium-preferred":
        target_difficulties = [2]
        strict_mode = False
    elif difficulty_filter == "hard-only":
        target_difficulties = [3]
        strict_mode = True
    elif difficulty_filter == "hard-preferred":
        target_difficulties = [3]
        strict_mode = False
    elif difficulty_filter in ["balanced", "custom"]:
        # Parse ratio
        if difficulty_filter == "balanced":
            ratio_parts = [1, 1, 1]
        else:
            if ratio:
                try:
                    ratio_parts = [int(x) for x in ratio.split(':')]
                    if len(ratio_parts) != 3:
                        print("Warning: Invalid ratio format, using equal distribution")
                        ratio_parts = [1, 1, 1]
                except ValueError:
                    print("Warning: Invalid ratio values, using equal distribution")
                    ratio_parts = [1, 1, 1]
            else:
                ratio_parts = [1, 1, 1]

        # Build target list based on ratio
        total_ratio = sum(ratio_parts)
        for cluster in clusters:
            count_easy = int(examples_per_cluster * ratio_parts[0] / total_ratio)
            count_medium = int(examples_per_cluster * ratio_parts[1] / total_ratio)
            count_hard = examples_per_cluster - count_easy - count_medium
            # We'll handle this per cluster, so just mark that we need balanced selection
        target_difficulties = None  # Special handling for balanced mode
        strict_mode = True  # Balanced mode is strict by default

    # Statistics tracking
    stats = {
        'exact_matches': 0,
        'approximate_matches': {},  # {(requested, actual): count}
        'total_selected': 0
    }

    selected = []

    for cluster in clusters:
        if difficulty_filter in ["balanced", "custom"]:
            # For balanced/custom, try to get proportional representation
            ratio_parts = [1, 1, 1] if difficulty_filter == "balanced" else [int(x) for x in (ratio or "1:1:1").split(':')]
            total_ratio = sum(ratio_parts)
            count_easy = int(examples_per_cluster * ratio_parts[0] / total_ratio)
            count_medium = int(examples_per_cluster * ratio_parts[1] / total_ratio)
            count_hard = examples_per_cluster - count_easy - count_medium

            # Group cluster tasks by difficulty
            by_diff = {1: [], 2: [], 3: []}
            for task in cluster:
                if task['difficulty'] in by_diff:
                    by_diff[task['difficulty']].append(task)

            # Try to select according to ratio
            cluster_selected = []
            needs = {1: count_easy, 2: count_medium, 3: count_hard}

            for diff_level in [1, 2, 3]:
                available = by_diff[diff_level]
                needed = needs[diff_level]
                taken = min(needed, len(available))
                cluster_selected.extend(available[:taken])
                stats['exact_matches'] += taken
                needs[diff_level] -= taken

            # If we still need more, try to fill with closest difficulty
            if len(cluster_selected) < examples_per_cluster:
                for target_diff in [1, 2, 3]:
                    if needs[target_diff] > 0:
                        # Try adjacent difficulties
                        for alt_diff in [target_diff - 1, target_diff + 1, target_diff - 2, target_diff + 2]:
                            if alt_diff in [1, 2, 3] and needs[target_diff] > 0:
                                # Get tasks not already selected
                                available = [t for t in by_diff[alt_diff] if t not in cluster_selected]
                                taken = min(needs[target_diff], len(available))
                                if taken > 0:
                                    cluster_selected.extend(available[:taken])
                                    key = (target_diff, alt_diff)
                                    stats['approximate_matches'][key] = stats['approximate_matches'].get(key, 0) + taken
                                    needs[target_diff] -= taken

            selected.extend(cluster_selected)
            stats['total_selected'] += len(cluster_selected)
        else:
            # For single difficulty modes (-only/-preferred)
            cluster_selected = []
            for task in cluster:
                if len(cluster_selected) >= examples_per_cluster:
                    break
                if task['difficulty'] in target_difficulties:
                    cluster_selected.append(task)
                    stats['exact_matches'] += 1

            # Check if we got enough samples for strict mode
            if strict_mode and len(cluster_selected) < examples_per_cluster:
                # In strict mode, raise error if not enough samples
                diff_level_name = {1: "easy", 2: "medium", 3: "hard"}[target_difficulties[0]]
                raise ValueError(
                    f"Not enough {diff_level_name} samples in cluster. "
                    f"Needed {examples_per_cluster}, found {len(cluster_selected)}. "
                    f"Use '{diff_level_name}-preferred' mode to allow closest matches as fallback."
                )

            # If we need more and in preferred mode, use closest difficulty
            if not strict_mode and len(cluster_selected) < examples_per_cluster:
                remaining = [t for t in cluster if t not in cluster_selected]

                for task in remaining:
                    if len(cluster_selected) >= examples_per_cluster:
                        break

                    # Find closest target difficulty
                    if target_difficulties:
                        closest_target = min(target_difficulties, key=lambda x: abs(x - task['difficulty']))
                        cluster_selected.append(task)
                        key = (closest_target, task['difficulty'])
                        stats['approximate_matches'][key] = stats['approximate_matches'].get(key, 0) + 1

            selected.extend(cluster_selected)
            stats['total_selected'] += len(cluster_selected)

    # Sort all selected tasks by original index to preserve overall order
    selected.sort(key=lambda x: x['original_index'])

    return selected, stats


def order_tasks(
    task_pairs: List[Tuple[str, int]],
    order_mode: str,
    random_seed: int = None
) -> List[Tuple[str, int]]:
    """Order tasks based on the specified mode."""
    if order_mode == "original":
        return task_pairs
    elif order_mode == "easy-to-hard":
        return sorted(task_pairs, key=lambda x: (x[1], x[0]))
    elif order_mode == "hard-to-easy":
        return sorted(task_pairs, key=lambda x: (-x[1], x[0]))
    elif order_mode == "random":
        if random_seed is not None:
            random.seed(random_seed)
        shuffled = task_pairs.copy()
        random.shuffle(shuffled)
        return shuffled
    else:
        raise ValueError(f"Unknown order mode: {order_mode}")


def save_dataset(task_ids: List[str], output_name: str):
    """Save selected task IDs to output file."""
    output_file = Path("data/datasets") / output_name

    with open(output_file, 'w') as f:
        for task_id in task_ids:
            f.write(f"{task_id}\n")

    print(f"Saved {len(task_ids)} tasks to: {output_file}")


def print_difficulty_matching_stats(stats: Dict):
    """Print statistics about difficulty matching in cluster selection."""
    print("\n" + "=" * 60)
    print("DIFFICULTY MATCHING STATISTICS")
    print("=" * 60)

    print(f"Total tasks selected: {stats['total_selected']}")
    print(f"Exact matches: {stats['exact_matches']} ({stats['exact_matches']/stats['total_selected']*100:.1f}%)")

    if stats['approximate_matches']:
        total_approx = sum(stats['approximate_matches'].values())
        print(f"Approximate matches: {total_approx} ({total_approx/stats['total_selected']*100:.1f}%)")
        print("\nApproximate match breakdown:")
        print(f"  {'Requested':<12} {'Actual':<12} {'Count':<8}")
        print("  " + "-" * 32)

        # Sort by requested difficulty, then actual
        for (requested, actual), count in sorted(stats['approximate_matches'].items()):
            req_label = f"Level {requested}"
            act_label = f"Level {actual}"
            print(f"  {req_label:<12} {act_label:<12} {count:<8}")
    else:
        print("Approximate matches: 0 (all exact matches!)")
    print()


def print_summary(task_pairs: List[Tuple[str, int]]):
    """Print summary of selected tasks."""
    total = len(task_pairs)
    if total == 0:
        print("No tasks selected!")
        return

    # Count by difficulty
    counts = {1: 0, 2: 0, 3: 0}
    for _, diff in task_pairs:
        if diff in counts:
            counts[diff] += 1

    print("\nSelection Summary:")
    print(f"Total tasks: {total}")
    print(f"  Easy (Level 1):   {counts[1]:3d} ({counts[1]/total*100:5.1f}%)")
    print(f"  Medium (Level 2): {counts[2]:3d} ({counts[2]/total*100:5.1f}%)")
    print(f"  Hard (Level 3):   {counts[3]:3d} ({counts[3]/total*100:5.1f}%)")


def run_llm_evaluation(
    task_ids: List[str],
    model_name: str,
    provider: str,
    experiment_name: str,
    max_cost_overall: float = 1000.0,
    max_cost_per_task: float = 10.0,
    max_steps: int = 40,
    temperature: float = 0.0,
    use_cache: bool = True,
) -> Tuple[List[str], Dict[str, Dict]]:
    """Run LLM evaluation on tasks and return failed task IDs and detailed results.

    Args:
        task_ids: List of task IDs to evaluate
        model_name: Name of the LLM model (e.g., "deepseek-ai/DeepSeek-V3.1")
        provider: Provider name (e.g., "together", "openai", "sambanova")
        experiment_name: Name for the experiment (used to create output folder)
        max_cost_overall: Maximum overall cost limit
        max_cost_per_task: Maximum cost per task
        max_steps: Maximum number of steps per task
        temperature: Temperature for LLM generation
        use_cache: Whether to use caching

    Returns:
        Tuple of (failed_task_ids, task_results_dict)
    """
    from datetime import datetime

    print(f"\n=== LLM EVALUATION MODE ===")
    print(f"Model: {model_name}")
    print(f"Provider: {provider}")
    print(f"Experiment name: {experiment_name}")
    print(f"Tasks to evaluate: {len(task_ids)}")

    # Import required modules
    try:
        from appworld_experiments.code.ace.evaluation_agent import Agent
        from appworld.evaluator import evaluate_task
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Make sure you're running from the repository root and all dependencies are installed.")
        sys.exit(1)

    # Find the generator prompt and playbook files
    # These are required for SimplifiedReActAgent
    prompt_file = "experiments/prompts/appworld_react_generator_prompt.txt"
    playbook_file = "experiments/playbooks/appworld_offline_trained_no_gt_playbook.txt"

    # Check if files exist
    if not Path(prompt_file).exists():
        print(f"Error: Generator prompt file not found: {prompt_file}")
        sys.exit(1)
    if not Path(playbook_file).exists():
        print(f"Error: Playbook file not found: {playbook_file}")
        sys.exit(1)

    # Create agent configuration
    agent_config = {
        "type": "ace_evaluation_react",
        "generator_prompt_file_path": prompt_file,
        "trained_playbook_file_path": playbook_file,
        "generator_model_config": {
            "name": model_name,
            "provider": provider,
            "temperature": temperature,
            "use_cache": use_cache,
            "max_retries": 50,
            "seed": 100,
        },
        "appworld_config": {
            "random_seed": 123,
        },
        "logger_config": {
            "color": True,
            "verbose": True,
        },
        "max_steps": max_steps,
        "max_cost_overall": max_cost_overall,
        "max_cost_per_task": max_cost_per_task,
        "log_lm_calls": True,
    }

    # Initialize agent using the registry system
    print("Initializing agent...")
    agent = Agent.from_dict(agent_config)

    # Run evaluation on all tasks
    print(f"\nRunning evaluation on {len(task_ids)} tasks...")
    agent.solve_tasks(
        task_ids=task_ids,
        experiment_name=experiment_name,
        num_processes=1,
        process_index=0,
    )

    print("\nEvaluating task results...")
    # Collect evaluation results
    task_results = {}
    failed_task_ids = []

    for task_id in task_ids:
        try:
            # Evaluate the task (returns tuple: test_tracker, report)
            test_tracker, report = evaluate_task(
                task_id=task_id,
                experiment_name=experiment_name,
                suppress_errors=True,
                save_report=True,
            )

            # Convert to dict for storage
            result_dict = test_tracker.to_dict(stats_only=False)
            task_results[task_id] = result_dict

            # Check if task failed
            if not result_dict.get("success", False):
                failed_task_ids.append(task_id)
                print(f"  ✗ {task_id} - FAILED")
            else:
                print(f"  ✓ {task_id} - PASSED")

        except Exception as e:
            print(f"  ✗ {task_id} - ERROR: {e}")
            task_results[task_id] = {
                "success": False,
                "error": str(e),
            }
            failed_task_ids.append(task_id)

    print(f"\nEvaluation complete:")
    print(f"  Total tasks: {len(task_ids)}")
    print(f"  Passed: {len(task_ids) - len(failed_task_ids)}")
    print(f"  Failed: {len(failed_task_ids)}")

    return failed_task_ids, task_results


def save_evaluation_results(
    task_results: Dict[str, Dict],
    experiment_name: str,
    dataset_name: str,
) -> Path:
    """Save detailed evaluation results to a JSON file in the datasets folder.

    Args:
        task_results: Dictionary mapping task_id to evaluation result dict
        experiment_name: Name of the experiment
        dataset_name: Name of the original dataset

    Returns:
        Path to the saved results file
    """
    from datetime import datetime

    # Create results directory under datasets
    results_dir = Path("data/datasets/llm_eval_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{dataset_name.replace('.txt', '')}_{timestamp}.json"
    results_path = results_dir / filename

    # Prepare full results with metadata
    full_results = {
        "experiment_name": experiment_name,
        "dataset_name": dataset_name,
        "timestamp": timestamp,
        "total_tasks": len(task_results),
        "failed_tasks": sum(1 for r in task_results.values() if not r.get("success", False)),
        "passed_tasks": sum(1 for r in task_results.values() if r.get("success", False)),
        "task_results": task_results,
    }

    # Save to file
    with open(results_path, 'w') as f:
        json.dump(full_results, f, indent=2)

    print(f"\nDetailed results saved to: {results_path}")
    return results_path


def main():
    parser = argparse.ArgumentParser(
        description="Select and reorder tasks from a dataset file",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Input dataset filename (e.g., train.txt)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output dataset filename (e.g., train_subset.txt)"
    )
    parser.add_argument(
        "--size",
        type=int,
        required=False,
        help="Number of tasks to select (ignored when using --cluster)"
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=[
            "easy-only", "easy-preferred",
            "medium-only", "medium-preferred",
            "hard-only", "hard-preferred",
            "balanced", "custom"
        ],
        default=None,
        help="Difficulty filter. -only modes require exact matches (error if unavailable). -preferred modes prefer the level but allow closest match. balanced/custom for proportional selection. If not specified, no difficulty filter is applied."
    )
    parser.add_argument(
        "--order",
        type=str,
        choices=["original", "easy-to-hard", "hard-to-easy", "random"],
        default="original",
        help="Task ordering (default: original)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (only used with --order random)"
    )
    parser.add_argument(
        "--ratio",
        type=str,
        default=None,
        help="Difficulty ratio for custom mode (e.g., '1:2:1' for easy:medium:hard)"
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default=None,
        help="Enable cluster-based selection with format 'metric:threshold' (e.g., 'cosine:0.8', 'oracle'). When used, --size is ignored but --difficulty is respected."
    )
    parser.add_argument(
        "--examples-per-cluster",
        type=int,
        default=None,
        help="Number of examples to select from each cluster (required when using --cluster)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-large",
        help="Embedding model to use when --cluster uses 'embedding' metric (default: text-embedding-3-large)"
    )
    parser.add_argument(
        "--llm-eval",
        action="store_true",
        help="Enable LLM evaluation mode: run base LLM on tasks and select failed ones"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="LLM model name for evaluation (e.g., 'deepseek-ai/DeepSeek-V3.1')"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="together",
        help="LLM provider for evaluation (e.g., 'together', 'openai', 'sambanova')"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for LLM evaluation (used to create output folder)"
    )
    parser.add_argument(
        "--max-cost-overall",
        type=float,
        default=1000.0,
        help="Maximum overall cost for LLM evaluation (default: 1000.0)"
    )
    parser.add_argument(
        "--max-cost-per-task",
        type=float,
        default=10.0,
        help="Maximum cost per task for LLM evaluation (default: 10.0)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=40,
        help="Maximum steps per task for LLM evaluation (default: 40)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.llm_eval:
        # LLM evaluation mode
        if not args.model_name:
            parser.error("--model-name is required when using --llm-eval")
        if not args.experiment_name:
            # Generate default experiment name
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.experiment_name = f"llm_eval_{timestamp}"
            print(f"No experiment name provided, using: {args.experiment_name}")
        if args.cluster or args.size or args.examples_per_cluster:
            print("Warning: --cluster, --size, and --examples-per-cluster are ignored in --llm-eval mode")
    elif args.cluster:
        if not args.examples_per_cluster:
            parser.error("--examples-per-cluster is required when using --cluster")
        if args.size:
            print("Warning: --size is ignored when using --cluster mode")
    else:
        if not args.size:
            parser.error("--size is required when not using --cluster mode")
        if args.examples_per_cluster:
            print("Warning: --examples-per-cluster is ignored when not using --cluster mode")

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    task_ids = load_dataset(args.dataset)
    print(f"Found {len(task_ids)} tasks")

    # Branch based on mode
    if args.llm_eval:
        # LLM evaluation mode: run LLM on tasks and select failed ones
        failed_task_ids, task_results = run_llm_evaluation(
            task_ids=task_ids,
            model_name=args.model_name,
            provider=args.provider,
            experiment_name=args.experiment_name,
            max_cost_overall=args.max_cost_overall,
            max_cost_per_task=args.max_cost_per_task,
            max_steps=args.max_steps,
            temperature=0.0,
            use_cache=True,
        )

        # Save detailed evaluation results
        save_evaluation_results(
            task_results=task_results,
            experiment_name=args.experiment_name,
            dataset_name=args.dataset,
        )

        # Get difficulty information for failed tasks
        failed_pairs = get_tasks_with_difficulty(failed_task_ids)

        # Print summary
        print_summary(failed_pairs)

        # Save failed task IDs to output file
        save_dataset(failed_task_ids, args.output)

        print(f"\nLLM evaluation mode complete!")
        print(f"Failed task IDs saved to: data/datasets/{args.output}")
        return

    elif args.cluster:
        # Parse cluster argument
        if ':' in args.cluster:
            parts = args.cluster.split(':', 1)
            similarity_metric = parts[0]
            try:
                threshold = float(parts[1])
            except ValueError:
                parser.error(f"Invalid threshold in --cluster argument: {parts[1]}")
        else:
            # For oracle, threshold is not needed
            similarity_metric = args.cluster
            threshold = 0.0  # Will be ignored for oracle

        # Validate similarity metric
        valid_metrics = ["jaccard", "cosine", "levenshtein", "oracle", "embedding"]
        if similarity_metric not in valid_metrics:
            parser.error(f"Invalid similarity metric: {similarity_metric}. Must be one of {valid_metrics}")

        print(f"\n=== CLUSTER-BASED SELECTION ===")
        print(f"Similarity metric: {similarity_metric}")
        if similarity_metric != "oracle":
            print(f"Threshold: {threshold}")
        print(f"Examples per cluster: {args.examples_per_cluster}")
        if args.difficulty:
            print(f"Difficulty filter: {args.difficulty}")
            if args.difficulty == "custom" and args.ratio:
                print(f"Custom ratio: {args.ratio}")
            elif args.difficulty == "balanced":
                print(f"Using balanced distribution (1:1:1)")
        else:
            print(f"Difficulty filter: None (all difficulties)")

        # Cluster tasks
        clusters = cluster_tasks_by_similarity(
            task_ids,
            similarity_metric,
            threshold,
            embedding_model=args.embedding_model
        )

        print(f"\nClustering complete: {len(clusters)} clusters found")

        # Print cluster statistics
        cluster_sizes = [len(cluster) for cluster in clusters]
        print(f"Cluster size range: {min(cluster_sizes)} to {max(cluster_sizes)}")
        print(f"Average cluster size: {sum(cluster_sizes) / len(cluster_sizes):.1f}")

        # Select from clusters with difficulty filtering
        filter_msg = f"with difficulty filter '{args.difficulty}'" if args.difficulty else "without difficulty filter"
        print(f"\nSelecting {args.examples_per_cluster} example(s) from each cluster {filter_msg}...")
        try:
            selected_tasks, difficulty_stats = select_from_clusters(
                clusters,
                args.examples_per_cluster,
                difficulty_filter=args.difficulty,
                ratio=args.ratio
            )
            print(f"Selected {len(selected_tasks)} tasks total")

            # Print difficulty matching statistics only if filter was applied
            if args.difficulty:
                print_difficulty_matching_stats(difficulty_stats)
        except ValueError as e:
            print(f"\nError: {e}")
            return

        # Convert to task_pairs format for ordering and summary
        selected_pairs = [(task['task_id'], task['difficulty']) for task in selected_tasks]

    else:
        # Standard selection mode
        print("\n=== STANDARD SELECTION ===")

        # Get difficulty information
        print("Reading task metadata...")
        task_pairs = get_tasks_with_difficulty(task_ids)

        # Filter by difficulty
        print(f"Filtering by difficulty: {args.difficulty}")
        filtered_pairs = filter_by_difficulty(task_pairs, args.difficulty)
        print(f"After filtering: {len(filtered_pairs)} tasks")

        # Select tasks
        print(f"Selecting {args.size} tasks...")
        if args.difficulty == "custom":
            if args.ratio:
                print(f"Using custom ratio: {args.ratio}")
            else:
                print("Warning: --ratio not specified for custom mode")
        elif args.difficulty == "balanced":
            print("Using balanced distribution (1:1:1)")
        selected_pairs = select_tasks(filtered_pairs, args.size, args.difficulty, args.ratio)

        if len(selected_pairs) < args.size:
            print(f"Warning: Only {len(selected_pairs)} tasks available, requested {args.size}")

    # Order tasks (applies to both modes)
    print(f"\nOrdering tasks: {args.order}")
    ordered_pairs = order_tasks(selected_pairs, args.order, args.seed)

    # Extract task IDs
    selected_task_ids = [task_id for task_id, _ in ordered_pairs]

    # Print summary
    print_summary(ordered_pairs)

    # Save output
    save_dataset(selected_task_ids, args.output)


if __name__ == "__main__":
    main()
