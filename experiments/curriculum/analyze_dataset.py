#!/usr/bin/env python3
"""
Analyze task difficulty distribution and instruction similarity from a dataset file.

Usage (run from repository root):
    python3 experiments/curriculum/analyze_dataset.py --dataset train.txt
    python3 experiments/curriculum/analyze_dataset.py --dataset train.txt --similarity-metric cosine:0.8
    python3 experiments/curriculum/analyze_dataset.py --dataset train.txt --similarity-metric oracle
    python3 experiments/curriculum/analyze_dataset.py --dataset train.txt --similarity-metric embedding:0.85 --embedding-model text-embedding-3-large

Available similarity metrics:
    - jaccard:THRESHOLD - Word-based set similarity (e.g., jaccard:0.7)
    - cosine:THRESHOLD - TF-IDF cosine similarity (e.g., cosine:0.8)
    - levenshtein:THRESHOLD - Character-level edit distance (e.g., levenshtein:0.75)
    - oracle - Ground truth clustering using task family IDs (e.g., "76f2c72" from "76f2c72_2")
               Note: threshold not needed for oracle since it uses exact family ID matching
    - embedding:THRESHOLD - Semantic similarity using OpenAI embeddings API (e.g., embedding:0.85)
                            Requires: OPENAI_API_KEY environment variable
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from similarity_metrics import calculate_similarity, compute_embeddings_batch, compute_idf_scores


def load_dataset(dataset_name: str) -> List[str]:
    """Load task IDs from dataset file."""
    dataset_file = Path("data/datasets") / dataset_name

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    with open(dataset_file, 'r') as f:
        task_ids = [line.strip() for line in f if line.strip()]
    return task_ids


def get_task_difficulty(task_id: str) -> Tuple[str, int, str]:
    """Get difficulty level and instruction for a task from its metadata files."""
    metadata_path = Path("data/tasks") / task_id / "ground_truth" / "metadata.json"
    specs_path = Path("data/tasks") / task_id / "specs.json"

    try:
        # Get difficulty from metadata.json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            difficulty = metadata.get('difficulty', -1)

        # Get instruction from specs.json
        instruction = ""
        try:
            with open(specs_path, 'r') as f:
                specs = json.load(f)
                instruction = specs.get('instruction', '')
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        return (task_id, difficulty, instruction)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not read metadata for task {task_id}: {e}")
        return (task_id, -1, "")


def analyze_difficulty_distribution(task_ids: List[str]) -> List[Tuple[str, int, str]]:
    """Analyze difficulty distribution and return task-difficulty-instruction tuples."""
    task_data = []

    for task_id in task_ids:
        task_id, difficulty, instruction = get_task_difficulty(task_id)
        task_data.append((task_id, difficulty, instruction))

    return task_data


def print_distribution(task_difficulty_pairs: List[Tuple[str, int, str]]):
    """Print difficulty distribution statistics."""
    # Count by difficulty
    difficulty_counts = {}
    total_tasks = len(task_difficulty_pairs)

    for _, difficulty, _ in task_difficulty_pairs:
        if difficulty >= 0:
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1

    print("\n" + "=" * 60)
    print("DIFFICULTY DISTRIBUTION")
    print("=" * 60)

    print(f"\n{'Difficulty':<12} {'Count':<10} {'Percentage':<12} {'Bar'}")
    print("-" * 60)

    for difficulty in sorted(difficulty_counts.keys()):
        count = difficulty_counts[difficulty]
        percentage = (count / total_tasks) * 100
        bar = "█" * int(percentage / 2)
        print(f"Level {difficulty:<6} {count:<10} {percentage:>6.2f}%     {bar}")

    print("-" * 60)
    print(f"{'Total':<12} {total_tasks:<10} {'100.00%':<12}")
    print()


def print_task_list(task_difficulty_pairs: List[Tuple[str, int, str]]):
    """Print tasks in original order with difficulty level appended."""
    print("\n" + "=" * 60)
    print("TASKS IN ORIGINAL ORDER (with difficulty)")
    print("=" * 60)
    print()

    for task_id, difficulty, _ in task_difficulty_pairs:
        diff_str = f"Level {difficulty}" if difficulty >= 0 else "Unknown"
        print(f"{task_id:<20} -> {diff_str}")

    print()


def _analyze_oracle_similarity(
    task_difficulty_pairs: List[Tuple[str, int, str]],
    threshold: float = 0.7
) -> Dict:
    """Analyze similarity using oracle (task family ID) clustering.

    Note: Threshold is ignored for oracle similarity since it produces binary results
    (1.0 for same family, 0.0 for different families). All tasks with the same family
    ID will always be clustered together.

    Args:
        task_difficulty_pairs: List of (task_id, difficulty, instruction) tuples
        threshold: Ignored for oracle metric

    Returns:
        Dictionary with clustering results based on task family IDs
    """
    from similarity_metrics import extract_task_family_id

    # Group tasks by family ID
    family_to_tasks = {}
    all_tasks_count = 0

    for task_id, difficulty, instruction in task_difficulty_pairs:
        family_id = extract_task_family_id(task_id)
        if family_id not in family_to_tasks:
            family_to_tasks[family_id] = []
        family_to_tasks[family_id].append({
            "task_id": task_id,
            "difficulty": difficulty,
            "instruction": instruction
        })
        all_tasks_count += 1

    # Build clusters (one cluster per family)
    detailed_clusters = []
    for idx, (family_id, tasks) in enumerate(family_to_tasks.items()):
        # Get unique instructions in this family
        unique_instructions = set(task["instruction"] for task in tasks if task["instruction"])

        detailed_clusters.append({
            "cluster_id": idx,
            "family_id": family_id,
            "size": len(tasks),
            "unique_instructions": len(unique_instructions),
            "tasks": tasks
        })

    # Sort clusters by size (largest first)
    detailed_clusters.sort(key=lambda x: x["size"], reverse=True)

    # Get cluster size distribution
    cluster_sizes = Counter(len(cluster["tasks"]) for cluster in detailed_clusters)

    # Count unique instructions across all tasks
    all_instructions = [instruction for _, _, instruction in task_difficulty_pairs if instruction]
    unique_instructions_total = len(set(all_instructions))
    exact_duplicates = len(all_instructions) - unique_instructions_total

    return {
        "total_instructions": all_tasks_count,
        "exact_duplicates": exact_duplicates,
        "unique_instructions": unique_instructions_total,
        "similarity_metric": "oracle",
        "clustering_threshold": "N/A (oracle uses exact family ID matching)",
        "num_clusters": len(detailed_clusters),
        "cluster_size_distribution": dict(cluster_sizes),
        "clusters": detailed_clusters
    }


def analyze_instruction_similarity(
    task_difficulty_pairs: List[Tuple[str, int, str]],
    similarity_metric: str = "jaccard",
    threshold: float = 0.7,
    embedding_model: str = "text-embedding-3-large"
) -> Dict:
    """Analyze instruction similarity patterns in the dataset."""
    # For oracle metric, we cluster by task ID directly
    if similarity_metric == "oracle":
        return _analyze_oracle_similarity(task_difficulty_pairs, threshold)

    # Build mapping of instruction to task IDs
    instruction_to_tasks = {}
    for task_id, difficulty, instruction in task_difficulty_pairs:
        if instruction:
            if instruction not in instruction_to_tasks:
                instruction_to_tasks[instruction] = []
            instruction_to_tasks[instruction].append({
                "task_id": task_id,
                "difficulty": difficulty
            })

    instructions = list(instruction_to_tasks.keys())

    if not instructions:
        return {
            "total_instructions": 0,
            "exact_duplicates": 0,
            "unique_instructions": 0,
            "similarity_metric": similarity_metric,
            "clustering_threshold": threshold,
            "num_clusters": 0,
            "clusters": []
        }

    # Count exact duplicates
    unique_instructions = len(instructions)
    total_instructions_count = sum(len(tasks) for tasks in instruction_to_tasks.values())
    exact_duplicates = total_instructions_count - unique_instructions  # Number of duplicate copies

    # Prepare metric-specific data
    idf_scores = None
    embeddings_cache = None
    api_key = None

    if similarity_metric == "cosine":
        idf_scores = compute_idf_scores(instructions)
    elif similarity_metric == "embedding":
        # Pre-compute all embeddings for efficiency
        print(f"  Computing embeddings for {len(instructions)} unique instructions using model '{embedding_model}'...")
        import os
        api_key = os.environ.get("OPENAI_API_KEY")
        embeddings_cache = compute_embeddings_batch(instructions, api_key, model=embedding_model)
        print(f"  Embeddings computed successfully")

    # Clustering based on similarity
    # Group instructions that are similar to each other
    clusters = []
    clustered = set()

    for i, inst1 in enumerate(instructions):
        if inst1 in clustered:
            continue

        cluster = [inst1]
        clustered.add(inst1)

        for inst2 in instructions[i+1:]:
            if inst2 in clustered:
                continue

            # Check similarity with any instruction in current cluster
            is_similar = False
            for cluster_inst in cluster:
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
                cluster.append(inst2)
                clustered.add(inst2)

        clusters.append(cluster)

    # Get cluster size distribution
    cluster_sizes = Counter(len(cluster) for cluster in clusters)

    # Build detailed cluster information with task IDs
    detailed_clusters = []
    for idx, cluster in enumerate(clusters):
        cluster_tasks = []
        for instruction in cluster:
            for task_info in instruction_to_tasks[instruction]:
                cluster_tasks.append({
                    "task_id": task_info["task_id"],
                    "difficulty": task_info["difficulty"],
                    "instruction": instruction
                })

        detailed_clusters.append({
            "cluster_id": idx,
            "size": len(cluster_tasks),
            "unique_instructions": len(cluster),
            "tasks": cluster_tasks
        })

    # Sort clusters by size (largest first)
    detailed_clusters.sort(key=lambda x: x["size"], reverse=True)

    return {
        "total_instructions": total_instructions_count,
        "exact_duplicates": exact_duplicates,
        "unique_instructions": unique_instructions,
        "similarity_metric": similarity_metric,
        "clustering_threshold": threshold,
        "num_clusters": len(clusters),
        "cluster_size_distribution": dict(cluster_sizes),
        "clusters": detailed_clusters
    }


def print_similarity_analysis(similarity_data: Dict):
    """Print instruction similarity analysis."""
    print("\n" + "=" * 60)
    print("INSTRUCTION CLUSTERING ANALYSIS")
    print("=" * 60)
    print()

    print(f"Total instructions: {similarity_data['total_instructions']}")
    print(f"Unique instructions: {similarity_data['unique_instructions']}")
    print(f"Exact duplicates (copies): {similarity_data['exact_duplicates']}")
    print()

    print(f"Clustering (metric: {similarity_data['similarity_metric']}, threshold ≥{similarity_data['clustering_threshold']}):")
    print(f"  Number of clusters: {similarity_data['num_clusters']}")
    print(f"  Cluster size distribution: {similarity_data['cluster_size_distribution']}")
    print()

    # Show top 10 largest clusters
    print("Top 10 largest clusters:")
    for cluster in similarity_data['clusters'][:10]:
        print(f"  Cluster {cluster['cluster_id']}: {cluster['size']} tasks ({cluster['unique_instructions']} unique instructions)")
        # Show first instruction as example
        if cluster['tasks']:
            example = cluster['tasks'][0]['instruction']
            print(f"    Example: {example[:80]}{'...' if len(example) > 80 else ''}")
    print()


def save_analysis(
    task_difficulty_pairs: List[Tuple[str, int, str]],
    dataset_name: str,
    similarity_data: Dict
):
    """Save formatted analysis to JSON file in datasets folder."""
    output_file = Path("data/datasets") / f"{dataset_name}_analysis.json"

    # Count by difficulty
    difficulty_counts = {}
    total_tasks = len(task_difficulty_pairs)

    for _, difficulty, _ in task_difficulty_pairs:
        if difficulty >= 0:
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1

    # Build distribution
    distribution = {}
    for difficulty in sorted(difficulty_counts.keys()):
        count = difficulty_counts[difficulty]
        percentage = (count / total_tasks) * 100
        distribution[f"level_{difficulty}"] = {
            "count": count,
            "percentage": round(percentage, 2)
        }

    # Filter out unknown difficulty tasks and include instruction text
    tasks_with_difficulty = [
        {"task_id": task_id, "difficulty": difficulty, "instruction": instruction}
        for task_id, difficulty, instruction in task_difficulty_pairs
        if difficulty >= 0
    ]

    # Create output data
    output_data = {
        "dataset": dataset_name,
        "total_tasks": total_tasks,
        "distribution": distribution,
        "instruction_similarity": similarity_data,
        "tasks": tasks_with_difficulty
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nAnalysis saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze task difficulty distribution from a dataset file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with default jaccard metric and threshold
  python3 experiments/curriculum/analyze_dataset.py --dataset train.txt

  # Use cosine similarity with custom threshold
  python3 experiments/curriculum/analyze_dataset.py --dataset train.txt --similarity-metric cosine:0.8

  # Use oracle clustering (task family IDs)
  python3 experiments/curriculum/analyze_dataset.py --dataset train.txt --similarity-metric oracle

  # Use embedding similarity with custom model
  python3 experiments/curriculum/analyze_dataset.py --dataset train.txt --similarity-metric embedding:0.85 --embedding-model text-embedding-3-large

Available similarity metrics:
  - jaccard:THRESHOLD (e.g., jaccard:0.7)
  - cosine:THRESHOLD (e.g., cosine:0.8)
  - levenshtein:THRESHOLD (e.g., levenshtein:0.75)
  - oracle (no threshold needed)
  - embedding:THRESHOLD (e.g., embedding:0.85, requires OPENAI_API_KEY)
        """
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset filename (e.g., train.txt, test.txt)"
    )
    parser.add_argument(
        "--similarity-metric",
        type=str,
        default="jaccard:0.7",
        help="Similarity metric with optional threshold in format 'metric:threshold' (e.g., 'cosine:0.8', 'oracle'). Default: jaccard:0.7"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-large",
        help="Embedding model to use when similarity-metric uses 'embedding' (default: text-embedding-3-large). Other OpenAI options: text-embedding-3-small, text-embedding-ada-002"
    )

    args = parser.parse_args()

    # Parse similarity metric and threshold
    if ':' in args.similarity_metric:
        parts = args.similarity_metric.split(':', 1)
        similarity_metric = parts[0]
        try:
            threshold = float(parts[1])
        except ValueError:
            parser.error(f"Invalid threshold in --similarity-metric argument: {parts[1]}")
    else:
        # For oracle or if user specifies metric without threshold
        similarity_metric = args.similarity_metric
        if similarity_metric == "oracle":
            threshold = 0.0  # Will be ignored for oracle
        else:
            # Default threshold for metrics without explicit threshold
            threshold = 0.7
            print(f"Note: No threshold specified, using default: {threshold}")

    # Validate similarity metric
    valid_metrics = ["jaccard", "cosine", "levenshtein", "oracle", "embedding"]
    if similarity_metric not in valid_metrics:
        parser.error(f"Invalid similarity metric: {similarity_metric}. Must be one of {valid_metrics}")

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    task_ids = load_dataset(args.dataset)
    print(f"Found {len(task_ids)} tasks")

    # Analyze difficulty distribution
    print("Analyzing difficulty levels...")
    task_difficulty_pairs = analyze_difficulty_distribution(task_ids)

    # Print results
    print_distribution(task_difficulty_pairs)
    print_task_list(task_difficulty_pairs)

    # Analyze and print instruction similarity
    print(f"Analyzing instruction similarity (metric: {similarity_metric}, threshold: {threshold})...")
    similarity_data = analyze_instruction_similarity(
        task_difficulty_pairs,
        similarity_metric=similarity_metric,
        threshold=threshold,
        embedding_model=args.embedding_model
    )
    print_similarity_analysis(similarity_data)

    # Save analysis
    dataset_name = Path(args.dataset).stem
    save_analysis(task_difficulty_pairs, dataset_name, similarity_data)


if __name__ == "__main__":
    main()
