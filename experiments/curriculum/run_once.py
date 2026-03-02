#!/usr/bin/env python3
"""
Run base ACE on each sample in a training dataset independently and save individual playbooks.

This script runs ACE adaptation on each sample in the training set exactly once,
generating one playbook file per sample. There are no dependencies between samples,
so each sample is processed independently starting with an empty playbook.

For a training set with N samples, this will generate:
- N individual playbook files (one per sample): playbook_<task_id>.txt
- 1 statistics file: run_once_stats.txt
  The stats file contains: sample ID, correctness (✓/✗), playbook filename, line count

Output Structure:
    output_dir/
    ├── playbook_6ea6792_3.txt      # Playbook for task 6ea6792_3
    ├── playbook_a1b2c3d_1.txt      # Playbook for task a1b2c3d_1
    ├── playbook_xyz789_2.txt       # Playbook for task xyz789_2
    ├── ...                          # One file per task
    └── run_once_stats.txt           # Summary statistics

Usage (run from repository root):
    # Full training set (sequential)
    python3 experiments/curriculum/run_once.py \
        --dataset train.txt \
        --output-dir experiments/playbooks/run_once_output \
        --config experiments/configs/ACE_run_once_empty_playbook.jsonnet \
        --experiment-name run_once_experiment

    # Small test with first 5 samples
    python3 experiments/curriculum/run_once.py \
        --dataset train.txt \
        --output-dir experiments/playbooks/run_once_test \
        --config experiments/configs/ACE_run_once_empty_playbook.jsonnet \
        --max-samples 5

    # Parallel processing with 4 workers
    venv/bin/python3 experiments/curriculum/run_once.py \
        --dataset train.txt \
        --output-dir experiments/playbooks/run_once_parallel \
        --config experiments/configs/ACE_run_once_empty_playbook.jsonnet \
        --num-processes 4

    # Auto-detect CPU count for parallel processing
    venv/bin/python3 experiments/curriculum/run_once.py \
        --dataset train_10_easy_only.txt \
        --output-dir experiments/playbooks/run_once_easy \
        --config experiments/configs/ACE_run_once_empty_playbook.jsonnet \
        --num-processes -1

Requirements:
    - Must run from repository root directory
    - Config file must be an ACE adaptation config (run_type: ace-adaptation)
    - Dataset file must exist in data/datasets/
    - Virtual environment should be activated (or use venv/bin/python3)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

# Add parent directories to path to import ace modules
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from appworld.task import Task, load_task_ids
from appworld.evaluator import evaluate_task
from appworld.common.utils import jsonnet_load
from appworld_experiments.code.ace.adaptation_agent import StarAgent


def load_dataset(dataset_name: str) -> List[str]:
    """Load task IDs from dataset file."""
    dataset_file = Path("data/datasets") / dataset_name

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    with open(dataset_file, 'r') as f:
        task_ids = [line.strip() for line in f if line.strip()]
    return task_ids


def load_config(config_path: str) -> dict:
    """Load configuration from jsonnet file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Set environment variable for jsonnet
    os.environ["APPWORLD_PROJECT_PATH"] = str(Path.cwd())

    # Load and parse jsonnet
    config = jsonnet_load(config_path)

    return config


def run_once_on_sample(
    task_id: str,
    agent_config: dict,
    experiment_name: str,
    output_dir: Path,
    sample_index: int,
    total_samples: int
) -> Dict:
    """
    Run ACE on a single sample and save the generated playbook.

    Args:
        task_id: The task ID to run
        agent_config: Agent configuration dictionary
        experiment_name: Name of the experiment
        output_dir: Directory to save playbooks
        sample_index: Index of current sample (0-based)
        total_samples: Total number of samples

    Returns:
        Dictionary with sample statistics
    """
    print(f"\n{'='*80}")
    print(f"Processing sample {sample_index + 1}/{total_samples}: {task_id}")
    print(f"{'='*80}\n")

    # Create a fresh agent instance for this sample (important for multiprocessing)
    agent = StarAgent.from_dict(agent_config)

    # Initialize the logger for single-task processing
    # This prevents NoneType errors in logger.complete_task()
    agent.logger.initialize(
        experiment_name=experiment_name,
        num_tasks=1,  # Each worker processes one task at a time
        num_processes=1,
        process_index=0
    )

    # Reset agent state for this task
    # Note: Agent already loads empty playbook from config (appworld_empty_playbook.txt)
    # But we reset it here to ensure complete independence between samples
    agent.playbook = ''  # Start with empty playbook for each sample
    agent.current_task_index = sample_index

    # Reset the next_global_id counter if it exists (for playbook item IDs)
    if hasattr(agent, 'next_global_id'):
        agent.next_global_id = 1

    print(f"  Initial playbook: '{agent.playbook}' (should be empty)")

    # Run the task
    try:
        agent.solve_task(task_id, experiment_name)

        # Get the generated playbook
        playbook_content = agent.playbook if hasattr(agent, 'playbook') else '(empty)'

        if not playbook_content or playbook_content.strip() == '' or playbook_content == '(empty)':
            print(f"  ⚠️  Warning: Agent playbook is empty! (No playbook generated during execution)")

        # Count playbook lines
        playbook_lines = len(playbook_content.split('\n')) if playbook_content else 0

        # Save playbook to individual file
        playbook_filename = f"playbook_{task_id}.txt"
        playbook_path = output_dir / playbook_filename

        with open(playbook_path, 'w') as f:
            f.write(playbook_content)

        print(f"\n✓ Saved playbook to: {playbook_path}")
        print(f"  Playbook lines: {playbook_lines}")
        print(f"  Playbook preview: {playbook_content[:200]}..." if len(playbook_content) > 200 else f"  Playbook content: {playbook_content}")

        # Evaluate the task to check if it's correct
        # For adaptation agents, evaluation happens inside solve_task, so check if results exist
        try:
            if hasattr(agent, 'test_report') and agent.test_report is not None:
                # Adaptation agent already evaluated - re-evaluate to get test_tracker
                test_tracker, test_report = evaluate_task(task_id, experiment_name)
                is_correct = len(test_tracker.failures) == 0
                print(f"  Task evaluation: {'✓ PASSED' if is_correct else '✗ FAILED'}")
            else:
                # No evaluation done, evaluate now
                test_tracker, test_report = evaluate_task(task_id, experiment_name)
                is_correct = len(test_tracker.failures) == 0
                print(f"  Task evaluation: {'✓ PASSED' if is_correct else '✗ FAILED'}")
        except Exception as e:
            print(f"  Warning: Could not evaluate task {task_id}: {e}")
            is_correct = False

        return {
            'sample_id': task_id,
            'correct': is_correct,
            'playbook_filename': playbook_filename,
            'playbook_lines': playbook_lines,
            'error': None
        }

    except Exception as e:
        print(f"\n✗ Error processing sample {task_id}: {e}")
        import traceback
        traceback.print_exc()

        # Try to save playbook even if there was an error
        # The agent may have generated partial playbook before crashing
        playbook_filename = None
        playbook_lines = 0
        try:
            if hasattr(agent, 'playbook') and agent.playbook:
                playbook_content = agent.playbook
                playbook_lines = len(playbook_content.split('\n')) if playbook_content else 0
                playbook_filename = f"playbook_{task_id}.txt"
                playbook_path = output_dir / playbook_filename
                with open(playbook_path, 'w') as f:
                    f.write(playbook_content)
                print(f"  ⚠️  Saved partial playbook ({playbook_lines} lines) despite error")
        except Exception as save_error:
            print(f"  Could not save playbook: {save_error}")

        return {
            'sample_id': task_id,
            'correct': False,
            'playbook_filename': playbook_filename,
            'playbook_lines': playbook_lines,
            'error': str(e)
        }


def save_stats(stats: List[Dict], output_dir: Path):
    """
    Save statistics to run_once_stats.txt file.

    Args:
        stats: List of sample statistics
        output_dir: Directory to save stats file
    """
    stats_file = output_dir / "run_once_stats.txt"

    with open(stats_file, 'w') as f:
        # Write header
        f.write("="*100 + "\n")
        f.write("RUN ONCE STATISTICS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*100 + "\n\n")

        # Write column headers
        f.write(f"{'Sample ID':<30} {'Correct':<10} {'Playbook File':<40} {'Lines':<10}\n")
        f.write("-"*100 + "\n")

        # Write each sample's stats
        for stat in stats:
            sample_id = stat['sample_id']
            correct = '✓' if stat['correct'] else '✗'
            playbook_file = stat['playbook_filename'] if stat['playbook_filename'] else 'N/A (error)'
            lines = stat['playbook_lines']

            f.write(f"{sample_id:<30} {correct:<10} {playbook_file:<40} {lines:<10}\n")

            if stat['error']:
                f.write(f"  Error: {stat['error']}\n")

        # Write summary
        f.write("\n" + "="*100 + "\n")
        f.write("SUMMARY\n")
        f.write("="*100 + "\n")

        total = len(stats)
        correct = sum(1 for s in stats if s['correct'])
        errors = sum(1 for s in stats if s['error'])

        f.write(f"Total samples: {total}\n")
        f.write(f"Correct: {correct} ({100*correct/total:.1f}%)\n")
        f.write(f"Incorrect: {total - correct} ({100*(total-correct)/total:.1f}%)\n")
        f.write(f"Errors: {errors}\n")

        total_lines = sum(s['playbook_lines'] for s in stats)
        avg_lines = total_lines / total if total > 0 else 0
        f.write(f"Total playbook lines: {total_lines}\n")
        f.write(f"Average lines per playbook: {avg_lines:.1f}\n")

    print(f"\n✓ Saved statistics to: {stats_file}")


def process_sample_wrapper(args_tuple):
    """
    Wrapper function for multiprocessing.Pool.map().

    Args:
        args_tuple: Tuple of (task_id, sample_index, agent_config, experiment_name,
                             output_dir, total_samples)

    Returns:
        Dictionary with sample statistics
    """
    task_id, sample_index, agent_config, experiment_name, output_dir, total_samples = args_tuple

    return run_once_on_sample(
        task_id=task_id,
        agent_config=agent_config,
        experiment_name=experiment_name,
        output_dir=output_dir,
        sample_index=sample_index,
        total_samples=total_samples
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run base ACE on each sample independently and save individual playbooks"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset file name (e.g., train.txt) from data/datasets/"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for playbooks and stats"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to ACE configuration jsonnet file"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name (default: run_once_TIMESTAMP)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=1,
        help="Number of parallel processes (default: 1, use -1 for auto-detect CPU count)"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"run_once_{timestamp}"
    else:
        experiment_name = args.experiment_name

    print(f"\n{'='*80}")
    print(f"RUN ONCE - ACE Individual Sample Training")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {output_dir}")
    print(f"Config: {args.config}")
    print(f"Experiment name: {experiment_name}")
    print(f"{'='*80}\n")

    # Load dataset
    print("Loading dataset...")
    task_ids = load_dataset(args.dataset)

    if args.max_samples:
        task_ids = task_ids[:args.max_samples]

    print(f"✓ Loaded {len(task_ids)} samples\n")

    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)

    # Verify it's an adaptation config
    if config.get('type') != 'ace':
        raise ValueError(f"Config type must be 'ace', got: {config.get('type')}")

    run_type = config['config'].get('run_type')
    if run_type != 'ace-adaptation':
        print(f"Warning: Expected run_type 'ace-adaptation', got '{run_type}'")

    agent_config = config['config']['agent']

    # Verify the config uses an empty playbook
    initial_playbook_path = agent_config.get('initial_playbook_file_path', '')
    if 'empty' not in initial_playbook_path.lower():
        print(f"⚠️  Warning: Config doesn't use empty playbook: {initial_playbook_path}")
        print(f"⚠️  Recommended: Use experiments/configs/ACE_run_once_empty_playbook.jsonnet")
        print(f"⚠️  Each sample should start with an empty playbook for independence.\n")

    print("✓ Configuration loaded\n")

    # Determine number of processes
    num_processes = args.num_processes
    if num_processes == -1:
        num_processes = cpu_count()
    num_processes = max(1, min(num_processes, len(task_ids)))

    print(f"Using {num_processes} process(es) for parallel execution\n")

    # Process samples
    stats = []
    total_samples = len(task_ids)

    if num_processes == 1:
        # Sequential processing
        print("Processing samples sequentially...\n")
        for i, task_id in enumerate(task_ids):
            stat = run_once_on_sample(
                task_id=task_id,
                agent_config=agent_config,
                experiment_name=experiment_name,
                output_dir=output_dir,
                sample_index=i,
                total_samples=total_samples
            )
            stats.append(stat)

            # Save stats after each sample (incremental save)
            save_stats(stats, output_dir)
    else:
        # Parallel processing
        print(f"Processing samples in parallel with {num_processes} workers...\n")

        # Prepare arguments for multiprocessing
        process_args = [
            (task_id, i, agent_config, experiment_name, output_dir, total_samples)
            for i, task_id in enumerate(task_ids)
        ]

        # Use multiprocessing Pool
        with Pool(processes=num_processes) as pool:
            # Use imap_unordered for better progress tracking
            for stat in pool.imap_unordered(process_sample_wrapper, process_args):
                stats.append(stat)

                # Save stats after each completed sample
                # Sort stats by sample_id for consistent output
                sorted_stats = sorted(stats, key=lambda s: s['sample_id'])
                save_stats(sorted_stats, output_dir)

                print(f"\n✓ Completed {len(stats)}/{total_samples} samples")

        # Final sort by sample_id
        stats = sorted(stats, key=lambda s: s['sample_id'])
        save_stats(stats, output_dir)

    # Final summary
    print(f"\n{'='*80}")
    print("ALL SAMPLES PROCESSED")
    print(f"{'='*80}")
    correct = sum(1 for s in stats if s['correct'])
    print(f"Total: {len(stats)}, Correct: {correct}, Incorrect: {len(stats) - correct}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
