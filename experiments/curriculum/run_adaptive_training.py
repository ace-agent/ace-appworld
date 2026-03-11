#!/usr/bin/env python3
"""
Run Adaptive Training Experiment

This script runs the adaptive training loop using AdaptiveQuestionSelector
for curriculum learning.

IMPORTANT NOTES:
- num_tasks_per_iteration: How many tasks to select per iteration (e.g., 5)
- Tasks within an iteration are processed SEQUENTIALLY (one after another)

Usage:
    # Basic usage
    python run_adaptive_training.py --config experiments/configs/ACE_adaptive_difficulty_progressive.jsonnet

    # With custom parameters
    python run_adaptive_training.py \
        --config experiments/configs/ACE_adaptive_random.jsonnet \
        --max-iterations 10 \
        --num-tasks-per-iteration 3 \
        --num-rollouts-for-reflection 5 \
        --pruning-strategy failure_first \
        --dataset data/datasets/train.txt \
        --experiment-name my_experiment
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project to path
project_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_path))

import _jsonnet
from appworld_experiments.code.ace.adaptive_training_loop import run_adaptive_training


def load_config(config_path: str) -> dict:
    """Load and parse Jsonnet config file"""
    # Get project path - try environment variable first, then use current path
    appworld_project_path = os.environ.get('APPWORLD_PROJECT_PATH', str(project_path))

    # Parse jsonnet with external variables
    ext_vars = {
        'APPWORLD_PROJECT_PATH': appworld_project_path,
        'PROJECT_HOME_PATH': appworld_project_path,  # Support both variable names
    }
    config_str = _jsonnet.evaluate_file(config_path, ext_vars=ext_vars)
    config = json.loads(config_str)

    return config


def main():
    parser = argparse.ArgumentParser(description='Run adaptive training experiment')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file (jsonnet or json)',
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Experiment name (overrides config)',
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=None,
        help='Maximum number of iterations to process (overrides config)',
    )
    parser.add_argument(
        '--num-tasks-per-iteration',
        type=int,
        default=None,
        help='Number of tasks to select per iteration (overrides config)',
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['random', 'difficulty_progressive', 'uncertainty_based'],
        default=None,
        help='Selection algorithm (overrides config)',
    )
    parser.add_argument(
        '--num-rollouts-for-reflection',
        type=int,
        default=None,
        help='Number of rollouts to keep after pruning (k parameter, overrides config)',
    )
    parser.add_argument(
        '--pruning-strategy',
        type=str,
        choices=['random', 'failure_first', 'high_cost', 'diverse', 'most_informative'],
        default=None,
        help='Pruning strategy to use (overrides config)',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Path to dataset file (overrides config)',
    )

    args = parser.parse_args()

    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Validate config structure
    if config.get('type') != 'ace':
        raise ValueError(f"Expected config type 'ace', got '{config.get('type')}'")

    if config['config'].get('run_type') != 'ace-adaptive-training':
        print(f"Warning: config run_type is '{config['config'].get('run_type')}', "
              f"expected 'ace-adaptive-training'")

    # Extract configs
    agent_config = config['config']['agent']
    selector_config = config['config'].get('selector', {})
    pruner_config = config['config'].get('pruner')  # Optional

    # Override with command-line args
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        experiment_name = None

    if args.max_iterations is not None:
        max_iterations = args.max_iterations
    else:
        max_iterations = config['config'].get('max_iterations')

    if args.num_tasks_per_iteration is not None:
        selector_config['num_tasks_per_iteration'] = args.num_tasks_per_iteration

    if args.algorithm is not None:
        selector_config['algorithm'] = args.algorithm

    if args.dataset is not None:
        selector_config['dataset_path'] = args.dataset

    if args.num_rollouts_for_reflection is not None:
        if pruner_config is None:
            pruner_config = {}
        pruner_config['num_rollouts_for_reflection'] = args.num_rollouts_for_reflection

    if args.pruning_strategy is not None:
        if pruner_config is None:
            pruner_config = {}
        pruner_config['strategy'] = args.pruning_strategy

    # Ensure pruner has a strategy if num_rollouts_for_reflection is set
    if pruner_config is not None and 'num_rollouts_for_reflection' in pruner_config:
        if 'strategy' not in pruner_config:
            pruner_config['strategy'] = 'random'  # Default to random pruning
            print(f"Warning: num_rollouts_for_reflection set without strategy, defaulting to 'random'")

    num_rollouts_per_task = config['config'].get('num_rollouts_per_task', 1)
    save_playbook_every_iteration = config['config'].get('save_playbook_every_iteration', True)
    playbook_save_dir = config['config'].get('playbook_save_dir')
    enable_logging = config['config'].get('enable_logging', True)
    log_base_dir = config['config'].get('log_base_dir', 'experiments/logs')

    # Print configuration summary
    print("\n" + "="*80)
    print("ADAPTIVE TRAINING CONFIGURATION")
    print("="*80)
    print(f"Algorithm: {selector_config.get('algorithm', 'not specified')}")
    print(f"Tasks per iteration (n): {selector_config.get('num_tasks_per_iteration', 'not specified')}")
    print(f"Rollouts per task (m): {num_rollouts_per_task}")
    print(f"Total rollouts per iteration: {selector_config.get('num_tasks_per_iteration', '?')} × {num_rollouts_per_task} = {selector_config.get('num_tasks_per_iteration', 0) * num_rollouts_per_task if selector_config.get('num_tasks_per_iteration') else '?'}")
    print(f"Dataset: {selector_config.get('dataset_path', 'not specified')}")
    print(f"Max iterations: {max_iterations if max_iterations is not None else 'unlimited'}")
    if pruner_config:
        print(f"\nPruning Strategy: {pruner_config.get('strategy', 'not specified')}")
        print(f"Rollouts for reflection (k): {pruner_config.get('num_rollouts_for_reflection', 'not specified')}")
    else:
        print(f"\nPruning: disabled (all rollouts sent to reflector)")
    print(f"\nExperiment name: {experiment_name or 'default'}")
    print(f"Save playbook every iteration: {save_playbook_every_iteration}")
    print(f"\nLogging: {'enabled' if enable_logging else 'disabled'}")
    if enable_logging:
        print(f"Experiment directory: {log_base_dir}/<experiment_name>_<timestamp>/")
        print(f"  - playbooks/           (playbook versions at each iteration)")
        print(f"  - iterations/          (iteration summaries)")
        print(f"  - rollouts/            (rollout details)")
        print(f"  - reflections/         (reflections)")
        print(f"  - playbook_analysis/   (rule contributions)")
        print(f"  - summary.json         (final summary)")
    elif playbook_save_dir:
        print(f"Playbook save directory: {playbook_save_dir}")
    print("="*80 + "\n")

    # Run training
    try:
        loop = run_adaptive_training(
            agent_config=agent_config,
            selector_config=selector_config,
            pruner_config=pruner_config,
            num_rollouts_per_task=num_rollouts_per_task,
            experiment_name=experiment_name,
            max_iterations=max_iterations,
            save_playbook_every_iteration=save_playbook_every_iteration,
            playbook_save_dir=playbook_save_dir,
            enable_logging=enable_logging,
            log_base_dir=log_base_dir,
        )

        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        progress = loop.selector.get_progress()
        print(f"Total tasks processed: {progress['tried_tasks']}/{progress['total_tasks']}")
        print(f"Total iterations: {loop.iteration_count}")
        print(f"Final playbook saved to: {agent_config.get('trained_playbook_file_path', 'not specified')}")
        print("="*80 + "\n")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
