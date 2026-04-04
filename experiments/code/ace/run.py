from typing import Any

from appworld.task import Task, load_task_ids
from appworld_experiments.code.ace.base_agent import BaseAgent
from appworld_experiments.code.ace.evaluation_agent import Agent
from appworld_experiments.code.ace.adaptation_agent import StarAgent

def run_experiment(
    experiment_name: str,
    runner_config: dict[str, Any],
    task_id: str | None = None,
    num_processes: int = 1,
    process_index: int = 0,
    batch_size: int = 1,
    training_size: int | None = None,
    test_size: int | None = None,
    valid_size: int | None = None,
) -> None:
    run_type = runner_config.pop("run_type")
    agent_config = runner_config.pop("agent")
    dataset_name = runner_config.pop("dataset", None)
    sample_size = runner_config.pop("sample_size", None)
    custom_task_ids = runner_config.pop("task_ids", None)
    num_epochs = runner_config.pop("num_epochs", 1)
    # batch_size, curator_batch_size, augmented_shuffling (ace / ACEBatch parity)
    config_batch_size = runner_config.pop("batch_size", None)
    if config_batch_size is not None:
        batch_size = int(config_batch_size)
    curator_batch_size = runner_config.pop("curator_batch_size", None)
    if curator_batch_size is not None:
        curator_batch_size = int(curator_batch_size)
    augmented_shuffling = runner_config.pop("augmented_shuffling", True)
    augmented_shuffling_factor = runner_config.pop("augmented_shuffling_factor", None)

    if runner_config:
        raise Exception(f"Unexpected keys in the runner config: {runner_config}")
    
    # Determine the size limit based on dataset name and CLI args
    dataset_size_map = {
        "train": training_size,
        "dev": valid_size,
        "test_normal": test_size,
        "test_challenge": test_size,
    }
    cli_size = dataset_size_map.get(dataset_name) if dataset_name else None

    if task_id:
        task_ids = [task_id] # execute a single task
    elif custom_task_ids:
        task_ids = custom_task_ids # use a custom list of tasks
        print(f"Using custom task list: {len(task_ids)} tasks")
    else:
        if dataset_name is None:
            raise Exception("Either 'dataset' or 'task_ids' must be specified in the config")
        task_ids = load_task_ids(dataset_name)
        if sample_size is not None:
            task_ids = task_ids[:sample_size]
        if cli_size is not None:
            task_ids = task_ids[:cli_size]

    # Make sure all the tasks can be loaded without running any of them
    for task_id in task_ids:
        Task.load(task_id=task_id)

    task_ids = task_ids * num_epochs

    if run_type == "ace-adaptation":
        # ACE adaptation — use parallel agent when batch_size > 1
        if batch_size > 1:
            import appworld_experiments.code.ace.adaptation_react_parallel  # noqa: F401 — ensure registration
            agent_config = dict(agent_config, type="ace_adaptation_react_parallel")
            if curator_batch_size is not None:
                agent_config = dict(agent_config, curator_batch_size=curator_batch_size)
            if not augmented_shuffling:
                aug_f = 1
            elif augmented_shuffling_factor is not None:
                aug_f = max(1, int(augmented_shuffling_factor))
            else:
                aug_f = 2
            agent_config = dict(agent_config, augmented_shuffling_factor=aug_f)
        agent = StarAgent.from_dict(agent_config)
    elif run_type == "ace-evaluation":
        # ACE evaluation
        agent = Agent.from_dict(agent_config)
    elif run_type == "non-ace-evaluation":
        # non-ACE evaluation
        agent = BaseAgent.from_dict(agent_config)
    else:
        raise ValueError(f"Unknown run_type: {run_type}")

    agent.solve_tasks(
        task_ids=task_ids,
        experiment_name=experiment_name,
        num_processes=num_processes,
        process_index=process_index,
        batch_size=batch_size,
    )