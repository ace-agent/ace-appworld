// Adaptive Training Configuration: Difficulty Progressive Selection
//
// This config uses the AdaptiveQuestionSelector with difficulty-based progressive curriculum:
// - Starts with easy tasks, gradually increases to medium, then hard
// - Batch size of 5 tasks per iteration
// - Each batch is fed to generator → reflector → curator pipeline
// - Playbook is updated after each batch

local project_home_path = std.extVar("APPWORLD_PROJECT_PATH");
local experiment_prompts_path = project_home_path + "/experiments/prompts";
local experiment_playbooks_path = project_home_path + "/experiments/playbooks";
local experiment_configs_path = project_home_path + "/experiments/configs";
local experiment_code_path = project_home_path + "/experiments/code";

local generator_model_config = {
    "name": "DeepSeek-V3.1",
    "provider": "sambanova",
    "temperature": 0,
    "seed": 100,
    "stop": ["<|endoftext|>", "<|eot_id|>", "<|start_header_id|>"],
    "logprobs": false,
    "top_logprobs": null,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "n": 1,
    "response_format": {"type": "text"},
    "retry_after_n_seconds": 10,
    "use_cache": true,
    "max_retries": 50,
};

local reflector_model_config = {
    "name": "DeepSeek-V3.1",
    "provider": "sambanova",
    "temperature": 0,
    "seed": 100,
    "stop": ["<|endoftext|>", "<|eot_id|>", "<|start_header_id|>"],
    "logprobs": false,
    "top_logprobs": null,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "n": 1,
    "response_format": {"type": "text"},
    "retry_after_n_seconds": 10,
    "use_cache": true,
    "max_retries": 50,
};

local curator_model_config = {
    "name": "DeepSeek-V3.1",
    "provider": "sambanova",
    "temperature": 0,
    "seed": 100,
    "stop": ["<|endoftext|>", "<|eot_id|>", "<|start_header_id|>"],
    "logprobs": false,
    "top_logprobs": null,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "n": 1,
    "response_format": {"type": "text"},
    "retry_after_n_seconds": 10,
    "use_cache": true,
    "max_retries": 50,
};

{
    "type": "ace",
    "config": {
        "run_type": "ace-adaptive-training",  // New run type for adaptive training

        // Agent configuration (same as standard adaptation)
        "agent": {
            "type": "ace_adaptation_react",
            "generator_model_config": generator_model_config,
            "reflector_model_config": reflector_model_config,
            "curator_model_config": curator_model_config,
            "appworld_config": {
                "random_seed": 123,
            },
            "logger_config": {
                "color": true,
                "verbose": true,
            },
            "generator_prompt_file_path": experiment_prompts_path + "/appworld_react_generator_prompt.txt",
            "reflector_prompt_file_path": experiment_prompts_path + "/appworld_react_reflector_with_gt_prompt.txt",
            "curator_prompt_file_path": experiment_prompts_path + "/appworld_react_curator_prompt.txt",
            "initial_playbook_file_path": experiment_playbooks_path + "/appworld_initial_playbook.txt",
            "trained_playbook_file_path": experiment_playbooks_path + "/appworld_adaptive_trained_difficulty_progressive.txt",
            "ignore_multiple_calls": true,
            "max_steps": 40,
            "max_cost_overall": 1000,
            "max_cost_per_task": 10,
            "log_lm_calls": true,
            "use_gt_code": true
        },

        // Adaptive selector configuration
        "selector": {
            "algorithm": "difficulty_progressive",          // Options: "random", "difficulty_progressive", "uncertainty_based"
            "num_tasks_per_iteration": 5,                  // Number of tasks to select per iteration
            "dataset_path": project_home_path + "/experiments/data/train.txt",  // Path to dataset file
            "no_repeat_tasks": true,                       // If true, each task used only once across iterations
            "random_seed": 42,                             // For reproducibility

            // Algorithm-specific parameters
            "difficulty_schedule": "progressive",          // Options: "progressive", "mixed", "reverse"
            // "mix_ratio": {"easy": 0.5, "medium": 0.3, "hard": 0.2},  // For "mixed" schedule
        },

        // Rollout pruning configuration
        "pruner": {
            "strategy": "failure_first",                   // Options: "random", "failure_first", "high_cost", "diverse", "most_informative"
            "num_rollouts_for_reflection": 3,              // Prune (n×m) rollouts down to k for reflection
            // "random_seed": 42,                          // Optional: for random pruner
        },

        // Training loop parameters
        "num_rollouts_per_task": 1,                        // m: Number of rollouts to generate per task
        "max_iterations": null,                            // null = train on all tasks
        "save_playbook_every_iteration": true,             // Save intermediate playbooks
        "playbook_save_dir": experiment_playbooks_path + "/adaptive_snapshots",
    }
}
