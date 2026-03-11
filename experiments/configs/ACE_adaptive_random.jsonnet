// Adaptive Training Configuration: Random Selection (Baseline)
//
// This config uses the AdaptiveQuestionSelector with random selection (no repeat):
// - Randomly selects tasks without replacement
// - Batch size of 5 tasks per iteration
// - Serves as baseline for comparison with smarter selection algorithms

local project_home_path = std.extVar("APPWORLD_PROJECT_PATH");
local experiment_prompts_path = project_home_path + "/experiments/prompts";
local experiment_playbooks_path = project_home_path + "/experiments/playbooks";
local experiment_configs_path = project_home_path + "/experiments/configs";
local experiment_code_path = project_home_path + "/experiments/code";

local generator_model_config = {
    // "name": "DeepSeek-V3.1",
    // "provider": "sambanova",
    "name": "deepseek-ai/DeepSeek-V3.1",
    "provider": "together",
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
    // "name": "DeepSeek-V3.1",
    // "provider": "sambanova",
    "name": "deepseek-ai/DeepSeek-V3.1",
    "provider": "together",
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
    // "name": "DeepSeek-V3.1",
    // "provider": "sambanova",
    "name": "deepseek-ai/DeepSeek-V3.1",
    "provider": "together",
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
        "run_type": "ace-adaptive-training",

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
            "trained_playbook_file_path": experiment_playbooks_path + "/appworld_adaptive_trained_random.txt",
            "ignore_multiple_calls": true,
            "max_steps": 40,
            "max_cost_overall": 1000,
            "max_cost_per_task": 10,
            "log_lm_calls": true,
            "use_reflector": true,
            "use_gt_code": true
        },

        "selector": {
            "algorithm": "random",
            "num_tasks_per_iteration": 5,
            "dataset_path": project_home_path + "/experiments/data/train.txt",
            "random_seed": 42,
        },

        "max_iterations": null,
        "save_playbook_every_iteration": true,
        "playbook_save_dir": experiment_playbooks_path + "/adaptive_snapshots",
    }
}
