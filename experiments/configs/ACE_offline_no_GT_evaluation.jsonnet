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

{
    "type": "ace",
    "config": {
        "run_type": "ace-evaluation",
        "agent": {
            "type": "ace_evaluation_react",
            "generator_model_config": generator_model_config,
            "appworld_config": {
                "random_seed": 123,
            },
            "logger_config": {
                "color": true,
                "verbose": true,
            },
            "generator_prompt_file_path": experiment_prompts_path + "/appworld_react_generator_prompt.txt",
            "trained_playbook_file_path": experiment_playbooks_path + "/appworld_offline_trained_no_gt_playbook_deepseek_3_1.txt",  
            "ignore_multiple_calls": true,
            "max_steps": 40,
            "max_cost_overall": 1000,
            "max_cost_per_task": 10,
            "log_lm_calls": true,
        },
        "dataset": "test_normal",
    }
}