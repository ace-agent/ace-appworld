local project_home_path = std.extVar("APPWORLD_PROJECT_PATH");
local experiment_prompts_path = project_home_path + "/experiments/prompts";
local experiment_playbooks_path = project_home_path + "/experiments/playbooks";
local experiment_configs_path = project_home_path + "/experiments/configs";
local experiment_code_path = project_home_path + "/experiments/code";

local generator_model_config = {
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
    "name": "/import/ml-sc-nlpcheckpoints-scratch3/jonathanl/generic_checkpoints/Qwen2.5-7B-Instruct",
    "temperature": 0,
    "lora_r":  16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],

    "sft_max_seq_len": 2048,
    "sft_microbatch_size": 1,
    "sft_grad_accum_steps": 8,
    "sft_lr": 2e-4, 
    "sft_epochs": 1,

    # Misc
    "bf16": true,
    "seed": 42
};

local curator_model_config = {
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
        "run_type": "ace-adaptation",
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
            "trained_playbook_file_path": experiment_playbooks_path + "/appworld_offline_trained_with_gt_playbook.txt",  
            "trained_checkpoints" : experiment_playbooks_path + "/appworld_offline_trained_with_gt_lora_checkpoints",
            "ignore_multiple_calls": true,
            "max_steps": 40,
            "max_cost_overall": 1000,
            "max_cost_per_task": 10,
            "log_lm_calls": true,
            "use_gt_code": true
        },
        "dataset": "train",
    }
}
