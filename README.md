# ACE + AppWorld Experiments

This repository provides the full setup and instructions for running AppWorld experiments and reproducing the reported metrics, including offline and online adaptation with ACE.

> **⚠️ Important:**  
> Do **NOT** install this repository using `pip install appworld`.  
> This version includes custom modifications and must be installed **from source**.
> This repo is a research preview; please use it with caution in high-stakes production environments.

## 1. Environment Setup

Follow these steps exactly. Skipping steps may cause missing-file errors or silent failures. Setting up this repo and running basic experiments do not require GPU access. All you need is API access from providers like Together AI, SambaNova, or OpenAI.

### 1.1 Install Git LFS
```bash
git lfs install
```

### 1.2 Clone the repository
```bash
git clone https://github.com/Alex-q-z/ace-appworld.git ace-appworld
cd ace-appworld
export APPWORLD_PROJECT_PATH="$(pwd)"
```

### 1.3 Install AppWorld from source
```bash
pip install -e .
appworld install --repo
```

### 1.4 Install appworld_experiments from source
```bash
cd experiments
pip install -e .
cd ..
```

### 1.5 Fetch data
```bash
appworld download data
```

## 2. Configure Experiment

### 2.1 Configure API Keys

API providers are configured via the ```provider``` field in the experiment config files. The framework currently supports Together AI, SambaNova, and OpenAI. Before running experiments, make sure to export the corresponding API keys that you need:
```bash
export TOGETHER_API_KEY=YOUR_API_KEY_HERE # export if necessary
export SAMBANOVA_API_KEY=YOUR_API_KEY_HERE # export if necessary
export OPENAI_API_KEY=YOUR_API_KEY_HERE # export if necessary
```

### 2.2 (Optional) Customize Configuration Files

Under ```experiments/configs```, you can customize the experiment you'd like to run by adding new or editing existing ```.jsonnet``` config files, including choice of language models and API providers, sampling parameters, system prompts, etc.

As an example, the following config snippet specifies that the reflector agent should use DeepSeek-V3.1 as its language model, rely on the SambaNova API as the provider, and run with a sampling temperature of zero.
```
local reflector_model_config = {
    "name": "DeepSeek-V3.1",
    "provider": "sambanova",
    "temperature": 0,
    ...
};
```

You do not have to edit the configuration files if you just want to reproduce the results in our paper. 

### 2.3 (Optional) Customize Your Own ACE Agent

The definition of the ACE pipeline is under ```experiments/code/ace```, mostly in ```adaptation_react.py``` and ```evaluation_react.py```. We will follow up soon with more instructions on how to customize your ACE agent to support additional functionalities like context compression, retrieval, etc.

## 3. Run Experiments

Here is the basic format of running an experiment: ```appworld run CONFIG_FILE_NAME```.

### 3.1 Offline Context Adaptation with ACE

As an example, run the AppWorld + ACE (offline adaptation) experiment on the training split with:
```bash
appworld run ACE_offline_no_GT_adaptation
```

After we obtain the offline-optimized context, run evaluation on the test-normal split with:
```bash
appworld run ACE_offline_no_GT_evaluation
```

### 3.1 Online Context Adaptation with ACE

As an example, run the AppWorld + ACE (online adaptation) experiment on the test-normal split with:
```bash
appworld run ACE_online_no_GT
```

## 4. Evaluate Results

After the run above completes, run the follow command to obtain the aggregated metrics. Replace ```CONFIG_FILE_NAME``` with the config file associated with your experiment (e.g., ```ACE_offline_no_GT_evaluation``` or ```ACE_online_no_GT```). This should take no more than 2-3 minutes:
```bash
appworld evaluate CONFIG_FILE_NAME test_normal
appworld evaluate CONFIG_FILE_NAME test_challenge
```

Here is an example of a generated evaluation report (on the test-normal split):
| type         | task_goal_completion | scenario_goal_completion |
|--------------|----------------------|---------------------------|
| aggregate    | 64.9                 | 51.8                      |
| difficulty_1 | 86.0                 | 79.0                      |
| difficulty_2 | 77.1                 | 68.8                      |
| difficulty_3 | 36.5                 | 14.3                      |

We report aggregate TGC (```task_goal_completion```) and SGC (```scenario_goal_completion```) for evaluations in the paper.

## 5. Contact

If you have any questions, feel free to open a new issue or email at ```qizhengz@stanford.edu```. We’ll also be setting up a Slack/Discord channel soon to make communication easier.

## 6. Reference

If you find our work helpful, please use the following citation. Thank you for your support!
```
@article{zhang2025agentic,
  title={{Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models}},
  author={Zhang, Qizheng and Hu, Changran and Upasani, Shubhangi and Ma, Boyuan and Hong, Fenglu and Kamanuru, Vamsidhar and Rainton, Jay and Wu, Chen and Ji, Mengmeng and Li, Hanchen and others},
  journal={arXiv preprint arXiv:2510.04618},
  year={2025}
}
```