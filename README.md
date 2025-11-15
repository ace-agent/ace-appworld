# ACE + AppWorld Experiments

This repository provides the full setup and instructions for running AppWorld experiments and reproducing the reported metrics, including offline and online adaptation with ACE.

> **⚠️ Important:**  
> Do **NOT** install this repository using `pip install appworld`.  
> This version includes custom modifications and must be installed **from source**.

## 1. Environment Setup

Follow these steps exactly. Skipping steps may cause missing-file errors or silent failures.

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

## 2. Configure API Keys

API providers are configured via the ```provider``` field in the experiment config files. The framework currently supports Together AI, SambaNova, and OpenAI. Before running experiments, make sure to export the corresponding API keys that you need:
```bash
export TOGETHER_API_KEY=YOUR_API_KEY_HERE # export if necessary
export SAMBANOVA_API_KEY=YOUR_API_KEY_HERE # export if necessary
export OPENAI_API_KEY=YOUR_API_KEY_HERE # export if necessary
```

## 3. Run Experiments

Example: run the AppWorld + ACE (online adaptation) experiment on the test-normal split
```bash
appworld run ACE_online_no_GT
```

## 4. Evaluate Results

After the run completes, compute aggregated metrics:
```bash
appworld evaluate ACE_online_no_GT test_normal
```

Here is an example of a generated evaluation report:
| type         | task_goal_completion | scenario_goal_completion |
|--------------|----------------------|---------------------------|
| aggregate    | 64.9                 | 51.8                      |
| difficulty_1 | 86.0                 | 79.0                      |
| difficulty_2 | 77.1                 | 68.8                      |
| difficulty_3 | 36.5                 | 14.3                      |