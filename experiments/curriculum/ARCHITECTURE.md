# Adaptive Training Architecture

## Overview

The adaptive training system combines three key components:
1. **Adaptive Question Selection**: Choose n tasks per iteration based on curriculum strategy
2. **Multiple Rollouts per Task**: Generate m rollouts for each task (for diversity)
3. **Shared Playbook State**: All n×m rollouts in an iteration share the same starting playbook
4. **Rollout Pruning**: Select top k rollouts (from n×m) to send to reflector, then curator for playbook updates

**Key Parameters**:
- **n** = `num_tasks_per_iteration`: Number of tasks selected per iteration (e.g., 5)
- **m** = `num_rollouts_per_task`: Number of rollouts generated per task (e.g., 1 or 3)
- **k** = `num_rollouts_for_reflection`: Number of rollouts kept after pruning (e.g., 3)

**Workflow**: Select n tasks → Generate n×m rollouts → Prune to k → Reflect on k → Curator batch update

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│ Iteration N                                                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. SELECTOR: Choose n tasks (e.g., n=5)                            │
│     - Analyzes current playbook                                     │
│     - Uses algorithm: difficulty_progressive/uncertainty_based/etc  │
│     - Selects: [task_A, task_B, task_C, task_D, task_E]           │
│                                                                     │
│  2. SAVE starting playbook state → playbook_v1                     │
│                                                                     │
│  3. GENERATE n×m rollouts (e.g., 5 tasks × 2 rollouts = 10)       │
│     All rollouts use playbook_v1, NO reflector yet                 │
│     For each task, generate m rollouts:                            │
│     ┌────────────────────────────────────────┐                     │
│     │ Task A, Rollout 1: Generator → Execute → SUCCESS            │
│     │ Task A, Rollout 2: Generator → Execute → FAILED             │
│     │ Task B, Rollout 1: Generator → Execute → FAILED             │
│     │ Task B, Rollout 2: Generator → Execute → FAILED             │
│     │ Task C, Rollout 1: Generator → Execute → SUCCESS            │
│     │ Task C, Rollout 2: Generator → Execute → SUCCESS            │
│     │ Task D, Rollout 1: Generator → Execute → FAILED             │
│     │ Task D, Rollout 2: Generator → Execute → SUCCESS            │
│     │ Task E, Rollout 1: Generator → Execute → FAILED             │
│     │ Task E, Rollout 2: Generator → Execute → FAILED             │
│     └────────────────────────────────────────┘                     │
│     Note: NO reflector or curator called yet                       │
│                                                                     │
│  4. PRUNE rollouts: n×m → k (e.g., 10 → 3)                        │
│     - Strategy: failure_first/high_cost/diverse/etc                │
│     - Selected: [B_rollout_1, B_rollout_2, E_rollout_2]           │
│     - Discarded: 7 other rollouts                                  │
│                                                                     │
│  5. GENERATE k reflections for pruned rollouts                     │
│     ┌────────────────────────────────────────┐                     │
│     │ Reflector(B_rollout_1) → reflection_1  │                     │
│     │ Reflector(B_rollout_2) → reflection_2  │                     │
│     │ Reflector(E_rollout_2) → reflection_3  │                     │
│     └────────────────────────────────────────┘                     │
│                                                                     │
│  6. CURATOR: Batch process k reflections → ONE update              │
│     - Input: playbook_v1 + k reflections                           │
│     - Curator(reflection_1) → update playbook                      │
│     - Curator(reflection_2) → update playbook                      │
│     - Curator(reflection_3) → update playbook                      │
│     - Output: playbook_v2                                          │
│                                                                     │
│  7. SAVE playbook_v2 snapshot                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Iteration N+1                                                       │
│  - Starts with playbook_v2                                         │
│  - Selects new n tasks...                                          │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Differences from Standard Adaptation

### Standard ACE Adaptation
```
Task 1 → Generator → Execute → Reflector → Curator → Update Playbook (v1 → v2)
Task 2 → Generator → Execute → Reflector → Curator → Update Playbook (v2 → v3)
Task 3 → Generator → Execute → Reflector → Curator → Update Playbook (v3 → v4)
...
```

**Issues:**
- Each task updates playbook immediately
- No curriculum learning
- No control over which tasks contribute to playbook
- Potentially noisy updates from every task

### Adaptive Training with Pruning
```
Iteration 1:
  Select n=3 tasks (curriculum-based)
  All tasks use playbook_v1

  Generate n×m rollouts (3 tasks × 2 rollouts/task = 6 total):
    Generator + execute only, NO reflector
    Task A, Rollout 1: Generator → Execute → SUCCESS
    Task A, Rollout 2: Generator → Execute → FAILED
    Task B, Rollout 1: Generator → Execute → FAILED
    Task B, Rollout 2: Generator → Execute → FAILED
    Task C, Rollout 1: Generator → Execute → SUCCESS
    Task C, Rollout 2: Generator → Execute → SUCCESS

  Prune to top k=3 rollouts (from 6 total)
    Select: B_rollout_2, B_rollout_1, A_rollout_2

  Generate reflections for k=3 pruned rollouts:
    Reflector(B_rollout_2) → reflection_1
    Reflector(B_rollout_1) → reflection_2
    Reflector(A_rollout_2) → reflection_3

  Curator batch processes 3 reflections → ONE update:
    Curator(reflection_1) → update playbook
    Curator(reflection_2) → update playbook
    Curator(reflection_3) → update playbook

  Result: playbook_v2

Iteration 2:
  Select n=3 different tasks
  All tasks use playbook_v2
  Generate n×m rollouts (generator + execute, NO reflector)
  Prune to top k rollouts
  Generate k reflections
  Curator batch processes k reflections → playbook_v3
...
```

**Benefits:**
- **Curriculum learning**: Smart task selection based on strategy
- **Multiple rollouts**: Generate m rollouts per task for diversity
- **Controlled updates**: Only k best rollouts (from n×m total) update playbook
- **Shared context**: All rollouts in iteration see same playbook state
- **Efficiency**: One batch playbook update per iteration (not per rollout)

## Components

### 1. AdaptiveQuestionSelector

Selects which n tasks to train on each iteration.

**Algorithms:**
- `random`: Random selection (baseline)
- `difficulty_progressive`: Start easy → medium → hard
- `uncertainty_based`: Select tasks where playbook is uncertain

**Configuration:**
```jsonnet
"selector": {
    "algorithm": "difficulty_progressive",
    "num_tasks_per_iteration": 5,  // n
    "dataset_path": "train.txt",
    "difficulty_schedule": "progressive",
}
```

### 2. RolloutPruner

Selects which k rollouts (from n×m total) to send to reflector.

**Strategies:**
- `random`: Random selection (baseline)
- `failure_first`: Prioritize failed rollouts
- `high_cost`: Prioritize expensive/complex rollouts
- `diverse`: Mix of failures and successes
- `most_informative`: Prioritize rollouts likely to generate useful reflections

**Configuration:**
```jsonnet
"pruner": {
    "strategy": "failure_first",
    "num_rollouts_for_reflection": 3,  // k
}
```

### 3. AdaptiveTrainingLoop

Orchestrates the entire workflow.

**Workflow:**
1. Selector chooses n tasks
2. Save current playbook state
3. Generate n×m rollouts (all use same playbook):
   - For each task, generate m rollouts: Generator → Execute (NO reflector yet)
   - Store rollout info
4. Prune n×m rollouts down to k rollouts
5. Generate k reflections:
   - For each pruned rollout: Reflector → reflection
6. Curator updates playbook using k reflections:
   - Batch process all k reflections → ONE playbook update
7. Repeat

## Example Scenario

```
Iteration 1:
├─ Selector: Choose n=3 tasks [A, B, C] (difficulty_progressive)
├─ Starting playbook: 10 bullets
├─ Generate n×m rollouts (3 tasks × 2 rollouts/task = 6 total):
│  All use playbook with 10 bullets, NO reflector yet
│  ├─ Task A, Rollout 1: Generator → Execute → SUCCESS (cost=2.5, test_failures=0)
│  ├─ Task A, Rollout 2: Generator → Execute → FAILED (cost=3.0, test_failures=1)
│  ├─ Task B, Rollout 1: Generator → Execute → FAILED (cost=3.1, test_failures=2)
│  ├─ Task B, Rollout 2: Generator → Execute → FAILED (cost=2.8, test_failures=3)
│  ├─ Task C, Rollout 1: Generator → Execute → SUCCESS (cost=1.8, test_failures=0)
│  └─ Task C, Rollout 2: Generator → Execute → SUCCESS (cost=2.0, test_failures=0)
├─ Pruner (failure_first, k=3): Select top 3 from 6 rollouts
│  ├─ Selected: [B_rollout_2, B_rollout_1, A_rollout_2]
│  ├─ Discarded: [A_rollout_1, C_rollout_1, C_rollout_2]
│  └─ Reason: Prioritize failures with most test failures
├─ Generate reflections for k=3 pruned rollouts:
│  ├─ Reflector(B_rollout_2) → "Need to handle pagination in list operations"
│  ├─ Reflector(B_rollout_1) → "API rate limiting not handled properly"
│  └─ Reflector(A_rollout_2) → "Missing input validation for edge cases"
├─ Curator: Batch process 3 reflections → ONE playbook update
│  ├─ Process reflection_1 → Add bullet about pagination
│  ├─ Process reflection_2 → Add bullet about rate limiting
│  └─ Process reflection_3 → Add bullet about input validation
└─ Updated playbook: 13 bullets (+3 new patterns from batch update)

Iteration 2:
├─ Selector: Choose 5 new tasks [F, G, H, I, J]
├─ Starting playbook: 13 bullets (from iteration 1)
├─ Generate rollouts: ...
└─ ...
```

## Parameters

| Parameter | Level | Description | Variable | Example |
|-----------|-------|-------------|----------|---------|
| `num_tasks_per_iteration` | Selector | How many tasks to select | n | 5 |
| `num_rollouts_per_task` | Loop | How many rollouts per task | m | 2 |
| `num_rollouts_for_reflection` | Pruner | How many rollouts to keep after pruning | k | 3 |
| `no_repeat_tasks` | Selector | If true, each task used only once across all iterations | - | true |
| `max_iterations` | Loop | How many iterations to run | - | 20 |

**Relationship:**
- **n** = number of tasks selected per iteration
- **m** = number of rollouts generated per task (for diversity)
- **Total rollouts per iteration** = n × m
- **k** = number of rollouts sent to reflector after pruning (k ≤ n×m)
- If k = n×m: no pruning (all rollouts processed)
- If k < n×m: prune n×m rollouts down to top k

**Example**: n=5 tasks, m=2 rollouts/task → 10 total rollouts → prune to k=3 → 3 reflections → 1 curator update

### Task Repetition Control

The `no_repeat_tasks` parameter controls whether tasks can be selected multiple times:

**`no_repeat_tasks=true` (default):**
- Each task is used at most once across all iterations
- Training automatically completes when all tasks tried
- Use case: Standard curriculum learning where you want to cover all tasks once

**`no_repeat_tasks=false`:**
- Tasks can be selected multiple times in different iterations
- Training never auto-completes (must set `max_iterations`)
- Use case: Fine-tuning where you want to repeatedly train on same tasks with improving playbook

**Example:**
```jsonnet
// Train on each task only once
"selector": {
    "num_tasks_per_iteration": 5,
    "no_repeat_tasks": true,  // Each task used once
}
// With 100 tasks, will run 20 iterations (100/5) then complete

// Allow task repetition for fine-tuning
"selector": {
    "num_tasks_per_iteration": 5,
    "no_repeat_tasks": false,  // Tasks can repeat
}
"max_iterations": 50,  // Must specify stopping condition
// Can select same tasks multiple times as playbook improves
```

## Pruning Strategies Comparison

| Strategy | When to Use | Pros | Cons |
|----------|-------------|------|------|
| `random` | Baseline comparison | Unbiased | No intelligence |
| `failure_first` | Learn from mistakes | Focuses on errors | May ignore successes |
| `high_cost` | Complex tasks matter | Targets hard problems | May ignore simple patterns |
| `diverse` | Balanced learning | Mix of failures/successes | Moderate on all |
| `most_informative` | Maximum learning signal | Prioritizes reflections | Requires reflections |

## Usage Examples

### Example 1: Basic Setup (n=5 tasks, m=1 rollout/task, k=3)

```jsonnet
{
    "selector": {
        "algorithm": "difficulty_progressive",
        "num_tasks_per_iteration": 5,  // n=5 tasks
        "dataset_path": "train.txt",
        "no_repeat_tasks": true,  // Each task used once
    },
    "num_rollouts_per_task": 1,  // m=1 (5 total rollouts)
    "pruner": {
        "strategy": "failure_first",
        "num_rollouts_for_reflection": 3,  // k=3 (prune 5→3)
    },
}
```

### Example 2: Multiple Rollouts Per Task (n=3, m=3, k=5)

```jsonnet
{
    "selector": {
        "algorithm": "difficulty_progressive",
        "num_tasks_per_iteration": 3,  // n=3 tasks
        "dataset_path": "train.txt",
    },
    "num_rollouts_per_task": 3,  // m=3 (9 total rollouts)
    "pruner": {
        "strategy": "diverse",
        "num_rollouts_for_reflection": 5,  // k=5 (prune 9→5)
    },
}
```

### Example 3: No Pruning (all rollouts to reflector)

```jsonnet
{
    "selector": {
        "algorithm": "random",
        "num_tasks_per_iteration": 3,
        "dataset_path": "train.txt",
    },
    "num_rollouts_per_task": 2,  // 6 total rollouts
    // No pruner configured - all 6 rollouts sent to reflector
}
```

### Example 4: Aggressive Pruning (n=10, m=2, k=3)

```jsonnet
{
    "selector": {
        "algorithm": "uncertainty_based",
        "num_tasks_per_iteration": 10,  // n=10 tasks
        "dataset_path": "train.txt",
        "no_repeat_tasks": true,
        "selection_strategy": "low_confidence",
    },
    "num_rollouts_per_task": 2,  // m=2 (20 total rollouts)
    "pruner": {
        "strategy": "most_informative",
        "num_rollouts_for_reflection": 3,  // k=3 (prune 20→3)
    },
}
```

### Example 5: Fine-tuning with Task Repetition (n=5, m=2, k=4)

```jsonnet
{
    "selector": {
        "algorithm": "difficulty_progressive",
        "num_tasks_per_iteration": 5,
        "dataset_path": "train.txt",
        "no_repeat_tasks": false,  // Allow task repetition
    },
    "num_rollouts_per_task": 2,  // m=2 (10 total rollouts)
    "pruner": {
        "strategy": "failure_first",
        "num_rollouts_for_reflection": 4,  // k=4 (prune 10→4)
    },
    "max_iterations": 100,  // Must set max_iterations when repeats allowed
}
```

## Command Line

```bash
# Run with default config
python run_adaptive_training.py \
    --config ACE_adaptive_difficulty_progressive.jsonnet

# Override parameters
python run_adaptive_training.py \
    --config ACE_adaptive_random.jsonnet \
    --num-tasks-per-iteration 10 \
    --max-iterations 5

# Check configuration before running
python run_adaptive_training.py \
    --config ACE_adaptive_uncertainty.jsonnet
# (Shows config summary before starting)
```

## Expected Output

```
ITERATION 1: Selected 5 tasks
Selection metadata: {'algorithm': 'difficulty_progressive', ...}
Tasks: ['task_001', 'task_023', 'task_045', 'task_067', 'task_089']

Generating rollouts for 5 tasks...
--- Generating rollout 1/5: task_001 ---
Rollout complete: success=True, cost=2.34, steps=8, test_failures=0
...

Generated 5 rollouts
  - Successful: 2
  - Failed: 3

Pruned to 3 rollouts for curator
  - Successful: 0
  - Failed: 3

Processing 3 rollouts through curator...
  Processing rollout for task_023 (success=False, test_failures=2)
  Processing rollout for task_067 (success=False, test_failures=1)
  Processing rollout for task_089 (success=False, test_failures=3)
Playbook updated based on pruned rollouts

Progress: 5/100 tasks (5.0%) - Iteration 1 complete
```

## Future Enhancements

1. **Parallel Rollout Generation**: Generate n×m rollouts in parallel (currently sequential)
2. **Reflection Aggregation**: Combine similar reflections before curator
3. **Dynamic k**: Adjust `num_rollouts_for_reflection` based on iteration progress
4. **Dynamic m**: Adjust `num_rollouts_per_task` based on task difficulty
5. **Confidence-Based Pruning**: Use playbook confidence scores for pruning
6. **Rollout Caching**: Reuse rollouts across iterations if playbook unchanged

## Summary

The adaptive training architecture provides:
- **Curriculum Learning**: Smart task selection via AdaptiveQuestionSelector
- **Multiple Rollouts**: Generate m rollouts per task for diversity
- **Shared Context**: All n×m rollouts in iteration use same playbook state
- **Selective Learning**: Pruning ensures only k best rollouts (from n×m) generate reflections
- **Efficiency**: Batch updates reduce reflection and curator overhead

This enables more efficient and controlled playbook learning compared to standard sequential adaptation.
