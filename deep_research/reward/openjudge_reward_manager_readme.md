# OpenJudge Reward Manager

## Overview

A modular reward computation system for VERL reinforcement learning training, integrating [OpenJudge](https://github.com/open-compass/OpenJudge) evaluation framework with clean separation between framework and business logic.

### Architecture

**Two-Layer Design:**
- **Framework Layer** (`openjudge_reward_manager.py`): Interfaces with VERL DataProto, handles token decoding and result reconstruction
- **Business Logic Layer** (`openjudge_reward_function_v0.py`): Implements grader configurations and scoring logic

### Key Features

- ✅ **Modular Architecture**: Framework layer + pluggable reward functions
- ✅ **Easy Extension**: Create custom reward functions by subclassing `BaseOpenJudgeRewardFunction`
- ✅ **VERL Integration**: Compatible with DataProto interface
- ✅ **Multi-dimensional Evaluation**: 6 professional graders (trajectory, action, observation, etc.)
- ✅ **High Concurrency**: Async processing with GradingRunner
- ✅ **Flexible Aggregation**: Customizable weights for combining grader scores

---

## Quick Start

### 1. Basic Usage

```python
from deep_research.reward.openjudge_reward_manager import OpenJudgeRewardManager
from deep_research.reward.openjudge_reward_function_v0 import OpenJudgeRewardFunctionV0
from openjudge.models.schema.prompt_template import LanguageEnum

# Create reward function (business logic)
reward_fn = OpenJudgeRewardFunctionV0(
    model_name="qwen3-max",
    temperature=0.0,
    language=LanguageEnum.ZH,
    max_concurrency=32,
    grader_weights={
        "financial_report_resolution": 0.3,
        "trajectory_comprehensive": 0.7,
    },
    show_progress=True,
)

# Create reward manager (framework interface)
reward_manager = OpenJudgeRewardManager(
    tokenizer=tokenizer,
    num_examine=3,
    compute_score=reward_fn,  # Inject reward function
)

# Use in VERL training
reward_tensor = reward_manager(data)  # data is DataProto
```

### 2. Custom Reward Function

Extend `BaseOpenJudgeRewardFunction` to create custom grader combinations:

```python
from deep_research.reward.openjudge_reward_function_v0 import OpenJudgeRewardFunctionV0

class CustomRewardFunction(OpenJudgeRewardFunctionV0):
    """Custom reward function with different grader combination."""
    
    def create_grader_configs(self, model, language):
        """Override to customize graders."""
        return {
            "rubrics_based_trajectory_performance": GraderConfig(
                grader=RubricsBasedTrajectoryPerformance(model=model, language=language),
                mapper=lambda data: {
                    "messages": data["messages"],
                    "rubrics": data.get("rubrics", [])
                },
            ),
            "action_loop_detection": GraderConfig(
                grader=ActionLoopDetectionGrader(similarity_threshold=1.0),
                mapper=lambda data: {"messages": data["messages"]},
            ),
        }

# Use custom reward function
custom_fn = CustomRewardFunction(model_name="qwen3-max")
reward_manager = OpenJudgeRewardManager(
    tokenizer=tokenizer,
    num_examine=3,
    compute_score=custom_fn,
)
```

---

## Architecture

### Data Flow

```
VERL DataProto
  ↓ OpenJudgeRewardManager.__call__
  ↓ Decode tokens → strings
List[RewardSample]
  ↓ Group by prompt
Dict[prompt, List[RewardSample]]
  ↓ compute_score (BaseOpenJudgeRewardFunction)
  ↓ Convert to OpenJudge format
  ↓ GradingRunner.arun_multiple_datasets
  ↓ Parse and aggregate results
List[RewardResult] (sorted by original_index)
  ↓ Reconstruct and validate
  ↓ Fill reward_tensor
reward_tensor
```

### Components

**Framework Layer** (`openjudge_reward_manager.py`):
- `RewardSample`: Input data structure (original_index, group_index, prompt, response, messages, extra)
- `RewardResult`: Output data structure (original_index, group_index, score, reward_info, details)
- `BaseOpenJudgeRewardFunction`: Base class for reward functions
- `OpenJudgeRewardManager`: VERL integration interface

**Business Logic Layer** (`openjudge_reward_function_v0.py`):
- `OpenJudgeRewardFunctionV0`: V0 implementation with 6 graders

### V0 Graders

| Grader | Purpose | Inputs | Dimensions |
|--------|---------|--------|------------|
| `financial_report_resolution` | Financial report quality | messages + chat_date | Structure, coverage, timeliness |
| `financial_trajectory_faithfulness` | Factual accuracy | messages | Citation accuracy, consistency |
| `rubrics_based_trajectory_performance` | Rubrics-based evaluation | messages + rubrics | Custom dimensions |
| `trajectory_comprehensive` | Trajectory comprehensiveness | messages | Completeness, coherence |
| `observation_information_gain` | Information gain | messages | Novelty, redundancy |
| `action_loop_detection` | Action loop detection | messages | Repetition, similarity |

---

## API Reference

### OpenJudgeRewardManager

```python
OpenJudgeRewardManager(
    tokenizer,                  # Required: Tokenizer for decoding
    num_examine: int,           # Required: Number of samples to print for debugging
    compute_score,              # Required: BaseOpenJudgeRewardFunction instance
    reward_fn_key: str = "data_source",  # Optional: Data source key
    is_val_mode: bool = False,  # Optional: Validation mode
    **reward_kwargs             # Optional: Additional arguments
)
```

### OpenJudgeRewardFunctionV0

```python
OpenJudgeRewardFunctionV0(
    model_name: str = "qwen3-max",      # OpenAI model name
    temperature: float = 0.0,           # Model temperature
    language: LanguageEnum = LanguageEnum.ZH,  # Evaluation language
    max_concurrency: int = 32,          # Max concurrent requests
    grader_weights: Optional[Dict[str, float]] = None,  # Grader weights
    **kwargs                            # Additional args (e.g., show_progress)
)
```

### VERL DataProto Requirements

```python
data.batch = {
    "prompts": torch.Tensor,        # shape: (batch_size, prompt_len)
    "responses": torch.Tensor,      # shape: (batch_size, response_len)
    "attention_mask": torch.Tensor, # shape: (batch_size, prompt_len + response_len)
}

data.non_tensor_batch = {
    "messages": List[List[Dict]],   # Conversation history
    "extra_info": List[Dict],       # Extra info (chat_date, rubrics, etc.)
}
```

**messages format**:
```python
[
    {"role": "system", "content": "You are a financial assistant"},
    {"role": "user", "content": "Please analyze..."},
    {"role": "assistant", "content": "Based on analysis..."},
]
```

**extra_info format**:
```python
{
    "chat_date": "2025-01-01",  # Optional: for timeliness evaluation
    "rubrics": [                # Optional: for rubrics-based evaluation
        {"dimension": "Accuracy", "description": "..."},
        {"dimension": "Logic", "description": "..."},
    ],
}
```

---

## Usage Examples

### Basic Usage

```python
# Return reward tensor (default)
reward_tensor = reward_manager(data, return_dict=False)
# shape: (batch_size, response_len), reward at last token position

# Get detailed information
result = reward_manager(data, return_dict=True)
# Contains: reward_tensor, reward_extra_info (grader scores), all_details
```

### VERL Training Integration

```python
from deep_research.reward.openjudge_reward_manager import OpenJudgeRewardManager
from deep_research.reward.openjudge_reward_function_v0 import OpenJudgeRewardFunctionV0

# Initialize reward function
reward_fn = OpenJudgeRewardFunctionV0(
    model_name="qwen3-max",
    temperature=0.0,
    grader_weights={
        "financial_report_resolution": 0.3,
        "trajectory_comprehensive": 0.7,
    },
)

# Initialize reward manager
reward_manager = OpenJudgeRewardManager(
    tokenizer=tokenizer,
    num_examine=5,
    compute_score=reward_fn,
)

# Use in training loop
for batch in train_dataloader:
    # ... model generates response ...
    reward_tensor = reward_manager(batch)
    # ... PPO update ...
```

### Grader Weight Strategies

**Financial Analysis Task**:
```python
grader_weights = {
    "financial_report_resolution": 0.3,
    "financial_trajectory_faithfulness": 0.3,
    "rubrics_based_trajectory_performance": 0.2,
    "trajectory_comprehensive": 0.2,
}
```

**General Dialogue Task**:
```python
grader_weights = {
    "trajectory_comprehensive": 0.4,
    "observation_information_gain": 0.3,
    "action_loop_detection": 0.3,
}
```

---

## Key Implementation Details

### Score Aggregation

The `_aggregate_grader_scores` method supports two modes:
- **Weighted Average** (recommended): If `grader_weights` is specified
- **Equal Weights** (default): If no weights provided

Failed graders (returning `GraderError`) contribute 0.0 and are excluded from weight normalization.

### Result Ordering

GradingRunner preserves sample order within each dataset, but the output follows prompt group order (not original input order). The framework automatically sorts results by `original_index` to restore input order.

### Error Handling

- Individual grader failures don't affect other graders
- Weights are automatically redistributed among successful graders
- Detailed error information is captured in `details` field

---

## FAQ

**Q: How to add custom graders?**

Extend `BaseOpenJudgeRewardFunction` or `OpenJudgeRewardFunctionV0` and override `create_grader_configs()`.

**Q: How to debug index mismatches?**

Set `num_examine` parameter to print detailed information for the first N samples.

**Q: How to optimize concurrency?**

Adjust `max_concurrency` based on API rate limits (recommended: 16-32).

**Q: What if a grader fails?**

Failed graders contribute 0.0 score. Other graders continue normally and weights are renormalized.

---

## Summary

OpenJudge Reward Manager provides a powerful, flexible, and easy-to-integrate reward computation solution:

✅ **Modular Design**: Clean separation between framework and business logic  
✅ **Easy Extension**: Subclass `BaseOpenJudgeRewardFunction` for custom graders  
✅ **VERL Compatible**: Seamless DataProto integration  
✅ **Multi-dimensional**: 6 professional graders for comprehensive evaluation  
✅ **High Performance**: Async processing with configurable concurrency  
✅ **Robust**: Automatic error handling and result validation