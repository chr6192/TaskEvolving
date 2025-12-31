"""
Example usage of OpenJudge Reward Manager with modular architecture.

This example demonstrates:
1. Using the default V0 reward function
2. Creating a custom reward function
3. Integration with VERL training
"""

from open_judge.models.schema.prompt_template import LanguageEnum
from open_judge.runner.grading_runner import GraderConfig
from open_judge.graders.agent.action.action_loop import ActionLoopDetectionGrader
from open_judge.graders.agent.observation.observation_information_gain import (
    ObservationInformationGainGrader,
)
from open_judge.graders.agent.trajectory.trajectory_comprehensive import (
    TrajectoryComprehensiveGrader,
)

from openjudge_reward_manager import OpenJudgeRewardManager
from openjudge_reward_function_v0 import OpenJudgeRewardFunctionV0


# Example 1: Using default V0 reward function
def example_basic_usage(tokenizer):
    """Basic usage with default V0 graders."""
    
    # Create V0 reward function
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
    
    # Create reward manager
    reward_manager = OpenJudgeRewardManager(
        tokenizer=tokenizer,
        num_examine=3,
        compute_score=reward_fn,
    )
    
    return reward_manager


# Example 2: Custom reward function with different graders
class CustomRewardFunction(OpenJudgeRewardFunctionV0):
    """
    Custom reward function for dialogue tasks.
    
    This version focuses on action quality and information gain,
    removing financial-specific graders.
    """
    
    def create_grader_configs(self, model, language):
        """Override to use custom grader combination."""
        return {
            "trajectory_comprehensive": GraderConfig(
                grader=TrajectoryComprehensiveGrader(model=model, language=language),
                mapper=lambda data: {"messages": data["messages"]},
            ),
            "observation_information_gain": GraderConfig(
                grader=ObservationInformationGainGrader(similarity_threshold=0.5),
                mapper=lambda data: {"messages": data["messages"]},
            ),
            "action_loop_detection": GraderConfig(
                grader=ActionLoopDetectionGrader(similarity_threshold=0.8),
                mapper=lambda data: {"messages": data["messages"]},
            ),
        }


def example_custom_function(tokenizer):
    """Using custom reward function."""
    
    # Create custom reward function
    custom_fn = CustomRewardFunction(
        model_name="qwen3-max",
        temperature=0.0,
        grader_weights={
            "trajectory_comprehensive": 0.4,
            "observation_information_gain": 0.3,
            "action_loop_detection": 0.3,
        },
    )
    
    # Create reward manager with custom function
    reward_manager = OpenJudgeRewardManager(
        tokenizer=tokenizer,
        num_examine=5,
        compute_score=custom_fn,
    )
    
    return reward_manager


# Example 3: Integration with VERL training
def example_verl_integration(tokenizer):
    """Complete VERL training integration example."""
    
    # Initialize reward function
    reward_fn = OpenJudgeRewardFunctionV0(
        model_name="qwen3-max",
        temperature=0.0,
        grader_weights={
            "financial_report_resolution": 0.25,
            "financial_trajectory_faithfulness": 0.25,
            "rubrics_based_trajectory_performance": 0.25,
            "trajectory_comprehensive": 0.25,
        },
    )
    
    # Initialize reward manager
    reward_manager = OpenJudgeRewardManager(
        tokenizer=tokenizer,
        num_examine=5,
        compute_score=reward_fn,
        is_val_mode=False,
    )
    
    # In training loop:
    # for batch in train_dataloader:
    #     # Model generates responses
    #     # ...
    #     
    #     # Compute rewards
    #     result = reward_manager(batch, return_dict=True)
    #     reward_tensor = result["reward_tensor"]
    #     reward_info = result["reward_extra_info"]
    #     
    #     # Use for PPO update
    #     # ...
    
    return reward_manager


if __name__ == "__main__":
    # This is just for demonstration
    # In actual usage, tokenizer would come from VERL setup
    print("OpenJudge Reward Manager Examples")
    print("==================================")
    print("\n1. Basic usage: See example_basic_usage()")
    print("2. Custom function: See example_custom_function()")
    print("3. VERL integration: See example_verl_integration()")
    print("\nFor actual usage, initialize with your tokenizer from VERL.")

