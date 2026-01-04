"""
OpenJudge Reward Function V0 - Business Logic Layer

This module implements the V0 version of reward computation using
OpenJudge GradingRunner with financial domain and agent graders.
"""

from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from openjudge.graders.agent.action.action_loop import ActionLoopDetectionGrader
from openjudge.graders.agent.observation.observation_information_gain import (
    ObservationInformationGainGrader,
)
from openjudge.graders.agent.trajectory.trajectory_comprehensive import (
    TrajectoryComprehensiveGrader,
)
from openjudge.graders.schema import GraderError
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum
from openjudge.runner.grading_runner import GraderConfig, GradingRunner
from openjudge.scenarios.deep_research.graders.financial_report_resolution import (
    FinancialReportResolutionGrader,
)
from openjudge.scenarios.deep_research.graders.financial_trajectory_faithfulness import (
    FinancialTrajectoryFaithfulGrader,
)
from openjudge.scenarios.deep_research.graders.rubrics_based_trajectory_performance import (
    RubricsBasedTrajectoryPerformance,
)

from deep_research.reward.openjudge_reward_manager import (
    BaseOpenJudgeRewardFunction,
    RewardResult,
    RewardSample,
)


class OpenJudgeRewardFunctionV0(BaseOpenJudgeRewardFunction):
    """
    V0 implementation using OpenJudge GradingRunner.
    
    This version includes:
    - Financial domain graders (report resolution, trajectory faithfulness)
    - Agent trajectory graders (comprehensive, information gain, loop detection)
    - Rubrics-based evaluation
    
    Users can extend this class and override create_grader_configs() to
    customize the grader combination.
    
    Example:
        reward_fn = OpenJudgeRewardFunctionV0(
            model_name="qwen3-max",
            temperature=0.0,
            grader_weights={
                "financial_report_resolution": 0.3,
                "trajectory_comprehensive": 0.7,
            }
        )
    """
    
    def __init__(
        self,
        model_name: str = "qwen3-max",
        temperature: float = 0.0,
        language: LanguageEnum = LanguageEnum.ZH,
        max_concurrency: int = 32,
        grader_weights: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """
        Initialize V0 reward function.
        
        Args:
            model_name: OpenAI model name
            temperature: Model temperature for evaluation
            language: Evaluation language (ZH or EN)
            max_concurrency: Max concurrent OpenAI requests
            grader_weights: Weights for aggregating grader scores.
                If None, uses equal weights. Will be normalized automatically.
            **kwargs: Additional arguments (e.g., show_progress)
        """
        self.model = OpenAIChatModel(
            model=model_name,
            temperature=temperature,
        )
        
        # Create grader configs
        grader_configs = self.create_grader_configs(self.model, language)
        
        # Initialize GradingRunner
        self.runner = GradingRunner(
            grader_configs=grader_configs,
            max_concurrency=max_concurrency,
            show_progress=kwargs.get("show_progress", True),
        )
        
        self.grader_weights = grader_weights
        self.grader_names = list(grader_configs.keys())
        
        logger.info(
            f"Initialized OpenJudgeRewardFunctionV0 with "
            f"{len(self.grader_names)} graders: {self.grader_names}"
        )
        logger.info(f"Max concurrency: {max_concurrency}")
    
    def create_grader_configs(
        self,
        model: OpenAIChatModel,
        language: LanguageEnum
    ) -> Dict[str, GraderConfig]:
        """
        Create grader configurations.
        
        Users can override this method in subclasses to customize
        the grader combination.
        
        Args:
            model: LLM model for evaluation
            language: Language for evaluation prompts
            
        Returns:
            Dictionary of grader configurations
        """
        return {
            # Financial report quality evaluation
            "financial_report_resolution": GraderConfig(
                grader=FinancialReportResolutionGrader(model=model, language=language),
                mapper=lambda data: {
                    "messages": data["messages"], 
                    "chat_date": data.get("chat_date")
                },
            ),
            # Factual accuracy evaluation
            "financial_trajectory_faithfulness": GraderConfig(
                grader=FinancialTrajectoryFaithfulGrader(model=model, language=language),
                mapper=lambda data: {"messages": data["messages"]},
            ),
            # Rubrics-based evaluation
            # "rubrics_based_trajectory_performance": GraderConfig(
            #     grader=RubricsBasedTrajectoryPerformance(model=model, language=language),
            #     mapper=lambda data: {
            #         "messages": data["messages"], 
            #         "rubrics": data.get("rubrics", [])
            #     },
            # ),
            # Comprehensive trajectory evaluation
            "trajectory_comprehensive": GraderConfig(
                grader=TrajectoryComprehensiveGrader(model=model, language=language),
                mapper=lambda data: {"messages": data["messages"]},
            ),
            # Information gain evaluation
            "observation_information_gain": GraderConfig(
                grader=ObservationInformationGainGrader(similarity_threshold=0.5),
                mapper=lambda data: {"messages": data["messages"]},
            ),
            # Action loop detection
            "action_loop_detection": GraderConfig(
                grader=ActionLoopDetectionGrader(similarity_threshold=1.0),
                mapper=lambda data: {"messages": data["messages"]},
            ),
        }
    
    async def compute_batch_scores(
        self,
        prompt_to_samples: Dict[str, List[RewardSample]]
    ) -> List[RewardResult]:
        """
        Compute scores using OpenJudge graders.
        
        This method:
        1. Converts samples to OpenJudge format
        2. Calls GradingRunner.arun_multiple_datasets
        3. Parses and aggregates results
        4. Returns sorted results
        
        Args:
            prompt_to_samples: Dict mapping prompts to samples
            
        Returns:
            List of RewardResult sorted by original_index
        """
        # Convert to OpenJudge format
        datasets = []
        for prompt, group_samples in prompt_to_samples.items():
            dataset = self._convert_to_openjudge_format(group_samples)
            datasets.append(dataset)
        
        logger.debug(
            f"Calling GradingRunner.arun_multiple_datasets with "
            f"{len(datasets)} groups"
        )
        
        # Call GradingRunner
        try:
            runner_results = await self.runner.arun_multiple_datasets(datasets)
        except Exception as e:
            logger.error(f"GradingRunner.arun_multiple_datasets failed: {e}")
            logger.exception("Detailed error:")
            
            # Return default results on error
            all_samples = []
            for samples in prompt_to_samples.values():
                all_samples.extend(samples)
            
            return [
                RewardResult(
                    original_index=sample.original_index,
                    group_index=sample.group_index,
                    score=0.0,
                    reward_info={f"{name}_score": 0.0 for name in self.grader_names},
                    details={"error": str(e)}
                )
                for sample in all_samples
            ]
        
        # Parse results
        result_items = self._parse_runner_results(runner_results, prompt_to_samples)
        
        return result_items
    
    def _convert_to_openjudge_format(
        self, 
        group_samples: List[RewardSample]
    ) -> List[Dict[str, Any]]:
        """
        Convert RewardSample to OpenJudge format.
        
        OpenJudge expects:
        {
            "messages": [...],     # Required: conversation history
            "chat_date": "...",    # Optional: for timeliness evaluation
            "rubrics": [...],      # Optional: for rubrics-based evaluation
        }
        
        Args:
            group_samples: List of samples in the same group
            
        Returns:
            List of dicts in OpenJudge format
        """
        dataset = []
        for sample in group_samples:
            data = {
                "messages": sample.messages,
                "chat_date": sample.extra.get("chat_date", None) if sample.extra else None,
                "rubrics": sample.extra.get("rubrics", []) if sample.extra else [],
            }
            dataset.append(data)
        return dataset
    
    def _parse_runner_results(
        self,
        runner_results: List[Dict[str, List[Any]]],
        prompt_to_samples: Dict[str, List[RewardSample]]
    ) -> List[RewardResult]:
        """
        Parse GradingRunner results and map to RewardResult.
        
        The runner returns results grouped by dataset (prompt group).
        This method:
        1. Validates dataset count consistency
        2. Iterates over each dataset and sample
        3. Aggregates scores from all graders
        4. Creates RewardResult objects
        5. Sorts by original_index to restore input order
        
        Args:
            runner_results: Results from arun_multiple_datasets
                Format: [
                    {  # dataset_0
                        "grader1": [result1_sample1, result1_sample2, ...],
                        "grader2": [result2_sample1, result2_sample2, ...],
                    },
                    {  # dataset_1
                        "grader1": [...],
                        "grader2": [...],
                    },
                ]
            prompt_to_samples: Dict mapping prompts to samples
            
        Returns:
            List of RewardResult sorted by original_index
        """
        all_results = []
        prompt_list = list(prompt_to_samples.keys())
        
        # Validate dataset count
        if len(runner_results) != len(prompt_list):
            raise ValueError(
                f"Result count ({len(runner_results)}) does not match "
                f"prompt group count ({len(prompt_list)})"
            )
        
        # Iterate over each dataset
        for dataset_idx, dataset_results in enumerate(runner_results):
            prompt = prompt_list[dataset_idx]
            group_samples = prompt_to_samples[prompt]
            num_samples = len(group_samples)
            
            # Pre-extract and validate all grader results
            grader_results_dict = {}
            for grader_name in self.grader_names:
                if grader_name in dataset_results:
                    results_list = dataset_results[grader_name]
                    
                    # Validate result count
                    if len(results_list) != num_samples:
                        logger.warning(
                            f"Dataset {dataset_idx}, Grader '{grader_name}' "
                            f"result count ({len(results_list)}) != "
                            f"sample count ({num_samples})"
                        )
                        # Pad or truncate to correct length
                        if len(results_list) < num_samples:
                            results_list = list(results_list) + [
                                GraderError(error=f"{grader_name} insufficient results")
                                for _ in range(num_samples - len(results_list))
                            ]
                        else:
                            results_list = results_list[:num_samples]
                    
                    grader_results_dict[grader_name] = results_list
                else:
                    # Grader did not return results
                    logger.warning(
                        f"Dataset {dataset_idx}, Grader '{grader_name}' "
                        f"did not return results"
                    )
                    grader_results_dict[grader_name] = [
                        GraderError(error=f"{grader_name} no results")
                        for _ in range(num_samples)
                    ]
            
            # Iterate over samples in this group
            for sample_idx in range(num_samples):
                # Collect all grader results for this sample
                sample_grader_results = {
                    name: results[sample_idx]
                    for name, results in grader_results_dict.items()
                }
                
                # Aggregate scores
                total_score, reward_info = self._aggregate_grader_scores(
                    sample_grader_results
                )
                
                # Extract details
                details = {
                    grader_name: self._extract_result_details(result)
                    for grader_name, result in sample_grader_results.items()
                }
                
                # Create RewardResult
                sample = group_samples[sample_idx]
                result_item = RewardResult(
                    original_index=sample.original_index,
                    group_index=sample.group_index,
                    score=total_score,
                    reward_info=reward_info,
                    details=details
                )
                all_results.append(result_item)
        
        # Sort by original_index to restore input order
        all_results.sort(key=lambda x: x.original_index)
        
        return all_results
    
    def _aggregate_grader_scores(
        self, 
        grader_results: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Aggregate scores from multiple graders.
        
        This method:
        1. Collects valid scores from all graders
        2. Calculates normalized weights (specified or equal)
        3. Computes weighted average score
        
        Failed graders (GraderError) contribute 0.0 and are excluded
        from weight calculation.
        
        Args:
            grader_results: {grader_name: GraderResult or GraderError}
        
        Returns:
            (total_score, reward_info_dict)
        """
        reward_info = {}
        
        # Collect valid scores
        valid_scores = {}
        for grader_name in self.grader_names:
            result = grader_results.get(grader_name)
            
            if result is None or isinstance(result, GraderError):
                # Error case: this grader contributes 0.0
                reward_info[f"{grader_name}_score"] = 0.0
                if result and isinstance(result, GraderError):
                    reward_info[f"{grader_name}_error"] = result.error
            else:
                score = result.score if hasattr(result, "score") else 0.0
                reward_info[f"{grader_name}_score"] = score
                valid_scores[grader_name] = score
        
        # If no valid scores, return 0
        if not valid_scores:
            return 0.0, reward_info
        
        # Calculate normalized weights
        if self.grader_weights:
            # Use specified weights, normalize only for valid graders
            weights = {
                name: self.grader_weights.get(name, 0.0) 
                for name in valid_scores.keys()
            }
            total_weight = sum(weights.values())
            if total_weight > 0:
                normalized_weights = {
                    name: w / total_weight 
                    for name, w in weights.items()
                }
            else:
                # All weights are 0, use equal weights
                normalized_weights = {
                    name: 1.0 / len(valid_scores) 
                    for name in valid_scores.keys()
                }
        else:
            # Default: equal weights
            normalized_weights = {
                name: 1.0 / len(valid_scores) 
                for name in valid_scores.keys()
            }
        
        # Calculate weighted total score
        total_score = sum(
            valid_scores[name] * normalized_weights[name] 
            for name in valid_scores.keys()
        )
        
        return total_score, reward_info

    def _extract_result_details(self, result: Any) -> Dict[str, Any]:
        """
        Extract details from a single grader result.
        
        Args:
            result: GraderScore or GraderError object
        
        Returns:
            Dict containing score, reason, metadata or error
        """
        if isinstance(result, GraderError):
            return {
                "error": result.error,
                "reason": getattr(result, "reason", None),
            }
        else:
            return {
                "score": getattr(result, "score", None),
                "reason": getattr(result, "reason", None),
                "metadata": getattr(result, "metadata", None),
            }


# =============================================================================
# Default Instance for VERL Integration
# =============================================================================

# Create default instance with standard parameters for training
compute_score_async = OpenJudgeRewardFunctionV0(
    model_name="qwen3-max",
    temperature=0.0,
    language=LanguageEnum.ZH,
    max_concurrency=32,
    grader_weights=None,  # Use equal weights by default
    show_progress=True,
)

