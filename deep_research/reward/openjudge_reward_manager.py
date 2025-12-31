"""
OpenJudge Reward Manager - Framework Integration Layer

This module provides the interface between VERL training framework and
OpenJudge reward computation logic.
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from loguru import logger
from verl import DataProto


@dataclass
class RewardSample:
    """A sample to be scored."""
    original_index: int  # Global index in batch
    group_index: int     # Index within group (same prompt)
    prompt: str
    response: str
    messages: List[Dict]
    extra: Any


@dataclass  
class RewardResult:
    """Scoring result for a sample."""
    original_index: int
    group_index: int
    score: float
    reward_info: Dict[str, float]
    details: Optional[Dict] = None


class BaseOpenJudgeRewardFunction:
    """
    Base class for OpenJudge reward computation.
    
    Subclasses should implement compute_batch_scores() to define
    custom grader configurations and scoring logic.
    
    Example:
        class MyRewardFunction(BaseOpenJudgeRewardFunction):
            async def compute_batch_scores(self, prompt_to_samples):
                # Custom implementation
                return results
    """
    
    async def __call__(
        self,
        prompt_to_samples: Dict[str, List[RewardSample]]
    ) -> List[RewardResult]:
        """
        Callable interface for reward computation.
        
        This allows instances to be passed as compute_score parameter
        to OpenJudgeRewardManager.
        
        Args:
            prompt_to_samples: Dict mapping prompts to their samples
            
        Returns:
            List of RewardResult sorted by original_index
        """
        return await self.compute_batch_scores(prompt_to_samples)
    
    async def compute_batch_scores(
        self,
        prompt_to_samples: Dict[str, List[RewardSample]]
    ) -> List[RewardResult]:
        """
        Compute scores for grouped samples.
        
        Must be implemented by subclasses.
        
        Args:
            prompt_to_samples: Dict mapping prompts to their samples
            
        Returns:
            List of RewardResult sorted by original_index
        """
        raise NotImplementedError("Subclasses must implement compute_batch_scores")


class OpenJudgeRewardManager:
    """
    Reward manager that integrates with VERL training framework.
    
    This manager handles DataProto batches from VERL and delegates scoring
    to a pluggable reward function (compute_score).
    
    The manager is responsible for:
    - Decoding tokens to strings
    - Grouping samples by prompt
    - Calling the reward function
    - Reconstructing results to match input order
    - Filling reward tensor
    
    Example:
        reward_fn = OpenJudgeRewardFunctionV0(model_name="qwen3-max")
        manager = OpenJudgeRewardManager(
            tokenizer=tokenizer,
            num_examine=5,
            compute_score=reward_fn,
        )
        rewards = manager(data_proto)
    """
    
    def __init__(
        self, 
        tokenizer, 
        num_examine: int,
        compute_score,
        reward_fn_key: str = "data_source",
        is_val_mode: bool = False,
        **reward_kwargs
    ):
        """
        Initialize OpenJudgeRewardManager.
        
        Args:
            tokenizer: Tokenizer for decoding prompts/responses
            num_examine: Number of samples to print for debugging
            compute_score: Reward function instance (BaseOpenJudgeRewardFunction)
            reward_fn_key: Key for data source (for compatibility)
            is_val_mode: Whether in validation mode
            **reward_kwargs: Additional arguments
            
        Raises:
            ValueError: If compute_score is not provided
        """
        if compute_score is None:
            raise ValueError(
                "compute_score must be provided. "
                "Please pass an instance of BaseOpenJudgeRewardFunction."
            )
        
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.is_val_mode = is_val_mode
        self.reward_kwargs = reward_kwargs
        
        logger.info(
            f"Initialized OpenJudgeRewardManager with "
            f"compute_score={type(compute_score).__name__}"
        )
    
    def __call__(self, data: DataProto, return_dict=False):
        """
        Compute rewards for DataProto batch.
        
        Args:
            data: DataProto containing prompts and responses
            return_dict: Whether to return dict with extra info
            
        Returns:
            reward_tensor or dict with reward_tensor and reward_extra_info
        """
        # Return cached scores if available
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        # Decode prompts and responses
        responses_str = []
        prompts_str = []
        messages_list = []
        extras_info = []

        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            response_str = self.tokenizer.decode(
                data.batch["responses"][i][:length], 
                skip_special_tokens=True
            )
            prompt_str = self.tokenizer.decode(
                data.batch["prompts"][i], 
                skip_special_tokens=True
            )
            extra_info = data.non_tensor_batch['extra_info'][i]

            messages_list.append(data.non_tensor_batch["messages"][i])
            responses_str.append(response_str)
            prompts_str.append(prompt_str)
            extras_info.append(extra_info)

        # Create RewardSample objects
        samples = []
        for i in range(len(data)):
            sample = RewardSample(
                original_index=i,
                group_index=-1,  # Will be set during grouping
                prompt=prompts_str[i],
                response=responses_str[i],
                messages=messages_list[i],
                extra=extras_info[i]
            )
            samples.append(sample)

        # Group by prompt and set group_index
        prompt_to_samples = defaultdict(list)
        for sample in samples:
            group_samples = prompt_to_samples[sample.prompt]
            sample.group_index = len(group_samples)
            group_samples.append(sample)

        # Validate group sizes
        group_sizes = [len(items) for items in prompt_to_samples.values()]
        if len(set(group_sizes)) > 1:
            raise AssertionError(
                f"All groups must have the same size, got: {group_sizes}"
            )

        logger.info(
            f"Processing {len(prompt_to_samples)} groups, "
            f"{group_sizes[0]} samples per group"
        )

        # Call reward function
        result_items = asyncio.run(self.compute_score(prompt_to_samples))

        # Reconstruct results
        all_rewards, all_reward_infos, all_details = self._reconstruct_results(
            result_items, len(data)
        )
        
        if not all_rewards:
            raise ValueError("Reward computation returned empty results")
        
        if not all_reward_infos:
            raise ValueError("Reward computation returned empty reward_info")

        # Fill reward tensor
        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            reward = all_rewards[i]
            reward_tensor[i, length - 1] = reward
            
            # Print for debugging
            if i < self.num_examine:
                logger.debug(f"[prompt] {prompts_str[i]}")
                logger.debug(f"[response] {responses_str[i]}")
                logger.debug(f"[score] {reward}")
                reward_info_dict = {k: v[i] for k, v in all_reward_infos.items()}
                logger.debug(f"[reward_info] {reward_info_dict}")
        
        # Add accuracy to batch
        data.batch["acc"] = torch.tensor(
            all_rewards, 
            dtype=torch.float32, 
            device=prompt_ids.device
        )
        
        if return_dict:
            result = {
                "reward_tensor": reward_tensor, 
                "reward_extra_info": dict(all_reward_infos)
            }
            if all_details:
                result["all_details"] = all_details
            return result
        else:
            return reward_tensor

    def _reconstruct_results(
        self, 
        result_items: List[RewardResult], 
        data_length: int
    ):
        """
        Convert RewardResult list to final output format with validation.
        
        This method:
        1. Validates result integrity and index consistency
        2. Extracts scores, reward_info, and details
        3. Fills missing reward_info fields across all samples
        
        Args:
            result_items: List of RewardResult (should be sorted)
            data_length: Expected data length
        
        Returns:
            (all_rewards, all_reward_infos, all_details)
        """
        # Collect all possible reward_info keys
        all_info_keys = set()
        for item in result_items:
            all_info_keys.update(item.reward_info.keys())
        
        # Validate integrity
        self._verify_results_integrity(result_items, data_length, all_info_keys)

        # Fill missing keys with default value
        for item in result_items:
            for key in all_info_keys:
                if key not in item.reward_info:
                    item.reward_info[key] = 0.0

        # Extract information
        all_rewards = [item.score for item in result_items]
        all_reward_infos = defaultdict(list)
        for key in all_info_keys:
            all_reward_infos[key] = [item.reward_info[key] for item in result_items]
        all_details = [item.details for item in result_items]

        return all_rewards, dict(all_reward_infos), all_details

    def _verify_results_integrity(
        self, 
        result_items: List[RewardResult], 
        expected_length: int,
        all_info_keys: set
    ):
        """
        Validate result integrity and consistency.
        
        Checks:
        1. Result count matches expected length
        2. Indices are continuous (0 to expected_length-1)
        3. All reward_info fields have consistent length
        
        Args:
            result_items: List of results (should be sorted)
            expected_length: Expected data length
            all_info_keys: Set of all reward_info keys
        
        Raises:
            AssertionError: If validation fails
        """
        # Validate count
        actual_length = len(result_items)
        if actual_length != expected_length:
            raise AssertionError(
                f"Result count mismatch: expected {expected_length}, got {actual_length}\n"
                f"First 5 result indices: {[item.original_index for item in result_items[:5]]}"
            )
        
        # Check index continuity
        expected_indices = list(range(expected_length))
        actual_indices = [item.original_index for item in result_items]
        
        if actual_indices != expected_indices:
            missing = set(expected_indices) - set(actual_indices)
            duplicated = [idx for idx in actual_indices if actual_indices.count(idx) > 1]
            raise AssertionError(
                f"Index discontinuity or not sorted:\n"
                f"  Expected: {expected_indices[:10]}{'...' if expected_length > 10 else ''}\n"
                f"  Actual: {actual_indices[:10]}{'...' if len(actual_indices) > 10 else ''}\n"
                f"  Missing: {sorted(missing) if missing else 'None'}\n"
                f"  Duplicated: {set(duplicated) if duplicated else 'None'}"
            )
        
        logger.info(
            f"✓ Result integrity validated: {actual_length} samples, "
            f"{len(all_info_keys)} reward_info fields"
        )

