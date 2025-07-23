# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


class BfclRewardManager:
    """The reward manager for BFCL."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", **reward_kwargs) -> None:
        """
        Initialize the BfclRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to "data_source".
            **reward_kwargs: Additional arguments for reward computation. Common kwargs include:
                - return_details: Whether to return detailed information (bool)
                - score_dir: Directory path for storing score details (str)
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.reward_kwargs = reward_kwargs  # Store additional reward arguments
        # 默认启用详细信息返回
        self.reward_kwargs.setdefault("return_details", True)

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        all_details = []  # 收集所有详细信息

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            messages = data_item.non_tensor_batch["messages"]
            original_test_entry = data_item.non_tensor_batch["original_test_entry"]

            try:
                result = self.compute_score(messages, original_test_entry, **self.reward_kwargs)
                
                if self.reward_kwargs.get("return_details", False) and isinstance(result, tuple) and len(result) == 3:
                    score, reward_info, details = result
                    
                    if details:
                        all_details.append(details)
                    
                    if reward_info and isinstance(reward_info, dict):
                        for key, value in reward_info.items():
                            reward_extra_info[key].append(value)
                else:
                    score = result
                    
            except Exception as e:
                print(f"Error in compute_score for item {i}: {e}")
                score = 0.0
                if self.reward_kwargs.get("return_details", False):
                    error_details = {
                        "test_id": original_test_entry.get("id", "unknown"),
                        "error": str(e),
                        "exception_type": type(e).__name__,
                        "item_index": i,
                    }
                    all_details.append(error_details)
                    reward_extra_info["error"].append(str(e))

            if isinstance(score, dict):
                reward = score.get("score", -99.0)
            else:
                reward = score if score is not None else -99.0

            reward_tensor[i, valid_response_length - 1] = reward

            # 打印调试信息
            if i < self.num_examine:
                print(f"[Sample {i}]")
                print(f"[prompt] {prompt_str}")
                print(f"[response] {response_str}")
                print(f"[score] {reward}")
                if all_details and i < len(all_details):
                    print(f"[details] {all_details[i]}")

        if return_dict:
            result = {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(reward_extra_info),
            }
            # 如果有详细信息，添加到返回结果中
            if all_details:
                result["all_details"] = all_details
            return result
        else:
            return reward_tensor
