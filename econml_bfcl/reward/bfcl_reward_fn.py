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

import re
from typing import Any, Dict, List, Optional, Tuple, Union
from econml_bfcl.tool.env_handler import EnvHandler


def compute_score(messages, original_test_entry, return_details=False, score_dir="/redacted/internal-path"):
    """The scoring function for BFCL.

    Args:
        messages: List of conversation messages
        original_test_entry: The original test entry data
        return_details: Whether to return detailed information
        score_dir: Directory path for storing score details
        
    Returns:
        If return_details=False: Float score representing the evaluation result
        If return_details=True: Tuple of (score, reward_info, details)
    """
    messages = messages.get("messages", messages) if isinstance(messages, dict) else messages
    assert isinstance(messages, list), f"messages is not a list: {messages}"
    messages = [msg.model_dump() if hasattr(msg, 'model_dump') else msg for msg in messages]
    assert isinstance(original_test_entry, dict), f"original_test_entry is not a dict: {original_test_entry}"
    
    try:
        # Initialize EnvHandler
        env_handler = EnvHandler()
        
        # Extract test_id from original_test_entry
        test_id = original_test_entry.get("id")
        
        # Count turns and tokens (basic implementation)
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        turn_count = len(user_messages)
        
        # Calculate basic token counts (simplified)
        total_input_tokens = 0
        total_output_tokens = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                tokens = len(content.split())  # Rough token estimation
                if msg.get("role") == "user":
                    total_input_tokens += tokens
                elif msg.get("role") == "assistant":
                    total_output_tokens += tokens
        
        # Check if conversation is completed (basic heuristic)
        completed = len(messages) > 0 and messages[-1].get("role") in ["assistant", "tool"]
        
        # Assemble test_entry in the format expected by EnvHandler.evaluate
        test_entry = {
            "test_id": test_id,
            "messages": messages,
            "turn_count": turn_count,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "completed": completed,
            "original_test_entry": original_test_entry
        }
        
        # Call EnvHandler.evaluate
        eval_result = env_handler.evaluate(test_entry, score_dir=score_dir)
        
        # Extract score from evaluation result
        if eval_result.get("valid", True):
            score = eval_result.get("accuracy", 0.0)
            
            if return_details:
                # 构建详细信息
                reward_info = {
                    "accuracy": score,
                    "valid": eval_result.get("valid", True),
                    "total_count": eval_result.get("total_count", 1),
                    "correct_count": eval_result.get("correct_count", 0),
                    "turn_count": turn_count
                }
                
                details = {
                    "test_id": test_id,
                    "test_category": eval_result.get("test_category", "unknown"),
                    "model_name": eval_result.get("model_name", "unknown"),
                    "completed": completed,
                    "eval_result": eval_result,
                    "original_test_entry": original_test_entry
                }
                
                return score, reward_info, details
            else:
                return score
        else:
            print(f"Evaluation error: {eval_result.get('error', 'Unknown error')}")
            error_msg = f"Evaluation error: {eval_result.get('error', 'Unknown error')}"
            
            if return_details:
                reward_info = {
                    "accuracy": -99.0,
                    "valid": False,
                    "error": eval_result.get('error', 'Unknown error'),
                    "turn_count": turn_count
                }
                details = {
                    "test_id": test_id,
                    "error": eval_result.get('error', 'Unknown error'),
                    "eval_result": eval_result,
                    "original_test_entry": original_test_entry
                }
                return -99.0, reward_info, details
            else:
                raise Exception(error_msg)
            
    except Exception as e:
        print(f"Error in compute_score: {str(e)}\nmessages: {messages}\nkeys of original_test_entry: {original_test_entry.keys()}")
        error_msg = f"Error in compute_score: {str(e)}\nmessages: {messages}\nkeys of original_test_entry: {original_test_entry.keys()}"
        
        if return_details:
            reward_info = {
                "accuracy": -99.0,
                "valid": False,
                "error": str(e),
                "turn_count": 0
            }
            details = {
                "test_id": original_test_entry.get("id", "unknown"),
                "error": str(e),
                "exception_type": type(e).__name__,
                "original_test_entry": original_test_entry
            }
            return -99.0, reward_info, details
        else:
            raise Exception(error_msg)