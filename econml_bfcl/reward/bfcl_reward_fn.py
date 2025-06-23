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
from econml_bfcl.tool.env_handler import EnvHandler


def compute_score(messages, original_test_entry):
    """The scoring function for BFCL.

    Args:
        messages: List of conversation messages
        original_test_entry: The original test entry data
        
    Returns:
        Float score representing the evaluation result
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
        eval_result = env_handler.evaluate(test_entry)
        # 将test_entry和eval_result写入本地json文件
        import json
        import os
        from datetime import datetime
        
        # 创建输出目录
        output_dir = "/redacted/internal-path"
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文件名,包含时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/reward_logx.json"
        
        # 准备要写入的数据
        log_data = {
            "messages": messages,
            "eval_result": eval_result
        }
        
        # 追加写入json文件
        try:
            with open(filename, "a", encoding="utf-8") as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"写入日志文件失败: {str(e)}")
        # Extract score from evaluation result
        # if eval_result.get("valid", True):
        #     return eval_result.get("accuracy")

        # else:
        #     print(f"Evaluation error: {eval_result.get('error', 'Unknown error')}")
        #     raise Exception(f"Evaluation error: {eval_result.get('error', 'Unknown error')}")
        #     return 0.0
        import random
        return random.choice([0, 1])
            
    except Exception as e:
        print(f"Error in compute_score: {str(e)}\nmessages: {messages}\nkeys of original_test_entry: {original_test_entry.keys()}")
        raise Exception(f"Error in compute_score: {str(e)}\nmessages: {messages}\nkeys of original_test_entry: {original_test_entry.keys()}")
        return 0.0
