from typing import Any, Dict, List, Optional, Tuple, Union
import random

def compute_score(prompt: str, group_responses: list[str], group_messages: list[list[dict]], group_extras: list[dict]=None, use_group_reward=False, return_details=False, return_statistics=True, **kwargs):
    details = []

    # 如果queries不是列表类型，将其转换为列表
    if not isinstance(group_responses, list):
        group_responses = [group_responses]
        if group_messages and not isinstance(group_messages, list):
            group_messages = [group_messages]
        if group_extras and not isinstance(group_extras, list):
            group_extras = [group_extras]

    batch_size = len(group_responses)
    
    # Mock最基本的0-1随机奖励
    scores = [random.uniform(0.0, 1.0) for _ in range(batch_size)]

    # 最简化的reward_info，只包含主要分数
    reward_info = {
        "reward_principle": scores,
    }
    
    if return_details: 
        return scores, reward_info, details
    return scores, reward_info
