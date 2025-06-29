from typing import Any, Dict, List, Optional, Tuple, Union
import random


def compute_score(prompt: str, responses: list[str], messages: list[list[dict]], extras=None, use_group_reward=False, return_details=False, return_statistics=True, **kwargs):
    details = []

    # 如果queries不是列表类型，将其转换为列表
    if not isinstance(responses, list):
        responses = [responses]
        if prompt and not isinstance(prompt, list):
            prompt = [prompt]

    batch_size = len(responses)
    
    # Mock最基本的0-1随机奖励
    scores = [random.uniform(0.0, 1.0) for _ in range(batch_size)]

    # 最简化的reward_info，只包含主要分数
    reward_info = {
        "reward_principle": scores,
    }
    
    if return_details: 
        return scores, reward_info, details
    return scores, reward_info
