import concurrent.futures
import copy
import datetime
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from filelock import FileLock
from meta_info.rewards.card_reward import cardRM_Qwen
from meta_info.rewards.express_reward import Map_peek_based_openai
from meta_info.rewards.faithful_reward import Map_faithful_model_based_openai
from meta_info.rewards.finance_personalization import personalizationRM
from meta_info.rewards.finance_rule import (LengthRM,
                                            Map_faithful_rule_based_num_filter,
                                            MapFormat, ThinkLengthRM)

from atomic_eval.simple_eval import SimpleEvaluator

INVALID_REWARD = -0.01

def extract_assistant_group_messages(group_messages: List[List[Dict[str, str]]]) -> List[str]:
    result = []
    for conversation in group_messages:
        # 提取当前对话中所有助手消息的内容
        assistant_contents = [
            msg["content"]
            for msg in conversation
            if msg.get("role") == "assistant"
        ]
        # 用两个换行符拼接内容并添加到结果列表
        result.append("\n\n".join(assistant_contents))
    return result

def extract_tool_calls(group_messages: List[List[Dict[str, str]]]) -> List[str]:
    result = []
    for conversation in group_messages:
        # 提取当前对话中所有工具调用的内容
        tool_calls = [
            msg["content"]
            for msg in conversation
            if msg.get("role") == "tool"
        ]
        # 用两个换行符拼接内容并添加到结果列表
        result.append("\n\n".join(tool_calls))
    return result

# 从所有子对话提取最后一轮输出内容
def extract_last_assistant_contents(group_messages: List[List[Dict[str, str]]]) -> List[str]:
    last_response_list = []
    for conversation in group_messages:
        last_assistant_content = None
        # 反向遍历对话以找到最后一个助手消息
        for message in reversed(conversation):
            if message.get('role') == 'assistant':
                last_assistant_content = message.get('content')
                break
        last_response_list.append(last_assistant_content)
    return last_response_list

# 从所有子对话中提取第一轮 user 输入，包含问题
def extract_first_user_contents(group_messages: List[List[Dict[str, str]]]) -> List[str]:
    first_user_contents = []
    for conversation in group_messages:
        first_user_content = None
        # 遍历对话以找到第一个用户消息
        for message in conversation:
            if message.get('role') == 'user':
                first_user_content = message.get('content')
                break
        first_user_contents.append(first_user_content)
    return first_user_contents
    

# group_messages: List[List[Dict[str, str]]]，输入是对话历史
def compute_score(prompt: str, group_responses: list[str], group_messages: list[list[dict]], group_extras=None, use_group_reward=False, return_details=False, return_statistics=True, **kwargs):
    # step1: 提取问题、工具调用、模型输出
    """
    prompt_contents: List[str]，模型第一轮输入，生成 prompt + query
    supply_contents: List[str]，工具调用内容列表
    assistant_contents: List[str]，模型输出内容列表
    """
    prompt_contents = extract_first_user_contents(group_messages)
    supply_contents = extract_tool_calls(group_messages)
    assistant_contents = extract_assistant_group_messages(group_messages)

    # step2: 分不同维度计算分数
    map_faithful_func = Map_faithful_model_based_openai()
    map_peek_func = Map_peek_based_openai()
    think_length_func = ThinkLengthRM()
    answer_length_func = LengthRM()
    number_filter_func = Map_faithful_rule_based_num_filter()
    map_format_func = MapFormat()

    # 基于工具调用的返回内容、模型生成的内容打分
    faithful_reward = map_faithful_func(assistant_contents, supply_contents)

    # 输入处理：1. 当前问题 2. 当前时间   
    # express_reward = map_peek_func(assistant_contents, prompt_contents)

    # 基于生成的思考长度打分，参数是思考长度上限
    # think_length_threshold = [4096 for _ in range(len(assistant_contents))]
    # think_length_reward = think_length_func(assistant_contents, think_length_threshold)

    # 基于最终答案长度打分，参数是答案长度下限
#     answer_length_threshold = [1000 for _ in range(len(assistant_contents))]
#     answer_length_reward = answer_length_func(assistant_contents, answer_length_threshold)

    # 基于供给内容，和最终答案打分
    faithful_num_exist_reward = number_filter_func(assistant_contents, supply_contents)

    # 基于供给内容，和最终答案打分
    map_format_reward = map_format_func(assistant_contents, supply_contents)

    # step3: 加权计算分数
    # total_score = (torch.tensor(faithful_reward) + torch.tensor(express_reward) + torch.tensor(
    #     think_length_reward) + torch.tensor(answer_length_reward) +
    #                torch.tensor(faithful_num_exist_reward) + torch.tensor(map_format_reward))
    total_score = (0.6 * torch.tensor(faithful_reward) + 0.2 * torch.tensor(faithful_num_exist_reward) + 0.2 * torch.tensor(map_format_reward))
    total_score = total_score.tolist()
    
    # 暂时不使用
    # personalization_func = personalizationRM()
    # card_reward_func = cardRM_Qwen()
    # # TODO: 输入处理：1. 当前问题 2. 卡片供给内容
    # card_reward = card_reward_func(assistant_contents, supply_contents)

    # # TODO: 输入内容：1. 历史对话 2. 思考内容与 answer 3. 用户画像 user_profile 字段 4. 问题
    # personlization_reward = personalization_func(assistant_contents, supply_contents)

    # step4: 记录每项分数
    reward_info = {
        "faithful_reward": faithful_reward,
        # "express_reward": express_reward,
        # "think_length_reward": think_length_reward,
        # "answer_length_reward": answer_length_reward,
        "faithful_num_exist_reward": faithful_num_exist_reward,
        "map_format_reward": map_format_reward,
    }

    return total_score, reward_info