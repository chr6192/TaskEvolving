import asyncio
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import time
from collections import defaultdict
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from deep_research.reward.utils import convert_to_data_samples
from rm_gallery.gallery.rm.agent import (
    create_agent_search_reward,
    create_tool_call_reward,
)


async def compute_score_async(
    prompt: str,
    group_responses: List[str],
    group_messages: List[List[dict]],
    rollout_data: Optional[Dict] = None,
    group_extras=None,
    use_group_reward=False,
    return_details=False,
    return_statistics=True,
    **kwargs,
) -> Union[
    Tuple[List[float], Dict[str, List[float]]],
    Tuple[List[float], Dict[str, List[float]], List[Dict[str, Any]]],
]:
    """
    异步reward计算函数 - 优化版本（只计算agent_search_reward）

    优化策略：
    1. 只计算agent_search_reward，获取agent_search_details
    2. 其他奖励（faithful_reward, express_reward, faithful_num_exist_reward, map_format_reward）从rollout.json读取
    3. 大幅减少计算开销，提高执行效率
    4. 保持原有的返回格式和接口兼容性

    Args:
        prompt: 当前问题
        group_responses: 每个子对话的模型输出
        group_messages: 每个子对话的消息列表
        rollout_data: rollout.json文件数据，包含预计算的奖励信息
        group_extras: 额外信息
        use_group_reward: 是否使用组奖励
        return_details: 是否返回详细信息
        return_statistics: 是否返回统计信息
        **kwargs: 其他参数

    Returns:
        total_score: 每个子对话的总分
        reward_info: 包含各个维度的分数
        reward_details: 每个子对话的详细分数信息（可选）
    """
    print(
        f"开始优化的异步reward计算（仅计算agent_search_reward），处理 {len(group_responses)} 个响应"
    )
    total_start_time = time.perf_counter()

    # 初始化结果存储
    total_score = []
    reward_info = defaultdict(list)
    reward_details = []

    # 从rollout_data中提取奖励信息和查询列表
    (
        faithful_reward,
        express_reward,
        faithful_num_exist_reward,
        map_format_reward,
    ) = _process_rollout_data(rollout_data, group_responses)

    # 从JSON文件读取tool_calls_reference
    tool_calls_reference = group_extras.get("tool_calls_reference", [])

    sample = convert_to_data_samples(
        group_messages, tool_calls_reference
    )  # 创建 sample 用于RM-Gallery的reward计算
    agent_search_reward = create_agent_search_reward()
    agent_tool_call_rewards = create_tool_call_reward(
        strict_mode=False, use_jaccard_similarity=True
    )  # 创建agent_tool_call_rewards

    start_time = time.perf_counter()
    agent_search_result = await agent_search_reward.async_evaluate(sample)
    agent_search_time = time.perf_counter() - start_time

    agent_tool_call_result = await agent_tool_call_rewards.async_evaluate(sample)

    # 处理agent_search_reward结果
    agent_search_output_list = agent_search_result.output if agent_search_result else []
    agent_search_reward_list = [
        output.answer.reward.score for output in agent_search_output_list
    ]
    agent_search_details = [
        output.answer.additional_kwargs for output in agent_search_output_list
    ]

    # 处理agent_tool_call_rewards结果
    tool_call_output_list = (
        agent_tool_call_result.output if agent_tool_call_result else []
    )
    tool_call_reward_list = [
        output.answer.reward.score for output in tool_call_output_list
    ]
    tool_call_details = [
        output.answer.additional_kwargs for output in tool_call_output_list
    ]

    # 从agent_search_details中提取额外的奖励分数
    search_info_reward_list = []
    action_loop_reward_list = []
    unnecessary_action_penalty_list = []

    for detail in agent_search_details:
        search_info_reward_list.append(
            detail.get("search_info_reward", {}).get("score", 0.0)
        )
        action_loop_reward_list.append(
            detail.get("action_loop_detection", {}).get("score", 0.0)
        )
        unnecessary_action_penalty_list.append(
            detail.get("unnecessary_action_penalty", {}).get("score", 0.0)
        )

    print(f"agent_search_reward计算完成，耗时: {agent_search_time:.4f}秒")

    # 确保所有奖励列表长度一致
    expected_length = len(group_responses)
    reward_lists = {
        "faithful_reward": faithful_reward,
        "express_reward": express_reward,
        "faithful_num_exist_reward": faithful_num_exist_reward,
        "map_format_reward": map_format_reward,
        "agent_search_reward": agent_search_reward_list,
        "search_info_reward": search_info_reward_list,
        "action_loop_reward": action_loop_reward_list,
        "unnecessary_action_penalty": unnecessary_action_penalty_list,
        "tool_call_reward": tool_call_reward_list,
    }

    for reward_name, reward_list in reward_lists.items():
        if len(reward_list) != expected_length:
            print(
                f"警告: {reward_name}长度不匹配，期望{expected_length}，实际{len(reward_list)}"
            )
            # 补齐或截断到期望长度
            if len(reward_list) < expected_length:
                reward_list.extend([0.0] * (expected_length - len(reward_list)))
            elif len(reward_list) > expected_length:
                reward_list = reward_list[:expected_length]
            reward_lists[reward_name] = reward_list

    # 计算总分
    total_score = (
        0.3 * torch.tensor(faithful_reward)
        + 0.3 * torch.tensor(express_reward)
        + 0.1 * torch.tensor(faithful_num_exist_reward)
        + 0.1 * torch.tensor(map_format_reward)
        + 0.1 * torch.tensor(agent_search_reward_list)
        + 0.1 * torch.tensor(tool_call_reward_list)
    ).tolist()

    # 构建reward_info
    reward_info = {k: v for k, v in reward_lists.items()}

    total_time = time.perf_counter() - total_start_time

    # 打印性能统计
    print("\n" + "=" * 80)
    print("奖励计算完成 - 性能统计（优化版本）")
    print("=" * 80)
    print(f"总计算时间: {total_time:.4f}秒")
    print(f"agent_search_reward计算耗时: {agent_search_time:.4f}秒")

    # 构建详细信息（如果需要）
    if return_details:
        for i in range(expected_length):
            detail = {
                "index": i,
                "response": group_responses[i],
                "total_score": total_score[i],
                "scores": {
                    "faithful_reward": faithful_reward[i],
                    "express_reward": express_reward[i],
                    "faithful_num_exist_reward": faithful_num_exist_reward[i],
                    "map_format_reward": map_format_reward[i],
                    "agent_search_reward": agent_search_reward_list[i],
                    "tool_call_reward": tool_call_reward_list[i],
                },
                "agent_search_details": agent_search_details[i],
                "tool_call_details": tool_call_details[i],
            }

            reward_details.append(detail)

        return total_score, reward_info, reward_details
    else:
        return total_score, reward_info


def _process_rollout_data(rollout_data, group_responses):
    """处理rollout_data提取奖励信息和查询列表"""
    faithful_reward = []
    express_reward = []
    faithful_num_exist_reward = []
    map_format_reward = []

    if rollout_data:
        print("从已有的 rollout 文件读取预计算的奖励信息...")
        for item in rollout_data:
            faithful_reward.append(item.get("faithful_reward", 0.0))
            express_reward.append(item.get("express_reward", 0.0))
            faithful_num_exist_reward.append(item.get("faithful_num_exist_reward", 0.0))
            map_format_reward.append(item.get("map_format_reward", 0.0))

    else:
        # 如果没有rollout_data，使用默认值
        print("警告: 没有提供rollout_data，使用默认奖励值")
        list_len = len(group_responses)
        faithful_reward = [0.0] * list_len
        express_reward = [0.0] * list_len
        faithful_num_exist_reward = [0.0] * list_len
        map_format_reward = [0.0] * list_len

    return (
        faithful_reward,
        express_reward,
        faithful_num_exist_reward,
        map_format_reward,
    )
