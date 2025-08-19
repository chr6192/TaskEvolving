import asyncio
import json
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from deep_research.reward.utils import (
    INVALID_REWARD,
    convert_to_data_samples,
    extract_assistant_group_messages,
    extract_first_user_contents,
    extract_packing_faithful,
    extract_tool_calls,
)
from meta_info.rewards.express_reward import AsyncMapPeekModel, SyncMapPeekModel
from meta_info.rewards.faithful_reward import (
    AsyncMapFaithfulModel,
    SyncMapFaithfulModel,
)
from meta_info.rewards.finance_rule import MapFaithfulRuleBasedNumFilter, MapFormat
from rm_gallery.gallery.rm.agent import (
    create_agent_search_reward,
    create_tool_call_reward,
)


class HybridRewardExecutor:
    """
    优化的混合并行执行器，统一使用 asyncio.to_thread
    新架构特点：
    - 所有任务：统一由asyncio.to_thread管理，避免异步协调开销
    - faithful/express任务：内部使用call_async+as_completed实现真正并行
    - 规则任务：纯CPU计算，使用线程池处理
    - 架构更清晰：中层专注业务逻辑，上层统一管理并发
    - 避免自维护线程池的复杂性和死锁风险
    """

    def __init__(self):
        self.timings = {}
        self.io_tasks = []  # IO密集型任务列表
        self.cpu_tasks = []  # CPU密集型任务列表

    def add_io_task(self, name: str, async_func, *args, **kwargs):
        """添加IO密集型异步任务"""
        self.io_tasks.append((name, async_func, args, kwargs))

    def add_cpu_task(self, name: str, func_instance, *args, **kwargs):
        """添加CPU密集型任务"""
        self.cpu_tasks.append((name, func_instance, args, kwargs))

    async def execute_all_parallel(self):
        """并行执行所有任务 - 统一使用asyncio管理"""
        results = {}

        try:
            # 创建所有任务的协程列表
            all_coroutines = []
            task_names = []

            # IO密集型任务：直接创建异步协程
            for name, async_func, args, kwargs in self.io_tasks:
                coro = self._execute_io_task(name, async_func, *args, **kwargs)
                all_coroutines.append(coro)
                task_names.append(name)

            # CPU密集型任务：使用 asyncio.to_thread 包装
            for name, func_instance, args, kwargs in self.cpu_tasks:
                coro = self._execute_cpu_task(name, func_instance, *args, **kwargs)
                all_coroutines.append(coro)
                task_names.append(name)

            # 真正的并行执行：所有任务同时运行
            print(f"开始并行执行 {len(all_coroutines)} 个任务")
            start_time = time.perf_counter()

            task_results = await asyncio.gather(*all_coroutines, return_exceptions=True)

            total_time = time.perf_counter() - start_time
            print(f"所有任务并行执行完成，总耗时: {total_time:.4f}秒")

            # 处理结果
            for i, (name, result) in enumerate(zip(task_names, task_results)):
                if isinstance(result, Exception):
                    print(f"任务 {name} 执行出错: {result}")
                    # 根据任务类型返回合适的默认值
                    if name in ["faithful_reward", "express_reward"]:
                        results[name] = ([], [])
                    else:
                        results[name] = []
                else:
                    results[name] = result

        except Exception as e:
            print(f"并行执行任务时发生错误: {e}")
            # 返回默认结果
            for name in [task[0] for task in self.io_tasks + self.cpu_tasks]:
                if name in ["faithful_reward", "express_reward"]:
                    results[name] = ([], [])
                else:
                    results[name] = []

        return results

    async def _execute_io_task(self, name: str, async_func, *args, **kwargs):
        """执行IO密集型任务 - 完全异步"""
        start_time = time.perf_counter()
        try:
            print(f"开始执行IO任务: {name}")
            result = await async_func(*args, **kwargs)
            print(f"IO任务 {name} 执行完成")
            return result
        except Exception as e:
            print(f"IO任务 {name} 执行失败: {e}")
            raise
        finally:
            end_time = time.perf_counter()
            self.timings[f"{name}_time"] = round(end_time - start_time, 4)

    async def _execute_cpu_task(self, name: str, func_instance, *args, **kwargs):
        """执行CPU密集型任务 - 使用 asyncio.to_thread"""
        start_time = time.perf_counter()
        try:
            print(f"开始执行CPU任务: {name}")

            # 关键改进：使用 asyncio.to_thread 代替自维护线程池
            result = await asyncio.to_thread(func_instance, *args)

            print(f"CPU任务 {name} 执行完成")
            return result
        except Exception as e:
            print(f"CPU任务 {name} 执行失败: {e}")
            raise
        finally:
            end_time = time.perf_counter()
            self.timings[f"{name}_time"] = round(end_time - start_time, 4)

    def get_timings(self):
        """获取所有任务的计时信息"""
        return self.timings.copy()


def _clean_detail_dict(detail_dict: Dict[str, Any]) -> Dict[str, Any]:
    """清理详细信息字典，移除临时字段"""
    if not isinstance(detail_dict, dict):
        return {}

    # 移除以_tmp结尾的字段
    cleaned = {k: v for k, v in detail_dict.items() if not k.endswith("_tmp")}
    return cleaned


def _process_extra_info(extra_info: Dict[str, Any]) -> Dict[str, Any]:
    skeleton = extra_info.get("skeleton", {})
    assert skeleton != {}, f"skeleton is not found for index {extra_info}"
    if "tool_calls_reference" in skeleton and isinstance(
        skeleton["tool_calls_reference"], str
    ):
        try:
            skeleton["tool_calls_reference"] = json.loads(
                skeleton["tool_calls_reference"]
            )
        except json.JSONDecodeError:
            print(
                f"无法将tool_calls_reference从str解析为dict,保持原样: {skeleton['tool_calls_reference']}"
            )

    extra_info["skeleton"] = skeleton
    return extra_info


def process_reward_result(result, score_key="score"):
    """
    通用处理 rm-gallery 返回reward结果的函数
    """
    output_list = result.output if result else []
    reward_list = [output.answer.reward.score for output in output_list]
    details_list = [output.answer.additional_kwargs for output in output_list]
    return reward_list, details_list


def unpack_result(result, default_key=None):
    if isinstance(result, tuple) and len(result) == 2:
        return result
    elif isinstance(result, list):
        if default_key:
            return result, [{default_key: score} for score in result]
        else:
            return result, []
    else:
        return [], []


async def compute_score_async(
    prompt: str,
    group_responses: List[str],
    group_messages: List[List[dict]],
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
    异步reward计算函数

    新架构优化策略：
    1. 所有任务：统一由asyncio.to_thread管理，避免异步协调开销
    2. faithful/express任务：内部使用call_async+as_completed实现真正并行
    3. 规则任务：纯CPU计算，使用线程池处理
    4. 4个reward函数间：真正的并行执行
    5. 架构更清晰：中层专注业务逻辑，上层统一管理并发

    Args:
        prompt: 当前问题
        group_responses: 每个子对话的模型输出
        group_messages: 每个子对话的消息列表
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
    print(f"开始混合并行优化的异步reward计算，处理 {len(group_responses)} 个响应")
    total_start_time = time.perf_counter()

    # step1: 数据预处理
    prompt_contents = extract_first_user_contents(group_messages)
    supply_contents = extract_tool_calls(group_messages)
    failthful_contents = extract_packing_faithful(group_responses)
    assistant_contents = extract_assistant_group_messages(group_messages)
    group_extras = [_process_extra_info(extra) for extra in group_extras]

    tool_calls_reference = group_extras[0]["skeleton"]["tool_calls_reference"]
    sample = convert_to_data_samples(
        group_messages, tool_calls_reference
    )  # 创建 sample 用于RM-Gallery的reward计算

    print(f"数据预处理完成: {len(assistant_contents)} 个助手回复")

    # step2: 创建优化的执行器
    executor = HybridRewardExecutor()

    # step3: 注册IO密集型任务（网络请求）- 使用批量并行同步版本
    faithful_func = SyncMapFaithfulModel()
    express_func = SyncMapPeekModel()

    executor.add_cpu_task(
        "faithful_reward", faithful_func, assistant_contents, failthful_contents
    )
    executor.add_cpu_task(
        "express_reward", express_func, assistant_contents, prompt_contents
    )

    # step4: 注册CPU密集型任务（本地计算）
    num_filter_func = MapFaithfulRuleBasedNumFilter()
    format_func = MapFormat()
    agent_search_reward = create_agent_search_reward()
    agent_tool_call_rewards = create_tool_call_reward(
        strict_mode=False, use_jaccard_similarity=True
    )

    executor.add_cpu_task(
        "faithful_num_exist_reward",
        num_filter_func,
        assistant_contents,
        supply_contents,
    )
    executor.add_cpu_task(
        "map_format_reward", format_func, assistant_contents, supply_contents
    )
    executor.add_io_task(
        "agent_search_reward", agent_search_reward.async_evaluate, sample
    )
    executor.add_io_task(
        "agent_tool_call_reward", agent_tool_call_rewards.async_evaluate, sample
    )

    # step5: 并行执行所有任务
    print("开始并行执行所有reward任务...")
    results = await executor.execute_all_parallel()
    timings = executor.get_timings()

    # step6: 统一处理reward结果
    faithful_reward, faithful_details = unpack_result(
        results.get("faithful_reward", ([], []))
    )
    express_reward, express_details = unpack_result(
        results.get("express_reward", ([], [])), "express_score"
    )
    faithful_num_exist_reward, num_exist_details = unpack_result(
        results.get("faithful_num_exist_reward", []), "num_exist_score"
    )
    map_format_reward, format_details = unpack_result(
        results.get("map_format_reward", []), "format_score"
    )

    # 处理agent_search_reward结果
    agent_search_reward_list, agent_search_details = process_reward_result(
        results.get("agent_search_reward")
    )

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

    # 处理agent_tool_call_rewards结果
    tool_call_reward_list, tool_call_details = process_reward_result(
        results.get("agent_tool_call_reward")
    )

    # step7: 校验reward长度
    expected_length = len(assistant_contents)
    reward_dict = {
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
    for reward_name, reward_list in reward_dict.items():
        assert (
            len(reward_list) == expected_length
        ), f"{reward_name}长度不正确: {len(reward_list)} != {expected_length}"

    # step8: 计算总分
    total_score = (
        0.3 * torch.tensor(faithful_reward)
        + 0.3 * torch.tensor(express_reward)
        + 0.1 * torch.tensor(faithful_num_exist_reward)
        + 0.1 * torch.tensor(map_format_reward)
        + 0.1 * torch.tensor(agent_search_reward_list)
        + 0.1 * torch.tensor(tool_call_reward_list)
    ).tolist()

    reward_info = {k: v for k, v in reward_dict.items()}
    total_time = time.perf_counter() - total_start_time

    # step9: 打印性能统计
    print("\n" + "=" * 80)
    print("奖励计算完成 - 性能统计")
    print("=" * 80)
    print(f"总计算时间: {total_time:.4f}秒")
    print(f"各任务详细耗时: {timings}")
    print(f"IO任务数={len(executor.io_tasks)}, CPU任务数={len(executor.cpu_tasks)}")
    for i in range(min(len(assistant_contents), 3)):
        print(f"\n--- 样本 {i} ---")
        print(f"忠实性奖励: {faithful_reward[i]:.4f}")
        print(f"体感奖励: {express_reward[i]:.4f}")
        print(f"数字规则奖励: {faithful_num_exist_reward[i]:.4f}")
        print(f"格式规则奖励: {map_format_reward[i]:.4f}")
        print(f"总分: {total_score[i]:.4f}")
    if len(assistant_contents) > 3:
        print(f"... 还有 {len(assistant_contents) - 3} 个样本")
    print("=" * 80 + "\n")

    # step10: 构建详细信息（如果需要）
    if return_details:
        reward_details = []
        detail_sources = [
            ("faithful_details", faithful_details),
            ("express_details", express_details),
            ("num_exist_details", num_exist_details),
            ("format_details", format_details),
            ("agent_search_details", agent_search_details),
            ("tool_call_details", tool_call_details),
        ]
        for i in range(expected_length):
            detail = {
                "index": i,
                "total_score": total_score[i],
                "performance_stats": {
                    "total_time": total_time,
                    "individual_timings": timings,
                },
            }
            for detail_key, detail_list in detail_sources:
                if i < len(detail_list):
                    detail[detail_key] = _clean_detail_dict(detail_list[i])
            reward_details.append(detail)
        return total_score, reward_info, reward_details
    else:
        return total_score, reward_info
