import asyncio
import concurrent.futures
import threading
from typing import Any, Dict, List, Optional, Tuple, Union
from functools import wraps
import time
import torch
import copy
import json
import os

from deep_research.reward.dr_reward_function import (
    extract_first_user_contents, extract_tool_calls, 
    extract_packing_yjx, extract_assistant_group_messages,
    INVALID_REWARD
)

from meta_info.rewards.faithful_reward import (
    AsyncMapFaithfulModelBasedOpenai, 
)
from meta_info.rewards.express_reward import (
    AsyncMapPeekBasedOpenai, 
)
from meta_info.rewards.finance_rule import (
    Map_faithful_rule_based_num_filter, MapFormat
)


class GlobalThreadPoolManager:
    """
    全局线程池管理器，避免多实例创建导致的资源竞争
    单例模式，确保整个应用只有一个线程池
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            # 根据CPU核心数设置合理的线程池大小
            # 对于规则计算这种轻量级CPU任务，可以设置较多线程
            max_workers = min(os.cpu_count() * 2, 16)  # 最多16个线程
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="DR_CPU_Worker"
            )
            self._initialized = True
            self._is_shutdown = False
            print(f"创建全局线程池，线程数: {max_workers}")
    
    def get_executor(self):
        """
        获取线程池执行器
        
        Returns:
            ThreadPoolExecutor: 线程池执行器
            
        Raises:
            RuntimeError: 如果线程池已经被关闭
        """
        if getattr(self, '_is_shutdown', False):
            raise RuntimeError("线程池已经被关闭，无法继续使用")
        return self.executor
    
    def shutdown(self):
        """
        关闭线程池
        
        注意：关闭后将无法再使用，需要重新创建实例
        """
        if hasattr(self, 'executor') and not getattr(self, '_is_shutdown', False):
            self.executor.shutdown(wait=True)
            self._is_shutdown = True
            print("全局线程池已关闭")
            
    def is_shutdown(self):
        """检查线程池是否已关闭"""
        return getattr(self, '_is_shutdown', False)


class OptimizedHybridRewardExecutor:
    """
    优化的混合并行执行器，采用混合并行优化策略：
    - IO密集型任务：完全异步并行，避免阻塞等待
    - CPU密集型任务：使用全局线程池，避免资源竞争
    - 三层并行架构:
        第一层: AntfinRewardManager
            ├── Group1 (相同prompt的样本)
            ├── Group2 (相同prompt的样本)
            └── GroupN (相同prompt的样本)
                ↓ 并行处理
        第二层: HybridRewardExecutor
            ├── faithful_reward (IO密集)
            ├── express_reward (IO密集)
            ├── faithful_num_exist_reward (CPU密集)
            └── map_format_reward (CPU密集)
                ↓ 每个reward内部并行
        第三层: 单个Reward函数内部
            ├── 样本1
            ├── 样本2
            └── 样本N
    """
    
    def __init__(self):
        self.timings = {}
        self.io_tasks = []  # IO密集型任务列表
        self.cpu_tasks = []  # CPU密集型任务列表
        self.thread_pool = GlobalThreadPoolManager().get_executor()
        
    def add_io_task(self, name: str, async_func, *args, **kwargs):
        """添加IO密集型异步任务"""
        self.io_tasks.append((name, async_func, args, kwargs))
        
    def add_cpu_task(self, name: str, func_instance, *args, **kwargs):
        """添加CPU密集型任务"""
        self.cpu_tasks.append((name, func_instance, args, kwargs))
        
    async def execute_all_parallel(self):
        """并行执行所有任务 - IO用异步，CPU用全局线程池"""
        results = {}
        
        # 创建所有任务的协程列表
        all_coroutines = []
        task_names = []
        
        # IO密集型任务：直接创建异步协程
        for name, async_func, args, kwargs in self.io_tasks:
            coro = self._execute_io_task(name, async_func, *args, **kwargs)
            all_coroutines.append(coro)
            task_names.append(name)
            
        # CPU密集型任务：使用全局线程池包装成协程
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
        """执行CPU密集型任务 - 使用全局线程池"""
        start_time = time.perf_counter()
        try:
            print(f"开始执行CPU任务: {name}")
            loop = asyncio.get_running_loop()
            
            # 使用全局线程池执行任务
            result = await loop.run_in_executor(
                self.thread_pool,
                func_instance,
                *args
            )
                
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


async def compute_score_async(
    prompt: str, 
    group_responses: List[str], 
    group_messages: List[List[dict]], 
    group_extras=None, 
    use_group_reward=False, 
    return_details=False, 
    return_statistics=True, 
    **kwargs
) -> Union[Tuple[List[float], Dict[str, List[float]]], 
           Tuple[List[float], Dict[str, List[float]], List[Dict[str, Any]]]]:
    """
    异步reward计算函数
    
    优化策略：
    1. IO密集型任务：完全异步并行，避免阻塞等待
    2. CPU密集型任务：使用全局线程池，避免资源竞争
    3. 4个reward函数间：真正的并行执行
    4. 保持三层并行架构的优势
    
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
    yjx_contents = extract_packing_yjx(group_responses)
    assistant_contents = extract_assistant_group_messages(group_messages)
    
    print(f"数据预处理完成: {len(assistant_contents)} 个助手回复")
    
    # step2: 创建优化的执行器
    executor = OptimizedHybridRewardExecutor()
    
    # step3: 注册IO密集型任务（网络请求）
    faithful_func = AsyncMapFaithfulModelBasedOpenai()
    express_func = AsyncMapPeekBasedOpenai()
    
    executor.add_io_task(
        "faithful_reward",
        faithful_func,
        assistant_contents, yjx_contents
    )
    executor.add_io_task(
        "express_reward", 
        express_func,
        assistant_contents, prompt_contents
    )
    
    # step4: 注册CPU密集型任务（本地计算）
    num_filter_func = Map_faithful_rule_based_num_filter()
    format_func = MapFormat()
    
    # 创建部分函数，绑定第二个参数
    def execute_num_filter():
        return num_filter_func(assistant_contents, supply_contents)
    
    def execute_format():
        return format_func(assistant_contents, supply_contents)
    
    executor.add_cpu_task("faithful_num_exist_reward", execute_num_filter)
    executor.add_cpu_task("map_format_reward", execute_format)
    
    # step5: 并行执行所有任务
    print("开始并行执行所有reward任务...")
    results = await executor.execute_all_parallel()
    timings = executor.get_timings()
    
    # step6: 处理结果
    print("处理reward计算结果...")
    
    # 处理faithful_reward结果
    faithful_result = results.get("faithful_reward", ([], []))
    if isinstance(faithful_result, tuple) and len(faithful_result) == 2:
        faithful_reward, faithful_details = faithful_result
    else:
        faithful_reward = faithful_result if isinstance(faithful_result, list) else []
        faithful_details = []
        
    # 处理express_reward结果
    express_result = results.get("express_reward", ([], []))
    if isinstance(express_result, tuple) and len(express_result) == 2:
        express_reward, express_details = express_result
    else:
        express_reward = express_result if isinstance(express_result, list) else []
        express_details = [{"express_score": score} for score in express_reward]
        
    # 处理CPU任务结果
    faithful_num_exist_reward = results.get("faithful_num_exist_reward", [])
    if isinstance(faithful_num_exist_reward, tuple):
        faithful_num_exist_reward, num_exist_details = faithful_num_exist_reward
    else:
        num_exist_details = [{"num_exist_score": score} for score in faithful_num_exist_reward]
        
    map_format_reward = results.get("map_format_reward", [])
    if isinstance(map_format_reward, tuple):
        map_format_reward, format_details = map_format_reward
    else:
        format_details = [{"format_score": score} for score in map_format_reward]
    
    # 确保所有reward列表长度一致
    expected_length = len(assistant_contents)
    faithful_reward = _ensure_length(faithful_reward, expected_length)
    express_reward = _ensure_length(express_reward, expected_length)
    faithful_num_exist_reward = _ensure_length(faithful_num_exist_reward, expected_length)
    map_format_reward = _ensure_length(map_format_reward, expected_length)
    
    # step7: 计算总分 - 加权平均
    total_score = (
        0.4 * torch.tensor(faithful_reward) + 
        0.4 * torch.tensor(express_reward) + 
        0.1 * torch.tensor(faithful_num_exist_reward) + 
        0.1 * torch.tensor(map_format_reward)
    )
    total_score = total_score.tolist()
    
    # step8: 构建返回结果
    reward_info = {
        "faithful_reward": faithful_reward,
        "express_reward": express_reward,
        "faithful_num_exist_reward": faithful_num_exist_reward,
        "map_format_reward": map_format_reward,
    }
    
    total_time = time.perf_counter() - total_start_time
    
    # step9: 打印性能统计
    print("\n" + "="*80)
    print("奖励计算完成 - 性能统计")
    print("="*80)
    print(f"总计算时间: {total_time:.4f}秒")
    print(f"各任务详细耗时: {timings}")
    print(f"IO任务数={len(executor.io_tasks)}, CPU任务数={len(executor.cpu_tasks)}")
    
    # 详细性能分析
    for i in range(min(len(assistant_contents), 3)):  # 只显示前3个样本
        print(f"\n--- 样本 {i} ---")
        print(f"忠实性奖励: {faithful_reward[i]:.4f}")
        print(f"体感奖励: {express_reward[i]:.4f}")
        print(f"数字规则奖励: {faithful_num_exist_reward[i]:.4f}")
        print(f"格式规则奖励: {map_format_reward[i]:.4f}")
        print(f"总分: {total_score[i]:.4f}")
        
    if len(assistant_contents) > 3:
        print(f"... 还有 {len(assistant_contents) - 3} 个样本")
        
    print("="*80 + "\n")
    
    # step10: 构建详细信息（如果需要）
    if return_details:
        reward_details = []
        for i in range(len(assistant_contents)):
            detail = {
                "index": i,
                "total_score": total_score[i],
                "performance_stats": {
                    "total_time": total_time,
                    "individual_timings": timings,
                }
            }
            
            # 合并各个reward的详细信息
            if i < len(faithful_details):
                detail.update({"faithful_details": _clean_detail_dict(faithful_details[i])})
            if i < len(express_details):
                detail.update({"express_details": _clean_detail_dict(express_details[i])})
            if i < len(num_exist_details):
                detail.update({"num_exist_details": _clean_detail_dict(num_exist_details[i])})
            if i < len(format_details):
                detail.update({"format_details": _clean_detail_dict(format_details[i])})
                
            reward_details.append(detail)
            
        return total_score, reward_info, reward_details
    else:
        return total_score, reward_info


def _ensure_length(reward_list: List[float], expected_length: int) -> List[float]:
    """确保reward列表长度正确"""
    if len(reward_list) == expected_length:
        return reward_list
    elif len(reward_list) < expected_length:
        # 长度不足，用0填充
        return reward_list + [0.0] * (expected_length - len(reward_list))
    else:
        # 长度过长，截断
        return reward_list[:expected_length]


def _clean_detail_dict(detail_dict: Dict[str, Any]) -> Dict[str, Any]:
    """清理详细信息字典，移除临时字段"""
    if not isinstance(detail_dict, dict):
        return {}
    
    # 移除以_tmp结尾的字段
    cleaned = {k: v for k, v in detail_dict.items() if not k.endswith("_tmp")}
    return cleaned


def compute_score(
    prompt: str, 
    group_responses: List[str], 
    group_messages: List[List[dict]], 
    group_extras=None, 
    use_group_reward=False, 
    return_details=False, 
    return_statistics=True, 
    **kwargs
) -> Union[Tuple[List[float], Dict[str, List[float]]], 
           Tuple[List[float], Dict[str, List[float]], List[Dict[str, Any]]]]:
    """
    同步版本compute_score，保持向后兼容性
    
    内部使用异步优化版本，通过asyncio.run包装
    适用于不在异步上下文中的调用场景
    """
    try:
        # 检查是否已经在事件循环中
        loop = asyncio.get_running_loop()
        # 如果已经在事件循环中，抛出错误提示
        raise RuntimeError(
            "compute_score不能在运行中的事件循环内调用。"
            "请在异步上下文中使用 await compute_score_async() 代替。"
        )
    except RuntimeError as e:
        if "no running event loop" in str(e):
            # 没有事件循环，可以安全使用 asyncio.run
            return asyncio.run(compute_score_async(
                prompt, group_responses, group_messages, 
                group_extras=group_extras, 
                use_group_reward=use_group_reward, 
                return_details=return_details, 
                return_statistics=return_statistics, 
                **kwargs
            ))
        else:
            # 其他RuntimeError，重新抛出
            raise


# 提供显式的清理方法，让调用者在适当的时候调用
def cleanup_thread_pool():
    """
    显式清理全局线程池
    
    注意：只有在确定不再需要使用DR reward计算时才调用此函数
    通常在程序结束或需要重新初始化时使用
    """
    try:
        GlobalThreadPoolManager().shutdown()
        print("手动清理全局线程池完成")
    except:
        pass

