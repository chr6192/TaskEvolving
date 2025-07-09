import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union
from functools import wraps
import time
import torch

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


class HybridRewardExecutor:
    """
    简化的混合并行执行器，统一使用 asyncio.to_thread
    - IO密集型任务：完全异步并行
    - CPU密集型任务：使用 asyncio.to_thread，由 asyncio 统一管理线程池
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
    executor = HybridRewardExecutor()
    
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
    for reward_list in [faithful_reward, express_reward, faithful_num_exist_reward, map_format_reward]:
        assert len(reward_list) == expected_length, f"{reward_list.__name__}长度不正确: {len(reward_list)} != {expected_length}"
    
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
    for i in range(min(len(assistant_contents), 3)):
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
            detail_sources = [
                ("faithful_details", faithful_details),
                ("express_details", express_details),
                ("num_exist_details", num_exist_details),
                ("format_details", format_details)
            ]
            
            for detail_key, detail_list in detail_sources:
                if i < len(detail_list):
                    detail.update({detail_key: _clean_detail_dict(detail_list[i])})
            reward_details.append(detail)
            
        return total_score, reward_info, reward_details
    else:
        return total_score, reward_info


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
    # 设置超时时间
    timeout = kwargs.get('timeout', 300.0)
    
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
            try:
                # 使用 asyncio.run 启动异步任务，但添加超时保护
                return asyncio.run(
                    asyncio.wait_for(
                        compute_score_async(
                            prompt, group_responses, group_messages, 
                            group_extras=group_extras, 
                            use_group_reward=use_group_reward, 
                            return_details=return_details, 
                            return_statistics=return_statistics, 
                            **kwargs
                        ),
                        timeout=timeout
                    )
                )
            except asyncio.TimeoutError:
                print(f"compute_score超时 ({timeout}秒)")
                # 返回默认值
                default_scores = [0.0] * len(group_responses)
                default_info = {}
                if return_details:
                    return default_scores, default_info, []
                else:
                    return default_scores, default_info
            except Exception as e:
                print(f"compute_score执行异常: {e}")
                # 返回默认值
                default_scores = [0.0] * len(group_responses)
                default_info = {}
                if return_details:
                    return default_scores, default_info, []
                else:
                    return default_scores, default_info
        else:
            # 其他RuntimeError，重新抛出
            raise


