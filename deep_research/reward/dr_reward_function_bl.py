"""
基于 RM-Gallery 的 Reward Function
使用 RM-Gallery 提供的金融深度研究奖励：
- financial_deep_research_resolution: 金融研究解决率奖励
- financial_deep_research_faithful: 金融研究忠实度奖励
- information_reward: 信息检索奖励
- loop_detection: 循环检测奖励
"""
import asyncio
import json
import logging
import os
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)
from deep_research.reward.utils import (
    INVALID_REWARD,
    convert_to_data_samples,
    extract_assistant_group_messages,
    extract_tool_calls,
)

try:
    from rm_gallery.gallery.rm.agent import (
        create_information_reward,
        create_loop_detection_reward,
    )
    from rm_gallery.gallery.rm.agent.financial_deep_research_resolution import (
        FinancialDeepResearchResolutionReward,
    )
    from rm_gallery.gallery.rm.agent.financial_deep_research_faithful import (
        FinancialDeepResearchFaithfulReward,
    )
    from rm_gallery.core.model.openai_llm import OpenaiLLM
    HAS_RM_GALLERY = True
except ImportError as e:
    logger.warning(f"rm_gallery 存在异常，将跳过相关奖励计算。详细错误: {e}")
    HAS_RM_GALLERY = False


class HybridRewardExecutor:
    """
    优化的混合并行执行器，统一使用 asyncio.to_thread
    新架构特点：
    - 所有任务：统一由asyncio.to_thread管理，避免异步协调开销
    - 规则任务：纯CPU计算，使用线程池处理
    - 架构更清晰：中层专注业务逻辑，上层统一管理并发
    - 避免自维护线程池的复杂性和死锁风险
    
    注意：此版本移除了依赖远程服务的 faithful/express 奖励
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
            logger.info(f"开始并行执行 {len(all_coroutines)} 个任务")
            start_time = time.perf_counter()

            task_results = await asyncio.gather(*all_coroutines, return_exceptions=True)

            total_time = time.perf_counter() - start_time
            logger.info(f"所有任务并行执行完成，总耗时: {total_time:.4f}秒")

            # 处理结果
            for i, (name, result) in enumerate(zip(task_names, task_results)):
                if isinstance(result, Exception):
                    logger.error(f"任务 {name} 执行出错: {result}")
                    # 返回合适的默认值
                    results[name] = []
                else:
                    results[name] = result

        except Exception as e:
            logger.error(f"并行执行任务时发生错误: {e}")
            # 返回默认结果
            for name in [task[0] for task in self.io_tasks + self.cpu_tasks]:
                results[name] = []

        return results

    async def _execute_io_task(self, name: str, async_func, *args, **kwargs):
        """执行IO密集型任务 - 完全异步"""
        start_time = time.perf_counter()
        try:
            logger.debug(f"开始执行IO任务: {name}")
            # 针对 rubrics_reward 添加详细日志
            if name == "rubrics_reward":
                logger.info(f"[{name}] 调用参数: args长度={len(args)}, 第一个arg类型={type(args[0]) if args else 'N/A'}, 长度={len(args[0]) if args and hasattr(args[0], '__len__') else 'N/A'}")
            
            result = await async_func(*args, **kwargs)
            
            # 针对 rubrics_reward 添加结果日志
            if name == "rubrics_reward":
                logger.info(f"[{name}] 返回结果类型={type(result)}, is tuple={isinstance(result, tuple)}, 长度={len(result) if isinstance(result, (tuple, list)) else 'N/A'}")
            
            logger.debug(f"IO任务 {name} 执行完成")
            return result
        except Exception as e:
            logger.error(f"IO任务 {name} 执行失败: {e}")
            raise
        finally:
            end_time = time.perf_counter()
            self.timings[f"{name}_time"] = round(end_time - start_time, 4)

    async def _execute_cpu_task(self, name: str, func_instance, *args, **kwargs):
        """执行CPU密集型任务 - 使用 asyncio.to_thread"""
        start_time = time.perf_counter()
        try:
            logger.debug(f"开始执行CPU任务: {name}")

            # 关键改进：使用 asyncio.to_thread 代替自维护线程池
            result = await asyncio.to_thread(func_instance, *args)

            logger.debug(f"CPU任务 {name} 执行完成")
            return result
        except Exception as e:
            logger.error(f"CPU任务 {name} 执行失败: {e}")
            raise
        finally:
            end_time = time.perf_counter()
            self.timings[f"{name}_time"] = round(end_time - start_time, 4)

    def get_timings(self):
        """获取所有任务的计时信息"""
        return self.timings.copy()


def _serialize_value(value: Any) -> Any:
    """
    递归序列化值，处理不可JSON序列化的对象

    Args:
        value: 要序列化的值

    Returns:
        可序列化的值
    """
    if isinstance(value, set):
        return list(value)
    elif isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    elif hasattr(value, "__dict__"):
        # 对于自定义对象，尝试序列化其__dict__
        try:
            return _serialize_value(value.__dict__)
        except:
            return str(value)
    else:
        try:
            # 测试是否可以直接JSON序列化
            import json

            json.dumps(value)
            return value
        except (TypeError, ValueError):
            # 如果不能序列化，转换为字符串
            return str(value)


def _clean_detail_dict(detail_dict: Dict[str, Any]) -> Dict[str, Any]:
    """清理详细信息字典，移除临时字段并处理不可序列化对象"""
    # 处理字符串格式的 JSON（先解析再清理）
    if isinstance(detail_dict, str):
        try:
            detail_dict = json.loads(detail_dict)
        except json.JSONDecodeError:
            return {"error": "invalid_json_string", "raw_value": detail_dict[:200]}
    
    if not isinstance(detail_dict, dict):
        return {}

    # 移除以_tmp结尾的字段
    cleaned = {k: v for k, v in detail_dict.items() if not k.endswith("_tmp")}

    # 递归处理所有值，确保可以JSON序列化
    cleaned = {k: _serialize_value(v) for k, v in cleaned.items()}

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
            logger.warning(
                f"无法将tool_calls_reference从str解析为dict,保持原样: {skeleton['tool_calls_reference']}"
            )

    extra_info["skeleton"] = skeleton
    return extra_info


def process_reward_result(result, score_key="score"):
    """
    通用处理 rm-gallery 返回reward结果的函数
    """
    logger.debug(f"result.output[0].answer.reward.score: {result.output[0].answer.reward.score}")
    logger.debug(f"result.output[0].answer.additional_kwargs: {result.output[0].answer.additional_kwargs}")
    output_list = result.output if result else []
    reward_list = [output.answer.reward.score for output in output_list]
    details_list = [output.answer.additional_kwargs for output in output_list]
    return reward_list, details_list


def process_financial_resolution_reward_result(result):
    """
    专门处理 financial resolution reward 结果的函数
    financial resolution reward 的结构不同，score存储在 additional_kwargs.resolution.total_score 中
    """
    if not result or not hasattr(result, "output"):
        return [], []

    output_list = result.output
    reward_list = []
    details_list = []

    for output in output_list:
        # 从 additional_kwargs 中提取 total_score
        additional_kwargs = output.answer.additional_kwargs
        if ("financial_deep_research_resolution" in additional_kwargs and "total_score" in additional_kwargs["financial_deep_research_resolution"]):
            total_score = additional_kwargs["financial_deep_research_resolution"]["total_score"]
            reward_list.append(total_score)
        else:
            # 如果没有找到 total_score，使用默认值 -99.0
            logger.warning(f"没有找到 total_score，使用默认值 -100, additional_kwargs: {additional_kwargs}")
            reward_list.append(-99)

        details_list.append(additional_kwargs)

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
    异步reward计算函数（基于 RM-Gallery）
    - search_info_reward - 搜索信息奖励（rm_gallery）
    - action_loop_reward - 循环检测奖励（rm_gallery）
    - financial_resolution_reward - 金融研究解决率奖励（rm_gallery）
    - financial_faithful_reward - 金融研究忠实度奖励（rm_gallery）
    - rubrics_reward - Rubrics评估奖励

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
    logger.info(f"开始基于 RM-Gallery 的异步reward计算，处理 {len(group_responses)} 个响应")
    total_start_time = time.perf_counter()
    
    # 从 group_extras 中提取 rubrics 和构建 trajectories（合并处理，避免冗余循环）
    rubrics_list = []
    trajectories_for_rubrics = []
    
    # 统计信息
    rubrics_stats = {
        'total': 0,
        'valid': 0,
        'not_exist': 0,
        'parse_failed': 0,
        'build_failed': 0
    }
    
    if group_extras:
        from deep_research.synthesis.core.data_structures import RubricsData, TrajectoryData, ToolCall, ToolResponse
        
        logger.debug(f"开始从 {len(group_extras)} 个 group_extras 中提取 rubrics")
        
        for idx, extra in enumerate(group_extras):
            # === Part 1: 提取 rubrics ===
            rubrics_stats['total'] += 1
            rubrics_dict = extra.get('rubrics', None)
            rubrics = None
            
            if not rubrics_dict:
                # rubrics 不存在
                rubrics_stats['not_exist'] += 1
                logger.debug(f"[{idx}] rubrics 不存在于 extra")
            elif isinstance(rubrics_dict, str):
                # 处理字符串格式（JSON字符串）
                logger.debug(f"[{idx}] rubrics 是字符串，长度={len(rubrics_dict)}")
                try:
                    rubrics_dict = json.loads(rubrics_dict)
                    logger.debug(f"[{idx}] ✓ 成功解析 rubrics 字符串为 dict")
                except json.JSONDecodeError as e:
                    rubrics_stats['parse_failed'] += 1
                    logger.warning(f"[{idx}] ✗ rubrics JSON解析失败: {e}, 前100字符: {rubrics_dict[:100]}")
                    rubrics_dict = None
            
            # 处理dict格式或RubricsData对象
            if isinstance(rubrics_dict, dict):
                try:
                    # Pydantic会自动验证和转换字段类型（包括@field_validator）
                    rubrics = RubricsData.from_dict(rubrics_dict)
                    rubrics_stats['valid'] += 1
                    logger.debug(f"[{idx}] ✓ 成功构建 RubricsData")
                except Exception as e:
                    rubrics_stats['build_failed'] += 1
                    logger.error(f"[{idx}] ✗ 从dict构建RubricsData失败: {e}", exc_info=True)
            elif isinstance(rubrics_dict, RubricsData):
                rubrics = rubrics_dict
                rubrics_stats['valid'] += 1
                logger.debug(f"[{idx}] rubrics 已经是 RubricsData 对象")
            
            rubrics_list.append(rubrics)
            
            # === Part 2: 构建 TrajectoryData ===
            tool_calls_dicts = extra.get('tool_calls', [])
            tool_responses_dicts = extra.get('tool_responses', [])
            
            # 处理 tool_calls 可能是字符串的情况
            if isinstance(tool_calls_dicts, str):
                try:
                    tool_calls_dicts = json.loads(tool_calls_dicts)
                except json.JSONDecodeError as e:
                    logger.warning(f"无法将tool_calls从str解析为list (idx={idx}): {e}")
                    tool_calls_dicts = []
            
            # 处理 tool_responses 可能是字符串的情况
            if isinstance(tool_responses_dicts, str):
                try:
                    tool_responses_dicts = json.loads(tool_responses_dicts)
                except json.JSONDecodeError as e:
                    logger.warning(f"无法将tool_responses从str解析为list (idx={idx}): {e}")
                    tool_responses_dicts = []
            
            tool_calls = [
                ToolCall(
                    name=tc.get('name', ''),
                    parameters=tc.get('parameters', {}),
                    tool_call_id=tc.get('tool_call_id', '')
                ) for tc in tool_calls_dicts
            ]
            
            tool_responses = [
                ToolResponse(
                    tool_call_id=tr.get('tool_call_id', ''),
                    content=tr.get('content', ''),
                    reward_score=tr.get('reward_score', 0.0),
                    metrics=tr.get('metrics', {}),
                    success=tr.get('success', True),
                    error_message=tr.get('error_message', None)
                ) for tr in tool_responses_dicts
            ]
            
            # 提取 query（从 skeleton 或 messages 中）
            skeleton = extra.get('skeleton', {})
            
            # 处理 skeleton 可能是字符串的情况
            if isinstance(skeleton, str):
                try:
                    skeleton = json.loads(skeleton)
                except json.JSONDecodeError as e:
                    logger.warning(f"无法将skeleton从str解析为dict (idx={idx}): {e}")
                    skeleton = {}
            
            query_content = skeleton.get('query', '')
            if not query_content and idx < len(group_messages):
                # 从 messages 中提取第一个 user 消息
                user_msgs = [m for m in group_messages[idx] if m.get('role') == 'user']
                query_content = user_msgs[0].get('content', '') if user_msgs else ''
            
            # 构建 metadata，确保包含 query
            metadata = extra.get('metadata', {})
            
            # 处理 metadata 可能是字符串的情况
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError as e:
                    logger.warning(f"无法将metadata从str解析为dict (idx={idx}): {e}")
                    metadata = {}
            
            metadata = metadata.copy()
            if 'query' not in metadata:
                metadata['query'] = query_content
            
            trajectory = TrajectoryData(
                trajectory_id=extra.get('trajectory_id', ''),
                skeleton_id=extra.get('skeleton_id', ''),
                query_id=skeleton.get('query_id', extra.get('trajectory_id', '')),
                query=query_content,  # 直接在创建时传入query字段
                messages=group_messages[idx] if idx < len(group_messages) else [],
                tool_calls=tool_calls,
                tool_responses=tool_responses,
                metadata=metadata
            )
            trajectories_for_rubrics.append(trajectory)
    
    # 输出详细的 rubrics 提取统计
    logger.info("=" * 60)
    logger.info("Rubrics 提取统计:")
    logger.info(f"  总数: {rubrics_stats['total']}")
    logger.info(f"  ✓ 有效: {rubrics_stats['valid']}")
    logger.info(f"  - 不存在: {rubrics_stats['not_exist']}")
    logger.info(f"  ✗ 解析失败: {rubrics_stats['parse_failed']}")
    logger.info(f"  ✗ 构建失败: {rubrics_stats['build_failed']}")
    logger.info(f"  Trajectories 构建数量: {len(trajectories_for_rubrics)}")
    logger.info("=" * 60)

    # step0: 初始化 LLM 用于 RM-Gallery 奖励
    financial_resolution_reward = None
    financial_faithful_reward = None
    
    if HAS_RM_GALLERY:
        try:
            llm = OpenaiLLM(
                model="qwen3-max",
                enable_thinking=True,
                base_url="http://dashscope.aliyuncs.com/compatible-mode/v1",
                openai_api_key=os.getenv("OPENAI_API_KEY"),
            )
            
            # 初始化 Financial Resolution Reward
            financial_resolution_reward = FinancialDeepResearchResolutionReward(llm=llm)
            logger.info("✅ Financial Resolution Reward 初始化成功")
            
            # 初始化 Financial Faithful Reward
            financial_faithful_reward = FinancialDeepResearchFaithfulReward(llm=llm)
            logger.info("✅ Financial Faithful Reward 初始化成功")
            
        except Exception as e:
            logger.warning(f"⚠️ RM-Gallery 奖励初始化失败: {e}")
            financial_resolution_reward = None
            financial_faithful_reward = None

    # step1: 数据预处理
    supply_contents = extract_tool_calls(group_messages)
    assistant_contents = extract_assistant_group_messages(group_messages)
    group_extras = [_process_extra_info(extra) for extra in group_extras]

    tool_calls_reference = group_extras[0]["skeleton"]["tool_calls_reference"]
    sample = convert_to_data_samples(group_messages, tool_calls_reference)  # 创建 sample 用于RM-Gallery的reward计算
    # logger.warning(f"sample: {sample.model_dump()}")
    logger.info(f"数据预处理完成: {len(assistant_contents)} 个助手回复")

    # step2: 创建优化的执行器
    executor = HybridRewardExecutor()

    # step3: 注册IO密集型任务（rm_gallery相关）
    if HAS_RM_GALLERY:
        try:
            agent_search_info_reward = create_information_reward()
            action_loop_detection_reward = create_loop_detection_reward()

            executor.add_io_task(
                "search_info_reward", agent_search_info_reward.async_evaluate, sample
            )
            executor.add_io_task(
                "action_loop_reward", action_loop_detection_reward.async_evaluate, sample
            )
            logger.info("✅ rm_gallery 基础奖励已添加（search_info, action_loop）")
        except Exception as e:
            logger.warning(f"⚠️ 添加 rm_gallery 基础奖励失败: {e}")

    # 添加 Financial Resolution Reward (如果初始化成功)
    if financial_resolution_reward is not None:
        executor.add_io_task(
            "financial_resolution_reward", financial_resolution_reward.async_evaluate, sample
        )
        logger.info("✅ Financial Resolution Reward 已添加")
    
    # 添加 Financial Faithful Reward (如果初始化成功)
    if financial_faithful_reward is not None:
        executor.add_io_task(
            "financial_faithful_reward", financial_faithful_reward.async_evaluate, sample
        )
        logger.info("✅ Financial Faithful Reward 已添加")
    
    # 添加 Rubrics Evaluation Reward
    valid_rubrics_count = sum(1 for r in rubrics_list if r is not None)
    logger.info(f"Rubrics任务准备: valid={valid_rubrics_count}, total={len(rubrics_list)}, trajectories={len(trajectories_for_rubrics)}")
    
    if valid_rubrics_count > 0:
        logger.info(f"✅ 启用Rubrics评估（并发），{valid_rubrics_count}/{len(rubrics_list)} 个trajectories有rubrics")
        
        # 初始化 RubricsEvaluator
        try:
            from deep_research.synthesis.core.rubrics_evaluator import RubricsEvaluator
            from omegaconf import OmegaConf
            
            # 从环境变量或默认配置获取
            config_dict = {
                'execution_mode': 'openai_api',
                'openai_api': {
                    'api_key': os.getenv('OPENAI_API_KEY'),
                    'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                    'timeout': 60,
                    'max_retries': 3
                },
                'model': {'name': 'qwen3-max'},
                'complexity_exploration': {
                    'rubrics_evaluation': {
                        'model_name': 'qwen3-max',
                        'temperature': 0.3,
                        'max_new_tokens': 2048
                    }
                }
            }
            rubrics_config = OmegaConf.create(config_dict)
            rubrics_evaluator = RubricsEvaluator(rubrics_config)
            logger.info(f"✓ RubricsEvaluator 初始化成功")
            
            executor.add_io_task(
                "rubrics_reward",
                rubrics_evaluator.batch_evaluate_with_optional_rubrics,
                trajectories_for_rubrics,
                rubrics_list
            )
            logger.info(f"✓ rubrics_reward 任务已添加到 executor")
        except Exception as e:
            logger.error(f"⚠️ 初始化RubricsEvaluator失败: {e}，跳过Rubrics评估", exc_info=True)
    else:
        logger.warning(f"ℹ️ 没有trajectory关联rubrics (valid_rubrics_count={valid_rubrics_count})，跳过Rubrics评估")

    # step5: 并行执行所有任务
    logger.info("开始并行执行所有reward任务...")
    results = await executor.execute_all_parallel()
    timings = executor.get_timings()

    # step6: 统一处理reward结果
    # 处理rm_gallery相关奖励结果
    search_info_reward_list, search_info_details = [], []
    action_loop_reward_list, action_loop_details = [], []
    
    if HAS_RM_GALLERY:
        search_info_reward_list, search_info_details = process_reward_result(
            results.get("search_info_reward")
        )
        action_loop_reward_list, action_loop_details = process_reward_result(
            results.get("action_loop_reward")
        )

    # 处理 financial resolution reward 结果
    financial_resolution_reward_list, financial_resolution_details = process_financial_resolution_reward_result(
        results.get("financial_resolution_reward"))
    
    # 处理 financial faithful reward 结果
    financial_faithful_reward_list, financial_faithful_details = process_reward_result(
        results.get("financial_faithful_reward"))
    
    # 处理 rubrics reward 结果
    rubrics_result = results.get("rubrics_reward", None)
    logger.debug(f"Rubrics结果: type={type(rubrics_result)}, is_tuple={isinstance(rubrics_result, tuple)}, len={len(rubrics_result) if isinstance(rubrics_result, (tuple, list)) else 'N/A'}")
    
    if rubrics_result and isinstance(rubrics_result, tuple) and len(rubrics_result) == 2:
        rubrics_reward_list, rubrics_details_list = rubrics_result
        logger.debug(f"Rubrics解包: rewards={len(rubrics_reward_list)}, details={len(rubrics_details_list)}, detail_types={[type(d).__name__ for d in rubrics_details_list[:3]]}")
    else:
        # 没有rubrics评估结果，使用默认值
        rubrics_reward_list = []
        rubrics_details_list = []
        logger.debug(f"Rubrics结果格式不符，使用默认空列表")
    
    # step7: 校验和填充reward长度
    expected_length = len(assistant_contents)
    
    # 确保所有reward列表长度正确
    if not search_info_reward_list or len(search_info_reward_list) != expected_length:
        search_info_reward_list = [0.0] * expected_length
        search_info_details = [{"note": "rm_gallery_not_available"}] * expected_length
    
    if not action_loop_reward_list or len(action_loop_reward_list) != expected_length:
        action_loop_reward_list = [0.0] * expected_length
        action_loop_details = [{"note": "rm_gallery_not_available"}] * expected_length
    
    if not financial_resolution_reward_list or len(financial_resolution_reward_list) != expected_length:
        logger.warning(f"⚠️ financial_resolution_reward长度不正确 ({len(financial_resolution_reward_list)} != {expected_length})，使用默认值填充")
        financial_resolution_reward_list = [0.0] * expected_length
        if not financial_resolution_details or len(financial_resolution_details) != expected_length:
            financial_resolution_details = [{"note": "financial_resolution_failed_or_disabled"}] * expected_length
    
    if not financial_faithful_reward_list or len(financial_faithful_reward_list) != expected_length:
        logger.warning(f"⚠️ financial_faithful_reward长度不正确 ({len(financial_faithful_reward_list)} != {expected_length})，使用默认值填充")
        financial_faithful_reward_list = [0.0] * expected_length
        if not financial_faithful_details or len(financial_faithful_details) != expected_length:
            financial_faithful_details = [{"note": "financial_faithful_failed_or_disabled"}] * expected_length
    
    if not rubrics_reward_list or len(rubrics_reward_list) != expected_length:
        rubrics_reward_list = [0.0] * expected_length
        rubrics_details_list = [{"note": "no_rubrics"}] * expected_length
    
    # step8: 构建reward字典并验证
    reward_dict = {
        "search_info_reward": search_info_reward_list,
        "action_loop_reward": action_loop_reward_list,
        "financial_resolution_reward": financial_resolution_reward_list,
        "financial_faithful_reward": financial_faithful_reward_list,
        "rubrics_reward": rubrics_reward_list,
    }
    
    for reward_name, reward_list in reward_dict.items():
        assert (
            len(reward_list) == expected_length
        ), f"{reward_name}长度不正确: {len(reward_list)} != {expected_length}"

    # step9: 计算总分
    total_score = (
        torch.tensor(search_info_reward_list)
        + torch.tensor(action_loop_reward_list)
        + torch.tensor(financial_resolution_reward_list)
        + torch.tensor(financial_faithful_reward_list)
        + torch.tensor(rubrics_reward_list)
    ).tolist()

    reward_info = {k: v for k, v in reward_dict.items()}
    total_time = time.perf_counter() - total_start_time

    # step10: 打印性能统计
    logger.info("\n" + "=" * 80)
    logger.info("基于 RM-Gallery 的奖励计算完成 - 性能统计")
    logger.info("=" * 80)
    logger.info(f"总计算时间: {total_time:.4f}秒")
    logger.info(f"各任务详细耗时: {timings}")
    logger.info(f"IO任务数={len(executor.io_tasks)}, CPU任务数={len(executor.cpu_tasks)}")
    logger.info("=" * 80)
    
    for i in range(min(len(assistant_contents), 10)):
        logger.info(f"\n--- 样本 {i} ---")
        logger.info(f"搜索信息奖励: {search_info_reward_list[i]:.4f}")
        logger.info(f"循环检测奖励: {action_loop_reward_list[i]:.4f}")
        if i < len(financial_resolution_reward_list):
            logger.info(f"金融解决率奖励: {financial_resolution_reward_list[i]:.4f}")
        if i < len(financial_faithful_reward_list):
            logger.info(f"金融忠实度奖励: {financial_faithful_reward_list[i]:.4f}")
        if i < len(rubrics_reward_list) and rubrics_reward_list[i] > 0:
            logger.info(f"Rubrics评估奖励: {rubrics_reward_list[i]:.4f}")
        logger.info(f"总分: {total_score[i]:.4f}")
    if len(assistant_contents) > 10:
        logger.info(f"... 还有 {len(assistant_contents) - 10} 个样本")
    logger.info("=" * 80 + "\n")

    # step11: 构建详细信息（如果需要）
    if return_details:
        reward_details = []
        detail_sources = [
            ("search_info_details", search_info_details),
            ("action_loop_details", action_loop_details),
            ("financial_resolution_details", financial_resolution_details),
            ("financial_faithful_details", financial_faithful_details),
            ("rubrics_details", rubrics_details_list),
        ]
        for i in range(expected_length):
            detail = {
                "index": i,
                "total_score": total_score[i],
                "performance_stats": {
                    "total_time": total_time,
                    "individual_timings": timings,
                },
                "note": "使用基于 RM-Gallery 的奖励计算"
            }
            for detail_key, detail_list in detail_sources:
                if i < len(detail_list):
                    raw_detail = detail_list[i]
                    # Debug: 针对 rubrics_details 的详细日志
                    if detail_key == "rubrics_details":
                        logger.debug(f"清理前 rubrics_details[{i}]: type={type(raw_detail).__name__}, is_str={isinstance(raw_detail, str)}, len={len(str(raw_detail)) if raw_detail else 0}")
                    detail[detail_key] = _clean_detail_dict(raw_detail)
            reward_details.append(detail)
        return total_score, reward_info, reward_details
    else:
        return total_score, reward_info

