"""
奖励计算核心服务
封装 compute_score_async，提供高并发异步计算能力
"""
import asyncio
import logging
import random
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from .persistence import SamplePersistence
from .hash_utils import compute_reward_input_hash

logger = logging.getLogger(__name__)


class RewardComputeService:
    """
    奖励计算服务
    - 封装 compute_score_async 函数
    - 提供缓存支持
    - 并发控制
    - 性能监控
    """
    
    def __init__(
        self,
        compute_score_async_fn,
        max_concurrent_groups: int = 10,
        enable_cache: bool = True,
        cache_backend=None,  # Redis或内存缓存
        enable_persistence: bool = True,
        persistence_dir: str = "./data/reward_samples",
    ):
        """
        初始化服务
        
        Args:
            compute_score_async_fn: compute_score_async 函数引用
            max_concurrent_groups: 最大并发group数量
            enable_cache: 是否启用缓存
            cache_backend: 缓存后端（Redis或MemoryCache）
            enable_persistence: 是否启用数据持久化
            persistence_dir: 数据持久化目录
        """
        self.compute_score_async = compute_score_async_fn
        self.max_concurrent_groups = max_concurrent_groups
        self.enable_cache = enable_cache
        self.cache_backend = cache_backend
        
        # 并发控制信号量
        self.semaphore = asyncio.Semaphore(max_concurrent_groups)
        
        # 数据持久化
        self.persistence = SamplePersistence(
            base_dir=persistence_dir,
            enabled=enable_persistence,
        )
        
        # 监控指标
        self.metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_compute_time": 0.0,
            "total_samples_processed": 0,
            "failed_requests": 0,
            "total_response_time": 0.0,
            "max_compute_time": 0.0,
            "min_compute_time": float('inf'),
        }
        
        # 指标更新锁（保证统计的线程安全）
        self.metrics_lock = asyncio.Lock()
        
        logger.info(
            f"RewardComputeService 初始化完成: "
            f"max_concurrent_groups={max_concurrent_groups}, "
            f"enable_cache={enable_cache}, "
            f"enable_persistence={enable_persistence}"
        )
    
    def _generate_cache_key(
        self,
        prompt: str,
        response_list: List[str],
        messages_list: List[List[dict]],
        extra_list: List[dict],
        config: dict
    ) -> str:
        """
        生成缓存key（用于结果缓存）
        
        Args:
            prompt: 问题文本
            response_list: 响应列表
            messages_list: 消息列表
            extra_list: 额外信息列表
            config: 配置参数
            
        Returns:
            缓存key字符串，格式：reward:compute:{hash}
            
        Note:
            使用统一的hash工具函数，确保与 TaskQueue 使用相同的hash算法。
        """
        return compute_reward_input_hash(
            prompt=prompt,
            response_list=response_list,
            messages_list=messages_list,
            extra_list=extra_list,
            config=config,
            length=None,  # 使用完整的64位hash
            prefix="reward:compute",
        )
    
    async def _get_from_cache(self, cache_key: str) -> Optional[dict]:
        """从缓存获取结果"""
        if not self.enable_cache or not self.cache_backend:
            return None
        
        try:
            result = await self.cache_backend.get(cache_key)
            if result:
                async with self.metrics_lock:
                    self.metrics["cache_hits"] += 1
                logger.info(f"缓存命中: {cache_key[:16]}...")
                return result
            else:
                async with self.metrics_lock:
                    self.metrics["cache_misses"] += 1
        except Exception as e:
            logger.warning(f"从缓存获取结果失败: {e}")
        
        return None
    
    async def _set_to_cache(self, cache_key: str, result: dict, ttl: int = 3600):
        """将结果保存到缓存"""
        if not self.enable_cache or not self.cache_backend:
            return
        
        try:
            await self.cache_backend.set(cache_key, result, ttl=ttl)
            logger.debug(f"结果已缓存: {cache_key[:16]}...")
        except Exception as e:
            logger.warning(f"保存到缓存失败: {e}")
    
    async def _log_metrics_summary(
        self, 
        sample_count: int,
        compute_time: float,
        total_response_time: float
    ):
        """
        导出统计数据到日志（在持久化保存后调用）
        
        Args:
            sample_count: 本次请求的样本数
            compute_time: 本次计算耗时
            total_response_time: 本次总响应耗时
        """
        async with self.metrics_lock:
            metrics_snapshot = self.metrics.copy()
        
        # 计算衍生指标
        total_requests = metrics_snapshot["total_requests"]
        cache_hits = metrics_snapshot["cache_hits"]
        cache_misses = metrics_snapshot["cache_misses"]
        failed_requests = metrics_snapshot["failed_requests"]
        
        # 成功请求数 = 总请求 - 失败请求
        success_requests = total_requests - failed_requests
        
        # 缓存命中率
        total_cache_checks = cache_hits + cache_misses
        cache_hit_rate = cache_hits / total_cache_checks if total_cache_checks > 0 else 0.0
        
        # 平均计算耗时（只计算缓存未命中的）
        avg_compute_time = (
            metrics_snapshot["total_compute_time"] / cache_misses 
            if cache_misses > 0 else 0.0
        )
        
        # 平均响应耗时（包含所有请求）
        avg_response_time = (
            metrics_snapshot["total_response_time"] / total_requests 
            if total_requests > 0 else 0.0
        )
        
        # 最小计算耗时（处理初始值）
        min_compute_time = metrics_snapshot["min_compute_time"]
        if min_compute_time == float('inf'):
            min_compute_time = 0.0
        
        # 导出到日志
        logger.info(
            f"📊 统计摘要 [总计] "
            f"请求: {total_requests} (成功: {success_requests}, 失败: {failed_requests}), "
            f"样本: {metrics_snapshot['total_samples_processed']}, "
            f"缓存命中率: {cache_hit_rate:.1%}"
        )
        logger.info(
            f"📊 统计摘要 [性能] "
            f"平均响应: {avg_response_time*1000:.1f}ms, "
            f"平均计算: {avg_compute_time:.2f}s, "
            f"最大计算: {metrics_snapshot['max_compute_time']:.2f}s, "
            f"最小计算: {min_compute_time:.2f}s"
        )
        logger.info(
            f"📊 统计摘要 [本次] "
            f"样本: {sample_count}, "
            f"计算: {compute_time:.2f}s, "
            f"响应: {total_response_time:.2f}s"
        )
    
    async def compute_single_group(
        self,
        prompt: str,
        response_list: List[str],
        messages_list: List[List[dict]],
        extra_list: List[dict],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        计算单个group的奖励
        
        Args:
            prompt: 问题文本
            response_list: 响应列表
            messages_list: 消息列表
            extra_list: 额外信息列表
            config: 配置参数
            
        Returns:
            包含total_scores, reward_info, details的字典
        """
        if config is None:
            config = {}
        
        # 统计：记录每个请求（无论是否缓存命中）
        sample_count = len(response_list)
        request_start_time = time.perf_counter()  # 记录请求开始时间（包含缓存查询）
        
        async with self.metrics_lock:
            self.metrics["total_requests"] += 1
            self.metrics["total_samples_processed"] += sample_count
        
        # 生成缓存key
        cache_key = self._generate_cache_key(prompt, response_list, messages_list, extra_list, config)
        
        # 尝试从缓存获取
        cached_result = await self._get_from_cache(cache_key)
        if cached_result is not None:
            # 更新缓存命中的meta信息
            if isinstance(cached_result, dict) and "meta" in cached_result:
                cached_result["meta"]["cache_hit"] = True
            
            # 统计缓存命中的响应时间
            response_time = time.perf_counter() - request_start_time
            async with self.metrics_lock:
                self.metrics["total_response_time"] += response_time
            
            logger.info(f"缓存命中返回: {sample_count} 个样本, 响应耗时 {response_time*1000:.2f}ms")
            return cached_result
        
        # 随机抖动：在获取semaphore前添加小延迟，避免同时发起大量LLM请求
        jitter_delay = random.uniform(0.05, 0.3)  # 50-300ms随机延迟
        await asyncio.sleep(jitter_delay)
        logger.debug(f"随机抖动延迟: {jitter_delay*1000:.1f}ms，减少并发冲击")
        
        # 记录等待并发槽位的开始时间
        semaphore_wait_start = time.perf_counter()
        
        # 并发控制
        async with self.semaphore:
            # 记录等待时间
            semaphore_wait_time = time.perf_counter() - semaphore_wait_start
            if semaphore_wait_time > 1.0:  # 等待超过1秒才记录
                logger.warning(f"⏱️ 等待并发槽位耗时: {semaphore_wait_time:.2f}s (可能存在队列拥堵)")
            else:
                logger.debug(f"获取并发槽位耗时: {semaphore_wait_time:.3f}s")
            start_time = time.perf_counter()
            compute_start_time = time.perf_counter()  # 记录纯计算开始时间
            
            # 获取超时配置（默认5分钟）
            timeout_seconds = config.get("timeout", 300)
            
            try:
                # 调用 compute_score_async
                # 将config中的所有参数传递给compute_score_async
                # 从config中提取参数，避免重复传递
                config_params = config.copy()
                return_details = config_params.pop("return_details", True)
                return_statistics = config_params.pop("return_statistics", True)
                config_params.pop("timeout", None)  # 移除timeout，避免传递给compute_score_async
                
                logger.info(f"🚀 开始计算reward (样本数={sample_count}, 超时={timeout_seconds}s)")
                
                # 使用 asyncio.wait_for 添加超时保护
                result = await asyncio.wait_for(
                    self.compute_score_async(
                        prompt=prompt,
                        group_responses=response_list,
                        group_messages=messages_list,
                        group_extras=extra_list,
                        return_details=return_details,
                        return_statistics=return_statistics,
                        **config_params,  # 传递其余config参数
                    ),
                    timeout=timeout_seconds
                )
                
                # 解析结果
                if config.get("return_details", True):
                    total_scores, reward_info, details = result
                else:
                    total_scores, reward_info = result
                    details = None
                
                # 构建返回结果
                compute_time = time.perf_counter() - start_time
                result_dict = {
                    "total_scores": total_scores,
                    "reward_info": reward_info,
                    "details": details,
                    "meta": {
                        "processing_time": compute_time,
                        "sample_count": len(response_list),
                        "timestamp": datetime.now().isoformat(),
                        "cache_hit": False,
                    }
                }
                
                # 保存到缓存
                await self._set_to_cache(cache_key, result_dict, ttl=config.get("cache_ttl", 3600))
                
                # 持久化样本数据
                await self.persistence.save_batch_samples(
                    prompt=prompt,
                    response_list=response_list,
                    messages_list=messages_list,
                    extra_list=extra_list,
                    total_scores=total_scores,
                    reward_info=reward_info,
                    details=details,
                )
                
                # 计算总响应时间（包含所有处理时间）
                total_response_time = time.perf_counter() - request_start_time
                
                # 更新统计指标（请求数和样本数已在函数开头统计）
                async with self.metrics_lock:
                    self.metrics["total_compute_time"] += compute_time
                    self.metrics["total_response_time"] += total_response_time
                    # 更新最大/最小计算时间
                    if compute_time > self.metrics["max_compute_time"]:
                        self.metrics["max_compute_time"] = compute_time
                    if compute_time < self.metrics["min_compute_time"]:
                        self.metrics["min_compute_time"] = compute_time
                
                # 导出统计数据到日志（在持久化后）
                await self._log_metrics_summary(
                    sample_count=sample_count,
                    compute_time=compute_time,
                    total_response_time=total_response_time
                )
                
                logger.info(
                    f"✅ 计算完成: {sample_count} 个样本, "
                    f"计算耗时 {compute_time:.2f}s, 总响应耗时 {total_response_time:.2f}s"
                )
                
                return result_dict
                
            except asyncio.TimeoutError:
                # 超时异常单独处理
                compute_time = time.perf_counter() - compute_start_time
                
                async with self.metrics_lock:
                    self.metrics["failed_requests"] += 1
                
                logger.error(
                    f"⏰ 计算超时: 样本数={sample_count}, "
                    f"超时阈值={timeout_seconds}s, 已耗时={compute_time:.2f}s"
                )
                logger.error(
                    f"📊 性能分析: "
                    f"等待槽位={semaphore_wait_time:.2f}s, "
                    f"实际计算={compute_time:.2f}s, "
                    f"总耗时={(time.perf_counter() - request_start_time):.2f}s"
                )
                
                raise TimeoutError(
                    f"Reward计算超时（超过{timeout_seconds}秒），样本数={sample_count}，"
                    f"可能原因：1) LLM API响应慢 2) 网络延迟 3) API限流排队"
                )
                
            except Exception as e:
                # 其他异常处理
                compute_time = time.perf_counter() - compute_start_time
                
                # 统计失败请求
                async with self.metrics_lock:
                    self.metrics["failed_requests"] += 1
                
                # 检测是否为API限流错误（通过HTTP 429状态码）
                is_rate_limit = False
                if hasattr(e, 'status_code') and e.status_code == 429:
                    # HTTP 429 Too Many Requests
                    is_rate_limit = True
                elif hasattr(e, 'response') and hasattr(e.response, 'status_code') and e.response.status_code == 429:
                    # httpx.HTTPStatusError or requests.HTTPError
                    is_rate_limit = True
                elif type(e).__name__ in ['RateLimitError', 'RateLimitException']:
                    # SDK特定的限流异常
                    is_rate_limit = True
                
                if is_rate_limit:
                    logger.warning(
                        f"⚠️ API限流检测 (HTTP 429): {e}\n"
                        f"  样本数: {sample_count}\n"
                        f"  已耗时: {compute_time:.2f}s\n"
                        f"  建议: 1) 降低并发数 2) 增加重试延迟 3) 使用API配额管理"
                    )
                else:
                    logger.error(
                        f"❌ 计算奖励时发生错误: {e}\n"
                        f"  样本数: {sample_count}\n"
                        f"  已耗时: {compute_time:.2f}s",
                        exc_info=True
                    )
                
                raise
    
    async def compute_batch_groups(
        self,
        groups: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        批量计算多个group的奖励
        
        Args:
            groups: group列表，每个group包含 prompt, response_list, messresponse wages_list, extra_list
            config: 配置参数
            
        Returns:
            结果列表
        """
        if config is None:
            config = {}
        
        logger.info(f"开始批量计算 {len(groups)} 个groups")
        
        # 创建所有任务
        tasks = []
        for i, group in enumerate(groups):
            task = self.compute_single_group(
                prompt=group["prompt"],
                response_list=group["response_list"],
                messages_list=group["messages_list"],
                extra_list=group["extra_list"],
                config=config,
            )
            tasks.append(task)
        
        # 并发执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果和异常
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Group {i} 计算失败: {result}")
                # 返回错误结果
                processed_results.append({
                    "error": str(result),
                    "group_index": i,
                    "status": "failed",
                })
            else:
                result["group_index"] = i
                result["status"] = "success"
                processed_results.append(result)
        
        return processed_results
    
    async def get_metrics(self) -> Dict[str, Any]:
        """获取服务指标"""
        async with self.metrics_lock:
            metrics = self.metrics.copy()
        
        # 计算衍生指标
        total_requests = metrics["total_requests"]
        cache_hits = metrics["cache_hits"]
        cache_misses = metrics["cache_misses"]
        failed_requests = metrics["failed_requests"]
        
        # 成功请求数
        success_requests = total_requests - failed_requests
        metrics["success_requests"] = success_requests
        metrics["success_rate"] = success_requests / total_requests if total_requests > 0 else 0.0
        
        # 平均响应时间（所有请求）
        if total_requests > 0:
            metrics["avg_response_time"] = metrics["total_response_time"] / total_requests
            metrics["avg_samples_per_request"] = metrics["total_samples_processed"] / total_requests
        
        # 平均计算时间（仅缓存未命中的）
        if cache_misses > 0:
            metrics["avg_compute_time"] = metrics["total_compute_time"] / cache_misses
        
        # 缓存命中率
        total_cache_checks = cache_hits + cache_misses
        if total_cache_checks > 0:
            metrics["cache_hit_rate"] = cache_hits / total_cache_checks
        
        # 处理最小计算时间的初始值
        if metrics["min_compute_time"] == float('inf'):
            metrics["min_compute_time"] = 0.0
        
        # 添加持久化统计
        metrics["persistence"] = self.persistence.get_stats()
        
        return metrics
    
    async def reset_metrics(self):
        """重置指标（带锁保护）"""
        async with self.metrics_lock:
            self.metrics = {
                "total_requests": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "total_compute_time": 0.0,
                "total_samples_processed": 0,
                "failed_requests": 0,
                "total_response_time": 0.0,
                "max_compute_time": 0.0,
                "min_compute_time": float('inf'),
            }
        logger.info("指标已重置")

