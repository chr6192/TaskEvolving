"""
奖励计算核心服务
封装 compute_score_async，提供高并发异步计算能力
"""
import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from .persistence import SamplePersistence

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
        }
        
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
        生成缓存key
        使用输入数据的hash作为key
        """
        # 构建用于hash的数据结构
        cache_data = {
            "prompt": prompt,
            "response_list": response_list,
            "messages_list": messages_list,
            # extra_list可能包含大量数据，只取关键字段
            "extras_keys": [
                {
                    "skeleton_id": e.get("skeleton_id", ""),
                    "trajectory_id": e.get("trajectory_id", ""),
                }
                for e in extra_list
            ],
            "config": config,
        }
        
        # 序列化并计算hash
        cache_str = json.dumps(cache_data, sort_keys=True, ensure_ascii=False)
        cache_key = hashlib.sha256(cache_str.encode()).hexdigest()
        return f"reward:compute:{cache_key}"
    
    async def _get_from_cache(self, cache_key: str) -> Optional[dict]:
        """从缓存获取结果"""
        if not self.enable_cache or not self.cache_backend:
            return None
        
        try:
            result = await self.cache_backend.get(cache_key)
            if result:
                self.metrics["cache_hits"] += 1
                logger.info(f"缓存命中: {cache_key[:16]}...")
                return result
            else:
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
        
        # 生成缓存key
        cache_key = self._generate_cache_key(prompt, response_list, messages_list, extra_list, config)
        
        # 尝试从缓存获取
        cached_result = await self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        # 并发控制
        async with self.semaphore:
            start_time = time.perf_counter()
            
            try:
                # 调用 compute_score_async
                # 将config中的所有参数传递给compute_score_async
                # 从config中提取参数，避免重复传递
                config_params = config.copy()
                return_details = config_params.pop("return_details", True)
                return_statistics = config_params.pop("return_statistics", True)
                
                result = await self.compute_score_async(
                    prompt=prompt,
                    group_responses=response_list,
                    group_messages=messages_list,
                    group_extras=extra_list,
                    return_details=return_details,
                    return_statistics=return_statistics,
                    **config_params,  # 传递其余config参数
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
                
                # 更新指标
                self.metrics["total_requests"] += 1
                self.metrics["total_compute_time"] += compute_time
                self.metrics["total_samples_processed"] += len(response_list)
                
                logger.info(
                    f"计算完成: {len(response_list)} 个样本, "
                    f"耗时 {compute_time:.2f}s"
                )
                
                return result_dict
                
            except Exception as e:
                logger.error(f"计算奖励时发生错误: {e}", exc_info=True)
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
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取服务指标"""
        metrics = self.metrics.copy()
        
        # 计算平均值
        if metrics["total_requests"] > 0:
            metrics["avg_compute_time"] = metrics["total_compute_time"] / metrics["total_requests"]
            metrics["avg_samples_per_request"] = metrics["total_samples_processed"] / metrics["total_requests"]
            
            if metrics["cache_hits"] + metrics["cache_misses"] > 0:
                metrics["cache_hit_rate"] = metrics["cache_hits"] / (metrics["cache_hits"] + metrics["cache_misses"])
        
        # 添加持久化统计
        metrics["persistence"] = self.persistence.get_stats()
        
        return metrics
    
    def reset_metrics(self):
        """重置指标"""
        self.metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_compute_time": 0.0,
            "total_samples_processed": 0,
        }
        logger.info("指标已重置")

