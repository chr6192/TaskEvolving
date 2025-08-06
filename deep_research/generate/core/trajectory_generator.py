"""
Trajectory生成器 - 基于skeleton生成完整对话轨迹
"""

import asyncio
import json
import logging
import uuid
from typing import List, Optional, Dict, Any

from omegaconf import DictConfig

from .data_structures import SkeletonData, TrajectoryData, QueryData, ToolCall, ToolResponse
from .skeleton_processor import SkeletonProcessor
from .agent_framework import AgentFramework

logger = logging.getLogger(__name__)


class TrajectoryGenerator:
    """
    基于skeleton生成完整的agent trajectory
    
    核心功能：
    1. 解析skeleton中的依赖关系和占位符
    2. 使用AgentFramework执行多轮对话
    3. 生成多个candidate trajectories
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.generation_config = config.generation
        
        # 核心组件
        self.skeleton_processor: Optional[SkeletonProcessor] = None
        self.agent_framework: Optional[AgentFramework] = None
        
        # 生成参数
        self.n_samples_per_skeleton = self.generation_config.n_samples_per_skeleton
        self.rollout_n = self.generation_config.rollout_n
        self.temperature = self.generation_config.temperature
        self.top_p = self.generation_config.top_p
        self.max_new_tokens = self.generation_config.max_new_tokens
        
        # 状态标记
        self._initialized = False
        
    async def initialize(self) -> None:
        """初始化trajectory生成器"""
        if self._initialized:
            return
            
        logger.info("初始化Trajectory生成器...")
        
        # 初始化skeleton处理器
        self.skeleton_processor = SkeletonProcessor(self.config)
        await self.skeleton_processor.initialize()
        
        # 初始化agent框架
        self.agent_framework = AgentFramework(self.config)
        await self.agent_framework.initialize()
        
        self._initialized = True
        logger.info("Trajectory生成器初始化完成")
        
    async def cleanup(self) -> None:
        """清理资源"""
        logger.info("清理Trajectory生成器资源...")
        
        cleanup_tasks = []
        if self.skeleton_processor:
            cleanup_tasks.append(self.skeleton_processor.cleanup())
        if self.agent_framework:
            cleanup_tasks.append(self.agent_framework.cleanup())
            
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
        logger.info("Trajectory生成器资源清理完成")
        
    async def generate_from_skeleton(self, skeleton: SkeletonData, n_samples: int) -> List[TrajectoryData]:
        """
        基于skeleton生成多个candidate trajectories
        
        Args:
            skeleton: 输入skeleton
            n_samples: 要生成的样本数量
            
        Returns:
            生成的trajectory列表
        """
        if not self._initialized:
            raise RuntimeError("Trajectory生成器未初始化")
            
        logger.info(f"基于skeleton {skeleton.skeleton_id} 生成 {n_samples} 个trajectory")
        
        try:
            # 处理skeleton，生成初始消息
            initial_messages = await self.skeleton_processor.process_skeleton(skeleton)
            
            # 并发生成多个trajectories
            tasks = []
            for i in range(n_samples):
                task = self._generate_single_trajectory(skeleton, initial_messages, i)
                tasks.append(task)
                
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            trajectories = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"生成trajectory {i} 失败: {result}")
                elif result is not None:
                    trajectories.append(result)
                    
            logger.info(f"成功生成 {len(trajectories)}/{n_samples} 个trajectory")
            return trajectories
            
        except Exception as e:
            logger.error(f"基于skeleton生成trajectory失败: {e}")
            return []
            
    async def generate_from_query(self, query: QueryData, n_samples: int) -> List[TrajectoryData]:
        """
        直接从query生成trajectories（不使用skeleton）
        
        Args:
            query: 输入查询
            n_samples: 要生成的样本数量
            
        Returns:
            生成的trajectory列表
        """
        if not self._initialized:
            raise RuntimeError("Trajectory生成器未初始化")
            
        logger.info(f"直接从查询生成 {n_samples} 个trajectory")
        
        try:
            # 构建初始消息
            initial_messages = [{"role": "user", "content": query.content}]
            
            # 并发生成多个trajectories
            tasks = []
            for i in range(n_samples):
                task = self._generate_single_trajectory_from_query(query, initial_messages, i)
                tasks.append(task)
                
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            trajectories = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"生成trajectory {i} 失败: {result}")
                elif result is not None:
                    trajectories.append(result)
                    
            logger.info(f"成功生成 {len(trajectories)}/{n_samples} 个trajectory")
            return trajectories
            
        except Exception as e:
            logger.error(f"从查询生成trajectory失败: {e}")
            return []
            
    async def _generate_single_trajectory(
        self, 
        skeleton: SkeletonData, 
        initial_messages: List[Dict[str, str]], 
        sample_index: int
    ) -> Optional[TrajectoryData]:
        """
        生成单个trajectory
        
        Args:
            skeleton: skeleton数据
            initial_messages: 初始消息
            sample_index: 样本索引
            
        Returns:
            生成的trajectory，失败时返回None
        """
        try:
            logger.debug(f"生成trajectory样本 {sample_index}")
            
            # 使用agent框架执行对话
            execution_result = await self.agent_framework.execute_trajectory(
                initial_messages=initial_messages.copy(),
                skeleton=skeleton,
                generation_params={
                    "temperature": self._get_varied_temperature(sample_index),
                    "top_p": self.top_p,
                    "max_new_tokens": self.max_new_tokens
                }
            )
            
            if not execution_result:
                logger.warning(f"Trajectory执行失败: sample {sample_index}")
                return None
                
            # 构建TrajectoryData
            trajectory = TrajectoryData(
                trajectory_id=str(uuid.uuid4()),
                skeleton_id=skeleton.skeleton_id,
                query_id=skeleton.query_id,
                messages=execution_result["messages"],
                tool_calls=execution_result.get("tool_calls", []),
                tool_responses=execution_result.get("tool_responses", []),
                metadata={
                    "sample_index": sample_index,
                    "generation_params": execution_result.get("generation_params", {}),
                    "skeleton_id": skeleton.skeleton_id
                }
            )
            
            return trajectory
            
        except Exception as e:
            logger.error(f"生成单个trajectory失败 (sample {sample_index}): {e}")
            return None
            
    async def _generate_single_trajectory_from_query(
        self, 
        query: QueryData, 
        initial_messages: List[Dict[str, str]], 
        sample_index: int
    ) -> Optional[TrajectoryData]:
        """
        从query直接生成单个trajectory
        
        Args:
            query: 查询数据
            initial_messages: 初始消息
            sample_index: 样本索引
            
        Returns:
            生成的trajectory，失败时返回None
        """
        try:
            logger.debug(f"从查询生成trajectory样本 {sample_index}")
            
            # 使用agent框架执行对话
            execution_result = await self.agent_framework.execute_trajectory(
                initial_messages=initial_messages.copy(),
                skeleton=None,  # 无skeleton
                generation_params={
                    "temperature": self._get_varied_temperature(sample_index),
                    "top_p": self.top_p,
                    "max_new_tokens": self.max_new_tokens
                }
            )
            
            if not execution_result:
                logger.warning(f"Trajectory执行失败: sample {sample_index}")
                return None
                
            # 构建TrajectoryData
            trajectory = TrajectoryData(
                trajectory_id=str(uuid.uuid4()),
                skeleton_id=None,
                query_id=query.query_id,
                messages=execution_result["messages"],
                tool_calls=execution_result.get("tool_calls", []),
                tool_responses=execution_result.get("tool_responses", []),
                metadata={
                    "sample_index": sample_index,
                    "generation_params": execution_result.get("generation_params", {}),
                    "direct_from_query": True
                }
            )
            
            return trajectory
            
        except Exception as e:
            logger.error(f"从查询生成单个trajectory失败 (sample {sample_index}): {e}")
            return None
            
    def _get_varied_temperature(self, sample_index: int) -> float:
        """
        为不同样本获取变化的temperature参数，增加多样性
        
        Args:
            sample_index: 样本索引
            
        Returns:
            调整后的temperature值
        """
        base_temp = self.temperature
        
        # 为不同样本增加温度变化，增加多样性
        if sample_index == 0:
            return base_temp  # 第一个样本使用基础温度
        elif sample_index % 3 == 1:
            return min(1.0, base_temp + 0.1)  # 稍微增加创造性
        elif sample_index % 3 == 2:
            return max(0.1, base_temp - 0.1)  # 稍微增加确定性
        else:
            return base_temp 