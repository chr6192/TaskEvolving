"""
原生Skeleton+Trajectory Workflow
将现有的skeleton_generator和trajectory_generator封装为标准workflow
"""
import logging
import uuid
from typing import Optional
from omegaconf import DictConfig

from ..base import WorkflowBase
from ...core.data_structures import WorkflowInput, WorkflowOutput
from deep_research.synthesis.core.skeleton_generator import SkeletonGenerator
from deep_research.synthesis.core.trajectory_generator import TrajectoryGenerator
from deep_research.synthesis.core.data_structures import QueryData

logger = logging.getLogger(__name__)


class SkeletonTrajectoryWorkflow(WorkflowBase):
    """将现有的skeleton+trajectory封装为标准workflow"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.skeleton_generator: Optional[SkeletonGenerator] = None
        self.trajectory_generator: Optional[TrajectoryGenerator] = None
        self._initialized = False
    
    @property
    def name(self) -> str:
        return "skeleton_trajectory_workflow"
    
    async def initialize(self) -> None:
        """初始化workflow"""
        if self._initialized:
            return
        
        self.skeleton_generator = SkeletonGenerator(self.config)
        self.trajectory_generator = TrajectoryGenerator(self.config)
        
        await self.skeleton_generator.initialize()
        await self.trajectory_generator.initialize()
        
        self._initialized = True
        logger.info("SkeletonTrajectoryWorkflow初始化完成")
    
    async def execute(self, input: WorkflowInput) -> WorkflowOutput:
        """
        执行完整的skeleton→trajectory流程
        
        Args:
            input: 标准输入（messages格式）
            
        Returns:
            标准输出
        """
        try:
            # 1. 从messages提取query
            query = self._extract_query_from_messages(input.messages)
            query_data = QueryData(
                query_id=str(uuid.uuid4()), 
                content=query, 
                metadata=input.metadata
            )
            
            # 2. 生成skeleton
            skeleton = await self.skeleton_generator.generate_skeleton(query_data)
            
            if not skeleton:
                return WorkflowOutput(
                    messages=input.messages,
                    success=False,
                    error="skeleton generation failed"
                )
            
            # 3. 生成trajectory（单个样本）
            trajectories = await self.trajectory_generator.generate_trajectories_from_skeleton(
                skeleton, 
                n_samples=1
            )
            
            if not trajectories:
                return WorkflowOutput(
                    messages=input.messages,
                    success=False,
                    error="trajectory generation failed"
                )
            
            trajectory = trajectories[0]
            
            # 4. 转换为标准输出
            return WorkflowOutput(
                messages=trajectory.messages,
                tool_calls=[tc.to_dict() for tc in trajectory.tool_calls],
                tool_responses=[tr.to_dict() for tr in trajectory.tool_responses],
                final_response=trajectory.get_final_response(),
                metadata={
                    'skeleton_id': skeleton.skeleton_id,
                    'trajectory_id': trajectory.trajectory_id,
                    'query_id': query_data.query_id,
                    'query': query,
                    **trajectory.metadata
                },
                success=True
            )
            
        except Exception as e:
            logger.error(f"Workflow执行失败: {e}", exc_info=True)
            return WorkflowOutput(
                messages=input.messages,
                success=False,
                error=str(e)
            )
    
    def _extract_query_from_messages(self, messages: list) -> str:
        """从messages中提取用户查询"""
        # 从后往前找第一条user消息
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                return msg.get('content', '')
        
        # 如果没有找到，返回空字符串
        return ''
    
    async def cleanup(self) -> None:
        """清理资源"""
        if self.skeleton_generator:
            await self.skeleton_generator.cleanup()
        if self.trajectory_generator:
            await self.trajectory_generator.cleanup()
        
        logger.info("SkeletonTrajectoryWorkflow清理完成")
