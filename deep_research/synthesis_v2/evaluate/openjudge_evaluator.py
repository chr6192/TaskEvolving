"""
OpenJudge评估器适配器
复用现有的ORMEvaluator，适配到新的接口
"""
import logging
import uuid
from typing import Dict, Any
from omegaconf import DictConfig

from .base import EvaluatorBase
from ..core.data_structures import WorkflowOutput, EvaluationResult
from deep_research.synthesis.core.orm_evaluator import ORMEvaluator
from deep_research.synthesis.core.data_structures import TrajectoryData

logger = logging.getLogger(__name__)


class OpenJudgeEvaluator(EvaluatorBase):
    """OpenJudge评估器适配器"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.orm_evaluator = ORMEvaluator(config)
        self._initialized = False
    
    async def initialize(self) -> None:
        """初始化评估器"""
        if self._initialized:
            return
        
        await self.orm_evaluator.initialize()
        self._initialized = True
        logger.info("OpenJudgeEvaluator初始化完成")
    
    async def evaluate(
        self, 
        workflow_output: WorkflowOutput, 
        context: Dict[str, Any]
    ) -> EvaluationResult:
        """
        评估workflow输出
        
        Args:
            workflow_output: workflow输出
            context: 上下文信息（包含query、rubrics等）
            
        Returns:
            评估结果
        """
        # 转换为TrajectoryData格式（复用现有evaluator）
        trajectory = self._convert_to_trajectory(workflow_output, context)
        
        # 调用现有的评估器
        eval_results = await self.orm_evaluator.batch_evaluate([trajectory])
        
        if not eval_results:
            return EvaluationResult(
                trajectory_id=trajectory.trajectory_id,
                scores={},
                passed=False
            )
        
        eval_result = eval_results[0]
        
        # 转换为新的EvaluationResult格式
        return EvaluationResult(
            trajectory_id=eval_result.trajectory_id,
            scores=eval_result.orm_scores,
            details=eval_result.orm_details,
            passed=eval_result.passed_quality_check
        )
    
    def _convert_to_trajectory(
        self, 
        workflow_output: WorkflowOutput, 
        context: Dict[str, Any]
    ) -> TrajectoryData:
        """将WorkflowOutput转换为TrajectoryData"""
        from deep_research.synthesis.core.data_structures import ToolCall, ToolResponse

        # evaluator 内部统一合并上下文：
        # - `context` 通常来自 WorkflowInput.metadata
        # - `workflow_output.metadata` 通常包含 workflow 运行时产出的关键字段（如 query、rubrics 等）
        # 这里以 workflow_output.metadata 为准（覆盖同名 key），保证评估输入完整且一致。
        merged_context = {**(context or {}), **(workflow_output.metadata or {})}
        
        trajectory_id = str(uuid.uuid4())
        
        # 转换tool_calls
        tool_calls = []
        for tc in workflow_output.tool_calls:
            tool_calls.append(ToolCall(
                name=tc.get('name', ''),
                parameters=tc.get('parameters', {}),
                tool_call_id=tc.get('tool_call_id')
            ))
        
        # 转换tool_responses
        tool_responses = []
        for tr in workflow_output.tool_responses:
            tool_responses.append(ToolResponse(
                tool_call_id=tr.get('tool_call_id', ''),
                content=tr.get('content', ''),
                success=tr.get('success', True)
            ))
        
        # 创建TrajectoryData
        trajectory = TrajectoryData(
            trajectory_id=trajectory_id,
            skeleton_id=merged_context.get('skeleton_id'),
            query_id=merged_context.get('query_id', str(uuid.uuid4())),
            query=merged_context.get('query'),
            messages=workflow_output.messages,
            tool_calls=tool_calls,
            tool_responses=tool_responses,
            metadata={
                **(context or {}),
                **(workflow_output.metadata or {}),
            }
        )
        
        return trajectory
    
    async def cleanup(self) -> None:
        """清理资源"""
        if self.orm_evaluator:
            await self.orm_evaluator.cleanup()
        logger.info("OpenJudgeEvaluator清理完成")
