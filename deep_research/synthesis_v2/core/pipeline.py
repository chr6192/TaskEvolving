"""
单条样本Pipeline
workflow执行 + evaluate
"""
import logging
from typing import Optional
from datetime import datetime

from .data_structures import WorkflowInput, PipelineResult
from .concurrency import ConcurrencyManager
from ..workflow.base import WorkflowBase
from ..evaluate.base import EvaluatorBase

logger = logging.getLogger(__name__)


class SamplePipeline:
    """单条样本处理Pipeline（workflow + evaluate）"""
    
    def __init__(
        self,
        workflow: WorkflowBase,
        evaluator: Optional[EvaluatorBase] = None,
        concurrency_manager: Optional[ConcurrencyManager] = None
    ):
        """
        Args:
            workflow: workflow实例
            evaluator: 评估器实例（可选）
            concurrency_manager: 并发管理器（可选，默认使用单例）
        """
        self.workflow = workflow
        self.evaluator = evaluator
        self.concurrency = concurrency_manager or ConcurrencyManager()
    
    async def process(self, input: WorkflowInput) -> PipelineResult:
        """
        处理单条样本：workflow执行 + evaluate
        
        Args:
            input: workflow输入
            
        Returns:
            pipeline结果
        """
        try:
            # 1. 执行workflow（受信号量控制）
            workflow_output = await self.concurrency.submit(
                self.workflow.execute(input)
            )
            
            # 2. 评估（如果有evaluator）
            # 注意：openjudge已有自己的并发控制，这里不需要额外控制
            evaluation = None
            if self.evaluator and workflow_output.success:
                evaluation = await self.evaluator.evaluate(
                    workflow_output, 
                    input.metadata
                )
            
            return PipelineResult(
                input=input,
                output=workflow_output,
                evaluation=evaluation,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Pipeline处理失败: {e}", exc_info=True)
            
            # 返回失败的结果
            from .data_structures import WorkflowOutput
            return PipelineResult(
                input=input,
                output=WorkflowOutput(
                    messages=input.messages,
                    success=False,
                    error=str(e)
                ),
                evaluation=None,
                timestamp=datetime.now()
            )
