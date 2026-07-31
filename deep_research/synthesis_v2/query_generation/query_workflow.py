"""
QueryGenerator作为特殊的workflow
"""
import logging
from typing import Optional
from omegaconf import DictConfig

from ..workflow.base import WorkflowBase
from ..core.data_structures import WorkflowInput, WorkflowOutput
from .query_generator import QueryGenerator

logger = logging.getLogger(__name__)


class QueryGeneratorWorkflow(WorkflowBase):
    """Query生成workflow"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.query_generator: Optional[QueryGenerator] = None
        self._initialized = False
    
    @property
    def name(self) -> str:
        return "query_generator_workflow"
    
    async def initialize(self) -> None:
        """初始化workflow"""
        if self._initialized:
            return
        
        self.query_generator = QueryGenerator(self.config)
        await self.query_generator.initialize()
        
        self._initialized = True
        logger.info("QueryGeneratorWorkflow初始化完成")
    
    async def execute(self, input: WorkflowInput) -> WorkflowOutput:
        """
        生成queries
        
        Args:
            input: 标准输入（metadata中包含num_queries参数）
            
        Returns:
            标准输出（metadata中包含生成的queries）
        """
        try:
            # 从metadata获取生成参数
            num_queries = input.metadata.get('num_queries', 1)
            
            # 生成queries
            queries = await self.query_generator.generate_queries(num_queries)
            
            if not queries:
                return WorkflowOutput(
                    messages=input.messages,
                    success=False,
                    error="query generation failed"
                )
            
            # 转换为标准输出
            # 每个query作为一条assistant消息
            messages = [
                {"role": "assistant", "content": q.content} 
                for q in queries
            ]
            
            return WorkflowOutput(
                messages=messages,
                metadata={
                    'queries': [q.to_dict() for q in queries],
                    'num_generated': len(queries)
                },
                success=True
            )
            
        except Exception as e:
            logger.error(f"QueryGeneratorWorkflow执行失败: {e}", exc_info=True)
            return WorkflowOutput(
                messages=input.messages,
                success=False,
                error=str(e)
            )
    
    async def cleanup(self) -> None:
        """清理资源"""
        if self.query_generator:
            await self.query_generator.cleanup()
        
        logger.info("QueryGeneratorWorkflow清理完成")
