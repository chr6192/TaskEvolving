"""
Evaluator抽象基类
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from ..core.data_structures import WorkflowOutput, EvaluationResult


class EvaluatorBase(ABC):
    """评估器抽象基类"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """初始化评估器"""
        pass
    
    @abstractmethod
    async def evaluate(
        self, 
        workflow_output: WorkflowOutput, 
        context: Dict[str, Any]
    ) -> EvaluationResult:
        """
        评估workflow输出
        
        Args:
            workflow_output: workflow输出
            context: 上下文信息（如query、rubrics等）
            
        Returns:
            评估结果
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """清理资源"""
        pass
