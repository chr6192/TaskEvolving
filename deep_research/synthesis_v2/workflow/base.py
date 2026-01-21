"""
Workflow抽象基类
"""
from abc import ABC, abstractmethod
from typing import Optional
from ..core.data_structures import WorkflowInput, WorkflowOutput


class WorkflowBase(ABC):
    """Agent Workflow抽象基类"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """初始化workflow资源"""
        pass
    
    @abstractmethod
    async def execute(self, input: WorkflowInput) -> WorkflowOutput:
        """
        执行workflow
        
        Args:
            input: 标准输入
            
        Returns:
            标准输出
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """清理资源"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Workflow名称"""
        pass
