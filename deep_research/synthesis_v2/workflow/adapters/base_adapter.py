"""
Workflow适配器基类
用于将外部Agent框架适配到标准接口
"""
from abc import ABC, abstractmethod
from typing import Any
from ...core.data_structures import WorkflowInput, WorkflowOutput


class WorkflowAdapter(ABC):
    """Workflow适配器抽象基类"""
    
    @abstractmethod
    def adapt_input(self, input: WorkflowInput) -> Any:
        """
        将标准输入转换为框架特定格式
        
        Args:
            input: 标准WorkflowInput
            
        Returns:
            框架特定的输入格式
        """
        pass
    
    @abstractmethod
    def adapt_output(self, framework_output: Any) -> WorkflowOutput:
        """
        将框架输出转换为标准格式
        
        Args:
            framework_output: 框架特定的输出
            
        Returns:
            标准WorkflowOutput
        """
        pass
