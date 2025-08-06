"""
工具管理器 - 管理和执行工具调用
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from uuid import uuid4

from omegaconf import DictConfig
from verl.tools.base_tool import initialize_tools_from_config

from ..core.data_structures import ToolCall, ToolResponse

logger = logging.getLogger(__name__)


class ToolManager:
    """
    管理工具的创建、执行和释放，复用deep_research/tool/dr_tool.py中的工具实现
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.tool_config_path = config.tool_config_path
        
        # 工具实例（延迟初始化）
        self.tools: Dict[str, Any] = {}
        self.tool_schemas: List[Dict[str, Any]] = []
        
        # 状态标记
        self._initialized = False
        
    async def initialize(self) -> None:
        """初始化工具管理器"""
        if self._initialized:
            return
            
        logger.info("初始化工具管理器...")
        
        # 初始化工具
        await self._initialize_tools()
        
        self._initialized = True
        logger.info(f"工具管理器初始化完成，加载了 {len(self.tools)} 个工具")
        
    async def cleanup(self) -> None:
        """清理资源"""
        logger.info("清理工具管理器资源...")
        
        # 释放所有工具实例
        for tool_name, tool in self.tools.items():
            try:
                if hasattr(tool, 'cleanup'):
                    await tool.cleanup()
            except Exception as e:
                logger.error(f"清理工具 {tool_name} 失败: {e}")
                
        logger.info("工具管理器资源清理完成")
        
    async def execute_tool_calls(self, tool_calls: List[ToolCall]) -> List[ToolResponse]:
        """
        执行工具调用列表
        
        Args:
            tool_calls: 工具调用列表
            
        Returns:
            工具响应列表
        """
        if not self._initialized:
            raise RuntimeError("工具管理器未初始化")
            
        logger.debug(f"执行 {len(tool_calls)} 个工具调用")
        
        # 并发执行所有工具调用
        tasks = [self.execute_single_tool_call(tool_call) for tool_call in tool_calls]
        tool_responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        results = []
        for i, response in enumerate(tool_responses):
            if isinstance(response, Exception):
                logger.error(f"工具调用 {i} 执行失败: {response}")
                error_response = ToolResponse(
                    tool_call_id=tool_calls[i].tool_call_id,
                    content=f"工具执行失败: {str(response)}",
                    success=False,
                    error_message=str(response)
                )
                results.append(error_response)
            else:
                results.append(response)
                
        logger.debug(f"工具调用执行完成，成功 {sum(1 for r in results if r.success)}/{len(results)} 个")
        return results
        
    async def execute_single_tool_call(self, tool_call: ToolCall) -> ToolResponse:
        """
        执行单个工具调用
        
        Args:
            tool_call: 工具调用
            
        Returns:
            工具响应
        """
        tool_name = tool_call.name
        
        if tool_name not in self.tools:
            error_msg = f"未知的工具: {tool_name}"
            logger.error(error_msg)
            return ToolResponse(
                tool_call_id=tool_call.tool_call_id,
                content=error_msg,
                success=False,
                error_message=error_msg
            )
            
        try:
            tool = self.tools[tool_name]
            
            # 创建工具实例
            instance_id = await tool.create()
            
            try:
                # 执行工具
                tool_response, tool_reward_score, tool_metrics = await tool.execute(
                    instance_id=instance_id,
                    parameters=tool_call.parameters
                )
                
                return ToolResponse(
                    tool_call_id=tool_call.tool_call_id,
                    content=tool_response,
                    reward_score=tool_reward_score,
                    metrics=tool_metrics,
                    success=True
                )
                
            finally:
                # 释放工具实例
                await tool.release(instance_id)
                
        except Exception as e:
            error_msg = f"执行工具 {tool_name} 失败: {str(e)}"
            logger.error(error_msg)
            return ToolResponse(
                tool_call_id=tool_call.tool_call_id,
                content=error_msg,
                success=False,
                error_message=str(e),
                metrics={"error": str(e)}
            )
            
    async def _initialize_tools(self) -> None:
        """初始化工具实例"""
        try:
            if not self.tool_config_path:
                logger.warning("未指定工具配置路径，跳过工具初始化")
                return
                
            # 使用verl的工具初始化方法
            tool_dict, tool_schemas = initialize_tools_from_config(self.tool_config_path)
            
            self.tools = tool_dict
            self.tool_schemas = tool_schemas
            
            logger.info(f"成功初始化 {len(self.tools)} 个工具: {list(self.tools.keys())}")
            
        except Exception as e:
            logger.error(f"工具初始化失败: {e}")
            # 创建空的工具字典以防止后续错误
            self.tools = {}
            self.tool_schemas = []
            
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """获取工具模式定义"""
        return self.tool_schemas.copy()
        
    def get_available_tools(self) -> List[str]:
        """获取可用工具列表"""
        return list(self.tools.keys())
        
    def has_tool(self, tool_name: str) -> bool:
        """检查工具是否可用"""
        return tool_name in self.tools