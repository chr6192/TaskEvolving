"""
智能体框架 - 管理工具调用和对话的执行
"""

import asyncio
import importlib
import logging
from typing import List, Dict, Any, Optional

from omegaconf import DictConfig

from .data_structures import SkeletonData, ToolCall, ToolResponse
from ..utils.tool_manager import ToolManager
from ..utils.conversation_manager import RayConversationManager

logger = logging.getLogger(__name__)


class AgentFramework:
    """
    管理工具调用和对话的智能体框架
    
    核心功能：
    1. 执行完整的对话轨迹
    2. 处理工具调用和响应
    3. 与现有的chat_scheduler集成
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.rollout_config = config.rollout
        
        # 核心组件
        self.tool_manager: Optional[ToolManager] = None
        self.conversation_manager: Optional[RayConversationManager] = None
        
        # Chat scheduler相关配置
        self.completion_callback_class = config.completion_callback
        self.tool_config_path = config.tool_config_path
        
        # 生成参数
        self.max_turns = self.rollout_config.multi_turn.max_turns
        self.multi_turn_format = self.rollout_config.multi_turn.format
        
        # 状态标记
        self._initialized = False
        
    async def initialize(self) -> None:
        """初始化agent框架"""
        if self._initialized:
            return
            
        logger.info("初始化Agent框架...")
        
        # 初始化工具管理器
        self.tool_manager = ToolManager(self.config)
        await self.tool_manager.initialize()
        
        # 根据执行模式初始化对话管理器
        execution_mode = self.config.get('execution_mode', 'local')
        if execution_mode == 'openai_api':
            logger.info("使用 OpenAI API 对话管理器")
            from ..utils.conversation_manager import OaiConversationManager
            self.conversation_manager = OaiConversationManager(self.config)
        else:
            logger.info("使用本地 Ray+LLM 对话管理器")
            from ..utils.conversation_manager import RayConversationManager
            self.conversation_manager = RayConversationManager(self.config)
        
        await self.conversation_manager.initialize()
        
        self._initialized = True
        logger.info("Agent框架初始化完成")
        
    async def cleanup(self) -> None:
        """清理资源"""
        logger.info("清理Agent框架资源...")
        
        cleanup_tasks = []
        if self.tool_manager:
            cleanup_tasks.append(self.tool_manager.cleanup())
        if self.conversation_manager:
            cleanup_tasks.append(self.conversation_manager.cleanup())
            
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
        logger.info("Agent框架资源清理完成")
        
    async def execute_trajectory(
        self, 
        initial_messages: List[Dict[str, str]], 
        skeleton: Optional[SkeletonData] = None,
        generation_params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        执行完整的对话轨迹
        
        Args:
            initial_messages: 初始对话消息
            skeleton: 可选的skeleton指导（如果有的话）
            generation_params: 生成参数
            
        Returns:
            执行结果，包含messages、tool_calls、tool_responses等
            失败时返回None
        """
        if not self._initialized:
            raise RuntimeError("Agent框架未初始化")
            
        logger.debug("开始执行对话轨迹")
        
        try:
            # 使用对话管理器执行多轮对话
            execution_result = await self.conversation_manager.execute_conversation(
                initial_messages=initial_messages,
                skeleton=skeleton,
                generation_params=generation_params or {}
            )
            
            if not execution_result:
                logger.warning("对话执行失败")
                return None
                
            # 提取工具调用和响应
            tool_calls, tool_responses = await self._extract_tool_interactions(
                execution_result["messages"]
            )
            
            # 构建完整的执行结果
            result = {
                "messages": execution_result["messages"],
                "tool_calls": tool_calls,
                "tool_responses": tool_responses,
                "generation_params": execution_result.get("generation_params", {}),
                "metadata": execution_result.get("metadata", {})
            }
            
            logger.debug(f"对话轨迹执行完成，共 {len(result['messages'])} 轮对话")
            return result
            
        except Exception as e:
            logger.error(f"执行对话轨迹失败: {e}")
            return None
            
    async def handle_tool_calls(self, tool_calls: List[ToolCall]) -> List[ToolResponse]:
        """
        处理工具调用
        
        Args:
            tool_calls: 工具调用列表
            
        Returns:
            工具响应列表
        """
        if not self.tool_manager:
            raise RuntimeError("工具管理器未初始化")
            
        logger.debug(f"处理 {len(tool_calls)} 个工具调用")
        
        try:
            tool_responses = await self.tool_manager.execute_tool_calls(tool_calls)
            logger.debug(f"工具调用处理完成，返回 {len(tool_responses)} 个响应")
            return tool_responses
            
        except Exception as e:
            logger.error(f"处理工具调用失败: {e}")
            # 返回错误响应
            error_responses = []
            for tool_call in tool_calls:
                error_response = ToolResponse(
                    tool_call_id=tool_call.tool_call_id,
                    content=f"工具调用失败: {str(e)}",
                    success=False,
                    error_message=str(e)
                )
                error_responses.append(error_response)
            return error_responses
            
    async def _extract_tool_interactions(
        self, 
        messages: List[Dict[str, str]]
    ) -> tuple[List[ToolCall], List[ToolResponse]]:
        """
        从对话消息中提取工具调用和响应
        
        Args:
            messages: 对话消息列表
            
        Returns:
            (tool_calls, tool_responses) 元组
        """
        tool_calls = []
        tool_responses = []
        
        try:
            for message in messages:
                role = message.get("role", "")
                
                # 提取assistant消息中的工具调用
                if role == "assistant" and "tool_calls" in message:
                    message_tool_calls = message["tool_calls"]
                    for tc in message_tool_calls:
                        tool_call = ToolCall(
                            name=tc["function"]["name"],
                            parameters=tc["function"]["arguments"],
                            tool_call_id=tc["id"]
                        )
                        tool_calls.append(tool_call)
                
                # 提取tool消息中的响应
                elif role == "tool":
                    tool_response = ToolResponse(
                        tool_call_id=message.get("tool_call_id", ""),
                        content=message.get("content", ""),
                        success=not message.get("content", "").startswith("Error:")
                    )
                    tool_responses.append(tool_response)
                    
        except Exception as e:
            logger.error(f"提取工具交互失败: {e}")
            
        return tool_calls, tool_responses
        
    def _create_completion_callback(self):
        """创建completion callback实例"""
        try:
            # 动态导入completion callback类
            module_path, class_name = self.completion_callback_class.rsplit('.', 1)
            module = importlib.import_module(module_path)
            callback_class = getattr(module, class_name)
            
            # 创建实例（需要config和scheduler参数）
            # 这里需要根据实际的callback类接口调整
            callback = callback_class(self.config, None)  # scheduler会在运行时设置
            
            return callback
            
        except Exception as e:
            logger.error(f"创建completion callback失败: {e}")
            return None 