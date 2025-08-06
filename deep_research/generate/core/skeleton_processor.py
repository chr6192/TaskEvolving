"""
Skeleton处理器 - 处理skeleton中的依赖关系和占位符
"""

import logging
import re
from typing import List, Dict, Any, Optional

from omegaconf import DictConfig

from .data_structures import SkeletonData, StepData, ToolCall

logger = logging.getLogger(__name__)


class SkeletonProcessor:
    """
    处理skeleton中的依赖关系和缺省值
    
    主要功能：
    1. 解析skeleton为初始对话消息
    2. 解析工具调用间的依赖关系
    3. 根据上下文填充占位符
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        
        # 依赖关系解析正则表达式
        self.dependency_pattern = re.compile(r'「无法明确」依赖step(\d+)中的(\w+)返回的(.+)')
        
        # 状态标记
        self._initialized = False
        
    async def initialize(self) -> None:
        """初始化skeleton处理器"""
        if self._initialized:
            return
            
        logger.info("初始化Skeleton处理器...")
        self._initialized = True
        logger.info("Skeleton处理器初始化完成")
        
    async def cleanup(self) -> None:
        """清理资源"""
        logger.info("清理Skeleton处理器资源...")
        logger.info("Skeleton处理器资源清理完成")
        
    async def process_skeleton(self, skeleton: SkeletonData) -> List[Dict[str, str]]:
        """
        处理skeleton，生成初始对话消息
        
        Args:
            skeleton: 输入skeleton数据
            
        Returns:
            初始对话消息列表
        """
        if not self._initialized:
            raise RuntimeError("Skeleton处理器未初始化")
            
        logger.debug(f"处理skeleton {skeleton.skeleton_id}")
        
        try:
            # 构建初始用户消息
            initial_messages = [
                {"role": "user", "content": skeleton.query_content}
            ]
            
            # 如果skeleton包含分析思路，可以作为系统消息或者引导性回复
            if skeleton.analysis_logic and len(skeleton.analysis_logic.strip()) > 0:
                # 构建包含分析思路的系统提示或引导
                analysis_prompt = self._build_analysis_guidance(skeleton)
                if analysis_prompt:
                    # 添加系统消息指导分析过程
                    system_message = {
                        "role": "system", 
                        "content": analysis_prompt
                    }
                    initial_messages.insert(0, system_message)
            
            logger.debug(f"生成了 {len(initial_messages)} 个初始消息")
            return initial_messages
            
        except Exception as e:
            logger.error(f"处理skeleton失败: {e}")
            # 返回基础的用户消息
            return [{"role": "user", "content": skeleton.query_content}]
            
    def resolve_dependencies(self, tool_calls: List[ToolCall]) -> List[ToolCall]:
        """
        解析工具调用间的依赖关系
        
        Args:
            tool_calls: 工具调用列表
            
        Returns:
            解析依赖关系后的工具调用列表
        """
        logger.debug(f"解析 {len(tool_calls)} 个工具调用的依赖关系")
        
        resolved_calls = []
        dependency_map = {}
        
        for tool_call in tool_calls:
            resolved_call = self._resolve_single_tool_call(tool_call, dependency_map)
            resolved_calls.append(resolved_call)
            
            # 记录工具调用结果供后续依赖使用
            dependency_map[tool_call.tool_call_id] = {
                "name": tool_call.name,
                "parameters": tool_call.parameters
            }
            
        return resolved_calls
        
    def fill_placeholders(self, tool_calls: List[ToolCall], context: Dict[str, Any]) -> List[ToolCall]:
        """
        根据上下文填充占位符
        
        Args:
            tool_calls: 工具调用列表
            context: 上下文数据，包含之前步骤的结果
            
        Returns:
            填充占位符后的工具调用列表
        """
        logger.debug(f"为 {len(tool_calls)} 个工具调用填充占位符")
        
        filled_calls = []
        
        for tool_call in tool_calls:
            filled_call = self._fill_single_tool_call_placeholders(tool_call, context)
            filled_calls.append(filled_call)
            
        return filled_calls
        
    def _build_analysis_guidance(self, skeleton: SkeletonData) -> Optional[str]:
        """
        构建分析指导消息
        
        Args:
            skeleton: skeleton数据
            
        Returns:
            分析指导消息，如果不需要则返回None
        """
        try:
            guidance_parts = []
            
            # 添加分析思路
            if skeleton.analysis_logic:
                guidance_parts.append(f"分析思路：{skeleton.analysis_logic}")
            
            # 添加步骤信息
            if skeleton.multi_step and skeleton.tool_results:
                guidance_parts.append(f"这是一个需要 {len(skeleton.tool_results)} 个步骤的多步分析任务。")
                
                # 添加步骤概述
                step_summary = []
                for step_data in skeleton.tool_results:
                    tools_in_step = [tc.name for tc in step_data.tool_calls]
                    step_summary.append(f"步骤{step_data.step}: 使用工具 {', '.join(tools_in_step)}")
                
                guidance_parts.append("分析步骤：\n" + "\n".join(step_summary))
            
            if guidance_parts:
                guidance = "请按照以下分析框架进行回答：\n\n" + "\n\n".join(guidance_parts)
                guidance += "\n\n请严格按照这个分析框架进行工具调用和分析。"
                return guidance
                
            return None
            
        except Exception as e:
            logger.error(f"构建分析指导失败: {e}")
            return None
            
    def _resolve_single_tool_call(self, tool_call: ToolCall, dependency_map: Dict[str, Any]) -> ToolCall:
        """
        解析单个工具调用的依赖关系
        
        Args:
            tool_call: 工具调用
            dependency_map: 依赖映射表
            
        Returns:
            解析依赖关系后的工具调用
        """
        # 创建工具调用副本
        resolved_call = ToolCall(
            name=tool_call.name,
            parameters=tool_call.parameters.copy(),
            tool_call_id=tool_call.tool_call_id
        )
        
        # 检查参数中的依赖关系
        for param_key, param_value in resolved_call.parameters.items():
            if isinstance(param_value, str):
                # 检查是否包含依赖标记
                match = self.dependency_pattern.search(param_value)
                if match:
                    step_num = match.group(1)
                    tool_name = match.group(2)
                    expected_content = match.group(3)
                    
                    logger.debug(f"发现依赖: step{step_num} -> {tool_name}.{expected_content}")
                    
                    # 这里可以根据实际需求解析依赖关系
                    # 当前保持占位符，在实际执行时动态解析
                    resolved_call.parameters[param_key] = f"[DEPENDENCY] step{step_num}.{tool_name}.{expected_content}"
        
        return resolved_call
        
    def _fill_single_tool_call_placeholders(self, tool_call: ToolCall, context: Dict[str, Any]) -> ToolCall:
        """
        为单个工具调用填充占位符
        
        Args:
            tool_call: 工具调用
            context: 上下文数据
            
        Returns:
            填充占位符后的工具调用
        """
        # 创建工具调用副本
        filled_call = ToolCall(
            name=tool_call.name,
            parameters=tool_call.parameters.copy(),
            tool_call_id=tool_call.tool_call_id
        )
        
        # 填充参数中的占位符
        for param_key, param_value in filled_call.parameters.items():
            if isinstance(param_value, str) and param_value.startswith("[DEPENDENCY]"):
                # 从上下文中查找对应的值
                filled_value = self._resolve_dependency_from_context(param_value, context)
                filled_call.parameters[param_key] = filled_value
                
        return filled_call
        
    def _resolve_dependency_from_context(self, dependency_placeholder: str, context: Dict[str, Any]) -> str:
        """
        从上下文中解析依赖关系
        
        Args:
            dependency_placeholder: 依赖占位符
            context: 上下文数据
            
        Returns:
            解析后的实际值
        """
        try:
            # 解析依赖占位符
            # 格式: [DEPENDENCY] step{num}.{tool_name}.{content}
            parts = dependency_placeholder.replace("[DEPENDENCY] ", "").split(".")
            if len(parts) >= 3:
                step_info = parts[0]  # step{num}
                tool_name = parts[1]
                content_key = parts[2]
                
                # 从上下文中查找对应的值
                step_results = context.get("step_results", {})
                if step_info in step_results:
                    step_result = step_results[step_info]
                    if tool_name in step_result:
                        tool_result = step_result[tool_name]
                        return tool_result.get(content_key, dependency_placeholder)
            
            # 如果无法解析，返回原占位符
            return dependency_placeholder
            
        except Exception as e:
            logger.error(f"解析依赖关系失败: {e}")
            return dependency_placeholder 