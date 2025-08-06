"""
数据合成专用的CompletionCallback
继承并扩展DrCompletionCallback，支持skeleton引导的对话生成
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig
from openai.types.chat.chat_completion import ChatCompletion
from tensordict import TensorDict
import numpy as np

from deep_research.dr_completion_callback import DrCompletionCallback
from verl.workers.rollout.chat_scheduler import CompletionCallback, initialize_tools_from_config
from verl.protocol import DataProto
from deep_research.generate.core.data_structures import SkeletonData

logger = logging.getLogger(__name__)


class DrSynthesisCompletionCallback(DrCompletionCallback):
    """
    数据合成专用的CompletionCallback
    
    在DrCompletionCallback基础上，增加对skeleton引导的支持
    """
    def __init__(self, config: DictConfig, scheduler: "ChatCompletionScheduler", skeleton: Optional[SkeletonData] = None):
        self.config = config
        self.scheduler = scheduler

        # Initialize tools from config file
        self.max_turns = config.actor_rollout_ref.rollout.multi_turn.max_turns
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self._tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        logger.info(f"Initialized tools: {self.tools}")
        
        # skeleton引导信息
        self.skeleton = skeleton
        self.skeleton_step_index = 0  # 当前skeleton步骤索引
        self.skeleton_guided = skeleton is not None
        
        # 扩展的生成控制
        self.max_skeleton_steps = config.skeleton.max_steps if hasattr(config, 'skeleton') else 5
        
    def set_skeleton(self, skeleton: Optional[SkeletonData]) -> None:
        """设置skeleton引导信息"""
        self.skeleton = skeleton
        self.skeleton_step_index = 0
        self.skeleton_guided = skeleton is not None
        
    async def __call__(self, messages: List[Dict[str, str]], completions: ChatCompletion, info: Dict[str, Any]):
        """
        处理completion回调，支持skeleton引导
        """
        message = completions.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)
        if "content" not in message:
            message["content"] = ""
        messages.append(message)
        finish_reason = completions.choices[0].finish_reason

        # STEP 0: check if we reach max turns
        if self.max_turns and len(messages) >= self.max_turns:
            logger.info(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] Reach max turns, done!\n{message}")
            return

        # STEP 1: skeleton guided logic
        if self.skeleton_guided and finish_reason != "tool_calls":
            # 如果有skeleton指导且模型没有调用工具，检查是否需要引导
            should_guide = await self._should_guide_with_skeleton(messages, completions)
            if should_guide:
                guided_messages = await self._guide_with_skeleton(messages, completions, info)
                if guided_messages:
                    return  # 已经重新提交了请求

        # STEP 2: check if the model called tools
        if finish_reason != "tool_calls":
            logger.info(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] No tool called, done!\n{message}")
            return

        # STEP 3: call tools (复用父类逻辑)
        tool_calls = completions.choices[0].message.tool_calls
        logger.info(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] Call {len(tool_calls)} tools")
        tasks = []
        for tool_call in tool_calls:
            tasks.append(self._call_tool(tool_call))
        tool_responses = await asyncio.gather(*tasks)
        if any(isinstance(item, Exception) for item in tool_responses):
            logger.error(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] Error when calling tools, done!")
            return
        messages.extend(tool_responses)

        logger.info(f"[id={completions.id},turn={len(messages)}, tool_calls: {tool_calls}, tool_responses: {tool_responses}]\nmessage: {message}")

        # STEP 4: update skeleton step index
        if self.skeleton_guided:
            self.skeleton_step_index += 1

        # STEP 5: resubmit completion request with tool responses
        self.scheduler.submit_chat_completions(messages=messages, request_id=completions.id, info=info)

    async def _should_guide_with_skeleton(self, messages: List[Dict[str, str]], completions: ChatCompletion) -> bool:
        """
        判断是否需要使用skeleton进行引导
        
        Args:
            messages: 当前对话消息
            completions: completion结果
            
        Returns:
            是否需要引导
        """
        if not self.skeleton or not self.skeleton.tool_results:
            return False
            
        # 如果还有未执行的skeleton步骤
        if self.skeleton_step_index < len(self.skeleton.tool_results):
            # 检查当前是否应该执行工具调用
            current_step = self.skeleton.tool_results[self.skeleton_step_index]
            if current_step.tool_calls:
                # 检查是否是多轮对话中的合适时机
                user_messages = [msg for msg in messages if msg.get("role") == "user"]
                assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
                
                # 简单的引导策略：如果用户有提问但assistant还没有调用相应工具
                return len(user_messages) > len([msg for msg in messages if msg.get("role") == "tool"])
                
        return False
        
    async def _guide_with_skeleton(self, messages: List[Dict[str, str]], completions: ChatCompletion, info: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
        """
        使用skeleton引导生成工具调用
        
        Args:
            messages: 当前对话消息
            completions: completion结果
            info: 额外信息
            
        Returns:
            引导后的消息列表，如果无需引导返回None
        """
        try:
            if self.skeleton_step_index >= len(self.skeleton.tool_results):
                return None
                
            current_step = self.skeleton.tool_results[self.skeleton_step_index]
            
            # 构建引导性的工具调用消息
            guided_tool_calls = []
            for tool_call in current_step.tool_calls:
                # 解析占位符
                resolved_parameters = self._resolve_parameters(tool_call.parameters, messages)
                
                guided_tool_call = {
                    "id": tool_call.tool_call_id or f"call_{len(guided_tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(resolved_parameters, ensure_ascii=False)
                    }
                }
                guided_tool_calls.append(guided_tool_call)
            
            # 修改最后的assistant消息以包含工具调用
            if messages and messages[-1].get("role") == "assistant":
                messages[-1]["tool_calls"] = guided_tool_calls
                
                # 执行工具调用
                tasks = []
                for tool_call_data in guided_tool_calls:
                    # 构建工具调用对象
                    class MockToolCall:
                        def __init__(self, data):
                            self.id = data["id"]
                            self.function = type('function', (), {
                                'name': data["function"]["name"],
                                'arguments': data["function"]["arguments"]
                            })()
                    
                    mock_tool_call = MockToolCall(tool_call_data)
                    tasks.append(self._call_tool(mock_tool_call))
                    
                tool_responses = await asyncio.gather(*tasks)
                if not any(isinstance(item, Exception) for item in tool_responses):
                    messages.extend(tool_responses)
                    
                    # 重新提交请求
                    self.scheduler.submit_chat_completions(messages=messages, request_id=completions.id, info=info)
                    return messages
                    
        except Exception as e:
            logger.error(f"Skeleton引导失败: {e}")
            
        return None
        
    def _resolve_parameters(self, parameters: Dict[str, Any], messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        解析工具参数中的占位符
        
        Args:
            parameters: 原始参数
            messages: 对话消息用于提取上下文
            
        Returns:
            解析后的参数
        """
        resolved_params = parameters.copy()
        
        # 提取用户查询作为默认query
        user_messages = [msg["content"] for msg in messages if msg.get("role") == "user"]
        if user_messages:
            latest_user_query = user_messages[-1]
            
            # 替换常见占位符
            for key, value in resolved_params.items():
                if isinstance(value, str):
                    if "{{query}}" in value:
                        resolved_params[key] = value.replace("{{query}}", latest_user_query)
                    elif "{{user_query}}" in value:
                        resolved_params[key] = value.replace("{{user_query}}", latest_user_query)
                    elif value == "{{query}}":
                        resolved_params[key] = latest_user_query
                        
        return resolved_params
        
    def get_skeleton_progress(self) -> Dict[str, Any]:
        """获取skeleton执行进度信息"""
        if not self.skeleton_guided:
            return {"guided": False}
            
        return {
            "guided": True,
            "skeleton_id": self.skeleton.skeleton_id,
            "current_step": self.skeleton_step_index,
            "total_steps": len(self.skeleton.tool_results) if self.skeleton else 0,
            "progress": self.skeleton_step_index / max(1, len(self.skeleton.tool_results)) if self.skeleton else 0.0
        }

    def postprocess(self, batch: DataProto, batch_conversations: List[List[Dict[str, str]]], n: int) -> DataProto:
        # batch = TensorDict()
        num_turns = np.array([len(conversation) for conversation in batch_conversations], dtype=np.int32)
        messages = np.array(batch_conversations, dtype=object)
        return DataProto(batch=None, non_tensor_batch={"__num_turns__": num_turns, "messages": messages})