"""Query-only 智能体执行框架。"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig

from deep_research.llm_utils import parse_tool_call_arguments
from .data_structures import ToolCall, ToolResponse

logger = logging.getLogger(__name__)


class AgentFramework:
    """管理 Query → Trajectory 的多轮对话和工具交互。"""

    def __init__(self, config: DictConfig):
        self.config = config
        self.conversation_manager = None
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        from .conversation_manager import OaiConversationManager

        self.conversation_manager = OaiConversationManager(self.config)
        await self.conversation_manager.initialize()
        self._initialized = True

    async def cleanup(self) -> None:
        if self.conversation_manager:
            await asyncio.gather(
                self.conversation_manager.cleanup(),
                return_exceptions=True,
            )

    async def execute_trajectory(
        self,
        initial_messages: List[Dict[str, Any]],
        generation_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self._initialized or self.conversation_manager is None:
            raise RuntimeError("AgentFramework未初始化")

        try:
            execution_result = await self.conversation_manager.execute_conversation(
                initial_messages=initial_messages,
                generation_params=generation_params or {},
            )
            if not execution_result:
                return None
            tool_calls, tool_responses = self._extract_tool_interactions(
                execution_result["messages"]
            )
            return {
                "messages": execution_result["messages"],
                "tool_calls": tool_calls,
                "tool_responses": tool_responses,
                "generation_params": execution_result.get("generation_params", {}),
                "metadata": execution_result.get("metadata", {}),
            }
        except Exception:
            logger.exception("执行 Query trajectory 失败")
            return None

    @staticmethod
    def _extract_tool_interactions(
        messages: List[Dict[str, Any]],
    ) -> tuple[List[ToolCall], List[ToolResponse]]:
        tool_calls: List[ToolCall] = []
        tool_responses: List[ToolResponse] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role == "assistant":
                for raw_call in message.get("tool_calls") or []:
                    function = raw_call.get("function") or {}
                    if not function.get("name") or not raw_call.get("id"):
                        continue
                    tool_calls.append(
                        ToolCall(
                            name=function["name"],
                            parameters=parse_tool_call_arguments(
                                function.get("arguments")
                            ),
                            tool_call_id=raw_call["id"],
                        )
                    )
            elif role == "tool":
                content = str(message.get("content", ""))
                tool_responses.append(
                    ToolResponse(
                        tool_call_id=str(message.get("tool_call_id", "")),
                        content=content,
                        success=not content.startswith("Error:"),
                    )
                )
        return tool_calls, tool_responses
