"""synthesis_v2 的 Query-only 工具调用回调。"""

import asyncio
import logging
from typing import Any, Dict, List, TYPE_CHECKING

import numpy as np
from omegaconf import DictConfig
from openai.types.chat.chat_completion import ChatCompletion

from deep_research.llm_utils import (
    is_invalid_tool_calls_completion,
    parse_tool_call_arguments,
    persist_invalid_tool_calls_response,
)
from verl.protocol import DataProto
from verl.workers.rollout.chat_scheduler import (
    CompletionCallback,
    initialize_tools_from_config,
)

if TYPE_CHECKING:
    from verl.workers.rollout.chat_scheduler import ChatCompletionScheduler

logger = logging.getLogger(__name__)


class DrSynthesisCompletionCallback(CompletionCallback):
    """执行工具调用，并在完成后继续提交多轮对话。"""

    def __init__(
        self,
        config: DictConfig,
        scheduler: "ChatCompletionScheduler",
    ):
        self.config = config
        self.scheduler = scheduler
        multi_turn = config.actor_rollout_ref.rollout.multi_turn
        self.max_turns = multi_turn.max_turns
        tool_config_path = multi_turn.tool_config_path
        tool_list = (
            initialize_tools_from_config(tool_config_path)
            if tool_config_path
            else []
        )
        self.tools = {tool.name: tool for tool in tool_list}
        self._tool_schemas = [
            tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)
            for tool in tool_list
        ]

    @property
    def tool_schemas(self):
        return self._tool_schemas

    async def __call__(
        self,
        messages: List[Dict[str, Any]],
        completions: ChatCompletion,
        info: Dict[str, Any],
    ):
        finish_reason = completions.choices[0].finish_reason
        if is_invalid_tool_calls_completion(completions):
            persist_invalid_tool_calls_response(
                source="dr_synthesis_completion_callback",
                completion_id=completions.id,
                turn=len(messages) + 1,
                completion=completions.model_dump(),
                messages=messages,
            )
            logger.warning("模型返回空壳 tool_calls，结束当前对话")
            return

        message = completions.choices[0].message.model_dump(
            exclude_unset=True,
            exclude_none=True,
        )
        message.setdefault("content", "")
        messages.append(message)

        if self.max_turns and len(messages) >= self.max_turns:
            return
        if finish_reason != "tool_calls":
            return

        tool_calls = completions.choices[0].message.tool_calls or []
        tool_responses = await asyncio.gather(
            *(self._call_tool(tool_call) for tool_call in tool_calls),
            return_exceptions=True,
        )
        if any(isinstance(response, Exception) for response in tool_responses):
            logger.error("工具调用出现未处理异常，结束当前对话")
            return
        messages.extend(tool_responses)
        self.scheduler.submit_chat_completions(
            messages=messages,
            request_id=completions.id,
            info=info,
        )

    async def _call_tool(self, tool_call) -> Dict[str, str]:
        tool_name = tool_call.function.name
        tool_args = parse_tool_call_arguments(tool_call.function.arguments)
        if not tool_name or tool_name not in self.tools:
            error = f"未知的工具: {tool_name!r}，可用工具: {list(self.tools)}"
            return {
                "role": "tool",
                "content": f"Error: {error}",
                "tool_call_id": tool_call.id,
            }

        tool = self.tools[tool_name]
        instance_id = await tool.create()
        try:
            tool_response, _, _ = await tool.execute(instance_id, tool_args)
        except Exception as exc:
            logger.exception("工具 %s 执行失败", tool_name)
            return {
                "role": "tool",
                "content": f"Error: {exc}",
                "tool_call_id": tool_call.id,
            }
        finally:
            await tool.release(instance_id)

        return {
            "role": "tool",
            "content": tool_response,
            "tool_call_id": tool_call.id,
        }

    def postprocess(
        self,
        batch: DataProto,
        batch_conversations: List[List[Dict[str, Any]]],
        n: int,
    ) -> DataProto:
        num_turns = np.array(
            [len(conversation) for conversation in batch_conversations],
            dtype=np.int32,
        )
        messages = np.array(batch_conversations, dtype=object)
        return DataProto(
            batch=None,
            non_tensor_batch={"__num_turns__": num_turns, "messages": messages},
        )
