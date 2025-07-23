import asyncio
import heapq
import logging
import time
from typing import Any, Dict, List
from uuid import uuid4

from omegaconf import DictConfig
from verl.workers.rollout.chat_scheduler import ChatCompletionScheduler
from verl.protocol import DataProto

logger = logging.getLogger(__file__)


class BfclChatCompletionScheduler(ChatCompletionScheduler):
    """
    BFCL专用的ChatCompletionScheduler，支持根据batch中的tools_kwargs动态获取tool_schemas。
    继承自原有的ChatCompletionScheduler，保持对verl核心代码的无侵入性。
    """
    
    def __init__(self, config: DictConfig, server_addresses: List[str], max_cache_size: int = 10000):
        super().__init__(config, server_addresses, max_cache_size)

    async def _submit_chat_completions_and_callback(
        self,
        messages: List[Dict[str, str]],
        request_id: str,
        info: Dict[str, Any],
    ):
        """重写_submit_chat_completions_and_callback方法，支持动态tool_schemas"""
        """Submit chat completion request, wait request finish and do callback."""
        if request_id:
            request_id = request_id.removeprefix("chatcmpl-")
            assert request_id in self.request_id_to_address
            address = self.request_id_to_address.pop(request_id)
        else:
            address = self.weighted_addresses[0][1]
            self.weighted_addresses[0][0] += 1
            heapq.heapreplace(self.weighted_addresses, self.weighted_addresses[0])

        # use new request_id to avoid duplicate request_id problem
        request_id = uuid4().hex
        self.request_id_to_address[request_id] = address

        completions, exception = None, None
        try:
            # 获取当前请求对应的tool_schemas
            tools_kwargs = info.get("tools_kwargs", None)
            if hasattr(self.completion_callback, 'get_tool_schemas_for_request'):
                # 如果有get_tool_schemas_for_request方法，使用它来动态获取tool schemas
                tools = self.completion_callback.get_tool_schemas_for_request(tools_kwargs)
            else:
                # 否则使用默认的tool_schemas属性
                tools = self.completion_callback.tool_schemas

            tools_kwargs_len = len(tools_kwargs.keys()) if tools_kwargs is not None else 0
            print(f"[BfclChatCompletionScheduler] tools length: {len(tools)}, tool_schemas length: {len(self.completion_callback.tool_schemas)}, tools_kwargs length: {tools_kwargs_len}")
            
            # NOTE: OpenAI client uses httpx, seems to have performance issue in high concurrency requests.
            completions = await self._chat_completions_aiohttp(
                address,
                messages=messages,
                tools=tools,
                extra_body=self.completion_callback.extra_body,
                extra_headers={"x-request-id": request_id},
                **info["__sampling_params__"],
            )
        except Exception as e:
            # Let user handle the exception
            exception = e

        info["__depth__"] -= 1

        if exception is not None:
            logger.exception(f"chat completion failed with exception: {exception}")
        else:
            try:
                await self.completion_callback(messages, completions, info)
            except Exception as e:
                logger.exception(f"completion callback failed with exception: {e}")

        # No more ongoing completion requests
        if info["__depth__"] == 0:
            info["__done__"].set()

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """重写generate_sequences方法，支持动态传递tools_kwargs"""
        t_start = time.time()
        kwargs = dict(
            model=self.model_name,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            kwargs["top_p"] = self.config.val_kwargs.top_p
            kwargs["temperature"] = self.config.val_kwargs.temperature

        print(f"[BfclChatCompletionScheduler] generate_sequences sampling params: {kwargs}")

        # NOTE: For multi-turn rollout, repeat raw_prompt n times and process each prompt independently,
        # validation dataset has already been repeated in `PPOTrainer._validate`.
        n = 1 if batch.meta_info.get("validate", False) else self.config.n
        tasks, batch_conversations = [], [None] * len(batch) * n
        for batch_index, conversation in enumerate(batch.non_tensor_batch["raw_prompt"].repeat(n, axis=0)):
            # raw_prompt: [{"role": "user", "content": ""}, ["role": "assistant", "content"], ...]
            batch_conversations[batch_index] = conversation.tolist()

            # 计算原始batch中的索引，用于获取对应的tools_kwargs
            original_batch_index = batch_index // n

            # 准备额外的信息，包括tools_kwargs（如果存在）
            extra_info = {}
            assert "tools_kwargs" in batch.non_tensor_batch, f"tools_kwargs is not found for index {batch.non_tensor_batch}"
            if "tools_kwargs" in batch.non_tensor_batch:
                extra_info["tools_kwargs"] = batch.non_tensor_batch["tools_kwargs"][original_batch_index]
                extra_info["original_test_entry"] = batch.non_tensor_batch["original_test_entry"][original_batch_index]

            tasks.append(
                asyncio.create_task(
                    self._submit_chat_completions_semaphore(
                        messages=batch_conversations[batch_index],
                        request_id=None,
                        sampling_params=kwargs,
                        extra_info=extra_info,
                    )
                )
            )

        await asyncio.gather(*tasks)
        output_batch = self.completion_callback.postprocess(batch, batch_conversations, n=n)
        output_batch.meta_info["timing"] = {"generate_sequences": time.time() - t_start}
        print("[BfclChatCompletionScheduler] generate_sequences done")
        return output_batch

    async def _submit_chat_completions_semaphore(self, messages: List[Dict[str, str]], request_id: str,
                                                 sampling_params: Dict[str, Any], extra_info: Dict[str, Any] = None):
        """重写_submit_chat_completions_semaphore方法，支持extra_info参数"""
        done = asyncio.Event()

        info = {
            "__done__": done,
            "__depth__": 0,  # indicate how many ongoing completion requests
            "__sampling_params__": sampling_params,
        }

        # 将extra_info合并到info中
        if extra_info:
            info.update(extra_info)

        self.submit_chat_completions(messages=messages, request_id=request_id, info=info)

        # Wait until all completion requests are done
        await done.wait()
