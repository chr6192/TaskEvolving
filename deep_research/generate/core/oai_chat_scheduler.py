"""
OpenAI API Chat Completion Scheduler
继承默认的 ChatCompletionScheduler，实现使用 OpenAI API 接口的自定义方法
"""

import asyncio
import logging
import os
import time
from typing import List, Dict, Any

from omegaconf import DictConfig
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion

from verl.protocol import DataProto
from verl.workers.rollout.chat_scheduler import ChatCompletionScheduler

logger = logging.getLogger(__name__)


class OaiChatCompletionScheduler(ChatCompletionScheduler):
    """
    使用 OpenAI API 的 Chat Completion Scheduler
    """
    
    def __init__(
        self,
        config: DictConfig,
        server_addresses: List[str] = None,  # 对于 OpenAI API，这个参数不会被使用
        max_cache_size: int = 10000,
    ):
        """
        初始化 OpenAI API Chat Completion Scheduler
        
        Args:
            config: DictConfig 配置
            server_addresses: 服务器地址列表（对 OpenAI API 无用，保持兼容性）
            max_cache_size: 最大缓存大小
        """
        # 调用父类初始化，但忽略 server_addresses 相关的负载均衡逻辑
        self.config = config.actor_rollout_ref.rollout
        model_path = config.actor_rollout_ref.model.path
        self.model_name = model_path  # 对于 OpenAI API，直接使用模型名称
        
        # OpenAI API 相关配置
        self.openai_config = config.get('openai_api', {})
        self.api_key = self.openai_config.get('api_key') or os.getenv('OPENAI_API_KEY')
        self.base_url = self.openai_config.get('base_url', 'https://api.openai.com/v1')
        self.organization = self.openai_config.get('organization') or os.getenv('OPENAI_ORG_ID')
        self.timeout = self.openai_config.get('timeout', 60)
        self.max_retries = self.openai_config.get('max_retries', 3)
        
        if not self.api_key:
            raise ValueError("OpenAI API key 未设置。请在配置文件中设置 openai_api.api_key 或设置环境变量 OPENAI_API_KEY")
        
        # 初始化 OpenAI 客户端
        self.openai_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            organization=self.organization,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        
        # 其他初始化逻辑
        self.background_tasks = set()
        
        # 初始化 completion callback
        if self.config.multi_turn.completion_callback is None:
            from verl.workers.rollout.chat_scheduler import ToolCompletionCallback
            self.completion_callback = ToolCompletionCallback(config, self)
            logger.warning("completion_callback is None, use ToolCompletionCallback")
        else:
            import importlib
            module_path, class_name = self.config.multi_turn.completion_callback.rsplit(".", 1)
            module = importlib.import_module(module_path)
            self.completion_callback = getattr(module, class_name)(config, self)
        
        logger.info(f"初始化 OaiChatCompletionScheduler，使用模型: {self.model_name}")
        logger.info(f"OpenAI API Base URL: {self.base_url}")

    async def _chat_completions_openai(self, **chat_complete_request) -> ChatCompletion:
        """
        使用 OpenAI API 进行 chat completion
        
        Args:
            **chat_complete_request: OpenAI API 请求参数
            
        Returns:
            ChatCompletion: OpenAI API 响应
        """
        # 移除不需要传递给 OpenAI API 的参数
        filtered_request = self._filter_openai_request(chat_complete_request)
        
        # 设置模型名称
        filtered_request['model'] = self.model_name
        
        try:
            # 记录请求开始时间
            start_time = time.time()
            
            # 调用 OpenAI API
            completion = await self.openai_client.chat.completions.create(**filtered_request)
            
            # 记录请求耗时
            duration = time.time() - start_time
            logger.debug(f"OpenAI API 请求完成，耗时: {duration:.2f}s")
            
            return completion
            
        except Exception as e:
            logger.error(f"OpenAI API 请求失败: {e}")
            raise

    def _filter_openai_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        过滤请求参数，只保留 OpenAI API 支持的参数
        
        Args:
            request: 原始请求参数
            
        Returns:
            Dict: 过滤后的请求参数
        """
        # OpenAI API 支持的标准参数
        supported_params = {
            'messages', 'model', 'temperature', 'top_p', 'n', 'stream', 
            'stop', 'max_tokens', 'presence_penalty', 'frequency_penalty',
            'logit_bias', 'user', 'tools', 'tool_choice', 'response_format',
            'seed', 'logprobs', 'top_logprobs', 'extra_body'
        }
        
        # 过滤参数
        filtered = {}
        
        for key, value in request.items():
            if key in supported_params:
                filtered[key] = value
            else:
                logger.debug(f"跳过不支持的参数: {key}")
        
        # 如果有 extra_body 参数，添加到请求中
        if filtered.get('stream', None) is not True:
            filtered['extra_body'] = {"enable_thinking": False}
        
        return filtered

    async def _chat_completions_aiohttp(self, address: str, **chat_complete_request) -> ChatCompletion:
        """
        重写此方法以支持 OpenAI API 的 HTTP 调用
        注意：对于 OpenAI API，address 参数会被忽略
        """
        return await self._chat_completions_openai(**chat_complete_request)

    def submit_chat_completions(self, *, messages: List[Dict[str, str]], request_id: str, info: Dict[str, Any]):
        """
        提交 chat completion 请求
        重写此方法以适配 OpenAI API
        """
        info["__depth__"] += 1
        task = asyncio.create_task(self._submit_chat_completions_and_callback(messages, request_id, info))

        # "fire-and-forget" background tasks
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    async def _submit_chat_completions_and_callback(
        self,
        messages: List[Dict[str, str]],
        request_id: str,
        info: Dict[str, Any],
    ):
        """
        提交 chat completion 请求并执行回调
        """
        completions, exception = None, None
        try:
            # 调用 OpenAI API
            completions = await self._chat_completions_openai(
                messages=messages,
                tools=self.completion_callback.tool_schemas,
                **info["__sampling_params__"],
            )
        except Exception as e:
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

        print(f"[ChatCompletionScheduler] generate_sequences sampling params: {kwargs}")

        # NOTE: For multi-turn rollout, repeat raw_prompt n times and process each prompt independently,
        # validation dataset has already been repeated in `PPOTrainer._validate`.
        n = 1 if batch.meta_info.get("validate", False) else self.config.n
        tasks, batch_conversations = [], [None] * len(batch) * n
        for batch_index, conversation in enumerate(batch.non_tensor_batch["raw_prompt"].repeat(n, axis=0)):
            # raw_prompt: [{"role": "user", "content": ""}, ["role": "assistant", "content"], ...]
            batch_conversations[batch_index] = conversation.tolist()

            tasks.append(
                asyncio.create_task(
                    self._submit_chat_completions_semaphore(
                        messages=batch_conversations[batch_index],
                        request_id=None,
                        sampling_params=kwargs,
                    )
                )
            )

        await asyncio.gather(*tasks)
        output_batch = self.completion_callback.postprocess(batch, batch_conversations, n=n)
        output_batch.meta_info["timing"] = {"generate_sequences": time.time() - t_start}
        print("[ChatCompletionScheduler] generate_sequences done")
        return output_batch
