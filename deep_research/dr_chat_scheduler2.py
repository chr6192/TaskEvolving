import asyncio
import heapq
import importlib
import itertools
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from uuid import uuid4

import aiohttp
import numpy as np
import torch
from cachetools import LRUCache
from omegaconf import DictConfig
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.tools.base_tool import initialize_tools_from_config
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.workers.rollout.chat_scheduler import ChatCompletionScheduler
from starlette.responses import JSONResponse, StreamingResponse
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse


logger = logging.getLogger(__file__)

class DrChatCompletionScheduler(ChatCompletionScheduler):
    """
    A robust version of ChatCompletionScheduler that handles tool parsing errors with retry mechanism.
    
    When tool parsing errors occur, it will retry the original request up to 3 times before falling back
    to a structured error response.
    """
    
    def __init__(self, config: DictConfig, server_addresses: List[str], max_cache_size: int = 10000):
        """
        Args:
            config: DictConfig.
            server_addresses: List[str], OpenAI compatible server addresses.
            max_cache_size: int, max cache size of request_id to address mapping.
            max_retries: int, maximum number of retries for tool parsing errors.
        """
        super().__init__(config, server_addresses, max_cache_size)
        # Safe access to max_retries config with default fallback
        try:
            self.max_retries = config.actor_rollout_ref.rollout.multi_turn.chat_scheduler_kwargs.max_retries
        except (AttributeError, KeyError):
            logger.warning("max_retries not found in config, using default value of 3")
            self.max_retries = 3
        
    async def _chat_completions_aiohttp(self, address: str, **chat_complete_request) -> ChatCompletion:
        session = None
        try:
            extra_body = chat_complete_request.pop("extra_body", {})
            chat_complete_request.update(extra_body or {})
            extra_headers = chat_complete_request.pop("extra_headers", {})
            timeout = aiohttp.ClientTimeout(total=None)
            session = aiohttp.ClientSession(timeout=timeout)
            async with session.post(
                url=f"http://{address}/v1/chat/completions",
                headers={"Authorization": "Bearer token-abc123", **extra_headers},
                json=chat_complete_request,
            ) as resp:
                # 检查响应状态
                if resp.status != 200:
                    logger.warning(f"Warning: HTTP {resp.status} response from {address}")
                    error_text = await resp.text()
                    logger.warning(f"Error response body: {error_text}")
                    
                    # 尝试解析错误响应为JSON
                    try:
                        # error_data = json.loads(error_text)
                        error_data = await resp.json()
                        logger.warning(f"Parsed error data: {error_data}")
                    except json.JSONDecodeError:
                        logger.exception("JSONDecodeError")
                        error_data = {"error": {"message": error_text, "type": "http_error", "code": resp.status}}
                    
                    # 构造一个表示错误的ChatCompletion对象
                    error_message = error_data.get("error", {}).get("message", error_text) if isinstance(error_data, dict) else str(error_text)
                    
                    # 创建错误响应的ChatCompletion格式
                    error_completion_data = {
                        "id": f"chatcmpl-error-{uuid4().hex[:8]}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": chat_complete_request.get("model", "unknown"),
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": f"Error: {error_message}"
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0
                        }
                    }
                    return ChatCompletion(**error_completion_data)
                
                data = await resp.json()
                # logger.info(f"Success response: {data}")
                
                # 直接返回 ChatCompletion 对象
                return ChatCompletion(**data)
        finally:
            if session:
                await session.close()