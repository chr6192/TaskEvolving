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
        try:
            extra_body = chat_complete_request.pop("extra_body", {})
            chat_complete_request.update(extra_body or {})
            extra_headers = chat_complete_request.pop("extra_headers")
            timeout = aiohttp.ClientTimeout(total=None)
            session = aiohttp.ClientSession(timeout=timeout)
            async with session.post(
                url=f"http://{address}/v1/chat/completions",
                headers={"Authorization": "Bearer token-abc123", **extra_headers},
                json=chat_complete_request,
            ) as resp:
                # data = await resp.json()
                generator = await resp.json()
                if isinstance(generator, ErrorResponse):
                    return JSONResponse(content=generator.model_dump(), status_code=generator.code)
                if chat_complete_request.stream:
                    return StreamingResponse(content=generator, media_type="text/event-stream")
                else:
                    assert isinstance(generator, ChatCompletionResponse)
                    return JSONResponse(content=generator.model_dump())

                return ChatCompletion(**data)
        finally:
            await session.close()