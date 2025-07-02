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
        """
        Enhanced chat completions with retry mechanism for tool parsing errors.
        
        Args:
            address: Server address
            **chat_complete_request: Chat completion request parameters
            
        Returns:
            ChatCompletion: Response or fallback response for tool parsing errors
        """
        original_request = chat_complete_request.copy()
        
        for attempt in range(self.max_retries + 1):  # 0, 1, 2, 3 (total 4 attempts)
            try:
                extra_body = chat_complete_request.pop("extra_body", {})
                chat_complete_request.update(extra_body or {})
                extra_headers = chat_complete_request.pop("extra_headers", {})
                timeout = aiohttp.ClientTimeout(total=None)
                session = aiohttp.ClientSession(timeout=timeout)
                
                try:
                    async with session.post(
                        url=f"http://{address}/v1/chat/completions",
                        headers={"Authorization": "Bearer token-abc123", **extra_headers},
                        json=chat_complete_request,
                    ) as resp:
                        data = await resp.json()
                        
                        # Check if response contains error related to tool parsing
                        if "error" in data:
                            error_msg = str(data.get("error", {}))
                            if self._is_tool_parsing_error(error_msg):
                                if attempt < self.max_retries:
                                    logger.warning(f"Tool parsing error detected (attempt {attempt + 1}/{self.max_retries + 1}): {error_msg}")
                                    # Reset request for retry
                                    chat_complete_request = original_request.copy()
                                    await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff
                                    continue
                                else:
                                    logger.error(f"Tool parsing error persists after {self.max_retries} retries: {error_msg}")
                                    return self._create_fallback_response(original_request, error_msg)
                        
                        # Try to create ChatCompletion object, catch validation errors
                        try:
                            return ChatCompletion(**data)
                        except Exception as validation_error:
                            validation_error_str = str(validation_error)
                            logger.warning(f"ChatCompletion validation error: {validation_error_str}")
                            
                            # Check if this is related to tool parsing
                            if self._is_tool_parsing_error(validation_error_str) or "error" in data:
                                if attempt < self.max_retries:
                                    logger.warning(f"Validation error due to tool parsing (attempt {attempt + 1}/{self.max_retries + 1})")
                                    # Reset request for retry
                                    chat_complete_request = original_request.copy()
                                    await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff
                                    continue
                                else:
                                    logger.error(f"Validation error persists after {self.max_retries} retries")
                                    return self._create_fallback_response(original_request, validation_error_str)
                            else:
                                # Other validation errors should be raised
                                raise validation_error
                        
                finally:
                    await session.close()
                    
            except Exception as e:
                error_str = str(e)
                # Check if this is a tool parsing related error
                if self._is_tool_parsing_error(error_str):
                    if attempt < self.max_retries:
                        logger.warning(f"Tool parsing exception (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                        # Reset request for retry
                        chat_complete_request = original_request.copy()
                        await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff
                        continue
                    else:
                        logger.error(f"Tool parsing exception persists after {self.max_retries} retries: {e}")
                        return self._create_fallback_response(original_request, error_str)
                else:
                    # Non-tool-parsing errors should be raised immediately
                    raise e
                    
        # This should not be reached, but just in case
        return self._create_fallback_response(original_request, "Maximum retries exceeded")

    def _is_tool_parsing_error(self, error_message: str) -> bool:
        """
        Check if the error message indicates a tool parsing error.
        
        Args:
            error_message: Error message to check
            
        Returns:
            bool: True if it's a tool parsing error
        """
        error_indicators = [
            "hermes_tool_parser",
            "tool_parsers/hermes_tool_parser", 
            "JSONDecodeError",
            "json.decoder.JSONDecodeError",
            "Unterminated string",
            "Expecting value",
            "Expecting property name",
            "Expecting ',' delimiter",
            "Expecting ':' delimiter", 
            "Error in extracting tool call",
            "invalid tool call format",
            "extract_tool_calls",
            "validation errors for ChatCompletion",
            "Field required",
            "Input should be 'chat.completion'"
        ]
        
        error_lower = error_message.lower()
        return any(indicator.lower() in error_lower for indicator in error_indicators)

    def _create_fallback_response(self, original_request: Dict[str, Any], error_message: str) -> ChatCompletion:
        """
        Create a fallback ChatCompletion response when tool parsing fails persistently.
        
        Args:
            original_request: Original chat completion request
            error_message: Error message to include in response
            
        Returns:
            ChatCompletion: Fallback response with error content
        """
        from openai.types.chat.chat_completion import Choice
        from openai.types.chat.chat_completion_message import ChatCompletionMessage
        from openai.types.completion_usage import CompletionUsage
        
        # Create error content that explains the issue
        error_content = f"Tool call format parsing failed. Error message: {error_message}"
        
        # Try to extract model name from request
        model_name = original_request.get("model", "unknown")
        
        # Create a structured fallback response
        fallback_response = ChatCompletion(
            id=f"chatcmpl-fallback-{uuid4().hex[:8]}",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        content=error_content,
                        role="assistant"
                    )
                )
            ],
            created=int(time.time()),
            model=model_name,
            object="chat.completion",
            usage=CompletionUsage(
                completion_tokens=len(error_content.split()),
                prompt_tokens=0,  # We don't have access to exact token count
                total_tokens=len(error_content.split())
            )
        )
        
        logger.info(f"Created fallback response for persistent tool parsing error: {error_content}")
        return fallback_response


# Usage example:
# To use the robust scheduler instead of the default one, replace:
#   scheduler = ChatCompletionScheduler(config, server_addresses)
# with:
#   scheduler = RobustChatCompletionScheduler(config, server_addresses, max_retries=3)
#
# The robust scheduler will automatically retry tool parsing errors up to 3 times
# before falling back to a structured error response.
