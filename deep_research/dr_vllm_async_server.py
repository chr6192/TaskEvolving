import logging
import os
import time
from typing import Any, Dict
from uuid import uuid4

import ray
from omegaconf import DictConfig
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.v1.engine.async_llm import AsyncLLM

from verl.utils.fs import copy_to_local
from verl.workers.rollout.async_server import AsyncServerBase
from verl.workers.rollout.vllm_rollout.vllm_async_server import ExternalRayDistributedExecutor

logger = logging.getLogger(__file__)


@ray.remote(num_cpus=1)
class DrAsyncvLLMServer(AsyncServerBase):
    """
    Deep Research项目专用的AsyncvLLMServer，继承原始AsyncvLLMServer并增加tool parsing错误处理功能。
    
    主要增强功能：
    1. 捕获和处理tool parsing相关异常
    2. 返回结构化的错误响应而非直接崩溃
    3. 保持与OpenAI API的兼容性
    4. 修复了Pydantic ChatCompletion验证错误问题，确保所有响应都符合正确的格式
    """

    def __init__(self, config: DictConfig, vllm_dp_size: int, vllm_dp_rank: int, wg_prefix: str):
        """
        初始化Deep Research专用的AsyncvLLMServer
        
        Args:
            config: DictConfig配置
            vllm_dp_size: vLLM数据并行大小
            vllm_dp_rank: vLLM数据并行排名
            wg_prefix: 工作组前缀
        """
        super().__init__()

        self.config = config.actor_rollout_ref
        self.vllm_dp_size = vllm_dp_size
        self.vllm_dp_rank = vllm_dp_rank
        self.wg_prefix = wg_prefix
        self.engine: AsyncLLM = None
        
        # 启用错误处理标志
        self.enable_error_handling = True

    async def init_engine(self):
        """Init vLLM AsyncLLM engine."""
        config = self.config
        model_path = config.model.path
        model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(model_path)
        trust_remote_code = config.model.get("trust_remote_code", False)
        config = config.rollout

        tensor_parallel_size = config.get("tensor_model_parallel_size", 1)
        max_num_batched_tokens = config.get("max_num_batched_tokens", 8192)
        max_model_len = config.max_model_len if config.max_model_len else config.prompt_length + config.response_length
        max_model_len = int(max_model_len)

        # Override default generation config from hugging face model config,
        # user can still override them by passing kwargs in each request.
        kwargs = dict(
            n=1,
            logprobs=0,
            repetition_penalty=1.0,
            max_new_tokens=config.response_length,
        )
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)
        print(f"override_generation_config: {kwargs}")

        engine_args = AsyncEngineArgs(
            model=local_path,
            enable_sleep_mode=True,
            override_generation_config=kwargs,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend=ExternalRayDistributedExecutor if os.environ.get("VERL_VLLM_USE_RAY_BACKEND", "1") == "1" else None,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format="auto",
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=self.vllm_dp_rank,
        )

        # init async llm engine
        vllm_config = engine_args.create_engine_config()
        namespace = ray.get_runtime_context().namespace
        vllm_config.instance_id = f"{namespace}:{self.wg_prefix}:{self.vllm_dp_size}:{self.vllm_dp_rank}"
        self.engine = AsyncLLM.from_vllm_config(vllm_config)

        # build serving chat
        model_config = self.engine.model_config
        BASE_MODEL_PATHS = [BaseModelPath(name=model_name, model_path=model_path)]
        models = OpenAIServingModels(self.engine, model_config, BASE_MODEL_PATHS)
        self.openai_serving_chat = OpenAIServingChat(
            self.engine,
            model_config,
            models,
            "assistant",
            request_logger=RequestLogger(max_log_len=4096),
            chat_template=None,
            chat_template_content_format="auto",
            enable_auto_tools=True,
            tool_parser=config.multi_turn.format,  # hermes, llama3_json, ...
        )

    async def wake_up(self):
        await self.engine.wake_up()

    async def sleep(self):
        await self.engine.reset_prefix_cache()
        await self.engine.sleep()

    async def chat_completion(self, raw_request: Request):
        """
        OpenAI兼容的HTTP端点，增强了tool parsing错误处理功能。

        API参考: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        if not self.enable_error_handling:
            # 如果未启用错误处理，使用默认的 AsyncvLLMServer 实现
            request_json = await raw_request.json()
            request = ChatCompletionRequest(**request_json)
            generator = await self.openai_serving_chat.create_chat_completion(request, raw_request)

            if isinstance(generator, ErrorResponse):
                return JSONResponse(content=generator.model_dump(), status_code=generator.code)
            if request.stream:
                return StreamingResponse(content=generator, media_type="text/event-stream")
            else:
                assert isinstance(generator, ChatCompletionResponse)
                return JSONResponse(content=generator.model_dump())
        
        request_json = None
        try:
            request_json = await raw_request.json()
            request = ChatCompletionRequest(**request_json)
            generator = await self.openai_serving_chat.create_chat_completion(request, raw_request)

            if isinstance(generator, ErrorResponse):
                return JSONResponse(content=generator.model_dump(), status_code=generator.code)
            if request.stream:
                return StreamingResponse(content=generator, media_type="text/event-stream")
            else:
                assert isinstance(generator, ChatCompletionResponse)
                return JSONResponse(content=generator.model_dump())
                
        except Exception as e:
            error_str = str(e)
            logger.exception(f"Chat completion failed with exception: {e}")
            
            # 检查是否为tool parsing相关错误
            if self._is_tool_parsing_error(error_str):
                logger.warning(f"Tool parsing error detected, creating fallback ChatCompletion response: {error_str}")
                return self._create_tool_parsing_error_response(request_json or {}, error_str)
            else:
                # 对于非tool parsing错误，返回通用错误响应
                logger.error(f"Non-tool-parsing error occurred, returning standard error response: {error_str}")
                return self._create_generic_error_response(error_str)

    def _is_tool_parsing_error(self, error_message: str) -> bool:
        """
        检查错误消息是否表示tool parsing错误。
        
        Args:
            error_message: 要检查的错误消息
            
        Returns:
            bool: 如果是tool parsing错误则返回True
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
            "Error in extracting tool call",
            "invalid tool call format",
            "tool_parsers",
            "extract_tool_calls",
            "Expecting ':' delimiter",
            "Invalid control character",
            "tool call parse",
            "function call parse",
            # 添加Pydantic验证错误相关的指示器
            "validation errors for ChatCompletion",
            "Field required",
            "Input should be 'chat.completion'",
            "literal_error",
            "pydantic"
        ]
        
        error_lower = error_message.lower()
        return any(indicator.lower() in error_lower for indicator in error_indicators)

    def _create_tool_parsing_error_response(self, request_json: Dict[str, Any], error_message: str) -> JSONResponse:
        """
        为tool parsing错误创建结构化的错误响应。
        
        Args:
            request_json: 原始请求JSON
            error_message: 错误消息
            
        Returns:
            JSONResponse: 带有回退内容的错误响应，符合ChatCompletion格式
        """
        model_name = request_json.get("model", "unknown")
        
        # 创建指示tool parsing失败的回退响应
        error_content = (
            f"Tool call parsing failed. The model generated an invalid tool call format that could not be parsed. "
            f"This may be due to malformed JSON in the tool call arguments. "
            f"Technical details: {error_message}"
        )
        
        # 创建符合ChatCompletion标准的响应结构
        chat_completion_response = {
            "id": f"chatcmpl-error-{uuid4().hex[:8]}",
            "object": "chat.completion",  # 必须是 "chat.completion"，不能是 "error"
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": error_content
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(error_content.split()),
                "total_tokens": len(error_content.split())
            }
        }
        
        logger.info(f"Created tool parsing error response for model {model_name}")
        # 返回200状态码以避免破坏pipeline，确保响应格式符合ChatCompletion标准
        return JSONResponse(content=chat_completion_response, status_code=200)

    def _create_generic_error_response(self, error_message: str) -> JSONResponse:
        """
        为非tool parsing错误创建通用错误响应。
        
        Args:
            error_message: 错误消息
            
        Returns:
            JSONResponse: 通用错误响应，使用标准OpenAI错误格式
        """
        # 使用标准的OpenAI错误响应格式
        error_response = {
            "error": {
                "message": f"Internal server error: {error_message}",
                "type": "internal_server_error",
                "code": "internal_server_error"
            }
        }
        
        return JSONResponse(content=error_response, status_code=500)
