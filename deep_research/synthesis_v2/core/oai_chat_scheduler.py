"""
OpenAI API Chat Completion Scheduler - 修复版本
继承默认的 ChatCompletionScheduler，实现使用 OpenAI API 接口的自定义方法
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, AsyncGenerator, Optional

from omegaconf import DictConfig
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

try:
    from verl.protocol import DataProto
    from verl.workers.rollout.chat_scheduler import ChatCompletionScheduler
except ImportError as e:
    raise ImportError(
        "无法导入 `verl`。请通过环境配置解决（推荐在仓库根目录执行 `pip install -e .`，"
        "或确保 `PYTHONPATH` 包含本仓库与 `./verl`）。"
    ) from e

try:
    from deep_research.llm_utils import (
        LLMModelUtils,
        configure_request_extra_body,
        is_invalid_tool_calls_completion,
        is_non_retryable_client_error,
        is_responses_api_payload,
        normalize_openai_base_url,
        normalize_responses_payload_to_chat_completion,
        persist_invalid_tool_calls_response,
        resolve_openai_api_key,
    )
    from .message_context_utils import (
        CHARS_PER_TOKEN_ESTIMATE,
        apply_request_input_char_budget,
        resolve_request_input_char_budget,
    )
except ImportError:
    # 如果在synthesis目录外运行，尝试直接导入
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from deep_research.llm_utils import (
        LLMModelUtils,
        configure_request_extra_body,
        is_invalid_tool_calls_completion,
        is_non_retryable_client_error,
        is_responses_api_payload,
        normalize_openai_base_url,
        normalize_responses_payload_to_chat_completion,
        persist_invalid_tool_calls_response,
        resolve_openai_api_key,
    )
    from deep_research.synthesis_v2.core.message_context_utils import (
        CHARS_PER_TOKEN_ESTIMATE,
        apply_request_input_char_budget,
        resolve_request_input_char_budget,
    )

logger = logging.getLogger(__name__)

class OaiChatCompletionScheduler(ChatCompletionScheduler):
    """
    使用 OpenAI API 的 Chat Completion Scheduler - oai模型去除部分参数的版本
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
        self.config = config.actor_rollout_ref.rollout if hasattr(config, 'actor_rollout_ref') else DictConfig({})
        self.full_config = config  # 保存完整config以访问各环节配置
        
        # OpenAI API 相关配置
        self.openai_config = config.get('openai_api', {})
        
        # 获取 trajectory generation 环节的配置（因为此scheduler仅用于trajectory generate）
        generation_config = config.get('generation', {})
        
        # 调试日志：查看配置内容
        logger.info(f"[配置检查] generation_config.model_name: {generation_config.get('model_name')}")
        logger.info(f"[配置检查] generation_config.api_key: {'***' if generation_config.get('api_key') else None}")
        logger.info(f"[配置检查] generation_config.base_url: {generation_config.get('base_url')}")
        logger.info(f"[配置检查] openai_config.default_model_name: {self.openai_config.get('default_model_name')}")
        
        # 模型名称优先级：generation环节配置 > 全局配置
        self.model_name = generation_config.get('model_name') or self.openai_config.get('default_model_name', 'qwen3.6-plus')
        
        # Base URL优先级：generation环节配置 > OPENAI_BASE_URL环境变量 > 全局配置
        # 允许通过端口转发后仅改环境变量实现“本地无缝切换”。
        self.base_url = normalize_openai_base_url(
            generation_config.get('base_url')
            or os.getenv('OPENAI_BASE_URL')
            or self.openai_config.get('base_url', 'https://api.openai.com/v1')
        )

        # API Key优先级：generation环节配置 > 全局配置 > 环境变量
        self.api_key = generation_config.get('api_key') or resolve_openai_api_key(
            self.openai_config.get('api_key'),
            base_url=self.base_url,
        )
        
        # self.organization = self.openai_config.get('organization') or os.getenv('OPENAI_ORG_ID')
        self.timeout = self.openai_config.get('timeout', 60)
        self.max_retries = self.openai_config.get('max_retries', 3)
        
        # 429 配额超限重试配置
        self.rate_limit_max_retries = self.openai_config.get('rate_limit_max_retries', 5)
        self.rate_limit_initial_wait = self.openai_config.get('rate_limit_initial_wait', 10)  # 初始等待时间(秒)
        self.rate_limit_max_wait = self.openai_config.get('rate_limit_max_wait', 120)  # 最大等待时间(秒)
        self.output_error_max_retries = int(
            self.openai_config.get('output_error_max_retries', 3)
        )
        
        # 获取默认LLM参数（从openai_api.default_*）
        self.default_temperature = self.openai_config.get('default_temperature', 0.7)
        self.default_top_p = self.openai_config.get('default_top_p', 0.9)
        self.default_top_k = self.openai_config.get('default_top_k', None)
        self.default_max_tokens = self.openai_config.get('default_max_tokens')

        self.max_request_input_char_budget = resolve_request_input_char_budget(
            max_request_input_tokens=generation_config.get("max_request_input_tokens"),
            max_request_input_chars=generation_config.get("max_request_input_chars"),
            fallback_tokens=self.openai_config.get("max_request_input_tokens"),
        )

        logger.info(
            "[配置检查] generation max_request_input: %s chars (~%s tokens)",
            self.max_request_input_char_budget,
            self.max_request_input_char_budget // CHARS_PER_TOKEN_ESTIMATE
            if self.max_request_input_char_budget
            else 0,
        )
        
        if not self.api_key:
            raise ValueError("OpenAI API key 未设置。请在配置文件中设置 openai_api.api_key 或设置环境变量 OPENAI_API_KEY")
        
        # 初始化 OpenAI 客户端（使用trajectory generation的配置）
        self.openai_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        
        # 其他初始化逻辑
        self.background_tasks = set()
        
        # 初始化 completion callback
        multi_turn_config = config.get('multi_turn', {})
        if multi_turn_config.get('completion_callback') is None:
            # 默认使用DrSynthesisCompletionCallback
            from deep_research.synthesis_v2.core.dr_synthesis_completion_callback import DrSynthesisCompletionCallback
            self.completion_callback = DrSynthesisCompletionCallback(config, self)
            logger.info("completion_callback未指定，使用默认: DrSynthesisCompletionCallback")
        else:
            import importlib
            module_path, class_name = multi_turn_config.completion_callback.rsplit(".", 1)
            module = importlib.import_module(module_path)
            self.completion_callback = getattr(module, class_name)(config, self)
            logger.info(f"使用自定义completion_callback: {class_name}")
        
        logger.info(f"初始化 OaiChatCompletionScheduler (Trajectory Generation)")
        logger.info(f"  使用模型: {self.model_name}")
        logger.info(f"  Base URL: {self.base_url}")
        logger.info(f"  429重试配置: max_retries={self.rate_limit_max_retries}, "
                   f"initial_wait={self.rate_limit_initial_wait}s, max_wait={self.rate_limit_max_wait}s")

    def _is_reasoning_model(self) -> bool:
        """
        检测当前模型是否为reasoning模型（只能通过stream调用）
        
        Returns:
            bool: 如果是reasoning模型返回True
        """
        is_reasoning = LLMModelUtils.is_reasoning_model(self.model_name)
        if is_reasoning:
            logger.info(f"检测到reasoning模型: {self.model_name}，将使用stream模式调用")
        return is_reasoning

    def _calculate_wait_time(self, retry_count: int) -> float:
        """指数退避等待时间（秒）。"""
        wait_time = self.rate_limit_initial_wait * (2 ** (retry_count - 1))
        return min(wait_time, self.rate_limit_max_wait)

    async def _collect_stream_response(self, stream: AsyncGenerator[ChatCompletionChunk, None]) -> ChatCompletion:
        """
        收集stream响应并组装成完整的ChatCompletion对象
        支持所有OpenAI API返回的字段，包括reasoning_content等
        
        Args:
            stream: OpenAI API的stream响应生成器
            
        Returns:
            ChatCompletion: 完整的响应对象
        """
        # 初始化响应数据结构
        collected_response = {
            "id": None,
            "object": "chat.completion",
            "created": None,
            "model": None,
            "choices": [],
            "usage": None,
            "system_fingerprint": None,
            "service_tier": None
        }
        
        # 用于收集消息内容和其他字段
        message_content = ""
        reasoning_content = ""
        tool_calls = []
        finish_reason = None
        logprobs = None
        
        try:
            async for chunk in stream:
                # 设置基础信息（只需要设置一次）
                if collected_response["id"] is None:
                    collected_response["id"] = chunk.id
                    collected_response["created"] = chunk.created
                    collected_response["model"] = chunk.model
                    collected_response["system_fingerprint"] = getattr(chunk, 'system_fingerprint', None)
                    collected_response["service_tier"] = getattr(chunk, 'service_tier', None)
                
                # 处理choices
                if chunk.choices:
                    choice = chunk.choices[0]  # 通常只有一个choice
                    delta = choice.delta
                    
                    # 收集消息内容
                    if delta.content:
                        message_content += delta.content
                    
                    # 收集reasoning content (推理模型的重要字段)
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        reasoning_content += delta.reasoning_content
                    
                    # 收集tool calls
                    if delta.tool_calls:
                        for tool_call_chunk in delta.tool_calls:
                            # 确保tool_calls列表足够长
                            while len(tool_calls) <= tool_call_chunk.index:
                                tool_calls.append({
                                    "id": None,
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""}
                                })
                            
                            tool_call = tool_calls[tool_call_chunk.index]
                            
                            if tool_call_chunk.id:
                                tool_call["id"] = tool_call_chunk.id
                            
                            if tool_call_chunk.function:
                                if tool_call_chunk.function.name:
                                    tool_call["function"]["name"] += tool_call_chunk.function.name
                                if tool_call_chunk.function.arguments:
                                    tool_call["function"]["arguments"] += tool_call_chunk.function.arguments
                    
                    # 记录finish_reason
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
                    
                    # 收集logprobs信息
                    if hasattr(choice, 'logprobs') and choice.logprobs:
                        logprobs = choice.logprobs
                
                # 收集usage信息（包括详细的token统计）
                if chunk.usage:
                    usage_dict = chunk.usage.model_dump()
                    # 确保包含所有usage字段
                    if not usage_dict.get("completion_tokens_details"):
                        usage_dict["completion_tokens_details"] = None
                    if not usage_dict.get("prompt_tokens_details"):
                        usage_dict["prompt_tokens_details"] = None
                    collected_response["usage"] = usage_dict
        
        except Exception as e:
            logger.error(f"收集stream响应时出错: {e}")
            raise
        
        # 组装最终的message，包含所有可能的字段
        message = {
            "role": "assistant",
            "content": message_content if message_content else None,
            "refusal": None,
            "annotations": None,
            "audio": None,
            "function_call": None
        }
        
        # 添加reasoning_content（推理模型特有）
        if reasoning_content:
            message["reasoning_content"] = reasoning_content
        
        # 添加tool_calls（过滤 name 为空的残缺 chunk）
        if tool_calls:
            tool_calls = [
                tc for tc in tool_calls
                if (tc.get("function") or {}).get("name")
            ]
        if tool_calls:
            message["tool_calls"] = tool_calls
        else:
            message["tool_calls"] = None
            if finish_reason == "tool_calls":
                finish_reason = "stop"
        
        # 组装choice，包含所有字段
        choice_dict = {
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
            "logprobs": logprobs
        }
        
        collected_response["choices"] = [choice_dict]
        
        logger.debug(f"Stream响应收集完成，内容长度: {len(message_content)}, 推理内容长度: {len(reasoning_content)}, 工具调用数: {len(tool_calls)}")
        
        # 创建ChatCompletion对象
        return ChatCompletion(**collected_response)

    def _is_rate_limit_error(self, exception: Exception) -> bool:
        """
        判断是否为429配额超限错误
        
        Args:
            exception: 异常对象
            
        Returns:
            bool: 是否为429错误
        """
        error_str = str(exception)
        # 检查错误码429或insufficient_quota
        return ('429' in error_str or 
                'insufficient_quota' in error_str.lower() or
                'rate_limit' in error_str.lower())

    def _is_connection_error(self, exception: Exception) -> bool:
        """网络连接失败、超时等瞬时错误，适合指数退避重试。"""
        if isinstance(exception, (APIConnectionError, APITimeoutError)):
            return True
        msg = str(exception).lower()
        needles = (
            "connecterror",
            "connection error",
            "all connection attempts failed",
            "name or service not known",
            "temporary failure in name resolution",
            "nodename nor servname provided",
            "connection refused",
            "network is unreachable",
            "timed out",
            "request timed out",
            "tls",
            "ssl",
        )
        return any(n in msg for n in needles)
    
    @staticmethod
    def _apply_max_output_tokens(
        request_params: Dict[str, Any],
        max_tokens: int,
        model_name: str,
    ) -> None:
        """gpt-5/o 系列使用 max_completion_tokens，其余模型使用 max_tokens。"""
        if LLMModelUtils.uses_max_completion_tokens(model_name):
            request_params["max_completion_tokens"] = int(max_tokens)
        else:
            request_params["max_tokens"] = int(max_tokens)

    def _resolve_trajectory_max_tokens(self, component_config: Dict[str, Any]) -> Optional[int]:
        """generation.max_tokens > openai_api.default_max_tokens。"""
        if component_config and component_config.get("max_tokens") is not None:
            return int(component_config["max_tokens"])
        if self.default_max_tokens is not None:
            return int(self.default_max_tokens)
        return None

        """
        计算指数退避等待时间
        
        Args:
            retry_count: 当前重试次数(从1开始)
            
        Returns:
            float: 等待时间(秒)
        """
        # 指数退避: initial_wait * (2 ^ (retry_count - 1))
        wait_time = self.rate_limit_initial_wait * (2 ** (retry_count - 1))
        # 限制最大等待时间
        wait_time = min(wait_time, self.rate_limit_max_wait)
        return wait_time

    def _log_request_payload(self, request_params: Dict[str, Any], retry_count: int) -> None:
        """打印每次 solver 模型调用入参（截断避免日志过大）。"""
        try:
            payload = json.dumps(request_params, ensure_ascii=False, default=str)
        except Exception:
            payload = str(request_params)
        if len(payload) > 12000:
            payload = payload[:12000] + "... <truncated>"
        logger.info(
            "solver_model_api=chat.completions.create retry=%s base_url=%s request_params=%s",
            retry_count,
            self.base_url,
            payload,
        )

    def _persist_418_payload(self, request_params: Dict[str, Any], retry_count: int, exception: Exception) -> None:
        """将 418/provider_failed 请求完整入参持久化到 jsonl，便于复盘。"""
        dump_path = Path(os.getenv("SYNTHESIS_418_DUMP_FILE", "outputs/llm_418_requests.jsonl"))
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "component": "solver",
            "api_name": "chat.completions.create",
            "retry_count": retry_count,
            "model": request_params.get("model", self.model_name),
            "base_url": self.base_url,
            "exception": str(exception),
            "request_params": request_params,
        }
        try:
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            with dump_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            logger.warning("418请求已落盘: %s", dump_path)
        except Exception as dump_err:  # noqa: BLE001
            logger.error("418请求落盘失败: %s", dump_err)

    def _normalize_responses_api_completion(
        self,
        completion: ChatCompletion,
        *,
        model: str,
    ) -> ChatCompletion:
        """仅 DashScope mr.gpt-5.5：Responses API 形态转 ChatCompletion。"""
        if not LLMModelUtils.needs_dashscope_gpt5_input_workaround(model, self.base_url):
            return completion
        data = completion.model_dump()
        if not is_responses_api_payload(data):
            return completion
        normalized = normalize_responses_payload_to_chat_completion(data)
        return ChatCompletion(**normalized)

    async def _create_chat_completion(self, **chat_complete_request) -> ChatCompletion:
        model = chat_complete_request.get("model", self.model_name)
        use_gpt55_workaround = LLMModelUtils.needs_dashscope_gpt5_input_workaround(
            model, self.base_url
        )
        if use_gpt55_workaround and not chat_complete_request.get("stream", False):
            raw = await self.openai_client.chat.completions.with_raw_response.create(
                **chat_complete_request
            )
            data = normalize_responses_payload_to_chat_completion(json.loads(raw.text))
            return ChatCompletion(**data)

        completion = await self.openai_client.chat.completions.create(**chat_complete_request)
        return self._normalize_responses_api_completion(completion, model=model)

    async def _chat_completions_openai(self, **chat_complete_request) -> ChatCompletion:
        """
        使用 OpenAI API 进行 chat completion
        支持reasoning模型的stream处理和429错误自动重试
        
        Args:
            **chat_complete_request: OpenAI API 请求参数（应该已包含model参数）
            
        Returns:
            ChatCompletion: OpenAI API 响应
        """
        # 获取实际使用的模型名称（应该已经在sampling_params中设置好了）
        actual_model = chat_complete_request.get('model', self.model_name)
        
        # 检测是否为reasoning模型，如果是则强制使用stream模式
        if LLMModelUtils.is_reasoning_model(actual_model):
            chat_complete_request['stream'] = True
            logger.info(f"Reasoning模型 {actual_model} 强制使用stream模式")
        
        # 处理 extra_body（thinking / DashScope GPT-5 input workaround）
        configure_request_extra_body(
            chat_complete_request,
            self.base_url,
            apply_thinking=not chat_complete_request.get("stream", False),
        )

        invalid_attempt = 0
        while invalid_attempt <= self.output_error_max_retries:
            completion, last_exception = await self._request_chat_completion_once(
                chat_complete_request,
                actual_model=actual_model,
            )
            if last_exception is not None:
                raise last_exception
            assert completion is not None

            if not is_invalid_tool_calls_completion(completion):
                if invalid_attempt > 0:
                    logger.info(
                        "空壳 tool_calls 重试成功 [attempt=%s,id=%s]",
                        invalid_attempt,
                        completion.id,
                    )
                return completion

            if invalid_attempt >= self.output_error_max_retries:
                persist_invalid_tool_calls_response(
                    source="oai_chat_scheduler_exhausted",
                    completion_id=completion.id,
                    completion=completion.model_dump(),
                )
                return completion

            invalid_attempt += 1
            wait_time = self._calculate_wait_time(invalid_attempt)
            logger.warning(
                "finish_reason=tool_calls 但无有效 tool_calls，"
                "第 %s/%s 次 API 重试，等待 %.1fs [model=%s]",
                invalid_attempt,
                self.output_error_max_retries,
                wait_time,
                actual_model,
            )
            await asyncio.sleep(wait_time)

        raise RuntimeError("unreachable: invalid tool_calls retry loop exhausted")

    async def _request_chat_completion_once(
        self,
        chat_complete_request: Dict[str, Any],
        *,
        actual_model: str,
    ) -> tuple[ChatCompletion | None, Exception | None]:
        """单次 chat completion 请求，含 429/418/连接类错误重试。"""
        last_exception: Exception | None = None
        for retry_count in range(self.rate_limit_max_retries + 1):
            try:
                start_time = time.time()

                if retry_count == 0:
                    logger.info("=" * 60)
                    logger.info("OpenAI API 请求参数:")
                    logger.info(f"  model: {chat_complete_request.get('model', self.model_name)}")
                    logger.info(f"  stream: {chat_complete_request.get('stream', False)}")
                    if 'messages' in chat_complete_request:
                        logger.info(f"  messages: {len(chat_complete_request['messages'])} 条消息")
                    if 'tools' in chat_complete_request:
                        logger.info(f"  tools: {len(chat_complete_request['tools'])} 个工具")
                    if 'extra_body' in chat_complete_request:
                        logger.info(f"  extra_body: {chat_complete_request['extra_body']}")
                    logger.info("=" * 60)
                else:
                    logger.info(f"第 {retry_count} 次重试 OpenAI API 请求...")
                self._log_request_payload(chat_complete_request, retry_count)

                if chat_complete_request.get('stream', False):
                    logger.info("使用stream模式调用OpenAI API")
                    stream = await self.openai_client.chat.completions.create(**chat_complete_request)
                    completion = await self._collect_stream_response(stream)
                    completion = self._normalize_responses_api_completion(
                        completion,
                        model=chat_complete_request.get("model", self.model_name),
                    )
                else:
                    logger.info("使用非stream模式调用OpenAI API")
                    completion = await self._create_chat_completion(**chat_complete_request)

                duration = time.time() - start_time
                logger.debug(f"OpenAI API 请求完成，耗时: {duration:.2f}s")

                if retry_count > 0:
                    logger.info(f"✓ 第 {retry_count} 次重试成功!")
                return completion, None

            except Exception as e:
                last_exception = e
                if is_non_retryable_client_error(e):
                    logger.error("客户端参数/配置错误，放弃重试: %s", e)
                    return None, e

                err_lower = str(e).lower()
                is_418_provider = "418" in err_lower and (
                    "provider_failed" in err_lower or "http code: 500" in err_lower
                )
                if is_418_provider:
                    self._persist_418_payload(chat_complete_request, retry_count, e)
                is_retryable = (
                    self._is_rate_limit_error(e)
                    or is_418_provider
                    or self._is_connection_error(e)
                )
                if is_retryable:
                    if retry_count < self.rate_limit_max_retries:
                        wait_time = self._calculate_wait_time(retry_count + 1)
                        logger.warning(
                            f"遇到429/418/超时/连接类可重试错误 "
                            f"(第 {retry_count + 1}/{self.rate_limit_max_retries} 次重试)"
                        )
                        logger.warning(f"等待 {wait_time:.1f} 秒后重试...")
                        logger.warning(f"错误详情: {e}")
                        await asyncio.sleep(wait_time)
                        continue
                    logger.error(f"已达到最大重试次数 ({self.rate_limit_max_retries})，放弃重试")
                    logger.error(f"最后的错误: {e}")
                    return None, e

                logger.error(f"OpenAI API 请求失败(不可重试错误): {e}")
                return None, e

        return None, last_exception

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
            api_messages = apply_request_input_char_budget(
                messages,
                max_request_input_chars=self.max_request_input_char_budget,
                tool_schemas=self.completion_callback.tool_schemas,
                log_prefix="trajectory API",
            )
            # 调用 OpenAI API（仅发送截断后的视图，messages 仍保留完整 tool 返回用于落盘）
            completions = await self._chat_completions_openai(
                messages=api_messages,
                tools=self.completion_callback.tool_schemas,
                **info["__sampling_params__"],
            )
        except Exception as e:
            exception = e

        info["__depth__"] -= 1

        if exception is not None:
            logger.exception(f"chat completion failed with exception: {exception}")
            err_text = str(exception)
            if err_text.strip():
                messages.append({"role": "assistant", "content": err_text})
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
        
        # 获取 generation/trajectory 环节的配置
        component_config = self.full_config.get('generation', {})
        
        # 参数优先级：generation环节配置 > rollout config > 全局默认值
        # 1. temperature
        temperature = self.config.get('temperature') if self.config else None  # rollout config默认值
        if component_config and component_config.get('temperature') is not None:
            temperature = component_config.get('temperature')
        elif temperature is None:
            temperature = self.default_temperature
        
        # 2. top_p
        top_p = self.config.get('top_p') if self.config else None  # rollout config默认值
        if component_config and component_config.get('top_p') is not None:
            top_p = component_config.get('top_p')
        elif top_p is None:
            top_p = self.default_top_p
        
        # 3. top_k
        top_k = None
        if component_config and component_config.get('top_k') is not None:
            top_k = component_config.get('top_k')
        else:
            top_k = self.default_top_k
        
        # 4. model_name - 支持环节级覆盖
        model_name = self.model_name  # 默认模型
        if component_config and component_config.get('model_name') is not None:
            model_name = component_config.get('model_name')

        # 5. max_tokens - generation.max_tokens > openai_api.default_max_tokens
        max_tokens = self._resolve_trajectory_max_tokens(component_config)
        
        kwargs = dict(
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            parallel_tool_calls=True,
        )
        if max_tokens is not None:
            self._apply_max_output_tokens(kwargs, max_tokens, model_name)
        
        # 添加top_k（如果指定）
        if top_k is not None:
            kwargs['top_k'] = top_k

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            if self.config and hasattr(self.config, 'val_kwargs'):
                kwargs["top_p"] = self.config.val_kwargs.get('top_p', 0.9)
                kwargs["temperature"] = self.config.val_kwargs.get('temperature', 0.7)
            kwargs["parallel_tool_calls"] = True

        logger.info(f"generate_sequences sampling params: {kwargs}")

        # NOTE: For multi-turn rollout, repeat raw_prompt n times and process each prompt independently,
        # validation dataset has already been repeated in `PPOTrainer._validate`.
        n = 1 if batch.meta_info.get("validate", False) else (self.config.get('n', 1) if self.config else 1)
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
        logger.info("generate_sequences done")
        return output_batch
