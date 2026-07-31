"""
通用的结构化输出LLM客户端
为query_reviser和rubrics_evaluator等模块提供统一的LLM调用接口
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar, Type, Optional, Any
from omegaconf import DictConfig
from openai import AsyncOpenAI
from openai import APIConnectionError, APITimeoutError, InternalServerError
from pydantic import BaseModel

from deep_research.llm_utils import (
    LLMModelUtils,
    configure_request_extra_body,
    is_non_retryable_client_error,
    normalize_openai_base_url,
    normalize_responses_payload_to_chat_completion,
    resolve_openai_api_key,
)

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class StructuredLLMClient:
    """
    通用的结构化输出LLM客户端
    
    功能：
    1. 统一的LLM调用接口
    2. 支持结构化输出（Structured Outputs with Pydantic）
    3. 自动降级到JSON mode + 手动解析
    4. 处理各种边界情况和错误
    """
    
    def __init__(self, config: DictConfig, model_name: Optional[str] = None, 
                 component_config: Optional[DictConfig] = None):
        """
        初始化LLM客户端
        
        Args:
            config: 配置对象，应包含 openai_api 配置
            model_name: 模型名称，如果不提供则从config中获取
            component_config: 特定组件的配置（如 query_generation、reviser），用于获取组件级别的参数
        """
        self.config = config
        self.component_config = component_config
        # 日志组件名（兼容旧调用方未传 component 的情况）
        self.component = "reviser"
        self.openai_config = config.get('openai_api', {})
        
        # 模型名称优先级：传入参数 > openai_api.default_model_name
        self.model_name = model_name or self.openai_config.get('default_model_name', 'qwen3.6-plus')
        
        # base_url优先级：component_config > OPENAI_BASE_URL环境变量 > openai_config
        # 方便在不改yaml的情况下，用端口转发无缝切换服务入口。
        if component_config and component_config.get('base_url'):
            self.base_url = normalize_openai_base_url(component_config.get('base_url'))
        else:
            self.base_url = normalize_openai_base_url(
                os.getenv('OPENAI_BASE_URL') or self.openai_config.get('base_url', 'https://api.openai.com/v1')
            )

        # api_key优先级：component_config > openai_config > 环境变量（结合 base_url 决定是否回退 DashScope）
        explicit_api_key = None
        if component_config and component_config.get('api_key'):
            explicit_api_key = component_config.get('api_key')
        else:
            explicit_api_key = self.openai_config.get('api_key')
        self.api_key = resolve_openai_api_key(explicit_api_key, base_url=self.base_url)
        
        # 429配额超限重试配置
        self.rate_limit_max_retries = self.openai_config.get('rate_limit_max_retries', 5)
        self.rate_limit_initial_wait = self.openai_config.get('rate_limit_initial_wait', 10)
        self.rate_limit_max_wait = self.openai_config.get('rate_limit_max_wait', 120)
        self.output_error_max_retries = self.openai_config.get('output_error_max_retries', 6)
        self._max_api_retries = max(self.rate_limit_max_retries, self.output_error_max_retries)
        
        # 获取默认LLM参数（从openai_api.default_*）
        self.default_temperature = self.openai_config.get('default_temperature', 0.7)
        self.default_top_p = self.openai_config.get('default_top_p', 0.9)
        self.default_top_k = self.openai_config.get('default_top_k', None)
        self.default_max_tokens = self.openai_config.get('default_max_tokens', 2048)
        
        self.llm_client = None
        self._initialized = False
        
    async def initialize(self) -> None:
        """初始化OpenAI客户端"""
        if self._initialized:
            return
            
        logger.info("初始化StructuredLLMClient...")
        
        # 获取OpenAI API配置
        # api_key已经在__init__中按优先级设置好了
        timeout = self.openai_config.get('timeout', 60)
        max_retries = self.openai_config.get('max_retries', 3)
        
        if not self.api_key:
            raise ValueError("OpenAI API key未设置")
        
        # 初始化AsyncOpenAI客户端
        self.llm_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        
        self._initialized = True
        # 记录关键网络配置，便于排查“连不上/走代理/走错基址”
        proxy_env = {
            "HTTPS_PROXY": os.getenv("HTTPS_PROXY"),
            "https_proxy": os.getenv("https_proxy"),
            "HTTP_PROXY": os.getenv("HTTP_PROXY"),
            "http_proxy": os.getenv("http_proxy"),
            "NO_PROXY": os.getenv("NO_PROXY"),
            "no_proxy": os.getenv("no_proxy"),
        }
        proxy_env = {k: v for k, v in proxy_env.items() if v}
        logger.info(
            "StructuredLLMClient初始化完成，模型: %s, base_url: %s, proxy_env_keys: %s",
            self.model_name,
            self.base_url,
            sorted(proxy_env.keys()),
        )
        logger.info(f"429重试配置: max_retries={self.rate_limit_max_retries}, "
                   f"initial_wait={self.rate_limit_initial_wait}s, max_wait={self.rate_limit_max_wait}s")
        
    async def cleanup(self) -> None:
        """清理资源"""
        if self.llm_client:
            try:
                if hasattr(self.llm_client, 'close'):
                    await self.llm_client.close()
                self.llm_client = None
                logger.debug("LLM客户端资源已清理")
            except Exception as e:
                logger.warning(f"清理LLM客户端时出现警告: {e}")
    
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
        """
        判断是否为“网络连接建立失败/连接中断”类错误。
        这类错误往往是网络不可达、DNS、代理、VPN、防火墙、服务端临时不可用导致，
        通常适合做指数退避重试。
        """
        if isinstance(exception, (APIConnectionError, APITimeoutError)):
            return True
        # 兜底：OpenAI SDK / httpx / httpcore 的错误文案会包含这些关键词
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

    def _is_internal_server_error(self, exception: Exception) -> bool:
        """HTTP 500 / internal_server_error：远端瞬时故障，适合退避重试。"""
        if isinstance(exception, InternalServerError):
            return True
        msg = str(exception).lower()
        return (
            "internal_server_error" in msg
            or "error code: 500" in msg
            or "status code: 500" in msg
            or "status_code=500" in msg
        )

    async def _sleep_before_transient_retry(
        self,
        *,
        retry_count: int,
        exc: Exception,
        label: str,
        max_retries: Optional[int] = None,
    ) -> bool:
        limit = self.rate_limit_max_retries if max_retries is None else max_retries
        if retry_count >= limit:
            return False
        wait_time = self._calculate_wait_time(retry_count + 1)
        logger.warning(
            "%s (第 %s/%s 次重试)，等待 %.1f 秒后重试。错误: %s",
            label,
            retry_count + 1,
            limit,
            wait_time,
            exc,
        )
        await asyncio.sleep(wait_time)
        return True

    def _is_retryable_output_error(self, exception: Exception) -> bool:
        """空响应、JSON 解析失败等瞬时输出问题，适合退避重试。"""
        if isinstance(exception, json.JSONDecodeError):
            return True
        msg = str(exception).lower()
        return (
            "空 content" in str(exception)
            or "无法解析json" in msg
            or "expecting value" in msg
        )

    def _is_empty_content_error(self, exception: Exception) -> bool:
        return "空 content" in str(exception)

    def _calculate_empty_response_wait(self, retry_count: int) -> float:
        """空响应至少等待 8 秒，再指数退避。"""
        wait_time = max(8.0, float(self.rate_limit_initial_wait)) * (2 ** (retry_count - 1))
        return min(wait_time, float(self.rate_limit_max_wait))

    async def _sleep_before_output_retry(
        self,
        *,
        retry_count: int,
        exc: Exception,
    ) -> bool:
        if retry_count >= self.output_error_max_retries:
            return False
        if self._is_empty_content_error(exc):
            wait_time = self._calculate_empty_response_wait(retry_count + 1)
            label = "LLM 返回为空"
        else:
            wait_time = self._calculate_wait_time(retry_count + 1)
            label = "JSON 解析失败"
        logger.warning(
            "%s (第 %s/%s 次重试)，等待 %.1f 秒后重试。错误: %s",
            label,
            retry_count + 1,
            self.output_error_max_retries,
            wait_time,
            exc,
        )
        await asyncio.sleep(wait_time)
        return True
    
    def _calculate_wait_time(self, retry_count: int) -> float:
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

    def _prefer_json_mode(self) -> bool:
        """为 true 时跳过 beta.chat.completions.parse，直接用 JSON mode。"""
        if self.component_config is not None and self.component_config.get("prefer_json_mode") is not None:
            return bool(self.component_config.get("prefer_json_mode"))
        if self.openai_config.get("prefer_json_mode"):
            return True
        # 默认：仅 GPT 系列尝试 structured parse；DeepSeek/Qwen 等走 JSON mode
        return not LLMModelUtils.is_gpt_series_model(self.model_name)

    def _set_max_output_tokens(self, request_params: dict, max_tokens: int) -> None:
        if LLMModelUtils.uses_max_completion_tokens(self.model_name):
            request_params["max_completion_tokens"] = max_tokens
        else:
            request_params["max_tokens"] = max_tokens

    def _uses_dashscope_gpt55_workaround(self) -> bool:
        return LLMModelUtils.needs_dashscope_gpt5_input_workaround(
            self.model_name, self.base_url
        )

    async def _chat_completion_data(self, request_params: dict) -> dict:
        if self._uses_dashscope_gpt55_workaround():
            raw = await self.llm_client.chat.completions.with_raw_response.create(
                **request_params
            )
            return normalize_responses_payload_to_chat_completion(json.loads(raw.text))
        response = await self.llm_client.chat.completions.create(**request_params)
        return response.model_dump()

    @staticmethod
    def _content_from_completion_data(data: dict) -> str:
        choices = data.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            if content:
                return str(content)
        for item in data.get("output") or []:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    return str(part.get("text") or "")
        return ""

    def _log_request_payload(self, api_name: str, request_params: dict, retry_count: int) -> None:
        """打印每次模型调用的入参（截断避免日志过大）。"""
        try:
            payload = json.dumps(request_params, ensure_ascii=False, default=str)
        except Exception:
            payload = str(request_params)
        if len(payload) > 12000:
            payload = payload[:12000] + "... <truncated>"
        logger.info(
            "[%s] model_api=%s retry=%s base_url=%s request_params=%s",
            getattr(self, "component", "reviser"),
            api_name,
            retry_count,
            self.base_url,
            payload,
        )

    def _persist_418_payload(self, api_name: str, request_params: dict, retry_count: int, exception: Exception) -> None:
        """将 418/provider_failed 请求完整入参持久化到 jsonl，便于复盘。"""
        dump_path = Path(os.getenv("SYNTHESIS_418_DUMP_FILE", "outputs/llm_418_requests.jsonl"))
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "component": getattr(self, "component", "reviser"),
            "api_name": api_name,
            "retry_count": retry_count,
            "model": self.model_name,
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
                
    async def call_with_structured_output(
        self,
        prompt: str,
        response_model: Type[T],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        system_message: Optional[str] = None,
        **kwargs
    ) -> T:
        """
        调用LLM获取结构化输出
        
        GPT 系列首选 beta.chat.completions.parse；其它模型默认 JSON mode。
        structured parse 失败时自动降级到 JSON mode + 手动解析。
        
        Args:
            prompt: 输入prompt
            response_model: Pydantic模型类，定义期望的输出结构
            temperature: 温度参数（None表示使用默认值）
            max_tokens: 最大token数（None表示使用默认值）
            top_p: top_p参数（None表示使用默认值）
            top_k: top_k参数（None表示不使用）
            system_message: 可选的系统消息，用于设定角色和任务
            **kwargs: 其他参数
            
        Returns:
            response_model的实例，包含LLM的结构化输出
            
        Raises:
            RuntimeError: 如果客户端未初始化
            ValueError: 如果无法解析LLM输出
        """
        if not self._initialized or not self.llm_client:
            raise RuntimeError("StructuredLLMClient未初始化，请先调用initialize()")
        
        # 参数优先级：传入参数 > component_config > 全局默认值
        # 1. temperature
        if temperature is None:
            if self.component_config and self.component_config.get('temperature') is not None:
                temperature = self.component_config.get('temperature')
            else:
                temperature = self.default_temperature
        
        # 2. max_tokens
        if max_tokens is None:
            if self.component_config and self.component_config.get('max_tokens') is not None:
                max_tokens = self.component_config.get('max_tokens')
            else:
                max_tokens = self.default_max_tokens
        
        # 3. top_p
        if top_p is None:
            if self.component_config and self.component_config.get('top_p') is not None:
                top_p = self.component_config.get('top_p')
            else:
                top_p = self.default_top_p
        
        # 4. top_k
        if top_k is None:
            if self.component_config and self.component_config.get('top_k') is not None:
                top_k = self.component_config.get('top_k')
            else:
                top_k = self.default_top_k
        
        # 构建消息列表
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        user_chars = len(prompt)
        system_chars = len(system_message) if system_message else 0
        logger.info(
            "[%s] structured 入参字符: user=%s system=%s total=%s",
            getattr(self, "component", "reviser"),
            user_chars,
            system_chars,
            user_chars + system_chars,
        )
        
        # 构建额外参数
        extra_params = kwargs.copy()
        extra_params['top_p'] = top_p
        if top_k is not None:
            extra_params['top_k'] = top_k
        
        if self._prefer_json_mode():
            if LLMModelUtils.is_gpt_series_model(self.model_name):
                logger.info(
                    "prefer_json_mode=true，模型 %s 使用 JSON mode（跳过 structured parse）",
                    self.model_name,
                )
            else:
                logger.info(
                    "非 GPT 系列模型 %s，默认使用 JSON mode（跳过 beta.parse）",
                    self.model_name,
                )
            return await self._call_with_json_mode(
                messages, response_model, temperature, max_tokens, **extra_params
            )

        # 首选方案：使用结构化输出
        try:
            return await self._call_with_structured_api(
                messages, response_model, temperature, max_tokens, **extra_params
            )
        except Exception as struct_error:
            if is_non_retryable_client_error(struct_error):
                logger.error("客户端参数/配置错误，不再重试或降级: %s", struct_error)
                raise
            # 降级方案：使用JSON mode + 手动解析
            logger.warning(f"结构化输出失败: {struct_error}，降级到JSON mode...")
            return await self._call_with_json_mode(
                messages, response_model, temperature, max_tokens, **extra_params
            )
            
    async def _call_with_structured_api(
        self,
        messages: list,
        response_model: Type[T],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> T:
        """
        使用OpenAI的结构化输出API（支持429重试）
        
        Args:
            messages: 消息列表
            response_model: Pydantic响应模型
            temperature: 温度
            max_tokens: 最大token数
            **kwargs: 其他参数
            
        Returns:
            解析后的response_model实例
        """
        if self._uses_dashscope_gpt55_workaround():
            logger.info("DashScope mr.gpt-5.5 跳过 beta.parse，使用 JSON mode + raw response")
            return await self._call_with_json_mode(
                messages, response_model, temperature, max_tokens, **kwargs
            )

        # 构建请求参数
        request_params = {
            'model': self.model_name,
            'messages': messages,
            'response_format': response_model,
            'temperature': temperature,
            **kwargs
        }
        self._set_max_output_tokens(request_params, max_tokens)

        configure_request_extra_body(request_params, self.base_url)
        
        # 429错误重试逻辑
        last_exception = None
        for retry_count in range(self._max_api_retries + 1):
            try:
                if retry_count == 0:
                    logger.debug(f"使用结构化输出API调用 {self.model_name}...")
                else:
                    logger.info(f"第 {retry_count} 次重试结构化API调用...")
                self._log_request_payload(
                    "beta.chat.completions.parse",
                    request_params,
                    retry_count,
                )
                start_time = time.time()
                response = await self.llm_client.beta.chat.completions.parse(**request_params)
                duration = time.time() - start_time
                
                parsed_output = response.choices[0].message.parsed
                
                if parsed_output is None:
                    # 如果结构化解析返回None，尝试从content手动解析
                    content = response.choices[0].message.content
                    if not (content or "").strip():
                        raise ValueError("LLM 返回空 content")
                    logger.warning("结构化解析返回None，尝试手动解析...")
                    parsed_output = self._manual_parse_json(content, response_model)
                
                logger.debug(f"结构化输出解析成功，耗时: {duration:.2f}s")
                if retry_count > 0:
                    logger.info(f"✓ 第 {retry_count} 次重试成功!")
                return parsed_output
                
            except Exception as e:
                last_exception = e
                if is_non_retryable_client_error(e):
                    err_lower = str(e).lower()
                    if "context_length" in err_lower:
                        msg_chars = sum(
                            len(str(m.get("content") or ""))
                            for m in request_params.get("messages", [])
                        )
                        logger.error(
                            "context_length_exceeded: messages_chars≈%s",
                            msg_chars,
                        )
                    logger.error("客户端参数/配置错误，放弃重试: %s", e)
                    raise

                # 429 或 418/provider_failed：统一按指数退避重试
                err_lower = str(e).lower()
                is_418_provider = "418" in err_lower and (
                    "provider_failed" in err_lower or "http code: 500" in err_lower
                )
                if is_418_provider:
                    self._persist_418_payload(
                        "beta.chat.completions.parse",
                        request_params,
                        retry_count,
                        e,
                    )
                if self._is_rate_limit_error(e) or is_418_provider:
                    if retry_count < self.rate_limit_max_retries:
                        wait_time = self._calculate_wait_time(retry_count + 1)
                        logger.warning(
                            f"遇到429/418可重试错误 (第 {retry_count + 1}/{self.rate_limit_max_retries} 次重试)"
                        )
                        logger.warning(f"等待 {wait_time:.1f} 秒后重试...")
                        logger.warning(f"错误详情: {e}")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"已达到最大重试次数 ({self.rate_limit_max_retries})，放弃重试")
                        logger.error(f"最后的错误: {e}")
                        raise
                # 连接类错误：做指数退避重试（默认复用 rate_limit_max_retries 配置）
                if self._is_connection_error(e):
                    if retry_count < self.rate_limit_max_retries:
                        wait_time = self._calculate_wait_time(retry_count + 1)
                        logger.warning(
                            "遇到连接类错误 (第 %s/%s 次重试)，等待 %.1f 秒后重试。错误: %s",
                            retry_count + 1,
                            self.rate_limit_max_retries,
                            wait_time,
                            e,
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    logger.error("连接类错误已达到最大重试次数，放弃重试。最后错误: %s", e)
                    raise
                if self._is_internal_server_error(e):
                    if await self._sleep_before_transient_retry(
                        retry_count=retry_count,
                        exc=e,
                        label="遇到 HTTP 500 internal_server_error",
                    ):
                        continue
                    logger.error("HTTP 500 已达到最大重试次数，放弃重试。最后错误: %s", e)
                    raise
                if self._is_retryable_output_error(e):
                    if await self._sleep_before_output_retry(retry_count=retry_count, exc=e):
                        continue
                    logger.error("输出解析错误已达到最大重试次数，放弃重试。最后错误: %s", e)
                    raise

                # 非429且非连接类错误，直接抛出
                logger.error(f"结构化API调用失败(非429错误): {e}")
                raise
        
        # 理论上不会到达这里
        if last_exception:
            raise last_exception
        
    async def _call_with_json_mode(
        self,
        messages: list,
        response_model: Type[T],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> T:
        """
        使用JSON mode + 手动解析（降级方案，支持429重试）
        
        Args:
            messages: 消息列表
            response_model: Pydantic响应模型
            temperature: 温度
            max_tokens: 最大token数
            **kwargs: 其他参数
            
        Returns:
            解析后的response_model实例
        """
        # 兼容性要求：当使用 response_format={"type":"json_object"} 时，
        # 上游 messages 需要在某处显式提及 "json"（服务端会做参数校验）。
        # 这里做兜底，避免调用方漏写导致 400。
        try:
            joined = " ".join(
                str(m.get("content", "")) for m in (messages or []) if isinstance(m, dict)
            )
        except Exception:
            joined = ""
        if "json" not in (joined or "").lower():
            messages = list(messages or [])
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": "Return a single valid JSON object only (json).",
                },
            )

        # 构建请求参数
        request_params = {
            'model': self.model_name,
            'messages': messages,
            'response_format': {"type": "json_object"},
            'temperature': temperature,
            **kwargs
        }
        self._set_max_output_tokens(request_params, max_tokens)

        configure_request_extra_body(request_params, self.base_url)
        
        # 429错误重试逻辑
        last_exception = None
        for retry_count in range(self._max_api_retries + 1):
            try:
                if retry_count == 0:
                    logger.debug(f"使用JSON mode调用 {self.model_name}...")
                else:
                    logger.info(f"第 {retry_count} 次重试JSON mode调用...")
                self._log_request_payload(
                    "chat.completions.create(json_mode)",
                    request_params,
                    retry_count,
                )
                start_time = time.time()
                data = await self._chat_completion_data(request_params)
                duration = time.time() - start_time

                content = self._content_from_completion_data(data)
                if not (content or "").strip():
                    logger.warning("LLM 返回空 content")
                    raise ValueError("LLM 返回空 content")
                parsed_output = self._manual_parse_json(content, response_model)
                
                logger.debug(f"JSON mode解析成功，耗时: {duration:.2f}s")
                if retry_count > 0:
                    logger.info(f"✓ 第 {retry_count} 次重试成功!")
                return parsed_output
                
            except Exception as e:
                last_exception = e
                if is_non_retryable_client_error(e):
                    logger.error("客户端参数/配置错误，放弃重试: %s", e)
                    raise

                # 429 或 418/provider_failed：统一按指数退避重试
                err_lower = str(e).lower()
                is_418_provider = "418" in err_lower and (
                    "provider_failed" in err_lower or "http code: 500" in err_lower
                )
                if is_418_provider:
                    self._persist_418_payload(
                        "chat.completions.create(json_mode)",
                        request_params,
                        retry_count,
                        e,
                    )
                if self._is_rate_limit_error(e) or is_418_provider:
                    if retry_count < self.rate_limit_max_retries:
                        wait_time = self._calculate_wait_time(retry_count + 1)
                        logger.warning(
                            f"遇到429/418可重试错误 (第 {retry_count + 1}/{self.rate_limit_max_retries} 次重试)"
                        )
                        logger.warning(f"等待 {wait_time:.1f} 秒后重试...")
                        logger.warning(f"错误详情: {e}")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"已达到最大重试次数 ({self.rate_limit_max_retries})，放弃重试")
                        logger.error(f"最后的错误: {e}")
                        raise
                # 连接类错误：做指数退避重试
                if self._is_connection_error(e):
                    if retry_count < self.rate_limit_max_retries:
                        wait_time = self._calculate_wait_time(retry_count + 1)
                        logger.warning(
                            "遇到连接类错误 (第 %s/%s 次重试)，等待 %.1f 秒后重试。错误: %s",
                            retry_count + 1,
                            self.rate_limit_max_retries,
                            wait_time,
                            e,
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    logger.error("连接类错误已达到最大重试次数，放弃重试。最后错误: %s", e)
                    raise
                if self._is_internal_server_error(e):
                    if await self._sleep_before_transient_retry(
                        retry_count=retry_count,
                        exc=e,
                        label="遇到 HTTP 500 internal_server_error",
                    ):
                        continue
                    logger.error("HTTP 500 已达到最大重试次数，放弃重试。最后错误: %s", e)
                    raise
                if self._is_retryable_output_error(e):
                    if await self._sleep_before_output_retry(retry_count=retry_count, exc=e):
                        continue
                    logger.error("输出解析错误已达到最大重试次数，放弃重试。最后错误: %s", e)
                    raise

                # 非429且非连接类错误，直接抛出
                logger.error(f"JSON mode调用失败(非429错误): {e}")
                raise
        
        # 理论上不会到达这里
        if last_exception:
            raise last_exception
        
    def _manual_parse_json(self, content: str, response_model: Type[T]) -> T:
        """
        手动解析JSON内容为Pydantic模型
        
        处理各种可能的格式问题：
        - JSON可能被markdown代码块包裹
        - 可能有额外的文本
        - 字段可能缺失（依赖Pydantic的默认值）
        
        Args:
            content: LLM返回的文本内容
            response_model: 目标Pydantic模型
            
        Returns:
            解析后的response_model实例
            
        Raises:
            ValueError: 如果无法解析
        """
        try:
            # 清洗JSON内容
            json_str = self._extract_json_from_content(content)
            
            # 解析JSON
            data = json.loads(json_str)
            
            # 基本结构验证（检测明显的错误）
            self._validate_json_structure(data, response_model)
            
            # 使用Pydantic模型验证和构建对象
            # Pydantic会自动填充默认值、验证类型等
            parsed_output = response_model(**data)
            
            logger.debug("手动JSON解析成功")
            return parsed_output
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            logger.error(f"原始内容（前500字符）: {content[:500]}")
            raise ValueError(f"无法解析JSON: {e}")
        except Exception as e:
            logger.error(f"构建Pydantic模型失败: {e}")
            logger.error(f"返回的数据类型: {type(data)}")
            if isinstance(data, dict):
                logger.error(f"返回的JSON顶层键: {list(data.keys())}")
            raise ValueError(f"无法构建响应模型: {e}")
            
    def _validate_json_structure(self, data: Any, response_model: Type[T]) -> None:
        """
        基本结构验证，检测LLM是否返回了明显错误的数据
        
        Args:
            data: 解析后的JSON数据
            response_model: 期望的Pydantic模型
            
        Raises:
            ValueError: 如果结构明显错误
        """
        if not isinstance(data, dict):
            raise ValueError(f"期望返回dict，但得到了 {type(data)}")
        
        # 获取模型的必需字段
        model_fields = response_model.model_fields
        
        # 检查是否有必需字段
        for field_name, field_info in model_fields.items():
            if field_info.is_required() and field_name not in data:
                logger.warning(f"缺少必需字段: {field_name}")
        
        # 特殊检查：对于数组字段，检查元素类型是否正确
        for field_name, field_value in data.items():
            if isinstance(field_value, list) and len(field_value) > 0:
                # 检查第一个元素是否是期望的类型
                first_elem = field_value[0]
                # 如果期望是对象但得到的是基本类型，报错
                if isinstance(first_elem, (int, float, str, bool)) and field_name in model_fields:
                    field_type = model_fields[field_name].annotation
                    # 简单检查：如果字段类型提示包含 List[SomeModel]，第一个元素应该是dict
                    if hasattr(field_type, '__origin__') and hasattr(field_type, '__args__'):
                        inner_type = field_type.__args__[0] if field_type.__args__ else None
                        if inner_type and hasattr(inner_type, 'model_fields'):
                            # 这是一个Pydantic模型，期望dict
                            logger.warning(
                                f"字段 '{field_name}' 期望包含对象，但第一个元素是基本类型: {type(first_elem)}"
                            )
                            logger.warning(f"可能的原因：LLM返回了原始数据而不是评估结果")
            
    def _extract_json_from_content(self, content: str) -> str:
        """
        从LLM返回的内容中提取JSON字符串
        
        处理以下情况：
        1. 纯JSON
        2. ```json ... ``` 包裹
        3. ``` ... ``` 包裹（无语言标记）
        4. 带有额外文本的JSON
        
        Args:
            content: 原始内容
            
        Returns:
            清洗后的JSON字符串
        """
        # 尝试提取markdown代码块中的JSON
        if "```json" in content:
            match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if match:
                return match.group(1).strip()
                
        if "```" in content:
            match = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # 尝试提取大括号包裹的JSON
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return match.group(0)
            
        # 如果都失败，返回原始内容
        return content.strip()

