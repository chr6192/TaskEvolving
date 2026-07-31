"""
LLM调用相关的通用工具函数 - 简化版
提供模型判断和参数准备的辅助功能
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

MessageInput = Union[str, List[Dict[str, Any]]]


def resolve_openai_api_key(
    api_key: Optional[str] = None,
    *,
    base_url: Optional[str] = None,
) -> Optional[str]:
    """合成/探索等通用 LLM：显式配置 > OPENAI_API_KEY > DASHSCOPE_API_KEY。

    当 base_url 指向 api.v3.cm 时，不回退 DASHSCOPE_API_KEY（该 key 不能用于 v3.cm）。
    """
    if api_key:
        return api_key
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return openai_key
    normalized = normalize_openai_base_url(base_url) if base_url else ""
    if "v3.cm" in normalized:
        return None
    return os.getenv("DASHSCOPE_API_KEY")


def resolve_rubric_api_key(api_key: Optional[str] = None) -> Optional[str]:
    """Rubric reward：显式配置 > DASHSCOPE_API_KEY > OPENAI_API_KEY。"""
    if api_key:
        return api_key
    return os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")


def normalize_openai_base_url(base_url: str | None) -> str:
    """
    OpenAI SDK 的 base_url 应为 API 根路径（如 https://host/v1），
    勿包含 /chat/completions，否则 SDK 会拼成 .../chat/completions/chat/completions。
    """
    url = (base_url or "").strip().rstrip("/")
    if not url:
        return "https://api.openai.com/v1"
    lower = url.lower()
    for suffix in ("/chat/completions", "/completions"):
        if lower.endswith(suffix):
            url = url[: -len(suffix)].rstrip("/")
            lower = url.lower()
    return url


class LLMModelUtils:
    """LLM模型特性判断工具"""
    
    @staticmethod
    def is_reasoning_model(model_name: str) -> bool:
        """
        检测当前模型是否为reasoning模型（必须通过stream调用）
        
        Reasoning模型（如o1系列）只能通过流式调用，不支持非流式调用
        
        Args:
            model_name: 模型名称
            
        Returns:
            bool: 如果是reasoning模型返回True
        """
        reasoning_model_prefixes = [
            'o1-',  # OpenAI o1 系列模型
            'o1',   # OpenAI o1 模型的其他变体
            'qwen3-235b-a22b-thinking-2507',
            'qwen3-235b-a22b',
        ]
        
        model_name_lower = model_name.lower()
        for prefix in reasoning_model_prefixes:
            if model_name_lower.startswith(prefix.lower()):
                return True
        
        return False
    
    @staticmethod
    def is_thinking_model(model_name: str) -> bool:
        """
        检测模型是否支持thinking模式
        
        Thinking模型支持在API调用时设置enable_thinking参数
        
        Args:
            model_name: 模型名称
            
        Returns:
            bool: 如果支持thinking模式返回True
        """
        return (
            model_name.startswith('o1') or 
            model_name.startswith('qwen3') or
            'thinking' in model_name.lower()
        )
    
    @staticmethod
    def should_enable_thinking(model_name: str) -> bool:
        """
        判断是否应该启用thinking模式
        
        只有特定的thinking专用模型才默认启用thinking
        
        Args:
            model_name: 模型名称
            
        Returns:
            bool: 是否应该启用thinking
        """
        return (
            "thinking" in model_name.lower() or 
            model_name == "qwen3-235b-a22b"
        )
    
    @staticmethod
    def is_dashscope_compatible_mode(base_url: str) -> bool:
        return "dashscope.aliyuncs.com/compatible-mode" in (base_url or "")

    @staticmethod
    def is_gpt5_model(model_name: str) -> bool:
        name = (model_name or "").lower()
        return "gpt-5" in name or name.startswith("gpt-5")

    @staticmethod
    def is_gpt_series_model(model_name: str) -> bool:
        """OpenAI GPT / o-series；仅这些模型默认尝试 beta.chat.completions.parse。"""
        name = (model_name or "").lower()
        if "gpt-" in name:
            return True
        base = name.rsplit(".", 1)[-1]
        return base.startswith(("o1", "o3", "o4"))

    _DASHSCOPE_RESPONSES_INPUT_MODELS = frozenset(
        {
            "azure.gpt-5.5-2026-04-24",
        }
    )

    @staticmethod
    def uses_dashscope_responses_input_format(model_name: str) -> bool:
        """DashScope 上走 Responses API input/tools 转换的模型。"""
        name = (model_name or "").strip().lower()
        if "completion" in name:
            return False
        if name == "mr.gpt-5.5" or name.startswith("mr.gpt-5.5-"):
            return True
        return name in LLMModelUtils._DASHSCOPE_RESPONSES_INPUT_MODELS

    @staticmethod
    def is_mr_gpt55_model(model_name: str) -> bool:
        return LLMModelUtils.uses_dashscope_responses_input_format(model_name)

    @staticmethod
    def needs_dashscope_gpt5_input_workaround(model_name: str, base_url: str) -> bool:
        """DashScope 白名单模型需要 input/tools/响应转换；名称含 completion 则走原 chat.completions。"""
        return (
            LLMModelUtils.is_dashscope_compatible_mode(base_url)
            and LLMModelUtils.uses_dashscope_responses_input_format(model_name)
        )

    @staticmethod
    def uses_fixed_sampling(model_name: str) -> bool:
        """保留给其它调用方；DashScope GPT-5.5 workaround 内单独处理采样参数。"""
        name = (model_name or "").lower()
        if LLMModelUtils.is_gpt5_model(model_name):
            return True
        if name.startswith(("o1", "o3", "o4")):
            return True
        return LLMModelUtils.is_reasoning_model(model_name)

    @staticmethod
    def uses_max_completion_tokens(model_name: str) -> bool:
        """部分新模型（如 gpt-5）要求使用 max_completion_tokens 而非 max_tokens。"""
        name = (model_name or "").lower()
        if name.startswith(("o1", "o3", "o4")):
            return True
        return "gpt-5" in name or name.startswith("gpt-5")

    @staticmethod
    def cap_completion_max_tokens(model_name: str, max_tokens: int) -> int:
        """
        按 DashScope 各模型 completion 上限裁剪 max_tokens，避免 400 InvalidParameter。

        参考报错示例：MiniMax-M2.5 [1,32768]；qwen3-32b [1,16384]；qwen3.6-flash [1,65536]。
        """
        requested = max(1, int(max_tokens))
        name = (model_name or "").strip().lower()
        if "minimax" in name:
            cap = 32768
        elif any(
            name.startswith(p)
            for p in ("qwen3-14b", "qwen3-8b", "qwen3-4b", "qwen3-1.7b", "qwen3-0.6b")
        ):
            cap = 8192
        elif name.startswith("qwen3-32b") or name.startswith("qwen3-30b"):
            cap = 16384
        elif "qwen3.6" in name or "qwen3.7" in name or "qwen3.5" in name:
            cap = 65536
        elif "deepseek" in name or "glm" in name or "kimi" in name:
            cap = 65536
        else:
            cap = 65536
        capped = min(requested, cap)
        if capped < requested:
            logger.debug(
                "cap_completion_max_tokens: model=%s requested=%s capped=%s",
                model_name,
                requested,
                capped,
            )
        return capped

    @staticmethod
    def is_official_openai_api(base_url: str) -> bool:
        """
        判断是否为官方OpenAI API或兼容API
        
        只有官方API和某些兼容API支持extra_body参数
        
        Args:
            base_url: API的base URL
            
        Returns:
            bool: 是否为官方或兼容API
        """
        official_urls = [
            'https://api.openai.com/v1',
            'https://dashscope.aliyuncs.com/compatible-mode/v1'
        ]
        return any(url in base_url for url in official_urls)


def prepare_thinking_extra_body(model_name: str, base_url: str) -> Dict[str, Any]:
    """
    为thinking模型准备extra_body参数
    
    这个函数负责在非流式调用时，根据模型类型和API提供商，
    准备合适的extra_body参数（主要是enable_thinking设置）
    
    规则：
    1. 只在官方OpenAI API或兼容API时才添加extra_body
    2. 只针对thinking模型添加
    3. qwen3.6-plus等非thinking专用模型：enable_thinking=False
    4. qwen3-235b-a22b等thinking专用模型：enable_thinking=True
    
    Args:
        model_name: 模型名称
        base_url: API的base URL
        
    Returns:
        Dict: extra_body参数字典，如果不需要则返回空字典
        
    Examples:
        >>> prepare_thinking_extra_body("qwen3.6-plus", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        {'enable_thinking': False}
        
        >>> prepare_thinking_extra_body("qwen3-235b-a22b", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        {'enable_thinking': True, 'thinking_budget': 8192}
        
        >>> prepare_thinking_extra_body("gpt-4", "https://api.openai.com/v1")
        {}
        
        >>> prepare_thinking_extra_body("qwen3.6-plus", "http://localhost:8000/v1")
        {}
    """
    # 非官方API不支持extra_body
    if not LLMModelUtils.is_official_openai_api(base_url):
        return {}
    
    # 非thinking模型不需要设置
    if not LLMModelUtils.is_thinking_model(model_name):
        return {}
    
    # 判断是否启用thinking
    enable_thinking = LLMModelUtils.should_enable_thinking(model_name)
    
    # 构建extra_body
    extra_body = {"enable_thinking": enable_thinking}
    
    # 如果启用thinking，添加budget参数
    if enable_thinking:
        extra_body["thinking_budget"] = 8192
    
    logger.debug(f"为模型 {model_name} 准备extra_body: {extra_body}")
    
    return extra_body


def _chat_messages_need_responses_conversion(messages: List[Dict[str, Any]]) -> bool:
    for msg in messages:
        role = msg.get("role")
        if role == "tool":
            return True
        if role == "assistant" and msg.get("tool_calls"):
            return True
    return False


def chat_messages_to_responses_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Chat Completions 多轮（含 tool_calls / tool）转为 Responses API input items。"""
    items: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role in ("system", "user"):
            items.append({"role": role, "content": msg.get("content") or ""})
            continue
        if role == "assistant":
            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls") or []
            if content:
                items.append({"role": "assistant", "content": content})
            for tc in tool_calls:
                fn = tc.get("function") or {}
                items.append(
                    {
                        "type": "function_call",
                        "call_id": tc.get("id"),
                        "name": fn.get("name"),
                        "arguments": coerce_tool_call_arguments(fn.get("arguments")),
                    }
                )
            if not content and not tool_calls:
                items.append({"role": "assistant", "content": ""})
            continue
        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id"),
                    "output": msg.get("content") or "",
                }
            )
    return items


def messages_to_dashscope_input(messages: List[Dict[str, Any]]) -> MessageInput:
    """将 chat messages 转为 DashScope GPT-5 workaround 所需的 input。"""
    if _chat_messages_need_responses_conversion(messages):
        return chat_messages_to_responses_input(messages)
    if len(messages) == 1 and messages[0].get("role") == "user":
        content = messages[0].get("content")
        if isinstance(content, str):
            return content
    return messages


def coerce_tool_call_arguments(raw: Any) -> str:
    """把 Responses/ChatCompletions 的 function.arguments 统一成 JSON 字符串。"""
    if raw is None:
        return "{}"
    if isinstance(raw, dict):
        return json.dumps(raw, ensure_ascii=False)
    if isinstance(raw, str):
        stripped = raw.strip()
        return stripped if stripped else "{}"
    return "{}"


def parse_tool_call_arguments(raw: Any) -> Dict[str, Any]:
    """安全解析 tool call arguments；空/非法时返回空 dict。"""
    coerced = coerce_tool_call_arguments(raw)
    try:
        parsed = json.loads(coerced)
    except json.JSONDecodeError:
        logger.warning("无法解析 tool call arguments: %r，使用空字典", raw)
        return {}
    if isinstance(parsed, dict):
        return parsed
    logger.warning("tool call arguments 不是 object: %r，使用空字典", parsed)
    return {}


def _function_call_arguments_from_output_item(item: Dict[str, Any]) -> str:
    nested = item.get("function")
    if isinstance(nested, dict):
        raw = nested.get("arguments", item.get("arguments"))
    else:
        raw = item.get("arguments")
    return coerce_tool_call_arguments(raw)


def _function_call_name_from_output_item(item: Dict[str, Any]) -> str:
    name = item.get("name") or ""
    nested = item.get("function")
    if not name and isinstance(nested, dict):
        name = nested.get("name") or ""
    return name


def _tool_calls_from_choices(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    tool_calls: List[Dict[str, Any]] = []
    for choice in data.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        msg = choice.get("message") or {}
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            if fn.get("name"):
                tool_calls.append(tc)
    return tool_calls


def _tool_call_names_by_id(tool_calls: List[Dict[str, Any]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for tc in tool_calls:
        call_id = tc.get("id")
        fn = tc.get("function") or {}
        name = fn.get("name") or ""
        if call_id and name:
            mapping[call_id] = name
    return mapping


def is_invalid_tool_calls_completion(completion: Any) -> bool:
    """finish_reason=tool_calls 但没有任何带 name 的有效 tool_calls。"""
    if hasattr(completion, "model_dump"):
        data = completion.model_dump()
    elif isinstance(completion, dict):
        data = completion
    else:
        return False

    choices = data.get("choices") or []
    if not choices:
        return False
    choice = choices[0] if isinstance(choices[0], dict) else choices[0].model_dump()
    if choice.get("finish_reason") != "tool_calls":
        return False

    message = choice.get("message") or {}
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        return True

    for tc in tool_calls:
        if not isinstance(tc, dict):
            tc = tc.model_dump() if hasattr(tc, "model_dump") else {}
        fn = tc.get("function") or {}
        if isinstance(fn, dict):
            name = fn.get("name")
        else:
            name = getattr(fn, "name", None)
        if name:
            return False
    return True


def persist_invalid_tool_calls_response(
    *,
    source: str,
    completion_id: Optional[str] = None,
    turn: Optional[int] = None,
    completion: Optional[Dict[str, Any]] = None,
    raw_response: Optional[Dict[str, Any]] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """finish_reason=tool_calls 但无有效 tool_calls 时落盘，便于排查模型原始返回。"""
    dump_path = Path(
        os.getenv(
            "SYNTHESIS_INVALID_TOOL_CALLS_DUMP_FILE",
            "outputs/invalid_tool_calls_responses.jsonl",
        )
    )
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "completion_id": completion_id,
        "turn": turn,
        "completion": completion,
        "raw_response": raw_response,
        "messages": messages,
    }
    try:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with dump_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        logger.warning("无效 tool_calls 响应已落盘: %s [source=%s]", dump_path, source)
    except Exception as dump_err:  # noqa: BLE001
        logger.error("无效 tool_calls 响应落盘失败: %s", dump_err)


def is_responses_api_payload(data: Dict[str, Any]) -> bool:
    if data.get("object") == "response":
        return True
    output = data.get("output")
    if not output:
        return False
    if data.get("choices") is None:
        return True
    # GPT-5 @ DashScope 可能同时带 choices 与 output；output 中的 function_call 更完整。
    return True


def normalize_responses_payload_to_chat_completion(data: Dict[str, Any]) -> Dict[str, Any]:
    """把 DashScope GPT-5 返回的 Responses API JSON 转成合法 ChatCompletion dict。"""
    if not is_responses_api_payload(data):
        return data

    message_content = ""
    tool_calls: List[Dict[str, Any]] = []
    finish_reason = "stop"

    choices_tool_calls = _tool_calls_from_choices(data)
    name_by_call_id = _tool_call_names_by_id(choices_tool_calls)

    for choice in data.get("choices") or []:
        if isinstance(choice, dict) and choice.get("finish_reason") == "tool_calls":
            finish_reason = "tool_calls"

    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "function_call":
            finish_reason = "tool_calls"
            call_id = item.get("call_id") or item.get("id")
            name = _function_call_name_from_output_item(item)
            if not name and call_id:
                name = name_by_call_id.get(call_id, "")
            if not name:
                logger.warning(
                    "Responses API function_call 缺少 name，已跳过: call_id=%s item=%s",
                    call_id,
                    item,
                )
                continue
            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": _function_call_arguments_from_output_item(item),
                    },
                }
            )
        elif item_type == "message":
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    message_content += part.get("text") or ""

    if not tool_calls and choices_tool_calls:
        tool_calls = choices_tool_calls
        finish_reason = "tool_calls"

    usage = dict(data.get("usage") or {})
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is None:
        completion_tokens = usage.get("output_tokens") or 0
    prompt_tokens = usage.get("prompt_tokens")
    if prompt_tokens is None:
        prompt_tokens = usage.get("input_tokens") or 0
    total_tokens = usage.get("total_tokens")
    if total_tokens is None:
        total_tokens = int(completion_tokens) + int(prompt_tokens)

    created = data.get("created")
    if created is None:
        created = data.get("created_at")
    if created is None:
        created = int(time.time())

    normalized = {
        "id": data.get("id") or "resp-converted",
        "object": "chat.completion",
        "created": int(created),
        "model": data.get("model") or "",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": message_content or None,
                    "tool_calls": tool_calls or None,
                },
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
        "usage": {
            "completion_tokens": int(completion_tokens),
            "prompt_tokens": int(prompt_tokens),
            "total_tokens": int(total_tokens),
        },
    }
    logger.debug(
        "Responses API 响应已转为 ChatCompletion: finish_reason=%s tool_calls=%s",
        finish_reason,
        len(tool_calls),
    )
    return normalized


_SAMPLING_PARAM_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "presence_penalty",
    "frequency_penalty",
)

_GPT5_EXTRA_UNSUPPORTED_KEYS = (
    "parallel_tool_calls",
)


def normalize_tools_for_responses_api(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Chat Completions 的 tools 转为 Responses API 所需的扁平格式（tools[i].name）。"""
    normalized: List[Dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("name"):
            normalized.append(tool)
            continue
        fn = tool.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            flat: Dict[str, Any] = {"type": tool.get("type", "function"), "name": fn["name"]}
            if fn.get("description") is not None:
                flat["description"] = fn["description"]
            if fn.get("parameters") is not None:
                flat["parameters"] = fn["parameters"]
            normalized.append(flat)
        else:
            normalized.append(tool)
    return normalized


def strip_unsupported_sampling_params(request_params: Dict[str, Any]) -> None:
    """GPT-5 / o 系列 / reasoning 等模型不支持 temperature 等采样参数。"""
    model = str(request_params.get("model", "") or "")
    if not LLMModelUtils.uses_fixed_sampling(model):
        return
    removed = [
        key
        for key in (*_SAMPLING_PARAM_KEYS, *_GPT5_EXTRA_UNSUPPORTED_KEYS)
        if key in request_params
    ]
    for key in removed:
        request_params.pop(key, None)
    if removed:
        logger.debug("已移除模型不支持的采样参数 %s [model=%s]", removed, model)


def _strip_dashscope_gpt55_unsupported_params(request_params: Dict[str, Any]) -> None:
    strip_unsupported_sampling_params(request_params)


def configure_request_extra_body(
    request_params: Dict[str, Any],
    base_url: str,
    *,
    apply_thinking: bool = True,
) -> None:
    """
    为请求补齐 extra_body。

    仅当 DashScope + 白名单模型（且名称不含 completion）时额外应用 input/tools workaround；
    其它模型保持原有 messages + thinking extra_body 逻辑不变。
    """
    model = request_params.get("model", "")
    messages = request_params.get("messages") or []

    extra_body = dict(request_params.get("extra_body") or {})
    if apply_thinking:
        extra_body.update(prepare_thinking_extra_body(model, base_url))

    if LLMModelUtils.needs_dashscope_gpt5_input_workaround(model, base_url):
        extra_body["input"] = messages_to_dashscope_input(messages)
        request_params["messages"] = []
        if request_params.get("tools"):
            request_params["tools"] = normalize_tools_for_responses_api(
                request_params["tools"]
            )
        logger.debug(
            "DashScope mr.gpt-5.5 workaround: 已将 %s 条 messages 迁入 extra_body.input",
            len(messages),
        )

    strip_unsupported_sampling_params(request_params)

    if extra_body:
        request_params["extra_body"] = extra_body
    elif "extra_body" in request_params:
        request_params.pop("extra_body", None)


def is_non_retryable_client_error(exception: Exception) -> bool:
    """客户端参数/配置错误，重试无意义。"""
    if is_input_token_limit_error(exception):
        return True
    msg = str(exception).lower()
    needles = (
        "is not supported for this model",
        "unsupported parameter",
        "unknown parameter",
        "missing required parameter",
        "missing_required_parameter",
    )
    return any(n in msg for n in needles)


def is_input_token_limit_error(exception: Exception) -> bool:
    """输入 token 超出模型/网关上限（HTTP 400），重试无意义。"""
    msg = str(exception).lower()
    needles = (
        "input tokens exceed",
        "tokens exceed the configured limit",
        "maximum context length",
        "context length exceeded",
        "context_length_exceeded",
        "prompt is too long",
    )
    return any(n in msg for n in needles)
