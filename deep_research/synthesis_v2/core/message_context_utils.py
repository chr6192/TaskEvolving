"""
多轮对话 API 入参的消息字符预算控制（trajectory / reviser 共用）。

按「整次请求」字符上限截断：先扣除 system/模板/tools 等固定开销，再对 tool+assistant 优先截 tool。
api.v3.cm + gpt-5.5 实测整请求 prompt_tokens 上限约 27 万。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN_ESTIMATE = 3
# 实测 api.v3.cm 整请求约 272k prompt_tokens；字符预算再留安全余量（JSON/中文 token 更密）
REQUEST_INPUT_CHAR_SAFETY_RATIO = 0.90
DEFAULT_MAX_REQUEST_INPUT_TOKENS = 240_000
DEFAULT_MAX_REQUEST_INPUT_CHARS = int(
    DEFAULT_MAX_REQUEST_INPUT_TOKENS
    * CHARS_PER_TOKEN_ESTIMATE
    * REQUEST_INPUT_CHAR_SAFETY_RATIO
)
TOOL_RESPONSE_TRUNCATION_SUFFIX = "... (truncated: content too long)"


def message_content_as_str(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(content)


def truncate_message_content(
    content: Any,
    max_chars: int,
    suffix: str = TOOL_RESPONSE_TRUNCATION_SUFFIX,
) -> Any:
    if max_chars <= 0:
        max_chars = len(suffix)
    text = message_content_as_str(content)
    if len(text) <= max_chars:
        return content
    keep = max(0, max_chars - len(suffix))
    return text[:keep] + suffix


def resolve_request_input_char_budget(
    *,
    max_request_input_tokens: Optional[Any] = None,
    max_request_input_chars: Optional[Any] = None,
    fallback_tokens: Optional[Any] = None,
) -> int:
    """环节配置 > openai_api 全局默认；tokens 优先于 chars。"""
    for tokens in (max_request_input_tokens, fallback_tokens):
        if tokens is None:
            continue
        try:
            value = int(tokens)
            if value == 0:
                return 0
            if value > 0:
                return int(value * CHARS_PER_TOKEN_ESTIMATE * REQUEST_INPUT_CHAR_SAFETY_RATIO)
        except (TypeError, ValueError):
            continue

    if max_request_input_chars is not None:
        try:
            return max(0, int(max_request_input_chars))
        except (TypeError, ValueError):
            pass

    return DEFAULT_MAX_REQUEST_INPUT_CHARS


def estimate_request_overhead_chars(
    messages: List[Dict[str, Any]],
    *,
    truncatable_roles: Optional[set[str]] = None,
    tool_schemas: Optional[Sequence[Any]] = None,
    extra_overhead_chars: int = 0,
) -> int:
    """
    整次请求中不参与 tool/assistant 内容预算的部分（system/user/tools 定义等）。
    """
    if truncatable_roles is None:
        truncatable_roles = {"tool", "assistant"}

    overhead = max(0, int(extra_overhead_chars))
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") in truncatable_roles:
            continue
        overhead += len(message_content_as_str(msg.get("content")))

    if tool_schemas:
        overhead += len(json.dumps(list(tool_schemas), ensure_ascii=False))

    return overhead


def _total_content_chars(messages: List[Dict[str, Any]], roles: Optional[set[str]] = None) -> int:
    total = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if roles is not None and role not in roles:
            continue
        total += len(message_content_as_str(msg.get("content")))
    return total


def _role_message_count(messages: List[Dict[str, Any]], role: str) -> int:
    return sum(
        1 for msg in messages if isinstance(msg, dict) and msg.get("role") == role
    )


def _min_role_footprint_chars(messages: List[Dict[str, Any]], role: str) -> int:
    count = _role_message_count(messages, role)
    if count == 0:
        return 0
    return len(TOOL_RESPONSE_TRUNCATION_SUFFIX) * count


def apply_role_content_char_budget(
    messages: List[Dict[str, Any]],
    role: str,
    max_total_chars: int,
    *,
    log_prefix: str = "messages",
) -> List[Dict[str, Any]]:
    """按指定 role 的消息条数均分总字符预算并截断 content。"""
    role_count = _role_message_count(messages, role)
    if role_count == 0:
        return messages

    if max_total_chars <= 0:
        max_total_chars = _min_role_footprint_chars(messages, role)

    per_msg_budget = max(1, max_total_chars // role_count)
    truncated_count = 0
    truncated_chars = 0
    result: List[Dict[str, Any]] = []
    changed = False

    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != role:
            result.append(msg)
            continue

        original_len = len(message_content_as_str(msg.get("content")))
        if original_len > per_msg_budget:
            changed = True
            new_msg = dict(msg)
            new_msg["content"] = truncate_message_content(
                msg.get("content"),
                per_msg_budget,
            )
            truncated_count += 1
            truncated_chars += original_len - per_msg_budget
            result.append(new_msg)
        else:
            result.append(msg)

    if truncated_count > 0:
        logger.warning(
            "%s 截断了 %s 条 %s 消息 "
            "(budget=%s chars, per_msg=%s, dropped_chars≈%s)",
            log_prefix,
            truncated_count,
            role,
            max_total_chars,
            per_msg_budget,
            truncated_chars,
        )

    return result if changed else messages


def apply_trajectory_view_char_budget(
    messages: List[Dict[str, Any]],
    *,
    max_view_total_chars: int = DEFAULT_MAX_REQUEST_INPUT_CHARS,
    log_prefix: str = "messages",
) -> List[Dict[str, Any]]:
    """
    tool+assistant 共享字符预算；超限时先砍 tool，assistant 仅在仍超限时再砍。
    """
    shrink_roles = {"tool", "assistant"}
    if max_view_total_chars <= 0:
        min_total = (
            _min_role_footprint_chars(messages, "tool")
            + _min_role_footprint_chars(messages, "assistant")
        )
        if min_total <= 0:
            return messages
        max_view_total_chars = min_total

    total = _total_content_chars(messages, shrink_roles)
    if total <= max_view_total_chars:
        return messages

    tool_chars = _total_content_chars(messages, {"tool"})
    overflow = total - max_view_total_chars
    view = messages

    if tool_chars > 0:
        min_tool_chars = _min_role_footprint_chars(messages, "tool")
        if overflow < tool_chars:
            new_tool_total = max(min_tool_chars, tool_chars - overflow)
            view = apply_role_content_char_budget(
                view,
                "tool",
                new_tool_total,
                log_prefix=f"{log_prefix} (tool-first)",
            )
            final_total = _total_content_chars(view, shrink_roles)
            if final_total <= max_view_total_chars:
                logger.info(
                    "%s 预算内仅截断 tool: %s -> %s chars (cap=%s)",
                    log_prefix,
                    total,
                    final_total,
                    max_view_total_chars,
                )
                return view

        view = apply_role_content_char_budget(
            view,
            "tool",
            min_tool_chars,
            log_prefix=f"{log_prefix} (tool-min)",
        )

    final_total = _total_content_chars(view, shrink_roles)
    overflow = final_total - max_view_total_chars
    assistant_chars = _total_content_chars(view, {"assistant"})
    if overflow > 0 and assistant_chars > 0:
        min_asst_chars = _min_role_footprint_chars(view, "assistant")
        new_asst_total = max(min_asst_chars, assistant_chars - overflow)
        view = apply_role_content_char_budget(
            view,
            "assistant",
            new_asst_total,
            log_prefix=f"{log_prefix} (assistant-fallback)",
        )
        final_total = _total_content_chars(view, shrink_roles)

    logger.warning(
        "%s 总预算压缩: %s -> %s chars (cap=%s, tool-first)",
        log_prefix,
        total,
        final_total,
        max_view_total_chars,
    )
    return view


def apply_request_input_char_budget(
    messages: List[Dict[str, Any]],
    *,
    max_request_input_chars: int,
    tool_schemas: Optional[Sequence[Any]] = None,
    extra_overhead_chars: int = 0,
    log_prefix: str = "request",
) -> List[Dict[str, Any]]:
    """
    按整次请求字符上限截断：固定开销 = 非 tool/assistant 消息 + tools 定义 + extra_overhead_chars。
    剩余预算再交给 apply_trajectory_view_char_budget（优先截 tool）。
    """
    if max_request_input_chars <= 0:
        return messages

    overhead = estimate_request_overhead_chars(
        messages,
        tool_schemas=tool_schemas,
        extra_overhead_chars=extra_overhead_chars,
    )
    trajectory_budget = max(0, max_request_input_chars - overhead)

    logger.info(
        "%s 整请求预算: cap=%s chars overhead=%s chars trajectory_content=%s chars",
        log_prefix,
        max_request_input_chars,
        overhead,
        trajectory_budget,
    )

    return apply_trajectory_view_char_budget(
        messages,
        max_view_total_chars=trajectory_budget,
        log_prefix=log_prefix,
    )
