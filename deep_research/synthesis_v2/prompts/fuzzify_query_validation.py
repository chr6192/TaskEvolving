"""
query_fuzzified 程序化校验（与 fuzzify_query_prompt 结构规则对齐）。

仅做可客观判定的结构化检查（长度、枚举等），
不调用 LLM 打分，也不判断语义对齐或子问是否遗漏。
"""

from __future__ import annotations

import re
from typing import List, Tuple

_MAX_LENGTH_RATIO = 0.70
_MIN_QUERY_LEN_FOR_RATIO_CHECK = 300
_MAX_COMMA_SEGMENTS = 3
_MAX_SENTENCES = 3
_MIN_LIST_ITEMS_FOR_CHECKLIST = 5

_ISSUE_EMPTY = "query_fuzzified 为空"
_ISSUE_ENUMERATION = "并列方面过多，可能有枚举情况"
_ISSUE_TOO_LONG = "表述仍偏长，不够简洁口语"
_ISSUE_TOO_MANY_SENTENCES = "句子过多，应合并为不超过三句的更自然问法"

_FEEDBACK_ISSUE_EN = {
    _ISSUE_EMPTY: "query_fuzzified is empty",
    _ISSUE_ENUMERATION: "Too many parallel facets; possible enumeration",
    _ISSUE_TOO_LONG: "Still too long; not concise or colloquial enough",
    _ISSUE_TOO_MANY_SENTENCES: "Too many sentences; merge into a more natural question",
}

# 显式「分面/方面」列举脚手架（含 including、from ... aspect、based on ... and 等）
_ASPECT_LISTING_PHRASES = (
    re.compile(r"\bincluding\b", re.I),
    re.compile(r"\bbased\s+on\b.{0,120}\band\b", re.I),
    re.compile(r"\bfrom\b.{0,80}\baspects?\b", re.I),
    re.compile(r"\bacross\b.{0,80}\baspects?\b", re.I),
    re.compile(r"\bin terms? of\b", re.I),
    re.compile(r"\bwith respect to\b", re.I),
    re.compile(r"\bwith regard to\b", re.I),
    re.compile(
        r"\b(?:at|on|from|in)\s+(?:the\s+)?(?:\w+\s+){0,3}(?:and|or)\s+(?:\w+\s+){0,3}(?:level|aspect|dimension|side|plane)s?\b",
        re.I,
    ),
    re.compile(
        r"\b(?:looking at|covering|considering)\b.{0,60}\b(?:aspect|dimension|angle|facet)s?\b",
        re.I,
    ),
    re.compile(
        r"\b(?:broader|wider|larger)\b.{0,50}\b(?:history|context|background|landscape)\b",
        re.I,
    ),
    re.compile(
        r"\bconsidering\b.{0,180}\b(?:and|as well as)\b.{0,120}\b(?:broader|wider|history|background|context|roles?\s+behind)\b",
        re.I,
    ),
    re.compile(r"\broles?\s+behind\b", re.I),
    re.compile(
        r"\b(?:creative|composition|writing)\b.{0,40}\b(?:and|or)\b.{0,40}\b(?:perform|sing|vocal)\w*\b",
        re.I,
    ),
    re.compile(
        r"\b(?:in|against)\s+the\s+(?:context|backdrop|background)\s+of\b",
        re.I,
    ),
)

# 在 from/including 等来源动词后的三项串列：more from X, Y, or Z
_FACET_SERIAL_LIST = re.compile(
    r"\b(?:"
    r"(?:come|stem)\s+(?:mainly\s+|more\s+)?from|including|such\s+as|across|"
    r"weigh|evaluate|consider|covering|looking\s+at"
    r")\b[^.?]{0,180}[^,]+,\s*[^,]+,\s*(?:and|or)\s+",
    re.I,
)


def _sentence_count(text: str) -> int:
    parts = [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]
    return len(parts)


def _comma_segment_count(text: str) -> int:
    return len([p for p in re.split(r",", text) if p.strip()])


def _parallel_list_item_count(text: str) -> int:
    parts = re.split(r",|\band\b|—", text, flags=re.I)
    return len([p for p in parts if len(p.strip()) > 3])


def _has_aspect_listing_phrase(text: str) -> bool:
    if _FACET_SERIAL_LIST.search(text):
        return True
    return any(pattern.search(text) for pattern in _ASPECT_LISTING_PHRASES)


def _has_enumeration_issue(text: str) -> bool:
    """逗号分段过多、并列项过多，或含方面列举脚手架短语。"""
    if _has_aspect_listing_phrase(text):
        return True
    if _comma_segment_count(text) > _MAX_COMMA_SEGMENTS:
        return True
    return _parallel_list_item_count(text) >= _MIN_LIST_ITEMS_FOR_CHECKLIST


def collect_fuzzify_issues(
    *,
    query: str,
    query_fuzzified: str,
) -> List[str]:
    """返回所有校验问题；空列表表示通过。"""
    query = (query or "").strip()
    fuzz = (query_fuzzified or "").strip()
    issues: List[str] = []

    if not fuzz:
        issues.append(_ISSUE_EMPTY)
        return issues

    if _has_enumeration_issue(fuzz):
        issues.append(_ISSUE_ENUMERATION)

    if len(query) > _MIN_QUERY_LEN_FOR_RATIO_CHECK:
        ratio = len(fuzz) / len(query)
        if ratio > _MAX_LENGTH_RATIO:
            issues.append(_ISSUE_TOO_LONG)

    sentences = _sentence_count(fuzz)
    if sentences > _MAX_SENTENCES:
        issues.append(_ISSUE_TOO_MANY_SENTENCES)

    return issues


def validate_fuzzify_output(
    *,
    query: str,
    query_fuzzified: str,
) -> None:
    issues = collect_fuzzify_issues(query=query, query_fuzzified=query_fuzzified)
    if issues:
        raise ValueError("; ".join(issues))


def _issues_for_feedback(issues: List[str]) -> str:
    return "; ".join(_FEEDBACK_ISSUE_EN.get(issue, issue) for issue in issues)


def _build_remediation_hints(issues: List[str]) -> str:
    hints: List[str] = []
    joined = "; ".join(issues)

    if _ISSUE_ENUMERATION in joined:
        hints.extend([
            "Remove aspect scaffolding: `including`, `in terms of`, `based on ... and`, `at ... levels`, `from ... aspect`, `considering ... and ... broader history`, `roles behind`.",
            "Do not restate background/context axes (creative roles + broader music history); keep only the main object.",
            "Do not keep 2+ parallel facets (story and tone, legal and practical); use umbrella phrasing such as what mainly drives X.",
            "Drop decorative style adjectives (character-driven, shifting tone) when the core ask stays complete.",
            "Read-aloud must sound like a casual user question, not a research rubric.",
        ])

    if _ISSUE_TOO_LONG in joined:
        hints.append(
            "Compress further: drop modifiers, parallel lists, and method words; keep only the core ask."
        )

    if _ISSUE_TOO_MANY_SENTENCES in joined:
        hints.append("Merge into at most three sentences, preferably one or two.")

    if not hints:
        return ""

    bullet = "\n".join(f"- {h}" for h in hints)
    return f"You MUST:\n{bullet}\n"


def build_fuzzify_retry_feedback(
    issues: List[str],
    *,
    language: str = "en",
) -> str:
    _ = language
    joined = _issues_for_feedback(issues)
    remediation = _build_remediation_hints(issues)
    return (
        "## Previous output failed automatic validation — rewrite\n\n"
        f"Issues: {joined}\n\n"
        f"{remediation}"
    )


def is_fuzzify_output_valid(
    *,
    query: str,
    query_fuzzified: str,
) -> Tuple[bool, List[str]]:
    issues = collect_fuzzify_issues(query=query, query_fuzzified=query_fuzzified)
    return not issues, issues
