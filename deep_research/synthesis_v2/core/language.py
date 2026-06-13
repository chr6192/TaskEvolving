"""Language switch for synthesis_v2 prompt / evaluation routing."""

from __future__ import annotations

from typing import Any, Literal

LanguageCode = Literal["cn", "en"]

FACTUAL_NODE_TYPES = frozenset({"事实型", "Factual"})
REASONING_NODE_TYPES = frozenset({"推理型", "Reasoning"})
VALID_NODE_TYPES = FACTUAL_NODE_TYPES | REASONING_NODE_TYPES

TRAJECTORY_PROMPT_CN = (
    "deep_research/synthesis_v2/prompts/trajectory_generate_prompt_general_search.py"
)
TRAJECTORY_PROMPT_EN = (
    "deep_research/synthesis_v2/prompts/trajectory_generate_prompt_general_search_en.py"
)

SEED_FILE_CN = "deep_research/synthesis_v2/data/nq_ch.jsonl"
SEED_FILE_EN = "deep_research/synthesis_v2/data/nq_open_seed_queries_1000.jsonl"

_LANGUAGE_PAIR_ALIASES: dict[str, tuple[LanguageCode, LanguageCode]] = {
    "cn-en": ("cn", "en"),
    "cn_en": ("cn", "en"),
    "zh-en": ("cn", "en"),
    "zh_en": ("cn", "en"),
    "en-cn": ("en", "cn"),
    "en_cn": ("en", "cn"),
}

_OUTPUT_INSTRUCTIONS: dict[tuple[LanguageCode, LanguageCode, str], str] = {
    ("cn", "en", "reviser"): """
## 输出语言（硬性要求）
- 以上任务说明为中文，但你的**最终输出必须使用英语**。
- JSON 中所有字符串字段（含 `mermaid.structure` 节点文本、`check_points`、`query`、`query_fuzzified`、`description` 等）一律使用英语。
- `node_type` 必须写 `"Factual"` 或 `"Reasoning"`（不要使用 `"事实型"` / `"推理型"`）。
- 保持 JSON 结构与字段名不变；仅内容语言为英语。
""".strip(),
    ("cn", "en", "trajectory"): """
# 输出语言（硬性要求）
- 以上 system 提示词为中文，但你的**全部回答内容（含 `<answer>` 内正文）必须使用英语**。
- 工具调用参数中的自然语言描述（如搜索关键词）也应使用英语，除非用户问题本身为中文且必须保留原文。
""".strip(),
    ("en", "cn", "reviser"): """
## Output language (hard requirement)
- The instructions above are in English, but your **final output must be in Chinese**.
- All string fields in JSON (`mermaid.structure` node labels, `check_points`, `query`, `query_fuzzified`, `description`, etc.) must be in Chinese.
- `node_type` must be `"事实型"` or `"推理型"` (do not use `"Factual"` / `"Reasoning"`).
- Keep JSON structure and field names unchanged; only the content language is Chinese.
""".strip(),
    ("en", "cn", "trajectory"): """
# Output language (hard requirement)
- The system prompt above is in English, but your **entire answer (including text inside `<answer>`) must be in Chinese**.
""".strip(),
}


def normalize_language(language: str | None) -> LanguageCode:
    if not language:
        return "cn"
    lang = str(language).strip().lower()
    if lang in {"en", "english"}:
        return "en"
    if lang in {"cn", "zh", "chinese", "zh-cn"}:
        return "cn"
    return "cn"


def parse_language_pair(
    language: str | None,
    output_language: str | None = None,
) -> tuple[LanguageCode, LanguageCode]:
    """
    解析提示词语言与模型输出语言。

    简写：
    - ``cn-en`` / ``cn_en`` / ``zh-en`` → 中文提示词 + 英语输出
    - ``en-cn`` / ``en_cn`` → 英文提示词 + 中文输出

    若 ``output_language`` 显式传入，则覆盖简写中的输出侧。
    """
    raw = str(language or "cn").strip().lower()
    if raw in _LANGUAGE_PAIR_ALIASES:
        prompt_lang, default_output_lang = _LANGUAGE_PAIR_ALIASES[raw]
    else:
        prompt_lang = normalize_language(raw)
        default_output_lang = prompt_lang

    if output_language is not None and str(output_language).strip():
        output_lang = normalize_language(output_language)
    else:
        output_lang = default_output_lang
    return prompt_lang, output_lang


def format_language_pair(prompt_language: LanguageCode, output_language: LanguageCode) -> str:
    if prompt_language == output_language:
        return prompt_language
    return f"{prompt_language}→{output_language}"


def is_reasoning_node_type(node_type: str | None) -> bool:
    return (node_type or "").strip() in REASONING_NODE_TYPES


def default_seed_file_for_language(language: str | None) -> str:
    return SEED_FILE_EN if normalize_language(language) == "en" else SEED_FILE_CN


def resolve_seed_file(
    language: str | None,
    input_path: str | None = None,
    output_language: str | None = None,
) -> str:
    """未显式指定 --input 时，按输出语言返回默认 seed jsonl 路径。"""
    if input_path:
        return input_path
    _, output_lang = parse_language_pair(language, output_language)
    return default_seed_file_for_language(output_lang)


def append_output_language_instruction(
    prompt: str,
    *,
    prompt_language: LanguageCode,
    output_language: LanguageCode,
    context: str = "reviser",
) -> str:
    """当提示词语言与输出语言不一致时，在 prompt 末尾追加输出语言硬性要求。"""
    if prompt_language == output_language:
        return prompt
    instruction = _OUTPUT_INSTRUCTIONS.get((prompt_language, output_language, context))
    if not instruction:
        return prompt
    return f"{prompt.rstrip()}\n\n{instruction}\n"


def apply_language_to_config(
    config: Any,
    language: str | None,
    output_language: str | None = None,
) -> str:
    """
    根据语言开关切换提示词路径、输出语言指令与评估语言。

    - ``language``：提示词模板语言（cn | en | cn-en 等）
    - ``output_language``：可选，模型输出语言；默认读取 config，再与 ``language`` 对齐

    返回便于日志展示的语言对字符串，如 ``cn`` 或 ``cn→en``。
    """
    if output_language is None:
        output_language = config.get("complexity_exploration", {}).get("output_language")

    prompt_lang, output_lang = parse_language_pair(language, output_language)

    if not config.get("complexity_exploration"):
        config.complexity_exploration = {}
    config.complexity_exploration.language = prompt_lang
    config.complexity_exploration.output_language = output_lang

    if not config.get("generation"):
        config.generation = {}
    config.generation.trajectory_prompt_template_path = (
        TRAJECTORY_PROMPT_EN if prompt_lang == "en" else TRAJECTORY_PROMPT_CN
    )
    config.generation.prompt_language = prompt_lang
    config.generation.output_language = output_lang

    if not config.get("reward_function"):
        config.reward_function = {}
    if not config.reward_function.get("init_params"):
        config.reward_function.init_params = {}
    config.reward_function.init_params.language = output_lang

    return format_language_pair(prompt_lang, output_lang)
