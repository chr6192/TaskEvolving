"""
Reviser 第二阶段（rubric生成）prompt 构造器。

第二阶段目标：
- 固定使用第一阶段产出的 query；
- 生成 rubric 与 query_fuzzified；
- 不重写第一阶段结果。
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


def build_revise_rubric_prompt(
    *,
    stage1_mermaid: str,
    stage1_query: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    metadata = metadata or {}
    _ = metadata  # 保留参数接口，提示词中不注入
    stage1_mermaid = (stage1_mermaid or "").strip()
    stage1_query = (stage1_query or "").strip()

    stage1_result_str = json.dumps(
        {"mermaid": stage1_mermaid, "query": stage1_query},
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"""你是一个专门为复杂深度任务生成 rubric 和 query_fuzzified 的助手。

## 已知信息

- 任务query：
{stage1_query}

- 任务mermaid，这是解决该query的标准路径：
{stage1_result_str}

## 你的工作（第二阶段：rubric生成）

1. 基于第一阶段 `mermaid` + `query` 生成 `rubric`
  - Rubric 必须覆盖 mermaid 与 query 的关键目标、关键约束与依赖关系，语义范围与 `query` 一致。
  - 每个关键节点应映射为一个或多个可判定 criterion。
  - `rubric` 字段结构必须为：
    - `query_id` (string)
    - `criteria` (array of objects)
    - `metadata` (object)
  - `criteria` 每个对象字段与约束：
    - `id` (string): 3-64 字符，仅小写字母/数字/下划线。
    - `description` (string): 人类可读，明确该项判定目标与逻辑。
    - `weight` (number): [0, 1]。
    - `check_points` (string[]): 非空；每条必须具体、可验证、可机械判定，避免“全面/深入/高质量”等主观词。

2. 基于第一阶段 `query` 生成 `query_fuzzified`（用户视角）
  - `query_fuzzified` 必须只以第一阶段 `query` 为语义来源。
  - 改写为更像真实用户的一句话/两句话：口语化、自然，聚焦终极需求。
  - 并列任务不能丢失；有逻辑依赖时可隐去中间路径但保留最终诉求。
  - 具体实现途径必须模糊掉。
  - 硬性禁止步骤化语言（如“第一步/然后/最后/先…再…”）。
  - 硬性禁止过程枚举（如“从a、b、c方面/结合a、b、c来源”）。
  - 去掉过细例子、过程性修饰和过多限定词，保留最核心任务。

## 输出形态（严格 JSON）

- `rubric`: {{
    "query_id": str,
    "criteria": [
      {{
        "id": str,
        "description": str,
        "weight": float,
        "check_points": [str, ...]
      }},
      ...
    ],
    "metadata": {{...}}
  }}
- `query_fuzzified`: str

现在，输出你的最终结果：
"""

    return prompt

