"""
根据当轮 mermaid 生成平实、贴切的研究题目 title_query。

输入：单条 mermaid（含结构与 check_points 结论）
输出：title_query
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def build_title_query_prompt(
    *,
    mermaid: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    metadata = metadata or {}
    _ = metadata
    mermaid = (mermaid or "").strip()
    if not mermaid:
        raise ValueError("mermaid 为空，无法生成 title_query")

    prompt = f"""以下 mermaid 刻画了我已完成 deep research 的**研究内容**与**结论**（节点描述研究过程与信息，check_points 是关键结论/证据要点）。

## 我的研究（mermaid）

{mermaid}

请给出**最平实、最贴切**的研究题目 `title_query`。`title_query` 应是信息型研究报告/课题会用的**一句话标题**（title），使得 mermaid 能够覆盖该题目的研究范围，而且不超出范围。不是问句，不是提纲，不是 checklist。

## 要求

- **标题体，不是问句**：名词短语或简短陈述性标题；不以 Why / How / What / Which 开头，不加问号。
- **必须保留公认的能够搜索到的研究主体（最重要）**：
  - 从 mermaid 的起源节点、上游事实节点和 check_points 中找出**具体的主要研究对象**（作品名、季集、人物全名、历史事件、地点+时间等），并写进标题。
  - **禁止**用绰号、子剧情线等不是被大众所知道的内容当研究主体。
- **信息型、去文学化**：
  - 用**中性、直白、可核对**的信息词；**禁止**文学化、修辞化、情绪化用词。
  - **禁止**把 check_point 改写成更「好看」的同义词；优先沿用 mermaid / check_points 里已有的**朴素名词**。
  - 读起来应像**研究课题名**，不像影评标题、散文标题或营销文案。
- **以推理结论定调**：在保留主体的前提下，概括最下游的推理型结论；不要写成事实检索 checklist。
- **比研究内容更短**：合并维度，不逐条枚举；通常 10–18 个英文词（含主体时可略长）。
- **不含解题路径，不含mermaid信息**：不写「先 A 后 B」、节点 ID、检索步骤。不要写节点A、B，也不要写这个，那个等指示代词。

## 输出前自检

- 读者读完标题，是否仍能知道研究的是**哪部作品/哪一集/哪个人物/哪件事**？若不能，补上主体。
- 是否出现隐喻、夸张形容词、情绪化评价？若有，改成中性信息词。
- 是否是一句高度凝练的话，概括了mermaid的研究内容。

## 输出格式（严格 JSON）

```json
{{
  "title_query": "string"
}}
```

最终的输出中的内容使用英文，现在输出结果：
"""

    return prompt
