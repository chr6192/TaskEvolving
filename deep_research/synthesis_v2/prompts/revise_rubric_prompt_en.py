"""
Reviser stage-2 prompt builder — English version.

Stage 2 goals:
- Take query and mermaid;
- Produce `query_fuzzified` and rubric.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def build_revise_rubric_prompt(
    *,
    query: str,
    mermaid: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    metadata = metadata or {}
    _ = metadata
    query = (query or "").strip()
    mermaid = (mermaid or "").strip()

    prompt = f"""You are an assistant that generates a fuzzified user question draft and evaluation rubric from a query and mermaid.

Based on the given query and mermaid, output:
1. `query_fuzzified`: a fuzzified version of the query (shorter, more colloquial, fewer constraints);
2. `rubric`: evaluation criteria aligned with the query.

## Known inputs

- query:
{query}

- mermaid (structure + nodes):
{mermaid}

## Task 1: generate `query_fuzzified`

Rewrite the query to produce a simpler, shorter, more direct, and easier-to-understand formulation of the same question, while preserving semantic alignment and not dropping sub-questions. Requirements:

- **Simplify the question**: remove professional terms and methods; do not restate mermaid nodes, edges, order, or execution details.
- **Fuzzify constraints**: delete or soften process constraints, step hints, fine-grained boundaries, scoring conditions, source requirements, and other additional limits; do not add new constraints absent from the query.
  - Delete aspect constraints: remove A, B, C from "from aspects A, B, C", e.g. rewrite "Please analyze the reason for X by combining algorithm mechanisms, special status, and content change" by omitting "combining algorithm mechanisms, special status, and content change", keeping only "analyze the reason for X".
  - Fuzzify process structure: the query may contain several progressively layered sub-questions; analyze the relations among them and embed dependent sub-questions into the main question. E.g. in "Who has the most followers on platform X in 2026? Why can their follower count lead other top users by a large margin?", hide the sub-question "Who has the most followers on platform X in 2026?" and obtain "Analyze why the top-followed account on platform X in 2026 can lead other top users by a large margin in follower count?"
- **Semantic alignment, no dropped questions**: for query "Why are growth plates prone to injury while enabling height growth? Please explain the internal causes of this dual attribute from tissue structure and physiological regulation, and explain how these biological characteristics specifically guide clinical injury recognition, treatment principles, and follow-up strategy.", you may simplify to "Explain from internal causes why growth plates both enable height and are especially injury-prone, and explain what clinical guidance these characteristics provide?" — but you must NOT simplify to "Growth plates both enable height and are especially injury-prone — what clinical guidance do these characteristics provide?" because the sub-question "explain from internal causes why growth plates both enable height and are especially injury-prone" was dropped.
- **Expression form**: a complete, natural user question, at most 3 sentences, preferably one main question that implicitly contains all sub-questions in the query; colloquial, plain, low jargon.

## Task 2: based on the query, generate one criterion for each node in the mermaid to judge answer quality.

Each criterion includes multiple verifiable checkpoints with the following fields:
- `node_id`: must match the corresponding `node_id` in `mermaid.nodes` **exactly** (case-sensitive). E.g. if the structure node is `A`, write `"A"`; do not use aliases like `node_a`.
- `description`: use the action+object text of that node in structure to explain what is being judged.
- `check_points`: reuse that node's `check_points` from mermaid.nodes; you may lightly adjust wording but must not invent new facts; used to explain how to judge.

Requirements:
- Concrete and directly judgeable:
  - Each item must be mechanically judgeable and executable; avoid subjective words such as comprehensive / deep / high-quality. Content must be specific, sourced from known information, non-fabricated, and must be verifiable.
  - Do not mention mermaid node-related information (e.g. "whether it depends on node A", or "node A's output is node B's input"); for the agent being evaluated, mermaid does not exist — replace with judgments on concrete content.
  - The judgment focus should be "whether the information holds, whether evidence supports it, whether reasoning is consistent", not "whether a specific retrieval action / click action / cross-verification step was performed".
- Modify content according to how each node serves the query; remove out-of-scope judgments; do not be overly strict especially for substitutable evaluation content:
  - Unless the question itself explicitly requires it, do not require specific tools, search engines, URLs, tool calls, or retrieval steps, because any search path may obtain the same information.
  - Unless terminology must appear, do not require specific keywords or fixed phrasing, because different expressions can convey the same meaning — judge meaning, not expression.
  - Unless a specific example must appear, allow different but reasonable examples. If an example is needed, write "provide at least one relevant instance or evidence".
  - Unless the question explicitly requires comparing source credibility, do not write rigid scoring items such as "cross-verify across how many databases / prioritization workflow"; evaluate the evidence sufficiency and logical consistency of the final conclusion.
  - If a node no longer helps answer the query, generate an empty criterion that still includes node_id, description, and other fields, but with empty `check_points`.

## Output format (strict JSON)

```json
{{
  "query_fuzzified": "string",
  "rubric": {{
    "query_id": "string",
    "criteria": [
      {{
        "node_id": "A",
        "description": "string",
        "check_points": ["string"]
      }}
    ]
  }}
}}
```

Now output your final result:
"""

    return prompt


def build_leaf_query_prompt(
    *,
    stage1_query: str,
    sub_mermaid: str,
    sub_rubric: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Leaf subgraph scenario: legacy interface."""
    _ = sub_rubric
    return build_revise_rubric_prompt(
        query=stage1_query,
        mermaid=sub_mermaid,
        metadata=metadata,
    )
