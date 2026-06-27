"""
Reviser stage-2 prompt builder — English version.

Stage 2 goals:
- Take the newly output mermaid and trajectory;
- Based on mermaid increments and mineable information in the trajectory, produce next-round `query`.
- `query_fuzzified` is produced separately by fuzzify_query_prompt.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


def build_revise_rubric_prompt(
    *,
    mermaid: str,
    traj_k: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    metadata = metadata or {}
    _ = metadata
    mermaid = (mermaid or "").strip()
    traj_k_str = json.dumps(traj_k, ensure_ascii=False, indent=2)

    prompt = f"""You are an assistant that generates next-round task questions from mermaid and the trajectory.

Based on the given mermaid and trajectory, output `query`: the next-round expanded task question.

## Known inputs

- mermaid (generated in the previous stage, structure + nodes):
{mermaid}

- trajectory:
{traj_k_str}

## Your workflow

1. Compare the previous-round mermaid with the expanded mermaid this round; identify new nodes, new check_points, and structural incremental changes.
2. Together with the trajectory, find information in the mermaid that has not yet been deeply mined but has evidence in the trajectory to continue deepening.
3. Based on the above increments, generate the next-round `query`.

## Generate `query`

From the expanded full mermaid, combined with incremental information, generate a new `query`:
  - Fuzzification: generate a fuzzified question for the full mermaid task — omit concrete reasoning paths and detailed information, keeping only the core problem to solve. The query should steer the answerer along the expanded mermaid direction, but must not restate mermaid node labels, edges, step order, or solution modules, avoiding translating the reasoning path into a directly copyable operation checklist.
    - Must be a single-task expression: keep only 1 main goal + at most 10 key constraints. Do not develop multiple independent questions in parallel. You may include 1–3 logically linked questions (progressive, deepening, synthetic), but not a set of mutually independent (or weakly related) questions. Use constraints to prevent the question from being too open-ended and to steer the answerer along the mermaid direction.
    - State broadly which aspects to address and what task to solve; remove redundant, overly specific, factual wording:
      - Avoid checkpoint-level specifics: for node A[query the strait between Madagascar and mainland Africa] with checkpoint ["The strait between Madagascar and mainland Africa is the Mozambique Channel"], do not ask "What are the spatial scale and natural characteristics of the Mozambique Channel?" — ask "What are the characteristics of the strait between Madagascar and mainland Africa?"
      - Hide concrete examples: rewrite "Compare Luna 9 (99kg landing mass, no ascent stage), Surveyor 1 (one-way engineering verification) and Luna 15 (crashed during Apollo 11 stay) on technical parameters and mission outcomes" as "Compare prior lunar mission technology and outcomes".
      - Hide decorative adjectives: rewrite "How can its follower count form an order-of-magnitude gap over other top users?" as "How can its follower count form a large gap over other top users?"
      - Hide depth information; use simple, easy-to-understand words: rewrite "Please synthesize recommendation algorithm weighting, structural privileges in non-content distribution, and audience reconstruction from content change during government tenure" as "Please analyze from recommendation mechanisms, structural privileges, and content change".
    - Without over-loosening constraints, rewrite in language close to ordinary users: do not preset conclusions in the query (e.g. specifying a shop must be high-risk) or preset fixed scores/thresholds (e.g. "2.4 points", ">8 points"). Remove specific terminology, formal tone, and professional methods; avoid academic phrasing such as "temporal coupling / dynamic calibration / cross-modal / semantic discrimination / tiered verification logic / final adjudication criteria"; use plain, simple, everyday language that is directly understandable.
    - Mermaid-invisible: the query must be independent of mermaid and trajectory, and is used to test whether a model can answer the task depicted by the mermaid. Therefore the query must not use referential phrasing such as "based on prior output / as above / as stated / previous section / previous round / final version / continue last time / according to node A".

  - Guiding question: on top of fuzzifying the mermaid into a question, also guide the agent toward tasks mentioned in the mermaid (must be existing nodes, not invented tasks) that have not yet been deeply explored, in order to generate a more difficult task that is concrete, reasonable, and meaningful.
    - **Trajectory-grounding hard constraint (overrides “guide deeper”)**:
      - Every sub-question, analysis dimension, and constraint aspect in `query` must be traceable to evidence in trajectory `tool_response`, or to content already retrieved and discussed in the trajectory query/reason/final answer. Concepts/entities/relations that appear only incidentally in a `tool_response` snippet but were not retrieved deeply and were not used in the answer **must not** appear in `query`.
      - If the trajectory only completed shallow retrieval — fact enumeration, list compilation, episode/entry listing, cast pairings, etc. — and did **not** analyze meaning, impact, patterns, centrality, or cross-work connections, then `query` may **only fuzzify**; do **not** add guiding analysis such as “what do these add / what is the overall pattern / how central / connections with X”.
      - Guiding analysis is allowed only when the trajectory already contains analyzable threads to deepen — e.g. the answer compared two sides but did not explain why, or mentioned a relation but did not expand it. Do not invent analysis frameworks absent from both trajectory and mermaid.
      - Counterexample: the trajectory only identified which Arrow episodes feature Supergirl, which crossover each belongs to, and a brief finale cameo; do **not** ask “what those appearances add to Arrow and the Arrowverse, how central Supergirl is, or the overall connection pattern involving Legends of Tomorrow” unless the trajectory already retrieved and discussed evidence for that analysis.
    - Meaningful questions: prefer "evidence-constrained deep analysis" — such as attribution, explanation, conditional constraints, comparison, etc. — requiring conclusions supportable by checkable evidence already in the mermaid. Including but not limited to:
      - Ask the agent to explain / verify / deepen a node
      - Ask the agent to synthesize / reason / compare across several nodes
    - Meaningless questions forbidden:
      - Cross-checking the same fact with multiple pieces of evidence: "repeatedly verify / compare the same fact across multiple searches/databases and give prioritization rules / confirm via multiple aspects" is a shallow task and meaningless.
      - Over-autonomous questions that do not depend on evidence or reasoning: "design methodology / design system / design mechanism / build framework / output manual" and similar abstract product-oriented tasks are over-autonomous and meaningless.

  Special note: **If there are no nodes left to deepen, or the question is already deep enough, and further guiding would become multiple parallel independent questions or meaningless, you may fuzzify only without adding a guiding question.**

## Output format (strict JSON)

```json
{{
  "query": "string"
}}
```

## Self-check before output
- If the query contains "based on prior output / as above / previous section / previous round / continue last time / final version", rewrite as a self-contained version.
- If the query contains fixed scores / fixed thresholds / coined abbreviation systems / preset conclusions / single-sentence extremely long checklist stacking, rewrite into a verifiable, executable version.

Now output your final result:
"""

    return prompt


def build_leaf_query_prompt(
    *,
    stage1_query: str,
    sub_mermaid: str,
    sub_rubric: Dict[str, Any],
    traj_k: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Leaf subgraph scenario: legacy interface."""
    _ = stage1_query
    _ = sub_rubric
    return build_revise_rubric_prompt(
        mermaid=sub_mermaid,
        traj_k=traj_k or {},
        metadata=metadata,
    )
