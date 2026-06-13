"""
Reviser stage-1 (expand task) prompt builder — English version.

Stage 1 goals:
- Expand the task mermaid from the trajectory;
- Generate the next-round expanded `query` from the expanded mermaid;
- `query_fuzzified` and rubric are produced in stage 2 from the stage-1 query.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


def build_revise_expand_task_prompt(
    *,
    mermaid_k: Optional[str],
    mermaid_nodes: Optional[list[Dict[str, Any]]] = None,
    traj_k: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    metadata = metadata or {}
    _ = metadata
    _ = mermaid_nodes
    traj_k_str = json.dumps(traj_k, ensure_ascii=False, indent=2)
    mermaid_block = (mermaid_k or "").strip()
    if mermaid_block:
        mermaid_section = f"""- Previous-round full mermaid (includes structure and nodes):
{mermaid_block}"""
    else:
        mermaid_section = "- Previous-round mermaid: (empty — build a new one from the trajectory this round)"

    prompt = f"""You are an assistant that expands the search scope and depth of deep-research tasks based on a trajectory, in order to construct more valuable tasks.
You must return a `mermaid` object and a `query`.

Definitions:
- `mermaid` describes the full execution process of a deep-research task (NOT a depiction of the trajectory itself). It includes:
  - `mermaid.structure`: a DAG mermaid flowchart structure (no back-edges or cycles), containing only node connections and node labels. Each node label captures a single **action + object**, stated freely without results, e.g. `A["Search Beijing weather"]`, `B["Estimate Zijin Mining valuation"]`. Both node labels and logical relations must be grounded in evidence; do not invent unsupported steps.
  - `mermaid.nodes`: structured node metadata — detailed information for every node in `mermaid.structure`. You MUST output one `mermaid.nodes` record for **every** node in `mermaid.structure`, including:
    - `node_id`: must match the node ID in `mermaid.structure` **exactly** (case-sensitive). E.g. if structure has `A["Search Beijing weather"]`, then `node_id` is `"A"`. Do not lowercase, add `node_` prefixes, or use aliases.
    - `node_type`: must be exactly `"Factual"` or `"Reasoning"`. Distinguish **strictly** by whether the information is directly verifiable:
      - Factual: information with a clear source or evidence that can stand without logical derivation; only information from `tool_response` may be labeled Factual.
      - Reasoning: information obtained by analyzing existing information (causal, inductive, comparative, predictive, etc.), but still non-fabricated and must originate from content already in the trajectory.
    - `check_points`: the **concrete results/conclusions** for this node, used to corroborate the node content. At least one per node; multiple allowed. Examples:
      - Factual node `A["Search Beijing weather"]`: `["Weather is sunny"]`
      - Reasoning node `B["Estimate Zijin Mining valuation"]`: `["Absolute valuation yields 32 yuan"]`
      - Reasoning node `C["Analyze reasons for gold price rise"]`: `["Fed rate cuts weakened USD, lifting gold", "Severe inflation reduced purchasing power", "Geopolitical conflict increased safe-haven demand for gold"]`
    - `source_type`: source type of the node information; must be exactly one of:
      - `previous_mermaid`: from a node in the input mermaid (unchanged / no new node information added this round).
      - `tool_call_response`: new content from tool calls and responses in this round's trajectory.
      - `query_or_reason_or_answer`: new content from query / reason / final answer in this round's trajectory.
    - `source_note`: short description of source + evidence (must be locatable in the input; no fabrication):
      - if `previous_mermaid`, write `from_previous_mermaid` plus the inheritance basis.
      - if `tool_call_response`, write `tool_call_id=<id>` plus tool name/keywords and key evidence.
      - if `query_or_reason_or_answer`, specify fragment location (e.g. `query` / `reason` / `final_answer`) plus key evidence.
- `query`: a task question generated from the mermaid.

## Mermaid requirements
- Node granularity: whether Factual or Reasoning, one node corresponds to at most a single operation — e.g. one object searched in one tool call, or one analytic question addressed in one reasoning step. Do not merge multiple operations into one node. There is no limit on node count; be as complete as possible, without fabrication or omission.

- **Failed/meaningless nodes forbidden (hard rule)**:
  - Any result that did not obtain valid information must NOT become a node, including but not limited to: request failure, scrape failure, unrendered dynamic placeholders, empty results, anti-bot blocks, timeouts, etc.
  - Such failure information may only be used internally to judge that a source is unavailable; it must not appear as structural node text, nor be assigned a standalone rubric criterion.
  - E.g. content like "failed to scrape Sina page / returned @now@ placeholder" is forbidden as a mermaid node output.

- **Meta nodes forbidden (hard rule)**:
  - Do not output workflow/meta-cognitive semantics in final nodes, such as: identify information gaps, choose deepening direction, plan next step, generate expanded task, build task framework, output final version, closed-loop playbook, etc.
  - You may internally perform gap identification / direction selection to decide mermaid expansion strategy, but such content is for internal reasoning only and must not appear as node labels. If you did perform gap identification internally, you must translate it into domain task nodes before output (e.g. "verify store 2026 operating status", "compare review tags over the last 3 months"), rather than outputting "identify gap" itself.
  - Node labels may only use concepts already present and alignable in this round's tool_response, query, or previous mermaid; do not introduce new thematic dimensions or coined terms. Prefer vocabulary already in the evidence; avoid artificially difficult terminology for pseudo-depth.
  - Structure node text must be concrete action + object; it must not describe "how to keep thinking / how to rewrite query". Evidence and results belong in `check_points`.
  - Forbid nodes with definitive answers but no upstream node support, such as abstract artifact nodes like "design methodology / build system / design mechanism / produce manual / build framework".

- Edge direction: edges must reflect actual information and information flow; multiple sources or sinks are allowed. Determine the logical relation between each node and its related nodes, then organize edges accordingly:
  - Derivation = vertical organization: a downstream node directly uses upstream information. Only organize vertically when the downstream node is definitively derived from upstream (the downstream node cannot be linked without upstream); otherwise treat as parallel:
    - E.g. node `B["Query shareholder count change"]` requires knowing which stock, so `A["Identify stock name"] --> B["Query shareholder count change"]` is valid vertical organization; `A`'s check_points include `["Stock is Zijin Mining"]`, `B`'s check_points include `["Shareholder count 1.252M as of 2026-03-31, up from prior period"]`.
  - Complementary (parallel) = horizontal organization: parallel nodes represent different aspects of solving the same task; even if the operation object is the same, they can be completed independently without depending on each other.

- **Connectivity & completeness (hard rule; must satisfy before output)**:
  - No dangling nodes: every node in the graph must appear in at least one edge (in-edge or out-edge).
  - No unsupported Reasoning nodes: every **Reasoning** node must have at least one in-edge, and in-edge sources must be the factual/prior reasoning nodes its inference depends on; forbid Reasoning nodes with no upstream that connect directly to the sink.
  - No dead-end intermediates: except **Factual leaf** nodes that only collect information and are not consumed downstream, any node whose information will be used in later analysis must have at least one out-edge pointing to a downstream node that actually consumes that result.
- Consistency between edges and `mermaid.nodes`: every `node_id` in `mermaid.nodes` must appear in `mermaid.structure`. Any upstream node cited in `source_note` must have a corresponding data-flow path in `mermaid.structure` (direct `X --> Y`, or via intermediate nodes `X --> ... --> Y`). If the note says "based on A/B/C", then A, B, and C must all be able to reach that node.
  - When extending an existing graph, fix edges first: after inheriting previous-round nodes, if existing nodes lack in/out edges, are inconsistent with `source_note`, analytic nodes do not feed into synthesis, Reasoning nodes lack in-edges, etc., repair and complete the mermaid first rather than only stacking new nodes. Every newly added node must include its upstream and downstream edges at the same time; forbid adding nodes without updating connectivity.

## Known inputs

{mermaid_section}

- Structured representation of the tool-call trajectory (structure and main content shown as-is):
{traj_k_str}

# Your workflow
1. If the previous-round mermaid is empty, first extract key nodes and logical relations from the trajectory, and generate a `mermaid` object that satisfies all requirements above; if not empty, skip this step and proceed directly to revise and expand the existing full mermaid.

2. Revise mermaid. Based on the user question in the trajectory, **delete nodes in the existing mermaid that are irrelevant to that user question**. Keep only the minimal sufficient nodes needed to complete the task; synchronously update mermaid structure and node information (especially check_points — delete entries irrelevant to the final question) so that all nodes directly or indirectly serve the final question. Then reorganize the mermaid so nodes and logic are more reasonable and clear.

3. Expand mermaid. Analyze information present in the trajectory but not yet written into the mermaid (do not re-add node information deleted in step 2), and produce a more complex mermaid:
  - Prefer depth expansion on the main path: make the whole task harder in depth, i.e. lengthen the path required to solve it. When extending vertically, prefer extending forward or backward rather than inserting middle nodes. The task after depth expansion may differ from before; focus on "small-step deepening" of one topic mentioned in the trajectory but not yet deep in the mermaid. You may change the overall task or existing nodes to achieve a more complex task, but forbid whole-graph rewrites. If the overall task changes, adjust the mermaid according to the expanded task and revise existing nodes and logic to preserve overall logical reasonableness and consistency. Add at most 2 depth nodes per round.
  - Horizontal expansion: around some node, supplement trajectory-related information absent from the mermaid; horizontal expansion generally does not change the overall task; you may add 0–10 nodes at once.
  - Refinement expansion: if a node label is unchanged but new verifiable content that serves the whole task has been added, you may add only related check_points to existing nodes.
  - Forbid meaningless expansion: every new node must bring incremental information. Zero incremental expansion or edits to existing nodes only are allowed if no high-value addition exists. Forbid meaningless nodes such as duplicate semantics (paraphrase counts as duplicate), or re-checking the same factual question with rephrased wording ("search again for the same fact", "verify message validity via another channel").
  - **Expansion principles**:
    - Expanded node information must come entirely from the trajectory; do not fabricate information; new nodes must maintain linkage with existing mermaid nodes.
    - At the end of each expansion round, check whether the expanded mermaid satisfies mermaid format and content requirements; repair if not.
    - Grounding is the first principle: the trajectory may contain hallucinations, so do not put information not obtained via retrieval in `tool_response` into Factual nodes. If new information conflicts with existing mermaid nodes or check_points, verify twice; otherwise, prefer omitting the node rather than blindly adding it.

4. Generate a new `query` from the expanded full mermaid:
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
    - Meaningful questions: prefer "evidence-constrained deep analysis" — such as attribution, explanation, conditional constraints, comparison, etc. — requiring conclusions supportable by checkable evidence already in the mermaid. Including but not limited to:
      - Ask the agent to explain / verify / deepen a node
      - Ask the agent to synthesize / reason / compare across several nodes
    - Meaningless questions forbidden:
      - Cross-checking the same fact with multiple pieces of evidence: "repeatedly verify / compare the same fact across multiple searches/databases and give prioritization rules / confirm via multiple aspects" is a shallow task and meaningless.
      - Over-autonomous questions that do not depend on evidence or reasoning: "design methodology / design system / design mechanism / build framework / output manual" and similar abstract product-oriented tasks are over-autonomous and meaningless.

  Special note: **If there are no nodes left to deepen, or the question is already deep enough, and further guiding would become multiple parallel independent questions or meaningless, you may fuzzify only without adding a guiding question.**

## Notes
- Grounding is the first principle throughout the entire process:
  - Strictly forbid adding any fabricated nodes or information: in both query and mermaid, do not introduce any undefined, non-commonly-known coined keywords (e.g. invented model names, abbreviations, scoring system names, method names). If widely known terms are used, they must come from `tool_response` verbatim or previous mermaid verbatim, and must be locatable in the evidence fragment pointed to by the corresponding `source_note`. Likewise, forbid outputting calculation/reasoning methods, numbers, formulas, or tier thresholds that have not been validated by standard methods (i.e. that could be obtained via query or search).
  - Do not deliberately package for depth or professionalism: forbid outputting any named entities, abbreviations, or numeric rules not in the evidence set; forbid "terminology upgrade" (rewriting ordinary evidence descriptions into more professional but absent terms, e.g. changing "concrete steps" to "SOP", packaging "compare" as "coupling analysis", packaging "verify truth" as "authenticity discrimination model"). Do not import AI-coined quoted vague adjectives from trajectory answers (e.g. "loss of irony", "sense of barrenness", "mythic historicism"); use plain, generic wording (e.g. "multi-dimensional risk assessment"); do not coin new proper nouns.

- Self-check before output:
  - Re-check all node labels, format, and granularity; ensure node content is concrete and compliant. Especially verify whether derivation edges strictly satisfy the definition "downstream node's premise is upstream node"; two tasks that can be completed independently should be defined as parallel.
  - If the query contains "based on prior output / as above / previous section / previous round / continue last time / final version", rewrite as a self-contained version.
  - If the query contains fixed scores / fixed thresholds / coined abbreviation systems / preset conclusions / single-sentence extremely long checklist stacking, rewrite into a verifiable, executable version.
  - If structure node text / check_points / query use "professional packaging" terms absent from evidence (or not widely defined and accepted), such as SOP, game theory, hedging, closed-loop mechanism, or quoted terms, replace with plain synonymous expressions from the evidence.
  - If query / structure node text / check_points use abstract product-oriented phrasing such as "design methodology / design system / design mechanism / build framework / execution manual", rewrite toward fact verification and analysis orientation.
  - If a term cannot be sourced in evidence, delete it and rewrite plainly; prefer being straightforward over deliberately sounding professional for depth.

## Output format
Your final output will be parsed as strict JSON:

```json
{{
  "mermaid": {{
    "structure": "string",
    "nodes": [
      {{
        "node_id": "string",
        "node_type": "Factual | Reasoning",
        "check_points": ["string", "string"],
        "source_type": "tool_call_response | query_or_reason_or_answer | previous_mermaid",
        "source_note": "string"
      }}
    ]
  }},
  "query": "string"
}}
```

Now output your final result:
"""

    return prompt
