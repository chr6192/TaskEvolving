"""
Rewrite query into a shorter, more colloquial query_fuzzified.

Input: a single query string
Output: query_fuzzified
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def build_fuzzify_query_prompt(
    *,
    query: str,
    metadata: Optional[Dict[str, Any]] = None,
    retry_feedback: Optional[str] = None,
) -> str:
    metadata = metadata or {}
    _ = metadata
    query = (query or "").strip()
    retry_block = ""
    if retry_feedback:
        retry_block = f"\n{retry_feedback.strip()}\n"

    prompt = f"""You are an assistant that rewrites a deep-research question into a shorter, more colloquial user-facing version.

Given a `query`, output `query_fuzzified` that preserves the same goal, semantic alignment, and all sub-questions. `query_fuzzified` should only ask the question itself, without constraints or solution pathways.

## Input query

{query}

## Rewrite rules

- **Semantic alignment, no dropped questions**: preserve the main object and every sub-question; do not drop causal or explanatory sub-goals when shortening.
- **Simplify the question**: remove professional terms and methods; do not restate execution details.
- **Width fuzzification (most important — most common failure)**:
  - "Do not drop sub-questions" means preserve semantic coverage of every goal in the query — **not** copy the query's parallel lists verbatim into `query_fuzzified`.
  - Delete facet/dimension constraints: remove A, B, C from "from aspects A, B, C" and keep only the core task.
  - **Comma budget**: split `query_fuzzified` on `,`; segment count must be **≤ 3**, strongly prefer **≤ 2**. Every extra comma-separated item is a checklist — replace with umbrella phrasing immediately.
  - **No serial listing (including pairs)**: do NOT use commas / semicolons / `and` / `or` / `as well as` / `including` / `such as` / `both ... and` to list **2+ parallel** analysis dimensions, plotlines, mechanisms, levels, conditions, or channels. Even "A and B" or "A, B, or C" must become one umbrella phrase + task verb (e.g. "what mainly drives X").
  - **Delete scaffold and background constraints**: drop `based on`, `depending on`, `in terms of`, `at the legal and practical level`, `from ... and ... aspects`, `considering ... and ...`, `in the context of`, `against the backdrop of`, `broader ... history`, etc. when they only enumerate angles, evidence, or paired "specific sub-factors + broader history/context" (e.g. creative/performing roles behind a theme song + broader film/TV music history). Keep only the core ask; background stays implicit.
  - **Delete decorative modifiers**: drop style/evaluative adjectives (freer, character-driven, shifting tone, nuanced, multifaceted) when the question stays complete without them.
  - **No entity roll call**: if the query names 5 characters, 5 mechanisms, or 5 release channels, **name none of them individually** in `query_fuzzified`; at most one collective noun + task verb.
  - Hide facet checklists without reducing task coverage; if it reads like a rubric / research outline / DAG node list, rewrite.
- **Process fuzzification**:
  - Delete process constraints, step hints, and solution pathways: `Focus on`, `Analyze by`, `Trace`, `Use evidence`, `Research`, `Investigate`, `when you separate`, and `Please compare` must not appear.
  - Hide solution order: embed layered sub-questions into one main question without exposing "first A, then B, then C".
- **Expression form**:
  - Prefer **1 sentence**, at most 2, never more than 3.
  - **Do not chain separate questions**: no `? ... ? ... ?` spray; at most **2 question marks**, merged into 1–2 sentences.
  - **Must be clearly shorter**: `len(query_fuzzified) / len(query) ≤ 0.70`. Longer queries require stronger compression.
  - Prefer one main question that implicitly contains all sub-questions; colloquial, plain, low jargon.

## Good vs bad examples

### Example 1 (width: facet checklist)

**query**:
When someone is picking their first apartment in a new city, what should they evaluate across commute time, monthly rent, neighborhood safety, nearby groceries, and building maintenance, and how should they balance cheaper rent against a longer commute?

**Bad** — still lists all five facets:
When picking a first apartment, what should renters weigh regarding commute time, monthly rent, neighborhood safety, nearby groceries, and building maintenance, and how should they balance rent against commute?

**Good**:
What should you look for when choosing your first apartment in a new city, and how do you balance cheaper rent against a longer commute?

### Example 2 (process + width)

**query**:
First identify which café on Main Street has the highest average rating this month. Then explain why it outperforms nearby competitors, and determine whether the gap comes mainly from drink quality, service speed, or price.

**Bad**:
Which café on Main Street has the highest rating this month? Then why does it outperform nearby competitors, and does the gap come mainly from drink quality, service speed, or price?

**Good**:
Why does the top-rated café on Main Street score higher than nearby competitors this month, and what mainly drives that advantage?

### Example 3 (width: multi-axis comma listing — high-frequency failure)

**query**:
Why is the 2007 film Fred Claus divisive, and does that stem more from its brother-conflict story, the lead performance, how its comedy-sentiment mix shapes Fred's redemption, critic-audience disagreement, or its holiday rewatch appeal?

**Bad** — 5 comma segments, mini-checklist:
Why is Fred Claus so divisive, and does that come more from its brother-conflict story or the lead performance, especially in how its comedy-sentiment mix shapes Fred's redemption, the critic-audience gap, and its holiday rewatch appeal?

**Good**:
Why is Fred Claus so divisive, and what mainly drives that split in reception?

### Example 6 (width: 2–3 facets must still merge)

**query**:
Why is the film so divisive, and does that stem more from its family-reconciliation story, shifting tone, or the leads' performances?

**Bad**:
Why is the film so divisive, and does that come more from its family-reconciliation story, shifting tone, or the leads' performances?

**Good**:
Why is the film so divisive, and what mainly drives that disagreement?

### Example 7 (drop "based on ..." scaffolding)

**query**:
Based on damage from the attack and subsequent recovery progress, when can the facility return to full operation?

**Bad**:
Based on damage from the attack and subsequent recovery progress, when can the facility return to full operation?

**Good**:
When can the facility return to full operation after the attack?

### Example 8 (drop level constraints and adjective stacks)

**query**:
At legal and practical levels, how does the show use a freer, more character-driven approach for its other major characters?

**Bad**:
At legal and practical levels, how does the show use a freer, more character-driven approach for its other major characters?

**Good**:
How does the show handle its other major characters?

### Example 9 (drop context scaffolding: sub-factors + broader history)

**query**:
Against the broader Pee-wee film and TV music history, and considering the different creative and performing roles behind its famous theme song, how should we understand the show's most famous theme song?

**Bad**:
How should we understand the show's famous theme song, considering the different creative and performing roles behind it and the broader Pee-wee film and TV music history?

**Good**:
How should we understand the show's famous theme song?

### Example 4 (width: character name roll call)

**query**:
What ultimately happened to Randal Holmes, Larry Connlee, Eddie Willis, Ian Glenn, Mika Willis, Travis Cain, Quan, Ramsey Walters, Kelly Isaacs, Joey Blackman, and other major characters in Tyler Perry's If Loving You Is Wrong, and who was confirmed dead, presumed dead, left unclear, or survived?

**Bad** — names 10+ characters:
What ultimately happened to the major and recurring characters in Tyler Perry's If Loving You Is Wrong, including Randal Holmes, Larry Connlee, Eddie Willis, Ian Glenn, Mika Willis, Travis Cain, Quan, Ramsey Walters, Kelly Isaacs, and Joey Blackman, and who was confirmed dead, presumed dead, left unclear, or actually survived?

**Good**:
What ultimately happened to the major and recurring characters in Tyler Perry's If Loving You Is Wrong, and who was confirmed dead, left unclear, or survived?

### Example 5 (expression: compression + single main question)

**query** (long):
A veteran actor famous for authoritative roles in Indian mythological TV dramas also played a major villain in a 1990s superhero series. Despite decades of varied roles, that superhero villain often stands out more in public memory. Explain why, considering dialogue spread, visual design, show genre, and series revival.

**Bad** — barely shorter, still lists analysis angles:
Why is the 1990s superhero villain played by this veteran Indian mythological TV actor remembered more vividly than many of his other roles, especially in terms of dialogue spread, visual design, show genre, and series revival?

**Good**:
Why is this actor's 1990s superhero villain role remembered more vividly than most of his other work?

## Output format (strict JSON)

```json
{{
  "query_fuzzified": "string"
}}
```

## Self-check before output (hard gates — rewrite if any fail)

1. **Main object & sub-questions**: any confusion or dropped sub-question?
2. **Comma segments**: split on `,`; count **≤ 3** (strongly prefer **≤ 2**)? If it still lists parallel items, use umbrella phrasing.
3. **Checklist smell**: still 2+ parallel facets (`A and B`, `A, B, or C`)? `based on` / `in terms of` / `at ... levels` / `including` / `considering ... and ... broader ... history` / `roles behind`? stacked style adjectives? Delete or merge all.
4. **Sentences & question marks**: split on `.!?` + space; **≤ 3 sentences**? **≤ 2 `?`**? Can you merge to one sentence? Do it.
5. **Length ratio**: `len(query_fuzzified) / len(query) ≤ 0.70`? If not, keep deleting modifiers, lists, and method words.
6. **Process leakage**: can a reader infer step order? Any First/Then/Investigate/Analyze by? Rewrite.
7. **Read-aloud test**: if it sounds like a research rubric rather than a question a real user would ask, rewrite.
{retry_block}
Now output your final result:
"""

    return prompt
