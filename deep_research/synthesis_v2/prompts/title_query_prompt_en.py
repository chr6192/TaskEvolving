"""
Generate a plain, fitting research title (title_query) from mermaid only.

Input: mermaid (structure + check_points as conclusions)
Output: title_query
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
        raise ValueError("mermaid is empty; cannot generate title_query")

    prompt = f"""The mermaid below describes my completed deep-research **content** and **conclusions** (nodes capture the research process and information; check_points are the key findings/evidence).

## My research (mermaid)

{mermaid}

What is the most plain, fitting research title `title_query` for this work? `title_query` must be a **one-sentence informational title** for a report or research project — not a question, outline, or checklist.

## Rules

- **Title form, not a question**: noun phrase or short declarative heading. **Do not** end with "?". **Do not** start with Why / How / What / Which.
- **Keep the research subject (most important)**:
  - Find the **object of study** from origin nodes, upstream factual nodes, and check_points in the mermaid (work title, season/episode, full person name, historical event, place + time, etc.) and include it in the title.
  - **Do not** use only an episode title, nickname, subplot, or character arc as the title subject while dropping the main anchor. Example: do not write only `Girls Versus Suits ...`; keep `How I Met Your Mother season 5 episode 12` (episode title may go in parentheses).
  - If both `show + season/episode` and `episode title/alias` exist, prefer: `subject (alias): research angle` or `subject: research angle`.
- **Informational and non-literary**:
  - Use **neutral, plain, verifiable** nouns; **do not** use literary, rhetorical, or emotional wording.
  - **Do not** upgrade check_points into prettier synonyms; prefer plain nouns from the mermaid / check_points.
  - It should read like a **research project title**, not a review headline or marketing copy.
- **Let reasoning conclusions set the tone**: after keeping the subject, summarize downstream Reasoning nodes; do not write a factual retrieval checklist.
- **Shorter than the mermaid**: merge facets; usually 10–18 English words (slightly longer when the subject string is included).
- **No solution pathways**: no "first ... then", node IDs, or retrieval steps.

## Examples

**Research subject**: How I Met Your Mother season 5 episode 12 = "Girls Versus Suits" (100th episode)

**Bad** (subject dropped, only episode title left):
Girls Versus Suits as Mother-story progress beyond a 100th-episode showcase

**Bad** (question):
Why does Girls Versus Suits matter as the 100th episode?

**Good** (subject kept + plain research angle):
How I Met Your Mother S5E12 (Girls Versus Suits): mother-story progress beyond the 100th-episode milestone

## Self-check before output

- After reading the title, can someone tell **which work / episode / person / event** is studied? If not, add the subject.
- Did an episode title, character line, or sub-theme become the grammatical subject? If yes, rewrite as `subject + supplement`.
- Any metaphor, hype adjective, or emotional judgment? Replace with neutral informational wording.
- Ends with "?" or starts with Why/How/What/Which? Rewrite as a title.

## Output format (strict JSON)

```json
{{
  "title_query": "string"
}}
```

Now output your final result:
"""

    return prompt
