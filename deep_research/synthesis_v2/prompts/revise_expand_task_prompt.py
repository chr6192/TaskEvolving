"""
Reviser 第一阶段（扩展任务）prompt 构造器。

第一阶段目标：
- 基于 trajectory 扩展任务流程图；
- 基于新流程图生成复杂度驱动的 query；
- 输出流程图文本字段、`query`，并为每个节点输出“源头溯源”信息。
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


def build_revise_expand_task_prompt(
    *,
    mermaid_k: Optional[str],
    traj_k: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    metadata = metadata or {}
    traj_k_str = json.dumps(traj_k, ensure_ascii=False, indent=2)
    metadata_str = json.dumps(metadata, ensure_ascii=False, indent=2)

    prompt = f"""你是一个根据trajectory，以扩展deep research任务搜索范围和深度的助手。你需要返回一个流程图mermaid和 `query`。前者是用来建模搜索过程，后者用自然语言刻画前者。

## 已知信息

- 工具调用轨迹（trajectory）的结构化表示（原样展示其结构和主要内容）：
{traj_k_str}


## mermaid的要求
- 总体要求：mermaid必须是 flowchart 语法的 DAG（不能出现回边/循环边）。是对trajectory的内部信息搜集/推理流程，*基于给你的 mermaid（可能为空） + trajectory**（包括query、reason、tool_call、tool_response、最终回答等hermes形式的字段，某些部分可能为空）得到节点内容及逻辑关系，不要凭空捏造。
- 节点内容：节点是内容必须是 trajectory 中存在的**具体行为，对象和结果**，统一设计为[(类型) 行为+对象：结果]的格式。**严格**根据使用trajectory中的信息来源分为两类：
  - 每一次tool_call以及其得到的对应tool_responses里的内容，是基于搜索得到的事实性信息，为一个事实性节点，用自然语言描述出来。举例：使用google搜索北京天气的tool_call以及返回的页面内容，的在变成节点时，至少需要包括其节点类型，具体的搜索对象是谁，搜索结果，即[(事实性) 搜索北京天气：晴天]。
  - query，reason，和最终回答中的信息，是基于推理得到的信息，为推理性节点。如果没有reason和最终回答，则不会出现推理性节点。举例：根据模型reason中的计算股票估值的过程，至少需要包括其节点类型，具体的计算对象，计算结果，即[(推理性) 计算得到紫金矿业估值价格：32元]。只有推理性节点才可以用“分析”，“计算”这种词。
- 节点粒度：无论是事实性节点还是推理性节点，一个节点最多对应单个操作，比如一次工具调用搜索的一个对象，或者针对一个问题的推理分析，不要把多个操作揉在一起。节点数目没有限制，尽量完备，不编造，不遗漏。
- 边的走向要求：能反映实际的信息和信息流，可以有多个起点或者终点。通过判断节点和与其相关的节点的逻辑关系，然后进行对应的关系组织。
  - 推导关系对应节点间的纵向组织：后续节点直接用到前序节点的信息。只有后续节点确定是由前序节点推导而来（没有前序节点无法链接至后续节点的情况）才能进行纵向组织，否则是并列关系。
  - 互补（并列）关系对应节点间的横向组织：并列节点代表了解决同一个任务的不同考虑方面，即使操作对象是同一个，但可以不依赖彼此独立完成。

## 节点源头（必须输出）
你必须为 mermaid 中**每一个节点**输出一条溯源记录，放在 `node_sources` 字段中。要求如下：
- `node_id`：节点ID（如 A/B/C...），必须和 mermaid 中一致。
- `node_text`：节点完整文本（即节点方括号里的内容）。
- `source_type`：节点信息来源类型，只能取以下之一：
  - `tool_call_response`：来自本轮 trajectory 的工具调用及返回。
  - `query_or_reason_or_answer`：来自 query/reason/final answer。
  - `previous_mermaid`：来自上一轮 mermaid 的继承节点（本轮 traj 未新增证据）。
- `source_ref`：来源定位信息（简短字符串）：
  - 若 `tool_call_response`，优先写 `tool_call_id=<id>`，并补充工具名/关键词。
  - 若 `query_or_reason_or_answer`，写明来源片段位置（如 `query` / `reason` / `final_answer`）。
  - 若 `previous_mermaid`，写明 `from_previous_mermaid`。
- `evidence`：不超过120字的证据摘要（必须是输入中可找到的内容，不可编造）。

# 你的工作
1. 如果mermaid为空，先从 trajectory 中得到关键节点及逻辑关系，并生成一个符合上述要求的 mermaid；不为空则跳过该步骤，直接根据已有的mermaid执行下一步扩充mermaid。
2. 扩展mermaid。通过分析trajectory中有，但是mermaid里没有被写入到的信息，扩展生成更复杂的mermaid。扩展后的mermaid应该包含当前mermaid的所有节点，并且应该包含当前mermaid没有的节点。扩展的信息可以不是解决当前task的，但一定是与当前某个节点逻辑相关的，同样根据该逻辑判断扩展到操作：
  - 推导关系对应纵向扩展：从深度上补充给你的mermaid没有的节点。
  - 互补（并列）关系对应横向扩展：新建兄弟节点并与其上下游依赖节点相连。

在扩展时，可以一次性扩展多个节点。但优先进行纵向扩展，使得整个task在深度上变难，即解决所需要经过的路径变长，无法增长时再考虑横向扩展。在纵向扩展时，优先考虑往后或者往前延伸，而不是添加中间节点。在必要时，可以修改已有节点和逻辑。

3. 基于扩充后的流程图生成 `query`：
    - 该query是对扩展后整个mermaid后的提问，不遗漏任何节点。
    - 每个节点只保留行为+对象，不要把结果直接写进query。
    - 尽量使用“首先”、“其次”、“然后”、“最后”、“以及”等连接词，使得query能够清晰，完整的描述出mermaid的所有逻辑，使得模型根据该query能够基本实现类似mermaid的推理逻辑。
    - 最后，在发问时，额外关注最后一层，可以继续往深度延伸的叶子节点，对其进行单独提问。

## 注意
- 在已传入mermaid的情况下，除非mermaid中节点内容有事实性错误，否则不要修改之前的节点内容，但是可以修改节点之间的边链接。不要删除节点，也不要删除节点内容，只做增量性添加。
- 再次检查所有节点内容，格式和粒度，确保节点内容具体，符合格式。
- 再次检查推导关系的节点是不是严格符合推导关系的定义，即后续节点存在的前提是前序节点。
- 实事求是是第一原则，不要为了mermaid的完整性或者逻辑合理性而添加任何编造的节点。严格遵循tool_call和其对应的tool_response对应事实性节点，reason，query和answer对应推理性节点的原则。当mermaid中不存在query，reason和answer的时候，不要添加推理性节点。
- 在扩展时，不需要保持mermaid整体的任务一致性，只需要保持节点间的逻辑性即可。



## 输出形态
你最终的输出会被解析为一个严格的 JSON 结构：

```json
{{
  "mermaid": "string",
  "query": "string",
  "node_sources": [
    {{
      "node_id": "string",
      "node_text": "string",
      "source_type": "tool_call_response | query_or_reason_or_answer | previous_mermaid",
      "source_ref": "string",
      "evidence": "string"
    }}
  ]
}}
```

现在，输出你的最终结果：
"""

    return prompt

