"""
Reviser 第二阶段 prompt 构造器。

第二阶段目标：
- 接收 query 与 mermaid；
- 生成 `query_fuzzified` 与 rubric。
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

    prompt = f"""你是一个根据 query 和 mermaid，生成模糊化提问稿和评估标准的助手。

你需要基于给定的 query 和 mermaid，输出：
1. `query_fuzzified`：对 query 的模糊化版本（更短、更口语、更少约束）；
2. `rubric`：与 query 对齐的评估标准。

## 已知信息

- query：
{query}

- mermaid（structure + nodes）：
{mermaid}


## 任务一：生成 `query_fuzzified`

对 query 做改写，生成比 query 更简单、更短、更直接、更易懂，但同时保证语义对齐且不丢失的同一个问题表述。要求：

- **问题简化**：去掉专业名词和方法，不复述 mermaid 的节点、边、顺序或执行细节。
- **约束模糊**：删除或模糊过程约束、步骤暗示、细化边界、打分条件、来源要求等附加限制，禁止增加 query 中没有的新限制。
    - 删除不同方面的约束：把“从A、B、C方面”里的A、B、C去掉，比如“请结合算法机制、特殊身份和内容变化来分析xxxx的原因”，省略“结合算法机制、特殊身份和内容变化”，只保留任务本身“分析xxxx的原因”，比如“分飞掠、撞击、软着陆、载人绕月、载人着陆等不同定义分析首次到达月球的任务是什么？”，简化为“从不同定义分析首次到达月球的任务是什么？”
    - 对过程模糊：query中可能是几个层层递进的问题，分析这几个问题之间的关系，有依赖关系的子问题隐含到大问题里。比如“2026年X平台粉丝最多的账号是谁？为什么他的粉丝数能断层领先其它头部用户？”里的“2026年X平台粉丝最多的账号是谁？”子问题隐去，得到，“分析2026年X平台粉丝最多账号为什么能在粉丝数上断层领先其他头部用户？。
- **语义对齐，不丢失问题**：把query "为什么儿童生长板在承担长高功能的同时特别容易受伤？请从组织结构与生理调节的角度解释这种双重属性的内在成因，并说明这些生物学特性如何具体指导临床上的损伤识别、治疗原则及随访策略。"，可以简化为 "从内在成因解释为什么儿童生长板既负责长高又特别容易受伤，并说明这些特性有什么临床指导意义？"，而不能简化为"儿童生长板既负责长高又特别容易受伤，这些特性有什么临床指导意义？"，因为丢失了“从内在成因解释为什么儿童生长板既负责长高又特别容易受伤”这个子问题。
- **表达形态**：完整、自然的用户提问，不超过 3 句，尽量是一个大问题隐含地包含了query中的所有小问题，口语化、直白、少术语。


## 任务二：根据query，为mermaid中的每个节点生成一条criteria，用来判定回答质量。

每个criteria包括多条可核验checkpoint，包括以下字段：
    - `node_id`：与 `mermaid.nodes` 中对应节点的 `node_id` **完全一致**（大小写敏感），如 structure 节点为 `A`，则写 `"A"`，不要用 `node_a` 等别名。
    - `description`：使用该节点在 structure 中的行为+对象文本，说明判定的内容是什么。
    - `check_points`：复用该节点在 mermaid.nodes 中的 check_points，可微调表述但不得编造新事实，用于说明如何判定。

要求：
- 具体，可直接判定：
    - 每条必须可机械判定，是可执行的，避免“全面/深入/高质量”等主观词。内容需要具体，来源于已知的信息，不要编造证据，且必须
    是可验证的。
    - 不要提及 mermaid 中节点相关信息（比如“是否依赖于节点A”，或者“节点A的输出是节点B的输入”），对于要评估的agent而言 
    mermaid 是不存在的，换成具体内容的判定。
    - 判定重点应是“信息是否成立、证据是否支撑、推理是否一致”，而不是“是否执行了某个检索动作/点击动作/交叉验证步骤”。
- 根据节点如何服务于query而修改内容，剔除对于超出范围的判定，尤其是对于可替代的评估内容，不要过分严格：
    - 除非题目本身明确要求，不要要求特定工具、搜索引擎、URL、tool call 或检索步骤，因为任何搜索路径都可以得到同样的信息。
    - 除非是必须出现的术语，不要要求必须出现某个关键词或固定表述。因为不同表达方式可以传达出相同含义，所以只需要判定含义，而
    不是判定表达。
    - 除非是必须出现的特定例子，否则允许不同但合理的例子。如果需要例子，应写成“提供至少一个相关实例或证据”
    - 除非题目明确要求比较信源可信度，否则不要把“跨多少数据库交叉验证/采信流程”写成刚性评分项，应评估最终结论的证据充分性与
    逻辑一致性。
    - 如果某个节点对于解决query已经没有帮助了，则生成一个空criteria，里边依然包含node_id，description等字段，但是check_points为空。

## 输出形态（严格 JSON），你需要用英语进行最终的输出：

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

## 输出前自检
- 检查改写完的query，有没有可以隐去的方面约束，如“从xxx，xxx，xxx方面”的表达，如果有，重新按照“任务一”中规则进行模糊化。
- 检查改写完的query如果是连续对几个问题发问，能不能在不丢失子问题的情况下，合并或者整理成一个大问题，如果有，重新按照“任务一”中规则进行模糊化。
- 检查rubric的数目是不是和mermaid节点数目一致，如果不一致，重新按照“任务二”中规则进行修改。
- 检查rubric里的判定内容有没有超过query的范围，即rubric的判定内容不能超出query的语义范围，所有checkpoint都能直接或间接的服务于query。如果存在与query最终目的无关的rubric，重新按照“任务二”中规则进行修改。

现在，输出你的最终结果：
"""

    return prompt


def build_leaf_query_prompt(
    *,
    stage1_query: str,
    sub_mermaid: str,
    sub_rubric: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """叶子拆分子图场景：保留旧接口。"""
    _ = sub_rubric
    return build_revise_rubric_prompt(
        query=stage1_query,
        mermaid=sub_mermaid,
        metadata=metadata,
    )
