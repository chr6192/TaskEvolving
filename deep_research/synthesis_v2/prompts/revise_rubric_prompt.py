"""
Reviser 第二阶段（从子 mermaid + 子 rubric 生成 query_fuzzified）prompt 构造器。

第二阶段目标：
- 接收程序化拆分后的单个子 mermaid 及其对应子 rubric；
- 仅生成一条 query_fuzzified。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def build_leaf_query_prompt(
    *,
    sub_mermaid: str,
    sub_rubric: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    metadata = metadata or {}
    _ = metadata  # 保留参数接口
    sub_mermaid = (sub_mermaid or "").strip()

    rubric_str = json.dumps(sub_rubric, ensure_ascii=False, indent=2)

    prompt = f"""你是一个专门基于 Mermaid 流程图生成复杂 query_fuzzified 的助手。

## 已知信息

- 子 mermaid（flowchart DAG，代表一条完整推理路径）：
```mermaid
{sub_mermaid}
```

- 该路径对应的评估标准（rubric）：
{rubric_str}

## 你的任务

基于上述 mermaid 和 rubric，生成一条 `query_fuzzified`。以下所有约束都针对 `query_fuzzified`：

- **方向锚定（必须）**：query_fuzzified 必须引导回答者沿子 mermaid 的方向推进（目标、约束、关注维度一致），并与 rubric 评估维度对齐。
- **必要且充分（必须）**：query_fuzzified 需覆盖任务成立所需的核心目标与关键约束，不额外扩展新目标，也不遗漏关键边界条件。
- **细节隐藏（必须）**：不要直接复述 mermaid 的节点、边、步骤顺序或解法模块；避免把推理路径翻译成可直接照抄的操作清单。
- **变量贴源（必须）**：核心变量、对象、边界条件来自 mermaid/rubric；禁止引入图中不存在的新变量、新问题或新任务方向。
- **细节抽象（必须）**：若 mermaid/rubric 含阈值、数值、公式、固定规则，不要原样搬运；改写为抽象条件（如“过长排队”“近期口碑异常”“附近可替代选项”“实时运营信号”）。
- **保留区分度（必须）**：让回答者必须自行完成证据核验、风险识别、触发条件判断与方案权衡；不要把关键判定逻辑提前写死。

- **信息压缩（必须）**：query_fuzzified 的信息量应显著少于 mermaid，仅表达“问题结构”而非“实现细节”；rubric 中可直接照抄执行的内容必须模糊化。
- **表达形态（必须）**：写成完整、自然的复杂问题（用户提问口吻），不是步骤说明或模板化指令。
- **语言风格（必须）**：1-2 句，贴近普通用户表达，少术语、少书面腔，聚焦“我到底该怎么判断/怎么选”这类实际目标，用最直白，最普通的话描述这个任务在做什么。
- **术语控制（硬性）**：能用日常词就不要用专业词。避免“时序耦合/动态校准/跨模态/语义判别/分级验证逻辑/终局裁决准则”等学术化表达；改成用户能直接听懂的话。
- **可读性约束（硬性）**：句子不要堆抽象名词，不要连续出现多个“机制/体系/策略/模型/范式”等词；优先用动词+具体对象表达。
- **禁止项（硬性）**：禁止步骤化语言（如“第一步/然后/最后/先…再…”）；禁止过程枚举（如“从a、b、c方面/结合a、b、c来源”）；禁止冗余修饰与过细例子；禁止为了“显得专业”而引入新术语。


最终目的：生成一条 query_fuzzified，能够引导回答者沿子 mermaid 的大致方向推进（目标、约束、关注维度一致），并与 rubric 评估维度对齐，使得 mermaid 是该query_fuzzified的充分且最小的解决路径。同时尽可能最大程度的模糊化任务，省略具体解决途径和检索到的信息，只保留最核心的终极任务，并贴近用户表达。

## 输出形态（严格 JSON）

```json
{{
  "query_fuzzified": "string"
}}
```

现在，输出你的最终结果：
"""

    return prompt


# 保留旧函数名兼容，但内部已不使用
def build_revise_rubric_prompt(
    *,
    stage1_mermaid: str,
    stage1_query: str,
    stage1_rubric: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """向后兼容：不再推荐使用，请改用 build_leaf_query_prompt。"""
    return build_leaf_query_prompt(
        sub_mermaid=stage1_mermaid,
        sub_rubric=stage1_rubric or {"query_id": "", "criteria": []},
        metadata=metadata,
    )
