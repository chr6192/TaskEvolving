"""
将 query 改写为更短、更口语的 query_fuzzified。

输入：单条 query
输出：query_fuzzified
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

    prompt = f"""
你是一个将 deep research 问题改写为更短、更口语用户提问的助手。

给定一条 `query`，输出与其目标一致、语义对齐且不丢失子问题的 `query_fuzzified`。`query_fuzzified` 只提问问题，不包括约束/解决问题的途径。

## 输入 query

{query}

## 改写规则

- **语义对齐，不丢失问题**：
    - 不省略掉核心的提问对象：比如“如何塑造《基督山伯爵》主人公的性格？”，不要简化为“如何塑造主人公的性格？”，这样会导致问题变得含糊。
    - 不改变 query 提问的主对象：query 询问“结合更宏大的皮威电视与电影音乐背景，并考虑到其著名主题曲背后不同的创作和演唱角色，应如何理解该剧最著名的主题曲？”，不可以改写为“如何理解主题曲以及更宏大的电视与电影音乐背景？”，也不可以保留“考虑到创作/演唱角色以及更广泛的皮威影视音乐史”这类背景脚手架，而应改写为“如何理解该剧最著名的主题曲？”，因为 query 的提问对象只有主题曲，背景与子要素都应隐含而非点名。
    - 不丢失子问题：把 query "为什么儿童生长板在承担长高功能的同时特别容易受伤？请从组织结构与生理调节的角度解释这种双重属性的内在成因，并说明这些生物学特性如何具体指导临床上的损伤识别、治疗原则及随访策略。"，可以简化为 "从内在成因解释为什么儿童生长板既负责长高又特别容易受伤，并说明这些特性有什么临床指导意义？"，而不能简化为"儿童生长板既负责长高又特别容易受伤，这些特性有什么临床指导意义？"，因为丢失了“从内在成因解释为什么儿童生长板既负责长高又特别容易受伤”这个子问题。
- **问题简化**：
    - 对宽度进行模糊，去掉专业名词、方法、执行细节。比如“为什么需要对空战飞行员的飞行技能进行一对一评估？”，可以简化为“为什么需要评估空战飞行员的飞行技能？”
    - 对过程进行模糊，去掉途径。比如“如何通过故事、语言和行为塑造《基督山伯爵》的主人公的性格？”，可以简化为“如何塑造《基督山伯爵》主人公的性格？”
- **宽度模糊（最重要），去掉约束**：
    - 「不丢失子问题」指语义上仍覆盖 query 的全部目标，**不是**把 query 里的并列清单原样搬进 query_fuzzified。
    - 删除分面/维度约束：把“从 A、B、C 方面”里的 A、B、C 去掉，只保留任务本身。例如“请结合算法机制、特殊身份和内容变化来分析 xxxx 的原因”→“分析 xxxx 的原因”；“分飞掠、撞击、软着陆、载人绕月、载人着陆等不同定义分析…”→“从不同定义分析…”。
    - **禁止串列（含 2 项）**：不得用逗号/分号/`and`/`or`/`as well as`/`including`/`such as`/`both ... and` 串起 **2 项及以上**平行分析维度、剧情线、机制、层面、情况、渠道。哪怕只有“A 和 B”“A、B 以及 C”，也必须合并为上位说法，只留 1 个总称 + 任务动词（如 "what mainly drives X" / "how Y is handled"），而不是保留各个分面名称。
    - **删除脚手架与背景约束**：凡 `based on` / `depending on` / `in terms of` / `at the legal and practical level` / `from ... and ... aspects` / `considering ... and ...` / `in the context of` / `against the backdrop of` / `broader ... history` 等只是在列分析角度、依据，或把「具体子要素 + 更广泛历史/语境」（如主题曲背后的创作/演唱角色 + 更广泛影视音乐史）搬进问句的，整段删掉，只留核心提问；背景只隐含，不得再列一遍。
    - **删除装饰性修饰语**：风格评价、程度副词、叙事方式形容词（如 freer、character-driven、shifting tone、nuanced、multifaceted）若删掉后问题仍完整，必须删掉；不要保留“更自由、由角色驱动的风格”这类 rubric 式描写。
- **过程模糊**：
    - 删除过程约束、步骤暗示、解题途径：不得出现 `Focus on` / `Analyze by` / `Trace` / `Use evidence` / `Research` / `Investigate` / `when you separate` / `Please compare` 等表达。
    - 删除打分条件、来源要求、细化边界等附加限制；禁止增加 query 中没有的新限制。
    - 隐藏求解顺序：若 query 含层层递进的多问，把有依赖的子问题隐含进一个大问，不暴露「先 A 后 B 再 C」的研究步骤。例如“2026 年 X 平台粉丝最多的账号是谁？为什么他能断层领先？”→“分析 2026 年 X 平台粉丝最多账号为什么能在粉丝数上断层领先其他头部用户？”。
- **表达形态**：
    - 完整、自然的用户提问；**优先 1 句，最多 2 句，绝不超过 3 句**。
    - 除提问的核心内容保持外，其余专有名次用概括表达。
    - 尽量是 1 个大问题隐含地包含了 query 中的所有小问题，口语化、直白、少术语。

### 例子

【例1 · 宽度模糊：分面清单】

**query**：
When someone is picking their first apartment in a new city, what should they evaluate across commute time, monthly rent, neighborhood safety, nearby groceries, and building maintenance, and how should they balance cheaper rent against a longer commute?

**反例**——仍把 query 里的五个分面逐项列出（commute / rent / safety / groceries / maintenance），宽度没模糊：
When picking a first apartment, what should renters weigh regarding commute time, monthly rent, neighborhood safety, nearby groceries, and building maintenance, and how should they balance rent against commute?

**正例**——用上位说法合并分面，但保留两个子问题：①选首套房该看什么；②便宜房租与更长通勤如何权衡：
What should you look for when choosing your first apartment in a new city, and how do you balance cheaper rent against a longer commute?

【例2 · 过程+宽度：分步与选项清单】

**query**：
First identify which café on Main Street has the highest average rating this month. Then explain why it outperforms nearby competitors, and determine whether the gap comes mainly from drink quality, service speed, or price.

**反例**——仍按「先 A 后 B 再 C」分步提问，且仍逐项列出 drink quality / service / price，过程与宽度都没模糊：
Which café on Main Street has the highest rating this month? Then why does it outperform nearby competitors, and does the gap come mainly from drink quality, service speed, or price?

**正例**——把「哪家店」隐含进「评分最高的店」，并把口味/服务/价格合并为「什么 mainly 驱动优势」，保留两个子问题：①为何高于周边；②优势主要来自什么：
Why does the top-rated café on Main Street score higher than nearby competitors this month, and what mainly drives that advantage?

【例3 · 宽度模糊：多维度仍用逗号串列（高频失败模式）】

**query**：
Why is the 2007 film Fred Claus divisive, and does that stem more from its brother-conflict story, the lead performance, how its comedy-sentiment mix shapes Fred's redemption, critic-audience disagreement, or its holiday rewatch appeal?

**反例**——把多个分析轴用逗号一项项列出（5 段逗号），像 mini-checklist：
Why is Fred Claus so divisive, and does that come more from its brother-conflict story or the lead performance, especially in how its comedy-sentiment mix shapes Fred's redemption, the critic-audience gap, and its holiday rewatch appeal?

**正例**——不再点名故事线/基调/表演等分面，只保留核心比较：
Why is Fred Claus so divisive, and what mainly drives that split in reception?

【例6 · 宽度模糊：2–3 个分面仍须合并】

**query**：
该片口碑为何两极分化？这种分歧更多来自家庭和解故事线、基调起伏，还是主演发挥？

**反例**——仍保留多个分析分面：
Why is the film so divisive, and does that come more from its family-reconciliation story, shifting tone, or the leads' performances?

**正例**——只问分歧本身及主要驱动，不列分面：
Why is the film so divisive, and what mainly drives that disagreement?

【例7 · 删除「基于…情况」脚手架】

**query**：
请结合袭击造成的损毁与后续修复进展，判断该设施何时能恢复全面运行。

**反例**——仍保留 damage / recovery 双依据：
Based on damage from the attack and subsequent recovery progress, when can the facility return to full operation?

**正例**——删掉依据脚手架，只留核心时间点问题：
When can the facility return to full operation after the attack?

【例8 · 删除「层面」约束与形容词堆砌】

**query**：
在法律与实践层面，该剧对其他主要角色采取了怎样更自由、更人物驱动的处理方式？

**反例**——仍保留 legal/practical 双层面 + 风格形容词：
At legal and practical levels, how does the show use a freer, more character-driven approach for its other major characters?

**正例**——只问处理对象与核心动作：
How does the show handle its other major characters?

【例9 · 删除「语境/背景」脚手架：子要素 + 更广泛历史】

**query**：
结合更宏大的皮威电视与电影音乐背景，并考虑到其著名主题曲背后不同的创作和演唱角色，应如何理解该剧最著名的主题曲？

**反例**——仍保留创作/演唱角色与更广泛音乐史等背景约束：
How should we understand the show's famous theme song, considering the different creative and performing roles behind it and the broader Pee-wee film and TV music history?

**正例**——只保留提问主对象（主题曲），背景与子要素全部隐含：
How should we understand the show's famous theme song?

【例4 · 宽度模糊：角色名 checklist】

**query**：
What ultimately happened to Randal Holmes, Larry Connlee, Eddie Willis, Ian Glenn, Mika Willis, Travis Cain, Quan, Ramsey Walters, Kelly Isaacs, Joey Blackman, and other major characters in Tyler Perry's If Loving You Is Wrong, and who was confirmed dead, presumed dead, left unclear, or survived?

**反例**——逐个点名 10+ 角色，完全未模糊宽度：
What ultimately happened to the major and recurring characters in Tyler Perry's If Loving You Is Wrong, including Randal Holmes, Larry Connlee, Eddie Willis, Ian Glenn, Mika Willis, Travis Cain, Quan, Ramsey Walters, Kelly Isaacs, and Joey Blackman, and who was confirmed dead, presumed dead, left unclear, or actually survived?

**正例**——只保留作品名 + 上位对象 + 结局类型：
What ultimately happened to the major and recurring characters in Tyler Perry's If Loving You Is Wrong, and who was confirmed dead, left unclear, or survived?

【例5 · 表达形态：必须压缩 + 单句大问】

**query**（较长中文原问）：
一位以在印度史诗神话电视剧中扮演权威人物而闻名的资深演员，同时也在上世纪90年代的一部超级英雄剧集中饰演了主要反派。尽管他拥有长达数十年的多样角色生涯，但这个超级英雄反派在公众记忆中往往比他的其他角色更为突出。请从台词传播、角色视觉设计、节目类型和系列复兴等角度，分析这一现象背后的原因。

**反例**——几乎没缩短（fuzz/query > 0.7），且仍暴露分析角度清单：
Why is the 1990s superhero villain played by this veteran Indian mythological TV actor remembered more vividly than many of his other roles, especially in terms of dialogue spread, visual design, show genre, and series revival?

**正例**——大幅压缩，合并子问题，无角度清单：
Why is this actor's 1990s superhero villain role remembered more vividly than most of his other work?

```json
{{
  "query_fuzzified": "string"
}}
```

## 输出前自检（逐项硬检，任一不通过必须重写）

1. **主对象**：有没有混淆 query 的提问主对象？有没有丢失任一子问题？
2. **逗号分段**：按 `,` 切分，段数是否 **≤ 3**（强烈建议 ≤ 2）？若仍像在列 parallel items，改写上位概念。
3. **checklist 感**：是否仍保留 2 个以上平行分面（含 `A and B`、`A, B, or C`）？是否出现 `based on` / `in terms of` / `at ... levels` / `including` / `considering ... and ... broader ... history` / `roles behind`？是否堆叠风格形容词？是则全部删掉或合并。
4. **句数与问号**：按 `.!?` 后空格切分，是否 **≤ 3 句**？问号是否 **≤ 2**？能否合并为 1 句、1 个问号？能则合并。
5. **长度比**：`len(query_fuzzified) / len(query)` 是否 **≤ 0.70**？否则继续删修饰语、删清单、删方法词。
6. **过程外露**：读者能否看出「先查 A 再查 B 再 C」？是否含 First/Then/Investigate/Analyze by 等？是则重写。
7. **朗读测试**：读 aloud 若像研究 rubric 或提纲，而不是真人会问的一句话，必须重写。
{retry_block}
最终的输出中的内容使用英文，现在输出结果：
"""

    return prompt
