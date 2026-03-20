from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BusinessMermaid:
    """
    Prompt 注入用的“业务流程 Mermaid”封装。

    当前阶段先提供一个写死的 Mermaid 测试；后续可替换为 mermaid_k 注入。
    """

    mermaid: str

    @staticmethod
    def test() -> "BusinessMermaid":
        return BusinessMermaid(
            mermaid=(
                "flowchart TD\n"
                "  A[用户提出问题] --> B[拆解任务与检索计划]\n"
                "  B --> C[调用工具收集信息]\n"
                "  C --> D[综合与写作输出]\n"
            )
        )

    def as_system_prompt_injection(self) -> str:
        return (
            "\n\n"
            "# 业务流程约束（business_process_mermaid）\n"
            "你需要参考以下业务流程图来组织你的思考与行动步骤，确保整体流程不偏离。\n"
            "```mermaid\n"
            f"{self.mermaid}"
            "```\n"
        )

