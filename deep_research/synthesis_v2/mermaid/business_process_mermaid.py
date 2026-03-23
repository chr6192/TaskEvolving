from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BusinessMermaid:
    """
    Prompt 注入用的“业务流程 Mermaid”封装。

    允许实例创建，并作为system prompt的一部分注入给agent。
    """
    mermaid: str

    def as_system_prompt_injection(self) -> str:
        return (
            "\n\n"
            "# 业务流程约束（business_process_mermaid）\n"
            "你需要参考以下业务流程图来组织你的思考与行动步骤，确保整体流程不偏离。\n"
            "```mermaid\n"
            f"{self.mermaid}"
            "```\n"
        )

