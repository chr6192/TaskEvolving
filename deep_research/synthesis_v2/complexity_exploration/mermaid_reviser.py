from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig
from pydantic import BaseModel, Field

from deep_research.synthesis.core.data_structures import TrajectoryData
from deep_research.synthesis.core.structured_llm_client import StructuredLLMClient
from deep_research.synthesis.prompts.mermaid_revise_prompt import (
    build_mermaid_revise_prompt,
)

logger = logging.getLogger(__name__)


class RubricCriterion(BaseModel):
    id: str = Field(description="唯一ID（稳定）")
    description: str = Field(description="评判标准描述")
    weight: float = Field(description="权重(0-1)", ge=0.0, le=1.0)
    check_points: List[str] = Field(default_factory=list, description="检查点列表")


class RubricOutput(BaseModel):
    query_id: str = Field(description="与 query 对齐的 query_id")
    criteria: List[RubricCriterion] = Field(description="rubric checklist")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MermaidRevisionOutput(BaseModel):
    mermaid: str = Field(description="DAG Mermaid flowchart 文本（可直接放入 prompt）")
    query: str = Field(description="下一轮 query_{k+1}")
    rubric: RubricOutput = Field(description="OpenJudge 兼容 rubric（dict 形态）")


@dataclass
class MermaidReviserAgent:
    """
    Mermaid + Query + Rubric 统一产出 reviser（用模型生成）。

    输入：mermaid_k（可空）、traj_k、metadata（可选；其中可包含 evaluation_output）
    输出：{"mermaid": str, "query": str, "rubric": dict}
    """

    config: DictConfig

    def __post_init__(self) -> None:
        reviser_cfg = self.config.complexity_exploration.reviser
        model_name = reviser_cfg.get("model_name") or self.config.get("openai_api", {}).get("default_model_name")
        if not model_name:
            raise ValueError(
                "MermaidReviser 未配置模型！请设置 complexity_exploration.reviser.model_name 或 openai_api.default_model_name"
            )

        self.llm_client = StructuredLLMClient(self.config, model_name=model_name, component_config=reviser_cfg)
        self._initialized = False

    async def initialize(self) -> None:
        if getattr(self, "_initialized", False):
            return
        await self.llm_client.initialize()
        self._initialized = True

    async def cleanup(self) -> None:
        if getattr(self, "llm_client", None):
            await self.llm_client.cleanup()
        self._initialized = False

    def _validate_mermaid(self, mermaid: str) -> None:
        """
        轻校验：
        - 非空
        - 长度上限（可配置，避免 prompt/落盘膨胀）
        """
        m = (mermaid or "").strip()
        if not m:
            raise ValueError("reviser 输出 mermaid 为空")

        limit = int(self.config.complexity_exploration.reviser.get("mermaid_char_limit", 2000))
        if len(m) > limit:
            raise ValueError(f"reviser 输出 mermaid 过长: {len(m)} > {limit}")

    async def revise(
        self,
        *,
        mermaid_k: Optional[str],
        traj_k: TrajectoryData,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not getattr(self, "_initialized", False):
            raise RuntimeError("MermaidReviserAgent 未初始化，请先调用 initialize()")

        # 控制 prompt 长度：只给 LLM 必要信息
        # 这里只传递一次完整的 traj_k 结构（包含 query、工具调用、工具返回和最终回答），
        # 不再重复拆分后以多种形式传入，避免信息冗余。
        prompt = build_mermaid_revise_prompt(
            mermaid_k=mermaid_k,
            traj_k=traj_k.to_dict(),
            metadata=metadata,
        )

        parsed = await self.llm_client.call_with_structured_output(
            prompt=prompt,
            response_model=MermaidRevisionOutput,
            system_message="你输出必须严格符合给定的结构化 schema。",
        )

        out = parsed.model_dump(mode="json")
        self._validate_mermaid(out.get("mermaid", ""))

        # 只保留轻量级、与追踪相关的字段，避免把完整 evaluation_output 挂在 rubric 上
        metadata = metadata or {}
        filtered_metadata = {}
        for key in ("iteration_id", "evolution_id"):
            if key in metadata:
                filtered_metadata[key] = metadata[key]

        out["rubric"]["metadata"] = {
            **(out["rubric"].get("metadata") or {}),
            "source": "MermaidReviserAgent(structured_llm)",
            **filtered_metadata,
        }
        return out

