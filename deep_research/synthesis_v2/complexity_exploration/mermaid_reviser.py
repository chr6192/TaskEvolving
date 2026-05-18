from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig
from pydantic import BaseModel, Field

from deep_research.synthesis.core.data_structures import TrajectoryData
from deep_research.synthesis.core.structured_llm_client import StructuredLLMClient
from deep_research.synthesis_v2.prompts.revise_expand_task_prompt import (
    build_revise_expand_task_prompt,
)
from deep_research.synthesis_v2.prompts.revise_rubric_prompt import (
    build_revise_rubric_prompt,
)
from ..mermaid import Mermaid

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
    query: str = Field(
        description="下一轮任务提问稿：尽量保留 Mermaid 所体现的多维度、依赖与约束，完整体现扩充后的复杂度，不刻意模糊，但仍不写逐步路径"
    )
    node_sources: List["MermaidNodeSource"] = Field(
        default_factory=list,
        description="每个节点的来源溯源信息（一节点一条）",
    )
    query_fuzzified: str = Field(
        description="在 query 基础上的用户口吻、口语化模糊版（仅落盘/对照；编排器下一轮 user message 使用 query）"
    )
    rubric: RubricOutput = Field(description="OpenJudge 兼容 rubric（dict 形态）")


class MermaidNodeSource(BaseModel):
    node_id: str = Field(description="节点ID（需与mermaid中的节点ID一致）")
    node_text: str = Field(description="节点完整文本")
    source_type: str = Field(
        description="节点来源类型：tool_call_response / query_or_reason_or_answer / previous_mermaid"
    )
    source_ref: str = Field(description="来源定位信息，例如tool_call_id、query、reason、final_answer等")
    evidence: str = Field(description="证据摘要（应可在输入中定位）")


class MermaidRevisionStage1Output(BaseModel):
    mermaid: str = Field(description="DAG Mermaid flowchart 文本（可直接放入 prompt）")
    query: str = Field(
        description="下一轮任务提问稿：尽量保留 Mermaid 所体现的多维度、依赖与约束，完整体现扩充后的复杂度，不刻意模糊，但仍不写逐步路径"
    )
    node_sources: List[MermaidNodeSource] = Field(
        default_factory=list,
        description="每个节点的来源溯源信息（一节点一条）",
    )


class MermaidRevisionStage2Output(BaseModel):
    query_fuzzified: str = Field(
        description="在 query 基础上的用户口吻、口语化模糊版（仅落盘/对照；编排器下一轮 user message 使用 query）"
    )
    rubric: RubricOutput = Field(description="OpenJudge 兼容 rubric（dict 形态）")


@dataclass
class MermaidReviserAgent:
    """
    Mermaid + Query + Rubric 统一产出 reviser（用模型生成）。

    输入：mermaid_k（可空）、traj_k、metadata（可选；其中可包含 evaluation_output）
    输出：{"mermaid": str, "query": str, "node_sources": list, "query_fuzzified": str, "rubric": dict}
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

    def _to_json_safe(self, value: Any) -> Any:
        """将 metadata 中的 Mermaid 等对象转换为可 JSON 序列化形态。"""
        if isinstance(value, Mermaid):
            return value.to_str()
        if isinstance(value, dict):
            return {k: self._to_json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_json_safe(v) for v in value]
        if isinstance(value, tuple):
            return [self._to_json_safe(v) for v in value]
        return value

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
        # 先不启用长度限制
        # if len(m) > limit:
        #     raise ValueError(f"reviser 输出 mermaid 过长: {len(m)} > {limit}")

    def _build_expand_task_prompt(
        self,
        *,
        mermaid: Optional[str],
        traj_k: TrajectoryData,
        metadata: Dict[str, Any],
    ) -> str:
        return build_revise_expand_task_prompt(
            mermaid_k=mermaid,
            traj_k=traj_k.to_dict(),
            metadata=metadata,
        )

    def _build_rubric_generation_prompt(
        self,
        *,
        metadata: Dict[str, Any],
        stage1_mermaid: str,
        stage1_query: str,
    ) -> str:
        return build_revise_rubric_prompt(
            stage1_mermaid=stage1_mermaid,
            stage1_query=stage1_query,
            metadata=metadata,
        )

    def _attach_rubric_metadata(
        self,
        *,
        out: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> None:
        # 只保留轻量级、与追踪相关的字段，避免把完整 evaluation_output 挂在 rubric 上
        filtered_metadata = {}
        for key in ("iteration_id", "evolution_id"):
            if key in metadata:
                filtered_metadata[key] = metadata[key]

        out["rubric"]["metadata"] = {
            **(out["rubric"].get("metadata") or {}),
            "source": "MermaidReviserAgent(structured_llm)",
            **filtered_metadata,
        }

    async def expand_task(
        self,
        *,
        mermaid: Optional[str],
        traj_k: TrajectoryData,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not getattr(self, "_initialized", False):
            raise RuntimeError("MermaidReviserAgent 未初始化，请先调用 initialize()")

        metadata = self._to_json_safe(metadata or {})
        prompt = self._build_expand_task_prompt(
            mermaid=mermaid,
            traj_k=traj_k,
            metadata=metadata,
        )

        parsed = await self.llm_client.call_with_structured_output(
            prompt=prompt,
            response_model=MermaidRevisionStage1Output,
            system_message="Your output must strictly conform to the provided structured schema.",
        )
        out = parsed.model_dump(mode="json")
        self._validate_mermaid(out.get("mermaid", ""))
        if not (out.get("query") or "").strip():
            raise ValueError("reviser 输出 query 为空")
        return out

    async def generate_rubric(
        self,
        *,
        stage1_mermaid: str,
        stage1_query: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not getattr(self, "_initialized", False):
            raise RuntimeError("MermaidReviserAgent 未初始化，请先调用 initialize()")

        metadata = self._to_json_safe(metadata or {})
        prompt = self._build_rubric_generation_prompt(
            metadata=metadata,
            stage1_mermaid=stage1_mermaid,
            stage1_query=stage1_query,
        )
        parsed = await self.llm_client.call_with_structured_output(
            prompt=prompt,
            response_model=MermaidRevisionStage2Output,
            system_message="Your output must strictly conform to the provided structured schema.",
        )
        out = parsed.model_dump(mode="json")
        if not (out.get("query_fuzzified") or "").strip():
            raise ValueError("reviser 输出 query_fuzzified 为空")
        self._attach_rubric_metadata(out=out, metadata=metadata)
        return out

    async def revise_stage1(
        self,
        *,
        mermaid_k: Optional[str],
        traj_k: TrajectoryData,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """兼容旧接口：等价于 expand_task。"""
        return await self.expand_task(mermaid=mermaid_k, traj_k=traj_k, metadata=metadata)

    async def revise_stage2(
        self,
        *,
        mermaid_k: Optional[str],
        traj_k: TrajectoryData,
        stage1_mermaid: str,
        stage1_query: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """兼容旧接口：等价于 generate_rubric。"""
        _ = mermaid_k
        _ = traj_k
        return await self.generate_rubric(
            stage1_mermaid=stage1_mermaid,
            stage1_query=stage1_query,
            metadata=metadata,
        )

    async def revise(
        self,
        *,
        mermaid_k: Optional[str],
        traj_k: TrajectoryData,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        stage1_out = await self.expand_task(
            mermaid=mermaid_k,
            traj_k=traj_k,
            metadata=metadata,
        )
        stage2_out = await self.generate_rubric(
            stage1_mermaid=stage1_out["mermaid"],
            stage1_query=stage1_out["query"],
            metadata=metadata,
        )
        out = MermaidRevisionOutput(
            mermaid=stage1_out["mermaid"],
            query=stage1_out["query"],
            node_sources=stage1_out.get("node_sources", []),
            query_fuzzified=stage2_out["query_fuzzified"],
            rubric=stage2_out["rubric"],
        ).model_dump(mode="json")
        return out

