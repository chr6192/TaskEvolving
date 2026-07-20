from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict, Field

from deep_research.synthesis.core.data_structures import TrajectoryData
from deep_research.synthesis.core.message_context_utils import (
    TOOL_RESPONSE_TRUNCATION_SUFFIX,
    apply_trajectory_view_char_budget,
    resolve_request_input_char_budget,
)
from deep_research.synthesis.core.structured_llm_client import StructuredLLMClient
from deep_research.synthesis_v2.core.language import VALID_NODE_TYPES, normalize_language
from deep_research.synthesis_v2.prompts.fuzzify_query_prompt import (
    build_fuzzify_query_prompt,
)
from deep_research.synthesis_v2.prompts.fuzzify_query_prompt_en import (
    build_fuzzify_query_prompt as build_fuzzify_query_prompt_en,
)
from deep_research.synthesis_v2.prompts.title_query_prompt import (
    build_title_query_prompt,
)
from deep_research.synthesis_v2.prompts.title_query_prompt_en import (
    build_title_query_prompt as build_title_query_prompt_en,
)
from deep_research.synthesis_v2.prompts.fuzzify_query_validation import (
    build_fuzzify_retry_feedback,
    collect_fuzzify_issues,
    validate_fuzzify_output,
)
from deep_research.synthesis_v2.prompts.revise_expand_task_prompt import (
    build_revise_expand_task_prompt,
)
from deep_research.synthesis_v2.prompts.revise_expand_task_prompt_en import (
    build_revise_expand_task_prompt as build_revise_expand_task_prompt_en,
)
from deep_research.synthesis_v2.prompts.revise_rubric_prompt import (
    build_leaf_query_prompt,
    build_revise_rubric_prompt,
)
from deep_research.synthesis_v2.prompts.revise_rubric_prompt_en import (
    build_leaf_query_prompt as build_leaf_query_prompt_en,
    build_revise_rubric_prompt as build_revise_rubric_prompt_en,
)
from ..mermaid import Mermaid

logger = logging.getLogger(__name__)

DEFAULT_FUZZIFY_MAX_ATTEMPTS = 3
REVISER_STRUCTURED_SYSTEM_MESSAGE = (
    "Your output must strictly conform to the provided structured schema."
)


class MermaidReviserValidationError(ValueError):
    """Stage1 校验失败时携带 LLM 原始输出，便于 reviser.json 落盘排查。"""

    def __init__(self, message: str, *, partial_output: dict):
        super().__init__(message)
        self.partial_output = partial_output


class RubricCriterionLLM(BaseModel):
    """Stage2 LLM 结构化输出（不含 weight，由程序按 mermaid 结构计算）。"""

    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(description="与 mermaid.nodes 中 node_id 完全一致")
    description: str = Field(description="评判标准描述")
    check_points: List[str] = Field(default_factory=list, description="检查点列表")


class RubricCriterion(RubricCriterionLLM):
    weight: float = Field(description="权重(0-1)，程序自下而上分配", ge=0.0, le=1.0)


class RubricOutputCore(BaseModel):
    """结构化输出用（不含自由 metadata，满足 strict json_schema）。"""

    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(description="与 query 对齐的 query_id")
    criteria: List[RubricCriterionLLM] = Field(description="rubric checklist")


class RubricOutput(BaseModel):
    query_id: str = Field(description="与 query 对齐的 query_id")
    criteria: List[RubricCriterion] = Field(description="rubric checklist")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MermaidGraphNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(description="节点ID（需与mermaid中的节点ID一致）")
    node_type: str = Field(description="节点类型：事实型/Factual 或 推理型/Reasoning")
    check_points: List[str] = Field(default_factory=list, description="该 Mermaid 节点对应的检查点")


class MermaidGraphNodeWithSource(MermaidGraphNode):
    source_type: str = Field(
        description="节点来源类型：tool_call_response / query_or_reason_or_answer / previous_mermaid"
    )
    source_note: str = Field(description="来源+证据的合成描述（本轮审计用，不进入代码 Mermaid）")


class MermaidGraphSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    structure: str = Field(description="DAG Mermaid flowchart 结构文本")
    nodes: List[MermaidGraphNodeWithSource] = Field(
        default_factory=list,
        description="结构化节点信息（含本轮溯源字段，解析后仅保留 node_id/node_type/check_points）",
    )


class MermaidRevisionOutput(BaseModel):
    mermaid: str = Field(description="完整 Mermaid 上下文（structure + nodes，可直接放入 prompt）")
    structure: str = Field(description="DAG Mermaid flowchart 结构文本")
    mermaid_nodes: List[MermaidGraphNode] = Field(
        default_factory=list,
        description="随 Mermaid 传递的结构化节点信息",
    )
    query: str = Field(
        description="下一轮任务提问稿：尽量保留 Mermaid 所体现的多维度、依赖与约束，完整体现扩充后的复杂度，不刻意模糊，但仍不写逐步路径"
    )
    query_fuzzified: str = Field(
        description="在 query 基础上的用户口吻、口语化模糊版（仅落盘/对照；编排器下一轮 user message 使用 query）"
    )
    leaf_queries: List["MermaidLeafQuery"] = Field(
        default_factory=list,
        description="按叶子节点拆分后的子 mermaid 与对应 query 列表",
    )
    rubric: RubricOutput = Field(description="OpenJudge 兼容 rubric（dict 形态）")


class MermaidRevisionStage1Output(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mermaid_new: MermaidGraphSpec = Field(
        description="完整 Mermaid 对象，包含 structure 和 nodes（nodes 含本轮溯源字段，解析后不进入代码 Mermaid）"
    )
    rubric: RubricOutputCore = Field(description="与 query 最终目的一致的 rubric")


class MermaidRevisionStage2QueryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="下一轮使用的扩展版 query")


class QueryFuzzifiedOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_fuzzified: str = Field(description="query 的用户口吻模糊版")


class TitleQueryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title_query: str = Field(description="query 对应的平实、贴切研究题目")


class MermaidRevisionStage2Output(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="下一轮使用的扩展版 query")
    query_fuzzified: str = Field(description="query 的用户口吻模糊版")


class LeafQueryOutput(BaseModel):
    """叶子拆分子图输出（兼容旧接口）。"""

    model_config = ConfigDict(extra="forbid")

    query_fuzzified: str = Field(description="扩展前 query 的模糊化版本")
    rubric: RubricOutputCore = Field(description="子图 rubric")


class MermaidLeafQuery(BaseModel):
    """叶子拆分结果（程序拆分 + LLM 生成 query_fuzzified）。"""

    leaf_node_id: str = Field(description="原始 mermaid 叶子节点ID")
    leaf_node_text: str = Field(description="原始 mermaid 叶子节点文本")
    mermaid: str = Field(description="通向该叶子节点的完整 Mermaid 上下文")
    mermaid_structure: str = Field(default="", description="通向该叶子节点的子 Mermaid 结构")
    mermaid_nodes: List[MermaidGraphNode] = Field(default_factory=list, description="子 Mermaid 节点字段")
    rubric: Dict[str, Any] = Field(default_factory=dict, description="子 rubric")
    query_fuzzified: str = Field(default="", description="query 的用户口吻模糊版")


@dataclass
class MermaidReviserAgent:
    """
    Mermaid + Query + Rubric 两阶段 reviser。

    Stage1 expand_task：接收 query、mermaid、trajectory，补充 mermaid，输出 mermaid + rubric
    Stage2 generate_query：接收新 mermaid、trajectory，输出 query
    fuzzify_query：接收 query，输出 query_fuzzified
    generate_title_query：接收 mermaid，输出 title_query
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
        openai_cfg = self.config.get("openai_api", {})
        self.max_request_input_char_budget = resolve_request_input_char_budget(
            max_request_input_tokens=reviser_cfg.get("max_request_input_tokens"),
            max_request_input_chars=reviser_cfg.get("max_request_input_chars"),
            fallback_tokens=openai_cfg.get("max_request_input_tokens"),
        )
        self._initialized = False

    @property
    def language(self) -> str:
        reviser_cfg = self.config.get("complexity_exploration", {}).get("reviser", {})
        lang = reviser_cfg.get("language") or self.config.get("complexity_exploration", {}).get("language")
        return normalize_language(lang)

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

    def _traj_dict_shell(self, traj_k: TrajectoryData) -> Dict[str, Any]:
        """
        Reviser prompt 的 trajectory 视图：仅保留 tool/assistant 消息。

        不嵌入 tool_responses / 根级 tool_calls 等字段——它们与 messages 中的 tool 角色重复，
        会导致 json.dumps 体积翻倍且使 traj_budget 被算成 0 从而跳过截断。
        """
        traj_dict = self._to_json_safe(traj_k.to_dict())
        messages = traj_dict.get("messages")
        if not isinstance(messages, list):
            return {"messages": []}

        filtered: List[Any] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") in ("system", "user"):
                continue
            filtered.append(msg)
        return {"messages": filtered}

    @staticmethod
    def _structured_schema_chars(response_model: Optional[type]) -> int:
        if response_model is None:
            return 0
        return len(json.dumps(response_model.model_json_schema(), ensure_ascii=False))

    def _reviser_estimated_request_chars(
        self,
        prompt: str,
        response_model: Optional[type],
    ) -> int:
        return (
            len(prompt)
            + len(REVISER_STRUCTURED_SYSTEM_MESSAGE)
            + self._structured_schema_chars(response_model)
        )

    def _build_reviser_prompt_fitted(
        self,
        builder,
        traj_k: TrajectoryData,
        response_model: Optional[type],
        log_label: str,
        **builder_kwargs: Any,
    ) -> str:
        """按整请求字符上限迭代截断 trajectory，以最终 prompt+system+schema 长度为准。"""
        cap = self.max_request_input_char_budget
        shell = self._traj_dict_shell(traj_k)
        messages = shell.get("messages")
        if not isinstance(messages, list):
            messages = []

        if cap <= 0:
            return builder(**builder_kwargs, traj_k=shell)

        empty_traj = dict(shell)
        empty_traj["messages"] = []
        base_prompt = builder(**builder_kwargs, traj_k=empty_traj)
        fixed = self._reviser_estimated_request_chars(base_prompt, response_model)
        traj_budget = max(0, cap - fixed)

        truncated = apply_trajectory_view_char_budget(
            list(messages),
            max_view_total_chars=traj_budget,
            log_prefix=f"reviser {log_label}",
        )
        prompt = builder(**builder_kwargs, traj_k={**shell, "messages": truncated})

        for attempt in range(8):
            total = self._reviser_estimated_request_chars(prompt, response_model)
            if total <= cap:
                logger.info(
                    "reviser %s fitted: prompt_chars=%s total=%s cap=%s "
                    "fixed=%s traj_budget=%s attempts=%s",
                    log_label,
                    len(prompt),
                    total,
                    cap,
                    fixed,
                    traj_budget,
                    attempt,
                )
                return prompt

            min_budget = (
                len(TOOL_RESPONSE_TRUNCATION_SUFFIX)
                * sum(
                    1
                    for msg in messages
                    if isinstance(msg, dict) and msg.get("role") in ("tool", "assistant")
                )
            )
            if traj_budget <= min_budget:
                logger.error(
                    "reviser %s 仍超 cap=%s (total=%s prompt=%s fixed=%s traj_budget=%s)，"
                    "已达最小 trajectory 预算，拒绝调用 API",
                    log_label,
                    cap,
                    total,
                    len(prompt),
                    fixed,
                    traj_budget,
                )
                raise ValueError(
                    f"reviser {log_label} 输入仍超限: total={total} cap={cap} "
                    f"prompt_chars={len(prompt)}"
                )

            traj_budget = max(min_budget, int(traj_budget * cap / total * 0.88))
            truncated = apply_trajectory_view_char_budget(
                list(messages),
                max_view_total_chars=traj_budget,
                log_prefix=f"reviser {log_label} (retry-{attempt + 1})",
            )
            prompt = builder(**builder_kwargs, traj_k={**shell, "messages": truncated})

        logger.warning(
            "reviser %s 拟合截断未收敛: total=%s cap=%s prompt=%s",
            log_label,
            self._reviser_estimated_request_chars(prompt, response_model),
            cap,
            len(prompt),
        )
        return prompt

    @staticmethod
    def split_mermaid_by_leaves(
        stage1_out: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        纯程序化拆分：按叶子节点拆 mermaid。

        输入：expand_task() 的输出（含 mermaid, mermaid_nodes）。
        输出：[{leaf_node_id, leaf_node_text, mermaid, mermaid_structure, mermaid_nodes}, ...]
        """
        mermaid_obj = Mermaid.from_str(
            MermaidReviserAgent._structure_from_out(stage1_out),
            nodes=stage1_out.get("mermaid_nodes") or [],
        )
        splits = mermaid_obj.split_by_leaves()
        if not splits:
            return []

        labels = mermaid_obj.node_labels()
        results: List[Dict[str, Any]] = []
        for leaf_id, sub_mermaid in splits:
            results.append({
                "leaf_node_id": leaf_id,
                "leaf_node_text": labels.get(leaf_id, leaf_id),
                "mermaid": sub_mermaid.to_str(),
                "mermaid_structure": sub_mermaid.to_structure_str(),
                "mermaid_nodes": sub_mermaid.node_dicts(),
            })
        return results

    def _validate_mermaid(self, mermaid: str) -> None:
        """轻校验：mermaid structure 非空。"""
        m = (mermaid or "").strip()
        if not m:
            raise ValueError("reviser 输出 mermaid 为空")

    def _build_expand_task_prompt(
        self,
        *,
        query: str,
        mermaid: Optional[str],
        traj_k: TrajectoryData,
        metadata: Dict[str, Any],
    ) -> str:
        builder = (
            build_revise_expand_task_prompt_en
            if self.language == "en"
            else build_revise_expand_task_prompt
        )
        return self._build_reviser_prompt_fitted(
            builder,
            traj_k,
            MermaidRevisionStage1Output,
            "expand_task",
            query=query,
            mermaid_k=mermaid,
            metadata=metadata,
        )

    def _build_revise_rubric_prompt(
        self,
        *,
        mermaid: str,
        traj_k: TrajectoryData,
        metadata: Dict[str, Any],
    ) -> str:
        builder = (
            build_revise_rubric_prompt_en
            if self.language == "en"
            else build_revise_rubric_prompt
        )
        return self._build_reviser_prompt_fitted(
            builder,
            traj_k,
            MermaidRevisionStage2QueryOutput,
            "revise_rubric",
            mermaid=mermaid,
            metadata=metadata,
        )

    def _build_leaf_query_prompt(
        self,
        *,
        stage1_query: str,
        sub_mermaid: str,
        sub_rubric: Dict[str, Any],
        traj_k: TrajectoryData,
        metadata: Dict[str, Any],
    ) -> str:
        builder = build_leaf_query_prompt_en if self.language == "en" else build_leaf_query_prompt
        return self._build_reviser_prompt_fitted(
            builder,
            traj_k,
            LeafQueryOutput,
            "leaf_query",
            stage1_query=stage1_query,
            sub_mermaid=sub_mermaid,
            sub_rubric=sub_rubric,
            metadata=metadata,
        )

    @staticmethod
    def _filter_rubric_to_nodes(rubric: Dict[str, Any], node_ids: set[str]) -> Dict[str, Any]:
        criteria = rubric.get("criteria") or []
        filtered = [
            dict(c)
            for c in criteria
            if MermaidReviserAgent._criterion_node_id(c) in node_ids
        ]
        return {**rubric, "criteria": filtered}

    def _validate_stage1_rubric_output(
        self,
        *,
        out: Dict[str, Any],
        stage1_out: Dict[str, Any],
    ) -> None:
        rubric = out.get("rubric") or {}
        criteria = rubric.get("criteria") or []
        if not criteria:
            raise ValueError("stage1 输出 rubric.criteria 为空")

        mermaid_nodes = stage1_out.get("mermaid_nodes") or []
        if len(criteria) != len(mermaid_nodes):
            raise ValueError(
                f"stage1 rubric 节点数不一致: criteria={len(criteria)} nodes={len(mermaid_nodes)}"
            )

        expected_ids = {
            (node or {}).get("node_id", "")
            for node in mermaid_nodes
            if (node or {}).get("node_id")
        }
        actual_ids = {
            self._criterion_node_id(c)
            for c in criteria
            if self._criterion_node_id(c)
        }
        if expected_ids != actual_ids:
            raise ValueError(
                f"stage1 rubric node_id 与 mermaid 节点不匹配: {sorted(actual_ids)} vs {sorted(expected_ids)}"
            )

    def _fuzzify_max_attempts(self) -> int:
        reviser_cfg = self.config.get("complexity_exploration", {}).get("reviser", {})
        raw = reviser_cfg.get("fuzzify_max_attempts", DEFAULT_FUZZIFY_MAX_ATTEMPTS)
        try:
            attempts = int(raw)
        except (TypeError, ValueError):
            attempts = DEFAULT_FUZZIFY_MAX_ATTEMPTS
        return max(1, attempts)

    def _build_fuzzify_query_prompt(
        self,
        *,
        query: str,
        metadata: Optional[Dict[str, Any]] = None,
        retry_feedback: Optional[str] = None,
    ) -> str:
        builder = (
            build_fuzzify_query_prompt_en
            if self.language == "en"
            else build_fuzzify_query_prompt
        )
        return builder(
            query=query,
            metadata=metadata or {},
            retry_feedback=retry_feedback,
        )

    def _build_title_query_prompt(
        self,
        *,
        mermaid: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        builder = (
            build_title_query_prompt_en
            if self.language == "en"
            else build_title_query_prompt
        )
        return builder(
            mermaid=mermaid,
            metadata=metadata or {},
        )

    def _validate_title_query_output(
        self,
        *,
        out: Dict[str, Any],
    ) -> None:
        title_query = (out.get("title_query") or "").strip()
        if not title_query:
            raise ValueError("title_query 为空")

    def _validate_stage2_query_output(
        self,
        *,
        out: Dict[str, Any],
    ) -> None:
        query = (out.get("query") or "").strip()
        if not query:
            raise ValueError("stage2 输出 query 为空")

    def _validate_fuzzify_output(
        self,
        *,
        out: Dict[str, Any],
        query: str,
    ) -> None:
        validate_fuzzify_output(
            query=query,
            query_fuzzified=(out.get("query_fuzzified") or "").strip(),
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

    @staticmethod
    def _criterion_node_id(criterion: Any) -> str:
        if not isinstance(criterion, dict):
            return ""
        node_id = (criterion.get("node_id") or "").strip()
        if node_id:
            return node_id
        # 兼容旧格式：id=node_<node_id>
        legacy_id = (criterion.get("id") or "").strip()
        if legacy_id.startswith("node_"):
            return legacy_id[len("node_") :]
        return legacy_id

    @staticmethod
    def _compute_bottom_up_node_weights(
        structure: str,
        *,
        focus_leaf_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        rubric 权重唯一算法入口：按 mermaid 结构自下而上分配节点权重。
        - focus_leaf_id 指定时：该叶子初始权重 1.0（子图拆分场景）；
        - 否则：所有叶子均分初始权重；
        - 逆拓扑向上：节点将自身权重均分给所有父节点；
        - 最后 sum 归一化。
        """
        flow = Mermaid.from_str(structure).to_flowchart()
        if not flow.edges:
            return {}

        nodes: set[str] = set()
        parents: Dict[str, List[str]] = {}
        indeg: Dict[str, int] = {}
        outdeg: Dict[str, int] = {}
        for e in flow.edges:
            nodes.add(e.src)
            nodes.add(e.dst)
        for nid in nodes:
            parents[nid] = []
            indeg[nid] = 0
            outdeg[nid] = 0
        for e in flow.edges:
            parents[e.dst].append(e.src)
            indeg[e.dst] += 1
            outdeg[e.src] += 1

        leaves = [nid for nid in nodes if outdeg.get(nid, 0) == 0]
        if not leaves:
            return {}

        node_weights: Dict[str, float] = {nid: 0.0 for nid in nodes}
        if focus_leaf_id and focus_leaf_id in node_weights:
            node_weights[focus_leaf_id] = 1.0
        else:
            leaf_weight = 1.0 / float(len(leaves))
            for leaf_id in leaves:
                node_weights[leaf_id] = leaf_weight

        topo: List[str] = []
        indeg_copy = dict(indeg)
        q: List[str] = [nid for nid, d in indeg_copy.items() if d == 0]
        while q:
            u = q.pop(0)
            topo.append(u)
            for edge in flow.edges:
                if edge.src == u:
                    indeg_copy[edge.dst] -= 1
                    if indeg_copy[edge.dst] == 0:
                        q.append(edge.dst)

        if len(topo) != len(nodes):
            return {}

        for nid in reversed(topo):
            w = node_weights.get(nid, 0.0)
            p = parents.get(nid, [])
            if not p or w <= 0:
                continue
            share = w / float(len(p))
            for pid in p:
                node_weights[pid] = node_weights.get(pid, 0.0) + share

        total = sum(node_weights.values())
        if total <= 0:
            return {}
        return {nid: w / total for nid, w in node_weights.items()}

    @staticmethod
    def _assign_weights_to_criteria(
        criteria: List[Dict[str, Any]],
        structure: str,
        *,
        focus_leaf_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        n = len(criteria)
        equal_fallback = {
            MermaidReviserAgent._criterion_node_id(c): 1.0 / float(n)
            for c in criteria
            if MermaidReviserAgent._criterion_node_id(c)
        } if n > 0 else {}

        node_weights = MermaidReviserAgent._compute_bottom_up_node_weights(
            structure,
            focus_leaf_id=focus_leaf_id,
        )

        criterion_node_ids = {
            MermaidReviserAgent._criterion_node_id(c)
            for c in criteria
            if MermaidReviserAgent._criterion_node_id(c)
        }

        if node_weights:
            weights = {nid: 0.0 for nid in criterion_node_ids}
            for nid, nw in node_weights.items():
                if nid in weights and nw > 0:
                    weights[nid] += nw
            total = sum(weights.values())
            if total > 0:
                weights = {nid: w / total for nid, w in weights.items()}
            else:
                weights = equal_fallback
        else:
            weights = equal_fallback

        result: List[Dict[str, Any]] = []
        for c in criteria:
            c_new = dict(c)
            nid = MermaidReviserAgent._criterion_node_id(c_new)
            c_new["weight"] = float(weights.get(nid, 0.0))
            result.append(c_new)
        return result

    def _apply_rubric_weights(
        self,
        *,
        out: Dict[str, Any],
        stage1_out: Dict[str, Any],
        focus_leaf_id: Optional[str] = None,
    ) -> None:
        rubric = out.get("rubric") or {}
        criteria = rubric.get("criteria") or []
        structure = self._structure_from_out(stage1_out)
        weighted = self._assign_weights_to_criteria(
            criteria,
            structure,
            focus_leaf_id=focus_leaf_id,
        )
        scorable: List[Dict[str, Any]] = []
        for c in weighted:
            if not isinstance(c, dict):
                continue
            check_points = c.get("check_points") or []
            if isinstance(check_points, list) and any(str(cp).strip() for cp in check_points):
                scorable.append(dict(c))
        if not scorable:
            raise ValueError("过滤 check_points 为空后 rubric.criteria 为空，无法评分")

        total_weight = sum(float(c.get("weight") or 0.0) for c in scorable)
        if total_weight <= 0:
            equal_weight = 1.0 / float(len(scorable))
            for c in scorable:
                c["weight"] = equal_weight
        else:
            for c in scorable:
                c["weight"] = float(c.get("weight") or 0.0) / total_weight

        out["rubric"]["criteria"] = scorable

        total_weight = sum(float((c or {}).get("weight") or 0.0) for c in scorable)
        if abs(total_weight - 1.0) > 0.05:
            raise ValueError(
                f"过滤空 checkpoint 后 rubric 权重之和应为 1.0，实际为 {total_weight:.4f}"
            )

    @staticmethod
    def _structure_from_out(out: Dict[str, Any]) -> str:
        if out.get("structure"):
            return str(out.get("structure") or "").strip()
        if out.get("mermaid_structure"):
            return str(out.get("mermaid_structure") or "").strip()
        return Mermaid.from_str(out.get("mermaid") or "").to_structure_str()

    def _validate_mermaid_nodes(self, out: Dict[str, Any]) -> None:
        mermaid_nodes = out.get("mermaid_nodes") or []
        if not mermaid_nodes:
            raise ValueError("reviser 输出 mermaid_nodes 为空")

        flow_node_ids: set[str] = set()
        mermaid_obj = Mermaid.from_str(self._structure_from_out(out))
        flow = mermaid_obj.to_flowchart()
        labels = mermaid_obj.node_labels()
        for edge in flow.edges:
            flow_node_ids.add(edge.src)
            flow_node_ids.add(edge.dst)

        seen_node_ids: set[str] = set()
        for node in mermaid_nodes:
            node_id = (node or {}).get("node_id", "")
            node_type = ((node or {}).get("node_type") or "").strip()
            check_points = (node or {}).get("check_points") or []
            source_type = ((node or {}).get("source_type") or "").strip()
            source_note = ((node or {}).get("source_note") or "").strip()

            if not node_id:
                raise ValueError("mermaid_nodes 存在空 node_id")
            if node_id in seen_node_ids:
                raise ValueError(f"mermaid_nodes 存在重复 node_id: {node_id}")
            if flow_node_ids and node_id not in flow_node_ids:
                raise ValueError(f"mermaid_nodes 节点 {node_id} 不存在于 mermaid 图中")
            node_text = (labels.get(node_id) or "").strip()
            if not node_text:
                raise ValueError(f"mermaid.structure 节点 {node_id} 缺少 node_text 标签")
            if node_type not in VALID_NODE_TYPES:
                raise ValueError(
                    f"mermaid_nodes 节点 {node_id} 的 node_type 非法: {node_type!r} "
                    f"(expected one of {sorted(VALID_NODE_TYPES)})"
                )
            if not check_points:
                raise ValueError(f"mermaid_nodes 节点 {node_id} 的 check_points 为空")
            if source_type not in {"tool_call_response", "query_or_reason_or_answer", "previous_mermaid"}:
                raise ValueError(f"mermaid_nodes 节点 {node_id} 的 source_type 非法")
            if not source_note:
                raise ValueError(f"mermaid_nodes 节点 {node_id} 的 source_note 为空")
            seen_node_ids.add(node_id)

    async def expand_task(
        self,
        *,
        query: str,
        mermaid: Optional[str],
        traj_k: TrajectoryData,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not getattr(self, "_initialized", False):
            raise RuntimeError("MermaidReviserAgent 未初始化，请先调用 initialize()")

        metadata = self._to_json_safe(metadata or {})
        prompt = self._build_expand_task_prompt(
            query=query,
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
        mermaid_payload = out.get("mermaid_new") or {}
        if not isinstance(mermaid_payload, dict):
            raise ValueError("reviser 输出 mermaid_new 必须是对象")
        structure = (mermaid_payload.get("structure") or "").strip()
        mermaid_nodes_raw = mermaid_payload.get("nodes") or []
        partial_output = {
            "mermaid_new": mermaid_payload,
            "structure": structure,
            "mermaid_nodes": mermaid_nodes_raw,
            "rubric": out.get("rubric"),
        }
        try:
            self._validate_mermaid(structure)
            out["structure"] = structure
            out["mermaid_nodes"] = mermaid_nodes_raw
            self._validate_mermaid_nodes(out)
        except ValueError as exc:
            raise MermaidReviserValidationError(
                str(exc),
                partial_output=partial_output,
            ) from exc
        mermaid_obj = Mermaid.from_str(structure, nodes=mermaid_nodes_raw)
        out["mermaid_nodes"] = mermaid_obj.node_dicts()
        out["mermaid_structure"] = mermaid_obj.to_structure_str()
        out["mermaid"] = mermaid_obj.to_str()

        self._validate_stage1_rubric_output(out=out, stage1_out=out)
        composed_query_id = (
            f"reviser_{metadata.get('evolution_id', 'unknown')}_{metadata.get('iteration_id', 'unknown')}"
        )
        out["rubric"]["query_id"] = composed_query_id
        self._apply_rubric_weights(out=out, stage1_out=out)
        self._attach_rubric_metadata(out=out, metadata=metadata)
        return out

    async def generate_query(
        self,
        *,
        stage1_out: Dict[str, Any],
        traj_k: TrajectoryData,
        metadata: Optional[Dict[str, Any]] = None,
        focus_leaf_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """基于 stage1 的 mermaid 与 trajectory，生成 query。"""
        _ = focus_leaf_id
        if not getattr(self, "_initialized", False):
            raise RuntimeError("MermaidReviserAgent 未初始化，请先调用 initialize()")

        metadata = self._to_json_safe(metadata or {})
        prompt = self._build_revise_rubric_prompt(
            mermaid=stage1_out.get("mermaid") or "",
            traj_k=traj_k,
            metadata=metadata,
        )
        parsed = await self.llm_client.call_with_structured_output(
            prompt=prompt,
            response_model=MermaidRevisionStage2QueryOutput,
            system_message="Your output must strictly conform to the provided structured schema.",
        )
        out = parsed.model_dump(mode="json")
        self._validate_stage2_query_output(out=out)
        return out

    def _fuzzify_log_context(self, metadata: Optional[Dict[str, Any]]) -> str:
        if not metadata:
            return ""
        parts: List[str] = []
        evolution_id = metadata.get("evolution_id")
        iteration_id = metadata.get("iteration_id")
        if evolution_id:
            parts.append(f"evolution_id={evolution_id}")
        if iteration_id is not None:
            parts.append(f"iteration_id={iteration_id}")
        return f" ({', '.join(parts)})" if parts else ""

    async def fuzzify_query(
        self,
        *,
        query: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """将 query 改写为 query_fuzzified。"""
        if not getattr(self, "_initialized", False):
            raise RuntimeError("MermaidReviserAgent 未初始化，请先调用 initialize()")

        metadata = self._to_json_safe(metadata or {})
        max_attempts = self._fuzzify_max_attempts()
        retry_feedback: Optional[str] = None
        candidates: List[Dict[str, Any]] = []
        log_ctx = self._fuzzify_log_context(metadata)

        for attempt in range(1, max_attempts + 1):
            prompt = self._build_fuzzify_query_prompt(
                query=query,
                metadata=metadata,
                retry_feedback=retry_feedback,
            )
            parsed = await self.llm_client.call_with_structured_output(
                prompt=prompt,
                response_model=QueryFuzzifiedOutput,
                system_message="Your output must strictly conform to the provided structured schema.",
            )
            out = parsed.model_dump(mode="json")
            candidates.append(out)
            try:
                self._validate_fuzzify_output(out=out, query=query)
                return out
            except ValueError as exc:
                failed_fuzz = (out.get("query_fuzzified") or "").strip()
                last_issues = collect_fuzzify_issues(
                    query=query,
                    query_fuzzified=failed_fuzz,
                )
                retry_feedback = build_fuzzify_retry_feedback(
                    last_issues,
                    language=self.language,
                )
                logger.warning(
                    "fuzzify 校验未通过%s (attempt %s/%s): %s | query_fuzzified: %s",
                    log_ctx,
                    attempt,
                    max_attempts,
                    exc,
                    failed_fuzz,
                )
                if attempt >= max_attempts:
                    best = min(
                        candidates,
                        key=lambda item: len((item.get("query_fuzzified") or "").strip()),
                    )
                    best_fuzz = (best.get("query_fuzzified") or "").strip()
                    best_issues = collect_fuzzify_issues(
                        query=query,
                        query_fuzzified=best_fuzz,
                    )
                    logger.warning(
                        "fuzzify %s 次均未通过校验%s，选用最短输出 (len=%s, issues=%s)",
                        max_attempts,
                        log_ctx,
                        len(best_fuzz),
                        "; ".join(best_issues),
                    )
                    logger.warning(
                        "fuzzify fallback query_fuzzified%s: %s",
                        log_ctx,
                        best_fuzz,
                    )
                    for idx, candidate in enumerate(candidates, start=1):
                        cand_fuzz = (candidate.get("query_fuzzified") or "").strip()
                        cand_issues = collect_fuzzify_issues(
                            query=query,
                            query_fuzzified=cand_fuzz,
                        )
                        logger.warning(
                            "fuzzify attempt %s candidate (len=%s, issues=%s): %s",
                            idx,
                            len(cand_fuzz),
                            "; ".join(cand_issues) or "none",
                            cand_fuzz,
                        )
                    return best

        raise RuntimeError("fuzzify_query 未产生任何输出")

    async def generate_title_query(
        self,
        *,
        mermaid: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """根据当轮 mermaid（研究内容与 check_points 结论）生成 title_query。"""
        if not getattr(self, "_initialized", False):
            raise RuntimeError("MermaidReviserAgent 未初始化，请先调用 initialize()")

        mermaid = (mermaid or "").strip()
        if not mermaid:
            raise ValueError("mermaid 为空，无法生成 title_query")

        metadata = self._to_json_safe(metadata or {})
        prompt = self._build_title_query_prompt(
            mermaid=mermaid,
            metadata=metadata,
        )
        parsed = await self.llm_client.call_with_structured_output(
            prompt=prompt,
            response_model=TitleQueryOutput,
            system_message="Your output must strictly conform to the provided structured schema.",
        )
        out = parsed.model_dump(mode="json")
        self._validate_title_query_output(out=out)
        return out

    async def generate_query_and_rubric(
        self,
        *,
        stage1_out: Dict[str, Any],
        traj_k: TrajectoryData,
        metadata: Optional[Dict[str, Any]] = None,
        focus_leaf_id: Optional[str] = None,
        include_title_query: bool = False,
    ) -> Dict[str, Any]:
        """基于 stage1 的 mermaid 与 trajectory，生成 query 和 query_fuzzified。"""
        query_out = await self.generate_query(
            stage1_out=stage1_out,
            traj_k=traj_k,
            metadata=metadata,
            focus_leaf_id=focus_leaf_id,
        )
        fuzz_out = await self.fuzzify_query(
            query=query_out["query"],
            metadata=metadata,
        )
        result = {**query_out, **fuzz_out}
        if include_title_query:
            mermaid_str = (stage1_out.get("mermaid") or "").strip()
            if mermaid_str:
                title_out = await self.generate_title_query(
                    mermaid=mermaid_str,
                    metadata=metadata,
                )
                result.update(title_out)
            else:
                result["title_query"] = None
        return result

    async def generate_leaf_queries(
        self,
        *,
        stage1_out: Dict[str, Any],
        traj_k: TrajectoryData,
        query: str,
        previous_mermaid: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        程序拆分 mermaid，然后对每个子图调用 Stage2 生成 query + query_fuzzified。

        返回: {"leaf_queries": [...]}
        """
        if not getattr(self, "_initialized", False):
            raise RuntimeError("MermaidReviserAgent 未初始化，请先调用 initialize()")

        metadata = self._to_json_safe(metadata or {})
        base_rubric = stage1_out.get("rubric") or {}

        splits = self.split_mermaid_by_leaves(stage1_out)
        if not splits:
            raise ValueError("mermaid 没有叶子节点，无法拆分")

        leaf_queries: List[Dict[str, Any]] = []
        for split_item in splits:
            sub_structure = split_item.get("mermaid_structure") or ""
            sub_node_ids = {
                (node or {}).get("node_id", "")
                for node in (split_item.get("mermaid_nodes") or [])
                if (node or {}).get("node_id")
            }
            sub_stage1_out = {
                **stage1_out,
                "mermaid": split_item["mermaid"],
                "mermaid_structure": sub_structure,
                "structure": sub_structure,
                "mermaid_nodes": split_item.get("mermaid_nodes") or [],
            }
            stage2_out = await self.generate_query_and_rubric(
                stage1_out=sub_stage1_out,
                traj_k=traj_k,
                metadata=metadata,
                focus_leaf_id=split_item["leaf_node_id"],
            )
            if not (stage2_out.get("query") or "").strip():
                raise ValueError(
                    f"LLM 为叶子 {split_item['leaf_node_id']} 生成的 query 为空"
                )
            if not (stage2_out.get("query_fuzzified") or "").strip():
                raise ValueError(
                    f"LLM 为叶子 {split_item['leaf_node_id']} 生成的 query_fuzzified 为空"
                )

            sub_rubric = self._filter_rubric_to_nodes(base_rubric, sub_node_ids)
            weighted_out = {"rubric": sub_rubric}
            self._apply_rubric_weights(
                out=weighted_out,
                stage1_out=sub_stage1_out,
                focus_leaf_id=split_item["leaf_node_id"],
            )

            leaf_queries.append({
                "leaf_node_id": split_item["leaf_node_id"],
                "leaf_node_text": split_item["leaf_node_text"],
                "mermaid": split_item["mermaid"],
                "mermaid_nodes": split_item.get("mermaid_nodes") or [],
                "rubric": weighted_out["rubric"],
                "query": stage2_out["query"],
                "query_fuzzified": stage2_out["query_fuzzified"],
            })

        return {"leaf_queries": leaf_queries}

    async def generate_query_fuzzified(
        self,
        *,
        query: str,
        metadata: Optional[Dict[str, Any]] = None,
        stage1_out: Optional[Dict[str, Any]] = None,
        traj_k: Optional[TrajectoryData] = None,
    ) -> Dict[str, Any]:
        """向后兼容：委托 fuzzify_query。"""
        _ = stage1_out
        _ = traj_k
        return await self.fuzzify_query(query=query, metadata=metadata)

    async def generate_rubric(
        self,
        *,
        stage1_mermaid: str,
        stage1_query: str,
        traj_k: TrajectoryData,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """向后兼容旧调用方。内部委托给 generate_leaf_queries。"""
        stage1_out = {
            "mermaid": stage1_mermaid,
            "structure": Mermaid.from_str(stage1_mermaid).to_structure_str(),
            "mermaid_nodes": [],
            "rubric": {"criteria": []},
            "query": stage1_query,
        }
        return await self.generate_leaf_queries(
            stage1_out=stage1_out,
            traj_k=traj_k,
            query=stage1_query,
            metadata=metadata,
        )

    async def revise_stage1(
        self,
        *,
        query: str,
        mermaid_k: Optional[str],
        traj_k: TrajectoryData,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """兼容旧接口：等价于 expand_task。"""
        return await self.expand_task(
            query=query,
            mermaid=mermaid_k,
            traj_k=traj_k,
            metadata=metadata,
        )

    async def revise_stage2(
        self,
        *,
        mermaid_k: Optional[str],
        traj_k: TrajectoryData,
        stage1_mermaid: str,
        stage1_query: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """兼容旧接口：等价于 generate_leaf_queries。"""
        _ = mermaid_k
        stage1_out = {
            "mermaid": stage1_mermaid,
            "structure": Mermaid.from_str(stage1_mermaid).to_structure_str(),
            "mermaid_nodes": [],
            "rubric": {"criteria": []},
        }
        return await self.generate_leaf_queries(
            stage1_out=stage1_out,
            traj_k=traj_k,
            query=stage1_query,
            previous_mermaid=mermaid_k,
            metadata=metadata,
        )

    async def revise(
        self,
        *,
        query: str,
        mermaid_k: Optional[str],
        traj_k: TrajectoryData,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        stage1_out = await self.expand_task(
            query=query,
            mermaid=mermaid_k,
            traj_k=traj_k,
            metadata=metadata,
        )
        stage2_out = await self.generate_query_and_rubric(
            stage1_out=stage1_out,
            traj_k=traj_k,
            metadata=metadata,
        )
        out = MermaidRevisionOutput(
            mermaid=stage1_out["mermaid"],
            structure=stage1_out.get("mermaid_structure") or stage1_out.get("structure", ""),
            mermaid_nodes=stage1_out.get("mermaid_nodes", []),
            query=stage2_out["query"],
            query_fuzzified=stage2_out["query_fuzzified"],
            leaf_queries=[],
            rubric=stage1_out["rubric"],
        ).model_dump(mode="json")
        out["stage1_query"] = stage2_out.get("query", "")
        return out

