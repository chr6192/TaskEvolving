from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict, Field

from deep_research.synthesis.core.data_structures import TrajectoryData
from deep_research.synthesis.core.structured_llm_client import StructuredLLMClient
from deep_research.synthesis_v2.core.language import VALID_NODE_TYPES, normalize_language
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

    mermaid: MermaidGraphSpec = Field(
        description="完整 Mermaid 对象，包含 structure 和 nodes（nodes 含本轮溯源字段，解析后不进入代码 Mermaid）"
    )
    query: str = Field(description="下一轮使用的扩展版 query（由 stage1 生成）")


class MermaidRevisionStage2Output(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_fuzzified: str = Field(description="扩展前 query 的模糊化版本")
    rubric: RubricOutputCore = Field(description="与 mermaid 对齐的 rubric")


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

    Stage1 expand_task：mermaid + query（不产出 rubric、不算权重）
    Stage2 generate_query_and_rubric：query_fuzzified + rubric（权重由程序按 mermaid 结构计算）
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

    @staticmethod
    def _looks_like_general_search_system_prompt(content: Any) -> bool:
        """
        判断是否为通用检索轨迹里冗长的 system 提示词。
        该类提示词会干扰 reviser 对真实任务轨迹的关注，需在传入 prompt 前剔除。
        """
        if not isinstance(content, str):
            return False
        text = content.strip()
        if not text:
            return False

        cn_markers = ("# 你扮演的角色", "# 当前问题(query)")
        en_markers = ("# Your role", "# Current question (query)")

        def _is_template_prompt(body: str) -> bool:
            return all(m in body for m in cn_markers) or all(m in body for m in en_markers)

        # 1) 原始 system prompt 文本
        if _is_template_prompt(text):
            return True

        # 2) 被序列化为 JSON 字符串的 {"role":"system","content":"..."} 片段
        try:
            parsed = json.loads(text)
        except Exception:
            return False
        if not isinstance(parsed, dict):
            return False
        if parsed.get("role") != "system":
            return False
        inner_content = parsed.get("content")
        return isinstance(inner_content, str) and _is_template_prompt(inner_content)

    def _sanitize_traj_for_reviser(self, traj_k: TrajectoryData) -> Dict[str, Any]:
        """
        仅对传给 reviser 的 trajectory 视图做轻量清洗：
        - 去掉通用检索模板中的长 system 提示词（包括其 JSON 串形态）
        """
        traj_dict = self._to_json_safe(traj_k.to_dict())
        messages = traj_dict.get("messages")
        if not isinstance(messages, list):
            return traj_dict

        sanitized_messages: List[Dict[str, Any]] = []
        dropped = 0
        for msg in messages:
            if not isinstance(msg, dict):
                sanitized_messages.append(msg)
                continue
            role = msg.get("role")
            content = msg.get("content")
            if role == "system" and self._looks_like_general_search_system_prompt(content):
                dropped += 1
                continue
            sanitized_messages.append(msg)

        if dropped > 0:
            logger.info("reviser trajectory 过滤了 %s 条 system 提示词消息", dropped)
        traj_dict["messages"] = sanitized_messages
        return traj_dict

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
        mermaid_nodes: Optional[List[Dict[str, Any]]],
        traj_k: TrajectoryData,
        metadata: Dict[str, Any],
    ) -> str:
        sanitized_traj = self._sanitize_traj_for_reviser(traj_k)
        builder = (
            build_revise_expand_task_prompt_en
            if self.language == "en"
            else build_revise_expand_task_prompt
        )
        return builder(
            mermaid_k=mermaid,
            mermaid_nodes=mermaid_nodes,
            traj_k=sanitized_traj,
            metadata=metadata,
        )

    def _build_revise_rubric_prompt(
        self,
        *,
        query: str,
        mermaid: str,
        metadata: Dict[str, Any],
    ) -> str:
        builder = (
            build_revise_rubric_prompt_en
            if self.language == "en"
            else build_revise_rubric_prompt
        )
        return builder(
            query=query,
            mermaid=mermaid,
            metadata=metadata,
        )

    def _build_leaf_query_prompt(
        self,
        *,
        stage1_query: str,
        sub_mermaid: str,
        sub_rubric: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> str:
        builder = build_leaf_query_prompt_en if self.language == "en" else build_leaf_query_prompt
        return builder(
            stage1_query=stage1_query,
            sub_mermaid=sub_mermaid,
            sub_rubric=sub_rubric,
            metadata=metadata,
        )

    def _validate_stage2_output(
        self,
        *,
        out: Dict[str, Any],
        stage1_out: Dict[str, Any],
    ) -> None:
        query_fuzzified = (out.get("query_fuzzified") or "").strip()
        if not query_fuzzified:
            raise ValueError("stage2 输出 query_fuzzified 为空")

        rubric = out.get("rubric") or {}
        criteria = rubric.get("criteria") or []
        if not criteria:
            raise ValueError("stage2 输出 rubric.criteria 为空")

        mermaid_nodes = stage1_out.get("mermaid_nodes") or []
        if len(criteria) != len(mermaid_nodes):
            raise ValueError(
                f"stage2 rubric 节点数不一致: criteria={len(criteria)} nodes={len(mermaid_nodes)}"
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
                f"stage2 rubric node_id 与 mermaid 节点不匹配: {sorted(actual_ids)} vs {sorted(expected_ids)}"
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
        mermaid: Optional[str],
        mermaid_nodes: Optional[List[Dict[str, Any]]] = None,
        traj_k: TrajectoryData,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not getattr(self, "_initialized", False):
            raise RuntimeError("MermaidReviserAgent 未初始化，请先调用 initialize()")

        metadata = self._to_json_safe(metadata or {})
        prompt = self._build_expand_task_prompt(
            mermaid=mermaid,
            mermaid_nodes=mermaid_nodes,
            traj_k=traj_k,
            metadata=metadata,
        )

        parsed = await self.llm_client.call_with_structured_output(
            prompt=prompt,
            response_model=MermaidRevisionStage1Output,
            system_message="Your output must strictly conform to the provided structured schema.",
        )
        out = parsed.model_dump(mode="json")
        mermaid_payload = out.get("mermaid") or {}
        if not isinstance(mermaid_payload, dict):
            raise ValueError("reviser 输出 mermaid 必须是对象")
        structure = (mermaid_payload.get("structure") or "").strip()
        mermaid_nodes_raw = mermaid_payload.get("nodes") or []
        self._validate_mermaid(structure)
        if not (out.get("query") or "").strip():
            raise ValueError("reviser stage1 输出 query 为空")
        out["structure"] = structure
        out["mermaid_nodes"] = mermaid_nodes_raw
        self._validate_mermaid_nodes(out)
        mermaid_obj = Mermaid.from_str(structure, nodes=mermaid_nodes_raw)
        out["mermaid_nodes"] = mermaid_obj.node_dicts()
        out["mermaid_structure"] = mermaid_obj.to_structure_str()
        out["mermaid"] = mermaid_obj.to_str()
        return out

    async def generate_query_and_rubric(
        self,
        *,
        stage1_out: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        focus_leaf_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """基于 stage1 的 query 与 mermaid，生成 query_fuzzified 和 rubric。"""
        if not getattr(self, "_initialized", False):
            raise RuntimeError("MermaidReviserAgent 未初始化，请先调用 initialize()")

        metadata = self._to_json_safe(metadata or {})
        query = (stage1_out.get("query") or "").strip()
        if not query:
            raise ValueError("stage1 输出 query 为空，无法生成 query_fuzzified 和 rubric")
        prompt = self._build_revise_rubric_prompt(
            query=query,
            mermaid=stage1_out.get("mermaid") or "",
            metadata=metadata,
        )
        parsed = await self.llm_client.call_with_structured_output(
            prompt=prompt,
            response_model=MermaidRevisionStage2Output,
            system_message="Your output must strictly conform to the provided structured schema.",
        )
        out = parsed.model_dump(mode="json")
        self._validate_stage2_output(out=out, stage1_out=stage1_out)
        composed_query_id = (
            f"reviser_{metadata.get('evolution_id', 'unknown')}_{metadata.get('iteration_id', 'unknown')}"
        )
        out["rubric"]["query_id"] = composed_query_id
        self._apply_rubric_weights(
            out=out,
            stage1_out=stage1_out,
            focus_leaf_id=focus_leaf_id,
        )
        self._attach_rubric_metadata(out=out, metadata=metadata)
        return out

    async def generate_leaf_queries(
        self,
        *,
        stage1_out: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        程序拆分 mermaid，然后对每个子图调用 Stage2 生成 query_fuzzified + rubric。

        返回: {"leaf_queries": [...]}
        """
        if not getattr(self, "_initialized", False):
            raise RuntimeError("MermaidReviserAgent 未初始化，请先调用 initialize()")

        metadata = self._to_json_safe(metadata or {})

        # 1. 程序化拆分
        splits = self.split_mermaid_by_leaves(stage1_out)
        if not splits:
            raise ValueError("mermaid 没有叶子节点，无法拆分")

        # 2. 对每个子图调用 LLM 生成 query + rubric
        leaf_queries: List[Dict[str, Any]] = []
        for split_item in splits:
            sub_structure = split_item.get("mermaid_structure") or ""
            stage2_out = await self.generate_query_and_rubric(
                stage1_out={
                    **stage1_out,
                    "mermaid": split_item["mermaid"],
                    "mermaid_structure": sub_structure,
                    "structure": sub_structure,
                    "mermaid_nodes": split_item.get("mermaid_nodes") or [],
                },
                metadata=metadata,
                focus_leaf_id=split_item["leaf_node_id"],
            )
            if not (stage2_out.get("query_fuzzified") or "").strip():
                raise ValueError(
                    f"LLM 为叶子 {split_item['leaf_node_id']} 生成的 query_fuzzified 为空"
                )

            leaf_queries.append({
                "leaf_node_id": split_item["leaf_node_id"],
                "leaf_node_text": split_item["leaf_node_text"],
                "mermaid": split_item["mermaid"],
                "mermaid_nodes": split_item.get("mermaid_nodes") or [],
                "rubric": stage2_out["rubric"],
                "query_fuzzified": stage2_out["query_fuzzified"],
            })

        return {"leaf_queries": leaf_queries}

    async def generate_query_fuzzified(
        self,
        *,
        stage1_out: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """向后兼容：委托 generate_query_and_rubric。"""
        return await self.generate_query_and_rubric(
            stage1_out=stage1_out,
            metadata=metadata,
        )

    async def generate_rubric(
        self,
        *,
        stage1_mermaid: str,
        stage1_query: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """向后兼容旧调用方。内部委托给 generate_leaf_queries。"""
        stage1_out = {
            "mermaid": stage1_mermaid,
            "structure": Mermaid.from_str(stage1_mermaid).to_structure_str(),
            "mermaid_nodes": [],
            "query": stage1_query,
        }
        return await self.generate_leaf_queries(stage1_out=stage1_out, metadata=metadata)

    async def revise_stage1(
        self,
        *,
        mermaid_k: Optional[str],
        mermaid_nodes: Optional[List[Dict[str, Any]]] = None,
        traj_k: TrajectoryData,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """兼容旧接口：等价于 expand_task。"""
        return await self.expand_task(
            mermaid=mermaid_k,
            mermaid_nodes=mermaid_nodes,
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
            mermaid_nodes=None,
            traj_k=traj_k,
            metadata=metadata,
        )
        stage2_out = await self.generate_query_and_rubric(
            stage1_out=stage1_out,
            metadata=metadata,
        )
        out = MermaidRevisionOutput(
            mermaid=stage1_out["mermaid"],
            structure=stage1_out.get("mermaid_structure") or stage1_out.get("structure", ""),
            mermaid_nodes=stage1_out.get("mermaid_nodes", []),
            query=stage1_out["query"],
            query_fuzzified=stage2_out["query_fuzzified"],
            leaf_queries=[],
            rubric=stage2_out["rubric"],
        ).model_dump(mode="json")
        out["stage1_query"] = stage1_out.get("query", "")
        return out

