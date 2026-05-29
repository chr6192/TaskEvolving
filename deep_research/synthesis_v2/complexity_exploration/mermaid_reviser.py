from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict, Field

from deep_research.synthesis.core.data_structures import TrajectoryData
from deep_research.synthesis.core.structured_llm_client import StructuredLLMClient
from deep_research.synthesis_v2.prompts.revise_expand_task_prompt import (
    build_revise_expand_task_prompt,
)
from deep_research.synthesis_v2.prompts.revise_rubric_prompt import (
    build_leaf_query_prompt,
)
from ..mermaid import Mermaid

logger = logging.getLogger(__name__)


class RubricCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="唯一ID（稳定）")
    description: str = Field(description="评判标准描述")
    weight: float = Field(description="权重(0-1)", ge=0.0, le=1.0)
    check_points: List[str] = Field(default_factory=list, description="检查点列表")


class RubricOutputCore(BaseModel):
    """结构化输出用（不含自由 metadata，满足 strict json_schema）。"""

    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(description="与 query 对齐的 query_id")
    criteria: List[RubricCriterion] = Field(description="rubric checklist")


class RubricOutput(BaseModel):
    query_id: str = Field(description="与 query 对齐的 query_id")
    criteria: List[RubricCriterion] = Field(description="rubric checklist")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NodeRubricCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="节点唯一对应的 rubric criterion id")
    description: str = Field(description="criterion 描述")
    check_points: List[str] = Field(default_factory=list, description="criterion 检查点")


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
    leaf_queries: List["MermaidLeafQuery"] = Field(
        default_factory=list,
        description="按叶子节点拆分后的子 mermaid 与对应 query 列表",
    )
    rubric: RubricOutput = Field(description="OpenJudge 兼容 rubric（dict 形态）")


class MermaidNodeSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(description="节点ID（需与mermaid中的节点ID一致）")
    node_text: str = Field(description="节点完整文本")
    rubric_criterion: NodeRubricCriterion = Field(
        description="该节点唯一对应的 rubric criterion（节点与criterion一一对应）",
    )
    source_type: str = Field(
        description="节点来源类型：tool_call_response / query_or_reason_or_answer / previous_mermaid"
    )
    source_note: str = Field(description="来源+证据的合成描述")


class MermaidRevisionStage1Output(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mermaid: str = Field(description="DAG Mermaid flowchart 文本（可直接放入 prompt）")
    query: str = Field(
        description="下一轮任务提问稿：尽量保留 Mermaid 所体现的多维度、依赖与约束，完整体现扩充后的复杂度，不刻意模糊，但仍不写逐步路径"
    )
    node_sources: List[MermaidNodeSource] = Field(
        default_factory=list,
        description="每个节点的来源溯源信息（一节点一条）",
    )


class LeafQueryOutput(BaseModel):
    """单条叶子子图的 LLM 输出：仅 query_fuzzified。"""

    model_config = ConfigDict(extra="forbid")

    query_fuzzified: str = Field(description="query 的用户口吻模糊版")


class MermaidLeafQuery(BaseModel):
    """叶子拆分结果（程序拆分 + LLM 生成 query_fuzzified）。"""

    leaf_node_id: str = Field(description="原始 mermaid 叶子节点ID")
    leaf_node_text: str = Field(description="原始 mermaid 叶子节点文本")
    mermaid: str = Field(description="通向该叶子节点的子 mermaid（flowchart 语法）")
    rubric: Dict[str, Any] = Field(default_factory=dict, description="子 rubric")
    query_fuzzified: str = Field(default="", description="query 的用户口吻模糊版")


@dataclass
class MermaidReviserAgent:
    """
    Mermaid + Query + Rubric 统一产出 reviser（用模型生成）。

    输入：mermaid_k（可空）、traj_k、metadata（可选；其中可包含 evaluation_output）
    输出：{"mermaid": str, "query": str, "node_sources": list, "query_fuzzified": str, "leaf_queries": list, "rubric": dict}
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

        # 1) 原始 system prompt 文本
        if "# 你扮演的角色" in text and "# 当前问题(query)" in text:
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
        return isinstance(inner_content, str) and (
            "# 你扮演的角色" in inner_content and "# 当前问题(query)" in inner_content
        )

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
    def split_mermaid_and_rubric(
        stage1_out: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        纯程序化拆分：按叶子节点拆 mermaid + rubric。

        输入：expand_task() 的输出（含 mermaid, rubric, node_sources）。
        输出：[{leaf_node_id, leaf_node_text, mermaid(str), rubric(dict)}, ...]

        拆分逻辑：
        1. 解析 mermaid 找到所有叶子节点；
        2. 对每个叶子回溯祖先，生成子 mermaid；
        3. 从 node_sources 收集子图涉及节点的 rubric_criterion_id；
        4. 从 rubric.criteria 过滤出对应子集，组成子 rubric。
        """
        mermaid_obj = Mermaid.from_str(stage1_out.get("mermaid") or "")
        splits = mermaid_obj.split_by_leaves()
        if not splits:
            return []

        # 构建 node_id -> rubric_criterion_id 映射
        node_sources = stage1_out.get("node_sources") or []
        node_to_criterion: Dict[str, str] = {}
        for ns in node_sources:
            nid = (ns or {}).get("node_id", "")
            criterion = (ns or {}).get("rubric_criterion") or {}
            cid = criterion.get("id", "")
            if nid and cid:
                node_to_criterion[nid] = cid

        # 全量 rubric criteria, 以 id 为 key
        rubric = stage1_out.get("rubric") or {}
        all_criteria = rubric.get("criteria") or []
        criteria_by_id: Dict[str, Any] = {}
        for c in all_criteria:
            if isinstance(c, dict) and c.get("id"):
                criteria_by_id[c["id"]] = c

        # 节点标签（用于 leaf_node_text）
        labels = mermaid_obj.node_labels()

        results: List[Dict[str, Any]] = []
        for leaf_id, sub_mermaid in splits:
            # 子图包含的节点集合
            sub_flow = sub_mermaid.to_flowchart()
            sub_node_ids: set[str] = set()
            for e in sub_flow.edges:
                sub_node_ids.add(e.src)
                sub_node_ids.add(e.dst)

            # --- 1) 按“叶子向上分配”规则计算节点权重 ---
            # 规则：
            # - 叶子节点初始权重 1；
            # - 节点若有 n 个上游父节点，则向每个父节点分配 node_weight / n；
            # - 节点值为来自所有下游分配的权重和；
            # - 最后对所有节点权重做 sum 归一化。
            parents: Dict[str, List[str]] = {nid: [] for nid in sub_node_ids}
            indeg: Dict[str, int] = {nid: 0 for nid in sub_node_ids}
            for e in sub_flow.edges:
                parents[e.dst].append(e.src)
                indeg[e.dst] += 1

            # 拓扑排序后逆序传播（从下游到上游）
            topo: List[str] = []
            q: List[str] = [nid for nid, d in indeg.items() if d == 0]
            # 使用简单 list 队列足够（节点规模小）
            while q:
                u = q.pop(0)
                topo.append(u)
                for edge in sub_flow.edges:
                    if edge.src == u:
                        indeg[edge.dst] -= 1
                        if indeg[edge.dst] == 0:
                            q.append(edge.dst)

            node_weights: Dict[str, float] = {nid: 0.0 for nid in sub_node_ids}
            if leaf_id in node_weights:
                node_weights[leaf_id] = 1.0

            for nid in reversed(topo):
                w = node_weights.get(nid, 0.0)
                p = parents.get(nid, [])
                if not p or w <= 0:
                    continue
                share = w / float(len(p))
                for pid in p:
                    node_weights[pid] = node_weights.get(pid, 0.0) + share

            # 节点权重 sum 归一化
            total_node_weight = sum(node_weights.values())
            if total_node_weight > 0:
                node_weights = {nid: w / total_node_weight for nid, w in node_weights.items()}

            # 收集该子图涉及的 rubric criteria ids
            relevant_criteria_ids: set[str] = set()
            for nid in sub_node_ids:
                cid = node_to_criterion.get(nid)
                if cid:
                    relevant_criteria_ids.add(cid)

            # 构建子 rubric
            sub_criteria = [criteria_by_id[cid] for cid in relevant_criteria_ids if cid in criteria_by_id]

            # --- 2) 用节点权重反推 criteria.weight ---
            # 一节点一 criterion：节点权重直接累加到对应 criterion。
            criterion_weights: Dict[str, float] = {
                c["id"]: 0.0 for c in sub_criteria if isinstance(c, dict) and c.get("id")
            }
            for nid, nw in node_weights.items():
                cid = node_to_criterion.get(nid)
                if not cid or cid not in criterion_weights or nw <= 0:
                    continue
                criterion_weights[cid] += nw

            # criterion 层再做一次归一化，确保最终 rubric weights 可直接使用
            total_criterion_weight = sum(criterion_weights.values())
            if total_criterion_weight > 0:
                criterion_weights = {
                    cid: w / total_criterion_weight for cid, w in criterion_weights.items()
                }

            sub_criteria_with_weight: List[Dict[str, Any]] = []
            for c in sub_criteria:
                cid = c.get("id")
                c_new = dict(c)
                if cid in criterion_weights:
                    c_new["weight"] = float(criterion_weights[cid])
                sub_criteria_with_weight.append(c_new)

            sub_rubric = {
                "query_id": rubric.get("query_id", ""),
                "criteria": sub_criteria_with_weight,
            }

            results.append({
                "leaf_node_id": leaf_id,
                "leaf_node_text": labels.get(leaf_id, leaf_id),
                "mermaid": sub_mermaid.to_str(),
                "rubric": sub_rubric,
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
        traj_k: TrajectoryData,
        metadata: Dict[str, Any],
    ) -> str:
        sanitized_traj = self._sanitize_traj_for_reviser(traj_k)
        return build_revise_expand_task_prompt(
            mermaid_k=mermaid,
            traj_k=sanitized_traj,
            metadata=metadata,
        )

    def _build_leaf_query_prompt(
        self,
        *,
        sub_mermaid: str,
        sub_rubric: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> str:
        return build_leaf_query_prompt(
            sub_mermaid=sub_mermaid,
            sub_rubric=sub_rubric,
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

    def _validate_node_sources_rubric_mapping(self, out: Dict[str, Any]) -> None:
        node_sources = out.get("node_sources") or []
        if not node_sources:
            raise ValueError("reviser 输出 node_sources 为空")

        used_criterion_ids: set[str] = set()
        for node in node_sources:
            node_id = (node or {}).get("node_id", "<unknown>")
            criterion = (node or {}).get("rubric_criterion") or {}
            criterion_id = criterion.get("id", "")
            check_points = criterion.get("check_points") or []
            description = (criterion.get("description") or "").strip()

            if not criterion_id:
                raise ValueError(f"节点 {node_id} 缺少 rubric_criterion.id")
            if not description:
                raise ValueError(f"节点 {node_id} 的 rubric_criterion.description 为空")
            if not check_points:
                raise ValueError(f"节点 {node_id} 的 rubric_criterion.check_points 为空")
            if criterion_id in used_criterion_ids:
                raise ValueError(
                    f"rubric_criterion.id 被多个节点复用（违反一对一约束）: {criterion_id}"
                )
            used_criterion_ids.add(criterion_id)

    def _compose_rubric_from_node_sources(
        self,
        *,
        out: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        从 node_sources 自动组装 rubric（单一事实源）。
        权重按整图“从底向上”分配：
        1) 最后一层叶子节点先均分；
        2) 每个节点将自身权重均分给所有父节点；
        3) 节点层归一化后映射到 criterion.weight。
        """
        node_sources = out.get("node_sources") or []
        criteria: List[Dict[str, Any]] = []
        node_to_criterion: Dict[str, str] = {}
        for node in node_sources:
            criterion = dict((node or {}).get("rubric_criterion") or {})
            node_id = (node or {}).get("node_id", "")
            criterion_id = criterion.get("id", "")
            if criterion:
                criteria.append(criterion)
            if node_id and criterion_id:
                node_to_criterion[node_id] = criterion_id

        n = len(criteria)
        if n <= 0:
            raise ValueError("无法从 node_sources 组装 rubric：criteria 为空")

        # 默认兜底等权（当 mermaid 非法/无边/无叶子/非 DAG 时使用）。
        criterion_weights: Dict[str, float] = {
            c["id"]: 1.0 / float(n)
            for c in criteria
            if isinstance(c, dict) and c.get("id")
        }

        mermaid_obj = Mermaid.from_str(out.get("mermaid") or "")
        flow = mermaid_obj.to_flowchart()
        if flow.edges:
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
            if leaves:
                # 叶子先均分，然后向上分配。
                node_weights: Dict[str, float] = {nid: 0.0 for nid in nodes}
                leaf_weight = 1.0 / float(len(leaves))
                for leaf_id in leaves:
                    node_weights[leaf_id] = leaf_weight

                topo: List[str] = []
                q: List[str] = [nid for nid, d in indeg.items() if d == 0]
                while q:
                    u = q.pop(0)
                    topo.append(u)
                    for edge in flow.edges:
                        if edge.src == u:
                            indeg[edge.dst] -= 1
                            if indeg[edge.dst] == 0:
                                q.append(edge.dst)

                # 仅在 DAG 时启用该权重计算；否则保留等权兜底。
                if len(topo) == len(nodes):
                    for nid in reversed(topo):
                        w = node_weights.get(nid, 0.0)
                        p = parents.get(nid, [])
                        if not p or w <= 0:
                            continue
                        share = w / float(len(p))
                        for pid in p:
                            node_weights[pid] = node_weights.get(pid, 0.0) + share

                    total_node_weight = sum(node_weights.values())
                    if total_node_weight > 0:
                        node_weights = {
                            nid: w / total_node_weight for nid, w in node_weights.items()
                        }

                        criterion_weights = {
                            c["id"]: 0.0
                            for c in criteria
                            if isinstance(c, dict) and c.get("id")
                        }
                        for nid, nw in node_weights.items():
                            cid = node_to_criterion.get(nid)
                            if not cid or cid not in criterion_weights or nw <= 0:
                                continue
                            criterion_weights[cid] += nw

                        total_criterion_weight = sum(criterion_weights.values())
                        if total_criterion_weight > 0:
                            criterion_weights = {
                                cid: w / total_criterion_weight
                                for cid, w in criterion_weights.items()
                            }
                        else:
                            criterion_weights = {
                                c["id"]: 1.0 / float(n)
                                for c in criteria
                                if isinstance(c, dict) and c.get("id")
                            }

        criteria_with_weight: List[Dict[str, Any]] = []
        for c in criteria:
            c_new = dict(c)
            cid = c_new.get("id")
            c_new["weight"] = float(criterion_weights.get(cid, 0.0))
            criteria_with_weight.append(c_new)

        composed_query_id = f"reviser_{metadata.get('evolution_id', 'unknown')}_{metadata.get('iteration_id', 'unknown')}"
        return {
            "query_id": composed_query_id,
            "criteria": criteria_with_weight,
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
        self._validate_node_sources_rubric_mapping(out)
        out["rubric"] = self._compose_rubric_from_node_sources(out=out, metadata=metadata)
        self._attach_rubric_metadata(out=out, metadata=metadata)
        return out

    async def generate_leaf_queries(
        self,
        *,
        stage1_out: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        程序拆分 mermaid + rubric，然后对每个子图调用 LLM 生成 query。

        返回: {"leaf_queries": [...]}
        """
        if not getattr(self, "_initialized", False):
            raise RuntimeError("MermaidReviserAgent 未初始化，请先调用 initialize()")

        metadata = self._to_json_safe(metadata or {})

        # 1. 程序化拆分
        splits = self.split_mermaid_and_rubric(stage1_out)
        if not splits:
            raise ValueError("mermaid 没有叶子节点，无法拆分")

        # 2. 对每个子图调用 LLM 生成 query_fuzzified
        leaf_queries: List[Dict[str, Any]] = []
        for split_item in splits:
            prompt = self._build_leaf_query_prompt(
                sub_mermaid=split_item["mermaid"],
                sub_rubric=split_item["rubric"],
                metadata=metadata,
            )
            parsed = await self.llm_client.call_with_structured_output(
                prompt=prompt,
                response_model=LeafQueryOutput,
                system_message="Your output must strictly conform to the provided structured schema.",
            )
            llm_out = parsed.model_dump(mode="json")
            if not (llm_out.get("query_fuzzified") or "").strip():
                raise ValueError(
                    f"LLM 为叶子 {split_item['leaf_node_id']} 生成的 query_fuzzified 为空"
                )

            leaf_queries.append({
                "leaf_node_id": split_item["leaf_node_id"],
                "leaf_node_text": split_item["leaf_node_text"],
                "mermaid": split_item["mermaid"],
                "rubric": split_item["rubric"],
                "query_fuzzified": llm_out["query_fuzzified"],
            })

        return {"leaf_queries": leaf_queries}

    async def generate_query_fuzzified(
        self,
        *,
        stage1_out: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        不拆分子图，直接基于整张 mermaid + rubric 生成一条 query_fuzzified。
        """
        if not getattr(self, "_initialized", False):
            raise RuntimeError("MermaidReviserAgent 未初始化，请先调用 initialize()")

        metadata = self._to_json_safe(metadata or {})
        sub_mermaid = (stage1_out.get("mermaid") or "").strip()
        sub_rubric = stage1_out.get("rubric") or {"query_id": "", "criteria": []}

        prompt = self._build_leaf_query_prompt(
            sub_mermaid=sub_mermaid,
            sub_rubric=sub_rubric,
            metadata=metadata,
        )
        parsed = await self.llm_client.call_with_structured_output(
            prompt=prompt,
            response_model=LeafQueryOutput,
            system_message="Your output must strictly conform to the provided structured schema.",
        )
        llm_out = parsed.model_dump(mode="json")
        query_fuzzified = (llm_out.get("query_fuzzified") or "").strip()
        if not query_fuzzified:
            raise ValueError("LLM 生成的 query_fuzzified 为空")
        return {"query_fuzzified": query_fuzzified}

    async def generate_rubric(
        self,
        *,
        stage1_mermaid: str,
        stage1_query: str,
        stage1_rubric: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """向后兼容旧调用方。内部委托给 generate_leaf_queries。"""
        # 构造一个最小 stage1_out 供 split 使用
        stage1_out = {
            "mermaid": stage1_mermaid,
            "query": stage1_query,
            "rubric": stage1_rubric or {"query_id": "", "criteria": []},
            "node_sources": [],
        }
        return await self.generate_leaf_queries(stage1_out=stage1_out, metadata=metadata)

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
            traj_k=traj_k,
            metadata=metadata,
        )
        stage2_out = await self.generate_leaf_queries(
            stage1_out=stage1_out,
            metadata=metadata,
        )
        leaf_queries = stage2_out.get("leaf_queries", [])
        # 取第一个叶子的 query_fuzzified 作为总述（向后兼容）
        query_fuzzified = leaf_queries[0].get("query_fuzzified", "") if leaf_queries else ""
        out = MermaidRevisionOutput(
            mermaid=stage1_out["mermaid"],
            query=stage1_out["query"],
            node_sources=stage1_out.get("node_sources", []),
            query_fuzzified=query_fuzzified,
            leaf_queries=leaf_queries,
            rubric=stage1_out["rubric"],
        ).model_dump(mode="json")
        return out

