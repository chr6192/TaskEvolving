from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MermaidNode:
    """结构化节点定义。"""

    node_id: str
    node_type: str
    check_points: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "check_points": list(self.check_points),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MermaidNode":
        return cls(
            node_id=str(data.get("node_id") or ""),
            node_type=str(data.get("node_type") or ""),
            check_points=[
                str(item)
                for item in (data.get("check_points") or [])
                if item is not None
            ],
        )


@dataclass
class MermaidEdge:
    """结构化边定义。"""

    src: str
    dst: str
    src_label: Optional[str] = None
    dst_label: Optional[str] = None


@dataclass
class MermaidFlowchart:
    """
    结构化 flowchart 定义（宽松模型）：
    - direction: TD/LR/RL/BT（默认 TD）
    - edges: 边列表；节点标签由边上的 src_label/dst_label 携带
    """

    direction: str = "TD"
    edges: list[MermaidEdge] = field(default_factory=list)


@dataclass(init=False)
class Mermaid:
    """
    Mermaid 的统一入口：
    - structure: 纯 flowchart 结构，用于解析、计算和渲染图
    - nodes: 随图传递的节点语义字段
    - to_str(): 注入 prompt 的完整 Mermaid 上下文（structure + nodes）
    - str <-> 结构化 flowchart
    - 统计最长路径节点深度（best-effort）
    """

    structure: str
    nodes: list[MermaidNode]

    def __init__(
        self,
        structure: str = "",
        nodes: Optional[list[MermaidNode | dict[str, Any]]] = None,
        mermaid: Optional[str] = None,
    ) -> None:
        # mermaid 参数仅为兼容旧调用；新代码请使用 structure。
        if mermaid is not None and not structure:
            structure = mermaid
        self.structure = (structure or "").strip()
        self.nodes = self._parse_nodes(nodes or [])

    @staticmethod
    def _parse_nodes(nodes: list[MermaidNode | dict[str, Any]]) -> list[MermaidNode]:
        parsed_nodes: list[MermaidNode] = []
        for node in nodes:
            if isinstance(node, MermaidNode):
                parsed_nodes.append(node)
            elif isinstance(node, dict):
                parsed_nodes.append(MermaidNode.from_dict(node))
        return parsed_nodes

    @property
    def mermaid(self) -> str:
        """兼容旧字段名；返回纯 flowchart 结构。"""
        return self.structure

    @staticmethod
    def _extract_node_id_and_label(side: str) -> tuple[Optional[str], Optional[str]]:
        s = side.strip()
        if not s:
            return None, None
        m = re.match(
            r'^([A-Za-z0-9_]+)\s*(?:\["([^"]*)"\]|\[([^\]]*)\]|\(([^\)]*)\)|\{([^\}]*)\})?\s*$',
            s,
        )
        if m:
            node_id = m.group(1)
            label = m.group(2) or m.group(3) or m.group(4) or m.group(5)
            return node_id, label
        m2 = re.match(r"^([A-Za-z0-9_]+)", s)
        return (m2.group(1), None) if m2 else (None, None)

    @classmethod
    def from_flowchart(
        cls,
        flowchart: MermaidFlowchart,
        nodes: Optional[list[MermaidNode]] = None,
    ) -> "Mermaid":
        return cls(
            structure=cls.flowchart_to_str(flowchart),
            nodes=nodes or [],
        )

    @classmethod
    def from_str(
        cls,
        mermaid: str,
        nodes: Optional[list[MermaidNode | dict[str, Any]]] = None,
    ) -> "Mermaid":
        text = (mermaid or "").strip()
        parsed_structure = cls._extract_fenced_block(text, "mermaid") or text
        parsed_nodes = cls._parse_nodes(nodes or [])
        if nodes is None:
            nodes_json = cls._extract_fenced_block(text, "json")
            if nodes_json:
                try:
                    raw_payload = json.loads(nodes_json)
                except Exception:
                    raw_payload = []
                if isinstance(raw_payload, dict):
                    parsed_nodes = cls._parse_nodes(raw_payload.get("nodes") or [])
                elif isinstance(raw_payload, list):
                    parsed_nodes = cls._parse_nodes(raw_payload)
        return cls(
            structure=parsed_structure,
            nodes=parsed_nodes,
        )

    @staticmethod
    def _extract_fenced_block(text: str, lang: str) -> Optional[str]:
        pattern = rf"```{re.escape(lang)}\s*\n(.*?)\n```"
        match = re.search(pattern, text or "", flags=re.DOTALL | re.IGNORECASE)
        if not match:
            return None
        return match.group(1).strip()

    def node_dicts(self) -> list[dict[str, Any]]:
        return [node.to_dict() for node in self.nodes]

    def to_flowchart(self) -> MermaidFlowchart:
        lines = [ln.strip() for ln in (self.structure or "").splitlines() if ln.strip()]
        if not lines:
            return MermaidFlowchart()

        direction = "TD"
        first = lines[0]
        if first.lower().startswith("flowchart"):
            parts = first.split()
            if len(parts) >= 2 and parts[1] in {"TD", "LR", "RL", "BT"}:
                direction = parts[1]

        edges: list[MermaidEdge] = []
        for ln in lines:
            if "-->" not in ln:
                continue
            left, right = ln.split("-->", 1)
            src, src_label = self._extract_node_id_and_label(left)
            dst, dst_label = self._extract_node_id_and_label(right)
            if not src or not dst:
                continue
            edges.append(
                MermaidEdge(
                    src=src,
                    dst=dst,
                    src_label=src_label,
                    dst_label=dst_label,
                )
            )

        return MermaidFlowchart(direction=direction, edges=edges)

    @staticmethod
    def flowchart_to_str(flowchart: MermaidFlowchart) -> str:
        direction = flowchart.direction if flowchart.direction in {"TD", "LR", "RL", "BT"} else "TD"
        lines = [f"flowchart {direction}"]
        for edge in flowchart.edges:
            src = edge.src
            dst = edge.dst
            if edge.src_label is not None:
                src = f'{src}["{edge.src_label}"]'
            if edge.dst_label is not None:
                dst = f'{dst}["{edge.dst_label}"]'
            lines.append(f"{src} --> {dst}")
        return "\n".join(lines)

    def to_structure_str(self) -> str:
        """返回纯 flowchart 结构，不包含节点字段。"""
        return (self.structure or "").strip()

    def to_str(self) -> str:
        """返回完整 Mermaid 上下文，包含 flowchart 结构和节点字段。"""
        metadata_json = json.dumps(
            {"nodes": self.node_dicts()},
            ensure_ascii=False,
            indent=2,
        )
        return (
            "## Mermaid Structure\n"
            "```mermaid\n"
            f"{self.to_structure_str()}\n"
            "```\n\n"
            "## Mermaid Metadata\n"
            "```json\n"
            f"{metadata_json}\n"
            "```"
        ).strip()

    def depth(self) -> Optional[int]:
        flow = self.to_flowchart()
        if not flow.edges:
            return None

        nodes: set[str] = set()
        for e in flow.edges:
            nodes.add(e.src)
            nodes.add(e.dst)
        if not nodes:
            return None

        in_deg = {n: 0 for n in nodes}
        adj: dict[str, list[str]] = {n: [] for n in nodes}
        for e in flow.edges:
            adj[e.src].append(e.dst)
            in_deg[e.dst] += 1

        q = deque([n for n, d in in_deg.items() if d == 0])
        dist = {n: 1 for n in nodes}
        visited = 0
        while q:
            u = q.popleft()
            visited += 1
            for v in adj[u]:
                if dist[v] < dist[u] + 1:
                    dist[v] = dist[u] + 1
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    q.append(v)

        if visited != len(nodes):
            return None
        return max(dist.values()) if dist else None

    def leaf_node_ids(self) -> list[str]:
        """返回 DAG 中所有叶子节点（出度为 0）的 ID 列表。"""
        flow = self.to_flowchart()
        if not flow.edges:
            return []

        all_nodes: set[str] = set()
        has_outgoing: set[str] = set()
        for e in flow.edges:
            all_nodes.add(e.src)
            all_nodes.add(e.dst)
            has_outgoing.add(e.src)

        return sorted(n for n in all_nodes if n not in has_outgoing)

    def _collect_ancestors(self, target: str) -> set[str]:
        """回溯收集从任意根到 target 的所有祖先节点（含 target 本身）。"""
        flow = self.to_flowchart()
        # 构建反向邻接表
        reverse_adj: dict[str, list[str]] = {}
        for e in flow.edges:
            reverse_adj.setdefault(e.dst, []).append(e.src)
            reverse_adj.setdefault(e.src, [])

        visited: set[str] = set()
        stack = [target]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for parent in reverse_adj.get(node, []):
                if parent not in visited:
                    stack.append(parent)
        return visited

    def split_by_leaves(self) -> list[tuple[str, "Mermaid"]]:
        """
        按叶子节点拆分：对每个叶子，回溯所有祖先，保留相关边，生成子 Mermaid。

        返回 [(leaf_node_id, sub_mermaid), ...] 列表。
        """
        leaves = self.leaf_node_ids()
        if not leaves:
            return []

        flow = self.to_flowchart()
        results: list[tuple[str, Mermaid]] = []
        for leaf in leaves:
            ancestor_nodes = self._collect_ancestors(leaf)
            sub_edges = [e for e in flow.edges if e.src in ancestor_nodes and e.dst in ancestor_nodes]
            sub_flow = MermaidFlowchart(direction=flow.direction, edges=sub_edges)
            sub_node_ids = {e.src for e in sub_edges} | {e.dst for e in sub_edges}
            sub_nodes = [node for node in self.nodes if node.node_id in sub_node_ids]
            results.append((
                leaf,
                Mermaid.from_str(
                    self.flowchart_to_str(sub_flow),
                    nodes=sub_nodes,
                ),
            ))
        return results

    def node_labels(self) -> dict[str, str]:
        """返回 {node_id: label} 映射（仅包含有标签的节点）。"""
        flow = self.to_flowchart()
        labels: dict[str, str] = {}
        for e in flow.edges:
            if e.src_label is not None:
                labels.setdefault(e.src, e.src_label)
            if e.dst_label is not None:
                labels.setdefault(e.dst, e.dst_label)
        return labels

    def as_system_prompt_injection(self) -> str:
        return (
            "\n\n"
            "# Mermaid 约束\n"
            "你需要参考以下完整 Mermaid 上下文来组织你的思考与行动步骤，确保整体流程不偏离。\n"
            f"{self.to_str()}\n"
        )

