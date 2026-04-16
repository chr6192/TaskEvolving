from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


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


@dataclass
class Mermaid:
    """
    Mermaid 的统一入口：
    - 原始字符串注入 prompt
    - str <-> 结构化 flowchart
    - 统计最长路径节点深度（best-effort）
    """

    mermaid: str

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
    def from_flowchart(cls, flowchart: MermaidFlowchart) -> "Mermaid":
        return cls(mermaid=cls.flowchart_to_str(flowchart))

    @classmethod
    def from_str(cls, mermaid: str) -> "Mermaid":
        return cls(mermaid=(mermaid or "").strip())

    def to_flowchart(self) -> MermaidFlowchart:
        lines = [ln.strip() for ln in (self.mermaid or "").splitlines() if ln.strip()]
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

    def to_str(self) -> str:
        return (self.mermaid or "").strip()

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

    def as_system_prompt_injection(self) -> str:
        return (
            "\n\n"
            "# Mermaid 约束\n"
            "你需要参考以下流程图来组织你的思考与行动步骤，确保整体流程不偏离。\n"
            "```mermaid\n"
            f"{self.mermaid}"
            "```\n"
        )

