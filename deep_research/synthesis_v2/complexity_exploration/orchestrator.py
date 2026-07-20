"""
复杂度探索编排器
调用新的pipeline系统实现复杂度探索
"""
import logging
import uuid
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from omegaconf import DictConfig

from ..core.data_structures import WorkflowInput
from ..core.batch_processor import BatchProcessor
from ..core.pipeline import SamplePipeline
from ..core.language import is_reasoning_node_type
from ..io.path_utils import resolve_under_repo
from ..io.streaming_writer import StreamingWriter
from deep_research.synthesis.core.data_structures import (
    QueryData,
    QueryEvolution,
    IterationData,
    TrajectoryData,
    ToolCall,
    ToolResponse,
)
from .mermaid_reviser import MermaidReviserAgent
from ..mermaid import Mermaid

logger = logging.getLogger(__name__)

_DIRNAME_SAFE_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _safe_dirname(name: str, max_len: int = 120) -> str:
    """
    将任意 query_id 清洗为安全的目录名（跨平台）。
    """
    name = (name or "").strip()
    if not name:
        return "empty_query_id"
    cleaned = _DIRNAME_SAFE_RE.sub("_", name)
    cleaned = cleaned.strip("._-")
    if not cleaned:
        cleaned = "query"
    return cleaned[:max_len]


def _allocate_unique_dir(base_dir):
    """
    若 base_dir 已存在，则追加 __2/__3...，避免覆盖。
    """
    if not base_dir.exists():
        return base_dir
    i = 2
    while True:
        candidate = base_dir.parent / f"{base_dir.name}__{i}"
        if not candidate.exists():
            return candidate
        i += 1


# 正常收敛：结构稳定 / 推理链过长等预期停止，不应续跑。
NORMAL_CONVERGENCE_REASON_PREFIXES = (
    "mermaid_depth_and_node_count_unchanged_for_2_rounds",
    "reasoning_chain_too_long",
)


class ComplexityExplorationV2:
    """使用新pipeline的复杂度探索编排器"""
    
    def __init__(
        self,
        pipeline: SamplePipeline,
        reviser: MermaidReviserAgent,
        config: DictConfig
    ):
        """
        Args:
            pipeline: 单条样本pipeline
            reviser: Mermaid+Query+Rubric 修订器
            config: 配置
        """
        self.pipeline = pipeline
        self.reviser = reviser
        self.config = config
        self.exploration_config = config.get('complexity_exploration', {})
        
        self.max_iterations = self.exploration_config.get('max_iterations', 5)
        self.max_consecutive_reasoning_nodes = int(
            self.exploration_config.get("max_consecutive_reasoning_nodes", 5)
        )

        # 用于复杂度探索中的 Best-of-N（每轮 iteration 内部采样 N 次再选择 1 条）
        try:
            self.default_best_of_n = int(self.config.get("batch_processor", {}).get("best_of_n", 1))
        except Exception:
            self.default_best_of_n = 1
        
        logger.info(
            f"ComplexityExplorationV2初始化: "
            f"max_iterations={self.max_iterations}"
        )

    @staticmethod
    def _build_iteration_summary(
        iteration_id: int,
        query: str,
        query_fuzzified: Optional[str],
        mermaid: Optional[Mermaid],
        rubric: Optional[dict],
        title_query: Optional[str] = None,
    ) -> dict:
        summary = {
            "iteration_id": iteration_id,
            "query": query,
            "query_fuzzified": query_fuzzified or query,
            "mermaid": mermaid.to_str() if mermaid else None,
            "mermaid_structure": mermaid.to_structure_str() if mermaid else None,
            "mermaid_nodes": mermaid.node_dicts() if mermaid else [],
            "depth": mermaid.depth() if mermaid else None,
            "node_count": len(mermaid.node_dicts()) if mermaid else None,
            "rubric": rubric,
        }
        if title_query is not None:
            summary["title_query"] = title_query
        return summary

    @staticmethod
    def _build_exploration_stop_record(
        evolution: QueryEvolution,
        *,
        stop_at_iteration_id: Optional[int] = None,
    ) -> dict:
        return {
            "converged": evolution.converged,
            "reason": evolution.convergence_reason or "completed",
            "stop_at_iteration_id": stop_at_iteration_id,
            "pipeline_iterations_completed": len(evolution.iterations),
            "evolution_id": evolution.evolution_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    @staticmethod
    def _write_iterations_file(
        path,
        summaries: List[dict],
        exploration_stop: Optional[dict] = None,
    ) -> None:
        payload: dict = {"iterations": summaries}
        if exploration_stop is not None:
            payload["exploration_stop"] = exploration_stop
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _build_reviser_stage1_record(
        *,
        iteration_id: int,
        query: str,
        mermaid: Optional[Mermaid],
        stage1_output: Optional[dict],
        metadata: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> dict:
        record = {
            "iteration_id": iteration_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input": {
                "query": query,
                "mermaid": mermaid.to_str() if mermaid else None,
                "mermaid_structure": mermaid.to_structure_str() if mermaid else None,
                "mermaid_nodes": mermaid.node_dicts() if mermaid else [],
            },
            "stage1_output": stage1_output,
            "metadata": metadata or {},
        }
        if error:
            record["error"] = error
        return record

    @staticmethod
    def _write_reviser_summaries(path, summaries: List[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)

    @staticmethod
    def find_evolution_dir(output_root, query_id: str):
        """查找 seed 对应的已有探索目录（优先无 __ 后缀）。"""
        base = _safe_dirname(query_id)
        direct = output_root / base
        if direct.is_dir() and (direct / "iterations.json").is_file():
            return direct
        for candidate in sorted(output_root.glob(f"{base}__*")):
            if candidate.is_dir() and (candidate / "iterations.json").is_file():
                return candidate
        return None

    @staticmethod
    def _structure_snapshot(summary: dict) -> tuple[Optional[int], Optional[int]]:
        depth = summary.get("depth")
        node_count = summary.get("node_count")
        if node_count is None:
            nodes = summary.get("mermaid_nodes")
            if isinstance(nodes, list):
                node_count = len(nodes)
        if isinstance(depth, float):
            depth = int(depth)
        return depth, node_count

    @classmethod
    def _structure_unchanged(cls, newer: dict, older: dict) -> bool:
        """与 explore 主循环一致：depth 与 node_count 均存在且相等才算结构不变。"""
        newer_depth, newer_count = cls._structure_snapshot(newer)
        older_depth, older_count = cls._structure_snapshot(older)
        if (
            newer_depth is None
            or older_depth is None
            or newer_count is None
            or older_count is None
        ):
            return False
        return newer_depth == older_depth and newer_count == older_count

    @classmethod
    def _compute_no_structure_change_streak(cls, summaries: List[dict]) -> int:
        """
        从 iterations 汇总回溯连续结构不变轮数。
        summaries[i+1] vs summaries[i] 对应第 i 轮 reviser 结束时的比较口径。
        """
        if len(summaries) < 2:
            return 0
        streak = 0
        for i in range(len(summaries) - 1, 0, -1):
            if cls._structure_unchanged(summaries[i], summaries[i - 1]):
                streak += 1
            else:
                break
        return streak

    @staticmethod
    def _mermaid_from_summary(summary: dict) -> Optional[Mermaid]:
        mermaid_str = summary.get("mermaid")
        if not isinstance(mermaid_str, str) or not mermaid_str.strip():
            return None
        nodes = summary.get("mermaid_nodes") or []
        return Mermaid.from_str(mermaid_str, nodes=nodes)

    @staticmethod
    def _query_from_summary(summary: dict, seed_query: QueryData) -> QueryData:
        return QueryData(
            query_id=seed_query.query_id,
            content=str(summary.get("query") or seed_query.content),
            metadata={
                **(seed_query.metadata or {}),
                "query_fuzzified": summary.get("query_fuzzified") or summary.get("query"),
            },
        )

    @classmethod
    def _is_normal_convergence_reason(cls, reason: str) -> bool:
        reason = (reason or "").strip()
        if not reason or reason == "completed":
            return True
        if reason == "resume_start_beyond_max_iterations":
            return True
        return any(reason.startswith(prefix) for prefix in NORMAL_CONVERGENCE_REASON_PREFIXES)

    @staticmethod
    def _max_traj_iteration_id(evolution_dir: Path) -> Optional[int]:
        traj_ids: List[int] = []
        for path in evolution_dir.glob("traj_*.jsonl"):
            suffix = path.stem.removeprefix("traj_")
            if suffix.isdigit():
                traj_ids.append(int(suffix))
        return max(traj_ids) if traj_ids else None

    @classmethod
    def _resolve_resume_position(
        cls,
        summaries: List[dict],
        stop: dict,
        evolution_dir: Path,
    ) -> dict:
        """
        解析续跑位置。

        - pending：reviser 已完成，下一轮 pipeline 尚未跑（如 max_iterations_reached）
        - retry：当前 iteration 中途失败，从该轮重新跑（如 error / timeout）
        """
        if not summaries:
            return {"status": "empty_iterations", "evolution_dir": str(evolution_dir)}

        reason = str(stop.get("reason") or "")
        has_stop = bool(stop)
        if has_stop and cls._is_normal_convergence_reason(reason):
            return {
                "status": "not_resumable",
                "evolution_dir": str(evolution_dir),
                "reason": reason,
            }

        last = summaries[-1]
        last_id = last.get("iteration_id")
        if not isinstance(last_id, int):
            return {
                "status": "invalid_last_iteration_id",
                "evolution_dir": str(evolution_dir),
                "reason": reason or "missing_exploration_stop",
            }

        stop_at = stop.get("stop_at_iteration_id") if has_stop else None
        resume_mode: Optional[str] = None
        resume_from: Optional[int] = None

        if has_stop and isinstance(stop_at, int):
            if last_id == stop_at + 1:
                resume_mode = "pending"
                resume_from = last_id
            elif last_id == stop_at:
                resume_mode = "retry"
                resume_from = stop_at
            else:
                return {
                    "status": "unexpected_checkpoint",
                    "evolution_dir": str(evolution_dir),
                    "reason": reason,
                    "stop_at_iteration_id": stop_at,
                    "last_iteration_id": last_id,
                }
        else:
            max_traj = cls._max_traj_iteration_id(evolution_dir)
            if max_traj is None:
                return {
                    "status": "missing_traj_for_crash_resume",
                    "evolution_dir": str(evolution_dir),
                    "reason": reason or "missing_exploration_stop",
                    "last_iteration_id": last_id,
                }
            if last_id == max_traj + 1:
                resume_mode = "pending"
                resume_from = last_id
                stop_at = last_id - 1
            elif last_id == max_traj:
                resume_mode = "retry"
                resume_from = last_id
                stop_at = last_id
            else:
                return {
                    "status": "ambiguous_crash_state",
                    "evolution_dir": str(evolution_dir),
                    "reason": reason or "missing_exploration_stop",
                    "last_iteration_id": last_id,
                    "max_traj_iteration_id": max_traj,
                }

        streak = cls._compute_no_structure_change_streak(summaries)
        return {
            "status": "resumable",
            "evolution_dir": str(evolution_dir),
            "reason": reason or "missing_exploration_stop",
            "resume_mode": resume_mode,
            "stop_at_iteration_id": stop_at,
            "resume_from_iteration_id": resume_from,
            "no_structure_change_streak": streak,
            "pipeline_iterations_completed": stop.get("pipeline_iterations_completed"),
        }

    @classmethod
    def inspect_resume(cls, evolution_dir) -> dict:
        """
        检查目录是否可断点续跑。返回 status/resume_from_iteration_id 等字段。
        """
        evolution_dir = Path(evolution_dir)
        iterations_path = evolution_dir / "iterations.json"
        if not iterations_path.is_file():
            return {"status": "missing_iterations", "evolution_dir": str(evolution_dir)}

        payload = json.loads(iterations_path.read_text(encoding="utf-8"))
        summaries = payload.get("iterations") or []
        stop = payload.get("exploration_stop") or {}
        return cls._resolve_resume_position(summaries, stop, evolution_dir)

    @classmethod
    def _load_resume_checkpoint(cls, evolution_dir, seed_query: QueryData) -> dict:
        info = cls.inspect_resume(evolution_dir)
        if info.get("status") != "resumable":
            raise ValueError(
                f"无法从 {evolution_dir} 续跑: {info.get('status')} ({info.get('reason', '')})"
            )

        iterations_path = Path(evolution_dir) / "iterations.json"
        payload = json.loads(iterations_path.read_text(encoding="utf-8"))
        summaries = list(payload.get("iterations") or [])
        stop = payload.get("exploration_stop") or {}
        resume_mode = str(info.get("resume_mode") or "pending")
        checkpoint = summaries.pop()
        start_iteration_id = int(checkpoint["iteration_id"])

        reviser_path = Path(evolution_dir) / "reviser.json"
        reviser_summaries: List[dict] = []
        if reviser_path.is_file():
            raw = json.loads(reviser_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                reviser_summaries = raw

        return {
            "evolution_id": str(stop.get("evolution_id") or uuid.uuid4()),
            "resume_mode": resume_mode,
            "start_iteration_id": start_iteration_id,
            "current_query": cls._query_from_summary(checkpoint, seed_query),
            "mermaid": cls._mermaid_from_summary(checkpoint),
            "rubric": checkpoint.get("rubric"),
            "title_query": checkpoint.get("title_query"),
            "iteration_summaries": summaries,
            "reviser_summaries": reviser_summaries,
            "no_structure_change_streak": int(info.get("no_structure_change_streak") or 0),
            "stop_reason": str(info.get("reason") or stop.get("reason") or ""),
        }
    
    def _review_expand_task_output(
        self,
        *,
        stage1_out: Dict[str, str],
        iteration_id: int,
        traj_k: TrajectoryData,
    ) -> Dict[str, str]:
        """
        扩展任务阶段后的可插拔钩子（默认直通）。
        可在子类重写：对 stage1 的 mermaid 节点做人工/规则判断，再返回修订后的 mermaid/query。
        """
        return stage1_out
    
    async def explore(
        self,
        seed_query: QueryData,
        best_of_n: Optional[int] = None,
        *,
        resume_dir: Optional[Path] = None,
    ) -> Optional[QueryEvolution]:
        """
        探索单个query的复杂度演化
        
        Args:
            seed_query: 种子query
            best_of_n: 覆盖 best-of-n 采样数（每轮 iteration 内部使用；<=1 等价不启用）
            resume_dir: 若指定，从该目录的 iterations.json 断点续跑（除正常结构收敛外，
                含 max_iterations_reached、error、timeout、进程中断等异常停止）
            
        Returns:
            演化过程（包含所有迭代）
        """
        resume_checkpoint = None
        if resume_dir is not None:
            resume_checkpoint = self._load_resume_checkpoint(Path(resume_dir), seed_query)
            logger.info(
                "断点续跑 %s: 从 iteration_id=%s 继续（此前因 %s 停止）",
                seed_query.query_id,
                resume_checkpoint["start_iteration_id"],
                resume_checkpoint["stop_reason"],
            )
        else:
            logger.info(f"开始复杂度探索: {seed_query.query_id}")

        effective_best_of_n = (
            self.default_best_of_n if best_of_n is None else int(best_of_n)
        )
        if effective_best_of_n < 1:
            effective_best_of_n = 1

        batch_processor: Optional[BatchProcessor] = None
        if effective_best_of_n > 1:
            # 先不注入 writer：等 evolution_dir 创建后再注入，确保写入目录正确
            batch_processor = BatchProcessor(
                pipeline=self.pipeline,
                batch_size=1,
                best_of_n=effective_best_of_n,
                writer=None,
            )
        
        if resume_checkpoint is not None:
            evolution = QueryEvolution(
                evolution_id=resume_checkpoint["evolution_id"],
                original_query=seed_query,
                iterations=[],
                metadata={
                    "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "resumed_from": resume_checkpoint["stop_reason"],
                    "resume_from_iteration_id": resume_checkpoint["start_iteration_id"],
                },
            )
            current_query = resume_checkpoint["current_query"]
            mermaid = resume_checkpoint["mermaid"]
            rubric = resume_checkpoint["rubric"]
            title_query = resume_checkpoint.get("title_query")
            evolution_dir = Path(resume_dir).resolve()
        else:
            evolution = QueryEvolution(
                evolution_id=str(uuid.uuid4()),
                original_query=seed_query,
                iterations=[],
                metadata={"start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            )
            current_query = seed_query
            mermaid = None
            rubric = None
            title_query = None

            output_root = resolve_under_repo(
                self.exploration_config.get("output_dir", "outputs/complexity_exploration"),
                anchor_file=__file__,
            )
            # 目录用 query_id 命名，便于断点续跑/人工检索；若冲突则自动加后缀避免覆盖
            evolution_dir = _allocate_unique_dir(output_root / _safe_dirname(seed_query.query_id))
            evolution_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ComplexityExplorationV2 落盘目录: {evolution_dir.resolve()}")

        # 如果启用 best-of-n，让 BatchProcessor 也把 selected/rejected 落盘到 pipeline_results.jsonl 系列文件
        if batch_processor:
            batch_processor.writer = StreamingWriter(
                str(evolution_dir / "best_of_n_pipeline_results")
            )

        # 汇总文件 iterations.json：每轮记录 pipeline 实际使用的 query / mermaid / rubric
        iterations_path = evolution_dir / "iterations.json"
        iteration_summaries: List[dict] = (
            list(resume_checkpoint["iteration_summaries"]) if resume_checkpoint else []
        )
        reviser_path = evolution_dir / "reviser.json"
        reviser_summaries: List[dict] = (
            list(resume_checkpoint["reviser_summaries"]) if resume_checkpoint else []
        )
        no_structure_change_streak = (
            int(resume_checkpoint["no_structure_change_streak"]) if resume_checkpoint else 0
        )
        final_evolved_query: Optional[QueryData] = None
        stop_at_iteration_id: Optional[int] = None
        start_iteration_id = (
            int(resume_checkpoint["start_iteration_id"]) if resume_checkpoint else 0
        )

        if start_iteration_id >= self.max_iterations:
            logger.warning(
                "Query %s 续跑起点 iteration_id=%s 已 >= max_iterations=%s，跳过",
                seed_query.query_id,
                start_iteration_id,
                self.max_iterations,
            )
            evolution.converged = True
            evolution.convergence_reason = "resume_start_beyond_max_iterations"
            evolution.total_iterations = len(iteration_summaries)
            return evolution

        def _finalize_iterations_file() -> None:
            self._write_iterations_file(
                iterations_path,
                iteration_summaries,
                self._build_exploration_stop_record(
                    evolution,
                    stop_at_iteration_id=stop_at_iteration_id,
                ),
            )

        for iteration_id in range(start_iteration_id, self.max_iterations):
            logger.info(
                f"Query {seed_query.query_id} - 迭代 {iteration_id + 1}/{self.max_iterations} "
                f"(evolution_id={evolution.evolution_id})"
            )
            
            try:
                # 1. 构建workflow input
                workflow_input = WorkflowInput(
                    messages=[{"role": "user", "content": current_query.content}],
                    metadata={
                        **current_query.metadata,
                        'query_id': current_query.query_id,
                        'query': current_query.content,
                        'iteration_id': iteration_id,
                        'mermaid': mermaid.to_str() if mermaid else None,
                        'mermaid_structure': mermaid.to_structure_str() if mermaid else None,
                        'mermaid_nodes': mermaid.node_dicts() if mermaid else [],
                    }
                )
                
                # 如果有上一轮的rubric，加入metadata（reviser 已过滤空 checkpoint 并归一化 weight）
                if rubric:
                    if isinstance(rubric, dict) and "criteria" in rubric:
                        workflow_input.metadata["rubrics"] = rubric["criteria"]
                    else:
                        # 兜底：保持兼容旧格式
                        workflow_input.metadata["rubrics"] = rubric
                
                # 2. 调用pipeline处理（每轮内部 Best-of-N + 拒绝采样）
                if batch_processor:
                    selected_results, rejected_results = await batch_processor.process_batch_with_rejection([workflow_input])
                    if not selected_results:
                        logger.warning(f"迭代 {iteration_id + 1} best-of-n 选优为空")
                        evolution.converged = True
                        evolution.convergence_reason = "best_of_n_empty"
                        stop_at_iteration_id = iteration_id
                        _finalize_iterations_file()
                        break
                    # selected_results 理论上只包含 1 条（因为只有 1 个原始 input）
                    result = selected_results[0]
                else:
                    result = await self.pipeline.process(workflow_input)
                    rejected_results = []
                
                if not result.success:
                    logger.warning(f"迭代 {iteration_id + 1} pipeline失败")
                    evolution.converged = True
                    evolution.convergence_reason = "pipeline_failed"
                    stop_at_iteration_id = iteration_id
                    _finalize_iterations_file()
                    break
                
                # 3. 计算质量分数 & 准备 evaluation_output
                quality_score = result.evaluation.score if result.evaluation else 0.0
                evaluation_output = (
                    {"scores": result.evaluation.scores, "details": result.evaluation.details}
                    if result.evaluation
                    else {"scores": {}, "details": {}}
                )
                logger.info(f"迭代 {iteration_id + 1} 质量分数: {quality_score:.4f}")

                # 4. 构造 traj_k（TrajectoryData 视图），供 reviser 使用
                # 用“被选中的那条结果”的 metadata 作为 context，保证每轮 best-of-n 的选择信息能跟着传下去
                traj_k = self._convert_to_trajectory(result.output, result.input.metadata)


                # 5. 落盘 traj_k.jsonl（每轮单独一个文件，便于复盘）
                traj_path = evolution_dir / f"traj_{iteration_id}.jsonl"
                with traj_path.open("w", encoding="utf-8") as f:
                    f.write(json.dumps(traj_k.to_dict(), ensure_ascii=False) + "\n")
                logger.info(f"已写入 traj: {traj_path.resolve()}")

                # 5.5 额外落盘：本轮 best-of-n 未被选中的候选（rejected）
                if rejected_results:
                    rejected_traj_path = evolution_dir / f"rejected_traj_{iteration_id}.jsonl"
                    with rejected_traj_path.open("w", encoding="utf-8") as f:
                        for rejected in rejected_results:
                            traj_rejected = self._convert_to_trajectory(
                                rejected.output,
                                rejected.input.metadata,
                            )
                            f.write(json.dumps(traj_rejected.to_dict(), ensure_ascii=False) + "\n")
                    logger.info(f"已写入 rejected traj: {rejected_traj_path.resolve()}")
                
                # 6. 保存迭代数据（进 evolution）
                iteration_data = IterationData(
                    iteration_id=iteration_id,
                    query=current_query,
                    trajectory=traj_k,
                    quality_score=quality_score,
                    metadata={
                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "evaluation_scores": result.evaluation.scores if result.evaluation else {},
                        "evaluation_details": result.evaluation.details if result.evaluation else {},
                        "mermaid": mermaid.to_str() if mermaid else None,
                        "mermaid_structure": mermaid.to_structure_str() if mermaid else None,
                        "mermaid_nodes": mermaid.node_dicts() if mermaid else [],
                        "rubric": rubric,
                        "traj_k_path": str(traj_path),
                        "query_fuzzified": (current_query.metadata or {}).get("query_fuzzified"),
                    }
                )
                evolution.iterations.append(iteration_data)

                # 7. 写入 iterations.json：记录本轮 pipeline 实际使用的 query / mermaid / rubric
                iteration_summaries.append(
                    self._build_iteration_summary(
                        iteration_id,
                        current_query.content,
                        (current_query.metadata or {}).get("query_fuzzified"),
                        mermaid,
                        rubric,
                        title_query=title_query,
                    )
                )
                self._write_iterations_file(iterations_path, iteration_summaries)

                # 8. 调用 reviser：stage1 接收 query+mermaid+trajectory 输出 mermaid+rubric；
                #    stage2 接收新 mermaid+trajectory 输出 query+query_fuzzified
                reviser_metadata = {
                    "iteration_id": iteration_id,
                    "evolution_id": evolution.evolution_id,
                    "evaluation_output": evaluation_output,
                }
                try:
                    revise_stage1_out = await self.reviser.expand_task(
                        query=current_query.content,
                        mermaid=mermaid.to_str() if mermaid else None,
                        traj_k=traj_k,
                        metadata=reviser_metadata,
                    )
                except Exception as e:
                    partial_output = getattr(e, "partial_output", None)
                    reviser_summaries.append(
                        self._build_reviser_stage1_record(
                            iteration_id=iteration_id,
                            query=current_query.content,
                            mermaid=mermaid,
                            stage1_output=partial_output,
                            metadata=reviser_metadata,
                            error=str(e),
                        )
                    )
                    self._write_reviser_summaries(reviser_path, reviser_summaries)
                    raise
                revise_stage1_out = self._review_expand_task_output(
                    stage1_out=revise_stage1_out,
                    iteration_id=iteration_id,
                    traj_k=traj_k,
                )
                reviser_summaries.append(
                    self._build_reviser_stage1_record(
                        iteration_id=iteration_id,
                        query=current_query.content,
                        mermaid=mermaid,
                        stage1_output=revise_stage1_out,
                        metadata=reviser_metadata,
                    )
                )
                self._write_reviser_summaries(reviser_path, reviser_summaries)
                try:
                    revise_stage2_out = await self.reviser.generate_query_and_rubric(
                        stage1_out=revise_stage1_out,
                        traj_k=traj_k,
                        metadata=reviser_metadata,
                        include_title_query=True,
                    )
                except Exception as e:
                    logger.warning(
                        "迭代 %s query 生成失败: %s",
                        iteration_id + 1,
                        e,
                    )
                    raise
                stage1_mermaid_nodes = revise_stage1_out.get("mermaid_nodes", [])
                mermaid_next = Mermaid.from_str(
                    revise_stage1_out["mermaid"] or "",
                    nodes=stage1_mermaid_nodes,
                )
                # stage1+stage2 都完成后赋值：下一轮 mermaid/rubric 来自 stage1，query/query_fuzzified 来自 stage2
                query_next = revise_stage2_out.get("query", "")
                rubric_next = revise_stage1_out["rubric"]
                query_fuzzified_next = revise_stage2_out["query_fuzzified"]
                title_query_next = revise_stage2_out.get("title_query")
                leaf_queries_next: List[dict] = []
                final_evolved_query = QueryData(
                    query_id=str(uuid.uuid4()),
                    content=query_next,
                    metadata={
                        **(current_query.metadata or {}),
                        "from_evolution_id": evolution.evolution_id,
                        "from_iteration_id": iteration_id,
                        "query_fuzzified": query_fuzzified_next,
                        "leaf_queries": leaf_queries_next,
                    },
                )

                current_mermaid_depth = mermaid.depth() if mermaid else None
                next_mermaid_depth = mermaid_next.depth() if mermaid_next else None
                current_mermaid_node_count = (
                    len(mermaid.node_dicts()) if mermaid else None
                )
                next_mermaid_node_count = (
                    len(mermaid_next.node_dicts()) if mermaid_next else None
                )
                if (
                    next_mermaid_depth is not None
                    and current_mermaid_depth is not None
                    and next_mermaid_node_count is not None
                    and current_mermaid_node_count is not None
                    and next_mermaid_depth == current_mermaid_depth
                    and next_mermaid_node_count == current_mermaid_node_count
                ):
                    no_structure_change_streak += 1
                else:
                    no_structure_change_streak = 0
                
                # 9. 判断是否继续
                should_continue, reason = self._should_continue_exploration(
                    iteration_id=iteration_id,
                    current_mermaid=mermaid,
                    next_mermaid=mermaid_next,
                    no_structure_change_streak=no_structure_change_streak,
                )
                
                if not should_continue:
                    logger.info(f"Query {seed_query.query_id} 收敛，原因: {reason}")
                    evolution.converged = True
                    evolution.convergence_reason = reason
                    # 最后一轮 reviser 产出未再跑 pipeline，单独落盘供复盘
                    iteration_summaries.append(
                        self._build_iteration_summary(
                            iteration_id + 1,
                            query_next,
                            query_fuzzified_next,
                            mermaid_next,
                            rubric_next,
                            title_query=title_query_next,
                        )
                    )
                    stop_at_iteration_id = iteration_id
                    _finalize_iterations_file()
                    break
                
                # 10. 准备下一轮：更新 (mermaid, query, rubric, title_query)
                mermaid = mermaid_next
                rubric = rubric_next
                title_query = title_query_next
                current_query = final_evolved_query
                
            except Exception as e:
                logger.error(f"迭代 {iteration_id + 1} 失败: {e}", exc_info=True)
                evolution.converged = True
                evolution.convergence_reason = f"error: {str(e)}"
                stop_at_iteration_id = iteration_id
                _finalize_iterations_file()
                break
        
        # 设置最终结果
        evolution.total_iterations = len(evolution.iterations)
        if stop_at_iteration_id is None:
            _finalize_iterations_file()
        if final_evolved_query is not None:
            evolution.final_query = final_evolved_query
        elif evolution.iterations:
            evolution.final_query = evolution.iterations[-1].query
        
        logger.info(f"Query {seed_query.query_id} 探索完成，共 {evolution.total_iterations} 次迭代")
        
        return evolution

    def _convert_to_trajectory(self, workflow_output, context) -> TrajectoryData:
        """
        将 WorkflowOutput 转成 TrajectoryData（用于 reviser/落盘复盘）。
        逻辑与 `OpenJudgeEvaluator._convert_to_trajectory` 对齐，避免口径漂移。
        """
        merged_context = {**(context or {}), **(getattr(workflow_output, "metadata", {}) or {})}
        trajectory_id = str(uuid.uuid4())

        tool_calls = []
        for tc in getattr(workflow_output, "tool_calls", []) or []:
            tool_calls.append(
                ToolCall(
                    name=tc.get("name", ""),
                    parameters=tc.get("parameters", {}),
                    tool_call_id=tc.get("tool_call_id"),
                )
            )

        tool_responses = []
        for tr in getattr(workflow_output, "tool_responses", []) or []:
            tool_responses.append(
                ToolResponse(
                    tool_call_id=tr.get("tool_call_id", ""),
                    content=tr.get("content", ""),
                    success=tr.get("success", True),
                )
            )

        return TrajectoryData(
            trajectory_id=trajectory_id,
            skeleton_id=merged_context.get("skeleton_id"),
            query_id=merged_context.get("query_id", str(uuid.uuid4())),
            query=merged_context.get("query"),
            messages=getattr(workflow_output, "messages", []) or [],
            tool_calls=tool_calls,
            tool_responses=tool_responses,
            metadata={
                **(context or {}),
                **(getattr(workflow_output, "metadata", {}) or {}),
            },
        )
    
    def _should_continue_exploration(
        self,
        iteration_id: int,
        current_mermaid: Optional[Mermaid],
        next_mermaid: Optional[Mermaid],
        no_structure_change_streak: int,
    ) -> tuple[bool, str]:
        """
        判断是否应该继续探索
        
        Returns:
            (should_continue, reason)
        """
        # 按“深度与节点数是否都不变”判断收敛
        current_mermaid_depth = current_mermaid.depth() if current_mermaid else None
        next_mermaid_depth = next_mermaid.depth() if next_mermaid else None
        current_mermaid_node_count = (
            len(current_mermaid.node_dicts()) if current_mermaid else None
        )
        next_mermaid_node_count = (
            len(next_mermaid.node_dicts()) if next_mermaid else None
        )
        # - next 深度或节点数缺失：视为无法继续有效增长
        # - 深度与节点数均不变：计入 no_structure_change_streak

        logger.info(
            "迭代 %s mermaid: depth current=%s next=%s, node_count current=%s next=%s",
            iteration_id + 1,
            current_mermaid_depth,
            next_mermaid_depth,
            current_mermaid_node_count,
            next_mermaid_node_count,
        )
        if next_mermaid_depth is None:
            return False, "mermaid_depth_unavailable"
        if next_mermaid_node_count is None:
            return False, "mermaid_node_count_unavailable"

        # 连续推理节点链过长时停止（防止“单条推理节点无限延伸”）。
        if self.max_consecutive_reasoning_nodes > 0 and next_mermaid:
            max_reasoning_streak = self._max_consecutive_reasoning_chain(next_mermaid)
            logger.info(
                "迭代 %s 最长连续推理链长度: %s (阈值=%s)",
                iteration_id + 1,
                max_reasoning_streak,
                self.max_consecutive_reasoning_nodes,
            )
            if max_reasoning_streak > self.max_consecutive_reasoning_nodes:
                return (
                    False,
                    "reasoning_chain_too_long"
                    f"_{max_reasoning_streak}_gt_{self.max_consecutive_reasoning_nodes}",
                )

        # 结构收敛优先于 max_iterations，避免最后一轮同时满足两条件时误报轮次上限。
        if no_structure_change_streak >= 2:
            return (
                False,
                "mermaid_depth_and_node_count_unchanged_for_2_rounds"
                f"_depth_{current_mermaid_depth}_nodes_{current_mermaid_node_count}",
            )

        if iteration_id >= self.max_iterations - 1:
            return False, "max_iterations_reached"
        
        # # 检查质量分数阈值
        # if quality_score < self.quality_threshold:
        #     return False, f"quality_below_threshold_{quality_score:.2f}"
        
        # # 检查最小复杂度增量
        # if iteration_id > 0 and len(evolution.iterations) > 1:
        #     prev_score = evolution.iterations[-2].quality_score
        #     score_change = abs(quality_score - prev_score)
        #     min_change = self.exploration_config.get("min_complexity_increase", 0.1)
        #     if score_change < min_change:
        #         return False, f"insufficient_complexity_increase_{score_change:.2f}"
        
        return True, "continue"

    def _max_consecutive_reasoning_chain(self, mermaid: Mermaid) -> int:
        """
        计算 DAG 中“连续推理性节点”最长链长度。
        推理节点判定：优先使用 mermaid.nodes 中的 node_type；兼容旧图时回退到 label 含“推理性”。
        """
        flow = mermaid.to_flowchart()
        if not flow.edges:
            return 0

        nodes: set[str] = set()
        indeg: Dict[str, int] = {}
        adj: Dict[str, List[str]] = {}
        for edge in flow.edges:
            nodes.add(edge.src)
            nodes.add(edge.dst)
        for node_id in nodes:
            indeg[node_id] = 0
            adj[node_id] = []
        for edge in flow.edges:
            adj[edge.src].append(edge.dst)
            indeg[edge.dst] += 1

        node_types = {node.node_id: node.node_type for node in mermaid.nodes}
        labels = mermaid.node_labels()

        def is_reasoning(node_id: str) -> bool:
            node_type = node_types.get(node_id)
            if node_type:
                return is_reasoning_node_type(node_type)
            label = labels.get(node_id, "")
            return "推理性" in label or "reasoning" in label.lower()

        # 拓扑 DP：dp[n] 表示“以 n 结尾”的最长连续推理链长度。
        dp: Dict[str, int] = {node_id: (1 if is_reasoning(node_id) else 0) for node_id in nodes}
        q: List[str] = [node_id for node_id, d in indeg.items() if d == 0]
        visited = 0

        while q:
            node_id = q.pop(0)
            visited += 1
            for nxt in adj[node_id]:
                if is_reasoning(nxt):
                    candidate = dp[node_id] + 1 if dp[node_id] > 0 else 1
                    if candidate > dp[nxt]:
                        dp[nxt] = candidate
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    q.append(nxt)

        if visited != len(nodes):
            return 0
        return max(dp.values()) if dp else 0
    
    async def explore_batch(
        self, 
        seed_queries: List[QueryData]
    ) -> List[QueryEvolution]:
        """
        批量探索多个seed queries
        
        Args:
            seed_queries: 种子query列表
            
        Returns:
            演化过程列表
        """
        import asyncio
        
        logger.info(f"开始批量复杂度探索: {len(seed_queries)} 个queries")
        
        # 并发探索
        results = await asyncio.gather(
            *[self.explore(query) for query in seed_queries],
            return_exceptions=True
        )
        
        # 过滤异常
        evolutions = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"探索失败: {result}")
            elif result:
                evolutions.append(result)
        
        logger.info(f"批量探索完成: {len(evolutions)}/{len(seed_queries)} 成功")
        
        return evolutions
