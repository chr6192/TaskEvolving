"""
复杂度探索编排器
调用新的pipeline系统实现复杂度探索
"""
import logging
import uuid
import json
import re
from typing import Dict, List, Optional
from datetime import datetime
from omegaconf import DictConfig

from ..core.data_structures import WorkflowInput
from ..core.batch_processor import BatchProcessor
from ..core.pipeline import SamplePipeline
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

        # 用于复杂度探索中的 Best-of-N（每轮 iteration 内部采样 N 次再选择 1 条）
        try:
            self.default_best_of_n = int(self.config.get("batch_processor", {}).get("best_of_n", 1))
        except Exception:
            self.default_best_of_n = 1
        
        logger.info(
            f"ComplexityExplorationV2初始化: "
            f"max_iterations={self.max_iterations}"
        )

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
    ) -> Optional[QueryEvolution]:
        """
        探索单个query的复杂度演化
        
        Args:
            seed_query: 种子query
            best_of_n: 覆盖 best-of-n 采样数（每轮 iteration 内部使用；<=1 等价不启用）
            
        Returns:
            演化过程（包含所有迭代）
        """
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
        
        evolution = QueryEvolution(
            evolution_id=str(uuid.uuid4()),
            original_query=seed_query,
            iterations=[],
            metadata={"start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        )
        
        current_query = seed_query
        mermaid: Optional[Mermaid] = None
        rubric: Optional[dict] = None

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

        # 汇总文件 iterations.json：每轮增量写入 query / mermaid / rubric 摘要，便于一次性复盘
        iterations_path = evolution_dir / "iterations.json"
        iteration_summaries: List[dict] = []
        no_depth_growth_streak = 0
        
        for iteration_id in range(self.max_iterations):
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
                    }
                )
                
                # 如果有上一轮的rubric，加入metadata（传给 OpenJudge 的是 criteria 列表）
                if rubric:
                    if isinstance(rubric, dict) and "criteria" in rubric:
                        # OpenJudge 只关心 rubrics checklist，本质是一个列表
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
                        "rubric": rubric,
                        "traj_k_path": str(traj_path),
                        "query_fuzzified": (current_query.metadata or {}).get("query_fuzzified"),
                    }
                )
                evolution.iterations.append(iteration_data)

                # 7. 先写入 iterations.json（每轮增量更新）：
                # 在 reviser 之前落盘，确保即便 reviser 失败/中断，也能保留已完成轮次。
                iteration_summaries.append(
                    {
                        "iteration_id": iteration_id,
                        "query": current_query.content,
                        "query_fuzzified": (current_query.metadata or {}).get("query_fuzzified"),
                        "mermaid": mermaid.to_str() if mermaid else None,
                        "depth": mermaid.depth() if mermaid else None,
                        "rubric": rubric,
                    }
                )
                with iterations_path.open("w", encoding="utf-8") as f:
                    json.dump(iteration_summaries, f, ensure_ascii=False, indent=2)

                # 8. 调用 reviser（两阶段）：
                # 扩展任务: mermaid + query
                # rubric生成: rubric + query_fuzzified
                reviser_metadata = {
                    "iteration_id": iteration_id,
                    "evolution_id": evolution.evolution_id,
                    "evaluation_output": evaluation_output,
                }
                revise_stage1_out = await self.reviser.expand_task(
                    mermaid=mermaid.to_str() if mermaid else None,
                    traj_k=traj_k,
                    metadata=reviser_metadata,
                )
                revise_stage1_out = self._review_expand_task_output(
                    stage1_out=revise_stage1_out,
                    iteration_id=iteration_id,
                    traj_k=traj_k,
                )
                revise_stage2_out = await self.reviser.generate_rubric(
                    stage1_mermaid=revise_stage1_out["mermaid"],
                    stage1_query=revise_stage1_out["query"],
                    metadata=reviser_metadata,
                )
                stage1_node_sources = revise_stage1_out.get("node_sources", [])
                mermaid_next = Mermaid.from_str(revise_stage1_out["mermaid"] or "")
                rubric_next = revise_stage2_out["rubric"]
                query_next = revise_stage1_out["query"]
                query_fuzzified_next = revise_stage2_out["query_fuzzified"]

                # 将 stage1 的节点来源信息挂到本轮 metadata，并刷新 iterations.json 便于复盘。
                iteration_data.metadata["node_sources"] = stage1_node_sources
                iteration_summaries[-1]["node_sources"] = stage1_node_sources
                with iterations_path.open("w", encoding="utf-8") as f:
                    json.dump(iteration_summaries, f, ensure_ascii=False, indent=2)

                current_mermaid_depth = mermaid.depth() if mermaid else None
                next_mermaid_depth = mermaid_next.depth() if mermaid_next else None
                if (
                    next_mermaid_depth is not None
                    and current_mermaid_depth is not None
                    and next_mermaid_depth <= current_mermaid_depth
                ):
                    no_depth_growth_streak += 1
                else:
                    no_depth_growth_streak = 0
                
                # 9. 判断是否继续
                should_continue, reason = self._should_continue_exploration(
                    iteration_id=iteration_id,
                    current_mermaid=mermaid,
                    next_mermaid=mermaid_next,
                    no_depth_growth_streak=no_depth_growth_streak,
                )
                
                if not should_continue:
                    logger.info(f"Query {seed_query.query_id} 收敛，原因: {reason}")
                    evolution.converged = True
                    evolution.convergence_reason = reason
                    break
                
                # 10. 准备下一轮：更新 (mermaid, query, rubric)
                mermaid = mermaid_next
                rubric = rubric_next
                current_query = QueryData(
                    query_id=str(uuid.uuid4()),
                    content=query_next,
                    metadata={
                        **(current_query.metadata or {}),
                        "from_evolution_id": evolution.evolution_id,
                        "from_iteration_id": iteration_id,
                        "query_fuzzified": query_fuzzified_next,
                    },
                )
                
            except Exception as e:
                logger.error(f"迭代 {iteration_id + 1} 失败: {e}", exc_info=True)
                evolution.converged = True
                evolution.convergence_reason = f"error: {str(e)}"
                break
        
        # 设置最终结果
        evolution.total_iterations = len(evolution.iterations)
        if evolution.iterations:
            last_iteration = evolution.iterations[-1]
            evolution.final_query = last_iteration.query
        
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
        no_depth_growth_streak: int,
    ) -> tuple[bool, str]:
        """
        判断是否应该继续探索
        
        Returns:
            (should_continue, reason)
        """
        # 检查迭代次数
        if iteration_id >= self.max_iterations - 1:
            return False, "max_iterations_reached"


        # 按“深度是否增长”判断收敛
        current_mermaid_depth = current_mermaid.depth() if current_mermaid else None
        next_mermaid_depth = next_mermaid.depth() if next_mermaid else None
        # - next 深度缺失：视为无法继续有效增长
        # - current 有值且 next <= current：深度不再增长（或退化）

        logger.info(
            "迭代 %s mermaid深度: current=%s, next=%s",
            iteration_id + 1,
            current_mermaid_depth,
            next_mermaid_depth,
        )
        if next_mermaid_depth is None:
            return False, "mermaid_depth_unavailable"
        if no_depth_growth_streak >= 2:
            return (
                False,
                "mermaid_depth_not_increasing_for_2_rounds"
                f"_{current_mermaid_depth}_to_{next_mermaid_depth}",
            )
        
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
