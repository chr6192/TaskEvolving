"""
复杂度探索编排器
调用新的pipeline系统实现复杂度探索
"""
import logging
import uuid
import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from omegaconf import DictConfig

from ..core.data_structures import WorkflowInput
from ..core.pipeline import SamplePipeline
from ..workflow.base import WorkflowBase
from ..io.path_utils import resolve_under_repo
from deep_research.synthesis.core.data_structures import (
    QueryData,
    QueryEvolution,
    IterationData,
    TrajectoryData,
    ToolCall,
    ToolResponse,
)
from .mermaid_reviser import MermaidReviserAgent

logger = logging.getLogger(__name__)


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
        self.quality_threshold = self.exploration_config.get('quality_threshold', 0.7)
        
        logger.info(
            f"ComplexityExplorationV2初始化: "
            f"max_iterations={self.max_iterations}, "
            f"quality_threshold={self.quality_threshold}"
        )
    
    async def explore(self, seed_query: QueryData) -> Optional[QueryEvolution]:
        """
        探索单个query的复杂度演化
        
        Args:
            seed_query: 种子query
            
        Returns:
            演化过程（包含所有迭代）
        """
        logger.info(f"开始复杂度探索: {seed_query.query_id}")
        
        evolution = QueryEvolution(
            evolution_id=str(uuid.uuid4()),
            original_query=seed_query,
            iterations=[],
            metadata={"start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        )
        
        current_query = seed_query
        mermaid: Optional[str] = None
        rubric: Optional[dict] = None

        output_root = resolve_under_repo(
            self.exploration_config.get("output_dir", "outputs/complexity_exploration"),
            anchor_file=__file__,
        )
        evolution_dir = output_root / evolution.evolution_id
        evolution_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ComplexityExplorationV2 落盘目录: {evolution_dir.resolve()}")
        
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
                        'mermaid': mermaid,
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
                
                # 2. 调用pipeline处理
                result = await self.pipeline.process(workflow_input)
                
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
                traj_k = self._convert_to_trajectory(result.output, workflow_input.metadata)

                # 5. 落盘 traj_k.jsonl（每轮单独一个文件，便于复盘）
                traj_path = evolution_dir / f"traj_{iteration_id}.jsonl"
                with traj_path.open("w", encoding="utf-8") as f:
                    f.write(json.dumps(traj_k.to_dict(), ensure_ascii=False) + "\n")
                logger.info(f"已写入 traj: {traj_path.resolve()}")
                
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
                        "mermaid": mermaid,
                        "rubric": rubric,
                        "traj_k_path": str(traj_path),
                    }
                )
                evolution.iterations.append(iteration_data)

                # 7. 调用 reviser：mermaid, traj_k -> mermaid_next, query_next, rubric_next
                revise_out = await self.reviser.revise(
                    mermaid_k=mermaid,
                    traj_k=traj_k,
                    metadata={
                        "iteration_id": iteration_id,
                        "evolution_id": evolution.evolution_id,
                        "evaluation_output": evaluation_output,
                    },
                )
                mermaid_next = revise_out["mermaid"]
                query_next = revise_out["query"]
                rubric_next = revise_out["rubric"]

                # 8. 落盘 iteration_k.json（按规划字段）
                iteration_path = evolution_dir / f"iteration_{iteration_id}.json"
                with iteration_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "query_k": current_query.content,
                            "mermaid_k": mermaid,
                            "rubric_k": rubric,
                            "score_k": quality_score,
                            "details_k": evaluation_output.get("details", {}),
                            "traj_k_path": str(traj_path),
                            "next": {
                                "mermaid_{k+1}": mermaid_next,
                                "query_{k+1}": query_next,
                                "rubric_{k+1}": rubric_next,
                            },
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                logger.info(f"已写入 iteration: {iteration_path.resolve()}")
                
                # 9. 判断是否继续
                should_continue, reason = self._should_continue_exploration(
                    quality_score, iteration_id, evolution
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
        quality_score: float,
        iteration_id: int,
        evolution: QueryEvolution
    ) -> tuple[bool, str]:
        """
        判断是否应该继续探索
        
        Returns:
            (should_continue, reason)
        """
        # 检查迭代次数
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
