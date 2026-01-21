"""
复杂度探索编排器
调用新的pipeline系统实现复杂度探索
"""
import logging
import uuid
from typing import List, Optional
from datetime import datetime
from omegaconf import DictConfig

from ..core.data_structures import WorkflowInput
from ..core.pipeline import SamplePipeline
from ..workflow.base import WorkflowBase
from deep_research.synthesis.core.data_structures import QueryData, QueryEvolution, IterationData
from deep_research.synthesis.core.query_reviser import QueryReviserAgent

logger = logging.getLogger(__name__)


class ComplexityExplorationV2:
    """使用新pipeline的复杂度探索编排器"""
    
    def __init__(
        self,
        pipeline: SamplePipeline,
        query_reviser: QueryReviserAgent,
        config: DictConfig
    ):
        """
        Args:
            pipeline: 单条样本pipeline
            query_reviser: query修订agent
            config: 配置
        """
        self.pipeline = pipeline
        self.query_reviser = query_reviser
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
        previous_rubrics = None
        
        for iteration_id in range(self.max_iterations):
            logger.info(f"Query {seed_query.query_id} - 迭代 {iteration_id + 1}")
            
            try:
                # 1. 构建workflow input
                workflow_input = WorkflowInput(
                    messages=[{"role": "user", "content": current_query.content}],
                    metadata={
                        **current_query.metadata,
                        'query_id': current_query.query_id,
                        'query': current_query.content,
                        'iteration_id': iteration_id
                    }
                )
                
                # 如果有上一轮的rubrics，加入metadata
                if previous_rubrics:
                    workflow_input.metadata['rubrics'] = previous_rubrics.to_dict()
                
                # 2. 调用pipeline处理
                result = await self.pipeline.process(workflow_input)
                
                if not result.success:
                    logger.warning(f"迭代 {iteration_id + 1} pipeline失败")
                    evolution.converged = True
                    evolution.convergence_reason = "pipeline_failed"
                    break
                
                # 3. 计算质量分数
                quality_score = result.evaluation.score if result.evaluation else 0.0
                logger.info(f"迭代 {iteration_id + 1} 质量分数: {quality_score:.4f}")
                
                # 4. 保存迭代数据
                iteration_data = IterationData(
                    iteration_id=iteration_id,
                    query=current_query,
                    quality_score=quality_score,
                    metadata={
                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "evaluation_scores": result.evaluation.scores if result.evaluation else {}
                    }
                )
                evolution.iterations.append(iteration_data)
                
                # 5. 判断是否继续
                should_continue, reason = self._should_continue_exploration(
                    quality_score, iteration_id, evolution
                )
                
                if not should_continue:
                    logger.info(f"Query {seed_query.query_id} 收敛，原因: {reason}")
                    evolution.converged = True
                    evolution.convergence_reason = reason
                    break
                
                # 6. 生成更复杂的query（使用现有的query_reviser）
                # 注意：这里需要适配现有的query_reviser接口
                # 暂时简化处理，实际可能需要转换数据格式
                logger.info("生成更复杂的query...")
                # TODO: 调用query_reviser生成下一个query和rubrics
                
                # 占位：直接break，实际需要实现完整的revise逻辑
                evolution.converged = True
                evolution.convergence_reason = "query_reviser_not_implemented"
                break
                
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
        
        # 检查质量分数阈值
        if quality_score < self.quality_threshold:
            return False, f"quality_below_threshold_{quality_score:.2f}"
        
        # 检查最小复杂度增量
        if iteration_id > 0 and len(evolution.iterations) > 1:
            prev_score = evolution.iterations[-2].quality_score
            score_change = abs(quality_score - prev_score)
            min_change = self.exploration_config.get("min_complexity_increase", 0.1)
            if score_change < min_change:
                return False, f"insufficient_complexity_increase_{score_change:.2f}"
        
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
