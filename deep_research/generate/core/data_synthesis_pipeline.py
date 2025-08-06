"""
数据合成Pipeline主协调器
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

from omegaconf import DictConfig

from .data_structures import (
    QueryData, SkeletonData, TrajectoryData, EvaluationResult, PipelineStats
)
from .skeleton_generator import SkeletonGenerator  
from .trajectory_generator import TrajectoryGenerator
from .prm_selector import PRMSelector
from .orm_evaluator import ORMEvaluator

logger = logging.getLogger(__name__)


class DataSynthesisPipeline:
    """
    数据合成主管道，协调阶段一和阶段二的执行
    
    - 阶段一：生成带有递进逻辑的agent tool calls skeleton
    - 阶段二：基于skeleton通过PRM拒绝采样构造完整trajectory
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.stats = PipelineStats()
        
        # 核心组件（延迟初始化）
        self.skeleton_generator: Optional[SkeletonGenerator] = None
        self.trajectory_generator: Optional[TrajectoryGenerator] = None
        self.prm_selector: Optional[PRMSelector] = None
        self.orm_evaluator: Optional[ORMEvaluator] = None
        
        # 状态标记
        self._initialized = False
        
    async def initialize(self) -> None:
        """初始化pipeline所有组件"""
        if self._initialized:
            return
            
        logger.info("初始化数据合成Pipeline...")
        
        # 初始化各个组件
        if self.config.stage in ["skeleton", "both"]:
            self.skeleton_generator = SkeletonGenerator(self.config)
            await self.skeleton_generator.initialize()
            
        if self.config.stage in ["trajectory", "both"]:
            self.trajectory_generator = TrajectoryGenerator(self.config)
            await self.trajectory_generator.initialize()
            
            if self.config.prm.enabled:
                self.prm_selector = PRMSelector(self.config)
                await self.prm_selector.initialize()
                
            if self.config.orm.enabled:
                self.orm_evaluator = ORMEvaluator(self.config)
                await self.orm_evaluator.initialize()
        
        self._initialized = True
        logger.info("Pipeline初始化完成")
        
    async def cleanup(self) -> None:
        """清理资源"""
        logger.info("清理Pipeline资源...")
        
        cleanup_tasks = []
        if self.skeleton_generator:
            cleanup_tasks.append(self.skeleton_generator.cleanup())
        if self.trajectory_generator:
            cleanup_tasks.append(self.trajectory_generator.cleanup())
        if self.prm_selector:
            cleanup_tasks.append(self.prm_selector.cleanup())
        if self.orm_evaluator:
            cleanup_tasks.append(self.orm_evaluator.cleanup())
            
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
        logger.info("Pipeline资源清理完成")
        
    async def run_stage1(self, queries: List[QueryData]) -> List[SkeletonData]:
        """
        阶段一：生成skeleton
        
        Args:
            queries: 输入查询列表
            
        Returns:
            生成的skeleton列表
        """
        if not self.skeleton_generator:
            raise RuntimeError("Skeleton生成器未初始化")
            
        logger.info(f"开始阶段一：为 {len(queries)} 个查询生成skeleton")
        start_time = time.time()
        
        self.stats.total_queries = len(queries)
        
        try:
            # 批量生成skeleton
            skeletons = await self.skeleton_generator.batch_generate(queries)
            
            # 验证和过滤
            valid_skeletons = []
            for skeleton in skeletons:
                if skeleton and skeleton.validate():
                    valid_skeletons.append(skeleton)
                else:
                    logger.warning(f"Invalid skeleton for query {skeleton.query_id if skeleton else 'unknown'}")
            
            self.stats.generated_skeletons = len(skeletons) 
            self.stats.valid_skeletons = len(valid_skeletons)
            self.stats.stage1_time = time.time() - start_time
            
            # 保存skeleton数据
            if self.config.output.save_intermediate:
                await self._save_skeletons(valid_skeletons)
                
            logger.info(f"阶段一完成：生成 {len(valid_skeletons)}/{len(skeletons)} 个有效skeleton，耗时 {self.stats.stage1_time:.2f}s")
            return valid_skeletons
            
        except Exception as e:
            logger.error(f"阶段一执行失败: {e}", exc_info=True)
            raise
            
    async def run_stage2(self, skeletons: List[SkeletonData]) -> List[TrajectoryData]:
        """
        阶段二：基于skeleton生成完整trajectory
        
        Args:
            skeletons: 输入skeleton列表
            
        Returns:
            高质量的trajectory列表
        """
        if not self.trajectory_generator:
            raise RuntimeError("Trajectory生成器未初始化")
            
        logger.info(f"开始阶段二：为 {len(skeletons)} 个skeleton生成trajectory")
        start_time = time.time()
        
        try:
            # 生成候选trajectories
            all_trajectories = []
            for skeleton in skeletons[:1]:
                logger.info(f"处理skeleton {skeleton.skeleton_id}")
                candidates = await self.trajectory_generator.generate_from_skeleton(
                    skeleton, self.config.generation.n_samples_per_skeleton
                )
                all_trajectories.extend(candidates)
                
            self.stats.generated_trajectories = len(all_trajectories)
            logger.info(f"生成了 {len(all_trajectories)} 个候选trajectory")
            
            # PRM筛选
            selected_trajectories = all_trajectories
            if self.prm_selector and self.config.prm.enabled:
                logger.info("使用PRM进行拒绝采样")
                selected_trajectories = await self.prm_selector.select_trajectories(all_trajectories)
                logger.info(f"PRM筛选后剩余 {len(selected_trajectories)} 个trajectory")
            
            # ORM评估
            high_quality_trajectories = []
            if self.orm_evaluator and self.config.orm.enabled:
                logger.info("使用ORM进行最终质量评估")
                evaluation_results = await self.orm_evaluator.batch_evaluate(selected_trajectories)
                
                # 应用质量控制
                for trajectory, eval_result in zip(selected_trajectories, evaluation_results):
                    trajectory.orm_scores = eval_result.orm_scores
                    if eval_result.passed_quality_check:
                        high_quality_trajectories.append(trajectory)
                        
                logger.info(f"ORM评估后剩余 {len(high_quality_trajectories)} 个高质量trajectory")
            else:
                high_quality_trajectories = selected_trajectories
                
            self.stats.high_quality_trajectories = len(high_quality_trajectories)
            self.stats.stage2_time = time.time() - start_time
            
            # 保存trajectory数据
            if self.config.output.save_intermediate:
                await self._save_trajectories(high_quality_trajectories)
                
            logger.info(f"阶段二完成：生成 {len(high_quality_trajectories)} 个高质量trajectory，耗时 {self.stats.stage2_time:.2f}s")
            return high_quality_trajectories
            
        except Exception as e:
            logger.error(f"阶段二执行失败: {e}", exc_info=True)
            raise
            
    async def run_full_pipeline(self, queries: List[QueryData]) -> List[TrajectoryData]:
        """端到端执行完整pipeline"""
        logger.info("开始执行完整数据合成pipeline")
        total_start_time = time.time()
        
        try:
            # 阶段一
            skeletons = await self.run_stage1(queries)
            
            # 阶段二
            trajectories = await self.run_stage2(skeletons)
            
            self.stats.total_time = time.time() - total_start_time
            
            # 保存统计信息
            await self._save_pipeline_stats()
            
            logger.info(f"Pipeline执行完成，总耗时 {self.stats.total_time:.2f}s")
            logger.info(f"最终生成 {len(trajectories)} 个高质量trajectory")
            
            return trajectories
            
        except Exception as e:
            logger.error(f"Pipeline执行失败: {e}", exc_info=True)
            raise
            
    def load_skeletons_from_dir(self, skeleton_dir: str) -> List[SkeletonData]:
        """从目录加载已有的skeleton数据"""
        skeleton_dir = Path(skeleton_dir)
        if not skeleton_dir.exists():
            raise FileNotFoundError(f"Skeleton目录不存在: {skeleton_dir}")
            
        skeletons = []
        for skeleton_file in skeleton_dir.glob("*.jsonl"):
            with open(skeleton_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    skeleton = SkeletonData.from_dict(data)
                    if skeleton.validate():
                        skeletons.append(skeleton)
                    else:
                        logger.warning(f"跳过无效skeleton: {skeleton.skeleton_id}")
                        
        logger.info(f"从 {skeleton_dir} 加载了 {len(skeletons)} 个有效skeleton")
        return skeletons
        
    async def _save_skeletons(self, skeletons: List[SkeletonData]) -> None:
        """保存skeleton数据"""
        output_dir = Path(self.config.output.skeleton_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "skeletons.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for skeleton in skeletons:
                f.write(json.dumps(skeleton.to_dict(), ensure_ascii=False) + '\n')
                
        logger.info(f"保存了 {len(skeletons)} 个skeleton到 {output_file}")
        
    async def _save_trajectories(self, trajectories: List[TrajectoryData]) -> None:
        """保存trajectory数据"""
        output_dir = Path(self.config.output.trajectory_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存完整trajectory数据
        output_file = output_dir / "trajectories.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for trajectory in trajectories:
                f.write(json.dumps(trajectory.to_dict(), ensure_ascii=False) + '\n')
                
        # 保存训练格式数据
        training_file = output_dir / "training_data.jsonl"
        with open(training_file, 'w', encoding='utf-8') as f:
            for trajectory in trajectories:
                f.write(json.dumps(trajectory.export_for_training(), ensure_ascii=False) + '\n')
                
        logger.info(f"保存了 {len(trajectories)} 个trajectory到 {output_dir}")
        
    async def _save_pipeline_stats(self) -> None:
        """保存pipeline统计信息"""
        if not self.config.output.log_dir:
            return
            
        log_dir = Path(self.config.output.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        stats_file = log_dir / "pipeline_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats.to_dict(), f, ensure_ascii=False, indent=2)
            
        logger.info(f"保存pipeline统计信息到 {stats_file}")
        
    def get_stats(self) -> PipelineStats:
        """获取pipeline统计信息"""
        return self.stats 