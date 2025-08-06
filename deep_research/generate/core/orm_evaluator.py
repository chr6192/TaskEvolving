"""
ORM评估器 - 使用业务侧奖励模型评估trajectory质量
"""

import asyncio
import importlib
import logging
from typing import List, Dict, Any, Optional

from omegaconf import DictConfig

from .data_structures import TrajectoryData, EvaluationResult

logger = logging.getLogger(__name__)


class ORMEvaluator:
    """
    使用业务侧ORM评估最终trajectory质量
    
    集成现有的AntfinRewardManager，支持：
    1. faithful_reward - 忠实性奖励
    2. express_reward - 体感奖励
    3. faithful_num_exist_reward - 数字规则奖励
    4. map_format_reward - 格式规则奖励
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.orm_config = config.orm
        
        # 奖励管理器配置
        self.reward_manager_class_path = self.orm_config.reward_manager_class_path
        self.async_mode = self.orm_config.async_mode
        self.timeout = self.orm_config.timeout
        self.max_workers = self.orm_config.max_workers
        self.evaluation_metrics = self.orm_config.evaluation_metrics
        
        # 质量控制配置
        self.quality_threshold = config.quality_control.quality_threshold
        
        # 奖励管理器实例（延迟初始化）
        self.reward_manager = None
        
        # 状态标记
        self._initialized = False
        
    async def initialize(self) -> None:
        """初始化ORM评估器"""
        if self._initialized:
            return
            
        logger.info("初始化ORM评估器...")
        
        # 初始化奖励管理器
        await self._initialize_reward_manager()
        
        self._initialized = True
        logger.info("ORM评估器初始化完成")
        
    async def cleanup(self) -> None:
        """清理资源"""
        logger.info("清理ORM评估器资源...")
        
        if self.reward_manager:
            # 清理奖励管理器资源
            try:
                if hasattr(self.reward_manager, 'cleanup'):
                    await self.reward_manager.cleanup()
            except Exception as e:
                logger.error(f"奖励管理器资源清理失败: {e}")
                
        logger.info("ORM评估器资源清理完成")
        
    async def evaluate_trajectory(self, trajectory: TrajectoryData) -> EvaluationResult:
        """
        评估单个trajectory
        
        Args:
            trajectory: 要评估的trajectory
            
        Returns:
            评估结果
        """
        if not self._initialized:
            raise RuntimeError("ORM评估器未初始化")
            
        logger.debug(f"评估trajectory {trajectory.trajectory_id}")
        
        try:
            # 准备输入数据
            input_data = self._prepare_trajectory_for_evaluation(trajectory)
            
            # 调用奖励管理器计算分数
            if self.async_mode:
                orm_scores, reward_info = await self._evaluate_async(input_data)
            else:
                orm_scores, reward_info = await self._evaluate_sync(input_data)
                
            # 构建评估结果
            evaluation_result = EvaluationResult(
                trajectory_id=trajectory.trajectory_id,
                orm_scores=orm_scores,
                prm_score=trajectory.prm_score,
                quality_metrics=self._calculate_quality_metrics(orm_scores, reward_info),
                detailed_feedback=reward_info,
                passed_quality_check=self._check_quality_threshold(orm_scores)
            )
            
            logger.debug(f"Trajectory {trajectory.trajectory_id} ORM评估完成，总分: {evaluation_result.get_overall_score():.4f}")
            return evaluation_result
            
        except Exception as e:
            logger.error(f"评估trajectory失败: {e}")
            # 返回默认评估结果
            return EvaluationResult(
                trajectory_id=trajectory.trajectory_id,
                orm_scores={},
                prm_score=trajectory.prm_score,
                passed_quality_check=False
            )
            
    async def batch_evaluate(self, trajectories: List[TrajectoryData]) -> List[EvaluationResult]:
        """
        批量评估trajectories
        
        Args:
            trajectories: 要评估的trajectory列表
            
        Returns:
            评估结果列表
        """
        if not trajectories:
            logger.warning("要评估的trajectories为空")
            return []
            
        logger.info(f"批量评估 {len(trajectories)} 个trajectory")
        
        try:
            # 并发执行评估
            tasks = [self.evaluate_trajectory(trajectory) for trajectory in trajectories]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            evaluation_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"评估trajectory {i} 失败: {result}")
                    # 创建默认评估结果
                    default_result = EvaluationResult(
                        trajectory_id=trajectories[i].trajectory_id,
                        orm_scores={},
                        prm_score=trajectories[i].prm_score,
                        passed_quality_check=False
                    )
                    evaluation_results.append(default_result)
                else:
                    evaluation_results.append(result)
                    
            success_count = sum(1 for result in evaluation_results if result.passed_quality_check)
            logger.info(f"批量评估完成：{success_count}/{len(trajectories)} 个通过质量检查")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"批量评估失败: {e}")
            # 返回默认评估结果列表
            return [
                EvaluationResult(
                    trajectory_id=traj.trajectory_id,
                    orm_scores={},
                    prm_score=traj.prm_score,
                    passed_quality_check=False
                ) for traj in trajectories
            ]
            
    async def _initialize_reward_manager(self) -> None:
        """初始化奖励管理器"""
        try:
            # 动态导入奖励管理器类
            module_path, class_name = self.reward_manager_class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            reward_manager_class = getattr(module, class_name)
            
            # 准备初始化参数
            # 这里需要根据AntfinRewardManager的实际接口调整
            tokenizer = None  # 需要实际的tokenizer
            num_examine = 0  # 用于调试的样本数量
            
            # 创建compute_score函数
            if self.async_mode:
                from deep_research.reward.dr_reward_function2 import compute_score_async
                compute_score = compute_score_async
            else:
                from deep_research.reward.dr_reward_function2 import compute_score
            
            # 奖励管理器参数
            reward_kwargs = {
                "async_mode": self.async_mode,
                "timeout": self.timeout,
                "max_workers": self.max_workers,
                "return_details": True
            }
            
            # 创建奖励管理器实例
            self.reward_manager = reward_manager_class(
                tokenizer=tokenizer,
                num_examine=num_examine,
                compute_score=compute_score,
                **reward_kwargs
            )
            
            logger.info(f"奖励管理器初始化成功: {self.reward_manager_class_path}")
            
        except Exception as e:
            logger.error(f"奖励管理器初始化失败: {e}")
            self.reward_manager = None
            
    def _prepare_trajectory_for_evaluation(self, trajectory: TrajectoryData) -> Dict[str, Any]:
        """
        准备trajectory数据用于评估
        
        Args:
            trajectory: trajectory数据
            
        Returns:
            准备好的输入数据
        """
        # 提取用户查询
        user_messages = [msg for msg in trajectory.messages if msg.get("role") == "user"]
        prompt = user_messages[0]["content"] if user_messages else ""
        
        # 提取助手回复
        assistant_messages = [msg for msg in trajectory.messages if msg.get("role") == "assistant"]
        responses = [msg["content"] for msg in assistant_messages if msg.get("content")]
        
        # 构建group_messages格式（AntfinRewardManager需要的格式）
        group_messages = [trajectory.messages]
        
        # 构建extras信息
        group_extras = [{
            "trajectory_id": trajectory.trajectory_id,
            "skeleton_id": trajectory.skeleton_id,
            "tool_calls": [tc.to_dict() for tc in trajectory.tool_calls],
            "tool_responses": [tr.to_dict() for tr in trajectory.tool_responses],
            "metadata": trajectory.metadata
        }]
        
        return {
            "prompt": prompt,
            "group_responses": responses,
            "group_messages": group_messages,
            "group_extras": group_extras
        }
        
    async def _evaluate_async(self, input_data: Dict[str, Any]) -> tuple[Dict[str, float], Dict[str, Any]]:
        """
        异步模式评估
        
        Args:
            input_data: 输入数据
            
        Returns:
            (orm_scores, reward_info) 元组
        """
        try:
            # 调用异步奖励计算
            from deep_research.reward.dr_reward_function2 import compute_score_async
            
            result = await compute_score_async(
                prompt=input_data["prompt"],
                group_responses=input_data["group_responses"],
                group_messages=input_data["group_messages"],
                group_extras=input_data["group_extras"],
                return_details=True
            )
            
            if len(result) == 3:
                scores, reward_info, details = result
            else:
                scores, reward_info = result
                details = []
            
            # 构建ORM分数字典
            orm_scores = {}
            if len(scores) > 0:
                # 假设只有一个response
                score = scores[0]
                orm_scores["overall_score"] = score
                
                # 添加各个维度的分数
                for metric in self.evaluation_metrics:
                    if metric in reward_info and len(reward_info[metric]) > 0:
                        orm_scores[metric] = reward_info[metric][0]
            
            return orm_scores, reward_info
            
        except Exception as e:
            logger.error(f"异步评估失败: {e}")
            return {}, {}
            
    async def _evaluate_sync(self, input_data: Dict[str, Any]) -> tuple[Dict[str, float], Dict[str, Any]]:
        """
        同步模式评估
        
        Args:
            input_data: 输入数据
            
        Returns:
            (orm_scores, reward_info) 元组
        """
        try:
            # 在线程池中执行同步奖励计算
            from deep_research.reward.dr_reward_function2 import compute_score
            
            result = await asyncio.to_thread(
                compute_score,
                prompt=input_data["prompt"],
                group_responses=input_data["group_responses"],
                group_messages=input_data["group_messages"],
                group_extras=input_data["group_extras"],
                return_details=True
            )
            
            if len(result) == 3:
                scores, reward_info, details = result
            else:
                scores, reward_info = result
                details = []
            
            # 构建ORM分数字典
            orm_scores = {}
            if len(scores) > 0:
                # 假设只有一个response
                score = scores[0]
                orm_scores["overall_score"] = score
                
                # 添加各个维度的分数
                for metric in self.evaluation_metrics:
                    if metric in reward_info and len(reward_info[metric]) > 0:
                        orm_scores[metric] = reward_info[metric][0]
            
            return orm_scores, reward_info
            
        except Exception as e:
            logger.error(f"同步评估失败: {e}")
            return {}, {}
            
    def _calculate_quality_metrics(
        self, 
        orm_scores: Dict[str, float], 
        reward_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        计算质量指标
        
        Args:
            orm_scores: ORM分数字典
            reward_info: 奖励详细信息
            
        Returns:
            质量指标字典
        """
        quality_metrics = {}
        
        try:
            # 基础质量指标
            if orm_scores:
                quality_metrics["avg_score"] = sum(orm_scores.values()) / len(orm_scores)
                quality_metrics["min_score"] = min(orm_scores.values())
                quality_metrics["max_score"] = max(orm_scores.values())
                
                # 各维度分数分布
                for metric_name, score in orm_scores.items():
                    quality_metrics[f"{metric_name}_normalized"] = max(0.0, min(1.0, score))
            
            # 从reward_info中提取额外指标
            if reward_info:
                # 添加统计信息
                for key, value in reward_info.items():
                    if isinstance(value, (int, float)):
                        quality_metrics[f"reward_info_{key}"] = value
                    elif isinstance(value, list) and value and isinstance(value[0], (int, float)):
                        quality_metrics[f"reward_info_{key}_avg"] = sum(value) / len(value)
                        
        except Exception as e:
            logger.error(f"计算质量指标失败: {e}")
            
        return quality_metrics
        
    def _check_quality_threshold(self, orm_scores: Dict[str, float]) -> bool:
        """
        检查是否通过质量阈值
        
        Args:
            orm_scores: ORM分数字典
            
        Returns:
            是否通过质量检查
        """
        if not orm_scores:
            return False
            
        try:
            # 计算平均分数
            avg_score = sum(orm_scores.values()) / len(orm_scores)
            
            # 检查是否超过阈值
            passed = avg_score >= self.quality_threshold
            
            logger.debug(f"质量检查: 平均分数 {avg_score:.4f}, 阈值 {self.quality_threshold}, 通过: {passed}")
            return passed
            
        except Exception as e:
            logger.error(f"质量阈值检查失败: {e}")
            return False 