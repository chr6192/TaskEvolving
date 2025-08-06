"""
PRM选择器 - 基于过程奖励模型进行拒绝采样
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
import heapq

from omegaconf import DictConfig

from .data_structures import TrajectoryData

logger = logging.getLogger(__name__)


class PRMSelector:
    """
    基于PRM进行拒绝采样
    
    支持多种选择策略：
    1. best_of_n: 选择分数最高的N个
    2. threshold: 选择分数超过阈值的所有样本
    3. top_k: 选择前K个
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.prm_config = config.prm
        
        # PRM模型配置
        self.model_path = self.prm_config.model_path
        self.selection_strategy = self.prm_config.selection_strategy
        self.threshold = self.prm_config.threshold
        self.top_k = self.prm_config.top_k
        self.scoring_method = self.prm_config.scoring_method
        
        # PRM模型实例（延迟初始化）
        self.prm_model = None
        
        # 状态标记
        self._initialized = False
        
    async def initialize(self) -> None:
        """初始化PRM选择器"""
        if self._initialized:
            return
            
        logger.info("初始化PRM选择器...")
        
        # 初始化PRM模型
        if self.model_path:
            await self._initialize_prm_model()
        else:
            logger.info("未指定PRM模型路径，将使用基于规则的评分")
            
        self._initialized = True
        logger.info("PRM选择器初始化完成")
        
    async def cleanup(self) -> None:
        """清理资源"""
        logger.info("清理PRM选择器资源...")
        
        if self.prm_model:
            # 清理PRM模型资源
            try:
                if hasattr(self.prm_model, 'cleanup'):
                    await self.prm_model.cleanup()
            except Exception as e:
                logger.error(f"PRM模型资源清理失败: {e}")
                
        logger.info("PRM选择器资源清理完成")
        
    async def select_trajectories(self, candidates: List[TrajectoryData]) -> List[TrajectoryData]:
        """
        从候选trajectories中选择高质量的
        
        Args:
            candidates: 候选trajectory列表
            
        Returns:
            选择后的trajectory列表
        """
        if not self._initialized:
            raise RuntimeError("PRM选择器未初始化")
            
        if not candidates:
            logger.warning("候选trajectories为空")
            return []
            
        logger.info(f"使用{self.selection_strategy}策略从 {len(candidates)} 个候选中选择")
        
        try:
            # 为所有候选计算PRM分数
            scored_trajectories = []
            for trajectory in candidates:
                prm_score = await self.score_trajectory(trajectory)
                trajectory.prm_score = prm_score
                scored_trajectories.append((prm_score, trajectory))
                
            # 根据策略选择trajectories
            selected = await self._apply_selection_strategy(scored_trajectories)
            
            logger.info(f"PRM选择完成：从 {len(candidates)} 个候选中选择了 {len(selected)} 个")
            return selected
            
        except Exception as e:
            logger.error(f"PRM选择失败: {e}")
            # 返回原始候选列表
            return candidates
            
    async def score_trajectory(self, trajectory: TrajectoryData) -> float:
        """
        为单个trajectory评分
        
        Args:
            trajectory: 要评分的trajectory
            
        Returns:
            PRM分数
        """
        try:
            if self.prm_model:
                # 使用训练好的PRM模型
                score = await self._score_with_model(trajectory)
            else:
                # 使用基于规则的评分
                score = await self._score_with_rules(trajectory)
                
            logger.debug(f"Trajectory {trajectory.trajectory_id} PRM分数: {score:.4f}")
            return score
            
        except Exception as e:
            logger.error(f"计算PRM分数失败: {e}")
            return 0.0
            
    async def _initialize_prm_model(self) -> None:
        """初始化PRM模型"""
        try:
            # TODO: 根据实际PRM模型接口实现
            logger.info(f"加载PRM模型: {self.model_path} (待实现)")
            # self.prm_model = load_prm_model(self.model_path)
            
        except Exception as e:
            logger.error(f"PRM模型初始化失败: {e}")
            self.prm_model = None
            
    async def _score_with_model(self, trajectory: TrajectoryData) -> float:
        """
        使用训练好的PRM模型评分
        
        Args:
            trajectory: 要评分的trajectory
            
        Returns:
            PRM分数
        """
        try:
            # TODO: 实现实际的PRM模型调用
            logger.debug("使用PRM模型评分 (待实现)")
            
            if self.scoring_method == "step_wise":
                # 逐步评分然后聚合
                step_scores = []
                for i, message in enumerate(trajectory.messages):
                    if message.get("role") == "assistant":
                        # 对每个assistant回复评分
                        step_score = await self._score_single_step(message, trajectory.messages[:i+1])
                        step_scores.append(step_score)
                
                # 聚合分数（例如取平均）
                if step_scores:
                    return sum(step_scores) / len(step_scores)
                else:
                    return 0.5
                    
            else:  # trajectory_wise
                # 对整个trajectory评分
                score = await self._score_full_trajectory(trajectory)
                return score
                
        except Exception as e:
            logger.error(f"PRM模型评分失败: {e}")
            return 0.0
            
    async def _score_with_rules(self, trajectory: TrajectoryData) -> float:
        """
        使用基于规则的评分
        
        Args:
            trajectory: 要评分的trajectory
            
        Returns:
            基于规则的分数
        """
        try:
            score_components = []
            
            # 1. 对话长度评分（适中的长度更好）
            turn_count = trajectory.get_turn_count()
            length_score = self._score_conversation_length(turn_count)
            score_components.append(("length", length_score, 0.2))
            
            # 2. 工具使用评分
            tool_score = self._score_tool_usage(trajectory)
            score_components.append(("tool_usage", tool_score, 0.3))
            
            # 3. 响应质量评分
            response_score = self._score_response_quality(trajectory)
            score_components.append(("response_quality", response_score, 0.3))
            
            # 4. 结构化程度评分
            structure_score = self._score_structure(trajectory)
            score_components.append(("structure", structure_score, 0.2))
            
            # 加权平均
            total_score = sum(score * weight for _, score, weight in score_components)
            
            logger.debug(f"规则评分详情: {score_components}, 总分: {total_score:.4f}")
            return max(0.0, min(1.0, total_score))  # 限制在[0,1]范围内
            
        except Exception as e:
            logger.error(f"基于规则的评分失败: {e}")
            return 0.5
            
    def _score_conversation_length(self, turn_count: int) -> float:
        """
        对话长度评分
        
        Args:
            turn_count: 对话轮数
            
        Returns:
            长度分数
        """
        if 4 <= turn_count <= 12:
            return 1.0
        elif turn_count < 4:
            return turn_count / 4.0
        else:
            return max(0.2, 12.0 / turn_count)
            
    def _score_tool_usage(self, trajectory: TrajectoryData) -> float:
        """
        工具使用评分
        
        Args:
            trajectory: trajectory数据
            
        Returns:
            工具使用分数
        """
        tool_calls = trajectory.tool_calls
        tool_responses = trajectory.tool_responses
        
        if not tool_calls:
            return 0.3  # 没有使用工具，给较低分数
            
        # 工具调用成功率
        success_rate = sum(1 for resp in tool_responses if resp.success) / len(tool_responses)
        
        # 工具多样性
        unique_tools = len(set(tc.name for tc in tool_calls))
        diversity_score = min(1.0, unique_tools / 3.0)  # 假设使用3种不同工具是理想的
        
        return 0.6 * success_rate + 0.4 * diversity_score
        
    def _score_response_quality(self, trajectory: TrajectoryData) -> float:
        """
        响应质量评分
        
        Args:
            trajectory: trajectory数据
            
        Returns:
            响应质量分数
        """
        final_response = trajectory.get_final_response()
        
        if not final_response:
            return 0.0
            
        # 基于响应长度和内容的简单评分
        length_score = min(1.0, len(final_response) / 500.0)  # 500字符为满分
        
        # 检查是否包含关键要素
        quality_indicators = [
            "分析" in final_response,
            "建议" in final_response or "推荐" in final_response,
            len(final_response.split('。')) >= 3,  # 至少3个句子
            any(keyword in final_response for keyword in ["因为", "由于", "根据"]),  # 包含解释性词汇
        ]
        
        content_score = sum(quality_indicators) / len(quality_indicators)
        
        return 0.4 * length_score + 0.6 * content_score
        
    def _score_structure(self, trajectory: TrajectoryData) -> float:
        """
        结构化程度评分
        
        Args:
            trajectory: trajectory数据
            
        Returns:
            结构化分数
        """
        messages = trajectory.messages
        
        # 检查对话结构
        has_proper_start = len(messages) > 0 and messages[0].get("role") == "user"
        has_proper_end = len(messages) > 1 and messages[-1].get("role") == "assistant"
        has_tool_interaction = any(msg.get("role") == "tool" for msg in messages)
        
        structure_score = sum([has_proper_start, has_proper_end, has_tool_interaction]) / 3.0
        
        return structure_score
        
    async def _apply_selection_strategy(
        self, 
        scored_trajectories: List[tuple[float, TrajectoryData]]
    ) -> List[TrajectoryData]:
        """
        应用选择策略
        
        Args:
            scored_trajectories: (score, trajectory)列表
            
        Returns:
            选择后的trajectory列表
        """
        # 按分数降序排序
        scored_trajectories.sort(key=lambda x: x[0], reverse=True)
        
        if self.selection_strategy == "best_of_n":
            # 选择分数最高的N个
            n = min(self.top_k, len(scored_trajectories))
            selected = [traj for _, traj in scored_trajectories[:n]]
            
        elif self.selection_strategy == "threshold":
            # 选择分数超过阈值的所有样本
            selected = [traj for score, traj in scored_trajectories if score >= self.threshold]
            
        elif self.selection_strategy == "top_k":
            # 选择前K个
            k = min(self.top_k, len(scored_trajectories))
            selected = [traj for _, traj in scored_trajectories[:k]]
            
        else:
            logger.warning(f"未知的选择策略: {self.selection_strategy}，使用top_k")
            k = min(self.top_k, len(scored_trajectories))
            selected = [traj for _, traj in scored_trajectories[:k]]
            
        logger.info(f"策略 {self.selection_strategy} 选择了 {len(selected)} 个trajectory")
        return selected
        
    async def _score_single_step(self, message: Dict[str, str], context: List[Dict[str, str]]) -> float:
        """
        对单个步骤评分
        
        Args:
            message: 当前消息
            context: 上下文消息
            
        Returns:
            步骤分数
        """
        # TODO: 实现实际的单步PRM评分
        return 0.5
        
    async def _score_full_trajectory(self, trajectory: TrajectoryData) -> float:
        """
        对完整trajectory评分
        
        Args:
            trajectory: trajectory数据
            
        Returns:
            trajectory分数
        """
        # TODO: 实现实际的完整trajectory PRM评分
        return 0.5 