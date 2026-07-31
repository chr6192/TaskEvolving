"""
ORM评估器 - 使用OpenJudge奖励模型评估trajectory质量
"""
import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict
from dataclasses import dataclass

from omegaconf import DictConfig

from .data_structures import ORMEvaluationResult, TrajectoryData

logger = logging.getLogger(__name__)


@dataclass
class RewardSample:
    """Reward评估样本，与OpenJudge RewardSample保持一致"""
    original_index: int  # 全局索引
    group_index: int     # 组内索引
    prompt: str
    response: str
    messages: List[Dict]
    extra: Any


@dataclass
class RewardResult:
    """Reward评估结果，与OpenJudge RewardResult保持一致"""
    original_index: int
    group_index: int
    score: float
    reward_info: Dict[str, float]
    details: Optional[Dict] = None


class ORMEvaluator:
    """
    使用OpenJudge奖励模型评估最终trajectory质量
    
    支持动态加载任意BaseOpenJudgeRewardFunction子类：
    - OpenJudgeRewardFunctionV0: 默认V0版本
    - 自定义RewardFunction: 用户自定义的评估函数
    
    支持的控制参数：
    - orm.enabled: 控制是否启用ORM评估功能
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.orm_config = config.orm
        
        # ORM评估启用控制
        self.orm_enabled = self.orm_config.get('enabled', True)
        
        # 质量控制配置
        self.quality_threshold = config.quality_control.quality_threshold
        
        # Reward Function实例（延迟初始化）
        self.reward_function: Optional[Any] = None
        
        # 状态标记
        self._initialized = False
        
    def _load_reward_function_from_config(self):
        """
        从配置中动态加载reward function类实例
        
        配置格式：
        reward_function:
          class_path: "deep_research.reward.openjudge_reward_function_v0.OpenJudgeRewardFunctionV0"
          init_params:
            model_name: "qwen3.6-plus"
            temperature: 0.0
            language: "zh"
            max_concurrency: 32
            grader_weights:
              financial_report_resolution: 0.3
              trajectory_comprehensive: 0.7
        
        Returns:
            BaseOpenJudgeRewardFunction实例
        """
        try:
            # 获取reward function配置
            reward_config = self.config.get('reward_function', {})
            
            if not reward_config:
                raise ValueError(
                    "reward_function配置不能为空！\n"
                    "请在配置文件中指定reward_function.class_path和init_params"
                )
            
            class_path = reward_config.get('class_path')
            if not class_path:
                raise ValueError(
                    "reward_function.class_path不能为空！\n"
                    "示例：class_path: 'deep_research.reward.openjudge_reward_function_v0.OpenJudgeRewardFunctionV0'"
                )
            
            # 解析模块路径和类名
            if '.' not in class_path:
                raise ValueError(f"class_path格式错误，应为 'module.path.ClassName'，当前为: {class_path}")
            
            module_path, class_name = class_path.rsplit('.', 1)
            
            # 动态导入模块
            import importlib
            module = importlib.import_module(module_path)
            
            # 获取类
            if not hasattr(module, class_name):
                raise AttributeError(f"模块 {module_path} 中没有类 {class_name}")
            
            reward_class = getattr(module, class_name)
            
            # 获取初始化参数
            init_params = dict(reward_config.get('init_params', {}))
            openai_api = self.config.get("openai_api", {})
            if init_params.get("api_timeout") is None:
                timeout = openai_api.get("timeout")
                if timeout is not None:
                    init_params["api_timeout"] = float(timeout)
            if init_params.get("max_retries") is None:
                max_retries = openai_api.get("max_retries")
                if max_retries is not None:
                    init_params["max_retries"] = int(max_retries)
            # 处理language参数（字符串转枚举）
            if 'language' in init_params and isinstance(init_params['language'], str):
                try:
                    from openjudge.models.schema.prompt_template import LanguageEnum
                    language_str = init_params['language'].upper()
                    init_params['language'] = LanguageEnum[language_str]
                except Exception as e:
                    logger.warning(f"语言参数转换失败，使用默认值: {e}")
                    init_params.pop('language', None)
            
            # 创建实例
            reward_function = reward_class(**init_params)
            
            logger.info(f"成功加载reward function: {class_path}")
            logger.info(f"初始化参数: {init_params}")
            
            return reward_function
            
        except Exception as e:
            logger.error(f"从配置加载reward function失败: {e}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
            raise RuntimeError(
                f"Reward function加载失败！\n"
                f"错误信息: {str(e)}\n"
                f"请检查配置文件中的reward_function配置是否正确"
            ) from e
        
    async def initialize(self) -> None:
        """初始化ORM评估器"""
        if self._initialized:
            return
            
        logger.info("初始化ORM评估器...")
        
        # 初始化reward function
        await self._initialize_reward_function()
        
        self._initialized = True
        logger.info("ORM评估器初始化完成")
        
    async def cleanup(self) -> None:
        """清理资源"""
        logger.info("清理ORM评估器资源...")
        
        if self.reward_function:
            # 清理reward function资源
            try:
                if hasattr(self.reward_function, 'cleanup'):
                    await self.reward_function.cleanup()
            except Exception as e:
                logger.error(f"Reward function资源清理失败: {e}")
                
        logger.info("ORM评估器资源清理完成")
          
    async def _initialize_reward_function(self) -> None:
        """初始化reward function实例"""
        try:
            if not self.orm_enabled:
                logger.info("ORM评估已禁用，跳过reward function初始化")
                self.reward_function = None
                return
            
            # 从配置中动态加载reward function类并创建实例
            self.reward_function = self._load_reward_function_from_config()
            
            logger.info(f"Reward function初始化成功: {type(self.reward_function).__name__}")
            
        except Exception as e:
            logger.error(f"Reward function初始化失败: {e}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
            self.reward_function = None
            raise

    async def evaluate_trajectory(self, trajectory: TrajectoryData) -> ORMEvaluationResult:
        """
        评估单个trajectory（使用批处理逻辑实现）
        
        Args:
            trajectory: 要评估的trajectory
            
        Returns:
            评估结果
        """
        # 使用批处理逻辑处理单个trajectory
        results = await self.batch_evaluate([trajectory])
        return results[0] if results else ORMEvaluationResult(
            trajectory_id=trajectory.trajectory_id,
            orm_scores={},
            passed_quality_check=False
        )
            
    async def batch_evaluate(self, trajectories: List[TrajectoryData]) -> List[ORMEvaluationResult]:
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
            
        if not self._initialized:
            raise RuntimeError("ORM评估器未初始化")
            
        logger.info(f"批量评估 {len(trajectories)} 个trajectory")
        
        try:
            # 1. 准备数据项（转换为RewardSample格式）
            reward_samples = self._prepare_trajectories_as_reward_samples(trajectories)
            
            # 2. 调用reward function进行评估
            orm_results = None
            if self.orm_enabled and self.reward_function:
                orm_results = await self._evaluate_with_reward_function(reward_samples)
            
            # 3. 合并结果
            evaluation_results = self._merge_evaluation_results(
                trajectories, orm_results
            )
            
            success_count = sum(1 for result in evaluation_results if result.passed_quality_check)
            logger.info(f"批量评估完成：{success_count}/{len(trajectories)} 个通过质量检查")
            
            return evaluation_results
            
        except Exception as e:
            import traceback
            from datetime import datetime
            
            logger.error(f"批量评估失败: {e}")
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            
            # 构建详细的错误信息
            error_details = {
                "error": str(e) if e else "批量评估过程中发生未知错误",
                "error_type": type(e).__name__ if e else "UnknownError",
                "error_timestamp": datetime.now().isoformat(),
                "batch_size": len(trajectories),
                "orm_enabled": self.orm_enabled,
                "traceback": traceback.format_exc(),
                "config_info": {
                    "quality_threshold": self.quality_threshold
                }
            }
            
            # 尝试获取更多上下文信息
            if hasattr(self, 'reward_function') and self.reward_function:
                error_details["reward_function_status"] = "initialized"
                error_details["reward_function_type"] = type(self.reward_function).__name__
            else:
                error_details["reward_function_status"] = "not_initialized"
            
            # 返回默认评估结果列表，确保trajectory_id正确对应
            logger.warning("批量评估失败，返回包含错误详情的默认评估结果")
            default_results = []
            for traj in trajectories:
                default_result = ORMEvaluationResult(
                    trajectory_id=traj.trajectory_id,
                    orm_scores={},
                    orm_details=error_details,  # 包含详细错误信息
                    quality_metrics={"error_occurred": True},
                    reward_info={"batch_evaluation_failed": True},
                    passed_quality_check=False
                )
                default_results.append(default_result)
            
            logger.info(f"为 {len(default_results)} 个trajectory创建了包含错误详情的默认评估结果")
            return default_results
      
    def _prepare_trajectories_as_reward_samples(self, trajectories: List[TrajectoryData]) -> List[RewardSample]:
        """
        将trajectories转换为RewardSample格式
        
        Args:
            trajectories: trajectory列表
            
        Returns:
            RewardSample列表
        """
        reward_samples = []
        for i, trajectory in enumerate(trajectories):
            # 提取用户查询
            user_messages = [msg for msg in trajectory.messages if msg.get("role") == "user"]
            prompt = user_messages[0]["content"] if user_messages else ""
            if not isinstance(prompt, str) or not prompt.strip():
                prompt = (trajectory.metadata or {}).get("query") or ""
            
            # 提取助手回复
            assistant_messages = [msg for msg in trajectory.messages if msg.get("role") == "assistant"]
            responses = [msg["content"] for msg in assistant_messages if msg.get("content")]
            response = responses[-1] if responses else ""  # 取最后一个回复
            
            grader_messages = [
                msg for msg in trajectory.messages if msg.get("role") != "system"
            ]
            if not any(msg.get("role") == "user" for msg in grader_messages):
                grader_messages.insert(0, {"role": "user", "content": prompt})
            
            # 构建额外信息
            extra = {
                "trajectory_id": trajectory.trajectory_id,
                                "tool_calls": [tc.to_dict() for tc in trajectory.tool_calls],
                "tool_responses": [tr.to_dict() for tr in trajectory.tool_responses],
                "metadata": trajectory.metadata,
                "rubrics": trajectory.metadata.get("rubrics", None),  # rubrics字段（用于reward function）
                "chat_date": trajectory.metadata.get("chat_date", None)  # 时间字段（用于timeliness评估）
            }
            
            # 创建RewardSample（与OpenJudge兼容）
            sample = RewardSample(
                original_index=i,
                group_index=-1,  # 稍后设置
                prompt=prompt,
                response=response,
                messages=grader_messages,
                extra=extra
            )
            reward_samples.append(sample)
            
        return reward_samples
    
    async def _evaluate_with_reward_function(
        self, 
        reward_samples: List[RewardSample]
    ) -> Dict[str, Any]:
        """
        使用reward function进行评估
        
        Args:
            reward_samples: RewardSample列表（rubrics已在extra中）
            
        Returns:
            ORM评估结果
        """
        try:
            # 1. 按prompt分组，设置group_index
            prompt_to_samples = defaultdict(list)
            for sample in reward_samples:
                group_samples = prompt_to_samples[sample.prompt]
                sample.group_index = len(group_samples)
                group_samples.append(sample)
            
            # 2. 检查每组样本数量
            group_sizes = [len(samples) for samples in prompt_to_samples.values()]
            if len(set(group_sizes)) > 1:
                logger.warning(f"每组的样本数量不相同，当前group_sizes: {group_sizes}")
            
            logger.info(f"共有{len(prompt_to_samples)}组，平均每组{sum(group_sizes)/len(group_sizes):.1f}个样本")
            
            # 3. 调用reward function（直接使用callable接口）
            result_items = await self.reward_function(prompt_to_samples)
            
            # 4. 验证结果完整性
            if len(result_items) != len(reward_samples):
                raise ValueError(
                    f"Reward function返回结果数量({len(result_items)})与输入样本数量({len(reward_samples)})不一致"
                )
            
            # 5. 转换为legacy格式
            rewards, reward_infos, details = self._convert_reward_results_to_legacy_format(result_items)
            
            return {
                "rewards": rewards,
                "reward_infos": reward_infos,
                "details": details
            }
            
        except Exception as e:
            import traceback
            from datetime import datetime
            
            logger.error(f"Reward function评估失败: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            
            # 构建错误详情
            error_details = {
                "reward_function_error": str(e),
                "error_type": type(e).__name__,
                "error_timestamp": datetime.now().isoformat(),
                "samples_count": len(reward_samples),
                "traceback": traceback.format_exc()
            }
            
            # 返回包含错误信息的默认结果
            return {
                "rewards": [0.0] * len(reward_samples),
                "reward_infos": {},
                "details": [error_details] * len(reward_samples)
            }
    
    def _convert_reward_results_to_legacy_format(
        self,
        result_items: List[RewardResult]
    ) -> tuple[List[float], Dict[str, List[float]], List[Dict]]:
        """
        将RewardResult列表转换为legacy格式
        
        Args:
            result_items: RewardResult列表（已按original_index排序）
            
        Returns:
            (rewards, reward_infos, details)
        """
        # 提取所有reward_info的key
        all_keys = set()
        for item in result_items:
            all_keys.update(item.reward_info.keys())
        
        # 提取数据
        rewards = [item.score for item in result_items]
        
        reward_infos = defaultdict(list)
        for key in all_keys:
            reward_infos[key] = [item.reward_info.get(key, 0.0) for item in result_items]
        
        details = [item.details if item.details else {} for item in result_items]
        
        return rewards, dict(reward_infos), details
    
    def _merge_evaluation_results(
        self, 
        trajectories: List[TrajectoryData], 
        orm_results: Optional[Dict[str, Any]]
    ) -> List[ORMEvaluationResult]:
        """
        合并ORM评估结果
        
        Args:
            trajectories: 原始trajectory列表
            orm_results: ORM评估结果
            
        Returns:
            最终评估结果列表
        """
        evaluation_results = []
        
        for i, trajectory in enumerate(trajectories):
            # 提取ORM结果
            orm_scores = {}
            reward_info = {}
            details = {}
            
            if orm_results:
                rewards = orm_results.get("rewards", [])
                reward_infos = orm_results.get("reward_infos", {})
                all_details = orm_results.get("details", [])
                
                if i < len(rewards):
                    # 添加各个维度的分数
                    for metric in reward_infos.keys():
                        if metric in reward_infos and i < len(reward_infos[metric]):
                            orm_scores[metric] = reward_infos[metric][i]
                    
                    reward_info = {k: [v[i]] if isinstance(v, list) and i < len(v) else v 
                                 for k, v in reward_infos.items()}
                    
                    if all_details and i < len(all_details):
                        details = all_details[i] if all_details[i] else {}
            
            # 将details信息保存到trajectory中
            if details:
                trajectory.orm_details = details
            
            # 构建最终评估结果
            evaluation_result = ORMEvaluationResult(
                trajectory_id=trajectory.trajectory_id,
                orm_scores=orm_scores,
                orm_details=trajectory.orm_details,
                quality_metrics=self._calculate_quality_metrics(orm_scores, reward_info),
                reward_info=reward_info,
                passed_quality_check=self._check_quality_threshold(orm_scores)
            )
            
            evaluation_results.append(evaluation_result)
        
        return evaluation_results
   
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
    
    
 