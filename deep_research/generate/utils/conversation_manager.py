"""
对话管理器 - 管理多轮对话的生成过程
"""

import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Optional

from omegaconf import DictConfig
from verl.protocol import DataProto
import ray
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
from verl.workers.rollout.async_server import AsyncLLMServerManager
from deep_research.generate.core.data_structures import SkeletonData

logger = logging.getLogger(__name__)


class RayConversationManager:
    """
    管理多轮对话的生成，集成verl的async_rollout_manager
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.rollout_config = config.rollout
        
        # async_rollout_manager将在初始化时设置
        self.async_rollout_manager = None
        
        # 状态标记
        self._initialized = False
        
    async def initialize(self) -> None:
        """初始化对话管理器"""
        if self._initialized:
            return
            
        logger.info("初始化对话管理器...")
        
        # 这里需要集成verl的async_rollout_manager初始化逻辑
        # 参考 tests/workers/rollout/async_rollout_utils.py
        await self._initialize_async_rollout_manager()
        
        self._initialized = True
        logger.info("对话管理器初始化完成")
        
    async def cleanup(self) -> None:
        """清理资源"""
        logger.info("清理对话管理器资源...")
        
        if self.async_rollout_manager:
            # 清理async_rollout_manager资源
            # 注意：具体的清理方法需要根据verl的API调整
            pass
            
        logger.info("对话管理器资源清理完成")
        
    async def execute_conversation(
        self,
        initial_messages: List[Dict[str, str]],
        skeleton: Optional[SkeletonData] = None,
        generation_params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        执行多轮对话生成
        
        Args:
            initial_messages: 初始对话消息
            skeleton: 可选的skeleton指导
            generation_params: 生成参数
            
        Returns:
            对话执行结果
        """
        if not self._initialized:
            raise RuntimeError("对话管理器未初始化")
            
        try:
            # 准备输入数据格式（DataProto格式）
            batch = DataProto(
                non_tensor_batch={
                    "raw_prompt": np.array([initial_messages], dtype=object),
                },
            )
            
            # 准备采样参数
            sampling_params = {
                "temperature": generation_params.get("temperature", 0.7),
                "top_p": generation_params.get("top_p", 0.9),
                "max_tokens": generation_params.get("max_new_tokens", 2048),
                "n": generation_params.get("n", 1)
            }
            
            # 使用async_rollout_manager生成序列
            result = self.async_rollout_manager.generate_sequences(
                prompts=batch, 
                **sampling_params
            )
            
            # 解析结果
            if result and len(result) > 0:
                # 提取生成的消息
                messages = result.non_tensor_batch.get("messages", [])
                if len(messages) > 0:
                    conversation_messages = messages[0]  # 取第一个结果
                    
                    return {
                        "messages": conversation_messages,
                        "generation_params": sampling_params,
                        "metadata": {
                            "num_turns": result.non_tensor_batch.get("__num_turns__", [0])[0]
                        }
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"RayConversationManager执行对话失败: {e}")
            return None
            
    async def _initialize_async_rollout_manager(self) -> None:
        """初始化async_rollout_manager"""
        try:
            # 准备配置
            config = self._prepare_rollout_config()
            
            # 如果Ray未初始化，先初始化
            if not ray.is_initialized():
                ray.init()
            
            # 创建resource pool
            role_worker_mapping = {
                Role.ActorRollout: ray.remote(AsyncActorRolloutRefWorker),
            }
            global_pool_id = "global_pool"
            resource_pool_spec = {
                global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
            }
            mapping = {
                Role.ActorRollout: global_pool_id,
            }
            resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
            resource_pool_manager.create_resource_pool()
            resource_pool_to_cls = {pool: {} for pool in resource_pool_manager.resource_pool_dict.values()}

            # 创建actor rollout workers
            resource_pool = resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=role_worker_mapping[Role.ActorRollout], 
                config=config.actor_rollout_ref, 
                role="actor_rollout"
            )
            resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls

            all_wg = {}
            for resource_pool, class_dict in resource_pool_to_cls.items():
                worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
                wg_dict = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
                spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
                all_wg.update(spawn_wg)
            
            actor_rollout_wg = all_wg["actor_rollout"]
            actor_rollout_wg.init_model()

            # 创建AsyncLLMServerManager
            self.async_rollout_manager = AsyncLLMServerManager(
                config=config,
                worker_group=actor_rollout_wg,
            )
            
            logger.info("async_rollout_manager初始化完成")
            
        except Exception as e:
            logger.error(f"async_rollout_manager初始化失败: {e}")
            # 创建一个简化的fallback实现
            self.async_rollout_manager = self._create_fallback_manager()
            logger.warning("使用fallback async_rollout_manager")
            
    def _prepare_rollout_config(self) -> DictConfig:
        """准备rollout配置"""
        # 基于当前配置构建rollout所需的完整配置
        # 这需要根据实际的配置结构调整
        
        rollout_config = self.config.copy()
        
        # 设置必要的rollout配置
        if not hasattr(rollout_config, 'actor_rollout_ref'):
            rollout_config.actor_rollout_ref = {}
            
        # 设置模型路径
        rollout_config.actor_rollout_ref.model = self.config.model
        
        # 设置rollout参数
        rollout_config.actor_rollout_ref.rollout = self.rollout_config
        
        # 设置trainer相关配置
        if not hasattr(rollout_config, 'trainer'):
            rollout_config.trainer = {
                'n_gpus_per_node': self.config.experiment.gpus_per_node,
                'nnodes': 1
            }
            
        # 设置completion callback
        rollout_config.actor_rollout_ref.rollout.multi_turn.completion_callback = self.config.completion_callback
        
        # 设置工具配置
        rollout_config.actor_rollout_ref.rollout.multi_turn.tool_config_path = self.config.tool_config_path
        
        return rollout_config
        
    def _create_fallback_manager(self):
        """创建fallback的async_rollout_manager"""
        class FallbackManager:
            def generate_sequences(self, prompts, **kwargs):
                # 简单的fallback实现，返回空结果
                logger.warning("使用fallback manager，返回空结果")
                
                # 构建空的返回结果
                batch_size = len(prompts.non_tensor_batch["raw_prompt"])
                return DataProto(
                    batch={},
                    non_tensor_batch={
                        "messages": [[] for _ in range(batch_size)],
                        "__num_turns__": np.zeros(batch_size, dtype=np.int32)
                    }
                )
        
        return FallbackManager()


class OaiConversationManager:
    """
    使用 OpenAI API 的简化对话管理器
    不需要 Ray 和本地 LLM 管理相关的逻辑
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.scheduler = None
        self._initialized = False
        
    async def initialize(self) -> None:
        """初始化对话管理器"""
        if self._initialized:
            return
            
        logger.info("初始化 OpenAI API 对话管理器...")
        
        # 导入并创建 OaiChatCompletionScheduler
        from deep_research.generate.core.oai_chat_scheduler import OaiChatCompletionScheduler
        
        # 准备配置，模拟 verl 的配置结构
        scheduler_config = self._prepare_scheduler_config()
        
        # 创建调度器
        self.scheduler = OaiChatCompletionScheduler(
            config=scheduler_config,
            server_addresses=[],  # OpenAI API 不需要服务器地址
        )
        
        self._initialized = True
        logger.info("OpenAI API 对话管理器初始化完成")
        
    async def cleanup(self) -> None:
        """清理资源"""
        logger.info("清理 OpenAI API 对话管理器资源...")
        # OpenAI API 不需要特殊清理
        logger.info("OpenAI API 对话管理器资源清理完成")
        
    async def execute_conversation(
        self,
        initial_messages: List[Dict[str, str]],
        skeleton: Optional[SkeletonData] = None,
        generation_params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        执行多轮对话生成
        
        Args:
            initial_messages: 初始对话消息
            skeleton: 可选的skeleton指导
            generation_params: 生成参数
            
        Returns:
            对话执行结果
        """
        if not self._initialized:
            raise RuntimeError("对话管理器未初始化")
            
        try:
            # 准备输入数据格式（DataProto格式）
            batch = DataProto(
                non_tensor_batch={
                    "raw_prompt": np.array([initial_messages], dtype=object),
                },
            )
            
            # 使用调度器生成序列
            # 注意：采样参数应该在配置中设置，而不是作为方法参数传递
            result = await self.scheduler.generate_sequences(batch)
            print(f"OaiConversationManage执行得到的result: {result}")
            # 解析结果
            if result and hasattr(result, 'non_tensor_batch'):
                # 提取生成的消息
                num_turns = result.non_tensor_batch.get("__num_turns__", [0])
                if len(num_turns) > 0:
                    return {
                        "messages": initial_messages,  # 完整的对话消息会在 completion_callback 中构建
                        "generation_params": generation_params or {},
                        "metadata": {
                            "num_turns": num_turns[0]
                        }
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"OaiConversationManager执行对话失败: {e}")
            return None
            
    def _prepare_scheduler_config(self) -> DictConfig:
        """准备调度器配置"""
        from omegaconf import OmegaConf
        
        # 构建符合 verl 结构的配置
        scheduler_config = OmegaConf.create({
            "actor_rollout_ref": {
                "model": {
                    "path": self.config.model.path,
                },
                "rollout": {
                    "temperature": self.config.generation.get("temperature", 0.7),
                    "top_p": self.config.generation.get("top_p", 0.9),
                    "n": self.config.generation.get("rollout_n", 1),
                    "multi_turn": {
                        "enable": True,
                        "max_turns": self.config.rollout.multi_turn.get("max_turns", 10),
                        "completion_callback": self.config.completion_callback,
                        "tool_config_path": self.config.tool_config_path,
                    }
                }
            },
            "openai_api": self.config.get("openai_api", {}),
        })
        
        return scheduler_config