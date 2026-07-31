"""Query-only OpenAI 多轮对话管理器。"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class OaiConversationManager:
    def __init__(self, config: DictConfig):
        self.config = config
        self.scheduler = None
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        from .oai_chat_scheduler import OaiChatCompletionScheduler

        self.scheduler = OaiChatCompletionScheduler(
            config=self._prepare_scheduler_config(),
            server_addresses=[],
        )
        self._initialized = True

    async def cleanup(self) -> None:
        self.scheduler = None
        self._initialized = False

    async def execute_conversation(
        self,
        initial_messages: List[Dict[str, Any]],
        generation_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self._initialized or self.scheduler is None:
            raise RuntimeError("对话管理器未初始化")

        try:
            from verl.protocol import DataProto

            batch = DataProto(
                non_tensor_batch={
                    "raw_prompt": np.array([initial_messages], dtype=object),
                },
            )
            result = await self.scheduler.generate_sequences(batch)
            if not result or not hasattr(result, "non_tensor_batch"):
                return None

            messages_data = result.non_tensor_batch.get("messages", [])
            if len(messages_data) == 0:
                return None
            messages_list = (
                messages_data.tolist()
                if hasattr(messages_data, "tolist")
                else messages_data
            )
            final_messages = (
                messages_list[0]
                if isinstance(messages_list, list)
                and messages_list
                and isinstance(messages_list[0], list)
                else messages_list
            )
            num_turns = result.non_tensor_batch.get("__num_turns__", [])
            return {
                "messages": final_messages,
                "generation_params": generation_params or {},
                "metadata": {
                    "num_turns": num_turns[0] if len(num_turns) else len(final_messages)
                },
            }
        except Exception:
            logger.exception("OpenAI Query 对话执行失败")
            return None

    def _prepare_scheduler_config(self) -> DictConfig:
        openai_config = self.config.get("openai_api", {})
        generation = self.config.get("generation", {})
        model_name = generation.get("model_name") or openai_config.get(
            "default_model_name"
        )
        if not model_name:
            raise ValueError(
                "Trajectory生成阶段未配置模型：请设置 generation.model_name "
                "或 openai_api.default_model_name"
            )

        return OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "model": {"path": None, "name": model_name},
                    "rollout": {
                        "temperature": generation.get("temperature", 0.7),
                        "top_p": generation.get("top_p", 0.9),
                        "n": generation.get("rollout_n", 1),
                        "multi_turn": {
                            "enable": True,
                            "max_turns": self.config.multi_turn.get("max_turns", 10),
                            "completion_callback": self.config.multi_turn.get(
                                "completion_callback"
                            ),
                            "tool_config_path": self.config.multi_turn.get(
                                "tool_config_path"
                            ),
                        },
                    },
                },
                "openai_api": openai_config,
                "generation": generation,
                "stage": self.config.get("stage", "trajectory"),
            }
        )
