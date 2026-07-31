"""直接从 Query 生成完整对话轨迹。"""

import asyncio
import importlib.util
import logging
import sys
import uuid
from typing import Any, Callable, Dict, List, Optional

from omegaconf import DictConfig

from ..mermaid import Mermaid
from .agent_framework import AgentFramework
from .data_structures import QueryData, TrajectoryData

logger = logging.getLogger(__name__)


class TrajectoryGenerator:
    """Query-only trajectory 生成器。"""

    def __init__(self, config: DictConfig):
        self.config = config
        self.generation_config = config.generation
        openai_config = config.get("openai_api", {})
        self.model_name = self.generation_config.get(
            "model_name"
        ) or openai_config.get("default_model_name")
        if not self.model_name:
            raise ValueError(
                "Trajectory生成阶段未配置模型：请设置 generation.model_name "
                "或 openai_api.default_model_name"
            )

        self.get_original_prompt_func = self._load_trajectory_prompt_module()
        self.temperature = self.generation_config.get(
            "temperature"
        ) or openai_config.get("default_temperature", 0.7)
        self.top_p = self.generation_config.get("top_p") or openai_config.get(
            "default_top_p", 0.9
        )
        self.max_tokens = self.generation_config.get(
            "max_tokens"
        ) or openai_config.get("default_max_tokens", 2048)
        self.agent_framework: Optional[AgentFramework] = None
        self._initialized = False

    def _load_trajectory_prompt_module(self) -> Callable[[str], str]:
        prompt_path = self.generation_config.get(
            "trajectory_prompt_template_path",
            "deep_research/synthesis_v2/prompts/"
            "trajectory_generate_prompt_general_search.py",
        )
        try:
            spec = importlib.util.spec_from_file_location(
                "synthesis_v2_trajectory_prompt",
                prompt_path,
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"无法加载 prompt 模块: {prompt_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module.get_original_prompt
        except Exception:
            logger.exception("加载 trajectory prompt 失败，使用 v2 默认模板")
            from ..prompts.trajectory_generate_prompt_general_search import (
                get_original_prompt,
            )

            return get_original_prompt

    def _build_initial_messages(
        self,
        query: str,
        mermaid: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        prompt = self.get_original_prompt_func(query)
        if (
            self.generation_config.get("inject_mermaid_in_prompt", False)
            and isinstance(mermaid, str)
            and mermaid.strip()
        ):
            prompt += Mermaid(
                mermaid=mermaid.strip()
            ).as_system_prompt_injection()
        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ]

    async def initialize(self) -> None:
        if self._initialized:
            return
        self.agent_framework = AgentFramework(self.config)
        await self.agent_framework.initialize()
        self._initialized = True

    async def cleanup(self) -> None:
        if self.agent_framework:
            await self.agent_framework.cleanup()
        self._initialized = False

    async def generate_from_query(
        self,
        query: QueryData,
        n_samples: int,
    ) -> List[TrajectoryData]:
        if not self._initialized or self.agent_framework is None:
            raise RuntimeError("Trajectory生成器未初始化")

        initial_messages = self._build_initial_messages(
            query=query.content,
            mermaid=(query.metadata or {}).get("mermaid"),
        )
        results = await asyncio.gather(
            *(
                self._generate_single_trajectory(
                    query,
                    initial_messages,
                    sample_index,
                )
                for sample_index in range(n_samples)
            ),
            return_exceptions=True,
        )
        trajectories: List[TrajectoryData] = []
        for result in results:
            if isinstance(result, Exception):
                logger.error("生成 trajectory 失败: %s", result)
            elif result is not None:
                trajectories.append(result)
        return trajectories

    async def _generate_single_trajectory(
        self,
        query: QueryData,
        initial_messages: List[Dict[str, str]],
        sample_index: int,
    ) -> Optional[TrajectoryData]:
        if self.agent_framework is None:
            return None
        result = await self.agent_framework.execute_trajectory(
            initial_messages=[dict(message) for message in initial_messages],
            generation_params={
                "temperature": self._get_varied_temperature(sample_index),
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
            },
        )
        if not result:
            return None
        return TrajectoryData(
            trajectory_id=str(uuid.uuid4()),
            query_id=query.query_id,
            query=query.content,
            messages=result["messages"],
            tool_calls=result.get("tool_calls", []),
            tool_responses=result.get("tool_responses", []),
            metadata={
                "sample_index": sample_index,
                "generation_params": result.get("generation_params", {}),
                "direct_from_query": True,
                **query.metadata,
            },
        )

    def _get_varied_temperature(self, sample_index: int) -> float:
        if sample_index == 0:
            return self.temperature
        if sample_index % 3 == 1:
            return min(1.0, self.temperature + 0.1)
        if sample_index % 3 == 2:
            return max(0.1, self.temperature - 0.1)
        return self.temperature
