"""
原生 Query→Trajectory Workflow（不使用 Skeleton）

将 deep_research.synthesis.core.trajectory_generator.TrajectoryGenerator.generate_from_query
封装为标准 workflow，保持与 synthesis_v2 pipeline/evaluator 的接口一致。
"""

import logging
import uuid
from typing import Optional

from omegaconf import DictConfig

from ..base import WorkflowBase
from ...core.data_structures import WorkflowInput, WorkflowOutput
from deep_research.synthesis.core.data_structures import QueryData
from deep_research.synthesis.core.trajectory_generator import TrajectoryGenerator

logger = logging.getLogger(__name__)


class QueryTrajectoryWorkflow(WorkflowBase):
    """从 query 直接生成 trajectory 的 workflow（完全绕过 Skeleton）。"""

    def __init__(self, config: DictConfig):
        self.config = config
        self.trajectory_generator: Optional[TrajectoryGenerator] = None
        self._initialized = False

    @property
    def name(self) -> str:
        return "query_trajectory_workflow"

    async def initialize(self) -> None:
        if self._initialized:
            return

        self.trajectory_generator = TrajectoryGenerator(self.config)
        await self.trajectory_generator.initialize()

        self._initialized = True
        logger.info("QueryTrajectoryWorkflow初始化完成")

    async def execute(self, input: WorkflowInput) -> WorkflowOutput:
        try:
            query = self._extract_query_from_messages(input.messages)
            query_data = QueryData(
                query_id=str(uuid.uuid4()),
                content=query,
                metadata=input.metadata,
            )

            trajectories = await self.trajectory_generator.generate_from_query(query_data, n_samples=1)
            if not trajectories:
                return WorkflowOutput(
                    messages=input.messages,
                    success=False,
                    error="trajectory generation failed",
                )

            trajectory = trajectories[0]

            return WorkflowOutput(
                messages=trajectory.messages,
                tool_calls=[tc.to_dict() for tc in trajectory.tool_calls],
                tool_responses=[tr.to_dict() for tr in trajectory.tool_responses],
                final_response=trajectory.get_final_response(),
                metadata={
                    "trajectory_id": trajectory.trajectory_id,
                    "query_id": query_data.query_id,
                    "query": query,
                    **trajectory.metadata,
                },
                success=True,
            )

        except Exception as e:
            logger.error(f"Workflow执行失败: {e}", exc_info=True)
            return WorkflowOutput(
                messages=input.messages,
                success=False,
                error=str(e),
            )

    def _extract_query_from_messages(self, messages: list) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    async def cleanup(self) -> None:
        if self.trajectory_generator:
            await self.trajectory_generator.cleanup()
        logger.info("QueryTrajectoryWorkflow清理完成")

