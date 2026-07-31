"""
标准接口和数据结构
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import uuid

from pydantic import BaseModel, Field, field_validator


@dataclass
class WorkflowInput:
    """Workflow标准输入 - 兼容OpenAI chat messages格式"""
    messages: List[Dict[str, Any]]  # [{"role": "user", "content": "..."}]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def input_id(self) -> str:
        """获取或生成input_id"""
        if 'input_id' not in self.metadata:
            import uuid
            self.metadata['input_id'] = str(uuid.uuid4())
        return self.metadata['input_id']


@dataclass
class WorkflowOutput:
    """Workflow标准输出"""
    messages: List[Dict[str, Any]]  # 完整对话历史
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_responses: List[Dict[str, Any]] = field(default_factory=list)
    final_response: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


@dataclass
class EvaluationResult:
    """评估结果"""
    trajectory_id: str
    scores: Dict[str, float] = field(default_factory=dict)  # 各维度得分
    details: Dict[str, Any] = field(default_factory=dict)  # 详细信息
    passed: bool = False
    
    @property
    def score(self) -> float:
        """综合得分（用于Best of N）"""
        return sum(self.scores.values()) / len(self.scores) if self.scores else 0.0


@dataclass
class PipelineResult:
    """单条样本Pipeline结果"""
    input: WorkflowInput
    output: WorkflowOutput
    evaluation: Optional[EvaluationResult] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def success(self) -> bool:
        return self.output.success
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            'input': {
                'messages': self.input.messages,
                'metadata': self.input.metadata
            },
            'output': {
                'messages': self.output.messages,
                'tool_calls': self.output.tool_calls,
                'tool_responses': self.output.tool_responses,
                'final_response': self.output.final_response,
                'metadata': self.output.metadata,
                'success': self.output.success,
                'error': self.output.error
            },
            'evaluation': {
                'trajectory_id': self.evaluation.trajectory_id,
                'scores': self.evaluation.scores,
                'details': self.evaluation.details,
                'passed': self.evaluation.passed,
                'score': self.evaluation.score
            } if self.evaluation else None,
            'timestamp': self.timestamp.isoformat()
        }


class RubricsData(BaseModel):
    """与 OpenJudge 兼容的评分标准。"""

    query_id: str
    criteria: List[Dict[str, Any]]
    complexity_analysis: Optional[str] = None
    cross_domain_connections: Optional[List[str]] = None
    fuzzy_query: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator("criteria", mode="before")
    @classmethod
    def parse_criteria(cls, value):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return []
        if isinstance(value, list):
            parsed = []
            for item in value:
                if isinstance(item, dict):
                    parsed.append(item)
                elif isinstance(item, str):
                    try:
                        candidate = json.loads(item)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(candidate, dict):
                        parsed.append(candidate)
            return parsed
        return value

    @field_validator("cross_domain_connections", mode="before")
    @classmethod
    def parse_cross_domain_connections(cls, value):
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return None
            return parsed if isinstance(parsed, list) else None
        return value

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RubricsData":
        return cls.model_validate(data)


class QueryData(BaseModel):
    """用户查询数据。"""

    query_id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    rubrics: Optional[RubricsData] = None
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = {"arbitrary_types_allowed": True}

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")


class ToolCall(BaseModel):
    """一次工具调用。"""

    name: str
    parameters: Dict[str, Any]
    tool_call_id: Optional[str] = None

    def model_post_init(self, __context):
        if self.tool_call_id is None:
            self.tool_call_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"name": self.name, "parameters": self.parameters}
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result


class ToolResponse(BaseModel):
    """一次工具响应。"""

    tool_call_id: str
    content: str
    reward_score: float = 0.0
    metrics: Dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")


class TrajectoryData(BaseModel):
    """完整对话轨迹。"""

    trajectory_id: str
    query_id: str
    query: Optional[str] = None
    messages: List[Dict[str, Any]]
    tool_calls: List[ToolCall] = Field(default_factory=list)
    tool_responses: List[ToolResponse] = Field(default_factory=list)
    orm_scores: Dict[str, float] = Field(default_factory=dict)
    orm_details: Dict[str, Any] = Field(default_factory=dict)
    selection_status: str = "pending"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = {"arbitrary_types_allowed": True}

    def get_final_response(self) -> str:
        for message in reversed(self.messages):
            if message.get("role") == "assistant" and message.get("content"):
                return str(message["content"])
        return ""

    def get_turn_count(self) -> int:
        return len(self.messages)

    def export_for_training(self) -> Dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "messages": self.messages,
            "orm_scores": self.orm_scores,
            "orm_details": self.orm_details,
            "quality_score": (
                sum(self.orm_scores.values()) / len(self.orm_scores)
                if self.orm_scores
                else 0.0
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryData":
        return cls.model_validate(data)


class ORMEvaluationResult(BaseModel):
    """ORM 对单条轨迹的评估结果。"""

    trajectory_id: str
    orm_scores: Dict[str, float]
    orm_details: Dict[str, Any] = Field(default_factory=dict)
    quality_metrics: Dict[str, Any] = Field(default_factory=dict)
    reward_info: Dict[str, Any] = Field(default_factory=dict)
    passed_quality_check: bool = False

    def get_overall_score(self) -> float:
        if not self.orm_scores:
            return 0.0
        return sum(self.orm_scores.values()) / len(self.orm_scores)

    def to_dict(self) -> Dict[str, Any]:
        result = self.model_dump(mode="json")
        result["overall_score"] = self.get_overall_score()
        return result


class IterationData(BaseModel):
    """复杂度探索的一轮迭代。"""

    iteration_id: int
    query: QueryData
    trajectory: Optional[TrajectoryData] = None
    rewards: Dict[str, float] = Field(default_factory=dict)
    rubrics: Optional[RubricsData] = None
    quality_score: float = 0.0
    rubrics_evaluation: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")


class QueryEvolution(BaseModel):
    """Query 的复杂度演化过程。"""

    evolution_id: str
    original_query: QueryData
    iterations: List[IterationData] = Field(default_factory=list)
    final_query: Optional[QueryData] = None
    final_rubrics: Optional[RubricsData] = None
    total_iterations: int = 0
    converged: bool = False
    convergence_reason: str = ""
    next_iteration_preparations: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = {"arbitrary_types_allowed": True}

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    def get_complexity_progression(self) -> List[float]:
        return [iteration.quality_score for iteration in self.iterations]
