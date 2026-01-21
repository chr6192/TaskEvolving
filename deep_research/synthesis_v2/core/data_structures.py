"""
标准接口和数据结构
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime


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
