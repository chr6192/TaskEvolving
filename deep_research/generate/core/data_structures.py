"""
数据合成Pipeline的核心数据结构定义
"""

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime


@dataclass
class QueryData:
    """用户查询数据结构"""
    query_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class ToolCall:
    """工具调用数据结构"""
    name: str
    parameters: Dict[str, Any]
    tool_call_id: Optional[str] = None
    
    def __post_init__(self):
        if self.tool_call_id is None:
            self.tool_call_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "parameters": self.parameters,
            "tool_call_id": self.tool_call_id
        }


@dataclass 
class StepData:
    """单个步骤的数据结构"""
    step: int
    tool_calls: List[ToolCall]
    dependencies: List[str] = field(default_factory=list)
    placeholder_values: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "dependencies": self.dependencies,
            "placeholder_values": self.placeholder_values
        }


@dataclass
class SkeletonData:
    """Skeleton数据结构"""
    skeleton_id: str
    query_id: str
    query_content: str
    analysis_logic: str
    multi_step: bool
    tool_results: List[StepData]
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def validate(self) -> bool:
        """验证skeleton格式正确性"""
        try:
            # 基本字段检查
            if not all([self.skeleton_id, self.query_id, self.analysis_logic]):
                return False
            
            # 检查tool_results
            if not self.tool_results:
                return False
                
            # 检查步骤顺序
            steps = [step.step for step in self.tool_results]
            if steps != list(range(len(steps))):
                return False
                
            # 检查工具调用
            for step_data in self.tool_results:
                if not step_data.tool_calls:
                    return False
                for tool_call in step_data.tool_calls:
                    if not tool_call.name or not isinstance(tool_call.parameters, dict):
                        return False
            
            return True
        except Exception:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "skeleton_id": self.skeleton_id,
            "query_id": self.query_id,  
            "query_content": self.query_content,
            "analysis_logic": self.analysis_logic,
            "multi_step": self.multi_step,
            "tool_results": [step.to_dict() for step in self.tool_results],
            "dependencies": self.dependencies,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkeletonData":
        """从字典创建SkeletonData实例"""
        tool_results = []
        for step_data in data["tool_results"]:
            tool_calls = [
                ToolCall(
                    name=tc["name"],
                    parameters=tc["parameters"],
                    tool_call_id=tc.get("tool_call_id")
                ) for tc in step_data["tool_calls"]
            ]
            tool_results.append(StepData(
                step=step_data["step"],
                tool_calls=tool_calls,
                dependencies=step_data.get("dependencies", []),
                placeholder_values=step_data.get("placeholder_values", {})
            ))
        
        return cls(
            skeleton_id=data["skeleton_id"],
            query_id=data["query_id"],
            query_content=data["query_content"],
            analysis_logic=data["analysis_logic"],
            multi_step=data["multi_step"],
            tool_results=tool_results,
            dependencies=data.get("dependencies", {}),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"])
        )


@dataclass
class ToolResponse:
    """工具响应数据结构"""
    tool_call_id: str
    content: str
    reward_score: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "content": self.content,
            "reward_score": self.reward_score,
            "metrics": self.metrics,
            "success": self.success,
            "error_message": self.error_message
        }


@dataclass
class TrajectoryData:
    """完整对话轨迹数据结构"""
    trajectory_id: str
    skeleton_id: Optional[str]
    query_id: str
    messages: List[Dict[str, str]]
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_responses: List[ToolResponse] = field(default_factory=list)
    prm_score: float = 0.0
    orm_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_final_response(self) -> str:
        """获取最终的助手回复"""
        for message in reversed(self.messages):
            if message.get("role") == "assistant" and message.get("content"):
                return message["content"]
        return ""
    
    def get_turn_count(self) -> int:
        """获取对话轮数"""
        return len(self.messages)
    
    def export_for_training(self) -> Dict[str, Any]:
        """导出为训练数据格式"""
        return {
            "trajectory_id": self.trajectory_id,
            "messages": self.messages,
            "orm_scores": self.orm_scores,
            "prm_score": self.prm_score,
            "quality_score": sum(self.orm_scores.values()) / len(self.orm_scores) if self.orm_scores else 0.0
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "skeleton_id": self.skeleton_id,
            "query_id": self.query_id,
            "messages": self.messages,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "tool_responses": [tr.to_dict() for tr in self.tool_responses], 
            "prm_score": self.prm_score,
            "orm_scores": self.orm_scores,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryData":
        """从字典创建TrajectoryData实例"""
        tool_calls = [
            ToolCall(
                name=tc["name"],
                parameters=tc["parameters"],
                tool_call_id=tc.get("tool_call_id")
            ) for tc in data.get("tool_calls", [])
        ]
        
        tool_responses = [
            ToolResponse(
                tool_call_id=tr["tool_call_id"],
                content=tr["content"],
                reward_score=tr.get("reward_score", 0.0),
                metrics=tr.get("metrics", {}),
                success=tr.get("success", True),
                error_message=tr.get("error_message")
            ) for tr in data.get("tool_responses", [])
        ]
        
        return cls(
            trajectory_id=data["trajectory_id"],
            skeleton_id=data.get("skeleton_id"),
            query_id=data["query_id"],
            messages=data["messages"],
            tool_calls=tool_calls,
            tool_responses=tool_responses,
            prm_score=data.get("prm_score", 0.0),
            orm_scores=data.get("orm_scores", {}),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"])
        )


@dataclass
class EvaluationResult:
    """评估结果数据结构"""
    trajectory_id: str
    orm_scores: Dict[str, float]
    prm_score: float
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    detailed_feedback: Dict[str, Any] = field(default_factory=dict)
    passed_quality_check: bool = False
    
    def get_overall_score(self) -> float:
        """获取综合质量分数"""
        if not self.orm_scores:
            return self.prm_score
        
        # 综合ORM和PRM分数，ORM权重更高
        orm_avg = sum(self.orm_scores.values()) / len(self.orm_scores)
        return 0.7 * orm_avg + 0.3 * self.prm_score
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "orm_scores": self.orm_scores,
            "prm_score": self.prm_score,
            "quality_metrics": self.quality_metrics,
            "detailed_feedback": self.detailed_feedback,
            "passed_quality_check": self.passed_quality_check,
            "overall_score": self.get_overall_score()
        }


@dataclass
class PipelineStats:
    """Pipeline统计信息"""
    total_queries: int = 0
    generated_skeletons: int = 0
    valid_skeletons: int = 0
    generated_trajectories: int = 0
    high_quality_trajectories: int = 0
    stage1_time: float = 0.0
    stage2_time: float = 0.0
    total_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "generated_skeletons": self.generated_skeletons,
            "valid_skeletons": self.valid_skeletons,
            "skeleton_success_rate": self.valid_skeletons / max(self.generated_skeletons, 1),
            "generated_trajectories": self.generated_trajectories,
            "high_quality_trajectories": self.high_quality_trajectories,
            "trajectory_success_rate": self.high_quality_trajectories / max(self.generated_trajectories, 1),
            "stage1_time": self.stage1_time,
            "stage2_time": self.stage2_time,
            "total_time": self.total_time
        } 