"""
API数据模型定义
使用Pydantic进行数据验证
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator


class ComputeConfig(BaseModel):
    """计算配置"""
    return_details: bool = Field(default=True, description="是否返回详细信息")
    return_statistics: bool = Field(default=True, description="是否返回统计信息")
    cache_ttl: int = Field(default=3600, description="缓存过期时间（秒）", ge=0, le=86400)
    timeout: int = Field(default=300, description="计算超时时间（秒）", ge=1, le=3600)


class ComputeRequest(BaseModel):
    """同步计算请求"""
    prompt: str = Field(..., description="问题文本")
    response_list: List[str] = Field(..., description="模型响应列表", min_items=1)
    messages_list: List[List[Dict[str, Any]]] = Field(..., description="对话消息列表", min_items=1)
    extra_list: List[Dict[str, Any]] = Field(..., description="额外信息列表", min_items=1)
    config: Optional[ComputeConfig] = Field(default=None, description="计算配置")
    
    @validator('response_list', 'messages_list', 'extra_list')
    def check_length_match(cls, v, values):
        """验证列表长度是否匹配"""
        if 'response_list' in values and len(v) != len(values['response_list']):
            raise ValueError("response_list, messages_list, extra_list 的长度必须一致")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "如何购买基金？",
                "response_list": ["您可以通过手机银行购买基金...", "建议您先了解基金知识..."],
                "messages_list": [
                    [
                        {"role": "user", "content": "如何购买基金？"},
                        {"role": "assistant", "content": "您可以通过手机银行购买基金..."}
                    ],
                    [
                        {"role": "user", "content": "如何购买基金？"},
                        {"role": "assistant", "content": "建议您先了解基金知识..."}
                    ]
                ],
                "extra_list": [
                    {
                        "skeleton_id": "skel_001",
                        "trajectory_id": "traj_001",
                        "skeleton": {"query": "如何购买基金？", "tool_calls_reference": []},
                        "tool_calls": [],
                        "tool_responses": []
                    },
                    {
                        "skeleton_id": "skel_001",
                        "trajectory_id": "traj_002",
                        "skeleton": {"query": "如何购买基金？", "tool_calls_reference": []},
                        "tool_calls": [],
                        "tool_responses": []
                    }
                ],
                "config": {
                    "return_details": True,
                    "cache_ttl": 3600
                }
            }
        }


class ComputeResponse(BaseModel):
    """同步计算响应"""
    success: bool = Field(..., description="是否成功")
    data: Optional[Dict[str, Any]] = Field(default=None, description="结果数据")
    meta: Optional[Dict[str, Any]] = Field(default=None, description="元信息")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "data": {
                    "total_scores": [0.85, 0.72],
                    "reward_info": {
                        "faithful_num_exist_reward": [0.1, 0.2],
                        "map_format_reward": [0.3, 0.4],
                        "rubrics_reward": [0.45, 0.12]
                    },
                    "details": [
                        {
                            "index": 0,
                            "total_score": 0.85,
                            "performance_stats": {"total_time": 2.34}
                        }
                    ]
                },
                "meta": {
                    "processing_time": 2.34,
                    "sample_count": 2,
                    "timestamp": "2025-11-27T10:00:00"
                }
            }
        }


class AsyncComputeRequest(BaseModel):
    """异步计算请求"""
    task_id: Optional[str] = Field(default=None, description="可选的自定义任务ID")
    data: ComputeRequest = Field(..., description="计算请求数据")
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "custom_task_123",
                "data": {
                    "prompt": "如何购买基金？",
                    "response_list": ["回复1", "回复2"],
                    "messages_list": [[{"role": "user", "content": "问题"}]],
                    "extra_list": [{"skeleton_id": "001"}],
                }
            }
        }


class AsyncComputeResponse(BaseModel):
    """异步计算响应"""
    success: bool = Field(..., description="是否成功")
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态")
    estimated_time: Optional[float] = Field(default=None, description="预估完成时间（秒）")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "task_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "pending",
                "estimated_time": 120
            }
        }


class TaskStatusResponse(BaseModel):
    """任务状态响应"""
    success: bool = Field(..., description="是否成功")
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态: pending/processing/completed/failed")
    progress: float = Field(..., description="进度 0-1", ge=0.0, le=1.0)
    created_at: str = Field(..., description="创建时间")
    updated_at: str = Field(..., description="更新时间")
    result: Optional[Dict[str, Any]] = Field(default=None, description="计算结果（仅completed状态）")
    error: Optional[str] = Field(default=None, description="错误信息（仅failed状态）")
    processing_time: Optional[float] = Field(default=None, description="处理时间（秒）")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "task_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "completed",
                "progress": 1.0,
                "created_at": "2025-11-27T10:00:00",
                "updated_at": "2025-11-27T10:02:00",
                "result": {
                    "total_scores": [0.85, 0.72],
                    "reward_info": {}
                },
                "processing_time": 120.5
            }
        }


class BatchGroupRequest(BaseModel):
    """单个group的请求"""
    group_id: str = Field(..., description="Group ID")
    prompt: str = Field(..., description="问题文本")
    response_list: List[str] = Field(..., description="响应列表", min_items=1)
    messages_list: List[List[Dict[str, Any]]] = Field(..., description="消息列表", min_items=1)
    extra_list: List[Dict[str, Any]] = Field(..., description="额外信息列表", min_items=1)


class BatchComputeRequest(BaseModel):
    """批量计算请求"""
    groups: List[BatchGroupRequest] = Field(..., description="Group列表", min_items=1)
    config: Optional[ComputeConfig] = Field(default=None, description="计算配置")
    
    class Config:
        json_schema_extra = {
            "example": {
                "groups": [
                    {
                        "group_id": "group_1",
                        "prompt": "问题1",
                        "response_list": ["回复1"],
                        "messages_list": [[{"role": "user", "content": "问题1"}]],
                        "extra_list": [{"skeleton_id": "001"}]
                    },
                    {
                        "group_id": "group_2",
                        "prompt": "问题2",
                        "response_list": ["回复2"],
                        "messages_list": [[{"role": "user", "content": "问题2"}]],
                        "extra_list": [{"skeleton_id": "002"}]
                    }
                ],
                "config": {"return_details": True}
            }
        }


class BatchComputeResponse(BaseModel):
    """批量计算响应"""
    success: bool = Field(..., description="是否成功")
    results: List[Dict[str, Any]] = Field(..., description="各group的结果")
    meta: Dict[str, Any] = Field(..., description="元信息")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "results": [
                    {
                        "group_id": "group_1",
                        "status": "success",
                        "total_scores": [0.85],
                        "reward_info": {}
                    },
                    {
                        "group_id": "group_2",
                        "status": "success",
                        "total_scores": [0.72],
                        "reward_info": {}
                    }
                ],
                "meta": {
                    "total_groups": 2,
                    "total_processing_time": 5.0
                }
            }
        }


class ErrorResponse(BaseModel):
    """错误响应"""
    success: bool = Field(default=False, description="是否成功")
    error: Dict[str, Any] = Field(..., description="错误信息")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Invalid input data",
                    "details": {}
                }
            }
        }


class MetricsResponse(BaseModel):
    """监控指标响应"""
    success: bool = Field(..., description="是否成功")
    metrics: Dict[str, Any] = Field(..., description="监控指标")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "metrics": {
                    "service": {
                        "total_requests": 1000,
                        "cache_hits": 500,
                        "cache_hit_rate": 0.5,
                        "avg_compute_time": 2.5
                    },
                    "queue": {
                        "queue_size": 10,
                        "total_tasks": 100,
                        "status_counts": {
                            "pending": 5,
                            "processing": 3,
                            "completed": 90,
                            "failed": 2
                        }
                    }
                }
            }
        }


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    timestamp: str = Field(..., description="时间戳")
    version: str = Field(default="1.0.0", description="API版本")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-11-27T10:00:00",
                "version": "1.0.0"
            }
        }

