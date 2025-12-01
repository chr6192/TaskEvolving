"""
配置管理
支持环境变量和配置文件
"""
import os
from typing import Optional

try:
    # Pydantic V2
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    # Pydantic V1
    from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """应用配置"""
    
    # ========== 服务配置 ==========
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=False, env="API_DEBUG")
    
    # ========== 并发控制 ==========
    max_concurrent_groups: int = Field(
        default=10,
        env="MAX_CONCURRENT_GROUPS",
        description="最大并发group数量"
    )
    num_task_processors: int = Field(
        default=8,
        env="NUM_TASK_PROCESSORS",
        description="进程内异步任务处理器数量"
    )
    max_queue_size: int = Field(
        default=1000,
        env="MAX_QUEUE_SIZE",
        description="最大队列大小"
    )
    
    # ========== 缓存配置 ==========
    enable_cache: bool = Field(default=True, env="ENABLE_CACHE")
    cache_type: str = Field(
        default="memory",
        env="CACHE_TYPE",
        description="缓存类型: memory 或 redis"
    )
    cache_max_size: int = Field(
        default=10000,
        env="CACHE_MAX_SIZE",
        description="内存缓存最大条目数"
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL",
        description="Redis连接URL"
    )
    
    # ========== 日志配置 ==========
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # ========== OpenAI配置（用于compute_score_async内部） ==========
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_base_url: Optional[str] = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        env="OPENAI_BASE_URL"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# 创建全局配置实例
settings = Settings()

# 确保环境变量设置（用于compute_score_async内部使用）
if settings.openai_api_key:
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key
if settings.openai_base_url:
    os.environ["OPENAI_BASE_URL"] = settings.openai_base_url

