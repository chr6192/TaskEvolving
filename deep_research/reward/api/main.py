"""
Reward API 主应用
FastAPI服务入口
"""
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    ComputeRequest,
    ComputeResponse,
    AsyncComputeRequest,
    AsyncComputeResponse,
    TaskStatusResponse,
    BatchComputeRequest,
    BatchComputeResponse,
    ErrorResponse,
    MetricsResponse,
    HealthResponse,
)
from .core.reward_service import RewardComputeService
from .core.task_queue import TaskQueue
from .core.cache import create_cache_backend
from .config import settings
from .middleware import setup_logging, request_logger

# 设置日志
setup_logging(settings.log_level, settings.log_file)
logger = logging.getLogger(__name__)

# 全局服务实例
reward_service: Optional[RewardComputeService] = None
task_queue: Optional[TaskQueue] = None
cache_backend = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global reward_service, task_queue, cache_backend
    
    logger.info("=" * 80)
    logger.info("启动 Reward API 服务")
    logger.info("=" * 80)
    
    try:
        # 1. 初始化缓存后端
        logger.info(f"初始化缓存后端: {settings.cache_type}")
        cache_backend = await create_cache_backend(
            cache_type=settings.cache_type,
            max_size=settings.cache_max_size,
            redis_url=settings.redis_url,
        )
        logger.info("✓ 缓存后端初始化成功")
        
        # 2. 初始化 compute_score_async 函数
        logger.info("加载 compute_score_async 函数")
        from deep_research.reward.dr_reward_function_bl import compute_score_async
        logger.info("✓ compute_score_async 加载成功")
        
        # 3. 初始化奖励计算服务
        logger.info(f"初始化奖励计算服务: max_concurrent_groups={settings.max_concurrent_groups}")
        reward_service = RewardComputeService(
            compute_score_async_fn=compute_score_async,
            max_concurrent_groups=settings.max_concurrent_groups,
            enable_cache=settings.enable_cache,
            cache_backend=cache_backend,
            enable_persistence=settings.enable_persistence,
            persistence_dir=settings.persistence_dir,
        )
        logger.info("✓ 奖励计算服务初始化成功")
        
        # 4. 初始化任务队列
        logger.info(f"初始化任务队列: max_queue_size={settings.max_queue_size}")
        task_queue = TaskQueue(
            cache_backend=cache_backend,
            max_queue_size=settings.max_queue_size,
        )
        logger.info("✓ 任务队列初始化成功")
        
        # 5. 启动后台异步任务处理器
        logger.info(f"启动 {settings.num_task_processors} 个异步任务处理器（async task processors）")
        await task_queue.start_processors(
            num_processors=settings.num_task_processors,
            reward_service=reward_service,
        )
        logger.info("✓ 异步任务处理器启动成功")
        
        logger.info("=" * 80)
        logger.info("✅ Reward API 服务启动完成")
        logger.info(f"📍 服务地址: http://{settings.host}:{settings.port}")
        logger.info(f"📖 API文档: http://{settings.host}:{settings.port}/docs")
        logger.info("=" * 80)
        
        yield  # 应用运行
        
    except Exception as e:
        logger.error(f"服务启动失败: {e}", exc_info=True)
        raise
    
    finally:
        # 清理资源
        logger.info("=" * 80)
        logger.info("关闭 Reward API 服务")
        logger.info("=" * 80)
        
        if task_queue:
            logger.info("停止异步任务处理器...")
            await task_queue.stop_processors()
            logger.info("✓ 异步任务处理器已停止")
        
        if cache_backend and hasattr(cache_backend, 'disconnect'):
            logger.info("关闭缓存连接...")
            await cache_backend.disconnect()
            logger.info("✓ 缓存连接已关闭")
        
        logger.info("=" * 80)
        logger.info("✅ Reward API 服务已关闭")
        logger.info("=" * 80)


# 创建FastAPI应用
app = FastAPI(
    title="Reward Compute API",
    description="高并发异步奖励计算API服务",
    version="1.0.0",
    lifespan=lifespan,
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加请求日志中间件
app.middleware("http")(request_logger)


# ============= 异常处理 =============

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": exc.detail,
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理"""
    logger.error(f"未处理的异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": str(exc),
            }
        }
    )


# ============= API路由 =============

@app.get("/", response_model=HealthResponse)
async def root():
    """根路径"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
    }


@app.post("/api/v1/rewards/compute", response_model=ComputeResponse)
async def compute_rewards_sync(request: ComputeRequest):
    """
    同步计算奖励
    - 立即计算并返回结果
    - 适合小批量、需要即时响应的场景
    """
    if reward_service is None:
        raise HTTPException(status_code=503, detail="服务未就绪")
    
    try:
        config = request.config.dict() if request.config else {}
        
        result = await reward_service.compute_single_group(
            prompt=request.prompt,
            response_list=request.response_list,
            messages_list=request.messages_list,
            extra_list=request.extra_list,
            config=config,
        )
        
        return {
            "success": True,
            "data": result,
        }
        
    except Exception as e:
        logger.error(f"同步计算失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/rewards/compute_async", response_model=AsyncComputeResponse)
async def compute_rewards_async(request: AsyncComputeRequest):
    """
    异步计算奖励
    - 任务入队，返回任务ID
    - 适合大批量、可以等待的场景
    """
    if task_queue is None:
        raise HTTPException(status_code=503, detail="服务未就绪")
    
    try:
        config = request.data.config.dict() if request.data.config else {}
        
        task_id = await task_queue.submit_task(
            prompt=request.data.prompt,
            responses=request.data.response_list,
            messages=request.data.messages_list,
            extras=request.data.extra_list,
            config=config,
            task_id=request.task_id,
        )
        
        # 估算完成时间（基于队列大小和历史平均处理时间）
        queue_stats = task_queue.get_stats()
        avg_time = 30.0  # 默认估算30秒
        if reward_service:
            metrics = reward_service.get_metrics()
            if metrics.get("avg_compute_time"):
                avg_time = metrics["avg_compute_time"]
        
        estimated_time = queue_stats["queue_size"] * avg_time / settings.num_task_processors
        
        return {
            "success": True,
            "task_id": task_id,
            "status": "pending",
            "estimated_time": estimated_time,
        }
        
    except ValueError as e:
        # 队列已满
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        logger.error(f"异步任务提交失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/rewards/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    查询任务状态
    - 根据任务ID查询任务的处理状态和结果
    """
    if task_queue is None:
        raise HTTPException(status_code=503, detail="服务未就绪")
    
    try:
        task_status = await task_queue.get_task_status(task_id)
        
        if task_status is None:
            raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
        
        return {
            "success": True,
            **task_status,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询任务状态失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/rewards/batch_compute", response_model=BatchComputeResponse)
async def compute_batch_rewards(request: BatchComputeRequest):
    """
    批量计算奖励
    - 同时计算多个group的奖励
    - 返回所有group的结果
    """
    if reward_service is None:
        raise HTTPException(status_code=503, detail="服务未就绪")
    
    try:
        config = request.config.dict() if request.config else {}
        
        # 构建groups列表
        groups = []
        for group in request.groups:
            groups.append({
                "group_id": group.group_id,
                "prompt": group.prompt,
                "response_list": group.response_list,
                "messages_list": group.messages_list,
                "extra_list": group.extra_list,
            })
        
        start_time = time.perf_counter()
        results = await reward_service.compute_batch_groups(groups, config)
        total_time = time.perf_counter() - start_time
        
        # 添加group_id到结果
        for i, result in enumerate(results):
            result["group_id"] = groups[i]["group_id"]
        
        return {
            "success": True,
            "results": results,
            "meta": {
                "total_groups": len(groups),
                "total_processing_time": total_time,
                "timestamp": datetime.now().isoformat(),
            }
        }
        
    except Exception as e:
        logger.error(f"批量计算失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    获取服务监控指标
    - 缓存命中率、请求统计等
    """
    try:
        metrics = {}
        
        if reward_service:
            metrics["service"] = reward_service.get_metrics()
        
        if task_queue:
            metrics["queue"] = task_queue.get_stats()
        
        if cache_backend and hasattr(cache_backend, 'get_stats'):
            if asyncio.iscoroutinefunction(cache_backend.get_stats):
                metrics["cache"] = await cache_backend.get_stats()
            else:
                metrics["cache"] = cache_backend.get_stats()
        
        return {
            "success": True,
            "metrics": metrics,
        }
        
    except Exception as e:
        logger.error(f"获取指标失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/cache/clear")
async def clear_cache():
    """
    清空缓存
    - 清空所有缓存数据
    """
    if cache_backend is None:
        raise HTTPException(status_code=503, detail="缓存未启用")
    
    try:
        await cache_backend.clear()
        
        # 重置服务指标
        if reward_service:
            reward_service.reset_metrics()
        
        return {
            "success": True,
            "message": "缓存已清空",
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"清空缓存失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "deep_research.reward.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )

