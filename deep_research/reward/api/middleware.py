"""
中间件
包括日志、监控等
"""
import logging
import sys
import time
from datetime import datetime
from typing import Callable

from fastapi import Request, Response


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """
    配置日志系统
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径（可选）
    """
    # 创建日志格式
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s | "
        "%(filename)s:%(lineno)d | %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # 配置根日志器
    handlers = []
    
    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    handlers.append(console_handler)
    
    # 文件输出（如果指定）
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        handlers.append(file_handler)
    
    # 配置根日志器
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=handlers,
        force=True,
    )
    
    # 设置第三方库的日志级别
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)


async def request_logger(request: Request, call_next: Callable) -> Response:
    """
    请求日志中间件
    记录每个请求的详细信息和处理时间
    """
    # 生成请求ID
    request_id = request.headers.get("X-Request-ID", f"req_{int(time.time() * 1000)}")
    
    # 记录请求开始
    start_time = time.perf_counter()
    logger = logging.getLogger("api.request")
    
    # 记录请求信息
    logger.info(
        f"[{request_id}] → {request.method} {request.url.path} "
        f"| Client: {request.client.host if request.client else 'unknown'}"
    )
    
    # 处理请求
    try:
        response = await call_next(request)
        
        # 记录响应信息
        process_time = time.perf_counter() - start_time
        logger.info(
            f"[{request_id}] ← {response.status_code} "
            f"| Time: {process_time:.3f}s"
        )
        
        # 添加响应头
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.3f}"
        
        return response
        
    except Exception as e:
        # 记录错误
        process_time = time.perf_counter() - start_time
        logger.error(
            f"[{request_id}] ✗ Error: {str(e)} "
            f"| Time: {process_time:.3f}s",
            exc_info=True
        )
        raise


class MetricsCollector:
    """
    监控指标收集器
    收集API请求的各种指标
    """
    
    def __init__(self):
        self.total_requests = 0
        self.total_errors = 0
        self.request_times = []
        self.endpoint_stats = {}
        
        # 启动时间
        self.start_time = datetime.now()
    
    def record_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        process_time: float,
    ):
        """记录请求"""
        self.total_requests += 1
        
        if status_code >= 400:
            self.total_errors += 1
        
        self.request_times.append(process_time)
        
        # 保持最近1000个请求的时间
        if len(self.request_times) > 1000:
            self.request_times = self.request_times[-1000:]
        
        # 记录端点统计
        key = f"{method} {endpoint}"
        if key not in self.endpoint_stats:
            self.endpoint_stats[key] = {
                "count": 0,
                "errors": 0,
                "total_time": 0.0,
            }
        
        self.endpoint_stats[key]["count"] += 1
        self.endpoint_stats[key]["total_time"] += process_time
        
        if status_code >= 400:
            self.endpoint_stats[key]["errors"] += 1
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        # 计算平均响应时间
        avg_time = 0.0
        if self.request_times:
            avg_time = sum(self.request_times) / len(self.request_times)
        
        # 计算各端点的统计
        endpoint_details = {}
        for key, stats in self.endpoint_stats.items():
            endpoint_details[key] = {
                "count": stats["count"],
                "errors": stats["errors"],
                "avg_time": stats["total_time"] / stats["count"] if stats["count"] > 0 else 0.0,
                "error_rate": stats["errors"] / stats["count"] if stats["count"] > 0 else 0.0,
            }
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate": self.total_errors / self.total_requests if self.total_requests > 0 else 0.0,
            "avg_response_time": avg_time,
            "qps": self.total_requests / uptime if uptime > 0 else 0.0,
            "endpoints": endpoint_details,
        }


# 全局指标收集器
metrics_collector = MetricsCollector()

