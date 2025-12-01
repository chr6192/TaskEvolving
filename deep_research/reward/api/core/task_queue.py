"""
任务队列管理
支持异步任务提交、查询和结果存储
"""
import asyncio
import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"      # 等待处理
    PROCESSING = "processing"  # 处理中
    COMPLETED = "completed"   # 已完成
    FAILED = "failed"         # 失败


@dataclass
class Task:
    """任务数据结构"""
    task_id: str
    status: TaskStatus
    created_at: str
    updated_at: str
    
    # 输入数据
    prompt: str
    response_list: list
    messages_list: list
    extra_list: list
    config: dict
    
    # 输出结果
    result: Optional[dict] = None
    error: Optional[str] = None
    
    # 元信息
    progress: float = 0.0
    processing_time: Optional[float] = None
    
    def to_dict(self) -> dict:
        """转换为字典"""
        data = asdict(self)
        data["status"] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        """从字典创建"""
        data["status"] = TaskStatus(data["status"])
        return cls(**data)


class TaskQueue:
    """
    任务队列管理器
    - 使用内存队列 + 可选Redis持久化
    - 支持任务提交、查询、结果存储
    """
    
    def __init__(self, cache_backend=None, max_queue_size: int = 1000):
        """
        Args:
            cache_backend: 缓存后端（用于持久化任务状态）
            max_queue_size: 最大队列大小
        """
        self.cache_backend = cache_backend
        self.max_queue_size = max_queue_size
        
        # 内存队列（用于快速访问）
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.tasks = {}  # task_id -> Task
        
        # 后台工作任务
        self.worker_tasks = []
        self.is_running = False
        
        logger.info(f"TaskQueue 初始化: max_queue_size={max_queue_size}")
    
    async def submit_task(
        self,
        prompt: str,
        responses: list,
        messages: list,
        extras: list,
        config: Optional[dict] = None,
        task_id: Optional[str] = None,
    ) -> str:
        """
        提交任务
        
        Args:
            prompt, responses, messages, extras: 计算奖励所需的输入
            config: 配置参数
            task_id: 可选的自定义任务ID（用于幂等性）
            
        Returns:
            任务ID
        """
        if config is None:
            config = {}
        
        # 生成或使用提供的task_id
        if task_id is None:
            task_id = str(uuid.uuid4())
        
        # 检查任务是否已存在（幂等性）
        if task_id in self.tasks:
            logger.info(f"任务已存在: {task_id}")
            return task_id
        
        # 创建任务
        now = datetime.now().isoformat()
        task = Task(
            task_id=task_id,
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
            prompt=prompt,
            response_list=responses,
            messages_list=messages,
            extra_list=extras,
            config=config,
        )
        
        # 检查队列是否已满
        if self.queue.full():
            raise ValueError(f"任务队列已满（最大 {self.max_queue_size} 个任务）")
        
        # 保存任务状态
        self.tasks[task_id] = task
        await self._save_task(task)
        
        # 加入队列
        await self.queue.put(task_id)
        
        logger.info(f"任务已提交: {task_id}, 队列大小: {self.queue.qsize()}")
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[dict]:
        """
        获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务状态字典，如果任务不存在返回None
        """
        # 先从内存查找
        task = self.tasks.get(task_id)
        
        # 如果内存中没有，尝试从缓存加载
        if task is None and self.cache_backend:
            task = await self._load_task(task_id)
            if task:
                self.tasks[task_id] = task
        
        if task is None:
            return None
        
        return task.to_dict()
    
    async def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        result: Optional[dict] = None,
        error: Optional[str] = None,
        progress: Optional[float] = None,
        processing_time: Optional[float] = None,
    ):
        """更新任务状态"""
        task = self.tasks.get(task_id)
        if task is None:
            logger.warning(f"任务不存在: {task_id}")
            return
        
        task.status = status
        task.updated_at = datetime.now().isoformat()
        
        if result is not None:
            task.result = result
        if error is not None:
            task.error = error
        if progress is not None:
            task.progress = progress
        if processing_time is not None:
            task.processing_time = processing_time
        
        # 持久化
        await self._save_task(task)
        
        logger.debug(f"任务状态已更新: {task_id} -> {status.value}")
    
    async def _save_task(self, task: Task):
        """保存任务到缓存"""
        if not self.cache_backend:
            return
        
        try:
            key = f"task:{task.task_id}"
            await self.cache_backend.set(
                key,
                task.to_dict(),
                ttl=86400  # 保存24小时
            )
        except Exception as e:
            logger.warning(f"保存任务失败: {e}")
    
    async def _load_task(self, task_id: str) -> Optional[Task]:
        """从缓存加载任务"""
        if not self.cache_backend:
            return None
        
        try:
            key = f"task:{task_id}"
            task_dict = await self.cache_backend.get(key)
            if task_dict:
                return Task.from_dict(task_dict)
        except Exception as e:
            logger.warning(f"加载任务失败: {e}")
        
        return None
    
    async def start_workers(self, num_workers: int, reward_service):
        """
        启动后台工作进程
        
        Args:
            num_workers: 工作进程数量
            reward_service: RewardComputeService实例
        """
        if self.is_running:
            logger.warning("工作进程已在运行")
            return
        
        self.is_running = True
        logger.info(f"启动 {num_workers} 个后台工作进程")
        
        for i in range(num_workers):
            worker = asyncio.create_task(
                self._worker(f"worker-{i}", reward_service)
            )
            self.worker_tasks.append(worker)
    
    async def stop_workers(self):
        """停止后台工作进程"""
        if not self.is_running:
            return
        
        logger.info("停止后台工作进程...")
        self.is_running = False
        
        # 取消所有工作任务
        for task in self.worker_tasks:
            task.cancel()
        
        # 等待所有任务完成
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.worker_tasks.clear()
        
        logger.info("后台工作进程已停止")
    
    async def _worker(self, worker_name: str, reward_service):
        """
        后台工作进程
        从队列中取出任务并处理
        """
        logger.info(f"{worker_name} 已启动")
        
        while self.is_running:
            try:
                # 从队列获取任务（超时避免阻塞）
                try:
                    task_id = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                task = self.tasks.get(task_id)
                if task is None:
                    logger.warning(f"{worker_name}: 任务不存在 {task_id}")
                    continue
                
                logger.info(f"{worker_name}: 开始处理任务 {task_id}")
                
                # 更新状态为处理中
                await self.update_task_status(
                    task_id,
                    TaskStatus.PROCESSING,
                    progress=0.1
                )
                
                # 执行计算
                import time
                start_time = time.perf_counter()
                
                try:
                    result = await reward_service.compute_single_group(
                        prompt=task.prompt,
                        response_list=task.response_list,
                        messages_list=task.messages_list,
                        extra_list=task.extra_list,
                        config=task.config,
                    )
                    
                    processing_time = time.perf_counter() - start_time
                    
                    # 更新状态为完成
                    await self.update_task_status(
                        task_id,
                        TaskStatus.COMPLETED,
                        result=result,
                        progress=1.0,
                        processing_time=processing_time,
                    )
                    
                    logger.info(
                        f"{worker_name}: 任务完成 {task_id}, "
                        f"耗时 {processing_time:.2f}s"
                    )
                    
                except Exception as e:
                    processing_time = time.perf_counter() - start_time
                    error_msg = str(e)
                    
                    # 更新状态为失败
                    await self.update_task_status(
                        task_id,
                        TaskStatus.FAILED,
                        error=error_msg,
                        progress=0.0,
                        processing_time=processing_time,
                    )
                    
                    logger.error(
                        f"{worker_name}: 任务失败 {task_id}, "
                        f"错误: {error_msg}",
                        exc_info=True
                    )
                
            except asyncio.CancelledError:
                logger.info(f"{worker_name} 被取消")
                break
            except Exception as e:
                logger.error(f"{worker_name} 发生错误: {e}", exc_info=True)
        
        logger.info(f"{worker_name} 已停止")
    
    def get_stats(self) -> dict:
        """获取队列统计信息"""
        status_counts = {status.value: 0 for status in TaskStatus}
        for task in self.tasks.values():
            status_counts[task.status.value] += 1
        
        return {
            "queue_size": self.queue.qsize(),
            "total_tasks": len(self.tasks),
            "status_counts": status_counts,
            "max_queue_size": self.max_queue_size,
        }

