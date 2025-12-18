"""
任务队列管理
支持异步任务提交、查询和结果存储
"""
import asyncio
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict

from .hash_utils import compute_reward_input_hash, TASK_HASH_LENGTH

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
    
    def __init__(self, cache_backend=None, max_queue_size: int = 1000, task_retention_seconds: int = 3600):
        """
        Args:
            cache_backend: 缓存后端（用于持久化任务状态）
            max_queue_size: 最大队列大小
            task_retention_seconds: 已完成任务在内存中保留的时间（秒），默认1小时
        """
        self.cache_backend = cache_backend
        self.max_queue_size = max_queue_size
        self.task_retention_seconds = task_retention_seconds
        
        # 内存队列（用于快速访问）
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.tasks = {}  # task_id -> Task
        
        # 后台任务处理器
        self.processor_tasks = []
        self.cleaner_task = None  # 内存清理任务
        self.is_running = False
        
        logger.info(
            f"TaskQueue 初始化: max_queue_size={max_queue_size}, "
            f"task_retention={task_retention_seconds}s"
        )
    
    @staticmethod
    def _compute_task_data_hash(
        prompt: str,
        response_list: list,
        messages_list: list,
        extra_list: list,
        config: dict
    ) -> str:
        """
        计算任务数据的hash值（用于任务去重和ID生成）
        
        Args:
            prompt: 问题文本
            response_list: 响应列表
            messages_list: 消息列表
            extra_list: 额外信息列表
            config: 配置参数
            
        Returns:
            16位hex hash字符串
            
        Note:
            使用统一的hash工具函数，确保与 RewardService 使用相同的hash算法。
        """
        return compute_reward_input_hash(
            prompt=prompt,
            response_list=response_list,
            messages_list=messages_list,
            extra_list=extra_list,
            config=config,
            length=TASK_HASH_LENGTH,
        )
    
    async def submit_task(
        self,
        prompt: str,
        response_list: list,
        messages_list: list,
        extra_list: list,
        config: Optional[dict] = None,
        task_id: Optional[str] = None,
    ) -> str:
        """
        提交任务
        
        Args:
            prompt, response_list, messages_list, extra_list: 计算奖励所需的输入
            config: 配置参数
            task_id: 可选的自定义任务ID前缀
            
        Returns:
            实际使用的任务ID (原task_id-hash)
        """
        if config is None:
            config = {}
        
        # 生成或使用提供的task_id作为前缀
        if task_id is None:
            task_id = str(uuid.uuid4())
        
        # 计算数据hash并生成最终task_id
        data_hash = self._compute_task_data_hash(prompt, response_list, messages_list, extra_list, config)
        final_task_id = f"{task_id}-{data_hash}"
        
        # 检查最终task_id是否已存在（相同数据的重复提交）
        if final_task_id in self.tasks:
            logger.info(f"任务已存在（相同数据）: {final_task_id}")
            return final_task_id
        
        # 创建任务
        now = datetime.now().isoformat()
        task = Task(
            task_id=final_task_id,
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
            prompt=prompt,
            response_list=response_list,
            messages_list=messages_list,
            extra_list=extra_list,
            config=config,
        )
        
        # 检查队列是否已满
        if self.queue.full():
            raise ValueError(f"任务队列已满（最大 {self.max_queue_size} 个任务）")
        
        # 保存任务状态
        self.tasks[final_task_id] = task
        await self._save_task(task)
        
        # 加入队列
        await self.queue.put(final_task_id)
        
        logger.info(f"任务已提交: {final_task_id}, 队列大小: {self.queue.qsize()}")
        return final_task_id
    
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
                # 只有正在进行的任务才重新加载到内存
                # 已完成/失败的任务直接返回，不占用内存
                if task.status in [TaskStatus.PENDING, TaskStatus.PROCESSING]:
                    self.tasks[task_id] = task
                    logger.debug(f"从缓存恢复进行中的任务到内存: {task_id}")
                else:
                    logger.debug(f"从缓存查询已完成任务（不加载到内存）: {task_id}")
        
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
    
    async def start_processors(self, num_processors: int, reward_service):
        """
        启动后台异步任务处理器
        
        注意：这里的"processor"是进程内的异步任务处理器，不是uvicorn的多进程worker
        
        Args:
            num_processors: 任务处理器数量
            reward_service: RewardComputeService实例
        """
        if self.is_running:
            logger.warning("任务处理器已在运行")
            return
        
        self.is_running = True
        logger.info(f"启动 {num_processors} 个异步任务处理器（async task processors）")
        
        for i in range(num_processors):
            processor = asyncio.create_task(
                self._task_processor(f"processor-{i}", reward_service)
            )
            self.processor_tasks.append(processor)
        
        # 启动内存清理任务
        logger.info(f"启动内存清理任务（每{self.task_retention_seconds}秒清理一次）")
        self.cleaner_task = asyncio.create_task(self._memory_cleaner())
    
    async def stop_processors(self):
        """停止后台异步任务处理器"""
        if not self.is_running:
            return
        
        logger.info("停止异步任务处理器...")
        self.is_running = False
        
        # 取消所有处理器任务
        for task in self.processor_tasks:
            task.cancel()
        
        # 取消内存清理任务
        if self.cleaner_task:
            self.cleaner_task.cancel()
        
        # 等待所有任务完成
        all_tasks = self.processor_tasks + ([self.cleaner_task] if self.cleaner_task else [])
        await asyncio.gather(*all_tasks, return_exceptions=True)
        self.processor_tasks.clear()
        self.cleaner_task = None
        
        logger.info("异步任务处理器已停止")
    
    async def _task_processor(self, processor_name: str, reward_service):
        """
        异步任务处理器
        从队列中取出任务并处理
        
        注意：这是进程内的异步任务处理协程，不是uvicorn的多进程worker
        """
        logger.info(f"{processor_name} 已启动")
        
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
                    logger.warning(f"{processor_name}: 任务不存在 {task_id}")
                    continue
                
                logger.info(f"{processor_name}: 开始处理任务 {task_id}")
                
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
                        f"{processor_name}: 任务完成 {task_id}, "
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
                        f"{processor_name}: 任务失败 {task_id}, "
                        f"错误: {error_msg}",
                        exc_info=True
                    )
                
            except asyncio.CancelledError:
                logger.info(f"{processor_name} 被取消")
                break
            except Exception as e:
                logger.error(f"{processor_name} 发生错误: {e}", exc_info=True)
        
        logger.info(f"{processor_name} 已停止")
    
    async def _memory_cleaner(self):
        """
        内存清理任务
        定期清理已完成/失败的任务，防止内存泄漏
        """
        logger.info("内存清理任务已启动")
        
        while self.is_running:
            try:
                # 每60秒检查一次
                await asyncio.sleep(60)
                
                if not self.is_running:
                    break
                
                from datetime import datetime, timedelta
                now = datetime.now()
                retention_delta = timedelta(seconds=self.task_retention_seconds)
                
                # 找到需要清理的任务
                tasks_to_remove = []
                for task_id, task in self.tasks.items():
                    # 只清理已完成或失败的任务
                    if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                        # 检查任务是否超过保留时间
                        updated_at = datetime.fromisoformat(task.updated_at)
                        if now - updated_at > retention_delta:
                            tasks_to_remove.append(task_id)
                
                # 清理任务
                if tasks_to_remove:
                    for task_id in tasks_to_remove:
                        del self.tasks[task_id]
                    logger.info(
                        f"内存清理: 移除 {len(tasks_to_remove)} 个已完成/失败的任务, "
                        f"当前内存任务数: {len(self.tasks)}"
                    )
                
            except asyncio.CancelledError:
                logger.info("内存清理任务被取消")
                break
            except Exception as e:
                logger.error(f"内存清理任务发生错误: {e}", exc_info=True)
        
        logger.info("内存清理任务已停止")
    
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

