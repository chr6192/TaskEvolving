"""
单例线程池 + 信号量并发控制
统一控制LLM API调用并发
"""
import asyncio
import logging
from typing import Coroutine, Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ConcurrencyManager:
    """单例并发控制管理器"""
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化会在第一次create实例时被调用"""
        pass
    
    async def initialize(self, max_workers: int = 32):
        """
        初始化并发控制器
        
        Args:
            max_workers: 最大并发数
        """
        async with self._lock:
            if self._initialized:
                logger.debug(f"ConcurrencyManager已初始化，max_workers={self._max_workers}")
                return
            
            self._max_workers = max_workers
            self._semaphore = asyncio.Semaphore(max_workers)
            self._initialized = True
            logger.info(f"ConcurrencyManager初始化完成，max_workers={max_workers}")
    
    async def submit(self, coro: Coroutine[Any, Any, T]) -> T:
        """
        提交协程任务，受信号量控制
        
        Args:
            coro: 要执行的协程
            
        Returns:
            协程执行结果
        """
        if not self._initialized:
            raise RuntimeError("ConcurrencyManager未初始化，请先调用initialize()")
        
        async with self._semaphore:
            return await coro
    
    @property
    def max_workers(self) -> int:
        """获取最大并发数"""
        if not self._initialized:
            return 0
        return self._max_workers
    
    def reset(self):
        """重置单例（主要用于测试）"""
        self._initialized = False
        self._semaphore = None
        self._max_workers = 0
