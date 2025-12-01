"""
缓存后端实现
支持Redis和内存缓存两种方式
"""
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CacheBackend(ABC):
    """缓存后端抽象基类"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[dict]:
        """获取缓存"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: dict, ttl: int = 3600):
        """设置缓存"""
        pass
    
    @abstractmethod
    async def delete(self, key: str):
        """删除缓存"""
        pass
    
    @abstractmethod
    async def clear(self):
        """清空所有缓存"""
        pass


class MemoryCache(CacheBackend):
    """内存缓存实现（适合单机部署）"""
    
    def __init__(self, max_size: int = 10000):
        """
        Args:
            max_size: 最大缓存条目数
        """
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}  # 记录访问时间，用于LRU
        self.expire_times = {}  # 记录过期时间
        logger.info(f"MemoryCache 初始化: max_size={max_size}")
    
    async def get(self, key: str) -> Optional[dict]:
        """获取缓存"""
        # 检查是否存在
        if key not in self.cache:
            return None
        
        # 检查是否过期
        if key in self.expire_times:
            if time.time() > self.expire_times[key]:
                # 已过期，删除
                await self.delete(key)
                return None
        
        # 更新访问时间
        self.access_times[key] = time.time()
        return self.cache[key]
    
    async def set(self, key: str, value: dict, ttl: int = 3600):
        """设置缓存"""
        # 如果缓存已满，删除最久未访问的项
        if len(self.cache) >= self.max_size and key not in self.cache:
            await self._evict_lru()
        
        self.cache[key] = value
        self.access_times[key] = time.time()
        self.expire_times[key] = time.time() + ttl
    
    async def delete(self, key: str):
        """删除缓存"""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.expire_times.pop(key, None)
    
    async def clear(self):
        """清空所有缓存"""
        self.cache.clear()
        self.access_times.clear()
        self.expire_times.clear()
        logger.info("内存缓存已清空")
    
    async def _evict_lru(self):
        """删除最久未访问的项（LRU策略）"""
        if not self.access_times:
            return
        
        # 找到最久未访问的key
        lru_key = min(self.access_times, key=self.access_times.get)
        await self.delete(lru_key)
        logger.debug(f"LRU淘汰: {lru_key[:16]}...")
    
    def get_stats(self) -> dict:
        """获取缓存统计信息"""
        return {
            "total_items": len(self.cache),
            "max_size": self.max_size,
            "usage_rate": len(self.cache) / self.max_size if self.max_size > 0 else 0,
        }


class RedisCache(CacheBackend):
    """Redis缓存实现（适合分布式部署，但当前不需要）"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """
        Args:
            redis_url: Redis连接URL
        """
        self.redis_url = redis_url
        self.redis = None
        logger.info(f"RedisCache 初始化: {redis_url}")
    
    async def connect(self):
        """连接到Redis"""
        try:
            import redis.asyncio as aioredis
            self.redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False  # 我们会手动处理JSON序列化
            )
            await self.redis.ping()
            logger.info("Redis连接成功")
        except ImportError:
            logger.error("redis包未安装，请执行: pip install redis")
            raise
        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
            raise
    
    async def disconnect(self):
        """断开Redis连接"""
        if self.redis:
            await self.redis.close()
            logger.info("Redis连接已关闭")
    
    async def get(self, key: str) -> Optional[dict]:
        """获取缓存"""
        if not self.redis:
            await self.connect()
        
        try:
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.warning(f"从Redis获取数据失败: {e}")
            return None
    
    async def set(self, key: str, value: dict, ttl: int = 3600):
        """设置缓存"""
        if not self.redis:
            await self.connect()
        
        try:
            value_str = json.dumps(value, ensure_ascii=False)
            await self.redis.setex(key, ttl, value_str)
        except Exception as e:
            logger.warning(f"保存到Redis失败: {e}")
    
    async def delete(self, key: str):
        """删除缓存"""
        if not self.redis:
            await self.connect()
        
        try:
            await self.redis.delete(key)
        except Exception as e:
            logger.warning(f"从Redis删除数据失败: {e}")
    
    async def clear(self):
        """清空所有缓存（危险操作）"""
        if not self.redis:
            await self.connect()
        
        logger.warning("清空Redis缓存（仅删除reward相关的key）")
        try:
            # 只删除以 reward: 开头的key
            cursor = 0
            pattern = "reward:*"
            while True:
                cursor, keys = await self.redis.scan(cursor, match=pattern, count=100)
                if keys:
                    await self.redis.delete(*keys)
                if cursor == 0:
                    break
            logger.info("Redis缓存已清空")
        except Exception as e:
            logger.error(f"清空Redis缓存失败: {e}")
    
    async def get_stats(self) -> dict:
        """获取Redis统计信息"""
        if not self.redis:
            await self.connect()
        
        try:
            info = await self.redis.info("memory")
            return {
                "used_memory": info.get("used_memory_human", "unknown"),
                "used_memory_peak": info.get("used_memory_peak_human", "unknown"),
            }
        except Exception as e:
            logger.warning(f"获取Redis统计信息失败: {e}")
            return {}


async def create_cache_backend(cache_type: str = "memory", **kwargs) -> CacheBackend:
    """
    创建缓存后端
    
    Args:
        cache_type: 缓存类型 ("memory" 或 "redis")
        **kwargs: 缓存后端的初始化参数
            - max_size: MemoryCache 的最大缓存条目数
            - redis_url: RedisCache 的连接URL
    
    Returns:
        缓存后端实例
    """
    if cache_type == "memory":
        # 只传递 MemoryCache 相关的参数
        memory_kwargs = {}
        if "max_size" in kwargs:
            memory_kwargs["max_size"] = kwargs["max_size"]
        return MemoryCache(**memory_kwargs)
    elif cache_type == "redis":
        # 只传递 RedisCache 相关的参数
        redis_kwargs = {}
        if "redis_url" in kwargs:
            redis_kwargs["redis_url"] = kwargs["redis_url"]
        cache = RedisCache(**redis_kwargs)
        await cache.connect()
        return cache
    else:
        raise ValueError(f"不支持的缓存类型: {cache_type}")

