"""
统一的Hash计算工具
确保task_queue、reward_service等模块使用一致的hash算法
"""
import hashlib
import json
from typing import Any, Dict, List, Optional


# 常用的hash长度常量
TASK_HASH_LENGTH = 16  # 用于task_id（2^64空间，约43亿任务后50%碰撞概率）


def compute_reward_input_hash(
    prompt: str,
    response_list: List[str],
    messages_list: List[List[dict]],
    extra_list: List[dict],
    config: Dict[str, Any],
    length: Optional[int] = None,
    prefix: str = "",
) -> str:
    """
    计算奖励输入数据的统一hash
    
    这是系统中唯一的hash计算函数，所有场景都使用它：
    - TaskQueue任务去重：length=16
    - RewardService缓存：length=None, prefix="reward:compute"
    - 其他需要识别相同输入的场景
    
    Args:
        prompt: 问题文本
        response_list: 响应列表
        messages_list: 消息列表（每个元素是一个消息列表）
        extra_list: 额外信息列表
        config: 配置参数
        length: hash长度（None=完整64位，16=任务ID，可自定义）
        prefix: 可选前缀（如 "reward:compute"），添加在hash前，用于命名空间隔离
        
    Returns:
        Hash字符串（可选前缀和截断）
        
    Examples:
        >>> # TaskQueue: 16位hash用于任务去重
        >>> task_hash = compute_reward_input_hash(
        ...     prompt="问题",
        ...     response_list=["回答"],
        ...     messages_list=[[{"role": "user", "content": "问题"}]],
        ...     extra_list=[{"id": "1"}],
        ...     config={},
        ...     length=16
        ... )
        >>> # 返回: 'ce21b30f493f126f'
        
        >>> # RewardService: 完整hash + 前缀用于缓存key
        >>> cache_key = compute_reward_input_hash(
        ...     prompt="问题",
        ...     response_list=["回答"],
        ...     messages_list=[[{"role": "user", "content": "问题"}]],
        ...     extra_list=[{"id": "1"}],
        ...     config={},
        ...     length=None,
        ...     prefix="reward:compute"
        ... )
        >>> # 返回: 'reward:compute:ce21b30f493f126f006cd0e9...'
    """
    # 使用标准化的字段名构建hash数据
    # 注意：字段名和顺序的任何变化都会导致hash值变化
    data = {
        "prompt": prompt,
        "response_list": response_list,
        "messages_list": messages_list,
        "extra_list": extra_list,
        "config": config,
    }
    
    # 序列化为JSON字符串（排序key确保一致性）
    data_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
    
    # 计算SHA256 hash
    full_hash = hashlib.sha256(data_str.encode()).hexdigest()
    
    # 截断hash（如果需要）
    if length is not None:
        full_hash = full_hash[:length]
    
    # 添加前缀（如果需要）
    if prefix:
        return f"{prefix}:{full_hash}"
    return full_hash

