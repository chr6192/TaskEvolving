"""
数据持久化模块
保存计算前后的样本数据，用于分析和调试
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SamplePersistence:
    """
    样本数据持久化
    保存计算前后的样本数据到本地文件
    """
    
    def __init__(self, base_dir: str = "./data/reward_samples", enabled: bool = True):
        """
        初始化持久化服务
        
        Args:
            base_dir: 数据保存根目录
            enabled: 是否启用持久化
        """
        self.base_dir = Path(base_dir)
        self.enabled = enabled
        
        if self.enabled:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"数据持久化已启用: {self.base_dir.absolute()}")
        else:
            logger.info("数据持久化已禁用")
    
    def _extract_unique_id(self, extra: Dict[str, Any]) -> str:
        """
        从extra中提取unique_id
        优先使用: unique_id > trajectory_id > skeleton_id
        """
        return (
            extra.get("unique_id") or 
            extra.get("trajectory_id") or 
            extra.get("skeleton_id") or 
            "unknown"
        )
    
    def _get_sample_file(self, unique_id: str) -> Path:
        """
        获取样本文件路径
        文件名包含unique_id
        """
        # 清理 unique_id 中的特殊字符
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in unique_id)
        filename = f"sample_{safe_id}.jsonl"
        return self.base_dir / filename
    
    async def save_sample_pair(
        self,
        prompt: str,
        response: str,
        messages: List[Dict[str, Any]],
        extra: Dict[str, Any],
        result: Dict[str, Any],
    ):
        """
        保存单个样本的计算前后数据
        
        Args:
            prompt: 问题文本
            response: 响应文本
            messages: 对话消息
            extra: 额外信息（包含unique_id）
            result: 计算结果（包含分数、奖励信息等）
        """
        if not self.enabled:
            return
        
        try:
            unique_id = self._extract_unique_id(extra)
            file_path = self._get_sample_file(unique_id)
            
            # 构建保存数据
            sample_data = {
                "timestamp": datetime.now().isoformat(),
                "unique_id": unique_id,
                "input": {
                    "prompt": prompt,
                    "response": response,
                    "messages": messages,
                    "extra": extra,
                },
                "output": {
                    "total_score": result.get("total_score"),
                    "reward_info": result.get("reward_info"),
                    "details": result.get("details"),
                }
            }
            
            # 追加写入文件
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(sample_data, ensure_ascii=False) + "\n")
            
            logger.debug(f"样本已保存: {unique_id} -> {file_path}")
            
        except Exception as e:
            logger.warning(f"保存样本数据失败: {e}", exc_info=True)
    
    async def save_batch_samples(
        self,
        prompt: str,
        response_list: List[str],
        messages_list: List[List[Dict[str, Any]]],
        extra_list: List[Dict[str, Any]],
        total_scores: List[float],
        reward_info: Dict[str, List[float]],
        details: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        批量保存一组样本的计算前后数据
        
        Args:
            prompt: 问题文本
            response_list: 响应列表
            messages_list: 消息列表
            extra_list: 额外信息列表
            total_scores: 总分列表
            reward_info: 奖励信息字典
            details: 详细信息列表
        """
        if not self.enabled:
            return
        
        try:
            num_samples = len(response_list)
            
            for i in range(num_samples):
                # 构建单个样本的结果
                sample_result = {
                    "total_score": total_scores[i] if i < len(total_scores) else None,
                    "reward_info": {
                        key: values[i] if i < len(values) else None
                        for key, values in reward_info.items()
                    },
                    "details": details[i] if details and i < len(details) else None,
                }
                
                # 保存单个样本
                await self.save_sample_pair(
                    prompt=prompt,
                    response=response_list[i],
                    messages=messages_list[i],
                    extra=extra_list[i],
                    result=sample_result,
                )
            
            logger.info(f"批量保存完成: {num_samples} 个样本")
            
        except Exception as e:
            logger.warning(f"批量保存样本数据失败: {e}", exc_info=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取持久化统计信息"""
        if not self.enabled:
            return {"enabled": False}
        
        try:
            files = list(self.base_dir.glob("sample_*.jsonl"))
            total_samples = 0
            
            for file in files:
                with open(file, "r", encoding="utf-8") as f:
                    total_samples += sum(1 for _ in f)
            
            return {
                "enabled": True,
                "base_dir": str(self.base_dir.absolute()),
                "total_files": len(files),
                "total_samples": total_samples,
            }
        except Exception as e:
            logger.warning(f"获取持久化统计失败: {e}")
            return {"enabled": True, "error": str(e)}

