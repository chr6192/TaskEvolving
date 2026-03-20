"""
Pipeline粒度追加写入
每个pipeline完成后即写入，支持断点续传
"""
import asyncio
import json
import logging
from pathlib import Path
from typing import List
from datetime import datetime

from ..core.data_structures import PipelineResult
from .path_utils import resolve_under_repo

logger = logging.getLogger(__name__)


class StreamingWriter:
    """Pipeline粒度的流式数据写入器"""
    
    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: 输出目录路径
        """
        # 统一把相对路径解析到 repo root 下，避免从 OpenJudge/ 等子目录启动导致输出落盘位置漂移
        self.output_dir = resolve_under_repo(output_dir, anchor_file=__file__)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 输出文件路径
        self.results_file = self.output_dir / "pipeline_results.jsonl"
        self.selected_file = self.output_dir / "selected_results.jsonl"
        self.rejected_file = self.output_dir / "rejected_results.jsonl"
        
        # 文件锁（避免并发写入冲突）
        self._write_lock = asyncio.Lock()
        
        logger.info(f"StreamingWriter初始化，输出目录: {self.output_dir}")
    
    async def append_result(self, result: PipelineResult) -> None:
        """
        追加写入单个pipeline结果
        
        Args:
            result: pipeline结果
        """
        async with self._write_lock:
            # 写入所有结果
            await self._append_to_file(self.results_file, result.to_dict())
            logger.debug(f"追加写入结果到 {self.results_file}")
    
    async def append_results(self, results: List[PipelineResult]) -> None:
        """
        批量追加写入pipeline结果
        
        Args:
            results: pipeline结果列表
        """
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"跳过异常结果: {result}")
                continue
            await self.append_result(result)
    
    async def append_selected_result(self, result: PipelineResult) -> None:
        """追加写入被选中的结果（用于Best of N）"""
        async with self._write_lock:
            await self._append_to_file(self.selected_file, result.to_dict())
            logger.debug(f"追加写入选中结果到 {self.selected_file}")
    
    async def append_rejected_result(self, result: PipelineResult) -> None:
        """追加写入被拒绝的结果（用于Best of N）"""
        async with self._write_lock:
            await self._append_to_file(self.rejected_file, result.to_dict())
            logger.debug(f"追加写入拒绝结果到 {self.rejected_file}")
    
    async def _append_to_file(self, file_path: Path, data: dict) -> None:
        """
        追加写入JSON行到文件
        
        Args:
            file_path: 文件路径
            data: 要写入的数据
        """
        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"写入文件失败 {file_path}: {e}")
    
    async def save_summary(self, summary: dict) -> None:
        """
        保存汇总信息
        
        Args:
            summary: 汇总信息
        """
        summary_file = self.output_dir / "summary.json"
        async with self._write_lock:
            try:
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                logger.info(f"保存汇总信息到 {summary_file}")
            except Exception as e:
                logger.error(f"保存汇总信息失败: {e}")
