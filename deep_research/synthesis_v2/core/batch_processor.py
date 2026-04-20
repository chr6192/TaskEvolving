"""
批处理控制器
支持Best of N和拒绝采样
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from copy import deepcopy

from .data_structures import WorkflowInput, PipelineResult
from .pipeline import SamplePipeline
from ..io.streaming_writer import StreamingWriter

logger = logging.getLogger(__name__)


class BatchProcessor:
    """顶层批处理控制器"""
    
    def __init__(
        self,
        pipeline: SamplePipeline,
        batch_size: int = 10,
        best_of_n: int = 1,
        writer: Optional[StreamingWriter] = None
    ):
        """
        Args:
            pipeline: 单条样本pipeline
            batch_size: 批处理大小
            best_of_n: Best of N采样数（1表示不启用）
            writer: 流式写入器（可选）
        """
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.best_of_n = best_of_n
        self.writer = writer
        
        logger.info(
            f"BatchProcessor初始化: batch_size={batch_size}, "
            f"best_of_n={best_of_n}"
        )
    
    async def process_batch(
        self, 
        inputs: List[WorkflowInput]
    ) -> List[PipelineResult]:
        """
        批量处理，支持Best of N
        
        Args:
            inputs: 输入列表
            
        Returns:
            处理结果列表（如果启用Best of N，则返回选中的结果）
        """
        selected_results, _ = await self.process_batch_with_rejection(inputs)
        return selected_results

    async def process_batch_with_rejection(
        self,
        inputs: List[WorkflowInput],
    ) -> Tuple[List[PipelineResult], List[PipelineResult]]:
        """
        批量处理，支持Best of N，并返回被选中与被拒绝的结果。

        Returns:
            (selected_results, rejected_results)
        """
        if not inputs:
            return [], []

        logger.info(f"开始批处理: {len(inputs)} 个输入")

        # 1. 如果best_of_n > 1，复制每个输入n次
        expanded_inputs = self._expand_inputs_for_best_of_n(inputs)
        logger.info(f"扩展后输入数量: {len(expanded_inputs)}")

        # 2. 并发处理所有样本
        all_results = await asyncio.gather(
            *[self.pipeline.process(inp) for inp in expanded_inputs],
            return_exceptions=True
        )

        # 3. 过滤异常结果
        valid_results = []
        for result in all_results:
            if isinstance(result, Exception):
                logger.error(f"Pipeline处理异常: {result}")
            else:
                valid_results.append(result)

        logger.info(f"成功处理 {len(valid_results)}/{len(all_results)} 个样本")

        # 4. 追加写入所有结果
        if self.writer:
            await self.writer.append_results(valid_results)

        # 5. 拒绝采样：如果启用Best of N
        if self.best_of_n > 1:
            selected_results, rejected_results = await self._apply_rejection_sampling(
                valid_results,
                inputs,
            )
            logger.info(
                f"拒绝采样选出 {len(selected_results)} 个最佳样本，拒绝 {len(rejected_results)} 个候选样本"
            )
            return selected_results, rejected_results

        return valid_results, []
    
    def _expand_inputs_for_best_of_n(
        self, 
        inputs: List[WorkflowInput]
    ) -> List[WorkflowInput]:
        """
        复制输入n次用于Best of N
        
        Args:
            inputs: 原始输入列表
            
        Returns:
            扩展后的输入列表
        """
        if self.best_of_n <= 1:
            return inputs
        
        expanded = []
        for inp in inputs:
            for i in range(self.best_of_n):
                cloned = deepcopy(inp)
                cloned.metadata['sample_index'] = i
                cloned.metadata['original_input_id'] = inp.input_id
                expanded.append(cloned)
        
        return expanded
    
    async def _apply_rejection_sampling(
        self, 
        all_results: List[PipelineResult], 
        original_inputs: List[WorkflowInput]
    ) -> Tuple[List[PipelineResult], List[PipelineResult]]:
        """
        拒绝采样：按原始输入分组，选择evaluate得分最高的
        
        Args:
            all_results: 所有pipeline结果
            original_inputs: 原始输入列表
            
        Returns:
            选中的最佳结果列表
        """
        # 按original_input_id分组
        groups = defaultdict(list)
        for result in all_results:
            orig_id = result.input.metadata.get('original_input_id')
            if orig_id:
                groups[orig_id].append(result)
            else:
                # 如果没有original_input_id，说明不是Best of N，直接加入
                groups[result.input.input_id].append(result)
        
        logger.info(f"拒绝采样分组: {len(groups)} 组")
        
        # 每组选最佳
        selected: List[PipelineResult] = []
        rejected: List[PipelineResult] = []
        for orig_id, group_results in groups.items():
            # 选择evaluation.score最高的
            best = max(
                group_results, 
                key=lambda r: r.evaluation.score if r.evaluation else 0.0
            )
            
            # 标记选择状态
            best.input.metadata['selection_status'] = 'selected'
            for r in group_results:
                if r != best:
                    r.input.metadata['selection_status'] = 'rejected'
                    rejected.append(r)
            
            selected.append(best)
            
            # 写入选中和拒绝的结果
            if self.writer:
                await self.writer.append_selected_result(best)
                for r in group_results:
                    if r != best:
                        await self.writer.append_rejected_result(r)
        
        return selected, rejected
    
    async def process_all(
        self, 
        all_inputs: List[WorkflowInput]
    ) -> List[PipelineResult]:
        """
        分批处理所有输入
        
        Args:
            all_inputs: 所有输入
            
        Returns:
            所有结果
        """
        all_results = []
        
        # 分批处理
        for i in range(0, len(all_inputs), self.batch_size):
            batch_inputs = all_inputs[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(all_inputs) + self.batch_size - 1) // self.batch_size
            
            logger.info(f"处理批次 {batch_num}/{total_batches}")
            
            batch_results = await self.process_batch(batch_inputs)
            all_results.extend(batch_results)
        
        logger.info(f"所有批次处理完成，共 {len(all_results)} 个结果")
        
        return all_results
