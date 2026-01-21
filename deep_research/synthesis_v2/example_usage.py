"""
使用示例：展示如何使用synthesis_v2框架
"""
import asyncio
from pathlib import Path
from omegaconf import OmegaConf

from .core.concurrency import ConcurrencyManager
from .core.data_structures import WorkflowInput
from .core.pipeline import SamplePipeline
from .core.batch_processor import BatchProcessor
from .workflow.native.skeleton_trajectory_workflow import SkeletonTrajectoryWorkflow
from .evaluate.openjudge_evaluator import OpenJudgeEvaluator
from .io.streaming_writer import StreamingWriter


async def example_single_sample():
    """示例1: 单条样本处理"""
    # 加载配置
    config_path = Path(__file__).parent / "config" / "synthesis_v2.yaml"
    config = OmegaConf.load(str(config_path))
    
    # 初始化并发管理器
    concurrency = ConcurrencyManager()
    await concurrency.initialize(max_workers=config.concurrency.max_workers)
    
    # 创建workflow和evaluator
    workflow = SkeletonTrajectoryWorkflow(config)
    evaluator = OpenJudgeEvaluator(config)
    
    await workflow.initialize()
    await evaluator.initialize()
    
    # 创建pipeline
    pipeline = SamplePipeline(
        workflow=workflow,
        evaluator=evaluator,
        concurrency_manager=concurrency
    )
    
    # 准备输入
    input_data = WorkflowInput(
        messages=[{"role": "user", "content": "帮我分析一下贵州茅台最近的财报数据"}],
        metadata={"query_id": "test_001"}
    )
    
    # 处理
    result = await pipeline.process(input_data)
    
    print(f"处理成功: {result.success}")
    if result.evaluation:
        print(f"评估得分: {result.evaluation.score}")
    
    # 清理
    await workflow.cleanup()
    await evaluator.cleanup()


async def example_batch_processing():
    """示例2: 批处理 + Best of N"""
    # 加载配置
    config_path = Path(__file__).parent / "config" / "synthesis_v2.yaml"
    config = OmegaConf.load(str(config_path))
    
    # 初始化并发管理器
    concurrency = ConcurrencyManager()
    await concurrency.initialize(max_workers=config.concurrency.max_workers)
    
    # 创建workflow和evaluator
    workflow = SkeletonTrajectoryWorkflow(config)
    evaluator = OpenJudgeEvaluator(config)
    
    await workflow.initialize()
    await evaluator.initialize()
    
    # 创建pipeline
    pipeline = SamplePipeline(workflow, evaluator, concurrency)
    
    # 创建writer
    writer = StreamingWriter(output_dir=config.writer.output_dir)
    
    # 创建batch processor（从配置读取，这里演示可以覆盖）
    batch_processor = BatchProcessor(
        pipeline=pipeline,
        batch_size=config.batch_processor.batch_size,
        best_of_n=3,  # 演示：覆盖配置，启用Best of N
        writer=writer
    )
    
    # 准备输入
    inputs = [
        WorkflowInput(
            messages=[{"role": "user", "content": f"查询{i}: 分析公司财报"}],
            metadata={"query_id": f"query_{i}"}
        )
        for i in range(20)
    ]
    
    # 批处理
    results = await batch_processor.process_all(inputs)
    
    print(f"处理完成: {len(results)} 个结果")
    
    # 保存汇总
    await writer.save_summary({
        "total_inputs": len(inputs),
        "total_results": len(results),
        "best_of_n": 3
    })
    
    # 清理
    await workflow.cleanup()
    await evaluator.cleanup()


async def example_query_generation():
    """示例3: Query生成workflow"""
    from .query_generation.query_workflow import QueryGeneratorWorkflow
    
    # 加载配置
    config_path = Path(__file__).parent / "config" / "synthesis_v2.yaml"
    config = OmegaConf.load(str(config_path))
    
    # Query生成不需要completion_callback
    
    # 创建query generator workflow
    query_workflow = QueryGeneratorWorkflow(config)
    await query_workflow.initialize()
    
    # 准备输入
    input_data = WorkflowInput(
        messages=[],  # query生成不需要输入messages
        metadata={"num_queries": 10}
    )
    
    # 执行
    result = await query_workflow.execute(input_data)
    
    if result.success:
        num_queries = result.metadata.get('num_generated', 0)
        print(f"生成了 {num_queries} 个queries")
        
        # 获取生成的queries
        queries = result.metadata.get('queries', [])
        for i, q in enumerate(queries[:3], 1):
            print(f"{i}. {q['content'][:100]}...")
    
    # 清理
    await query_workflow.cleanup()


if __name__ == "__main__":
    print("=" * 50)
    print("示例1: 单条样本处理")
    print("=" * 50)
    asyncio.run(example_single_sample())
    
    print("\n" + "=" * 50)
    print("示例2: 批处理 + Best of N")
    print("=" * 50)
    asyncio.run(example_batch_processing())
    
    print("\n" + "=" * 50)
    print("示例3: Query生成")
    print("=" * 50)
    asyncio.run(example_query_generation())
