"""
使用示例：展示如何使用synthesis_v2框架
"""
import asyncio
import json
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf

from .core.concurrency import ConcurrencyManager
from .core.data_structures import WorkflowInput
from .core.pipeline import SamplePipeline
from .core.batch_processor import BatchProcessor
from .workflow.native.query_trajectory_workflow import QueryTrajectoryWorkflow
from .workflow.native.skeleton_trajectory_workflow import SkeletonTrajectoryWorkflow
from .evaluate.openjudge_evaluator import OpenJudgeEvaluator
from .io.streaming_writer import StreamingWriter
from .io.path_utils import resolve_under_repo
from .complexity_exploration.orchestrator import ComplexityExplorationV2
from .complexity_exploration.mermaid_reviser import MermaidReviserAgent
from deep_research.synthesis.core.data_structures import QueryData


async def example_single_sample():
    """示例1: 单条样本处理"""
    # 加载配置
    config_path = Path(__file__).parent / "config" / "synthesis_v2.yaml"
    config = OmegaConf.load(str(config_path))
    
    # 初始化并发管理器
    concurrency = ConcurrencyManager()
    await concurrency.initialize(max_workers=config.concurrency.max_workers)
    
    # 创建workflow和evaluator
    workflow = QueryTrajectoryWorkflow(config)
    # workflow = SkeletonTrajectoryWorkflow(config)
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

    # 保存单条结果（推荐）：复用 StreamingWriter，写入 outputs/synthesis_v2/pipeline_results.jsonl
    writer = StreamingWriter(output_dir=config.writer.output_dir)
    await writer.append_result(result)
    
    # 清理
    await workflow.cleanup()
    await evaluator.cleanup()


async def example_complexity_exploration(best_of_n: int = 1):
    """示例4: Complexity exploration 调用 pipeline

    Args:
        best_of_n: 每轮 iteration 内部 best-of-n 采样数；<=1 等价于不启用（使用配置默认）。
    """
    # 加载配置
    config_path = Path(__file__).parent / "config" / "synthesis_v2.yaml"
    config = OmegaConf.load(str(config_path))

    # 初始化并发管理器
    concurrency = ConcurrencyManager()
    await concurrency.initialize(max_workers=config.concurrency.max_workers)

    # 创建workflow和evaluator
    workflow = QueryTrajectoryWorkflow(config)
    evaluator = OpenJudgeEvaluator(config)

    await workflow.initialize()
    await evaluator.initialize()

    # 创建pipeline
    pipeline = SamplePipeline(
        workflow=workflow,
        evaluator=evaluator,
        concurrency_manager=concurrency,
    )

    # 创建reviser + exploration orchestrator
    reviser = MermaidReviserAgent(config)
    await reviser.initialize()
    explorer = ComplexityExplorationV2(pipeline=pipeline, reviser=reviser, config=config)

    # 准备seed query
    seed_query = QueryData(
        query_id="seed_001",
        content="帮我分析一下贵州茅台最近的财报数据，并给出关键指标趋势与风险点。",
        metadata={"query_id": "seed_001"},
    )

    # 执行 exploration（内部会逐轮调用 pipeline.process，并把 traj/iteration 落盘到 outputs/complexity_exploration/<evolution_id>/）
    # best_of_n<=1 时传 None，让 orchestrator 使用配置默认值（通常 best_of_n=1 表示不启用拒绝采样）。
    evolution = await explorer.explore(
        seed_query,
        best_of_n=None if best_of_n <= 1 else best_of_n,
    )

    if evolution is None:
        print("exploration失败：未返回evolution")
    else:
        print(f"exploration完成: converged={getattr(evolution, 'converged', None)}")
        print(f"total_iterations={getattr(evolution, 'total_iterations', None)}")
        if getattr(evolution, "final_query", None):
            print(f"final_query: {evolution.final_query.content}")
        if getattr(evolution, "convergence_reason", None):
            print(f"convergence_reason: {evolution.convergence_reason}")

    # 清理
    await reviser.cleanup()
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
    workflow = QueryTrajectoryWorkflow(config)
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

        # 保存queries到 outputs/synthesis_v2/queries.jsonl
        output_dir = resolve_under_repo(config.writer.output_dir, anchor_file=__file__)
        output_dir.mkdir(parents=True, exist_ok=True)
        queries_file = output_dir / "queries.jsonl"
        with queries_file.open("a", encoding="utf-8") as f:
            for q in queries:
                record = {
                    "content": q.get("content"),
                    "metadata": {
                        "generated_at": datetime.now().isoformat()
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    # 清理
    await query_workflow.cleanup()


if __name__ == "__main__":
    # print("=" * 50)
    # print("示例1: 单条样本处理")
    # print("=" * 50)
    # asyncio.run(example_single_sample())

    # 如果你想运行“每轮 best-of-n + 拒绝采样后再 revise”的版本，可以取消注释：
    print("\n" + "=" * 50)
    print("示例4b: 每轮 best-of-n + 只选1条再 revise")
    print("=" * 50)
    asyncio.run(example_complexity_exploration(best_of_n=3))
    
    # print("\n" + "=" * 50)
    # print("示例2: 批处理 + Best of N")
    # print("=" * 50)
    # asyncio.run(example_batch_processing())
    
    # print("\n" + "=" * 50)
    # print("示例3: Query生成")
    # print("=" * 50)
    # asyncio.run(example_query_generation())
