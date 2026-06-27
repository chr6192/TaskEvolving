"""
读取 seeds jsonl，并对每条 QueryData 执行 ComplexityExplorationV2.explore。

用法示例：
  python deep_research/synthesis_v2/experiments/run_complexity_exploration_on_seed_file.py \
    --language en \
    --best-of-n 3

  # 最多 3 条 seed 并行探索（默认）
  python deep_research/synthesis_v2/experiments/run_complexity_exploration_on_seed_file.py \
    --max-concurrent-seeds 3 \
    --limit 20

  # 显式指定 seed 文件（覆盖语言默认数据集）
  python deep_research/synthesis_v2/experiments/run_complexity_exploration_on_seed_file.py \
    --input deep_research/synthesis_v2/data/nq_ch.jsonl \
    --language cn

  # 断点续跑：对已因 max_iterations_reached 触顶的 case，在调高 max_iterations 后继续
  python deep_research/synthesis_v2/experiments/run_complexity_exploration_on_seed_file.py \
    --resume \
    --max-concurrent-seeds 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

from omegaconf import OmegaConf


def _prefer_local_repo_packages() -> None:
    """
    避免误用 site-packages 里的同名包（尤其是 `verl`）。
    这里把仓库根目录与 ./verl 放到 sys.path 最前面。
    """
    repo_root = Path(__file__).resolve().parents[3]
    verl_dir = repo_root / "verl"
    sys.path.insert(0, str(repo_root))
    if verl_dir.exists():
        sys.path.insert(0, str(verl_dir))


_prefer_local_repo_packages()

from deep_research.synthesis.core.data_structures import QueryData
from deep_research.synthesis_v2.complexity_exploration.mermaid_reviser import MermaidReviserAgent
from deep_research.synthesis_v2.complexity_exploration.orchestrator import ComplexityExplorationV2
from deep_research.synthesis_v2.core.concurrency import ConcurrencyManager
from deep_research.synthesis_v2.core.language import apply_language_to_config, resolve_seed_file
from deep_research.synthesis_v2.core.pipeline import SamplePipeline
from deep_research.synthesis_v2.evaluate.openjudge_evaluator import OpenJudgeEvaluator
from deep_research.synthesis_v2.io.path_utils import resolve_under_repo
from deep_research.synthesis_v2.workflow.native.query_trajectory_workflow import QueryTrajectoryWorkflow


def _resolve_repo_path(p: Path) -> Path:
    """
    将相对路径稳定解析到 repo root，避免从不同 cwd 启动导致 progress/input 指向不同文件。
    """
    if p.is_absolute():
        return p
    repo_root = Path(__file__).resolve().parents[3]
    return (repo_root / p).resolve()


_DIRNAME_SAFE_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _safe_dirname(name: str, max_len: int = 120) -> str:
    name = (name or "").strip()
    if not name:
        return "empty_query_id"
    cleaned = _DIRNAME_SAFE_RE.sub("_", name)
    cleaned = cleaned.strip("._-")
    if not cleaned:
        cleaned = "query"
    return cleaned[:max_len]


def _already_ran(output_root: Path, query_id: str) -> bool:
    """
    以输出目录是否存在来判断是否已跑过（支持 query_id 与 query_id__N）。
    """
    return ComplexityExplorationV2.find_evolution_dir(output_root, query_id) is not None


def _resolve_seed_job(
    output_root: Path,
    query_id: str,
    *,
    resume: bool,
) -> Tuple[str, Optional[Path]]:
    """
    返回 (action, resume_dir)。
    action: run | resume | skip
    """
    evolution_dir = ComplexityExplorationV2.find_evolution_dir(output_root, query_id)
    if evolution_dir is None:
        return ("run", None) if not resume else ("skip", None)
    if not resume:
        return "skip", None
    info = ComplexityExplorationV2.inspect_resume(evolution_dir)
    if info.get("status") == "resumable":
        return "resume", evolution_dir
    return "skip", None


def _iter_seed_records(path: Path) -> Iterable[QueryData]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            try:
                yield QueryData.model_validate(obj)
            except Exception as e:
                raise ValueError(f"Invalid QueryData at {path}:{line_no}: {e}") from e


async def _explore_seed(
    *,
    job_index: int,
    seed_query: QueryData,
    explorer: ComplexityExplorationV2,
    best_of_n: int,
    resume_dir: Optional[Path],
    semaphore: asyncio.Semaphore,
    stats_lock: asyncio.Lock,
    counters: dict[str, int],
) -> None:
    async with semaphore:
        mode = "resume" if resume_dir is not None else "run"
        print(f"\n===== [{job_index}] {mode} query_id={seed_query.query_id} =====")
        try:
            evolution = await explorer.explore(
                seed_query,
                best_of_n=None if best_of_n <= 1 else best_of_n,
                resume_dir=resume_dir,
            )
            async with stats_lock:
                if evolution is None:
                    counters["fail"] += 1
                    print(f"[{job_index}] FAILED: evolution=None")
                else:
                    key = "resumed" if resume_dir is not None else "ok"
                    counters[key] += 1
                    converged = getattr(evolution, "converged", None)
                    iters = getattr(evolution, "total_iterations", None)
                    reason = getattr(evolution, "convergence_reason", None)
                    print(
                        f"[{job_index}] OK: converged={converged} "
                        f"total_iterations={iters} reason={reason}"
                    )
        except Exception as e:
            async with stats_lock:
                counters["fail"] += 1
                print(f"[{job_index}] EXCEPTION: {type(e).__name__}: {e}")


async def run(
    seed_file: Optional[Path],
    best_of_n: int,
    limit: Optional[int],
    start: int,
    api_key: Optional[str],
    language: str,
    max_concurrent_seeds: int,
    resume: bool,
) -> None:
    config_path = Path(__file__).resolve().parents[1] / "config" / "synthesis_v2.yaml"
    config = OmegaConf.load(str(config_path))
    language = apply_language_to_config(config, language)

    seed_path = resolve_seed_file(language, str(seed_file) if seed_file else None)
    seed_file = _resolve_repo_path(Path(seed_path))
    if not seed_file.is_file():
        raise FileNotFoundError(f"Seed file not found: {seed_file}")

    print(f"Using language={language} pipeline")
    print(f"Using seed_file={seed_file}")

    # 允许通过 CLI 传入 key（优先级最高），否则依赖环境变量 OPENAI_API_KEY 或配置文件内的 key
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        try:
            config.openai_api.api_key = api_key
        except Exception:
            pass

    concurrency = ConcurrencyManager()
    await concurrency.initialize(max_workers=config.concurrency.max_workers)

    workflow = QueryTrajectoryWorkflow(config)
    evaluator = OpenJudgeEvaluator(config)
    await workflow.initialize()
    await evaluator.initialize()

    pipeline = SamplePipeline(
        workflow=workflow,
        evaluator=evaluator,
        concurrency_manager=concurrency,
    )

    reviser = MermaidReviserAgent(config)
    await reviser.initialize()
    explorer = ComplexityExplorationV2(pipeline=pipeline, reviser=reviser, config=config)

    output_root = resolve_under_repo(
        config.get("complexity_exploration", {}).get("output_dir", "outputs/complexity_exploration"),
        anchor_file=__file__,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Using output_root={output_root}")
    print(f"Using max_concurrent_seeds={max_concurrent_seeds}")
    print(f"Using resume={resume}")

    pending: list[tuple[int, QueryData, Optional[Path]]] = []
    skipped_prefilter = 0
    skipped_not_resumable = 0
    total_seen = 0
    for idx, seed_query in enumerate(_iter_seed_records(seed_file), start=0):
        if idx < start:
            continue
        total_seen += 1
        if limit is not None and total_seen > limit:
            break
        action, resume_dir = _resolve_seed_job(
            output_root,
            seed_query.query_id,
            resume=resume,
        )
        if action == "skip":
            skipped_prefilter += 1
            if skipped_prefilter % 50 == 1:
                print(f"[skip] existing: {seed_query.query_id} (skipped={skipped_prefilter})")
            continue
        if action == "resume" and resume_dir is None:
            skipped_not_resumable += 1
            continue
        pending.append((total_seen, seed_query, resume_dir))

    counters = {
        "ok": 0,
        "resumed": 0,
        "fail": 0,
        "skipped": skipped_prefilter,
        "skipped_not_resumable": skipped_not_resumable,
    }
    semaphore = asyncio.Semaphore(max(1, max_concurrent_seeds))
    stats_lock = asyncio.Lock()

    try:
        if pending:
            await asyncio.gather(
                *[
                    _explore_seed(
                        job_index=job_index,
                        seed_query=seed_query,
                        explorer=explorer,
                        best_of_n=best_of_n,
                        resume_dir=resume_dir,
                        semaphore=semaphore,
                        stats_lock=stats_lock,
                        counters=counters,
                    )
                    for job_index, seed_query, resume_dir in pending
                ]
            )
    finally:
        print(
            f"\nDone. ok={counters['ok']} resumed={counters['resumed']} fail={counters['fail']} "
            f"skipped={counters['skipped']} skipped_not_resumable={counters['skipped_not_resumable']} "
            f"pending={len(pending)}"
        )
        await reviser.cleanup()
        await workflow.cleanup()
        await evaluator.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Optional seed jsonl path. If omitted: cn -> data/nq_ch.jsonl, "
            "en -> data/nq_open_seed_queries_1000.jsonl."
        ),
    )
    parser.add_argument("--best-of-n", type=int, default=3)
    parser.add_argument(
        "--max-concurrent-seeds",
        type=int,
        default=2,
        help="Max number of seeds to explore in parallel (default: 3; set 1 for serial).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only run N seeds (after --start).")
    parser.add_argument("--start", type=int, default=0, help="Skip first N seeds.")
    parser.add_argument(
        "--language",
        type=str,
        default="cn",
        choices=["en", "cn"],
        help="Language switch: prompts, evaluation, and default seed dataset (default: cn).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "断点续跑：对已存在且 exploration_stop.reason=max_iterations_reached 的 case，"
            "从 iterations.json 中 pending 的下一轮继续（需先把 max_iterations 调高）。"
        ),
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional. If provided, sets OPENAI_API_KEY and overrides config.openai_api.api_key at runtime.",
    )
    args = parser.parse_args()

    asyncio.run(
        run(
            seed_file=Path(args.input) if args.input else None,
            best_of_n=args.best_of_n,
            limit=args.limit,
            start=args.start,
            api_key=args.api_key,
            language=args.language,
            max_concurrent_seeds=args.max_concurrent_seeds,
            resume=args.resume,
        )
    )


if __name__ == "__main__":
    main()

