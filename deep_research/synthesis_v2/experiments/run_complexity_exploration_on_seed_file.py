"""
读取 seeds jsonl，并对每条 QueryData 执行 ComplexityExplorationV2.explore。

用法示例：
  python /Users/xiaocan/agentRL/deep_research/synthesis_v2/experiments/run_complexity_exploration_on_seed_file.py \
    --input deep_research/synthesis_v2/data/nq_ch.jsonl \
    --best-of-n 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

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
    base = _safe_dirname(query_id)
    if (output_root / base).exists():
        return True
    # 兼容同名目录冲突时的 __2/__3...
    return any(output_root.glob(f"{base}__*"))


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


async def run(
    seed_file: Path,
    best_of_n: int,
    limit: Optional[int],
    start: int,
    api_key: Optional[str],
) -> None:
    seed_file = _resolve_repo_path(seed_file)
    config_path = Path(__file__).resolve().parents[1] / "config" / "synthesis_v2.yaml"
    config = OmegaConf.load(str(config_path))

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

    ok = 0
    fail = 0
    skipped = 0
    total_seen = 0
    try:
        for idx, seed_query in enumerate(_iter_seed_records(seed_file), start=0):
            if idx < start:
                continue
            total_seen += 1
            if limit is not None and total_seen > limit:
                break

            if _already_ran(output_root, seed_query.query_id):
                skipped += 1
                if skipped % 50 == 1:
                    print(f"[skip] already done: {seed_query.query_id} (skipped={skipped})")
                continue

            print(f"\n===== [{total_seen}] explore query_id={seed_query.query_id} =====")
            try:
                evolution = await explorer.explore(
                    seed_query,
                    best_of_n=None if best_of_n <= 1 else best_of_n,
                )
                if evolution is None:
                    fail += 1
                    print(f"[{total_seen}] FAILED: evolution=None")
                else:
                    ok += 1
                    converged = getattr(evolution, "converged", None)
                    iters = getattr(evolution, "total_iterations", None)
                    print(f"[{total_seen}] OK: converged={converged} total_iterations={iters}")
            except Exception as e:
                fail += 1
                print(f"[{total_seen}] EXCEPTION: {type(e).__name__}: {e}")

    finally:
        print(f"\nDone. ok={ok} fail={fail} skipped={skipped}")
        await reviser.cleanup()
        await workflow.cleanup()
        await evaluator.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Seed jsonl file. Each line must be QueryData-compatible JSON.",
    )
    parser.add_argument("--best-of-n", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="Only run N seeds (after --start).")
    parser.add_argument("--start", type=int, default=0, help="Skip first N seeds.")
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional. If provided, sets OPENAI_API_KEY and overrides config.openai_api.api_key at runtime.",
    )
    args = parser.parse_args()

    asyncio.run(
        run(
            seed_file=Path(args.input),
            best_of_n=args.best_of_n,
            limit=args.limit,
            start=args.start,
            api_key=args.api_key,
        )
    )


if __name__ == "__main__":
    main()

