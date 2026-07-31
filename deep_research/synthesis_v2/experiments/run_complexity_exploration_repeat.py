"""
对单个 seed case 重复跑 n 次复杂度探索，用于观察随机性带来的输出多样性。

每次运行写入 outputs/repeat/ 下独立目录（与主流程相同，冲突时自动追加 __2/__3…）：
  outputs/repeat/{query_id}/
  outputs/repeat/{query_id}__2/
  ...

跑完后生成多样性汇总：
  outputs/repeat/{query_id}_diversity.json

用法示例：

  # 对 nq_open_seed_0029 重复跑 5 次
  python deep_research/synthesis_v2/experiments/run_complexity_exploration_repeat.py \
    --query-id nq_open_seed_0029 \
    --n-repeats 5

  # 从 seed 文件第 0 条开始，跑 3 次
  python deep_research/synthesis_v2/experiments/run_complexity_exploration_repeat.py \
    --seed-index 0 \
    --n-repeats 3 \
    --language en

  # 中文提示词 + 英语 seed 数据集 + 英语模型输出
  python deep_research/synthesis_v2/experiments/run_complexity_exploration_repeat.py \
    --language cn-en \
    --start 0 \
    --limit 5 \
    --n-repeats 10 \
    --skip-existing

  # 等价写法：中文提示词，显式指定英语 seed
  python deep_research/synthesis_v2/experiments/run_complexity_exploration_repeat.py \
    --language cn \
    --output-language en \
    --input deep_research/synthesis_v2/data/nq_open_seed_queries_1000.jsonl \
    --start 0 \
    --limit 5 \
    --n-repeats 10 \
    --skip-existing
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from omegaconf import OmegaConf


def _prefer_local_repo_packages() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    verl_dir = repo_root / "verl"
    openjudge_dir = repo_root / "OpenJudge"
    sys.path.insert(0, str(repo_root))
    if verl_dir.exists():
        sys.path.insert(0, str(verl_dir))
    if openjudge_dir.exists():
        sys.path.insert(0, str(openjudge_dir))


_prefer_local_repo_packages()

from deep_research.synthesis_v2.core.data_structures import QueryData
from deep_research.synthesis_v2.complexity_exploration.mermaid_reviser import MermaidReviserAgent
from deep_research.synthesis_v2.complexity_exploration.orchestrator import ComplexityExplorationV2
from deep_research.synthesis_v2.core.concurrency import ConcurrencyManager
from deep_research.llm_utils import (
    normalize_openai_base_url,
    resolve_openai_api_key,
    resolve_rubric_api_key,
)
from deep_research.synthesis_v2.core.language import apply_language_to_config, resolve_seed_file
from deep_research.synthesis_v2.core.pipeline import SamplePipeline
from deep_research.synthesis_v2.evaluate.openjudge_evaluator import OpenJudgeEvaluator
from deep_research.synthesis_v2.io.path_utils import resolve_under_repo
from deep_research.synthesis_v2.workflow.native.query_trajectory_workflow import QueryTrajectoryWorkflow

_DIRNAME_SAFE_RE = re.compile(r"[^a-zA-Z0-9._-]+")
DEFAULT_REPEAT_OUTPUT = "outputs/repeat"
DEFAULT_SELECTION = "outputs/complexity_exploration/retained_case_ids_quality_filter.json"


def _resolve_repo_path(p: Path) -> Path:
    if p.is_absolute():
        return p
    repo_root = Path(__file__).resolve().parents[3]
    return (repo_root / p).resolve()


def _mask_secret(value: Optional[str]) -> str:
    if not value:
        return "unset"
    value = value.strip()
    if len(value) <= 8:
        return f"set(len={len(value)})"
    return f"set(prefix={value[:4]}..., suffix=...{value[-4:]}, len={len(value)})"


def _exploration_base_url(config: Any) -> str:
    generation = config.get("generation", {})
    openai_api = config.get("openai_api", {})
    return normalize_openai_base_url(
        generation.get("base_url")
        or os.getenv("OPENAI_BASE_URL")
        or openai_api.get("base_url", "https://api.openai.com/v1")
    )


def _apply_env_openai_key_to_config(config: Any) -> None:
    """将 OPENAI_API_KEY 写入探索链路各环节的 config，避免仅依赖运行时 env 回退。"""
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        return
    try:
        config.openai_api.api_key = key
        config.generation.api_key = key
        config.complexity_exploration.reviser.api_key = key
    except Exception:
        pass


def _log_and_validate_api_env(config: Any) -> None:
    """打印探索/rubric 实际使用的 endpoint 与 key 来源，避免与 test_context_capacity_v3cm 行为不一致。"""
    exploration_base = _exploration_base_url(config)
    rubric_base = normalize_openai_base_url(
        config.get("reward_function", {}).get("init_params", {}).get("base_url")
    )
    openai_env = (os.getenv("OPENAI_API_KEY") or "").strip() or None
    dashscope_env = (os.getenv("DASHSCOPE_API_KEY") or "").strip() or None
    exploration_key = resolve_openai_api_key(
        config.get("openai_api", {}).get("api_key"),
        base_url=exploration_base,
    )
    rubric_key = resolve_rubric_api_key(
        config.get("reward_function", {}).get("init_params", {}).get("api_key")
    )

    exploration_key_source = "unset"
    if exploration_key:
        if openai_env and exploration_key == openai_env:
            exploration_key_source = "OPENAI_API_KEY"
        elif dashscope_env and exploration_key == dashscope_env:
            exploration_key_source = "DASHSCOPE_API_KEY (fallback)"
        else:
            exploration_key_source = "config/unknown"

    rubric_key_source = "unset"
    if rubric_key:
        if dashscope_env and rubric_key == dashscope_env:
            rubric_key_source = "DASHSCOPE_API_KEY"
        elif openai_env and rubric_key == openai_env:
            rubric_key_source = "OPENAI_API_KEY (fallback)"
        else:
            rubric_key_source = "config/unknown"

    print("===== API 配置检查 =====")
    print(f"exploration_base_url={exploration_base}")
    print(f"exploration_key={_mask_secret(exploration_key)} source={exploration_key_source}")
    print(f"rubric_base_url={rubric_base}")
    print(f"rubric_key={_mask_secret(rubric_key)} source={rubric_key_source}")
    print(f"OPENAI_BASE_URL env={os.getenv('OPENAI_BASE_URL') or 'unset'}")
    print(
        "注意: test_context_capacity_v3cm.py 固定使用 https://api.v3.cm/v1，"
        "且不会读取 OPENAI_BASE_URL。"
    )

    if "v3.cm" in exploration_base and not exploration_key:
        raise RuntimeError(
            "探索链路将请求 api.v3.cm，但未找到可用的 OPENAI_API_KEY。"
            "请 export OPENAI_API_KEY='你的 v3.cm key' 后再运行；"
            "DASHSCOPE_API_KEY 不能用于 api.v3.cm。"
        )

    if "v3.cm" in exploration_base and exploration_key_source == "DASHSCOPE_API_KEY (fallback)":
        raise RuntimeError(
            "探索链路将请求 api.v3.cm，但当前只有 DASHSCOPE_API_KEY 可用。"
            "请 export OPENAI_API_KEY='你的 v3.cm key' 后再运行；"
            "或在同一终端执行: echo ${OPENAI_API_KEY:+set} ${DASHSCOPE_API_KEY:+set}"
        )

    if os.getenv("OPENAI_BASE_URL") and normalize_openai_base_url(os.getenv("OPENAI_BASE_URL")) != normalize_openai_base_url(
        config.get("openai_api", {}).get("base_url")
    ):
        print(
            "警告: OPENAI_BASE_URL 会覆盖 synthesis_v2.yaml 的 openai_api.base_url，"
            "这会导致 repeat 与 test_context_capacity_v3cm.py 走不同 endpoint。"
        )


def _safe_dirname(name: str, max_len: int = 120) -> str:
    name = (name or "").strip()
    if not name:
        return "empty_query_id"
    cleaned = _DIRNAME_SAFE_RE.sub("_", name)
    cleaned = cleaned.strip("._-")
    if not cleaned:
        cleaned = "query"
    return cleaned[:max_len]


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


def _load_all_seeds(seed_file: Path) -> List[QueryData]:
    seeds = list(_iter_seed_records(seed_file))
    if not seeds:
        raise ValueError(f"No seeds found in {seed_file}")
    return seeds


def _load_case_ids_from_selection(selection_path: Path) -> List[str]:
    data = json.loads(selection_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        case_ids = data.get("retained_case_ids")
        if isinstance(case_ids, list) and case_ids:
            return [str(case_id) for case_id in case_ids if case_id]
    raise ValueError(f"{selection_path} 缺少 retained_case_ids 列表")


def _load_seed_query(
    seed_file: Path,
    *,
    query_id: Optional[str],
    seed_index: Optional[int],
) -> QueryData:
    seeds = _load_all_seeds(seed_file)

    if query_id is not None:
        for seed in seeds:
            if seed.query_id == query_id:
                return seed
        known = ", ".join(s.query_id for s in seeds[:10])
        raise ValueError(f"query_id={query_id!r} not found in {seed_file}. First ids: {known}")

    index = 0 if seed_index is None else seed_index
    if index < 0 or index >= len(seeds):
        raise IndexError(f"seed_index={index} out of range (file has {len(seeds)} seeds)")
    return seeds[index]


def _select_seed_queries(
    seed_file: Path,
    *,
    query_id: Optional[str],
    seed_index: Optional[int],
    start: int,
    limit: Optional[int],
    selection_file: Optional[Path],
) -> List[QueryData]:
    if query_id is not None:
        return [_load_seed_query(seed_file, query_id=query_id, seed_index=None)]

    seeds = _load_all_seeds(seed_file)
    seed_by_id = {seed.query_id: seed for seed in seeds}

    if seed_index is not None:
        if seed_index < 0 or seed_index >= len(seeds):
            raise IndexError(f"seed_index={seed_index} out of range (file has {len(seeds)} seeds)")
        return [seeds[seed_index]]

    eff_limit = 1 if limit is None else limit
    if selection_file is not None:
        retained_ids = _load_case_ids_from_selection(selection_file)
        selected_ids: List[str] = []
        for idx, case_id in enumerate(retained_ids):
            if idx < start:
                continue
            selected_ids.append(case_id)
            if len(selected_ids) >= eff_limit:
                break
        if not selected_ids:
            raise ValueError(
                f"No retained cases selected (start={start}, limit={limit}, "
                f"retained_total={len(retained_ids)}, selection={selection_file})"
            )
        selected: List[QueryData] = []
        missing: List[str] = []
        for case_id in selected_ids:
            seed = seed_by_id.get(case_id)
            if seed is None:
                missing.append(case_id)
            else:
                selected.append(seed)
        if missing:
            raise ValueError(
                f"retained case ids not found in seed file {seed_file}: {missing[:10]}"
            )
        return selected

    selected = []
    for idx, seed in enumerate(seeds):
        if idx < start:
            continue
        selected.append(seed)
        if len(selected) >= eff_limit:
            break
    if not selected:
        raise ValueError(f"No seeds selected (start={start}, limit={limit}, total={len(seeds)})")
    return selected


def _repeat_run_is_successful(case_dir: Path) -> bool:
    return bool(_summarize_run(case_dir).get("has_mermaid"))


def _successful_repeat_run_dirs(output_root: Path, query_id: str) -> List[Path]:
    return [d for d in _list_repeat_run_dirs(output_root, query_id) if _repeat_run_is_successful(d)]


def _list_repeat_run_dirs(output_root: Path, query_id: str) -> List[Path]:
    """列出某 query_id 在 repeat 输出根目录下的所有已完成 run 目录。"""
    base = _safe_dirname(query_id)
    dirs: List[Path] = []
    direct = output_root / base
    if direct.is_dir() and (direct / "iterations.json").is_file():
        dirs.append(direct)
    for candidate in sorted(output_root.glob(f"{base}__*")):
        if candidate.is_dir() and (candidate / "iterations.json").is_file():
            dirs.append(candidate)
    return dirs


def _load_iterations_payload(case_dir: Path) -> Dict[str, Any]:
    path = case_dir / "iterations.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        iterations = data.get("iterations")
        if not isinstance(iterations, list):
            raise ValueError(f"iterations.json 格式错误: {path}")
        return data
    if isinstance(data, list):
        return {"iterations": data}
    raise ValueError(f"iterations.json 格式错误: {path}")


def _final_iteration_with_mermaid(case_dir: Path) -> Dict[str, Any]:
    iterations = _load_iterations_payload(case_dir).get("iterations") or []
    candidates = [it for it in iterations if isinstance(it, dict) and it.get("mermaid")]
    if candidates:
        return max(candidates, key=lambda it: it.get("iteration_id", -1))
    if iterations:
        return max(iterations, key=lambda it: it.get("iteration_id", -1))
    return {}


def _token_jaccard(a: str, b: str) -> float:
    ta = {t for t in re.split(r"\W+", (a or "").lower()) if t}
    tb = {t for t in re.split(r"\W+", (b or "").lower()) if t}
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _pairwise_mean_jaccard(texts: Sequence[str]) -> Optional[float]:
    if len(texts) < 2:
        return None
    scores: List[float] = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            scores.append(_token_jaccard(texts[i], texts[j]))
    return statistics.mean(scores) if scores else None


def _numeric_stats(values: Sequence[int]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "stdev": None, "unique": []}
    unique_sorted = sorted(set(values))
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 4),
        "stdev": round(stdev, 4),
        "unique": unique_sorted,
    }


def _summarize_run(case_dir: Path) -> Dict[str, Any]:
    payload = _load_iterations_payload(case_dir)
    iterations = payload.get("iterations") or []
    stop = payload.get("exploration_stop") if isinstance(payload, dict) else None
    final_it = _final_iteration_with_mermaid(case_dir)

    query = final_it.get("query_fuzzified") or final_it.get("query")
    return {
        "run_dir": str(case_dir),
        "run_name": case_dir.name,
        "total_iterations": len(iterations),
        "exploration_stop": stop,
        "final_iteration_id": final_it.get("iteration_id"),
        "final_query": final_it.get("query"),
        "final_query_fuzzified": final_it.get("query_fuzzified"),
        "final_query_key": query,
        "final_mermaid_structure": final_it.get("mermaid_structure"),
        "final_depth": final_it.get("depth"),
        "final_node_count": final_it.get("node_count"),
        "has_mermaid": bool(final_it.get("mermaid")),
    }


def compute_diversity_summary(
    *,
    query_id: str,
    seed_query: str,
    run_dirs: Sequence[Path],
    n_repeats_target: int,
) -> Dict[str, Any]:
    runs = [_summarize_run(case_dir) for case_dir in run_dirs]
    successful = [r for r in runs if r.get("has_mermaid")]

    final_queries = [r.get("final_query_key") or "" for r in successful]
    final_structures = [r.get("final_mermaid_structure") or "" for r in successful if r.get("final_mermaid_structure")]
    depths = [r["final_depth"] for r in successful if isinstance(r.get("final_depth"), int)]
    node_counts = [r["final_node_count"] for r in successful if isinstance(r.get("final_node_count"), int)]
    iteration_counts = [r["total_iterations"] for r in runs]

    stop_reasons = Counter(
        (r.get("exploration_stop") or {}).get("reason", "unknown") for r in runs
    )

    unique_queries = sorted(set(final_queries))
    unique_structures = sorted(set(final_structures))

    return {
        "query_id": query_id,
        "seed_query": seed_query,
        "n_repeats_target": n_repeats_target,
        "n_runs_completed": len(runs),
        "n_runs_with_mermaid": len(successful),
        "run_dirs": [str(p) for p in run_dirs],
        "runs": runs,
        "diversity": {
            "final_query": {
                "unique_count": len(unique_queries),
                "duplicate_rate": (
                    round(1 - len(unique_queries) / len(final_queries), 4) if final_queries else None
                ),
                "pairwise_token_jaccard_mean": _pairwise_mean_jaccard(final_queries),
                "unique_samples": unique_queries[:20],
            },
            "final_mermaid_structure": {
                "unique_count": len(unique_structures),
                "duplicate_rate": (
                    round(1 - len(unique_structures) / len(final_structures), 4)
                    if final_structures
                    else None
                ),
                "pairwise_exact_match_rate": (
                    round(
                        sum(
                            1
                            for i in range(len(final_structures))
                            for j in range(i + 1, len(final_structures))
                            if final_structures[i] == final_structures[j]
                        )
                        / (len(final_structures) * (len(final_structures) - 1) / 2),
                        4,
                    )
                    if len(final_structures) > 1
                    else None
                ),
            },
            "final_depth": _numeric_stats(depths),
            "final_node_count": _numeric_stats(node_counts),
            "total_iterations": _numeric_stats(iteration_counts),
            "convergence_reasons": dict(stop_reasons),
        },
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _write_diversity_summary(output_root: Path, query_id: str, summary: Dict[str, Any]) -> Path:
    path = output_root / f"{_safe_dirname(query_id)}_diversity.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


async def _explore_once(
    *,
    run_index: int,
    seed_query: QueryData,
    explorer: ComplexityExplorationV2,
    best_of_n: int,
    output_root: Path,
    known_dirs: Set[Path],
) -> Tuple[int, Optional[Path], Optional[str]]:
    print(f"\n===== [repeat {run_index}] query_id={seed_query.query_id} =====")
    try:
        evolution = await explorer.explore(
            seed_query,
            best_of_n=None if best_of_n <= 1 else best_of_n,
        )
        if evolution is None:
            print(f"[repeat {run_index}] FAILED: evolution=None")
            return run_index, None, "evolution_none"

        all_dirs = _list_repeat_run_dirs(output_root, seed_query.query_id)
        new_dirs = [d for d in all_dirs if d.resolve() not in known_dirs]
        latest_dir = new_dirs[-1] if new_dirs else (all_dirs[-1] if all_dirs else None)
        if latest_dir is not None:
            known_dirs.add(latest_dir.resolve())

        converged = getattr(evolution, "converged", None)
        iters = getattr(evolution, "total_iterations", None)
        reason = getattr(evolution, "convergence_reason", None)
        print(
            f"[repeat {run_index}] OK: dir={latest_dir.name if latest_dir else None} "
            f"converged={converged} total_iterations={iters} reason={reason}"
        )
        return run_index, latest_dir, None
    except Exception as e:
        print(f"[repeat {run_index}] EXCEPTION: {type(e).__name__}: {e}")
        return run_index, None, f"{type(e).__name__}: {e}"


async def _run_one_seed(
    *,
    seed_query: QueryData,
    n_repeats: int,
    best_of_n: int,
    language: str,
    seed_file_resolved: Path,
    output_root: Path,
    skip_existing: bool,
    summarize_only: bool,
    explorer: Optional[ComplexityExplorationV2],
    failures: List[str],
) -> Dict[str, Any]:
    print(f"\n########## case query_id={seed_query.query_id} ##########")
    print(f"seed_query={seed_query.content[:120]}...")

    all_existing_dirs = _list_repeat_run_dirs(output_root, seed_query.query_id)
    successful_dirs = _successful_repeat_run_dirs(output_root, seed_query.query_id)
    failed_dirs = [d for d in all_existing_dirs if d not in successful_dirs]

    if skip_existing:
        runs_needed = max(0, n_repeats - len(successful_dirs))
        if failed_dirs:
            print(
                f"Found {len(failed_dirs)} failed run(s) (401/无 mermaid，不计入 --skip-existing)；"
                f"successful={len(successful_dirs)}/{n_repeats}, will run {runs_needed} more."
            )
        elif successful_dirs:
            print(f"Found {len(successful_dirs)} successful run(s); will run {runs_needed} more.")
    else:
        runs_needed = n_repeats

    if not summarize_only and runs_needed > 0:
        if explorer is None:
            raise RuntimeError("explorer is required when running exploration")
        known_dirs = {d.resolve() for d in all_existing_dirs}
        before_count = len(all_existing_dirs)
        for run_i in range(1, runs_needed + 1):
            _, _, err = await _explore_once(
                run_index=before_count + run_i,
                seed_query=seed_query,
                explorer=explorer,
                best_of_n=best_of_n,
                output_root=output_root,
                known_dirs=known_dirs,
            )
            if err:
                failures.append(f"{seed_query.query_id}:{err}")

    all_dirs = _list_repeat_run_dirs(output_root, seed_query.query_id)
    summary = compute_diversity_summary(
        query_id=seed_query.query_id,
        seed_query=seed_query.content,
        run_dirs=all_dirs,
        n_repeats_target=n_repeats,
    )
    summary["config"] = {
        "language": language,
        "best_of_n": best_of_n,
        "output_dir": str(output_root),
        "seed_file": str(seed_file_resolved),
    }
    summary["failures"] = [f for f in failures if f.startswith(f"{seed_query.query_id}:")]
    summary_path = _write_diversity_summary(output_root, seed_query.query_id, summary)

    div = summary["diversity"]
    print("\n===== Diversity Summary =====")
    print(f"completed_runs={summary['n_runs_completed']} with_mermaid={summary['n_runs_with_mermaid']}")
    print(
        f"final_query unique={div['final_query']['unique_count']} "
        f"jaccard_mean={div['final_query']['pairwise_token_jaccard_mean']}"
    )
    print(
        f"mermaid_structure unique={div['final_mermaid_structure']['unique_count']} "
        f"exact_match_rate={div['final_mermaid_structure']['pairwise_exact_match_rate']}"
    )
    print(f"depth stats={div['final_depth']}")
    print(f"node_count stats={div['final_node_count']}")
    print(f"convergence_reasons={div['convergence_reasons']}")
    print(f"Wrote {summary_path}")
    return summary


async def run(
    *,
    seed_file: Optional[Path],
    query_id: Optional[str],
    seed_index: Optional[int],
    start: int,
    limit: Optional[int],
    selection_file: Optional[Path],
    n_repeats: int,
    best_of_n: int,
    api_key: Optional[str],
    language: str,
    output_language: Optional[str],
    output_dir: str,
    skip_existing: bool,
    summarize_only: bool,
) -> None:
    config_path = Path(__file__).resolve().parents[1] / "config" / "synthesis_v2.yaml"
    config = OmegaConf.load(str(config_path))
    language_label = apply_language_to_config(config, language, output_language)

    seed_path = resolve_seed_file(
        language,
        str(seed_file) if seed_file else None,
        output_language,
    )
    seed_file_resolved = _resolve_repo_path(Path(seed_path))
    if not seed_file_resolved.is_file():
        raise FileNotFoundError(f"Seed file not found: {seed_file_resolved}")

    selection_resolved: Optional[Path] = None
    if selection_file is not None:
        selection_resolved = _resolve_repo_path(selection_file)
        if not selection_resolved.is_file():
            raise FileNotFoundError(f"Selection file not found: {selection_resolved}")

    seed_queries = _select_seed_queries(
        seed_file_resolved,
        query_id=query_id,
        seed_index=seed_index,
        start=start,
        limit=limit,
        selection_file=selection_resolved,
    )

    output_root = resolve_under_repo(output_dir, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Using language={language_label}")
    print(f"Using seed_file={seed_file_resolved}")
    if selection_resolved is not None and query_id is None and seed_index is None:
        print(f"Using selection_file={selection_resolved}")
    print(f"Using cases={[s.query_id for s in seed_queries]}")
    print(f"Using output_root={output_root}")
    print(f"Using n_repeats={n_repeats}")
    print(f"Using skip_existing={skip_existing}")
    print(f"Using summarize_only={summarize_only}")

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        try:
            config.openai_api.api_key = api_key
        except Exception:
            pass

    _apply_env_openai_key_to_config(config)
    _log_and_validate_api_env(config)

    # 所有 repeat 写入同一 output_root，由 orchestrator 自动 __2/__3 避重
    config.complexity_exploration.output_dir = str(output_dir)

    explorer: Optional[ComplexityExplorationV2] = None
    reviser: Optional[MermaidReviserAgent] = None
    workflow: Optional[QueryTrajectoryWorkflow] = None
    evaluator: Optional[OpenJudgeEvaluator] = None

    failures: List[str] = []
    batch_summaries: List[Dict[str, Any]] = []

    try:
        needs_explore = not summarize_only and any(
            (
                (not skip_existing)
                or max(
                    0,
                    n_repeats
                    - len(_successful_repeat_run_dirs(output_root, s.query_id)),
                )
                > 0
            )
            for s in seed_queries
        )

        if needs_explore:
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

        for seed_query in seed_queries:
            summary = await _run_one_seed(
                seed_query=seed_query,
                n_repeats=n_repeats,
                best_of_n=best_of_n,
                language=language_label,
                seed_file_resolved=seed_file_resolved,
                output_root=output_root,
                skip_existing=skip_existing,
                summarize_only=summarize_only,
                explorer=explorer,
                failures=failures,
            )
            batch_summaries.append(summary)

        if len(batch_summaries) > 1:
            batch_path = output_root / "batch_diversity_summary.json"
            batch_path.write_text(
                json.dumps(
                    {
                        "cases": batch_summaries,
                        "n_cases": len(batch_summaries),
                        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"\nWrote batch summary: {batch_path}")
    finally:
        if reviser is not None:
            await reviser.cleanup()
        if workflow is not None:
            await workflow.cleanup()
        if evaluator is not None:
            await evaluator.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="对单个 seed case 重复跑复杂度探索，并汇总输出多样性。",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional seed jsonl path.",
    )
    parser.add_argument(
        "--query-id",
        type=str,
        default=None,
        help="要重复的 seed query_id（与 --seed-index 二选一）。",
    )
    parser.add_argument(
        "--seed-index",
        type=int,
        default=None,
        help="从 seed 文件选取第 N 条（0-based）。",
    )
    parser.add_argument(
        "--selection",
        type=str,
        default=DEFAULT_SELECTION,
        help=(
            "批量模式：从该 JSON 读取 retained_case_ids，"
            f"--start/--limit 作用于保留列表（default: {DEFAULT_SELECTION}）。"
            "传 none 则改为按 seed 文件顺序选取。"
        ),
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="批量模式：跳过 retained_case_ids 前 N 条（或 --selection none 时跳过 seed 前 N 条）。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="批量模式：最多处理 N 个 case（默认 1）。",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=3,
        help="重复运行次数（default: 3）。",
    )
    parser.add_argument("--best-of-n", type=int, default=3)
    parser.add_argument(
        "--language",
        type=str,
        default="cn",
        help=(
            "提示词模板语言：cn | en | cn-en（中文提示词+英语 seed/输出）| en-cn | "
            "也支持 cn_en / zh-en 等别名。"
        ),
    )
    parser.add_argument(
        "--output-language",
        type=str,
        default=None,
        choices=["cn", "en"],
        help="可选：模型输出与默认 seed 语言（cn | en）。未指定时由 --language 决定。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_REPEAT_OUTPUT,
        help=f"重复实验输出根目录（default: {DEFAULT_REPEAT_OUTPUT}）。",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="若已有成功 run（含 mermaid），只补足到 --n-repeats 次；失败 run 不计入。",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="不跑探索，仅根据已有 run 目录重新计算多样性汇总。",
    )
    parser.add_argument("--api-key", type=str, default=None)
    args = parser.parse_args()

    if args.n_repeats < 1:
        parser.error("--n-repeats must be >= 1")

    if args.query_id is not None and (args.seed_index is not None or args.start or args.limit):
        parser.error("--query-id 不能与 --seed-index/--start/--limit 同时使用")

    selection_arg = (args.selection or "").strip()
    selection_file: Optional[Path] = None
    if selection_arg.lower() not in {"", "none", "null"}:
        selection_file = Path(selection_arg)

    asyncio.run(
        run(
            seed_file=Path(args.input) if args.input else None,
            query_id=args.query_id,
            seed_index=args.seed_index,
            start=args.start,
            limit=args.limit,
            selection_file=selection_file,
            n_repeats=args.n_repeats,
            best_of_n=args.best_of_n,
            api_key=args.api_key,
            language=args.language,
            output_language=args.output_language,
            output_dir=args.output_dir,
            skip_existing=args.skip_existing,
            summarize_only=args.summarize_only,
        )
    )


if __name__ == "__main__":
    main()
