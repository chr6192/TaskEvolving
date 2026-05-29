#!/usr/bin/env python3
"""
最小可运行示例：测试 google_search 工具。

用法：
  python deep_research/tool/test_google_search.py
  python deep_research/tool/test_google_search.py --q "artificial intelligence news" --num 5 --tbs "qdr:w"
"""

import argparse
import asyncio
from pathlib import Path
import sys

import yaml
from verl.tools.schemas import OpenAIFunctionToolSchema

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deep_research.tool.dr_tool_as import GoogleSearchTool

DEFAULT_QUERY = "Python programming language"


def _load_google_search_schema(config_path: Path) -> OpenAIFunctionToolSchema:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for tool_cfg in config.get("tools", []):
        fn = tool_cfg.get("tool_schema", {}).get("function", {})
        if fn.get("name") == "google_search":
            return OpenAIFunctionToolSchema.model_validate(tool_cfg["tool_schema"])

    raise ValueError(f"未在配置中找到 google_search: {config_path}")


async def _run(q: str, gl: str, hl: str, num: int, tbs: str, config_path: Path) -> int:
    schema = _load_google_search_schema(config_path)
    tool = GoogleSearchTool(config={"type": "native"}, tool_schema=schema)
    instance_id = await tool.create()
    try:
        parameters = {
            "q": q,
            "gl": gl,
            "hl": hl,
            "num": num,
        }
        if tbs:
            parameters["tbs"] = tbs

        response, score, metrics = await tool.execute(
            instance_id=instance_id,
            parameters=parameters,
        )
    finally:
        await tool.release(instance_id)

    print("=" * 80)
    print("google_search 调用完成")
    print(f"q: {q}")
    print(f"gl: {gl}")
    print(f"hl: {hl}")
    print(f"num: {num}")
    print(f"tbs: {tbs or '(empty)'}")
    print(f"score: {score}")
    print(f"metrics: {metrics}")
    print("-" * 80)
    preview = response[:1200] if isinstance(response, str) else str(response)
    print(preview)
    if isinstance(response, str) and len(response) > 1200:
        print("\n... (已截断，完整内容见工具日志)")
    print("=" * 80)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="测试 google_search 工具")
    parser.add_argument("--q", default=DEFAULT_QUERY, help="搜索关键词")
    parser.add_argument("--gl", default="us", help="国家/地区代码")
    parser.add_argument("--hl", default="en", help="语言代码")
    parser.add_argument("--num", type=int, default=10, help="返回结果数量")
    parser.add_argument("--tbs", default="", help="时间/筛选条件，例如 qdr:w")
    parser.add_argument(
        "--config",
        default="deep_research/config/tool_config/normal.yaml",
        help="工具配置文件路径",
    )
    args = parser.parse_args()

    return asyncio.run(_run(args.q, args.gl, args.hl, args.num, args.tbs, Path(args.config)))


if __name__ == "__main__":
    raise SystemExit(main())
