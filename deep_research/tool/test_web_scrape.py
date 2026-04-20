#!/usr/bin/env python3
"""
最小可运行示例：测试 web_scrape 工具。

用法：
  python deep_research/tool/test_web_scrape.py
  python deep_research/tool/test_web_scrape.py --url "https://example.com"
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

from deep_research.tool.dr_tool_as import WebScrapeTool

DEFAULT_URL = "https://hk.trip.com/blog/jinan-128-1day-itinerary-9998650110/"


def _load_web_scrape_schema(config_path: Path) -> OpenAIFunctionToolSchema:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for tool_cfg in config.get("tools", []):
        fn = tool_cfg.get("tool_schema", {}).get("function", {})
        if fn.get("name") == "web_scrape":
            return OpenAIFunctionToolSchema.model_validate(tool_cfg["tool_schema"])

    raise ValueError(f"未在配置中找到 web_scrape: {config_path}")


async def _run(url: str, config_path: Path) -> int:
    schema = _load_web_scrape_schema(config_path)
    tool = WebScrapeTool(config={"type": "native"}, tool_schema=schema)
    instance_id = await tool.create()
    try:
        response, score, metrics = await tool.execute(
            instance_id=instance_id,
            parameters={"url": url},
        )
    finally:
        await tool.release(instance_id)

    print("=" * 80)
    print("web_scrape 调用完成")
    print(f"url: {url}")
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
    parser = argparse.ArgumentParser(description="测试 web_scrape 工具")
    parser.add_argument("--url", default=DEFAULT_URL, help="待抓取网页 URL")
    parser.add_argument(
        "--config",
        default="deep_research/config/tool_config/normal.yaml",
        help="工具配置文件路径",
    )
    args = parser.parse_args()

    return asyncio.run(_run(args.url, Path(args.config)))


if __name__ == "__main__":
    raise SystemExit(main())
