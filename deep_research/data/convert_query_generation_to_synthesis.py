#!/usr/bin/env python3
"""
Query Generation输出格式转换工具

将query_generation阶段生成的数据转换为skeleton generation和trajectory generation能够接受的parquet格式。

输入格式（query_generation输出）:
{
  "query_id": "uuid",
  "content": "查询内容",
  "metadata": {
    "complexity_level": "medium",
    "query_type": "个股分析",
    "expected_tools": ["tool1", "tool2"],
    ...
  },
  "rubrics": null,
  "created_at": "timestamp"
}

输出格式（parquet）:
{
  "data_source": "query_generation",
  "prompt": [],
  "extra_info": {
    "need_tools_kwargs": false,
    "skeleton": {
      "query": "查询内容",
      "analysis": "",
      "intent": "",
      "need_multi_step": true,
      "tool_calls_reference": "",
      "uid": ""
    },
    "tools_kwargs": {"dummy_field": null}
  }
}
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_json_data(input_path: Path) -> List[Dict[str, Any]]:
    """
    从JSON或JSONL文件加载数据
    
    Args:
        input_path: 输入文件路径
        
    Returns:
        查询数据列表
    """
    queries = []
    
    try:
        if input_path.suffix.lower() == '.jsonl':
            # JSONL格式：每行一个JSON对象
            logger.info(f"从JSONL文件读取数据: {input_path}")
            with open(input_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        queries.append(data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"跳过无效JSON行 {line_num}: {e}")
                        continue
                        
        elif input_path.suffix.lower() == '.json':
            # JSON格式：完整的JSON数组或对象
            logger.info(f"从JSON文件读取数据: {input_path}")
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                queries = data
            elif isinstance(data, dict):
                queries = [data]
            else:
                raise ValueError(f"不支持的JSON格式: {type(data)}")
        else:
            raise ValueError(f"不支持的文件格式: {input_path.suffix}，仅支持.json和.jsonl")
        
        logger.info(f"成功读取 {len(queries)} 条查询数据")
        return queries
        
    except Exception as e:
        logger.error(f"读取文件失败: {e}")
        raise


def get_tool_names_from_config(config_path: Optional[Path] = None) -> List[str]:
    """
    从工具配置文件获取工具名称列表
    
    Args:
        config_path: 工具配置文件路径（可选）
        
    Returns:
        工具名称列表
    """
    # 默认工具列表（基于dr_tool_config.yaml）
    default_tools = [
        "user_info",
        "investment_analysis",
        "portfolio_allocation",
        "recommend_investments",
        "finance_search",
        "fund_custom_premium"
    ]
    
    if config_path and config_path.exists():
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            tool_names = []
            for tool in config.get('tools', []):
                tool_schema = tool.get('tool_schema', {})
                function = tool_schema.get('function', {})
                name = function.get('name', '')
                if name:
                    tool_names.append(name)
            
            if tool_names:
                logger.info(f"从配置文件读取到 {len(tool_names)} 个工具")
                return tool_names
        except Exception as e:
            logger.warning(f"读取工具配置文件失败: {e}，使用默认工具列表")
    
    return default_tools


def create_tools_kwargs(tool_names: List[str], include_dummy: bool = True) -> Dict[str, Any]:
    """
    创建tools_kwargs字典
    
    Args:
        tool_names: 工具名称列表
        include_dummy: 是否包含dummy_field（用于空tools_kwargs的情况）
        
    Returns:
        tools_kwargs字典
    """
    if not tool_names and include_dummy:
        # 空的tools_kwargs需要有dummy_field避免parquet写入错误
        return {"dummy_field": None}
    
    # 为每个工具创建空参数字典
    tools_kwargs = {}
    for tool_name in tool_names:
        tools_kwargs[tool_name] = {"dummy_field": None}
    
    return tools_kwargs


def convert_query_to_synthesis_format(
    query: Dict[str, Any],
    data_source: str,
    tools_kwargs: Dict[str, Any],
    use_expected_tools: bool = True
) -> Dict[str, Any]:
    """
    将单个query转换为synthesis格式
    
    Args:
        query: query_generation输出的查询数据
        data_source: 数据源标识
        tools_kwargs: 工具参数字典
        use_expected_tools: 是否使用metadata中的expected_tools来填充tool_calls_reference
        
    Returns:
        转换后的数据
    """
    # 提取基本字段
    query_id = query.get('query_id', '')
    content = query.get('content', '')
    metadata = query.get('metadata', {})
    
    # 构建skeleton字段
    skeleton = {
        "query": content,
        "query_id": query_id,
        "analysis": "",
        "intent": metadata.get('query_type', ''),
        "need_multi_step": True,
        "tool_calls_reference": "",
        "uid": ""
    }
    
    # 如果metadata中有expected_tools，可以将其转换为简单的tool_calls_reference
    if use_expected_tools and 'expected_tools' in metadata:
        expected_tools = metadata.get('expected_tools', [])
        if expected_tools:
            # 创建简单的工具调用引用格式
            tool_calls = []
            for idx, tool_name in enumerate(expected_tools):
                tool_calls.append({
                    "step": idx,
                    "tool": [{
                        "name": tool_name,
                        "parameters": {}
                    }]
                })
            skeleton["tool_calls_reference"] = json.dumps(tool_calls, ensure_ascii=False)
    
    # 保留原始metadata信息
    skeleton["metadata"] = metadata
    skeleton["created_at"] = query.get('created_at', datetime.now().isoformat())
    
    # 构建最终格式
    result = {
        "data_source": data_source,
        "prompt": [],  # 空prompt列表
        "extra_info": {
            "need_tools_kwargs": False,  # 默认不需要tools_kwargs
            "skeleton": skeleton,
            "tools_kwargs": tools_kwargs
        }
    }
    
    return result


def convert_and_save(
    queries: List[Dict[str, Any]],
    output_path: Path,
    data_source: str,
    tool_config_path: Optional[Path] = None,
    use_expected_tools: bool = True
) -> None:
    """
    转换数据并保存为parquet格式
    
    Args:
        queries: 查询数据列表
        output_path: 输出文件路径
        data_source: 数据源标识
        tool_config_path: 工具配置文件路径
        use_expected_tools: 是否使用metadata中的expected_tools
    """
    try:
        import pandas as pd
    except ImportError:
        logger.error("需要安装pandas库: pip install pandas pyarrow")
        sys.exit(1)
    
    # 获取工具列表
    tool_names = get_tool_names_from_config(tool_config_path)
    tools_kwargs = create_tools_kwargs(tool_names)
    
    logger.info(f"开始转换 {len(queries)} 条查询...")
    
    # 转换所有查询
    converted_data = []
    for i, query in enumerate(queries):
        try:
            converted = convert_query_to_synthesis_format(
                query, 
                data_source, 
                tools_kwargs,
                use_expected_tools
            )
            converted_data.append(converted)
        except Exception as e:
            logger.warning(f"跳过第 {i+1} 条数据（转换失败）: {e}")
            continue
    
    if not converted_data:
        logger.error("没有成功转换的数据")
        sys.exit(1)
    
    logger.info(f"成功转换 {len(converted_data)} 条数据")
    
    # 创建DataFrame
    df = pd.DataFrame(converted_data)
    
    # 保存为parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    logger.info(f"成功保存到: {output_path}")
    logger.info(f"文件大小: {output_path.stat().st_size / 1024:.2f} KB")


def validate_input_data(queries: List[Dict[str, Any]]) -> bool:
    """
    验证输入数据格式
    
    Args:
        queries: 查询数据列表
        
    Returns:
        是否验证通过
    """
    if not queries:
        logger.error("输入数据为空")
        return False
    
    errors = []
    for i, query in enumerate(queries):
        if not isinstance(query, dict):
            errors.append(f"第 {i+1} 条数据不是字典类型")
            continue
        
        # 检查必需字段
        if 'query_id' not in query:
            errors.append(f"第 {i+1} 条数据缺少 query_id 字段")
        if 'content' not in query:
            errors.append(f"第 {i+1} 条数据缺少 content 字段")
        elif not query['content'].strip():
            errors.append(f"第 {i+1} 条数据的 content 字段为空")
    
    if errors:
        logger.error("数据验证失败:")
        for error in errors[:10]:  # 只显示前10个错误
            logger.error(f"  - {error}")
        if len(errors) > 10:
            logger.error(f"  ... 还有 {len(errors) - 10} 个错误")
        return False
    
    logger.info("数据验证通过")
    return True


def print_sample_output(queries: List[Dict[str, Any]], tools_kwargs: Dict[str, Any]) -> None:
    """打印示例输出"""
    if not queries:
        return
    
    sample_query = queries[0]
    sample_converted = convert_query_to_synthesis_format(
        sample_query,
        "query_generation",
        tools_kwargs
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("示例输入数据:")
    logger.info("=" * 60)
    logger.info(json.dumps(sample_query, ensure_ascii=False, indent=2))
    
    logger.info("\n" + "=" * 60)
    logger.info("示例输出数据:")
    logger.info("=" * 60)
    logger.info(json.dumps(sample_converted, ensure_ascii=False, indent=2, default=str))
    logger.info("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="将query_generation输出转换为synthesis输入格式（parquet）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法
  python convert_query_generation_to_synthesis.py queries.json output.parquet
  
  # 指定数据源名称
  python convert_query_generation_to_synthesis.py queries.jsonl output.parquet --data-source my_queries
  
  # 指定工具配置文件
  python convert_query_generation_to_synthesis.py queries.json output.parquet \\
      --tool-config ../config/tool_config/dr_tool_config.yaml
  
  # 仅验证数据格式
  python convert_query_generation_to_synthesis.py queries.json output.parquet --validate-only
  
  # 显示示例输出
  python convert_query_generation_to_synthesis.py queries.json output.parquet --show-sample
        """
    )
    
    parser.add_argument(
        'input_file',
        type=Path,
        help='输入文件路径（JSON或JSONL格式）'
    )
    
    parser.add_argument(
        'output_file',
        type=Path,
        help='输出文件路径（parquet格式）'
    )
    
    parser.add_argument(
        '--data-source',
        type=str,
        default='query_generation',
        help='数据源标识（默认: query_generation）'
    )
    
    parser.add_argument(
        '--tool-config',
        type=Path,
        help='工具配置文件路径（可选，默认使用内置工具列表）'
    )
    
    parser.add_argument(
        '--use-expected-tools',
        action='store_true',
        default=True,
        help='使用metadata中的expected_tools填充tool_calls_reference（默认: True）'
    )
    
    parser.add_argument(
        '--no-expected-tools',
        action='store_true',
        help='不使用expected_tools（保持tool_calls_reference为空）'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='仅验证数据格式，不保存文件'
    )
    
    parser.add_argument(
        '--show-sample',
        action='store_true',
        help='显示示例输入输出数据'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not args.input_file.exists():
        logger.error(f"输入文件不存在: {args.input_file}")
        sys.exit(1)
    
    # 检查工具配置文件
    if args.tool_config and not args.tool_config.exists():
        logger.warning(f"工具配置文件不存在: {args.tool_config}，将使用默认工具列表")
        args.tool_config = None
    
    # 加载数据
    try:
        queries = load_json_data(args.input_file)
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        sys.exit(1)
    
    # 验证数据
    if not validate_input_data(queries):
        sys.exit(1)
    
    # 获取工具配置
    tool_names = get_tool_names_from_config(args.tool_config)
    tools_kwargs = create_tools_kwargs(tool_names)
    
    # 显示示例
    if args.show_sample:
        print_sample_output(queries, tools_kwargs)
    
    # 仅验证模式
    if args.validate_only:
        logger.info("✓ 验证完成，数据格式正确")
        logger.info(f"  共 {len(queries)} 条查询")
        return
    
    # 处理use_expected_tools参数
    use_expected_tools = args.use_expected_tools and not args.no_expected_tools
    
    # 转换并保存
    try:
        convert_and_save(
            queries,
            args.output_file,
            args.data_source,
            args.tool_config,
            use_expected_tools
        )
    except Exception as e:
        logger.error(f"转换失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 打印使用提示
    logger.info("\n" + "=" * 60)
    logger.info("✓ 转换完成！")
    logger.info("=" * 60)
    logger.info(f"输入: {args.input_file}")
    logger.info(f"输出: {args.output_file}")
    logger.info(f"数量: {len(queries)} 条查询")
    logger.info("\n使用提示:")
    logger.info("  1. Skeleton generation:")
    logger.info(f"     ./deep_research/synthesis/run_synthesis_unified.sh \\")
    logger.info(f"         --stage skeleton \\")
    logger.info(f"         --input-file {args.output_file}")
    logger.info("\n  2. Trajectory generation:")
    logger.info(f"     ./deep_research/synthesis/run_synthesis_unified.sh \\")
    logger.info(f"         --stage trajectory \\")
    logger.info(f"         --input-file {args.output_file}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

