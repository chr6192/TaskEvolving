"""
将query_generation阶段生成的数据转换为skeleton generation和trajectory generation能够接受的格式

支持的输入格式:
- JSON文件 (.json)
- JSONL文件 (.jsonl)

支持的输出格式:
- JSONL文件 (.jsonl) - 默认
- Parquet文件 (.parquet)

用法示例:
    python convert_query_to_synthesis_format.py input.json output.jsonl
    python convert_query_to_synthesis_format.py input.json output.parquet --format parquet
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


def load_queries_from_file(input_file: Path) -> List[Dict[str, Any]]:
    """
    从文件加载查询数据
    
    Args:
        input_file: 输入文件路径
        
    Returns:
        查询数据列表
    """
    print(f"正在读取输入文件: {input_file}")
    
    if not input_file.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    suffix = input_file.suffix.lower()
    
    if suffix == '.json':
        # JSON格式 - 假设是查询数组
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                queries = data
            elif isinstance(data, dict) and 'queries' in data:
                queries = data['queries']
            else:
                raise ValueError("JSON文件格式不正确，需要是查询数组或包含'queries'字段的对象")
    
    elif suffix == '.jsonl':
        # JSONL格式 - 每行一个查询
        queries = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    query = json.loads(line)
                    queries.append(query)
                except json.JSONDecodeError as e:
                    print(f"警告: 第{line_num}行JSON解析失败，跳过: {e}")
                    continue
    
    else:
        raise ValueError(f"不支持的输入文件格式: {suffix}，仅支持 .json 和 .jsonl")
    
    print(f"成功读取 {len(queries)} 条查询数据")
    return queries


def validate_and_convert_query(query: Dict[str, Any]) -> Dict[str, Any]:
    """
    验证并转换查询数据为synthesis pipeline所需格式
    
    Args:
        query: 原始查询数据
        
    Returns:
        转换后的查询数据
    """
    # 检查必需字段
    if 'query_id' not in query:
        raise ValueError("查询数据缺少必需字段: query_id")
    if 'content' not in query:
        raise ValueError("查询数据缺少必需字段: content")
    
    # 构建标准格式
    converted = {
        'query_id': query['query_id'],
        'content': query['content'],
        'metadata': query.get('metadata', {}),
        'rubrics': query.get('rubrics', None),
        'created_at': query.get('created_at', datetime.now().isoformat())
    }
    
    # 确保created_at是ISO格式字符串
    if isinstance(converted['created_at'], str):
        # 验证是否是有效的ISO格式
        try:
            datetime.fromisoformat(converted['created_at'].replace('Z', '+00:00'))
        except ValueError:
            # 如果不是有效格式，使用当前时间
            converted['created_at'] = datetime.now().isoformat()
    else:
        converted['created_at'] = datetime.now().isoformat()
    
    return converted


def save_queries_to_jsonl(queries: List[Dict[str, Any]], output_file: Path) -> None:
    """
    保存查询数据为JSONL格式
    
    Args:
        queries: 查询数据列表
        output_file: 输出文件路径
    """
    print(f"正在保存为JSONL格式: {output_file}")
    
    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for query in queries:
            json_line = json.dumps(query, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"成功保存 {len(queries)} 条查询到 {output_file}")


def save_queries_to_parquet(queries: List[Dict[str, Any]], output_file: Path) -> None:
    """
    保存查询数据为Parquet格式
    
    Args:
        queries: 查询数据列表
        output_file: 输出文件路径
    """
    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError(
            "保存为Parquet格式需要安装pandas和pyarrow: "
            "pip install pandas pyarrow"
        ) from e
    
    print(f"正在保存为Parquet格式: {output_file}")
    
    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 转换为DataFrame
    df = pd.DataFrame(queries)
    
    # 将嵌套的字典/列表字段转换为JSON字符串（Parquet要求）
    for col in df.columns:
        if df[col].dtype == 'object':
            # 检查是否包含字典或列表
            sample = df[col].iloc[0] if len(df) > 0 else None
            if isinstance(sample, (dict, list)):
                df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if x is not None else None)
    
    # 保存为Parquet
    df.to_parquet(output_file, engine='pyarrow', index=False)
    
    print(f"成功保存 {len(queries)} 条查询到 {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='将query_generation数据转换为synthesis pipeline格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 转换JSON到JSONL
  python convert_query_to_synthesis_format.py queries.json queries.jsonl
  
  # 转换JSON到Parquet
  python convert_query_to_synthesis_format.py queries.json queries.parquet --format parquet
  
  # 转换JSONL到JSONL
  python convert_query_to_synthesis_format.py queries.jsonl output.jsonl
        """
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        help='输入文件路径 (.json 或 .jsonl)'
    )
    
    parser.add_argument(
        'output_file',
        type=str,
        help='输出文件路径 (.jsonl 或 .parquet)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['jsonl', 'parquet'],
        default=None,
        help='输出格式 (默认根据输出文件扩展名自动判断)'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='仅验证数据格式，不保存输出文件'
    )
    
    args = parser.parse_args()
    
    # 解析文件路径
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    
    # 确定输出格式
    if args.format:
        output_format = args.format
    else:
        output_suffix = output_file.suffix.lower()
        if output_suffix == '.jsonl':
            output_format = 'jsonl'
        elif output_suffix == '.parquet':
            output_format = 'parquet'
        else:
            print(f"错误: 无法从文件扩展名判断输出格式: {output_suffix}")
            print("请使用 --format 参数指定输出格式 (jsonl 或 parquet)")
            sys.exit(1)
    
    try:
        # 1. 加载输入数据
        queries = load_queries_from_file(input_file)
        
        if not queries:
            print("警告: 没有找到任何查询数据")
            sys.exit(0)
        
        # 2. 验证并转换数据
        print(f"正在验证和转换 {len(queries)} 条查询...")
        converted_queries = []
        failed_count = 0
        
        for i, query in enumerate(queries):
            try:
                converted = validate_and_convert_query(query)
                converted_queries.append(converted)
            except Exception as e:
                print(f"警告: 第{i+1}条查询转换失败，跳过: {e}")
                failed_count += 1
                continue
        
        print(f"成功转换 {len(converted_queries)} 条查询")
        if failed_count > 0:
            print(f"失败 {failed_count} 条查询")
        
        if not converted_queries:
            print("错误: 没有成功转换的查询数据")
            sys.exit(1)
        
        # 3. 保存输出（如果不是仅验证模式）
        if args.validate_only:
            print("\n✓ 数据验证通过（仅验证模式，未保存文件）")
        else:
            if output_format == 'jsonl':
                save_queries_to_jsonl(converted_queries, output_file)
            elif output_format == 'parquet':
                save_queries_to_parquet(converted_queries, output_file)
            
            print(f"\n✓ 转换完成！")
            print(f"  输入: {input_file}")
            print(f"  输出: {output_file}")
            print(f"  格式: {output_format.upper()}")
            print(f"  数量: {len(converted_queries)} 条查询")
        
        # 4. 显示使用提示
        if not args.validate_only:
            print(f"\n📝 使用提示:")
            print(f"  - Skeleton generation: 在配置文件中设置 input.queries_file: '{output_file}'")
            print(f"  - Trajectory generation: 同样使用上述配置")
            print(f"  - 可以通过命令行参数覆盖: --data.input.queries_file='{output_file}'")
        
    except Exception as e:
        print(f"\n错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

