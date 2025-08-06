"""
数据加载器 - 加载和保存查询数据
"""

import json
import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any
from ..core.data_structures import QueryData, SkeletonData, StepData, ToolCall

logger = logging.getLogger(__name__)


def load_skeleton_data_from_parquet(file_path: str) -> List[SkeletonData]:
    """
    从parquet文件读取skeleton数据并转换为标准格式
    
    Args:
        file_path: parquet文件路径
        
    Returns:
        SkeletonData列表
    """
    import pandas as pd
    
    try:
        logger.info(f"从 {file_path} 读取skeleton数据")
        
        # 读取parquet文件
        df = pd.read_parquet(file_path)
        skeleton_data_list = []
        
        for idx, row in df.iterrows():
            try:
                # 解析extra_info中的skeleton信息
                extra_info = row['extra_info']
                skeleton_info = extra_info['skeleton']
                prompt = row['prompt']
                
                # 提取基本信息
                query_content = skeleton_info.get('query', '')
                analysis_logic = skeleton_info.get('analysis', '')
                multi_step = skeleton_info.get('need_multi_step', False)
                uid = skeleton_info.get('uid', '')
                
                # 解析tool_calls_reference如果有的话
                tool_results = []
                tool_calls_reference = skeleton_info.get('tool_calls_reference', '')
                
                if tool_calls_reference and multi_step:
                    # 如果有tool_calls_reference，解析成tool_results
                    tool_results = _parse_tool_calls_from_reference(tool_calls_reference, extra_info.get('tools_kwargs', {}))
                else:
                    # 默认创建一个基础的搜索步骤
                    default_tool_call = ToolCall(
                        name="search",
                        parameters={
                            "pool": ["互联网"],
                            "query": query_content,
                            "entity": "相关主体"
                        }
                    )
                    tool_results = [StepData(
                        step=0,
                        tool_calls=[default_tool_call]
                    )]
                
                # 创建SkeletonData
                skeleton = SkeletonData(
                    skeleton_id=str(uuid.uuid4()),
                    query_id=uid or str(uuid.uuid4()),
                    query_content=query_content,
                    analysis_logic=analysis_logic or f"分析查询: {query_content}",
                    multi_step=multi_step,
                    tool_results=tool_results,
                    metadata={
                        'data_source': row.get('data_source', ''),
                        'tools_kwargs': extra_info.get('tools_kwargs', {}),
                        'original_prompt': prompt,
                        'intent': skeleton_info.get('intent', ''),
                        'need_tools_kwargs': extra_info.get('need_tools_kwargs', False)
                    }
                )
                
                skeleton_data_list.append(skeleton)
                
            except Exception as e:
                logger.error(f"解析第 {idx} 行skeleton数据失败: {e}")
                continue
                
        logger.info(f"成功读取 {len(skeleton_data_list)} 个skeleton数据")
        return skeleton_data_list
        
    except Exception as e:
        logger.error(f"读取parquet文件失败: {e}")
        return []


def _parse_tool_calls_from_reference(tool_calls_reference: str, tools_kwargs: Dict[str, Any]) -> List[StepData]:
    """
    从tool_calls_reference解析工具调用步骤
    
    Args:
        tool_calls_reference: 工具调用引用字符串
        tools_kwargs: 工具参数配置
        
    Returns:
        StepData列表
    """
    try:
        # 尝试解析tool_calls_reference为JSON
        if tool_calls_reference.strip():
            try:
                import json
                tool_calls_data = json.loads(tool_calls_reference)
                
                if isinstance(tool_calls_data, list):
                    tool_results = []
                    for i, step_data in enumerate(tool_calls_data):
                        if isinstance(step_data, dict) and 'tool_calls' in step_data:
                            tool_calls = []
                            for tool_call_data in step_data['tool_calls']:
                                tool_call = ToolCall(
                                    name=tool_call_data.get('name', 'search'),
                                    parameters=tool_call_data.get('parameters', {})
                                )
                                tool_calls.append(tool_call)
                            
                            step = StepData(
                                step=i,
                                tool_calls=tool_calls,
                                dependencies=step_data.get('dependencies', []),
                                placeholder_values=step_data.get('placeholder_values', {})
                            )
                            tool_results.append(step)
                    
                    return tool_results
                    
            except json.JSONDecodeError:
                logger.warning(f"tool_calls_reference解析为JSON失败，使用默认逻辑")
                
    except Exception as e:
        logger.error(f"解析tool_calls_reference失败: {e}")
        
    # 返回默认的工具调用步骤
    return []


def load_queries_from_file(file_path: str) -> List[QueryData]:
    """
    从文件加载查询数据
    
    Args:
        file_path: 查询文件路径，支持.jsonl和.json格式
        
    Returns:
        查询数据列表
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"查询文件不存在: {file_path}")
    
    logger.info(f"从文件加载查询数据: {file_path}")
    
    queries = []
    
    try:
        if file_path.suffix.lower() == '.jsonl':
            # JSONL格式：每行一个JSON对象
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        data = json.loads(line)
                        query = _parse_query_data(data, f"line_{line_num}")
                        if query:
                            queries.append(query)
                    except json.JSONDecodeError as e:
                        logger.warning(f"跳过无效JSON行 {line_num}: {e}")
                        continue
                        
        elif file_path.suffix.lower() == '.json':
            # JSON格式：完整的JSON数组或对象
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                # JSON数组格式
                for i, item in enumerate(data):
                    query = _parse_query_data(item, f"item_{i}")
                    if query:
                        queries.append(query)
            elif isinstance(data, dict):
                # 单个JSON对象
                query = _parse_query_data(data, "single_query")
                if query:
                    queries.append(query)
            else:
                raise ValueError(f"不支持的JSON格式: {type(data)}")
                
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
            
        logger.info(f"成功加载 {len(queries)} 个查询")
        return queries
        
    except Exception as e:
        logger.error(f"加载查询文件失败: {e}")
        raise


def save_queries_to_file(queries: List[QueryData], file_path: str) -> None:
    """
    保存查询数据到文件
    
    Args:
        queries: 查询数据列表
        file_path: 输出文件路径
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"保存 {len(queries)} 个查询到文件: {file_path}")
    
    try:
        if file_path.suffix.lower() == '.jsonl':
            # JSONL格式
            with open(file_path, 'w', encoding='utf-8') as f:
                for query in queries:
                    json.dump(query.to_dict(), f, ensure_ascii=False)
                    f.write('\n')
                    
        elif file_path.suffix.lower() == '.json':
            # JSON格式
            with open(file_path, 'w', encoding='utf-8') as f:
                data = [query.to_dict() for query in queries]
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
            
        logger.info(f"查询数据保存成功: {file_path}")
        
    except Exception as e:
        logger.error(f"保存查询文件失败: {e}")
        raise


def _parse_query_data(data: Dict[str, Any], default_id: str) -> QueryData:
    """
    解析单个查询数据
    
    Args:
        data: 原始数据字典
        default_id: 默认ID（如果数据中没有提供）
        
    Returns:
        QueryData对象，解析失败时返回None
    """
    try:
        # 提取查询内容
        content = None
        
        # 支持多种可能的字段名
        content_fields = ['content', 'query', 'question', 'text', 'prompt']
        for field in content_fields:
            if field in data and data[field]:
                content = str(data[field]).strip()
                break
                
        if not content:
            logger.warning(f"查询内容为空，跳过: {data}")
            return None
            
        # 提取或生成ID
        query_id = data.get('id') or data.get('query_id') or str(uuid.uuid4())
        
        # 提取元数据
        metadata = {}
        excluded_fields = {'id', 'query_id', 'content', 'query', 'question', 'text', 'prompt'}
        for key, value in data.items():
            if key not in excluded_fields:
                metadata[key] = value
                
        # 创建QueryData对象
        query = QueryData(
            query_id=str(query_id),
            content=content,
            metadata=metadata
        )
        
        return query
        
    except Exception as e:
        logger.error(f"解析查询数据失败: {e}, 数据: {data}")
        return None


def create_sample_queries_file(output_path: str, num_samples: int = 10) -> None:
    """
    创建示例查询文件，用于测试
    
    Args:
        output_path: 输出文件路径
        num_samples: 生成的示例数量
    """
    sample_queries = [
        "帮我分析一下2024年Q2家电行业利润环比上升的公司有哪些，并给出投资建议",
        "最近哪些新能源汽车股票表现较好？请分析一下原因",
        "我有10万元资金，想投资基金，请推荐一些稳健型的产品",
        "请分析一下当前A股市场的整体走势和投资机会",
        "比较一下银行理财和货币基金的收益和风险",
        "医药行业近期有哪些投资机会？请详细分析",
        "如何构建一个均衡的投资组合？",
        "请分析一下美联储加息对中国股市的影响",
        "科技股是否还有投资价值？请给出具体建议",
        "房地产行业的投资前景如何？"
    ]
    
    queries = []
    for i in range(min(num_samples, len(sample_queries))):
        query = QueryData(
            query_id=f"sample_{i+1}",
            content=sample_queries[i],
            metadata={
                "source": "sample_data",
                "category": "financial_analysis", 
                "difficulty": "medium"
            }
        )
        queries.append(query)
        
    save_queries_to_file(queries, output_path)
    logger.info(f"创建了包含 {len(queries)} 个示例查询的文件: {output_path}")


if __name__ == "__main__":
    # 创建示例文件用于测试
    create_sample_queries_file("deep_research/data/synthesis_queries.jsonl") 