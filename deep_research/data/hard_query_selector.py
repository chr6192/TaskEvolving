import asyncio
import json
import os
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from openai import AsyncOpenAI
import logging
from tqdm.asyncio import tqdm
import time
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QueryComplexityClassifier:
    """使用LLM as Judge的方式对query进行复杂度和模糊度分类"""
    
    def __init__(self, api_key: str = None, base_url: str = None, max_concurrent: int = 10, model_name: str = None):
        """
        初始化分类器
        
        Args:
            api_key: OpenAI API密钥，如果为None则从环境变量获取
            base_url: API基础URL，可选
            max_concurrent: 最大并发数
            model_name: 模型名称，如果为None则使用默认模型
        """
        self.client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url
        )
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.model_name = model_name or "qwen3-235b-a22b-instruct-2507"
        
        # 评判prompt模板
        self.evaluation_prompt = """
你是一个专业的查询分析专家。请分析以下查询的复杂度和模糊度，并按照以下标准进行评判：

**复杂度评判标准：**
- 复杂：需要多步推理、涉及多个概念、需要专业知识、或需要综合分析多个信息源
- 简单：直接询问事实、单一概念查询、或可以通过简单搜索回答

**模糊度评判标准：**
- 模糊：查询意图不明确、缺少关键信息、或有多种可能的理解方式
- 非模糊：查询意图明确、信息完整、目标清晰

请分析以下查询：
"{query}"

请严格按照以下JSON格式输出结果：
{{
    "complexity": "复杂" 或 "简单",
    "ambiguity": "模糊" 或 "非模糊",
    "reasoning": "简要说明判断理由"
}}
"""

    async def classify_single_query(self, query: str) -> Dict[str, Any]:
        """
        对单个query进行分类
        
        Args:
            query: 要分析的查询文本
            
        Returns:
            包含分类结果的字典
        """
        async with self.semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一个专业的查询分析专家，能够准确判断查询的复杂度和模糊度。请严格按照JSON格式输出结果。"
                        },
                        {
                            "role": "user", 
                            "content": self.evaluation_prompt.format(query=query)
                        }
                    ],
                    temperature=0.1,
                    max_tokens=200
                )
                
                content = response.choices[0].message.content.strip()
                
                # 尝试解析JSON
                try:
                    result = json.loads(content)
                    return {
                        "complexity": result.get("complexity", "未知"),
                        "ambiguity": result.get("ambiguity", "未知"),
                        "reasoning": result.get("reasoning", ""),
                        "success": True
                    }
                except json.JSONDecodeError:
                    logger.warning(f"JSON解析失败，原始响应: {content}")
                    return {
                        "complexity": "未知",
                        "ambiguity": "未知", 
                        "reasoning": f"解析失败: {content}",
                        "success": False
                    }
                    
            except Exception as e:
                logger.error(f"API调用失败: {str(e)}")
                return {
                    "complexity": "未知",
                    "ambiguity": "未知",
                    "reasoning": f"API调用失败: {str(e)}",
                    "success": False
                }

    async def classify_queries_batch(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        批量分类查询
        
        Args:
            queries: 查询列表
            
        Returns:
            分类结果列表
        """
        tasks = [self.classify_single_query(query) for query in queries]
        results = await tqdm.gather(*tasks, desc="分类查询")
        return results

    def extract_query_from_prompt(self, prompt_data: List[Dict]) -> str:
        """
        从prompt数据中提取实际的用户查询
        
        Args:
            prompt_data: prompt数组数据
            
        Returns:
            提取的查询文本
        """
        try:
            # 查找用户角色的最后一条消息
            user_messages = [msg for msg in prompt_data if msg.get('role') == 'user']
            if user_messages:
                last_user_msg = user_messages[-1]
                content = last_user_msg.get('content', '')
                
                # 尝试从content中提取实际查询
                # 查找"当前问题"部分
                if '当前问题：' in content:
                    query = content.split('当前问题：')[-1].strip()
                    return query
                else:
                    # 如果没有找到标记，返回整个content的前200字符作为查询
                    return content[:200]
            else:
                return "无法提取查询"
        except Exception as e:
            logger.warning(f"提取查询失败: {str(e)}")
            return "提取失败"

    async def process_dataset_batch(self, input_file: str, output_dir: str = None, batch_size: int = 100) -> Tuple[str, str]:
        """
        批量处理数据集，支持大文件的增量处理
        
        Args:
            input_file: 输入的parquet文件路径
            output_dir: 输出目录，如果为None则使用输入文件的目录
            batch_size: 每个批次的大小
            
        Returns:
            (复杂且非模糊数据文件路径, 简单或模糊数据文件路径)
        """
        if output_dir is None:
            output_dir = os.path.dirname(input_file)
            
        # 确保输出目录存在
        Path(output_dir).mkdir(parents=True, exist_ok=True)
            
        # 读取数据
        logger.info(f"读取数据文件: {input_file}")
        df = pd.read_parquet(input_file)
        total_records = len(df)
        logger.info(f"数据集大小: {total_records} 条记录")
        logger.info(f"批次大小: {batch_size}")
        
        # 准备输出文件路径
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        complex_clear_file = os.path.join(output_dir, f"{base_name}_complex_clear.parquet")
        simple_or_ambiguous_file = os.path.join(output_dir, f"{base_name}_simple_or_ambiguous.parquet")
        detailed_results_file = os.path.join(output_dir, f"{base_name}_classification_details.parquet")
        
        # 清理之前的输出文件（如果存在）
        for file_path in [complex_clear_file, simple_or_ambiguous_file, detailed_results_file]:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"清理旧文件: {file_path}")
        
        # 统计变量
        total_complex_clear = 0
        total_simple_or_ambiguous = 0
        total_processed = 0
        
        # 按批次处理数据
        for batch_start in range(0, total_records, batch_size):
            batch_end = min(batch_start + batch_size, total_records)
            batch_df = df.iloc[batch_start:batch_end].copy()
            
            logger.info(f"处理批次 {batch_start//batch_size + 1}/{(total_records-1)//batch_size + 1}: "
                       f"记录 {batch_start+1}-{batch_end} (共 {batch_end-batch_start} 条)")
            
            # 提取查询
            queries = []
            for idx, row in batch_df.iterrows():
                query = self.extract_query_from_prompt(row['prompt'])
                queries.append(query)
            
            # 批量分类
            start_time = time.time()
            classification_results = await self.classify_queries_batch(queries)
            end_time = time.time()
            
            logger.info(f"批次分类耗时: {end_time - start_time:.2f} 秒")
            
            # 添加分类结果到批次数据框
            batch_df['extracted_query'] = queries
            batch_df['complexity'] = [r['complexity'] for r in classification_results]
            batch_df['ambiguity'] = [r['ambiguity'] for r in classification_results]
            batch_df['classification_reasoning'] = [r['reasoning'] for r in classification_results]
            batch_df['classification_success'] = [r['success'] for r in classification_results]
            
            # 分类批次数据
            complex_clear_mask = (batch_df['complexity'] == '复杂') & (batch_df['ambiguity'] == '非模糊')
            simple_or_ambiguous_mask = (batch_df['complexity'] == '简单') | (batch_df['ambiguity'] == '模糊')
            
            complex_clear_batch = batch_df[complex_clear_mask].copy()
            simple_or_ambiguous_batch = batch_df[simple_or_ambiguous_mask].copy()
            
            # 更新统计
            batch_complex_clear = len(complex_clear_batch)
            batch_simple_or_ambiguous = len(simple_or_ambiguous_batch)
            total_complex_clear += batch_complex_clear
            total_simple_or_ambiguous += batch_simple_or_ambiguous
            total_processed += len(batch_df)
            
            logger.info(f"批次结果 - 复杂且非模糊: {batch_complex_clear}, 简单或模糊: {batch_simple_or_ambiguous}")
            logger.info(f"累计结果 - 复杂且非模糊: {total_complex_clear}, 简单或模糊: {total_simple_or_ambiguous}, 总处理: {total_processed}")
            
            # 保存批次结果（追加模式）
            columns_to_keep = ['data_source', 'prompt', 'extra_info']
            
            if len(complex_clear_batch) > 0:
                complex_clear_output = complex_clear_batch[columns_to_keep].copy()
                if os.path.exists(complex_clear_file):
                    # 追加写入
                    existing_df = pd.read_parquet(complex_clear_file)
                    combined_df = pd.concat([existing_df, complex_clear_output], ignore_index=True)
                    combined_df.to_parquet(complex_clear_file, index=False)
                else:
                    # 首次写入
                    complex_clear_output.to_parquet(complex_clear_file, index=False)
                logger.info(f"复杂且非模糊数据已追加到: {complex_clear_file}")
            
            if len(simple_or_ambiguous_batch) > 0:
                simple_or_ambiguous_output = simple_or_ambiguous_batch[columns_to_keep].copy()
                if os.path.exists(simple_or_ambiguous_file):
                    # 追加写入
                    existing_df = pd.read_parquet(simple_or_ambiguous_file)
                    combined_df = pd.concat([existing_df, simple_or_ambiguous_output], ignore_index=True)
                    combined_df.to_parquet(simple_or_ambiguous_file, index=False)
                else:
                    # 首次写入
                    simple_or_ambiguous_output.to_parquet(simple_or_ambiguous_file, index=False)
                logger.info(f"简单或模糊数据已追加到: {simple_or_ambiguous_file}")
            
            # 保存详细分类结果（追加模式）
            if os.path.exists(detailed_results_file):
                existing_df = pd.read_parquet(detailed_results_file)
                combined_df = pd.concat([existing_df, batch_df], ignore_index=True)
                combined_df.to_parquet(detailed_results_file, index=False)
            else:
                batch_df.to_parquet(detailed_results_file, index=False)
            
            logger.info(f"批次 {batch_start//batch_size + 1} 处理完成\n")
        
        # 最终统计
        logger.info("=== 最终处理统计 ===")
        logger.info(f"总处理记录数: {total_processed}")
        logger.info(f"复杂且非模糊数据: {total_complex_clear} 条")
        logger.info(f"简单或模糊数据: {total_simple_or_ambiguous} 条")
        logger.info(f"复杂且非模糊数据文件: {complex_clear_file}")
        logger.info(f"简单或模糊数据文件: {simple_or_ambiguous_file}")
        logger.info(f"详细分类结果文件: {detailed_results_file}")
        
        return complex_clear_file, simple_or_ambiguous_file

async def main(input_file: str = None, output_dir: str = None, model_name: str = None, 
              batch_size: int = None):
    """
    主函数，直接运行批处理模式
    
    Args:
        input_file: 输入文件路径
        output_dir: 输出目录
        model_name: 模型名称
        batch_size: 批次大小
    """
    # 使用默认配置参数
    final_input_file = input_file or "/Users/chr/Dev/agentic-rl/deep_research/data/4k_w_skeleton/train_250731.parquet"
    final_output_dir = output_dir
    final_model_name = model_name or "qwen3-235b-a22b-instruct-2507"
    final_batch_size = batch_size or 100
    max_concurrent = 15
    
    logger.info(f"运行参数:")
    logger.info(f"  输入文件: {final_input_file}")
    logger.info(f"  输出目录: {final_output_dir}")
    logger.info(f"  模型名称: {final_model_name}")
    logger.info(f"  批次大小: {final_batch_size}")
    logger.info(f"  最大并发数: {max_concurrent}")
    
    # 创建分类器
    classifier = QueryComplexityClassifier(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", 
        max_concurrent=max_concurrent,
        model_name=final_model_name
    )

    try:
        # 批处理模式处理数据集
        complex_file, simple_file = await classifier.process_dataset_batch(
            final_input_file, final_output_dir, final_batch_size
        )
        
        print(f"\n✅ 数据筛选完成！")
        print(f"复杂且非模糊数据: {complex_file}")
        print(f"简单或模糊数据: {simple_file}")
        
        return complex_file, simple_file
        
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        raise

def run_batch_processing(input_file: str, output_dir: str = None, model_name: str = None, batch_size: int = 100):
    """
    外部调用接口：批处理模式
    
    Args:
        input_file: 输入文件路径
        output_dir: 输出目录
        model_name: 模型名称
        batch_size: 批次大小
        
    Returns:
        (复杂且非模糊数据文件路径, 简单或模糊数据文件路径)
    """
    return asyncio.run(main(
        input_file=input_file,
        output_dir=output_dir, 
        model_name=model_name,
        batch_size=batch_size
    ))


if __name__ == "__main__":
    asyncio.run(main())
