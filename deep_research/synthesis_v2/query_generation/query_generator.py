"""
Query生成器 - 根据工具环境生成多样化的查询
使用结构化输出一次性生成多个queries
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field
from omegaconf import DictConfig

from ..core.data_structures import QueryData
from ..core.structured_llm_client import StructuredLLMClient
from ..utils.tool_loader import load_tool_schemas_from_yaml
from ..prompts.query_generate_prompt import get_query_generate_prompt, format_tools_for_prompt

logger = logging.getLogger(__name__)


class SingleQuery(BaseModel):
    """单个查询模型"""
    model_config = ConfigDict(extra="ignore")

    content: str = Field(description="查询内容")
    complexity_level: str = Field(description="复杂度级别：simple/medium/high")
    query_type: str = Field(description="查询类型：如选股推荐、信息查询、数据分析等")
    expected_tools: List[str] = Field(description="预期需要使用的工具列表", default_factory=list)
    
class QueriesResponse(BaseModel):
    """
    Query生成的LLM响应模型
    一次性生成多个queries的结构化输出
    """
    model_config = ConfigDict(extra="ignore")

    queries: List[SingleQuery] = Field(
        description="生成的查询列表",
        default_factory=list
    )
    
class QueryGenerator:
    """
    Query生成器
    根据工具环境生成多样化的查询
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.query_gen_config = config.get('query_generation', {})
        
        # 模型配置 - 优先使用环节配置，否则fallback到全局默认
        self.model_name = self.query_gen_config.get('model_name') or config.get('openai_api', {}).get('default_model_name')
        if not self.model_name:
            raise ValueError(
                "Query生成阶段未配置模型！"
                "请在config/data_synthesis.yaml中设置 query_generation.model_name 或 openai_api.default_model_name"
            )
        
        # 生成参数
        self.num_queries_per_batch = self.query_gen_config.get('num_queries_per_batch', 10)
        self.tool_config_path = config.multi_turn.tool_config_path
        
        # LLM参数 - fallback到全局默认值
        openai_config = config.get('openai_api', {})
        self.temperature = self.query_gen_config.get('temperature') or openai_config.get('default_temperature', 0.9)
        self.top_p = self.query_gen_config.get('top_p') or openai_config.get('default_top_p', 0.9)
        self.max_tokens = self.query_gen_config.get('max_tokens') or openai_config.get('default_max_tokens', 4096)
        
        # 结构化LLM客户端，传递query_gen_config以支持完整的LLM参数
        self.llm_client = StructuredLLMClient(
            config, 
            model_name=self.model_name,
            component_config=self.query_gen_config
        )
        
        # Prompt模板
        self.prompt_template: Optional[str] = None
        self.tools_info: List[dict] = []
        
        # 状态标记
        self._initialized = False
        
    async def initialize(self) -> None:
        """初始化query生成器"""
        if self._initialized:
            return
            
        logger.info("初始化Query生成器...")
        logger.info(f"  使用模型: {self.model_name}")
        
        # 加载prompt模板
        self.prompt_template = get_query_generate_prompt()
        
        # 加载工具信息
        try:
            self.tools_info = load_tool_schemas_from_yaml(self.tool_config_path)
            logger.info(f"成功加载 {len(self.tools_info)} 个工具定义")
        except Exception as e:
            logger.error(f"加载工具配置失败: {e}")
            raise
        
        # 初始化结构化LLM客户端
        await self.llm_client.initialize()
        
        self._initialized = True
        logger.info("Query生成器初始化完成")
        
    async def cleanup(self) -> None:
        """清理资源"""
        logger.info("清理Query生成器资源...")
        
        if self.llm_client:
            await self.llm_client.cleanup()
            
        logger.info("Query生成器资源清理完成")
        
    async def generate_queries(self, num_queries: int = None) -> List[QueryData]:
        """
        生成指定数量的多样化查询
        
        Args:
            num_queries: 要生成的查询数量，如果为None则使用配置中的值
            
        Returns:
            生成的QueryData列表
        """
        if not self._initialized:
            raise RuntimeError("Query生成器未初始化")
            
        if num_queries is None:
            num_queries = self.num_queries_per_batch
            
        logger.info(f"开始生成 {num_queries} 个多样化查询")
        
        all_queries = []
        
        # 如果需要生成的查询数量较大，分批生成
        batch_size = self.query_gen_config.get('max_queries_per_call', 20)
        num_batches = (num_queries + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, num_queries - len(all_queries))
            logger.info(f"生成批次 {batch_idx + 1}/{num_batches}，当前批次大小: {current_batch_size}")
            
            try:
                # 调用LLM生成queries
                queries_response = await self._call_llm_for_queries(current_batch_size)
                
                # 转换为QueryData对象
                batch_queries = self._build_query_data_from_response(queries_response)
                all_queries.extend(batch_queries)
                
                logger.info(f"批次 {batch_idx + 1} 成功生成 {len(batch_queries)} 个查询")
                
            except Exception as e:
                logger.error(f"批次 {batch_idx + 1} 生成失败: {e}")
                # 继续尝试下一批次
                continue
        
        logger.info(f"总共成功生成 {len(all_queries)}/{num_queries} 个查询")
        return all_queries
        
    async def _call_llm_for_queries(self, num_queries: int) -> QueriesResponse:
        """
        调用LLM生成queries（使用结构化输出）
        
        Args:
            num_queries: 要生成的查询数量
            
        Returns:
            解析后的QueriesResponse对象
        """
        if not self.llm_client:
            raise RuntimeError("StructuredLLMClient未初始化")
            
        logger.debug(f"调用LLM生成 {num_queries} 个查询")
        
        # 构建prompt
        tools_json = format_tools_for_prompt(self.tools_info)
        prompt = self.prompt_template.format(
            tools_json=tools_json,
            num_queries=num_queries
        )
        
        # 使用结构化LLM客户端调用
        response = await self.llm_client.call_with_structured_output(
            prompt=prompt,
            response_model=QueriesResponse,
            temperature=self.temperature,  # 使用配置的temperature参数
            max_tokens=self.max_tokens  # 使用配置的max_tokens参数
        )
        
        logger.debug(f"成功生成 {len(response.queries)} 个查询")
        
        return response
    
    def _build_query_data_from_response(self, response: QueriesResponse) -> List[QueryData]:
        """
        从结构化响应构建QueryData列表
        
        Args:
            response: LLM返回的结构化响应
            
        Returns:
            QueryData对象列表
        """
        queries = []
        
        for single_query in response.queries:
            query_id = str(uuid.uuid4())
            
            query_data = QueryData(
                query_id=query_id,
                content=single_query.content,
                metadata={
                    "complexity_level": single_query.complexity_level,
                    "query_type": single_query.query_type,
                    "expected_tools": single_query.expected_tools,
                    "generated_by": "query_generator",
                    "model_name": self.model_name,
                    "source": "tool_environment_exploration"
                },
                created_at=datetime.now()
            )
            
            queries.append(query_data)
            
        logger.debug(f"成功构建 {len(queries)} 个QueryData对象")
        return queries
    
    async def batch_generate(self, num_queries_list: List[int]) -> List[List[QueryData]]:
        """
        批量生成多组查询
        
        Args:
            num_queries_list: 每组要生成的查询数量列表
            
        Returns:
            每组生成的QueryData列表的列表
        """
        if not num_queries_list:
            return []
            
        logger.info(f"批量生成 {len(num_queries_list)} 组查询")
        
        # 获取并发控制参数
        max_concurrent = self.query_gen_config.get('max_concurrent_generation', 5)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_with_limit(num_queries, index):
            async with semaphore:
                try:
                    result = await self.generate_queries(num_queries)
                    logger.info(f"完成组 {index+1}/{len(num_queries_list)}，生成 {len(result)} 个查询")
                    return result
                except Exception as e:
                    logger.error(f"处理组 {index+1} 失败: {e}")
                    return []
        
        # 并发生成
        results = await asyncio.gather(
            *[generate_with_limit(num_queries, i) for i, num_queries in enumerate(num_queries_list)],
            return_exceptions=True
        )
        
        # 处理结果
        all_query_groups = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"组 {i+1} 生成失败: {result}")
                all_query_groups.append([])
            else:
                all_query_groups.append(result)
                
        success_count = sum(1 for group in all_query_groups if len(group) > 0)
        logger.info(f"批量生成完成: {success_count}/{len(num_queries_list)} 组成功")
        
        return all_query_groups

