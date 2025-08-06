"""
Skeleton生成器 - 阶段一框架实现
"""

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import List, Optional

from omegaconf import DictConfig

from .data_structures import QueryData, SkeletonData, StepData, ToolCall

logger = logging.getLogger(__name__)


class SkeletonGenerator:
    """
    生成带有递进逻辑的agent tool calls skeleton
    
    阶段一框架实现 - 核心逻辑待实现
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.skeleton_config = config.skeleton
        
        # 模型相关配置
        self.model_name = config.model.name
        self.model_path = config.model.path
        
        # Prompt模板
        self.prompt_template_path = self.skeleton_config.prompt_template_path
        self.prompt_template: Optional[str] = None
        
        # 生成参数
        self.max_steps = self.skeleton_config.max_steps
        self.validation_enabled = self.skeleton_config.validation_enabled
        
        # LLM客户端 (待初始化)
        self.llm_client = None
        
        # 状态标记
        self._initialized = False
        
    async def initialize(self) -> None:
        """初始化skeleton生成器"""
        if self._initialized:
            return
            
        logger.info("初始化Skeleton生成器...")
        
        # 加载prompt模板
        await self._load_prompt_template()
        
        # 初始化LLM客户端
        await self._initialize_llm_client()
        
        self._initialized = True
        logger.info("Skeleton生成器初始化完成")
        
    async def cleanup(self) -> None:
        """清理资源"""
        logger.info("清理Skeleton生成器资源...")
        
        if self.llm_client:
            # TODO: 清理LLM客户端资源
            pass
            
        logger.info("Skeleton生成器资源清理完成")
        
    async def generate_skeleton(self, query: QueryData) -> Optional[SkeletonData]:
        """
        为单个查询生成skeleton
        
        Args:
            query: 输入查询
            
        Returns:
            生成的skeleton，失败时返回None
        """
        if not self._initialized:
            raise RuntimeError("Skeleton生成器未初始化")
            
        logger.debug(f"为查询 {query.query_id} 生成skeleton")
        
        try:
            # TODO: 阶段一具体实现 - 调用LLM生成skeleton
            skeleton_json = await self._call_llm_for_skeleton(query)
            
            # 解析生成结果
            skeleton = await self._parse_skeleton_response(query, skeleton_json)
            
            # 验证skeleton
            if self.validation_enabled and skeleton and not skeleton.validate():
                logger.warning(f"生成的skeleton验证失败: {query.query_id}")
                return None
                
            return skeleton
            
        except Exception as e:
            logger.error(f"生成skeleton失败 {query.query_id}: {e}")
            return None
            
    async def batch_generate(self, queries: List[QueryData]) -> List[Optional[SkeletonData]]:
        """
        批量生成skeleton
        
        Args:
            queries: 输入查询列表
            
        Returns:
            生成的skeleton列表，失败的项为None
        """
        logger.info(f"批量生成 {len(queries)} 个skeleton")
        
        # 并发生成skeleton
        tasks = [self.generate_skeleton(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        skeletons = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"查询 {queries[i].query_id} 生成失败: {result}")
                skeletons.append(None)
            else:
                skeletons.append(result)
                
        success_count = sum(1 for s in skeletons if s is not None)
        logger.info(f"批量生成完成: {success_count}/{len(queries)} 个成功")
        
        return skeletons
        
    async def _load_prompt_template(self) -> None:
        """加载prompt模板"""
        try:
            template_path = Path(self.prompt_template_path)
            if template_path.exists():
                with open(template_path, 'r', encoding='utf-8') as f:
                    self.prompt_template = f.read()
                logger.info(f"加载prompt模板: {template_path}")
            else:
                logger.warning(f"Prompt模板文件不存在: {template_path}")
                # 使用默认模板
                self.prompt_template = self._get_default_prompt_template()
        except Exception as e:
            logger.error(f"加载prompt模板失败: {e}")
            self.prompt_template = self._get_default_prompt_template()
            
    def _get_default_prompt_template(self) -> str:
        """获取默认prompt模板"""
        return """你是一个专业的金融分析师助手，需要为用户查询设计递进式的工具调用分析框架。

请根据用户的查询，生成一个包含递进逻辑的分析思路和工具调用骨架(skeleton)。

用户查询：{query}

请输出JSON格式的分析框架：
```json
{
    "分析思路": "详细描述分析的递进逻辑和思路",
    "是否需要多步": "是/否",
    "工具结果": []
}
```"""
        
    async def _initialize_llm_client(self) -> None:
        """初始化LLM客户端"""
        # TODO: 阶段一具体实现 - 初始化LLM客户端
        # 可以复用现有的chat_scheduler或创建新的客户端
        logger.info("初始化LLM客户端 (待实现)")
        pass
        
    async def _call_llm_for_skeleton(self, query: QueryData) -> str:
        """
        调用LLM生成skeleton
        
        Args:
            query: 输入查询
            
        Returns:
            LLM生成的JSON字符串
        """
        # TODO: 阶段一具体实现 - 调用LLM生成skeleton
        logger.debug("调用LLM生成skeleton (待实现)")
        
        # 构建prompt
        prompt = self.prompt_template.format(query=query.content)
        
        # 调用LLM
        # response = await self.llm_client.generate(prompt)
        
        # 临时返回示例结果
        example_skeleton = {
            "分析思路": f"针对查询'{query.content}'的分析框架",
            "是否需要多步": "是",
            "工具结果": [
                {
                    "step": 0,
                    "tool_calls": [
                        {
                            "name": "search",
                            "parameters": {
                                "pool": ["互联网"],
                                "query": query.content,
                                "entity": "相关主体"
                            }
                        }
                    ]
                }
            ]
        }
        
        return json.dumps(example_skeleton, ensure_ascii=False)
        
    async def _parse_skeleton_response(self, query: QueryData, skeleton_json: str) -> Optional[SkeletonData]:
        """
        解析skeleton生成响应
        
        Args:
            query: 原始查询
            skeleton_json: LLM返回的JSON字符串
            
        Returns:
            解析后的SkeletonData，失败时返回None
        """
        try:
            # 解析JSON
            skeleton_dict = json.loads(skeleton_json)
            
            # 提取字段
            analysis_logic = skeleton_dict.get("分析思路", "")
            multi_step = skeleton_dict.get("是否需要多步", "否") == "是"
            tool_results_raw = skeleton_dict.get("工具结果", [])
            
            # 构建tool_results
            tool_results = []
            dependencies = {}
            
            for step_data in tool_results_raw:
                step = step_data.get("step", 0)
                tool_calls_raw = step_data.get("tool_calls", [])
                
                # 构建tool_calls
                tool_calls = []
                step_dependencies = []
                placeholder_values = {}
                
                for tool_call_raw in tool_calls_raw:
                    tool_name = tool_call_raw.get("name", "")
                    parameters = tool_call_raw.get("parameters", {})
                    
                    # 检查依赖关系和占位符
                    for param_key, param_value in parameters.items():
                        if isinstance(param_value, str) and "「无法明确」依赖" in param_value:
                            step_dependencies.append(param_value)
                            placeholder_values[f"{tool_name}.{param_key}"] = param_value
                    
                    tool_call = ToolCall(name=tool_name, parameters=parameters)
                    tool_calls.append(tool_call)
                
                step_data_obj = StepData(
                    step=step,
                    tool_calls=tool_calls,
                    dependencies=step_dependencies,
                    placeholder_values=placeholder_values
                )
                tool_results.append(step_data_obj)
                
                if step_dependencies:
                    dependencies[str(step)] = step_dependencies
            
            # 创建SkeletonData
            skeleton = SkeletonData(
                skeleton_id=str(uuid.uuid4()),
                query_id=query.query_id,
                query_content=query.content,
                analysis_logic=analysis_logic,
                multi_step=multi_step,
                tool_results=tool_results,
                dependencies=dependencies,
                metadata={"generated_by": "skeleton_generator"}
            )
            
            return skeleton
            
        except Exception as e:
            logger.error(f"解析skeleton响应失败: {e}")
            return None 