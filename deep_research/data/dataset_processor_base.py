"""
数据集处理基类

提供了数据预处理的通用框架，封装了：
- UID加载与采样
- 工具配置加载
- 去重逻辑
- 标准数据结构构建
- Parquet保存

子类只需实现特定的数据读取和解析逻辑。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import pandas as pd
import json
import random
import yaml
import os
from datetime import datetime


class DatasetProcessor(ABC):
    """数据集处理基类
    
    封装了数据处理的通用流程，子类只需实现特定的数据读取和解析逻辑。
    
    使用示例:
        ```python
        class MyProcessor(DatasetProcessor):
            def load_raw_data(self):
                # 实现数据加载
                pass
            
            def parse_raw_item(self, raw_item):
                # 实现数据解析
                pass
            
            def get_data_source_name(self):
                return "my_dataset"
        
        processor = MyProcessor(output_path="output.parquet")
        df = processor.process()
        ```
    """
    
    def __init__(
        self,
        output_path: Optional[str] = None,
        n_uid_samples: int = 1,
        tools_kwargs_empty: bool = False,
        tool_config_path: str = 'deep_research/config/tool_config/dr_tool_config.yaml',
        uid_path: str = 'deep_research/data/uid.jsonl',
        random_seed: Optional[int] = None
    ):
        """
        Args:
            output_path: 输出文件路径，如果为None则不保存
            n_uid_samples: 每个query采样的uid数量
            tools_kwargs_empty: 是否使用空的tools_kwargs
            tool_config_path: 工具配置文件路径
            uid_path: UID列表文件路径
            random_seed: 随机种子（可选），用于结果复现
        """
        self.output_path = output_path or self.get_default_output_path()
        self.n_uid_samples = n_uid_samples
        self.tools_kwargs_empty = tools_kwargs_empty
        self.tool_config_path = tool_config_path
        self.uid_path = uid_path
        
        # 设置随机种子
        if random_seed is not None:
            random.seed(random_seed)
        
        # 初始化
        self.uid_list = self._load_uid_list()
        self.tools_config = self._load_tools_config()
        self.processed_combinations = set()
        self.processed_data = []
    
    # ========== 通用资源加载方法 ==========
    
    def _load_uid_list(self) -> List[str]:
        """加载UID列表"""
        try:
            with open(self.uid_path, 'r', encoding='utf-8') as f:
                uid_list = json.load(f)
            print(f"成功加载 {len(uid_list)} 个UID")
            return uid_list
        except Exception as e:
            print(f"加载UID列表失败: {e}")
            raise
    
    def _load_tools_config(self) -> Dict[str, Any]:
        """加载工具配置"""
        if self.tools_kwargs_empty:
            return self._get_default_tools_config()
        
        try:
            with open(self.tool_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                tools_config = {}
                for tool_item in config.get('tools', []):
                    if 'tool_schema' in tool_item and 'function' in tool_item['tool_schema']:
                        tool_name = tool_item['tool_schema']['function']['name']
                        tools_config[tool_name] = {"dummy_field": None}
                print(f"成功加载工具配置，共 {len(tools_config)} 个工具")
                return tools_config
        except FileNotFoundError:
            print(f"工具配置文件未找到: {self.tool_config_path}，使用默认配置")
            return self._get_default_tools_config()
    
    def _get_default_tools_config(self) -> Dict[str, Any]:
        """获取默认工具配置"""
        return {
            "investment_analysis": {"dummy_field": None},
            "portfolio_allocation": {"dummy_field": None},
            "finance_search": {"dummy_field": None},
            "recommend_investments": {"dummy_field": None},
            "user_info": {"dummy_field": None}
        }
    
    # ========== 去重与采样 ==========
    
    def _is_duplicate(self, query: str, uid: str) -> bool:
        """检查query+uid组合是否重复"""
        combination_key = f"{query}|{uid}"
        if combination_key in self.processed_combinations:
            print(f"跳过重复组合: query='{query[:50]}...' uid='{uid}'")
            return True
        self.processed_combinations.add(combination_key)
        return False
    
    def _sample_uids(self, fixed_uid: Optional[str] = None) -> List[str]:
        """从uid列表中采样
        
        Args:
            fixed_uid: 如果指定，则使用固定的UID（用于某些数据集已经包含UID的情况）
        """
        if fixed_uid:
            return [fixed_uid]
        return random.sample(self.uid_list, min(self.n_uid_samples, len(self.uid_list)))
    
    # ========== 数据构建方法 ==========
    
    def _build_prompt(self, uid: str, query: str, include_prompt: bool = True) -> List[Dict]:
        """构建prompt字段
        
        Args:
            uid: 用户ID
            query: 查询内容
            include_prompt: 是否包含prompt内容（某些数据集为空）
        """
        if not include_prompt:
            return []
        
        from prompt_v0729 import PROMPT
        return [
            {
                "role": "user",
                "content": PROMPT.format(uid=uid, query=query)
            }
        ]
    
    def _build_skeleton(
        self,
        uid: str,
        query: str,
        analysis: str = "",
        intent: str = "",
        need_multi_step: bool = True,
        tool_calls_reference: str = ""
    ) -> Dict:
        """构建skeleton字段"""
        return {
            "uid": uid,
            "query": query,
            "analysis": analysis,
            "intent": intent,
            "need_multi_step": need_multi_step,
            "tool_calls_reference": tool_calls_reference
        }
    
    def _build_data_item(
        self,
        data_source: str,
        uid: str,
        query: str,
        include_prompt: bool = True,
        include_skeleton: bool = True,
        **skeleton_kwargs
    ) -> Dict:
        """构建单条数据记录
        
        Args:
            data_source: 数据源标识
            uid: 用户ID
            query: 查询内容
            include_prompt: 是否包含prompt
            include_skeleton: 是否包含skeleton
            **skeleton_kwargs: skeleton的额外参数（analysis, intent等）
        """
        data_item = {
            "data_source": data_source,
            "prompt": self._build_prompt(uid, query, include_prompt),
            "extra_info": {
                "need_tools_kwargs": not self.tools_kwargs_empty,
                "tools_kwargs": self.tools_config
            }
        }
        
        if include_skeleton:
            data_item["extra_info"]["skeleton"] = self._build_skeleton(
                uid, query, **skeleton_kwargs
            )
        
        return data_item
    
    # ========== 主流程（模板方法） ==========
    
    def process(self) -> pd.DataFrame:
        """主处理流程（模板方法）
        
        Returns:
            处理后的DataFrame
        """
        print(f"\n{'='*60}")
        print(f"开始处理数据集: {self.get_data_source_name()}")
        print(f"{'='*60}\n")
        
        # 1. 读取原始数据
        raw_data = self.load_raw_data()
        print(f"✓ 成功读取原始数据，共 {len(raw_data)} 条\n")
        
        # 2. 处理每条数据
        for idx, raw_item in enumerate(raw_data):
            if (idx + 1) % 100 == 0:
                print(f"  处理进度: {idx + 1}/{len(raw_data)}")
            self._process_single_item(raw_item)
        
        # 3. 后处理（可选）
        self.post_process()
        
        # 4. 保存
        df = pd.DataFrame(self.processed_data)
        
        if self.output_path:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            df.to_parquet(self.output_path, index=False)
            print(f"\n✓ 已保存到 {self.output_path}")
        
        print(f"✓ 共处理 {len(self.processed_data)} 条有效记录")
        print(f"✓ 总共 {len(self.processed_combinations)} 个唯一的query+uid组合")
        print(f"\n{'='*60}\n")
        
        return df
    
    def _process_single_item(self, raw_item: Any):
        """处理单条原始数据
        
        该方法可以被子类override，用于处理特殊情况（如UID已经在数据中）
        """
        # 1. 解析为标准格式
        parsed_items = self.parse_raw_item(raw_item)
        
        # 2. 为每个解析结果采样uid
        for parsed_item in parsed_items:
            query = parsed_item['query']
            
            # 检查是否有固定的UID
            fixed_uid = parsed_item.get('fixed_uid')
            selected_uids = self._sample_uids(fixed_uid)
            
            for uid in selected_uids:
                # 去重检查
                if self._is_duplicate(query, uid):
                    continue
                
                # 构建数据项
                data_item = self._build_data_item(
                    data_source=self.get_data_source_name(),
                    uid=uid,
                    query=query,
                    include_prompt=parsed_item.get('include_prompt', True),
                    include_skeleton=parsed_item.get('include_skeleton', True),
                    analysis=parsed_item.get('analysis', ''),
                    intent=parsed_item.get('intent', ''),
                    need_multi_step=parsed_item.get('need_multi_step', True),
                    tool_calls_reference=parsed_item.get('tool_calls_reference', '')
                )
                
                self.processed_data.append(data_item)
    
    # ========== 子类需要实现的抽象方法 ==========
    
    @abstractmethod
    def load_raw_data(self) -> List[Any]:
        """加载原始数据
        
        Returns:
            原始数据列表，每个元素的格式由子类定义
        """
        pass
    
    @abstractmethod
    def parse_raw_item(self, raw_item: Any) -> List[Dict]:
        """解析单条原始数据
        
        Args:
            raw_item: 单条原始数据
        
        Returns:
            解析后的标准格式列表，每个元素包含：
            {
                'query': str,                      # 必需
                'include_prompt': bool,            # 可选，默认True
                'include_skeleton': bool,          # 可选，默认True
                'fixed_uid': str,                  # 可选，如果数据中已包含UID
                'analysis': str,                   # 可选
                'intent': str,                     # 可选
                'need_multi_step': bool,           # 可选，默认True
                'tool_calls_reference': str        # 可选
            }
        """
        pass
    
    @abstractmethod
    def get_data_source_name(self) -> str:
        """返回数据源名称（用于data_source字段）"""
        pass
    
    # ========== 可选的钩子方法 ==========
    
    def post_process(self):
        """后处理钩子，子类可选实现
        
        例如：随机打乱、数据增强等
        """
        pass
    
    def get_default_output_path(self) -> str:
        """生成默认输出路径"""
        date_suffix = datetime.now().strftime("_%y%m%d")
        return f"deep_research/data/{self.get_data_source_name()}{date_suffix}.parquet"
    
    # ========== 辅助方法 ==========
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return {
            "total_records": len(self.processed_data),
            "unique_combinations": len(self.processed_combinations),
            "data_source": self.get_data_source_name(),
            "output_path": self.output_path
        }

