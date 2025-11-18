"""
具体的数据集处理器实现

包含各种数据源的处理器：
- XuanguProcessor: 选股Excel处理
- KefuTextProcessor: 客服文本处理（UID+query在同一行）
- KefuExcelProcessor: 客服Excel处理（包含analysis）
- SkeletonJsonProcessor: 带skeleton的JSON处理
"""

import pandas as pd
import json
from typing import List, Dict, Any
from dataset_processor_base import DatasetProcessor


class XuanguProcessor(DatasetProcessor):
    """选股Excel数据处理器
    
    特点：
    - 从Excel第一列读取queries
    - 不包含prompt
    - 包含空skeleton
    - 随机匹配uid
    
    使用示例:
        processor = XuanguProcessor(
            excel_path="data/xuangu/xuangu1104.xlsx",
            output_path="data/xuangu/xuangu_processed_251104.parquet",
            n_uid_samples=1,
            tools_kwargs_empty=False
        )
        df = processor.process()
    """
    
    def __init__(self, excel_path: str, **kwargs):
        """
        Args:
            excel_path: Excel文件路径
            **kwargs: 传递给基类的其他参数
        """
        self.excel_path = excel_path
        super().__init__(**kwargs)
    
    def load_raw_data(self) -> List[str]:
        """从Excel读取queries"""
        df = pd.read_excel(self.excel_path, sheet_name=0)
        # 获取第一列作为queries，过滤空值
        queries = [
            str(q).strip() 
            for q in df.iloc[:, 0] 
            if pd.notna(q) and str(q).strip()
        ]
        return queries
    
    def parse_raw_item(self, raw_item: str) -> List[Dict]:
        """解析单个query"""
        query = raw_item.strip()
        
        if not query:
            return []
        
        return [{
            'query': query,
            'include_prompt': False,  # 该数据集不包含prompt
            'include_skeleton': True,
            'analysis': '',
            'intent': '',
            'need_multi_step': True,
            'tool_calls_reference': ''
        }]
    
    def get_data_source_name(self) -> str:
        return "xuangu1104_test"


class KefuTextProcessor(DatasetProcessor):
    """客服文本数据处理器（UID+query在同一行）
    
    特点：
    - 从txt文件读取，每行格式为：前16位UID + query
    - 不包含prompt
    - 包含空skeleton
    - UID已在数据中，不需要随机采样
    
    使用示例:
        processor = KefuTextProcessor(
            txt_path="data/kefu0903/47test_data.txt",
            output_path="data/kefu0903/47test_250903.parquet",
            tools_kwargs_empty=False
        )
        df = processor.process()
    """
    
    def __init__(self, txt_path: str, **kwargs):
        """
        Args:
            txt_path: 文本文件路径
            **kwargs: 传递给基类的其他参数
        """
        self.txt_path = txt_path
        super().__init__(**kwargs)
    
    def load_raw_data(self) -> List[str]:
        """从txt文件读取"""
        with open(self.txt_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines
    
    def parse_raw_item(self, raw_item: str) -> List[Dict]:
        """解析UID和query（前16位为UID）"""
        line = raw_item
        
        # 验证格式
        if len(line) < 16:
            print(f"行数据长度不足16位，跳过: {line[:50]}...")
            return []
        
        uid = line[:16]
        query = line[16:].strip()
        
        # 验证UID格式
        if not uid.isdigit():
            print(f"UID格式不正确（非数字），跳过: {uid}")
            return []
        
        if not query:
            print(f"query为空，跳过UID: {uid}")
            return []
        
        return [{
            'query': query,
            'fixed_uid': uid,  # 使用数据中的固定UID
            'include_prompt': False,
            'include_skeleton': True,
            'analysis': '',
            'intent': '',
            'need_multi_step': True,
            'tool_calls_reference': ''
        }]
    
    def get_data_source_name(self) -> str:
        return "kefu0903_47test_data"


class KefuExcelProcessor(DatasetProcessor):
    """客服Excel数据处理器（包含DR建议链路）
    
    特点：
    - 从Excel读取，包含"用户Query"和"DR建议链路"两列
    - 包含prompt
    - 包含skeleton（有analysis字段）
    - 随机匹配uid
    
    使用示例:
        processor = KefuExcelProcessor(
            excel_path="data/kefu/客服DR典型Query.xlsx",
            output_path="data/kefu/test_250823.parquet",
            n_uid_samples=1,
            tools_kwargs_empty=False
        )
        df = processor.process()
    """
    
    def __init__(self, excel_path: str, **kwargs):
        """
        Args:
            excel_path: Excel文件路径
            **kwargs: 传递给基类的其他参数
        """
        self.excel_path = excel_path
        super().__init__(**kwargs)
    
    def load_raw_data(self) -> List[Dict]:
        """从Excel读取"""
        df = pd.read_excel(self.excel_path, sheet_name=0)
        
        # 检查必要字段
        required_columns = ["用户Query（***隔开多轮）", "DR建议链路"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            # 尝试寻找相似的列名
            print(f"警告：缺少必要字段 {missing_columns}")
            print(f"可用的列名: {list(df.columns)}")
            raise ValueError(f"Excel缺少必要字段: {missing_columns}")
        
        return df.to_dict('records')
    
    def parse_raw_item(self, raw_item: Dict) -> List[Dict]:
        """解析包含analysis的Excel行"""
        query = raw_item.get("用户Query（***隔开多轮）")
        analysis = raw_item.get("DR建议链路")
        
        # 跳过空值
        if pd.isna(query) or pd.isna(analysis):
            return []
        
        query = str(query).strip()
        analysis = str(analysis).strip()
        
        if not query or not analysis:
            return []
        
        return [{
            'query': query,
            'include_prompt': True,
            'include_skeleton': True,
            'analysis': analysis,
            'intent': '',
            'need_multi_step': True,
            'tool_calls_reference': ''
        }]
    
    def get_data_source_name(self) -> str:
        return "kefu_31_w_skeleton_test"


class SkeletonJsonProcessor(DatasetProcessor):
    """带skeleton的JSON数据处理器
    
    特点：
    - 从JSON文件读取
    - 包含完整的skeleton信息（analysis, intent, tool_calls_list）
    - 包含prompt
    - 随机匹配uid
    - 使用固定随机种子确保可复现
    
    使用示例:
        processor = SkeletonJsonProcessor(
            json_path="data/4k_w_skeleton/processed_multi_tool_result.json",
            output_path="data/4k_w_skeleton/train_241104.parquet",
            n_uid_samples=1,
            tools_kwargs_empty=False,
            random_seed=42
        )
        df = processor.process()
    """
    
    def __init__(self, json_path: str, **kwargs):
        """
        Args:
            json_path: JSON文件路径
            **kwargs: 传递给基类的其他参数
        """
        self.json_path = json_path
        # 默认使用固定随机种子
        kwargs.setdefault('random_seed', 42)
        super().__init__(**kwargs)
    
    def load_raw_data(self) -> List[Dict]:
        """从JSON文件读取"""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 过滤有效数据：必须包含必要字段
        required_fields = ['query', 'analysis', 'intent', 'need_multi_step', 'tool_calls_list']
        valid_data = [
            item for item in data
            if all(field in item for field in required_fields)
        ]
        
        filtered_count = len(data) - len(valid_data)
        if filtered_count > 0:
            print(f"过滤了 {filtered_count} 条缺少必要字段的数据")
        
        return valid_data
    
    def parse_raw_item(self, raw_item: Dict) -> List[Dict]:
        """解析带skeleton的JSON数据"""
        query = raw_item.get('query')
        if not query:
            return []
        
        return [{
            'query': query,
            'include_prompt': True,
            'include_skeleton': True,
            'analysis': raw_item.get('analysis', ''),
            'intent': raw_item.get('intent', ''),
            'need_multi_step': raw_item.get('need_multi_step', True),
            'tool_calls_reference': json.dumps(
                raw_item.get('tool_calls_list', []), 
                ensure_ascii=False
            )
        }]
    
    def get_data_source_name(self) -> str:
        return "yiwen_processed_4k_w_skeleton_train"
    
    def post_process(self):
        """后处理：随机打乱数据"""
        import random
        random.shuffle(self.processed_data)
        print("✓ 数据已随机打乱")


class SimpleQueryProcessor(DatasetProcessor):
    """简单Query列表处理器
    
    特点：
    - 直接从内存中的query列表读取
    - 包含prompt
    - 不包含skeleton
    - 随机匹配uid
    
    使用示例:
        queries = ["查询1", "查询2", "查询3"]
        processor = SimpleQueryProcessor(
            queries=queries,
            output_path="data/custom/queries_241104.parquet",
            n_uid_samples=1,
            tools_kwargs_empty=False
        )
        df = processor.process()
    """
    
    def __init__(self, queries: List[str], data_source_name: str = "custom_queries", **kwargs):
        """
        Args:
            queries: 查询列表
            data_source_name: 数据源名称
            **kwargs: 传递给基类的其他参数
        """
        self.queries = queries
        self.data_source_name = data_source_name
        super().__init__(**kwargs)
    
    def load_raw_data(self) -> List[str]:
        """直接返回queries"""
        return self.queries
    
    def parse_raw_item(self, raw_item: str) -> List[Dict]:
        """解析单个query"""
        query = raw_item.strip()
        
        if not query:
            return []
        
        return [{
            'query': query,
            'include_prompt': True,
            'include_skeleton': False,  # 不包含skeleton
            'analysis': '',
            'intent': '',
            'need_multi_step': True,
            'tool_calls_reference': ''
        }]
    
    def get_data_source_name(self) -> str:
        return self.data_source_name


# ========== 便捷函数 ==========

def process_xuangu_excel(
    excel_path: str,
    output_path: str = None,
    n_uid_samples: int = 1,
    tools_kwargs_empty: bool = False
) -> pd.DataFrame:
    """处理选股Excel数据的便捷函数"""
    processor = XuanguProcessor(
        excel_path=excel_path,
        output_path=output_path,
        n_uid_samples=n_uid_samples,
        tools_kwargs_empty=tools_kwargs_empty
    )
    return processor.process()


def process_kefu_text(
    txt_path: str,
    output_path: str = None,
    tools_kwargs_empty: bool = False
) -> pd.DataFrame:
    """处理客服文本数据的便捷函数"""
    processor = KefuTextProcessor(
        txt_path=txt_path,
        output_path=output_path,
        tools_kwargs_empty=tools_kwargs_empty
    )
    return processor.process()


def process_kefu_excel(
    excel_path: str,
    output_path: str = None,
    n_uid_samples: int = 1,
    tools_kwargs_empty: bool = False
) -> pd.DataFrame:
    """处理客服Excel数据的便捷函数"""
    processor = KefuExcelProcessor(
        excel_path=excel_path,
        output_path=output_path,
        n_uid_samples=n_uid_samples,
        tools_kwargs_empty=tools_kwargs_empty
    )
    return processor.process()


def process_skeleton_json(
    json_path: str,
    output_path: str = None,
    n_uid_samples: int = 1,
    tools_kwargs_empty: bool = False,
    random_seed: int = 42
) -> pd.DataFrame:
    """处理带skeleton的JSON数据的便捷函数"""
    processor = SkeletonJsonProcessor(
        json_path=json_path,
        output_path=output_path,
        n_uid_samples=n_uid_samples,
        tools_kwargs_empty=tools_kwargs_empty,
        random_seed=random_seed
    )
    return processor.process()


def process_simple_queries(
    queries: List[str],
    output_path: str,
    data_source_name: str = "custom_queries",
    n_uid_samples: int = 1,
    tools_kwargs_empty: bool = False
) -> pd.DataFrame:
    """处理简单query列表的便捷函数"""
    processor = SimpleQueryProcessor(
        queries=queries,
        data_source_name=data_source_name,
        output_path=output_path,
        n_uid_samples=n_uid_samples,
        tools_kwargs_empty=tools_kwargs_empty
    )
    return processor.process()


# ========== 示例代码 ==========

if __name__ == "__main__":
    import os
    
    # 示例1: 处理选股Excel
    print("示例1: 处理选股Excel")
    excel_path = "deep_research/data/xuangu/xuangu_2511_test.xlsx"
    if os.path.exists(excel_path):
        df = process_xuangu_excel(
            excel_path=excel_path,
            output_path="deep_research/data/xuangu/xuangu_2511_test.parquet",
            n_uid_samples=1,
            tools_kwargs_empty=False
        )
        print(f"处理完成，共 {len(df)} 条记录\n")
    else:
        print(f"文件不存在: {excel_path}\n")
    
    # 示例2: 处理客服文本
    # print("示例2: 处理客服文本")
    # df = process_kefu_text(
    #     txt_path="deep_research/data/kefu0903/47test_data.txt",
    #     output_path="deep_research/data/kefu0903/47test_test.parquet",
    #     tools_kwargs_empty=False
    # )
    # print(f"处理完成，共 {len(df)} 条记录\n")
    
    # 示例3: 处理简单query列表
    # print("示例3: 处理简单query列表")
    # queries = [
    #     "今年以来房地产板块下跌比较少的公司",
    #     "易方达增强回报债券A基金购买有限额么？",
    #     "如何选择优质基金？"
    # ]
    # df = process_simple_queries(
    #     queries=queries,
    #     output_path="deep_research/data/custom/test_queries.parquet",
    #     data_source_name="test_queries",
    #     n_uid_samples=1,
    #     tools_kwargs_empty=False
    # )
    # print(f"处理完成，共 {len(df)} 条记录\n")

