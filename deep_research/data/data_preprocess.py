import pandas as pd
import json
import random
from datetime import datetime
import yaml
import os
import sys
from typing import List, Optional

from prompt_v0729 import PROMPT

def preprocess_data(n: int = 1, tools_kwargs_empty: bool = True, output_date_suffix: str = None):
    """
    数据预处理函数
    
    Args:
        n: 控制随机获取uid的数量，当n>1时将数据复制n份
        tools_kwargs_empty: 控制tools_kwargs是否为空
        output_date_suffix: 输出文件日期后缀，如果为None则使用当前日期
    """
    if output_date_suffix is None:
        output_date_suffix = datetime.now().strftime("_%y%m%d")
    
    # 读取uid列表
    with open('deep_research/data/uid.jsonl', 'r', encoding='utf-8') as f:
        uid_list = json.load(f)
    
    # 读取工具配置
    tools_config = {}
    if not tools_kwargs_empty:
        try:
            with open('deep_research/config/tool_config/dr_tool_config.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                # 构建tools_kwargs格式
                for tool_item in config.get('tools', []):
                    if 'tool_schema' in tool_item and 'function' in tool_item['tool_schema']:
                        tool_name = tool_item['tool_schema']['function']['name']
                        tools_config[tool_name] = {"dummy_field": None}
        except FileNotFoundError:
            print("工具配置文件未找到，使用默认search工具")
            tools_config = {"search": {"dummy_field": None}}
    
    # 用于去重的集合，存储query+uid组合
    processed_combinations = set()
    
    # 处理训练和测试数据
    for dataset_type in ['train', 'test']:
        
        if not os.path.exists(input_path):
            print(f"文件 {input_path} 不存在，跳过处理")
            continue
            
        # 读取parquet文件
        df = pd.read_parquet(input_path)
        
        processed_data = []
        
        for _, row in df.iterrows():
            # 随机选择uid
            selected_uids = random.sample(uid_list, min(n, len(uid_list)))
            
            for uid in selected_uids:
                # 先提取query用于去重检查
                query = None
                for msg in row['prompt']:
                    if msg['role'] == 'user':
                        query = msg['content'].split('。')[-1]
                        break
                
                if query is None:
                    continue  # 如果没有找到user消息，跳过
                
                # 创建query+uid组合的唯一标识
                combination_key = f"{query}|{uid}"
                
                # 检查是否已处理过该组合
                if combination_key in processed_combinations:
                    print(f"跳过重复组合: query='{query[:50]}...' uid='{uid}'")
                    continue
                
                # 添加到已处理集合
                processed_combinations.add(combination_key)
                
                # 复制原始数据
                new_row = row.copy()
                
                new_row['extra_info']['uid'] = uid
                # 修改prompt
                prompt = new_row['prompt']
                modified_prompt = []
                
                for msg in prompt:
                    if msg['role'] == 'system':  # 去掉system角色
                        continue
                    elif msg['role'] == 'user':
                        # 在user content末尾添加uid信息
                        # msg['content'] += f"。用户的uid为{uid}。用户的问题为：{query}"
                        msg['content'] = PROMPT.format(uid=uid, query=query)
                        modified_prompt.append(msg)
                        break
                
                new_row['prompt'] = modified_prompt
                
                # 修改tools_kwargs
                if tools_kwargs_empty:
                    new_row['extra_info']['tools_kwargs'] = {}
                    new_row['extra_info']['need_tools_kwargs'] = False
                else:
                    new_row['extra_info']['tools_kwargs'] = tools_config
                    new_row['extra_info']['need_tools_kwargs'] = True
                
                processed_data.append(new_row)
        
        # 保存处理后的数据
        processed_df = pd.DataFrame(processed_data)
        processed_df.to_parquet(output_path, index=False)
        print(f"已保存处理后的{dataset_type}数据到 {output_path}，共 {len(processed_data)} 条记录")
        
    print(f"总共处理了 {len(processed_combinations)} 个唯一的query+uid组合")


def preprocess_skeleton(n: int = 1, tools_kwargs_empty: bool = True, output_date_suffix: str = None):
    """
    数据预处理函数
    
    Args:
        n: 控制随机获取uid的数量，当n>1时将数据复制n份
        tools_kwargs_empty: 控制tools_kwargs是否为空
        output_date_suffix: 输出文件日期后缀，如果为None则使用当前日期
    """
    # 设置固定随机种子，确保结果可复现
    random.seed(42)
    
    if output_date_suffix is None:
        output_date_suffix = datetime.now().strftime("_%y%m%d")
    
    # 读取uid列表
    with open('deep_research/data/uid.jsonl', 'r', encoding='utf-8') as f:
        uid_list = json.load(f)
    
    # 读取工具配置
    tools_config = {}
    if not tools_kwargs_empty:
        try:
            with open('deep_research/config/tool_config/dr_tool_config.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                # 构建tools_kwargs格式
                for tool_item in config.get('tools', []):
                    if 'tool_schema' in tool_item and 'function' in tool_item['tool_schema']:
                        tool_name = tool_item['tool_schema']['function']['name']
                        tools_config[tool_name] = {"dummy_field": None}
        except FileNotFoundError:
            print("工具配置文件未找到，使用默认search工具")
            tools_config = {"search": {"dummy_field": None}}
    
    # 用于去重的集合，存储query+uid组合
    processed_combinations = set()
    
            # 处理训练和测试数据
    for dataset_type in ['train']:
        input_path = f'deep_research/data/4k_w_skeleton/processed_multi_tool_result.json'
        output_path = f'deep_research/data/4k_w_skeleton/{dataset_type}{output_date_suffix}.parquet'
        
        if not os.path.exists(input_path):
            print(f"文件 {input_path} 不存在，跳过处理")
            continue
            
        # 读取json文件
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed_data = []
        
        for row in data:
            if "analysis" not in row or "intent" not in row or "need_multi_step" not in row or "tool_calls_list" not in row:
                continue

            # 随机选择uid
            selected_uids = random.sample(uid_list, min(n, len(uid_list)))
            
            for uid in selected_uids:
                # 先提取query用于去重检查
                query = row['query']
                
                if query is None:
                    continue  # 如果没有找到user消息，跳过
                
                # 创建query+uid组合的唯一标识
                combination_key = f"{query}|{uid}"
                
                # 检查是否已处理过该组合
                if combination_key in processed_combinations:
                    print(f"跳过重复组合: query='{query[:50]}...' uid='{uid}'")
                    continue
                
                # 添加到已处理集合
                processed_combinations.add(combination_key)
                
                # 复制原始数据
                new_row = {}
                new_row['data_source'] = "yiwen_processed_4k_w_skeleton_train"

                new_row['prompt'] = [
                    {
                        "role": "user",
                        "content": PROMPT.format(uid=uid, query=query)
                    }
                ]
                skeleton = {}
                skeleton['uid'] = uid
                skeleton['query'] = query
                skeleton['analysis'] = row['analysis']
                skeleton['intent'] = row['intent']
                skeleton['need_multi_step'] = row['need_multi_step']
                # new_row['extra_info']['original_result'] = row['original_result']
                skeleton['tool_calls_reference'] = json.dumps(row['tool_calls_list'], ensure_ascii=False)
                
                new_row['extra_info'] = {}
                new_row['extra_info']['skeleton'] = skeleton
                
                # 修改tools_kwargs - 确保结构一致性
                if tools_kwargs_empty:
                    new_row['extra_info']['tools_kwargs'] = None  # 使用None而非空字典
                    new_row['extra_info']['need_tools_kwargs'] = False
                else:
                    new_row['extra_info']['tools_kwargs'] = tools_config
                    new_row['extra_info']['need_tools_kwargs'] = True
                
                processed_data.append(new_row)
        
        # 保存处理后的数据 - 规范化数据结构
        processed_df = pd.DataFrame(processed_data)
        
        # 确保所有复杂对象字段类型一致
        for i in range(len(processed_df)):
            # 确保所有None值保持为None，避免混合类型
            if processed_df.iloc[i]['extra_info']['tools_kwargs'] == {}:
                processed_df.iloc[i]['extra_info']['tools_kwargs'] = None
        
        # 对数据集进行随机排序
        processed_df = processed_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        processed_df.to_parquet(output_path, index=False)
        print(f"已保存处理后的{dataset_type}数据到 {output_path}，共 {len(processed_data)} 条记录")
        
    print(f"总共处理了 {len(processed_combinations)} 个唯一的query+uid组合")

def preprocess_queries(n: int = 1, queries: list = None, output_fp: str=None, tools_kwargs_empty: bool = False):
    """
    数据预处理函数
    
    Args:
        n: 控制随机获取uid的数量，当n>1时将数据复制n份
        tools_kwargs_empty: 控制tools_kwargs是否为空
        output_date_suffix: 输出文件日期后缀，如果为None则使用当前日期
    """

    # 读取uid列表
    with open('deep_research/data/uid.jsonl', 'r', encoding='utf-8') as f:
        uid_list = json.load(f)
    
    # 读取工具配置
    tools_config = {}
    if not tools_kwargs_empty:
        try:
            with open('deep_research/config/tool_config/dr_tool_config.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                # 构建tools_kwargs格式
                for tool_item in config.get('tools', []):
                    if 'tool_schema' in tool_item and 'function' in tool_item['tool_schema']:
                        tool_name = tool_item['tool_schema']['function']['name']
                        tools_config[tool_name] = {"dummy_field": None}
        except FileNotFoundError:
            print("工具配置文件未找到，使用默认search工具")
            tools_config = {"search": {"dummy_field": None}}
    
    # 用于去重的集合，存储query+uid组合
    processed_combinations = set()
    

    processed_data = []
    
    for query in queries:
        # 随机选择uid
        selected_uids = random.sample(uid_list, min(n, len(uid_list)))
        
        for uid in selected_uids:

            # 创建query+uid组合的唯一标识
            combination_key = f"{query}|{uid}"
            
            # 检查是否已处理过该组合
            if combination_key in processed_combinations:
                print(f"跳过重复组合: query='{query[:50]}...' uid='{uid}'")
                continue
            
            # 添加到已处理集合
            processed_combinations.add(combination_key)
            
            # 复制原始数据
            new_row = {}
            new_row['extra_info'] = {}
            
            new_row['extra_info']['uid'] = uid

            new_row['prompt'] = [{"role":"user","content": PROMPT.format(uid=uid, query=query)}]
            
            # 修改tools_kwargs
            if tools_kwargs_empty:
                new_row['extra_info']['tools_kwargs'] = {}
                new_row['extra_info']['need_tools_kwargs'] = False
            else:
                new_row['extra_info']['tools_kwargs'] = tools_config
                new_row['extra_info']['need_tools_kwargs'] = True
            
            processed_data.append(new_row)
    
    # 保存处理后的数据
    processed_df = pd.DataFrame(processed_data)
    processed_df.to_parquet(output_fp, index=False)
    print(f"已保存处理后的数据到 {output_fp}，共 {len(processed_data)} 条记录")
    print(f"总共处理了 {len(processed_combinations)} 个唯一的query+uid组合")

def create_custom_dataset(query_list: List[str], output_filename: str = None, n: int = 1, tools_kwargs_empty: bool = True):
    """
    创建自定义数据集
    
    Args:
        query_list: 查询列表
        output_filename: 输出文件名，如果为None则使用默认名称
        n: 控制随机获取uid的数量
        tools_kwargs_empty: 控制tools_kwargs是否为空
    """
    if output_filename is None:
        date_suffix = datetime.now().strftime("_%y%m%d")
        output_filename = f"custom_dataset{date_suffix}.parquet"
    
    # 读取uid列表
    with open('deep_research/uid.jsonl', 'r', encoding='utf-8') as f:
        content = f.read()
        # 将Python列表格式转换为标准JSON格式
        content = content.replace("'", '"')
        uid_list = json.loads(content)
    
    # 读取系统提示词
    with open('deep_research/long_text_2025-06-20-11-25-16.txt', 'r', encoding='utf-8') as f:
        system_content = f.read()
    
    # 读取工具配置
    tools_config = {}
    if not tools_kwargs_empty:
        try:
            with open('/Users/chr/Dev/agentic-rl/deep_research/config/tool_config/dr_tool_config.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                for tool_item in config.get('tools', []):
                    if 'tool_schema' in tool_item and 'function' in tool_item['tool_schema']:
                        tool_name = tool_item['tool_schema']['function']['name']
                        tools_config[tool_name] = {"dummy_field": None}
        except FileNotFoundError:
            print("工具配置文件未找到，使用默认search工具")
            tools_config = {"search": {"dummy_field": None}}
    
    processed_data = []
    
    for idx, query in enumerate(query_list):
        # 随机选择uid
        selected_uids = random.sample(uid_list, min(n, len(uid_list)))
        
        for uid in selected_uids:
            # 构建数据结构
            data_item = {
                "data_source": "custom_dataset",
                "prompt": [
                    {
                        "content": system_content,
                        "role": "system"
                    },
                    {
                        "content": f"{query}当前用户的uid为{uid}",
                        "role": "user"
                    }
                ],
                "ability": "deep_research",
                "reward_model": {
                    "ground_truth": "",
                    "style": "custom"
                },
                "extra_info": {
                    "index": idx,
                    "need_tools_kwargs": True,
                    "question": query,
                    "split": "custom",
                    "tools_kwargs": tools_config if not tools_kwargs_empty else {}
                }
            }
            
            processed_data.append(data_item)
    
    # 保存数据
    df = pd.DataFrame(processed_data)
    df.to_parquet(output_filename, index=False)
    print(f"已保存自定义数据集到 {output_filename}")

def add_skeleton_to_parquet(input_path: str, output_path: str = None):
        """
        为指定路径的parquet文件中的每条数据增加一个空的skeleton字段
        
        Args:
            input_path: 输入parquet文件路径
            output_path: 输出parquet文件路径，如果为None则覆盖原文件
        """
        import pandas as pd
        
        if not os.path.exists(input_path):
            print(f"文件 {input_path} 不存在")
            return
            
        df = pd.read_parquet(input_path)

        def extract_uid_and_query(row):
            # 从extra_info中提取uid
            uid = row.get('extra_info', {}).get('uid', '')
            
            # 从prompt中提取query
            prompt_content = row.get('prompt', [])
            query = ''
            for msg in prompt_content:
                if msg.get('role') == 'user':
                    content = msg.get('content', '')
                    # 提取"当前问题"后的内容
                    if '当前问题：' in content:
                        query = content.split('当前问题：')[-1].strip()
                    elif '当前问题:' in content:
                        query = content.split('当前问题:')[-1].strip()
                    else:
                        # 如果没有明确标记，取最后一段
                        query = content.split('\n')[-1].strip()
                    break
            
            skeleton = {
                'uid': uid,
                'query': query,
                'analysis': '',
                'intent': '',
                'need_multi_step': False,
                'tool_calls_reference': ""
            }
            
            return skeleton
        
        def add_skeleton_and_remove_uid(row):
            extra_info = row.get('extra_info', {}).copy()  # 创建副本避免修改原数据
            
            # 添加skeleton
            skeleton = extract_uid_and_query(row)
            extra_info['skeleton'] = skeleton
            
            # 移除uid
            if isinstance(extra_info, dict) and 'uid' in extra_info:
                extra_info.pop('uid', None)
                
            return extra_info
        
        df['extra_info'] = df.apply(add_skeleton_and_remove_uid, axis=1)

        df['data_source'] = "shiyu_250730_241_test"

        if output_path is None:
            raise ValueError("output_path is required")

        df = df[['data_source', 'prompt', 'extra_info']]

        df.to_parquet(output_path, index=False)
        print(f"已处理完成，共 {len(df)} 条记录，保存到 {output_path}")


def process_kefu0903_to_parquet(txt_path: str = "deep_research/data/kefu0903/47test_data.txt", output_path: str = None, tools_kwargs_empty: bool = False):
    """
    读取kefu0903目录下的47test_data.txt文件，解析UID和query对，转换为parquet格式
    
    Args:
        txt_path: 文本文件路径，默认为kefu0903/47test_data.txt
        output_path: 输出parquet文件路径，如果为None则使用默认路径
        tools_kwargs_empty: 控制tools_kwargs是否为空
    """
    if output_path is None:
        date_suffix = datetime.now().strftime("_%y%m%d")
        output_path = f"deep_research/data/kefu0903/47test_{date_suffix}.parquet"
    
    # 读取文本文件
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"成功读取文本文件，共{len(lines)}行记录")
    except Exception as e:
        print(f"读取文本文件失败: {e}")
        return
    
    # 读取工具配置
    tools_config = {}
    if not tools_kwargs_empty:
        try:
            with open('deep_research/config/tool_config/dr_tool_config.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                # 构建tools_kwargs格式
                for tool_item in config.get('tools', []):
                    if 'tool_schema' in tool_item and 'function' in tool_item['tool_schema']:
                        tool_name = tool_item['tool_schema']['function']['name']
                        tools_config[tool_name] = {"dummy_field": None}
        except FileNotFoundError:
            print("工具配置文件未找到，使用默认工具配置")
            tools_config = {
                "investment_analysis": {"dummy_field": None},
                "portfolio_allocation": {"dummy_field": None},
                "finance_search": {"dummy_field": None},
                "recommend_investments": {"dummy_field": None},
                "user_info": {"dummy_field": None}
            }
    else:
        tools_config = {
            "investment_analysis": {"dummy_field": None},
            "portfolio_allocation": {"dummy_field": None},
            "finance_search": {"dummy_field": None},
            "recommend_investments": {"dummy_field": None},
            "user_info": {"dummy_field": None}
        }
    
    processed_data = []
    processed_combinations = set()
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:  # 跳过空行
            continue
        
        # 解析UID和query：假设UID为16位数字
        if len(line) < 16:
            print(f"第{line_num}行数据长度不足，跳过: {line}")
            continue
        
        # 提取前16位作为UID
        uid = line[:16]
        query = line[16:]
        
        # 验证UID是否为纯数字
        if not uid.isdigit():
            print(f"第{line_num}行UID格式不正确，跳过: {line}")
            continue
        
        # 跳过空query
        if not query.strip():
            print(f"第{line_num}行query为空，跳过: {line}")
            continue
        
        query = query.strip()
        
        # 创建query+uid组合的唯一标识
        combination_key = f"{query}|{uid}"
        
        # 检查是否已处理过该组合
        if combination_key in processed_combinations:
            print(f"跳过重复组合: query='{query[:50]}...' uid='{uid}'")
            continue
        
        # 添加到已处理集合
        processed_combinations.add(combination_key)
        
        # 构建数据结构
        data_item = {
            "data_source": "kefu0903_47test_data",
            "prompt": [
                # {
                #     "content": PROMPT.format(uid=uid, query=query),
                #     "role": "user"
                # }
            ],
            "extra_info": {
                "need_tools_kwargs": not tools_kwargs_empty,
                "skeleton": {
                    "analysis": "",  # prompt为空，所以analysis为空
                    "intent": "",
                    "need_multi_step": True,
                    "query": query,
                    "tool_calls_reference": "",
                    "uid": uid
                },
                "tools_kwargs": tools_config
            }
        }
        
        processed_data.append(data_item)
    
    # 保存处理后的数据
    if processed_data:
        processed_df = pd.DataFrame(processed_data)
        processed_df.to_parquet(output_path, index=False)
        print(f"已保存处理后的数据到 {output_path}，共 {len(processed_data)} 条记录")
        print(f"总共处理了 {len(processed_combinations)} 个唯一的query+uid组合")
    else:
        print("没有有效的数据可以处理")


def process_kefu_excel_to_parquet(excel_path: str, output_path: str = None, n: int = 1, tools_kwargs_empty: bool = False):
    """
    读取Excel文件并转换为目标parquet格式
    
    Args:
        excel_path: Excel文件路径
        output_path: 输出parquet文件路径，如果为None则使用默认路径
        n: 控制随机获取uid的数量
        tools_kwargs_empty: 控制tools_kwargs是否为空
    """
    if output_path is None:
        date_suffix = datetime.now().strftime("_%y%m%d")
        output_path = f"deep_research/data/excel_processed{date_suffix}.parquet"
    
    # 读取Excel文件的第一个sheet
    try:
        df = pd.read_excel(excel_path, sheet_name=0)
        print(f"成功读取Excel文件，共{len(df)}条记录")
    except Exception as e:
        print(f"读取Excel文件失败: {e}")
        return
    
    # 检查必要字段是否存在
    required_columns = ["用户Query（***隔开多轮）", "DR建议链路"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Excel文件缺少必要字段: {missing_columns}")
        print(f"可用字段: {list(df.columns)}")
        return
    
    # 读取uid列表
    try:
        with open('deep_research/data/uid.jsonl', 'r', encoding='utf-8') as f:
            uid_list = json.load(f)
    except Exception as e:
        print(f"读取uid.jsonl失败: {e}")
        return
    
    # 读取工具配置
    tools_config = {}
    if not tools_kwargs_empty:
        try:
            with open('deep_research/config/tool_config/dr_tool_config.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                # 构建tools_kwargs格式
                for tool_item in config.get('tools', []):
                    if 'tool_schema' in tool_item and 'function' in tool_item['tool_schema']:
                        tool_name = tool_item['tool_schema']['function']['name']
                        tools_config[tool_name] = {"dummy_field": None}
        except FileNotFoundError:
            print("工具配置文件未找到，使用默认工具配置")
            tools_config = {
                "investment_analysis": {"dummy_field": None},
                "portfolio_allocation": {"dummy_field": None},
                "finance_search": {"dummy_field": None},
                "recommend_investments": {"dummy_field": None},
                "user_info": {"dummy_field": None}
            }
    else:
        tools_config = {
            "investment_analysis": {"dummy_field": None},
            "portfolio_allocation": {"dummy_field": None},
            "finance_search": {"dummy_field": None},
            "recommend_investments": {"dummy_field": None},
            "user_info": {"dummy_field": None}
        }
    
    processed_data = []
    processed_combinations = set()
    
    for _, row in df.iterrows():
        # 提取query和analysis字段
        query = row["用户Query（***隔开多轮）"]
        analysis = row["DR建议链路"]
        
        # 跳过空值
        if pd.isna(query) or pd.isna(analysis):
            continue
        
        query = str(query).strip()
        analysis = str(analysis).strip()
        
        if not query or not analysis:
            continue
        
        # 随机选择uid
        selected_uids = random.sample(uid_list, min(n, len(uid_list)))
        
        for uid in selected_uids:
            # 创建query+uid组合的唯一标识
            combination_key = f"{query}|{uid}"
            
            # 检查是否已处理过该组合
            if combination_key in processed_combinations:
                print(f"跳过重复组合: query='{query[:50]}...' uid='{uid}'")
                continue
            
            # 添加到已处理集合
            processed_combinations.add(combination_key)
            
            # 构建数据结构
            data_item = {
                "data_source": "kefu_31_w_skeleton_test",
                "prompt": [
                    {
                        "content": PROMPT.format(uid=uid, query=query),
                        "role": "user"
                    }
                ],
                "extra_info": {
                    "need_tools_kwargs": not tools_kwargs_empty,
                    "skeleton": {
                        "analysis": analysis,
                        "intent": "",
                        "need_multi_step": True,
                        "query": query,
                        "tool_calls_reference": "",
                        "uid": uid
                    },
                    "tools_kwargs": tools_config
                }
            }
            
            processed_data.append(data_item)
    
    # 保存处理后的数据
    if processed_data:
        processed_df = pd.DataFrame(processed_data)
        processed_df.to_parquet(output_path, index=False)
        print(f"已保存处理后的数据到 {output_path}，共 {len(processed_data)} 条记录")
        print(f"总共处理了 {len(processed_combinations)} 个唯一的query+uid组合")
    else:
        print("没有有效的数据可以处理")


# 示例使用
if __name__ == "__main__":
    # preprocess_data(n=1, tools_kwargs_empty=False)
   

    ################################0729prompt+ shiyu 730提供的长难query=241条test################################

    # fp = '/ossfs/workspace/dr_test_25_7_30.xlsx'
    # df = pd.read_excel(fp)

    # output_date_suffix = datetime.now().strftime("_%y%m%d")

    # queries = df['query'].tolist()
    # print(f'总共样本数量为{len(queries)}')
    # preprocess_queries(n=1, queries=queries, output_fp=output_fp, tools_kwargs_empty=False)

    # ################################0729prompt+ yiwen处理后的4k条带skeleton = 4k条train################################
    # preprocess_skeleton(n=1, tools_kwargs_empty=False)

     # ################################0729prompt+ 客服场景未解决query 31条，仅有skeleton analysis################################
    # process_kefu_excel_to_parquet(excel_path="deep_research/data/kefu/客服DR典型Query.xlsx", output_path="deep_research/data/kefu/test_250823.parquet", n=1, tools_kwargs_empty=False)
    
    # ################################0903 kefu 47test_data.txt 处理UID和query对################################
    process_kefu0903_to_parquet(txt_path="deep_research/data/kefu0903/47test_data.txt", output_path="deep_research/data/kefu0903/47test_250903.parquet", tools_kwargs_empty=False)
    
    ################################other################################

    # # 创建自定义数据集示例
    # custom_queries = [
    #     "今年以来房地产板块下跌比较少的公司",
    #     "易方达增强回报债券A基金购买有限额么？"
    # ]
    # create_custom_dataset(custom_queries, n=1, tools_kwargs_empty=False)

