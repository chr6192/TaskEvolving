

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

# 示例使用
if __name__ == "__main__":
    # 处理原始数据
    preprocess_data(n=1, tools_kwargs_empty=False)
    
    # # 创建自定义数据集示例
    # custom_queries = [
    #     "今年以来房地产板块下跌比较少的公司",
    #     "易方达增强回报债券A基金购买有限额么？"
    # ]
    # create_custom_dataset(custom_queries, n=1, tools_kwargs_empty=False)
