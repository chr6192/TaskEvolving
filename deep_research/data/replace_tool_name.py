#!/usr/bin/env python3
"""
数据处理脚本：将parquet数据集中的search工具替换为finance_search
处理两个字段：
1. extra_info.skeleton.tool_calls_reference (JSON字符串)
2. extra_info.tools_kwargs.search (dict)
同时将「无法明确」替换为<|占位符|>
"""

import pandas as pd
import json
import re
from typing import Dict, Any, List


def process_tool_calls_reference(tool_calls_ref: str) -> str:
    """
    处理tool_calls_reference字段：
    1. 解析JSON字符串
    2. 替换search工具名为finance_search
    3. 替换所有工具parameters中的「无法明确」为<|占位符|>
    4. 替换parameters中"依赖于search"为"依赖于finance_search"
    5. 返回处理后的JSON字符串
    """
    try:
        # 解析JSON字符串
        tool_calls = json.loads(tool_calls_ref)
        
        # 遍历所有步骤和工具调用
        for step_info in tool_calls:
            if 'tool' in step_info:
                for tool in step_info['tool']:
                    if tool.get('name') == 'search_and_recommend_investments':
                        tool['name'] = 'recommend_investments'
                    if tool.get('name') == 'search':
                        tool['name'] = 'finance_search'
                    
                    # 处理所有工具的parameters
                    if 'parameters' in tool:
                        parameters_str = json.dumps(tool['parameters'], ensure_ascii=False)
                        parameters_str = parameters_str.replace('search_and_recommend_investments', 'recommend_investments')
                        parameters_str = parameters_str.replace('search', 'finance_search')
                        if '「无法明确」' in parameters_str:
                            parameters_str = parameters_str.replace('「无法明确」', '<|占位符|>')
                        tool['parameters'] = json.loads(parameters_str)
        
        # 返回处理后的JSON字符串
        return json.dumps(tool_calls, ensure_ascii=False)
    
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Error processing tool_calls_reference: {e}")
        print(f"Original content: {tool_calls_ref}")
        return tool_calls_ref

def process_tools_kwargs(tools_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理tools_kwargs字段：将search key改为finance_search
    """
    new_tools_kwargs = tools_kwargs.copy()
    if 'search_and_recommend_investments' in new_tools_kwargs:
        new_tools_kwargs['recommend_investments'] = new_tools_kwargs.pop('search_and_recommend_investments')
    if 'search' in new_tools_kwargs:
        new_tools_kwargs['finance_search'] = new_tools_kwargs.pop('search')
    return new_tools_kwargs

def process_dataset(input_file: str, output_file: str) -> None:
    """
    处理整个数据集
    """
    print(f"Loading dataset from {input_file}...")
    df = pd.read_parquet(input_file)
    print(f"Dataset shape: {df.shape}")
    
    # 统计处理的记录数
    tool_calls_processed = 0
    tools_kwargs_processed = 0
    placeholder_replacements = 0
    dependency_replacements = 0
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"Processing row {idx}/{len(df)}")
        
        extra_info = row['extra_info']
        modified = False
        if 'skeleton' in extra_info and 'analysis' in extra_info['skeleton']:
            analysis = extra_info['skeleton']['analysis']
            if 'search' in analysis:
                if 'search_and_recommend_investments' in analysis:
                    analysis = analysis.replace('search_and_recommend_investments', 'recommend_investments')
                if 'search' in analysis:
                    analysis = analysis.replace('search', 'finance_search')
                    
                extra_info['skeleton']['analysis'] = analysis
                modified = True

        # 处理tool_calls_reference
        if 'skeleton' in extra_info and 'tool_calls_reference' in extra_info['skeleton']:
            original_ref = extra_info['skeleton']['tool_calls_reference']
            processed_ref = process_tool_calls_reference(original_ref)
            
            if processed_ref != original_ref:
                extra_info['skeleton']['tool_calls_reference'] = processed_ref
                tool_calls_processed += 1
                modified = True
                
                # 检查是否有占位符替换
                if '<|占位符|>' in processed_ref and '<|占位符|>' not in original_ref:
                    placeholder_replacements += 1
                
                # 检查是否有依赖关系替换
                if '依赖于finance_search' in processed_ref and '依赖于finance_search' not in original_ref:
                    dependency_replacements += 1
        
        # 处理tools_kwargs
        if 'tools_kwargs' in extra_info:
            original_kwargs = extra_info['tools_kwargs']
            processed_kwargs = process_tools_kwargs(original_kwargs)
            
            if processed_kwargs != original_kwargs:
                extra_info['tools_kwargs'] = processed_kwargs
                tools_kwargs_processed += 1
                modified = True
        
        # 更新DataFrame
        if modified:
            df.at[idx, 'extra_info'] = extra_info
    
    print(f"\nProcessing completed:")
    print(f"- tool_calls_reference processed: {tool_calls_processed}")
    print(f"- tools_kwargs processed: {tools_kwargs_processed}")
    print(f"- placeholder replacements: {placeholder_replacements}")
    print(f"- dependency replacements: {dependency_replacements}")
    
    # 保存处理后的数据
    print(f"\nSaving processed dataset to {output_file}...")
    df.to_parquet(output_file, index=False)
    print("Dataset saved successfully!")
    
    # 验证处理结果
    print("\nVerifying results...")
    verify_processing(output_file)


def verify_processing(file_path: str) -> None:
    """
    验证处理结果
    """
    df = pd.read_parquet(file_path)
    
    search_count = 0
    finance_search_count = 0
    placeholder_count = 0
    dependency_count = 0
    finance_search_in_kwargs = 0
    
    for _, row in df.iterrows():
        extra_info = row['extra_info']
        
        # 检查tool_calls_reference
        if 'skeleton' in extra_info and 'tool_calls_reference' in extra_info['skeleton']:
            tool_calls_ref = extra_info['skeleton']['tool_calls_reference']
            
            # 统计search和finance_search出现次数
            try:
                tool_calls = json.loads(tool_calls_ref)
                for step_info in tool_calls:
                    if 'tool' in step_info:
                        for tool in step_info['tool']:
                            if tool.get('name') == 'search':
                                search_count += 1
                            elif tool.get('name') == 'finance_search':
                                finance_search_count += 1
                            
                            # 检查占位符和依赖关系
                            if 'parameters' in tool:
                                for param_value in tool['parameters'].values():
                                    if isinstance(param_value, str):
                                        if '<|占位符|>' in param_value:
                                            placeholder_count += 1
                                        if '依赖于finance_search' in param_value or re.search(r'依赖step\d+中finance_search', param_value):
                                            dependency_count += 1
            except:
                pass
        
        # 检查tools_kwargs
        if 'tools_kwargs' in extra_info:
            if 'finance_search' in extra_info['tools_kwargs']:
                finance_search_in_kwargs += 1
    
    print(f"Verification results:")
    print(f"- Remaining 'search' tools in tool_calls_reference: {search_count}")
    print(f"- 'finance_search' tools in tool_calls_reference: {finance_search_count}")
    print(f"- Placeholder replacements found: {placeholder_count}")
    print(f"- Dependency replacements found: {dependency_count}")
    print(f"- 'finance_search' keys in tools_kwargs: {finance_search_in_kwargs}")


if __name__ == "__main__":
    input_file = "/Users/chr/Dev/agentic-rl/deep_research/data/4k_w_skeleton/train_250731_complex_clear.parquet"
    output_file = "/Users/chr/Dev/agentic-rl/deep_research/data/4k_w_skeleton/train_250731_complex_clear_replaced_tool_name.parquet"
    
    process_dataset(input_file, output_file)
