#!/usr/bin/env python3
"""
将JSONL格式的对话轨迹数据转换为Excel格式，便于人类标注
"""

import json
import pandas as pd
import argparse
from pathlib import Path
from typing import List, Dict, Any
import re


def extract_messages_info(messages: List[Dict]) -> Dict[str, Any]:
    """提取对话消息的关键信息"""
    user_query = ""
    assistant_responses = []
    tool_calls_count = 0
    
    for msg in messages:
        if msg["role"] == "user":
            user_query = msg["content"]
        elif msg["role"] == "assistant":
            if msg.get("content"):
                assistant_responses.append(msg["content"])
            if msg.get("tool_calls"):
                tool_calls_count += len(msg["tool_calls"])
    
    return {
        "user_query": user_query,
        "assistant_response": " | ".join(assistant_responses) if assistant_responses else "",
        "tool_calls_count": tool_calls_count
    }


def extract_tool_calls_info(tool_calls: List[Dict]) -> Dict[str, Any]:
    """提取工具调用信息"""
    if not tool_calls:
        return {
            "tool_names": "",
            "tool_calls_summary": "",
            "total_tool_calls": 0
        }
    
    tool_names = [call["name"] for call in tool_calls]
    tool_calls_summary = " | ".join([f"{call['name']}({len(call.get('parameters', ''))} chars)" 
                                   for call in tool_calls])
    
    return {
        "tool_names": ", ".join(set(tool_names)),
        "tool_calls_summary": tool_calls_summary,
        "total_tool_calls": len(tool_calls)
    }


def extract_tool_responses_info(tool_responses: List[Dict]) -> Dict[str, Any]:
    """提取工具响应信息"""
    if not tool_responses:
        return {
            "successful_calls": 0,
            "failed_calls": 0,
            "avg_reward_score": 0
        }
    
    successful = sum(1 for resp in tool_responses if resp.get("success", False))
    failed = len(tool_responses) - successful
    avg_reward = sum(resp.get("reward_score", 0) for resp in tool_responses) / len(tool_responses)
    
    return {
        "successful_calls": successful,
        "failed_calls": failed,
        "avg_reward_score": round(avg_reward, 3)
    }


def extract_orm_scores(orm_scores: Dict) -> Dict[str, float]:
    """提取ORM评分信息"""
    return {
        "unnecessary_action_penalty": round(orm_scores.get("unnecessary_action_penalty", 0), 3),
        "search_info_reward": round(orm_scores.get("search_info_reward", 0), 3),
        "action_loop_reward": round(orm_scores.get("action_loop_reward", 0), 3),
        "financial_resolution_reward": round(orm_scores.get("financial_resolution_reward", 0), 3),
        "faithful_reward": round(orm_scores.get("faithful_reward", 0), 3),
        "tool_call_reward": round(orm_scores.get("tool_call_reward", 0), 3),
        "resolution_score": round(orm_scores.get("resolution_score", 0), 3)
    }


def extract_orm_details_info(orm_details: Dict) -> Dict[str, Any]:
    """提取ORM详细信息，将所有字段以orm_details/xxx格式保存"""
    if not orm_details:
        return {}
    
    result = {}
    for key, value in orm_details.items():
        # 将字段名前缀设置为orm_details/
        field_name = f"orm_details/{key}"
        
        # 根据值的类型进行适当的处理
        if isinstance(value, (int, float)):
            result[field_name] = round(value, 3) if isinstance(value, float) else value
        elif isinstance(value, dict):
            # 如果是字典，转换为JSON字符串
            result[field_name] = json.dumps(value, ensure_ascii=False, indent=2)
        elif isinstance(value, list):
            # 如果是列表，转换为JSON字符串
            result[field_name] = json.dumps(value, ensure_ascii=False, indent=2)
        else:
            # 其他类型直接转换为字符串
            result[field_name] = str(value)
    
    return result


def clean_text_for_excel(text: str, max_length: int = None) -> str:
    """清理文本，使其适合在Excel中显示"""
    if not text:
        return ""
    
    # 移除特殊字符，但保留内容完整性
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # 只有指定了max_length且文本超长时才截断
    if max_length and len(text) > max_length:
        text = text[:max_length] + "..."
    
    return text


def process_jsonl_to_excel(input_file: str, output_file: str):
    """将JSONL文件转换为Excel格式"""
    
    print(f"正在读取文件: {input_file}")
    
    data_rows = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # 提取基本信息
                basic_info = {
                    "trajectory_id": data.get("trajectory_id", ""),
                    "skeleton_id": data.get("skeleton_id", ""),
                    "query_id": data.get("query_id", ""),
                    "prm_score": round(data.get("prm_score", 0), 3)
                }
                
                # 提取消息信息
                messages_info = extract_messages_info(data.get("messages", []))
                
                # 提取工具调用信息
                tool_calls_info = extract_tool_calls_info(data.get("tool_calls", []))
                
                # 提取工具响应信息
                tool_responses_info = extract_tool_responses_info(data.get("tool_responses", []))
                
                # 提取ORM评分信息
                orm_scores_info = extract_orm_scores(data.get("orm_scores", {}))
                
                # 提取ORM详细信息
                orm_details_info = extract_orm_details_info(data.get("orm_details", {}))
                
                # 保留原始的tool_calls和tool_responses字段（JSON格式）
                original_fields = {
                    "tool_calls_json": json.dumps(data.get("tool_calls", []), ensure_ascii=False, indent=2),
                    "tool_responses_json": json.dumps(data.get("tool_responses", []), ensure_ascii=False, indent=2)
                }
                
                # 合并所有信息
                row_data = {
                    **basic_info,
                    **messages_info,
                    **tool_calls_info,
                    **tool_responses_info,
                    **orm_scores_info,
                    **orm_details_info,
                    **original_fields
                }
                
                # for key, value in row_data.items():
                #     if isinstance(value, str):
                #         if key in ["assistant_response", "user_query", "tool_calls_json", "tool_responses_json"]:
                #             row_data[key] = clean_text_for_excel(value)  # 不截断
                #         else:
                #             row_data[key] = clean_text_for_excel(value, max_length=500)
                
                # 添加标注列
                row_data.update({
                    "人工标注_质量评分": "",
                    "人工标注_问题类型": "",
                    "人工标注_是否解决": "",
                    "人工标注_工具使用合理性": "",
                    "人工标注_备注": ""
                })
                
                data_rows.append(row_data)
                
            except json.JSONDecodeError as e:
                print(f"警告: 第{line_num}行JSON解析失败: {e}")
                continue
            except Exception as e:
                print(f"警告: 第{line_num}行处理失败: {e}")
                continue
    
    if not data_rows:
        print("错误: 没有成功处理任何数据行")
        return
    
    # 创建DataFrame
    df = pd.DataFrame(data_rows)
    
    # 调整列顺序，将重要信息放在前面
    column_order = [
        "trajectory_id", "query_id", "user_query", "assistant_response",
        "tool_names", "total_tool_calls", "successful_calls", "failed_calls",
        "prm_score", "financial_resolution_reward", "resolution_score",
        "人工标注_质量评分", "人工标注_问题类型", "人工标注_是否解决", 
        "人工标注_工具使用合理性", "人工标注_备注"
    ]
    
    # 添加其他列
    remaining_columns = [col for col in df.columns if col not in column_order]
    final_columns = column_order + remaining_columns
    
    # 只保留存在的列
    final_columns = [col for col in final_columns if col in df.columns]
    df = df[final_columns]
    
    # 保存到Excel
    print(f"正在保存到: {output_file}")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 主数据表
        df.to_excel(writer, sheet_name='对话轨迹数据', index=False)
        
        # 创建说明表
        explanation_data = {
            "字段名": [
                "trajectory_id", "query_id", "user_query", "assistant_response",
                "tool_names", "total_tool_calls", "successful_calls", "failed_calls",
                "prm_score", "financial_resolution_reward", "resolution_score",
                "人工标注_质量评分", "人工标注_问题类型", "人工标注_是否解决",
                "人工标注_工具使用合理性", "人工标注_备注"
            ],
            "说明": [
                "轨迹唯一标识符",
                "查询唯一标识符", 
                "用户提出的问题",
                "助手的回复内容",
                "使用的工具名称列表",
                "工具调用总次数",
                "成功的工具调用次数",
                "失败的工具调用次数",
                "PRM模型评分",
                "金融问题解决奖励分数",
                "问题解决评分",
                "人工评估的整体质量(1-5分)",
                "问题类型分类(如:基金查询/投资建议等)",
                "问题是否得到解决(是/否/部分)",
                "工具使用是否合理(合理/不合理/部分合理)",
                "其他备注信息"
            ]
        }
        
        explanation_df = pd.DataFrame(explanation_data)
        explanation_df.to_excel(writer, sheet_name='字段说明', index=False)
        
        # 调整列宽
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                
                adjusted_width = min(max_length + 2, 50)  # 最大宽度限制为50
                worksheet.column_dimensions[column_letter].width = adjusted_width
    
    print(f"转换完成! 共处理 {len(data_rows)} 条记录")
    print(f"输出文件: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='将JSONL对话轨迹文件转换为Excel格式用于人工标注')
    parser.add_argument('input_file', help='输入的JSONL文件路径')
    parser.add_argument('-o', '--output', help='输出的Excel文件路径(可选)', default=None)
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {args.input_file}")
        return
    
    # 确定输出文件路径
    if args.output:
        output_file = args.output
    else:
        output_file = str(input_path.with_suffix('.xlsx'))
    
    # 执行转换
    try:
        process_jsonl_to_excel(args.input_file, output_file)
    except Exception as e:
        print(f"转换过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
