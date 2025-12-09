import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from rm_gallery.core.data.schema import DataOutput, DataSample, Step
from rm_gallery.core.model.message import ChatMessage, ChatTool, MessageRole, Tool

INVALID_REWARD = -0.01


# 从所有子对话中提取第一轮 user 输入，包含问题
def extract_first_user_contents(
    group_messages: List[List[Dict[str, str]]],
) -> List[str]:
    first_user_contents = []
    for conversation in group_messages:
        first_user_content = None
        # 遍历对话以找到第一个用户消息
        for message in conversation:
            if message.get("role") == "user":
                first_user_content = message.get("content")
                break
        first_user_contents.append(first_user_content)
    return first_user_contents


def extract_tool_calls(group_messages: List[List[Dict[str, str]]]) -> List[str]:
    result = []
    for conversation in group_messages:
        # 提取当前对话中所有工具调用的内容
        tool_calls = [
            msg["content"] for msg in conversation if msg.get("role") == "tool"
        ]
        # 用两个换行符拼接内容并添加到结果列表
        result.append("\n\n".join(tool_calls))
    return result


def extract_packing_faithful(group_responses: list[str]) -> List[Tuple[str, bool]]:
    pattern = r"<tool_response>([\s\S]*?)</tool_response>"
    current_time = time.strftime("%Y年%m月%d日 %H:%M:%S 星期%A", time.localtime())
    results = []
    for output in group_responses:
        faithful_rm_consume_template = "# 之前的聊天内容\n\n# 用户画像\n<user_info>\n# 参考资料\n## 平台专业服务内容\n<platform_info>\n## 权威金融查询\n<fin_data_info>\n## 全网搜索内容\n<search_info>\n## 平台服务卡片\n<card_info>\n# 当前时间\n<current_time>\n# 当前问题\n"
        save_dict = {
            "<user_info>": "",
            "<platform_info>": "",
            "<fin_data_info>": [],
            "<search_info>": [],
            "<card_info>": "",
            "<current_time>": current_time,
        }
        matches = re.findall(pattern, output)
        for matched_text in matches:
            try:
                tool_response = json.loads(matched_text)
                if "用户持仓结构" in tool_response[0]:
                    save_dict["<user_info>"] = (
                        "用户基础信息：\n"
                        + tool_response[1]["用户基础信息"]
                        + "\n用户持仓特征：\n"
                        + tool_response[0]["用户持仓结构"]
                    )
                else:
                    for data_dict in tool_response:
                        key_name = list(data_dict.keys())[0]
                        if data_dict[key_name] == "[]":
                            continue
                        data_list = data_dict[key_name].split(
                            "------「webpage_spliter」------"
                        )
                        if key_name == "金融指标":
                            save_dict["<fin_data_info>"].extend(data_list)
                        else:
                            save_dict["<search_info>"].extend(data_list)
            except:
                continue

        for tag in ["<fin_data_info>", "<search_info>"]:
            if len(save_dict[tag]) == 0:
                save_dict[tag] = ""
            else:
                prompt_tag = {"<fin_data_info>": "data", "<search_info>": "webpage"}[
                    tag
                ]
                supply_list = []

                for i, supply in enumerate(save_dict[tag]):
                    if tag == "<fin_data_info>":
                        supply = supply.split("\n发布时间:")[-1].strip()
                    elif tag == "<search_info>":
                        supply = "发布时间:" + supply.split("\n发布时间:")[-1].strip()
                    supply_list.append(
                        f"[{prompt_tag} {i} begin]\n{supply}\n[{prompt_tag} {i} end]"
                    )
                save_dict[tag] = "\n".join(supply_list)
        supply_empty_flag = True
        for tag, value in save_dict.items():
            if (value != "") and (tag != "<current_time>"):
                supply_empty_flag = False
            faithful_rm_consume_template = faithful_rm_consume_template.replace(
                tag, value
            )
        results.append((faithful_rm_consume_template, supply_empty_flag))
    return results


def extract_packing_faithful_from_messages(group_messages: List[List[Dict[str, str]]]) -> List[Tuple[str, bool]]:
    """
    处理标准messages格式的函数，遵循extract_packing_faithful的逻辑
    
    Args:
        group_messages: List of conversations, each conversation is a list of message dicts
                       with keys: "role" (user/assistant/tool), "content"
    
    Returns:
        List of tuples (formatted_template, supply_empty_flag)
    """
    current_time = time.strftime("%Y年%m月%d日 %H:%M:%S 星期%A", time.localtime())
    results = []
    
    for conversation in group_messages:
        faithful_rm_consume_template = "# 之前的聊天内容\n\n# 用户画像\n<user_info>\n# 参考资料\n## 平台专业服务内容\n<platform_info>\n## 权威金融查询\n<fin_data_info>\n## 全网搜索内容\n<search_info>\n## 平台服务卡片\n<card_info>\n# 当前时间\n<current_time>\n# 当前问题\n"
        save_dict = {
            "<user_info>": "",
            "<platform_info>": "",
            "<fin_data_info>": [],
            "<search_info>": [],
            "<card_info>": "",
            "<current_time>": current_time,
        }
        
        # 从tool消息中提取工具响应内容
        for message in conversation:
            if message.get("role") == "tool":
                tool_content = message.get("content", "")
                # 尝试直接解析工具内容作为JSON
                try:
                    tool_response = json.loads(tool_content)
                    if isinstance(tool_response, list) and len(tool_response) > 0:
                        if "用户持仓结构" in tool_response[0]:
                            save_dict["<user_info>"] = (
                                "用户基础信息：\n"
                                + tool_response[1]["用户基础信息"]
                                + "\n用户持仓特征：\n"
                                + tool_response[0]["用户持仓结构"]
                            )
                        else:
                            for data_dict in tool_response:
                                key_name = list(data_dict.keys())[0]
                                if data_dict[key_name] == "[]":
                                    continue
                                data_list = data_dict[key_name].split(
                                    "------「webpage_spliter」------"
                                )
                                if key_name == "金融指标":
                                    save_dict["<fin_data_info>"].extend(data_list)
                                else:
                                    save_dict["<search_info>"].extend(data_list)
                except:
                    # 如果不是JSON格式，尝试查找<tool_response>标签
                    pattern = r"<tool_response>([\s\S]*?)</tool_response>"
                    matches = re.findall(pattern, tool_content)
                    for matched_text in matches:
                        try:
                            tool_response = json.loads(matched_text)
                            if isinstance(tool_response, list) and len(tool_response) > 0:
                                if "用户持仓结构" in tool_response[0]:
                                    save_dict["<user_info>"] = (
                                        "用户基础信息：\n"
                                        + tool_response[1]["用户基础信息"]
                                        + "\n用户持仓特征：\n"
                                        + tool_response[0]["用户持仓结构"]
                                    )
                                else:
                                    for data_dict in tool_response:
                                        key_name = list(data_dict.keys())[0]
                                        if data_dict[key_name] == "[]":
                                            continue
                                        data_list = data_dict[key_name].split(
                                            "------「webpage_spliter」------"
                                        )
                                        if key_name == "金融指标":
                                            save_dict["<fin_data_info>"].extend(data_list)
                                        else:
                                            save_dict["<search_info>"].extend(data_list)
                        except:
                            continue

        # 格式化数据
        for tag in ["<fin_data_info>", "<search_info>"]:
            if len(save_dict[tag]) == 0:
                save_dict[tag] = ""
            else:
                prompt_tag = {"<fin_data_info>": "data", "<search_info>": "webpage"}[
                    tag
                ]
                supply_list = []

                for i, supply in enumerate(save_dict[tag]):
                    if tag == "<fin_data_info>":
                        supply = supply.split("\n发布时间:")[-1].strip()
                    elif tag == "<search_info>":
                        supply = "发布时间:" + supply.split("\n发布时间:")[-1].strip()
                    supply_list.append(
                        f"[{prompt_tag} {i} begin]\n{supply}\n[{prompt_tag} {i} end]"
                    )
                save_dict[tag] = "\n".join(supply_list)
        
        # 检查是否有有效数据
        supply_empty_flag = True
        for tag, value in save_dict.items():
            if (value != "") and (tag != "<current_time>"):
                supply_empty_flag = False
            faithful_rm_consume_template = faithful_rm_consume_template.replace(
                tag, value
            )
        
        results.append((faithful_rm_consume_template, supply_empty_flag))
    
    return results


def extract_assistant_group_messages(
    group_messages: List[List[Dict[str, str]]],
) -> List[str]:
    result = []
    for conversation in group_messages:
        # 提取当前对话中所有助手消息的内容
        assistant_contents = [
            msg["content"] for msg in conversation if msg.get("role") == "assistant"
        ]
        # 用两个换行符拼接内容并添加到结果列表
        result.append("\n\n".join(assistant_contents))
    return result


def convert_tool_calls_format(msg: Dict[str, Any]) -> List[ChatTool]:
    """将工具调用转换为 RM-Gallery 的 ChatTool 格式"""
    tool_calls = msg.get("tool_calls")
    if not tool_calls:  # 检查是否为None或空列表
        return []

    tool = [
        ChatTool(
            id=call["id"],
            function=Tool(
                arguments=call["function"]["arguments"],
                name=call["function"]["name"],
            ),
            type=call["type"],
        )
        for call in tool_calls
    ]
    return tool


def convert_to_data_samples(
    group_messages: List[List[dict]], tool_calls_reference
) -> DataSample:
    """将group_messages 转换为 RM-Gallery 的 DataSample格式"""
    
    # 检查第一条消息是否为 system prompt
    first_msg = group_messages[0][0]
    skip_messages = 0
    
    if first_msg["role"] == "system":
        # 检查所有样本的第二条消息是否都是 user 且 content 相同
        all_have_same_user = True
        first_user_content = None
        
        for messages in group_messages:
            if len(messages) < 2 or messages[1]["role"] != "user":
                all_have_same_user = False
                break
            
            if first_user_content is None:
                first_user_content = messages[1]["content"]
            elif first_user_content != messages[1]["content"]:
                all_have_same_user = False
                break
        
        if all_have_same_user:
            # 剔除 system prompt，使用第一条 user message 作为 shared_input
            shared_input_messages = group_messages[0][1:2]  # 获取第二条消息
            skip_messages = 2  # 跳过前两条消息（system + user）
        else:
            # 保留 system prompt 作为 shared_input
            shared_input_messages = group_messages[0][:1]
            skip_messages = 1
    else:
        # 第一条不是 system prompt，保持原逻辑
        shared_input_messages = group_messages[0][:1]
        skip_messages = 1
    
    # 将共享输入转换为 ChatMessage 对象
    shared_input = []
    for msg in shared_input_messages:
        shared_input.append(
            ChatMessage(
                role=msg["role"],
                content=msg["content"],
            )
        )

    # 为每组消息创建一个DataOutput
    outputs = []
    for i, messages in enumerate(group_messages):
        # 将当前组的所有消息转换为 ChatMessage 对象
        output_messages = []
        for msg in messages[skip_messages:]:  # 跳过共享输入消息
            role = msg["role"]
            if role == "tool":
                role = MessageRole.FUNCTION
            output_messages.append(
                ChatMessage(
                    role=role,
                    content=msg["content"],
                    tool_calls=convert_tool_calls_format(msg),
                )
            )

        # 最后一个消息作为 final_answer，其余作为 steps
        final_answer = Step(
            role=output_messages[-1].role,
            content=output_messages[-1].content,
            tool_calls=output_messages[-1].tool_calls,
        )

        steps = [
            Step(
                role=msg.role,
                content=msg.content,
                tool_calls=msg.tool_calls,
            )
            for msg in output_messages
        ]

        output = DataOutput(answer=final_answer, steps=steps)
        outputs.append(output)

    # 创建一个包含共享输入和所有输出的单一 DataSample
    sample = DataSample(
        unique_id="sample_deep_research",
        input=shared_input,
        output=outputs,
        metadata={
            "tool_calls_reference": tool_calls_reference,
            "chat_date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        },
    )
    return sample
