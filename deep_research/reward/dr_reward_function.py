import concurrent.futures
import copy
import datetime
import json
import os
import re
import time
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from filelock import FileLock
from meta_info.rewards.card_reward import cardRM_Qwen
from meta_info.rewards.express_reward import (AsyncMapPeekBasedOpenai,
                                               Map_peek_based_openai)
from meta_info.rewards.faithful_reward import (
    AsyncMapFaithfulModelBasedOpenai, Map_faithful_model_based_openai)
from meta_info.rewards.finance_personalization import personalizationRM
from meta_info.rewards.finance_rule import (LengthRM,
                                            Map_faithful_rule_based_num_filter,
                                            MapFormat, ThinkLengthRM)

from atomic_eval.simple_eval import SimpleEvaluator

INVALID_REWARD = -0.01

def extract_assistant_group_messages(group_messages: List[List[Dict[str, str]]]) -> List[str]:
    result = []
    for conversation in group_messages:
        # 提取当前对话中所有助手消息的内容
        assistant_contents = [
            msg["content"]
            for msg in conversation
            if msg.get("role") == "assistant"
        ]
        # 用两个换行符拼接内容并添加到结果列表
        result.append("\n\n".join(assistant_contents))
    return result

def extract_tool_calls(group_messages: List[List[Dict[str, str]]]) -> List[str]:
    result = []
    for conversation in group_messages:
        # 提取当前对话中所有工具调用的内容
        tool_calls = [
            msg["content"]
            for msg in conversation
            if msg.get("role") == "tool"
        ]
        # 用两个换行符拼接内容并添加到结果列表
        result.append("\n\n".join(tool_calls))
    return result

import time
import os
import re
os.environ['TZ'] = 'Asia/Shanghai'
time.tzset()
import json
def extract_packing_yjx(group_responses: list[str]) -> List[str]:
    pattern = r"<tool_response>([\s\S]*?)</tool_response>"
    current_time = time.strftime('%Y年%m月%d日 %H:%M:%S 星期%A', time.localtime())
    results = []
    for output in group_responses:
        yjx_rm_consume_template = '# 之前的聊天内容\n\n# 用户画像\n<user_info>\n# 参考资料\n## 平台专业服务内容\n<platform_info>\n## 权威金融查询\n<fin_data_info>\n## 全网搜索内容\n<search_info>\n## 平台服务卡片\n<card_info>\n# 当前时间\n<current_time>\n# 当前问题\n'
        save_dict = {'<user_info>':'','<platform_info>':'','<fin_data_info>':[],'<search_info>':[],'<card_info>':'','<current_time>':current_time}
        matches = re.findall(pattern, output)
        for matched_text in matches:
            
            # tool_call,tool_response = matched_text.split('</tool_call>')
            # tool_response = tool_response.split('<tool_response>')[-1]
            try:
                tool_response = json.loads(matched_text)
                # tool_call = json.loads(tool_call)
                # tool_response = json.loads(tool_response)
                if '用户持仓结构' in tool_response[0]:
                    save_dict['<user_info>'] = '用户基础信息：\n'+tool_response[1]['用户基础信息']+'\n用户持仓特征：\n'+tool_response[0]['用户持仓结构']
                else: 
                    for data_dict in tool_response:
                        key_name = list(data_dict.keys())[0]
                        if data_dict[key_name]=='[]':continue
                        data_list = data_dict[key_name].split('------「webpage_spliter」------')
                        if key_name == '金融指标':
                            save_dict['<fin_data_info>'].extend(data_list)
                        else:
                            save_dict['<search_info>'].extend(data_list)
            except:
                continue

        for tag in ['<fin_data_info>','<search_info>']:
            if len(save_dict[tag])==0:
                save_dict[tag] = ''
            else:
                prompt_tag = {'<fin_data_info>':'data','<search_info>':'webpage'}[tag]
                supply_list = []
                
                for i,supply in enumerate(save_dict[tag]):
                    if tag=='<fin_data_info>':
                        supply = supply.split('\n发布时间:')[-1].strip()
                    elif tag=='<search_info>':
                        supply = '发布时间:' + supply.split('\n发布时间:')[-1].strip()
                    supply_list.append(f'[{prompt_tag} {i} begin]\n{supply}\n[{prompt_tag} {i} end]')
                save_dict[tag] = '\n'.join(supply_list)
        supply_empty_flag = True
        for tag,value in save_dict.items():
            if (value!='') and (tag!='<current_time>'):
                supply_empty_flag = False
            yjx_rm_consume_template = yjx_rm_consume_template.replace(tag,value)
        results.append((yjx_rm_consume_template,supply_empty_flag))
    return results

# 从所有子对话提取最后一轮输出内容
def extract_last_assistant_contents(group_messages: List[List[Dict[str, str]]]) -> List[str]:
    last_response_list = []
    for conversation in group_messages:
        last_assistant_content = None
        # 反向遍历对话以找到最后一个助手消息
        for message in reversed(conversation):
            if message.get('role') == 'assistant':
                last_assistant_content = message.get('content')
                break
        last_response_list.append(last_assistant_content)
    return last_response_list

# 从所有子对话中提取第一轮 user 输入，包含问题
def extract_first_user_contents(group_messages: List[List[Dict[str, str]]]) -> List[str]:
    first_user_contents = []
    for conversation in group_messages:
        first_user_content = None
        # 遍历对话以找到第一个用户消息
        for message in conversation:
            if message.get('role') == 'user':
                first_user_content = message.get('content')
                break
        first_user_contents.append(first_user_content)
    return first_user_contents
    

class HybridRewardExecutor:
    """混合并行执行器，支持异步和同步reward函数的并行执行"""
    
    def __init__(self):
        self.timings = {}
        self.async_tasks = []
        self.sync_tasks = []
    
    def add_async_reward(self, name: str, func, *args, **kwargs):
        """添加异步reward任务"""
        self.async_tasks.append((name, func, args, kwargs))
    
    def add_sync_reward(self, name: str, func, *args, **kwargs):
        """添加同步reward任务"""
        self.sync_tasks.append((name, func, args, kwargs))
    
    async def execute_all_async(self):
        """异步执行所有任务"""
        results = {}
        
        # 并行执行异步任务
        if self.async_tasks:
            async_coroutines = []
            async_names = []
            for name, func, args, kwargs in self.async_tasks:
                # 检查函数是否为异步函数
                if asyncio.iscoroutinefunction(func) or hasattr(func, '__call__') and asyncio.iscoroutinefunction(func.__call__):
                    # 直接调用异步函数
                    async_coroutines.append(self._timed_async_call(name, func, *args, **kwargs))
                else:
                    # 同步函数包装为异步
                    async_coroutines.append(self._timed_sync_to_async_call(name, func, *args, **kwargs))
                async_names.append(name)
            
            # 等待所有异步任务完成
            async_results = await asyncio.gather(*async_coroutines, return_exceptions=True)
            for i, name in enumerate(async_names):
                if isinstance(async_results[i], Exception):
                    print(f"异步reward {name} 执行出错: {async_results[i]}")
                    results[name] = ([], []) if name == "faithful_reward" else []
                else:
                    results[name] = async_results[i]
        
        # 并行执行同步任务（使用线程池）
        if self.sync_tasks:
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=len(self.sync_tasks)) as executor:
                sync_futures = []
                sync_names = []
                for name, func, args, kwargs in self.sync_tasks:
                    future = loop.run_in_executor(
                        executor, 
                        self._timed_sync_call, 
                        name, func, *args, **kwargs
                    )
                    sync_futures.append(future)
                    sync_names.append(name)
                
                # 等待所有同步任务完成
                sync_results = await asyncio.gather(*sync_futures, return_exceptions=True)
                for i, name in enumerate(sync_names):
                    if isinstance(sync_results[i], Exception):
                        print(f"同步reward {name} 执行出错: {sync_results[i]}")
                        results[name] = []
                    else:
                        results[name] = sync_results[i]
        
        return results
    
    def execute_all_sync(self):
        """同步执行所有任务（检测事件循环状态）"""
        try:
            # 尝试获取当前事件循环
            loop = asyncio.get_running_loop()
            # 如果已经在事件循环中，抛出错误提示
            raise RuntimeError(
                "execute_all_sync不能在运行中的事件循环内调用。"
                "请在异步上下文中使用 await execute_all_async() 代替。"
            )
        except RuntimeError as e:
            if "no running event loop" in str(e):
                # 没有事件循环，可以安全使用 asyncio.run
                return asyncio.run(self.execute_all_async())
            else:
                # 其他RuntimeError，重新抛出
                raise
    
    async def _timed_async_call(self, name: str, func, *args, **kwargs):
        """带计时的异步函数调用 - 直接调用异步函数"""
        start_time = time.perf_counter()
        try:
            # 直接await异步函数，不使用线程
            result = await func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            self.timings[f"{name}_time"] = round(end_time - start_time, 4)
    
    async def _timed_sync_to_async_call(self, name: str, func, *args, **kwargs):
        """带计时的同步函数异步包装调用"""
        start_time = time.perf_counter()
        try:
            # 同步函数包装为异步
            result = await asyncio.to_thread(func, *args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            self.timings[f"{name}_time"] = round(end_time - start_time, 4)
    
    def _timed_sync_call(self, name: str, func, *args, **kwargs):
        """带计时的同步函数调用"""
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            self.timings[f"{name}_time"] = round(end_time - start_time, 4)
    
    def get_timings(self):
        return self.timings.copy()

# group_messages: List[List[Dict[str, str]]]，输入是对话历史
async def compute_score_async(prompt: str, group_responses: list[str], group_messages: list[list[dict]], group_extras=None, use_group_reward=False, return_details=False, return_statistics=True, **kwargs):
    """
    异步版本的compute_score，用于高性能场景和批量处理
    
    输入：
        - prompt: str，当前问题
        - group_responses: List[str]，每个子对话的模型输出
        - group_messages: List[List[Dict[str, str]]]，每个子对话的消息列表
    输出：
        - total_score: List[float]，每个子对话的总分
        - reward_info: Dict[str, List[float]]，包含各个维度的分数
        - reward_details: List[Dict[str, Any]]，每个子对话的详细分数信息
    """
    # step1: 提取问题、工具调用、模型输出
    # prompt_contents: List[str]，模型第一轮输入，生成 prompt + query
    # supply_contents: List[str]，工具调用内容列表
    # assistant_contents: List[str]，模型输出内容列表
    prompt_contents = extract_first_user_contents(group_messages)
    supply_contents = extract_tool_calls(group_messages)
    yjx_contents = extract_packing_yjx(group_responses)
    assistant_contents = extract_assistant_group_messages(group_messages)

    # step2: 分不同维度计算分数
    # map_faithful_func = Map_faithful_model_based_openai()
    # map_peek_func = Map_peek_based_openai()
    async_map_faithful_func = AsyncMapFaithfulModelBasedOpenai()
    async_map_peek_func = AsyncMapPeekBasedOpenai()
    # think_length_func = ThinkLengthRM()
    # answer_length_func = LengthRM()
    number_filter_func = Map_faithful_rule_based_num_filter()
    map_format_func = MapFormat()
    
    executor = HybridRewardExecutor()
    
    # step3: 注册reward任务
    # 异步任务
    # 基于工具调用的返回内容、模型生成的内容打分
    executor.add_async_reward(
        "faithful_reward", 
        async_map_faithful_func, 
        assistant_contents, yjx_contents
    )
    executor.add_async_reward(
        "express_reward",
        async_map_peek_func,
        assistant_contents, prompt_contents
    )
    
    # 基于生成的思考长度打分，参数是思考长度上限
    # think_length_threshold = [4096 for _ in range(len(assistant_contents))]
    # think_length_reward = think_length_func(assistant_contents, think_length_threshold)

    # 基于最终答案长度打分，参数是答案长度下限
#     answer_length_threshold = [1000 for _ in range(len(assistant_contents))]
#     answer_length_reward = answer_length_func(assistant_contents, answer_length_threshold)


    # 同步任务
    executor.add_sync_reward(
        "faithful_num_exist_reward",
        number_filter_func,
        assistant_contents, supply_contents
    )
    executor.add_sync_reward(
        "map_format_reward",
        map_format_func,
        assistant_contents, supply_contents
    )
    
    # step4: 并行执行所有任务
    results = await executor.execute_all_async()
    timings = executor.get_timings()
    
    # step5: 提取结果
    faithful_reward, faithful_details = results["faithful_reward"]
    
    express_result = results["express_reward"]
    if isinstance(express_result, tuple):
        express_reward, express_details = express_result
    else:
        express_reward = express_result
        express_details = [{"express_score": score} for score in express_result]
    
    num_exist_result = results["faithful_num_exist_reward"]
    if isinstance(num_exist_result, tuple):
        faithful_num_exist_reward, num_exist_details = num_exist_result
    else:
        faithful_num_exist_reward = num_exist_result
        num_exist_details = [{"num_exist_score": score} for score in num_exist_result]
    
    format_result = results["map_format_reward"]
    if isinstance(format_result, tuple):
        map_format_reward, format_details = format_result
    else:
        map_format_reward = format_result
        format_details = [{"format_score": score} for score in format_result]


    # total_score = (torch.tensor(faithful_reward) + torch.tensor(express_reward) + torch.tensor(
    #     think_length_reward) + torch.tensor(answer_length_reward) +
    #                torch.tensor(faithful_num_exist_reward) + torch.tensor(map_format_reward))
    # total_score = (0.6 * torch.tensor(faithful_reward) + 0.2 * torch.tensor(faithful_num_exist_reward) + 0.2 * torch.tensor(map_format_reward))
    total_score = (0.4 * torch.tensor(faithful_reward) + 0.4 * torch.tensor(express_reward) + 
                   0.1 * torch.tensor(faithful_num_exist_reward) + 0.1 * torch.tensor(map_format_reward))
    total_score = total_score.tolist()

    # 暂时不使用
    # personalization_func = personalizationRM()
    # card_reward_func = cardRM_Qwen()
    # # # TODO: 输入处理：1. 当前问题 2. 卡片供给内容
    # start = time.perf_counter()
    # card_reward = card_reward_func(supply_contents, assistant_contents)
    # timings['card_reward_time'] = round(time.perf_counter() - start, 4)

    # # TODO: 输入内容：1. 历史对话 2. 思考内容与 answer 3. 用户画像 user_profile 字段 4. 问题
    # start = time.perf_counter()
    # personlization_reward = personalization_func(supply_contents, assistant_contents)
    # timings['personalization_reward_time'] = time.perf_counter() - start

    print("\n" + "="*80)
    print("奖励计算完成 - 详细信息")
    print("="*80)
    print(f"混合并行reward计算耗时: {timings}")
    print(f"异步任务数: {len(executor.async_tasks)}, 同步任务数: {len(executor.sync_tasks)}")
    print(f"总计算时间: {sum(timings.values()):.4f}秒")
    
    for i in range(len(assistant_contents)):
        print(f"\n--- 样本 {i} ---")
        print(f"忠实性奖励: {faithful_reward[i]:.4f}")
        print(f"体感奖励: {express_reward[i]:.4f}")
        print(f"数字规则奖励: {faithful_num_exist_reward[i]:.4f}")
        print(f"格式规则奖励: {map_format_reward[i]:.4f}")
        print(f"总分: {total_score[i]:.4f}")
        
        if i < len(faithful_details):
            print(f"  忠实性错误数: {faithful_details[i].get('faithful_error_count', 'N/A')}")
        if i < len(express_details):
            print(f"  体感原始分: {express_details[i].get('express_raw_score', 'N/A')}")
        if i < len(num_exist_details):
            print(f"  无中生有的数字: {num_exist_details[i].get('num_exist_errors', [])}")
        if i < len(format_details):
            print(f"  格式错误数: {format_details[i].get('format_error_count', 0)}")
    
    print("="*80 + "\n")
    
    reward_info = {
        "faithful_reward": faithful_reward,
        "express_reward": express_reward,
        # "think_length_reward": think_length_reward,
        # "answer_length_reward": answer_length_reward,
        "faithful_num_exist_reward": faithful_num_exist_reward,
        "map_format_reward": map_format_reward,
    }
    
    reward_details = []

    # 构建综合的reward_details
    for i in range(len(assistant_contents)):
        detail = {
            "index": i,
            "total_score": total_score[i]
        }
        
        # 合并faithful详情（去除tmp后缀的字段）
        if i < len(faithful_details):
            faithful_detail = {key: value for key, value in faithful_details[i].items() if not key.endswith("tmp")}
            detail.update(faithful_detail)
        
        # 合并express详情
        if i < len(express_details):
            detail.update(express_details[i])
        
        # 合并num_exist详情
        if i < len(num_exist_details):
            detail.update(num_exist_details[i])
        
        # 合并format详情
        if i < len(format_details):
            detail.update(format_details[i])
        
        # 添加耗时信息
        detail.update(timings)
        
        reward_details.append(detail)

    if return_details:
        return total_score, reward_info, reward_details
    else:
        return total_score, reward_info


def compute_score(prompt: str, group_responses: list[str], group_messages: list[list[dict]], group_extras=None, use_group_reward=False, return_details=False, return_statistics=True, **kwargs):
    """
    同步版本的compute_score，保持向后兼容性
    
    这是对 compute_score_async 的同步包装器，适用于：
    - 现有代码迁移，无需修改调用方式
    - 单次调用场景
    - 不需要与其他异步操作并发的场景
    
    对于高频调用或批量处理，建议使用 compute_score_async 获得更好的性能。
    
    输入：
        - prompt: str，当前问题
        - group_responses: List[str]，每个子对话的模型输出
        - group_messages: List[List[Dict[str, str]]]，每个子对话的消息列表
    输出：
        - total_score: List[float]，每个子对话的总分
        - reward_info: Dict[str, List[float]]，包含各个维度的分数
        - reward_details: List[Dict[str, Any]]，每个子对话的详细分数信息
    """
    return asyncio.run(compute_score_async(
        prompt, group_responses, group_messages, 
        group_extras=group_extras, 
        use_group_reward=use_group_reward, 
        return_details=return_details, 
        return_statistics=return_statistics, 
        **kwargs
    ))
