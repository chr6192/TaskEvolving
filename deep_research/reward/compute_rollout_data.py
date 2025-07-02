import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from dr_reward_function import compute_score


def compute_reward_statistics(all_reward_infos, all_scores):
    """
    统计每项 reward 及总分的均值、最大值、最小值、方差
    """
    reward_stats = {}
    for k, v in all_reward_infos.items():
        arr = np.array(v)
        reward_stats[k] = {
            'mean': round(float(np.mean(arr)), 4),
            'max': round(float(np.max(arr)), 4),
            'min': round(float(np.min(arr)), 4),
            'var': round(float(np.var(arr)), 4)
        }
    # 统计总分
    all_scores_arr = np.array(all_scores)
    # 四位小数
    reward_stats['total_score'] = {
        'mean': round(float(np.mean(all_scores_arr)), 4),
        'max': round(float(np.max(all_scores_arr)), 4),
        'min': round(float(np.min(all_scores_arr)), 4),
        'var': round(float(np.var(all_scores_arr)), 4)
    }
    return reward_stats

def load_all_messages(input_path: str) -> List[List[Dict[str, str]]]:
    """
    从 JSONL 文件中读取所有记录
    """
    all_messages = []
    with open(input_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            record = json.loads(line)

            # 如果 record 包含 'messages' 字段，使用其值；否则，直接当作 messages
            if isinstance(record, dict) and 'messages' in record:
                msgs = record['messages']
            else:
                msgs = record

            # msgs 应为 List[List[Dict[str, str]]] 或 List[Dict[str, str]]
            # 若为单一对话列表，则包裹成一项列表
            if msgs and isinstance(msgs[0], dict):
                all_messages.append(msgs)
            else:
                all_messages.extend(msgs)
    return all_messages


def shiyu_load(input_path,test_num):
    with open(input_path) as f:
        data = f.readlines()
    import pandas as pd
    data_list = []
    for dp in data:
        data_list.append(eval(dp))
    df = pd.DataFrame(data_list)
    return df['messages'].tolist()[:test_num],df['output'].tolist()[:test_num]



def process_all(input_path: str, output_path: str, test_num=1024) -> None:
    """
    加载所有对话消息，一次性调用 compute_score，
    将总分和 reward_info 保存到 JSON 文件。
    """
    # messages_list = load_all_messages(input_path)[:8]

    messages_list,output_list = shiyu_load(input_path,test_num)

    total_score, reward_info, reward_details = compute_score("", output_list, messages_list, return_details=True)
    print(reward_details)

    all_results = []
    all_scores = []
    all_reward_infos = {k: [] for k in reward_info.keys()}

    for idx, messages in enumerate(messages_list):
        # import pdb;pdb.set_trace()
        result = {
            'messages': messages,
            'total_score': total_score[idx] if isinstance(total_score, list) else total_score,
            'reward_info': {k: reward_info[k][idx] if isinstance(reward_info[k], list) else reward_info[k]
                            for k in reward_info}
        }
        all_results.append(result)
        all_scores.append(result['total_score'])
        for k in all_reward_infos:
            all_reward_infos[k].append(result['reward_info'][k])

    # 统计 reward 各项的均值、最大值、方差
    reward_stats = compute_reward_statistics(all_reward_infos, all_scores)

    # 写入 jsonl 文件
    with open(output_path, 'w', encoding='utf-8') as fout:
        for item in all_results:
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        # 最后一行写统计信息
        fout.write(json.dumps({'reward_statistics': reward_stats}, ensure_ascii=False) + '\n')

    print(f"Scoring complete. Results saved to {output_path}")


if __name__ == '__main__':
    import time
    input_file = '/mnt4/code/congling/deep_research/Qwen3-32B-20250701-002445/rollout_data/40.jsonl'
    output_file = '/mnt2/sst/data/temp/parallel.jsonl'
    start_time = time.time()
    process_all(input_file, output_file)
    end_time = time.time()
    import subprocess
    result = subprocess.run(['wc', '-l', output_file], capture_output=True, text=True)
    count = int(result.stdout.split()[0])
    print(f"行数：{count}")
    print(f"totnum {count} cost time {end_time-start_time}\nProcessing complete. Results saved to {output_file}")

