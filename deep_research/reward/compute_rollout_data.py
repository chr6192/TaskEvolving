import json
from dr_reward_function import compute_score
from typing import Any, Dict, List, Optional, Tuple, Union

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


def process_all(input_path: str, output_path: str) -> None:
    """
    加载所有对话消息，一次性调用 compute_score，
    将总分和 reward_info 保存到 JSON 文件。
    """
    messages = load_all_messages(input_path)
    total_score, reward_info = compute_score("", [""], messages)

    result = {
        'total_score': total_score,
        'reward_info': reward_info
    }

    with open(output_path, 'w', encoding='utf-8') as fout:
        json.dump(result, fout, ensure_ascii=False, indent=2)

    print(f"Scoring complete. Results saved to {output_path}")


if __name__ == '__main__':
    input_file = '/mnt4/code/congling/deep_research/Qwen3-8B-20250628/rollout_data/1.jsonl'
    output_file = '/mnt4/code/dingyiwen/data/1_scored_normal.jsonl'
    process_all(input_file, output_file)
    print(f"Processing complete. Results saved to {output_file}")
