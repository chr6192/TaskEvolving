import json

import datetime
from dr_reward_parallel import compute_score

def make_sample_dialogue():
    """
    构造一个包含多轮子对话的样例 messages：
    - 第一轮用户问候
    - 第一轮助手回复
    - 第二轮用户提问
    - 工具返回
    - 助手给出最终答案
    """
    messages = [
        # 第一子对话：用户发起，助手简单应答
        [
            {"role": "user", "content": "你好，能给我介绍一下北京的天气吗？"},
            {"role": "assistant", "content": "当然可以，请稍等，我正在查询天气信息。"},
            {"role": "tool", "content": json.dumps({
                "location": "Beijing",
                "forecast": [
                    {"date": "2025-06-27", "high": 30, "low": 20, "condition": "晴"},
                    {"date": "2025-06-28", "high": 28, "low": 19, "condition": "多云"}
                ]
            })}

        ],
        # 第二子对话：工具调用返回天气
        [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "tool", "content": json.dumps({
                "location": "Beijing",
                "forecast": [
                    {"date": "2025-06-27", "high": 30, "low": 20, "condition": "晴"},
                    {"date": "2025-06-28", "high": 28, "low": 19, "condition": "多云"}
                ]
            })}
        ],
        # 第三子对话：助手基于工具结果给出回答
        [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": (
                "北京今天晴，最高温度 30 ℃，最低温度 20 ℃；"
                "明天多云，最高温度 28 ℃，最低温度 19 ℃。"
            )}
        ]
    ]
    return messages

def test_compute_score_basic():
    # 构造样例输入
    messages = make_sample_dialogue()
    
    # extras 和 kwargs 均可省略
    total_score, reward_info = compute_score(messages)
    
    print("Total score:", total_score)
    print("Detailed rewards:")
    for k, v in reward_info.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    # 直接运行脚本时执行测试
    test_compute_score_basic()
    print("测试完成。")
