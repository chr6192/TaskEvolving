"""
Reward API 测试用例
"""
import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from deep_research.reward.api.example_client import RewardAPIClient

# 加载测试数据
def load_test_data():
    """从JSON文件加载测试数据"""
    json_path = Path(__file__).parent / "test_data_with_rubrics.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 全局加载测试数据
TEST_DATA = load_test_data()
query = TEST_DATA["query"]
messages = TEST_DATA["messages"]
rubrics = TEST_DATA["rubrics"]
assistant_response = TEST_DATA["assistant_response"]


def prepare_multi_sample_data(num_samples: int = 2):
    """
    准备多样本测试数据：一个prompt对应多个样本
    
    Args:
        num_samples: 样本数量，默认3个
    """
    response_list = []
    messages_list = []
    extra_list = []
    
    for i in range(num_samples):
        # 为每个样本创建略微不同的response（模拟不同模型输出）
        response = assistant_response if i == 0 else assistant_response[:500] + f"\n\n[样本{i+1}的变体回答]"
        response_list.append(response)
        
        # 每个样本使用相同的messages
        messages_list.append(messages)
        
        # 为每个样本创建独立的extra
        extra_list.append({
            "skeleton_id": f"test_skel_{i+1:03d}",
            "trajectory_id": f"test_traj_{i+1:03d}",
            "skeleton": {
                "query": query,
                "query_id": rubrics["query_id"],
                "tool_calls_reference": []
            },
            "tool_calls": [],
            "tool_responses": [],
            "metadata": {"sample_index": i},
            "rubrics": rubrics
        })
    
    config = {"return_details": True, "enable_rubrics_evaluation": True}
    return query, response_list, messages_list, extra_list, config


def test_sync_with_rubrics():
    """测试: 同步调用（含rubrics）"""
    print("\n" + "="*80)
    print("测试: 同步调用（含rubrics）")
    print("="*80)
    
    client = RewardAPIClient()
    prompt, response_list, messages_list, extra_list, config = prepare_multi_sample_data(num_samples=2)
    
    # 打印输入
    print("\n【输入】")
    print(json.dumps({
        "prompt": prompt[:100] + "...",
        "sample_count": len(response_list),
        "config": config
    }, indent=2, ensure_ascii=False))
    
    # 调用API
    result = client.compute_sync(prompt=prompt, response_list=response_list, messages_list=messages_list, extra_list=extra_list, config=config)
    
    # 打印输出
    print("\n【输出】")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def test_async_with_rubrics():
    """测试: 异步调用（含rubrics）"""
    print("\n" + "="*80)
    print("测试: 异步调用（含rubrics）")
    print("="*80)
    
    client = RewardAPIClient()
    prompt, response_list, messages_list, extra_list, config = prepare_multi_sample_data(num_samples=2)
    
    # 打印输入
    print("\n【输入】")
    print(json.dumps({
        "task_id": "test_async_001",
        "prompt": prompt[:100] + "...",
        "sample_count": len(response_list),
        "config": config
    }, indent=2, ensure_ascii=False))
    
    # 提交异步任务
    task_id = client.compute_async(prompt=prompt, response_list=response_list, messages_list=messages_list, extra_list=extra_list, config=config, task_id="test_async_001")
    print(f"\n任务已提交: {task_id}")
    
    # 等待完成
    result = client.wait_for_task(task_id, poll_interval=1.0)
    
    # 打印输出
    print("\n【输出】")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # 运行所有测试
    try:
        test_sync_with_rubrics()
        test_async_with_rubrics()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
