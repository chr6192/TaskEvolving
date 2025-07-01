#!/usr/bin/env python3
"""
测试线程安全的并发奖励函数
验证多线程访问时不会出现SIGSEGV错误
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from meta_info.rewards.faithful_reward import Map_faithful_model_based_openai
import threading
import time

def test_concurrent_reward_function():
    print("开始测试线程安全的并发奖励函数...")
    
    # 创建奖励函数实例
    reward_func = Map_faithful_model_based_openai()
    
    # 构造测试数据 - 多个样本以测试并发
    completions = [
        "根据市场数据分析，当前股市表现良好，建议适度投资。",
        "基金净值上涨2.5%，预期年化收益率达到8%。",
        "债券市场相对稳定，适合风险偏好较低的投资者。",
        "科技股近期波动较大，建议谨慎操作。"
    ]
    
    prompts = [
        "# 之前的聊天内容\n市场数据显示上证指数3000点\n# 当前问题\n请分析市场情况",
        "# 之前的聊天内容\n基金数据更新\n# 当前问题\n基金收益如何",
        "# 之前的聊天内容\n债券信息\n# 当前问题\n债券投资建议",
        "# 之前的聊天内容\n科技股动态\n# 当前问题\n科技股投资策略"
    ]
    
    print(f"准备测试 {len(completions)} 个样本的并发处理...")
    
    # 测试不同的线程数
    for num_threads in [1, 2, 4, 8]:
        print(f"\n测试使用 {num_threads} 个线程...")
        start_time = time.time()
        
        try:
            # 调用奖励函数，这会触发多线程处理
            scores = reward_func(
                completions=completions,
                prompts=prompts,
                num_threads=num_threads
            )
            
            end_time = time.time()
            
            print(f"✅ {num_threads} 线程测试成功!")
            print(f"   处理时间: {end_time - start_time:.2f}秒")
            print(f"   得分结果: {scores}")
            
        except Exception as e:
            print(f"❌ {num_threads} 线程测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n🎉 所有并发测试通过！线程安全问题已解决。")
    return True

def test_thread_local_isolation():
    """测试线程本地存储的隔离性"""
    print("\n测试线程本地存储的隔离性...")
    
    from meta_info.rewards.faithful_reward import get_thread_local_ray_client
    
    client_ids = []
    
    def get_client_id():
        client = get_thread_local_ray_client()
        client_id = id(client)
        client_ids.append(client_id)
        print(f"线程 {threading.current_thread().name}: 客户端ID = {client_id}")
    
    # 创建多个线程，每个线程获取自己的客户端
    threads = []
    for i in range(3):
        thread = threading.Thread(target=get_client_id, name=f"TestThread-{i}")
        threads.append(thread)
        thread.start()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    # 检查是否每个线程都有独立的客户端实例
    unique_clients = len(set(client_ids))
    print(f"创建了 {unique_clients} 个独立的客户端实例")
    
    if unique_clients == len(client_ids):
        print("✅ 线程隔离测试通过！每个线程都有独立的客户端实例。")
        return True
    else:
        print("❌ 线程隔离测试失败！存在客户端实例共享。")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("线程安全并发奖励函数测试")
    print("=" * 60)
    
    # 测试线程本地存储隔离性
    isolation_success = test_thread_local_isolation()
    
    # 测试并发奖励函数
    concurrent_success = test_concurrent_reward_function()
    
    overall_success = isolation_success and concurrent_success
    
    print("\n" + "=" * 60)
    if overall_success:
        print("🎉 所有测试通过！多线程并发问题已解决。")
    else:
        print("❌ 部分测试失败，需要进一步调试。")
    print("=" * 60)
    
    exit(0 if overall_success else 1) 