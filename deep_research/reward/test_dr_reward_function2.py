#!/usr/bin/env python3
"""
测试优化后的DR reward计算功能
演示正确的使用方式和资源管理
"""

import asyncio
import time
from dr_reward_function2 import compute_score_async, cleanup_thread_pool

def create_test_data():
    """创建测试数据"""
    # 模拟prompt
    prompt = "请帮我计算一下这个数学问题"
    
    # 模拟group_responses
    group_responses = [
        "根据您的问题，我需要使用计算工具来帮您解答。",
        "让我为您详细分析这个问题。",
        "我来逐步计算这个问题。"
    ]
    
    # 模拟group_messages
    group_messages = [
        [
            {"role": "user", "content": "请帮我计算一下这个数学问题"},
            {"role": "assistant", "content": "根据您的问题，我需要使用计算工具来帮您解答。"}
        ],
        [
            {"role": "user", "content": "请帮我计算一下这个数学问题"},
            {"role": "assistant", "content": "让我为您详细分析这个问题。"}
        ],
        [
            {"role": "user", "content": "请帮我计算一下这个数学问题"},
            {"role": "assistant", "content": "我来逐步计算这个问题。"}
        ]
    ]
    
    return prompt, group_responses, group_messages


async def test_single_call():
    """测试单次调用"""
    print("=" * 60)
    print("测试1: 单次调用")
    print("=" * 60)
    
    prompt, group_responses, group_messages = create_test_data()
    
    start_time = time.perf_counter()
    
    try:
        total_score, reward_info = await compute_score_async(
            prompt=prompt,
            group_responses=group_responses,
            group_messages=group_messages,
            return_details=False
        )
        
        end_time = time.perf_counter()
        
        print(f"单次调用成功完成，耗时: {end_time - start_time:.4f}秒")
        print(f"总分: {total_score}")
        print(f"奖励信息: {list(reward_info.keys())}")
        
    except Exception as e:
        print(f"单次调用失败: {e}")
        return False
    
    return True


async def test_concurrent_calls():
    """测试并发调用 - 模拟上层manager的多组并发"""
    print("\n" + "=" * 60)
    print("测试2: 并发调用（模拟多组并发）")
    print("=" * 60)
    
    prompt, group_responses, group_messages = create_test_data()
    
    # 创建多个并发任务，模拟AntfinRewardManager的多组并发场景
    num_concurrent = 5
    tasks = []
    
    for i in range(num_concurrent):
        task = compute_score_async(
            prompt=f"{prompt} - 组{i+1}",
            group_responses=group_responses,
            group_messages=group_messages,
            return_details=False
        )
        tasks.append(task)
    
    start_time = time.perf_counter()
    
    try:
        # 并发执行所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.perf_counter()
        
        # 检查结果
        success_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"组{i+1}执行失败: {result}")
            else:
                success_count += 1
                total_score, reward_info = result
                print(f"组{i+1}完成，总分: {total_score}")
        
        print(f"\n并发调用完成:")
        print(f"- 总任务数: {num_concurrent}")
        print(f"- 成功数: {success_count}")
        print(f"- 总耗时: {end_time - start_time:.4f}秒")
        print(f"- 平均每组耗时: {(end_time - start_time) / num_concurrent:.4f}秒")
        
        return success_count == num_concurrent
        
    except Exception as e:
        print(f"并发调用失败: {e}")
        return False


async def test_resource_management():
    """测试资源管理"""
    print("\n" + "=" * 60)
    print("测试3: 资源管理")
    print("=" * 60)
    
    from dr_reward_function2 import GlobalThreadPoolManager
    
    # 获取线程池管理器
    manager = GlobalThreadPoolManager()
    
    print(f"线程池是否已关闭: {manager.is_shutdown()}")
    
    # 测试正常使用
    try:
        executor = manager.get_executor()
        print(f"成功获取线程池执行器: {type(executor)}")
    except Exception as e:
        print(f"获取线程池执行器失败: {e}")
        return False
    
    # 测试一次正常计算
    prompt, group_responses, group_messages = create_test_data()
    
    try:
        total_score, reward_info = await compute_score_async(
            prompt=prompt,
            group_responses=group_responses,
            group_messages=group_messages,
            return_details=False
        )
        print("资源管理测试中的计算成功完成")
    except Exception as e:
        print(f"资源管理测试中的计算失败: {e}")
        return False
    
    return True


async def main():
    """主测试函数"""
    print("开始测试优化后的DR reward计算功能")
    print("=" * 80)
    
    test_results = []
    
    # 测试1: 单次调用
    result1 = await test_single_call()
    test_results.append(("单次调用", result1))
    
    # 测试2: 并发调用
    result2 = await test_concurrent_calls()
    test_results.append(("并发调用", result2))
    
    # 测试3: 资源管理
    result3 = await test_resource_management()
    test_results.append(("资源管理", result3))
    
    # 打印测试结果
    print("\n" + "=" * 80)
    print("测试结果总结")
    print("=" * 80)
    
    for test_name, success in test_results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in test_results)
    
    if all_passed:
        print("\n🎉 所有测试都通过了！")
        print("优化后的DR reward计算功能工作正常")
    else:
        print("\n⚠️  部分测试失败，请检查实现")
    
    # 演示显式清理（可选）
    print("\n" + "=" * 80)
    print("资源清理演示")
    print("=" * 80)
    print("注意：通常情况下不需要手动清理线程池")
    print("只有在程序结束或需要重新初始化时才需要调用")
    
    # 在这里我们不调用cleanup_thread_pool()，因为这只是演示
    # 实际使用中，让调用者决定何时清理
    
    return all_passed


if __name__ == "__main__":
    # 运行测试
    success = asyncio.run(main())
    
    if success:
        print("\n✅ 测试完成，所有功能正常")
    else:
        print("\n❌ 测试发现问题，请检查实现")
        exit(1) 