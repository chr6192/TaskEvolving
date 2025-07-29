import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import asyncio
from analysis_compute_reward import compute_score_async


def compute_statistics(scores):
    """计算分数的统计信息"""
    if not scores:
        return {
            "mean": 0.0,
            "std": 0.0,
            "var": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "count": 0,
        }
    scores_array = np.array(scores)
    return {
        "mean": np.mean(scores_array),
        "std": np.std(scores_array),
        "var": np.var(scores_array),
        "min": np.min(scores_array),
        "max": np.max(scores_array),
        "median": np.median(scores_array),
        "count": len(scores_array),
    }


async def analyze_steps_reward(directory_path):
    """分析不同步骤的奖励均值"""

    # 获取目录下所有jsonl文件并按数字排序
    jsonl_files = [f for f in os.listdir(directory_path) if f.endswith(".jsonl")]
    jsonl_files.sort(key=lambda x: int(x.split(".")[0]))  # 按照文件名中的数字排序

    # 存储每个步骤的奖励统计信息
    step_rewards = defaultdict(list)  # {reward_type: [step1_mean, step2_mean, ...]}
    steps = []  # 存储步骤编号

    # 奖励维度名称映射
    reward_names = {
        "total_score": "Total Score",
        "faithful_reward": "Faithful Reward",
        "express_reward": "Express Reward",
        "faithful_num_exist_reward": "Faithful Number Exist Reward",
        "map_format_reward": "Map Format Reward",
        "agent_search_reward": "Agent Search Weighted Reward",
        "search_info_reward": "Information Gain Reward",
        "action_loop_reward": "Action Repetition Reward",
        "unnecessary_action_penalty": "Action Necessity Reward",
    }

    print(f"找到 {len(jsonl_files)} 个步骤文件")

    # 逐个处理每个步骤文件
    for i, filename in enumerate(jsonl_files):
        step_num = int(filename.split(".")[0])
        steps.append(step_num)
        file_path = os.path.join(directory_path, filename)

        print(f"处理步骤 {step_num} ({filename})...")

        # 读取文件数据
        group_responses = []
        group_messages = []
        rollout_data = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    messages = data.get("messages", [])
                    response = data.get("output", "")
                    group_responses.append(response)
                    group_messages.append(messages)
                    rollout_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"警告: 文件 {filename} 第{line_num}行JSON解析错误: {e}")
                    continue

        print(f"  成功加载 {len(group_responses)} 条数据")

        # 计算奖励分数
        prompt = ""
        total_score, reward_info, reward_details = await compute_score_async(
            prompt,
            group_responses,
            group_messages,
            rollout_data=rollout_data,
            return_details=True,
        )

        # 存储每个奖励维度的均值
        for reward_key, scores in reward_info.items():
            stats = compute_statistics(scores)
            step_rewards[reward_key].append(stats["mean"])

        # 添加总分维度
        stats = compute_statistics(total_score)
        step_rewards["total_score"].append(stats["mean"])

        print(f"  步骤 {step_num} 处理完成")

    # 创建可视化图表
    print("\n开始生成步骤奖励均值图表...")
    create_step_visualizations(step_rewards, steps, reward_names)

    return step_rewards, steps


def create_step_visualizations(step_rewards, steps, reward_names, output_dir="step_reward_analysis_plots"):
    """创建步骤奖励均值图表"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 计算实际需要绘制的子图数量
    num_subplots = len(step_rewards)

    # 计算合适的子图布局（每行最多3个子图）
    cols = 3
    rows = (num_subplots + cols - 1) // cols  # 向上取整

    # 确保至少有1行
    rows = max(1, rows)

    # 创建图表
    plt.figure(figsize=(6 * cols, 4 * rows))

    subplot_idx = 1
    for reward_key, means in step_rewards.items():
        if means and subplot_idx <= num_subplots:
            plt.subplot(rows, cols, subplot_idx)
            plt.plot(steps, means, marker="o", linewidth=2, markersize=6)
            reward_name = reward_names.get(reward_key, reward_key)
            plt.title(f"{reward_name} Mean", fontsize=12)
            plt.xlabel("Step")
            plt.ylabel("Reward Mean")
            plt.grid(True, alpha=0.3)
            subplot_idx += 1

    plt.tight_layout()
    plt.savefig(f"{output_dir}/step_reward_means.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 创建汇总图 - 所有奖励维度在同一图中
    plt.figure(figsize=(12, 8))
    for reward_key, means in step_rewards.items():
        if means:
            reward_name = reward_names.get(reward_key, reward_key)
            plt.plot(steps, means, marker="o", linewidth=2, markersize=6, label=reward_name)

    plt.title("Reward Dimension Means Across Steps", fontsize=16, fontweight="bold")
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Reward Mean", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/step_reward_means_combined.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"步骤奖励分析图表已保存到目录: {output_dir}/")
    print("生成的图表包括:")
    print("- step_reward_means.png: 各奖励维度均值子图")
    print("- step_reward_means_combined.png: 所有奖励维度均值汇总图")


async def main():
    # 指定包含JSONL文件的目录
    directory_path = "/Users/cindy/Desktop/data/rollout_data"

    # 分析步骤奖励
    step_rewards, steps = await analyze_steps_reward(directory_path)

    print("\n分析完成！")
    print(f"总共处理了 {len(steps)} 个步骤: {steps}")
    print(f"分析了 {len(step_rewards)} 个奖励维度")


if __name__ == "__main__":
    asyncio.run(main())
