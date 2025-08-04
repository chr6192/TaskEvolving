import asyncio
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from analysis_compute_reward import compute_score_async


def load_tools_reference_file(query):
    """从JSON文件加载tool_calls_reference"""
    with open(
        "/Users/cindy/Desktop/code/data/processed_multi_tool_result.json", "r"
    ) as f:
        tool_data = json.load(f)

    # 在tool_data中查找匹配的query
    matching_entry = next(
        (entry for entry in tool_data if entry.get("query") == query), None
    )
    if matching_entry and "tool_calls_list" in matching_entry:
        tool_calls_reference = matching_entry["tool_calls_list"]
    else:
        tool_calls_reference = []

    return tool_calls_reference


def compute_statistics(scores):
    """计算分数的统计信息"""
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
        all_data = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    all_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"警告: 文件 {filename} 第{line_num}行JSON解析错误: {e}")
                    continue

        print(f"  成功加载 {len(all_data)} 条数据")

        # 按query分组数据
        query_groups = defaultdict(list)
        for data in all_data:
            # 提取query（与analysis_compute_reward.py中的逻辑保持一致）
            content = data["messages"][0]["content"]
            if "当前问题：" in content:
                content_parts = content.split("当前问题：")
                if len(content_parts) > 1:
                    query = content_parts[1].strip()
                else:
                    query = content.strip()
            else:
                query = content.strip()
            query_groups[query].append(data)

        print(f"  按query分组后，共有 {len(query_groups)} 个不同的query")

        # 汇总当前步骤的所有奖励分数
        all_total_scores = []
        all_reward_info = defaultdict(list)

        # 对每个query组分别计算奖励
        for query_idx, (query, query_data_list) in enumerate(query_groups.items()):
            # 为当前query准备数据
            group_responses = [data.get("output", "") for data in query_data_list]
            group_messages = [data.get("messages", []) for data in query_data_list]
            rollout_data = query_data_list
            tool_calls_reference = load_tools_reference_file(query)

            # 计算奖励分数
            total_score, reward_info, reward_details = await compute_score_async(
                query,  # 传入当前query作为prompt
                group_responses,
                group_messages,
                rollout_data=rollout_data,
                group_extras={
                    "tool_calls_reference": tool_calls_reference,
                },
                return_details=True,
            )

            # 汇总结果
            all_total_scores.extend(total_score)
            all_reward_info["total_score"].extend(total_score)
            for reward_key, scores in reward_info.items():
                all_reward_info[reward_key].extend(scores)

        # 添加总分维度
        stats = compute_statistics(all_total_scores)

        # 存储每个奖励维度的均值
        for reward_key, scores in all_reward_info.items():
            stats = compute_statistics(scores)
            step_rewards[reward_key].append(stats["mean"])

        print(f"  步骤 {step_num} 处理完成，总样本数: {len(all_total_scores)}")

    # 创建可视化图表
    print("\n开始生成步骤奖励均值图表...")
    create_step_visualizations(step_rewards, steps, reward_names)

    return step_rewards, steps


def create_step_visualizations(
    step_rewards, steps, reward_names, output_dir="step_reward_analysis_plots"
):
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
            plt.plot(steps, means, marker="o", linewidth=2, markersize=4)
            reward_name = reward_names.get(reward_key, reward_key)
            plt.title(f"{reward_name}", fontsize=16)
            plt.xlabel("Step", fontsize=16)
            plt.ylabel("Reward Mean", fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
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
            plt.plot(
                steps, means, marker="o", linewidth=2, markersize=4, label=reward_name
            )

    plt.title("Reward Dimension Means Across Steps", fontsize=16, fontweight="bold")
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Reward Mean", fontsize=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/step_reward_means_combined.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"步骤奖励分析图表已保存到目录: {output_dir}/")
    print("生成的图表包括:")
    print("- step_reward_means.png: 各奖励维度均值子图")
    print("- step_reward_means_combined.png: 所有奖励维度均值汇总图")


async def main():
    # 指定包含JSONL文件的目录
    directory_path = "/Users/cindy/Desktop/data/rollout/tool_rollout/rollout_data/"

    # 分析步骤奖励
    step_rewards, steps = await analyze_steps_reward(directory_path)

    print("\n分析完成！")
    print(f"总共处理了 {len(steps)} 个步骤: {steps}")
    print(f"分析了 {len(step_rewards)} 个奖励维度")


if __name__ == "__main__":
    asyncio.run(main())
