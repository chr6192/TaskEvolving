import asyncio
import json
import os
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from analysis_compute_reward import compute_score_async


def load_tools_reference_file(query):
    """从JSON文件加载tool_calls_reference"""
    with open(
        "/Users/cindy/Desktop/code/data/processed_multi_tool_update.json", "r"
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


def print_detailed_statistics(reward_info, total_score):
    """打印详细统计信息"""
    print("\n" + "=" * 100)
    print("详细统计分析报告")
    print("=" * 100)

    # 创建统计表格
    stats_data = []

    # 添加总分统计
    total_stats = compute_statistics(total_score)
    stats_data.append(
        [
            "总分",
            total_stats["count"],
            f"{total_stats['mean']:.4f}",
            f"{total_stats['std']:.4f}",
            f"{total_stats['var']:.4f}",
            f"{total_stats['min']:.4f}",
            f"{total_stats['max']:.4f}",
            f"{total_stats['median']:.4f}",
        ]
    )

    # 添加各个reward维度的统计
    reward_names = {
        "faithful_reward": "Faithful Reward",
        "express_reward": "Express Reward",
        "faithful_num_exist_reward": "Faithful Number Exist Reward",
        "map_format_reward": "Map Format Reward",
        "agent_search_reward": "Agent Search Weighted Reward",
        "search_info_reward": "Information Gain Reward",
        "action_loop_reward": "Action Repetition Reward",
        "unnecessary_action_penalty": "Action Necessity Reward",
    }

    for reward_key, scores in reward_info.items():
        if scores:  # 确保分数列表不为空
            stats = compute_statistics(scores)
            reward_name = reward_names.get(reward_key, reward_key)
            stats_data.append(
                [
                    reward_name,
                    stats["count"],
                    f"{stats['mean']:.4f}",
                    f"{stats['std']:.4f}",
                    f"{stats['var']:.4f}",
                    f"{stats['min']:.4f}",
                    f"{stats['max']:.4f}",
                    f"{stats['median']:.4f}",
                ]
            )

    # 打印表格
    headers = [
        "奖励类型",
        "样本数",
        "均值",
        "标准差",
        "方差",
        "最小值",
        "最大值",
        "中位数",
    ]
    col_widths = [15, 8, 10, 10, 10, 10, 10, 10]

    # 打印表头
    header_line = ""
    for i, header in enumerate(headers):
        header_line += f"{header:^{col_widths[i]}}"
    print(header_line)
    print("-" * sum(col_widths))

    # 打印数据行
    for row in stats_data:
        line = ""
        for i, cell in enumerate(row):
            line += f"{str(cell):^{col_widths[i]}}"
        print(line)

    print("=" * 100)


def create_visualizations(reward_info, total_score, output_dir="reward_analysis_plots"):
    """创建可视化图表"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    reward_names = {
        "faithful_reward": "Faithfulness Reward",
        "express_reward": "Expression Reward",
        "faithful_num_exist_reward": "Numeric Rule Reward",
        "map_format_reward": "Format Rule Reward",
        "agent_search_reward": "Agent Search Reward",
        "search_info_reward": "Information Gain Reward",
        "action_loop_reward": "Action Repetition Reward",
        "unnecessary_action_penalty": "Action Necessity Reward",
    }

    # 计算实际需要绘制的子图数量（总分 + 各个维度）
    num_subplots = 1  # 总分图
    for scores in reward_info.values():
        if scores:
            num_subplots += 1

    # 计算合适的子图布局（每行最多3个子图）
    cols = 3
    rows = (num_subplots + cols - 1) // cols  # 向上取整

    # 确保至少有1行
    rows = max(1, rows)

    # 创建分布直方图
    plt.figure(figsize=(6 * cols, 4 * rows))

    # 总分布图
    plt.subplot(rows, cols, 1)
    plt.hist(total_score, bins=30, alpha=0.7, color="navy", edgecolor="black")
    plt.title("Total Score Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)

    # 各个维度分布图
    subplot_idx = 2
    for reward_key, scores in reward_info.items():
        if scores and subplot_idx <= num_subplots:
            plt.subplot(rows, cols, subplot_idx)
            plt.hist(scores, bins=20, alpha=0.7, edgecolor="black")
            reward_name = reward_names.get(reward_key, reward_key)
            plt.title(f"{reward_name} Distribution", fontsize=12)
            plt.xlabel("Score")
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)
            subplot_idx += 1

    plt.tight_layout()
    plt.savefig(f"{output_dir}/reward_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. 创建汇总统计图
    plt.figure(figsize=(15, 10))

    # 准备数据
    means = []
    stds = []
    names = []

    # 总分统计
    total_stats = compute_statistics(total_score)
    means.append(total_stats["mean"])
    stds.append(total_stats["std"])
    names.append("Total Score")

    # 各维度统计
    for reward_key, scores in reward_info.items():
        if scores:
            stats = compute_statistics(scores)
            means.append(stats["mean"])
            stds.append(stats["std"])
            reward_name = reward_names.get(reward_key, reward_key)
            names.append(reward_name)

    # 创建双y轴图
    fig, ax1 = plt.subplots(figsize=(15, 8))

    x_pos = np.arange(len(names))

    # 均值柱状图
    bars1 = ax1.bar(x_pos, means, alpha=0.7, color="skyblue", label="Mean")
    ax1.set_xlabel("Reward Dimensions", fontsize=12)
    ax1.set_ylabel("Mean", color="blue", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names, rotation=45, ha="right")

    # 在柱状图上显示数值
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.001,
            f"{means[i]:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # 标准差折线图
    ax2 = ax1.twinx()
    line1 = ax2.plot(
        x_pos, stds, color="red", marker="o", linewidth=2, label="Standard Deviation"
    )
    ax2.set_ylabel("Standard Deviation", color="red", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="red")

    # 在标准差点上显示数值
    for i, std in enumerate(stds):
        ax2.text(
            x_pos[i],
            std + 0.001,
            f"{std:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="red",
        )

    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title(
        "Mean and Standard Deviation Comparison Across Reward Dimensions",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/reward_mean_std_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"\n可视化图表已保存到目录: {output_dir}/")
    print("生成的图表包括:")
    print("- reward_distributions.png: 分布直方图")
    print("- reward_mean_std_comparison.png: 均值与标准差对比图")


def save_detailed_results(
    reward_info,
    total_score,
    reward_details,
    output_file="detailed_reward_analysis.json",
):
    """保存详细结果到JSON文件"""

    # 计算统计信息
    stats_summary = {}

    # 总分统计
    stats_summary["total_score"] = compute_statistics(total_score)

    # 各维度统计
    stats_summary["reward_dimensions"] = {}
    for reward_key, scores in reward_info.items():
        if scores:
            stats_summary["reward_dimensions"][reward_key] = compute_statistics(scores)

    # 构建完整结果
    results = {
        "summary_statistics": stats_summary,
        # "raw_scores": {"total_score": total_score, "reward_info": reward_info},
        "detailed_results": reward_details if reward_details else [],
        "analysis_metadata": {
            "total_samples": len(total_score),
            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reward_dimensions_count": len(reward_info),
        },
    }

    # 保存到文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4, default=str)

    print(f"\n详细分析结果已保存到: {output_file}")


async def analysis_reward():
    # 读取对话数据，已计算的文件
    jsonl_file_path = "/Users/cindy/Desktop/data/rollout/Qwen3-8B-20250814-115823_w_penalty/rollout_data/1.jsonl"
    all_data = []

    with open(jsonl_file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                all_data.append(data)
            except json.JSONDecodeError as e:
                print(f"警告: 第{line_num}行JSON解析错误: {e}")
                continue

    print(f"成功加载 {len(all_data)} 条数据")

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

    print(f"按query分组后，共有 {len(query_groups)} 个不同的query")
    # 汇总所有结果
    all_total_scores = []
    all_reward_info = defaultdict(list)
    all_reward_details = []

    start_time = time.perf_counter()

    # 对每个query组分别计算奖励
    for query_idx, (query, query_data_list) in enumerate(query_groups.items()):
        print(f"\n处理第 {query_idx + 1}/{len(query_groups)} 个query")

        # 为当前query准备数据
        group_responses = [data.get("output", "") for data in query_data_list]
        group_messages = [data.get("messages", []) for data in query_data_list]
        rollout_data = query_data_list
        tool_calls_reference = load_tools_reference_file(query)

        print(f"当前query有 {len(group_responses)} 个样本")
        print(f"tool_calls_reference：{tool_calls_reference}")

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
        for reward_key, scores in reward_info.items():
            all_reward_info[reward_key].extend(scores)
        all_reward_details.extend(reward_details)

    end_time = time.perf_counter()
    print(
        f"\n所有奖励计算完成，总耗时: {end_time - start_time:.4f}秒，总样本数: {len(all_total_scores)}"
    )

    # 执行详细统计分析
    print_detailed_statistics(all_reward_info, all_total_scores)

    # 创建可视化图表
    print("\n开始生成可视化图表...")
    create_visualizations(all_reward_info, all_total_scores)

    # 保存详细结果
    print("\n保存详细分析结果...")
    save_detailed_results(all_reward_info, all_total_scores, all_reward_details)

    print("\n分析完成！")
    print(f"样本总数: {len(all_total_scores)}")
    print(f"奖励维度数: {len(all_reward_info)}")
    print(f"详细信息条目数: {len(all_reward_details) if all_reward_details else 0}")


if __name__ == "__main__":
    asyncio.run(analysis_reward())
