import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime

# 脚本调用说明：

# TODOs:
# 1. 工具调用行为统计
# 调用频率分析：每个batch内工具调用总次数分布、平均调用次数 - 热力图：Step × 工具类型，显示使用频率
# 工具类型分布：不同工具（如search）的使用频率和占比 - 饼图：当前batch的工具类型分布
# 调用成功率：成功vs失败的工具调用比例，错误类型分布 - 时间序列：跨step的调用成功率、平均调用次数趋势

# 2. 工具调用模式分析
# 调用深度：单个对话中的工具调用链长度分布
# 调用序列模式：连续调用、重试调用、错误后修正等模式
# 参数复杂度：工具参数的丰富度（如arguments字段的复杂性）

# 3. 工具调用与奖励关系
# 调用效果分析：工具调用次数/成功率与各种reward的相关性 - 散点图：工具调用次数 vs 总奖励分数，按step着色
# 工具价值评估：不同工具对final score的贡献度
# 错误成本：工具调用失败对总体奖励的影响

# 4. 学习进度分析
# 跨step演化：工具使用策略随训练步数的变化趋势 - 箱线图：不同step间工具调用次数的分布差异
# 错误率衰减：工具调用错误率是否随训练减少 - 堆叠柱状图：每个step的成功/失败调用分布
# 策略收敛：工具选择和参数设置是否趋于稳定

# 5. 异常行为检测
# 异常调用模式：过度调用、无效重试等
# 参数错误模式：常见的参数格式错误、取值错误

# 6. 模式挖掘
# 工具调用序列流向（工具A → 工具B → 结果）- 桑基图
# 工具调用的转移关系 - 网络图
# 工具调用错误演化 - 热力图：参数错误类型 × step，

class BatchToolCallsAnalyzer:
    """批次工具调用分析器"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.step_data = {}  # {step: [samples]}
        self.analysis_results = {}
        
    def load_data(self) -> None:
        """加载所有步骤的jsonl数据"""
        jsonl_files = glob.glob(str(self.data_dir / "*.jsonl"))
        print(f"Found {len(jsonl_files)} jsonl files")
        
        for file_path in jsonl_files:
            # 从文件名提取步骤号，假设格式为 step_xx.jsonl 或 xx.jsonl
            filename = Path(file_path).stem
            if filename.isdigit():
                step = int(filename)
            elif filename.startswith('step_'):
                step = int(filename.split('_')[1])
            else:
                continue
                
            samples = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
            
            self.step_data[step] = samples
            print(f"Loaded step {step}: {len(samples)} samples")
    
    def extract_tool_calls_info(self, sample: Dict) -> Dict:
        """从单个样本中提取工具调用信息"""
        info = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'tools_used': [],
            'call_sequence': [],
            'error_types': [],
            'call_depths': 0,
            'max_retries': 0,
            'avg_call_time': 0
        }
        
        messages = sample.get('messages', [])
        tool_call_id_to_success = {}
        
        # 统计工具调用
        for msg in messages:
            if msg.get('role') == 'assistant' and 'tool_calls' in msg:
                tool_calls = msg.get('tool_calls', [])
                info['total_calls'] += len(tool_calls)
                
                for call in tool_calls:
                    tool_name = call.get('function', {}).get('name')
                    if tool_name:
                        info['tools_used'].append(tool_name)
                        info['call_sequence'].append(tool_name)
                        tool_call_id_to_success[call.get('id')] = False  # 默认失败
            
            elif msg.get('role') == 'tool':
                # 检查工具调用结果
                call_id = msg.get('tool_call_id')
                content = msg.get('content', '')
                
                if call_id in tool_call_id_to_success:
                    # 简单判断：包含error字段的为失败
                    if 'error' in content.lower():
                        info['failed_calls'] += 1
                        # 提取错误类型
                        if '入参只能属于' in content:
                            info['error_types'].append('invalid_parameter')
                        else:
                            info['error_types'].append('general_error')
                    else:
                        info['successful_calls'] += 1
                        tool_call_id_to_success[call_id] = True
        
        # 计算调用深度（连续调用链长度）
        info['call_depths'] = len(info['call_sequence'])
        
        # 计算重试次数（同一工具连续调用）
        retries = 0
        for i in range(1, len(info['call_sequence'])):
            if info['call_sequence'][i] == info['call_sequence'][i-1]:
                retries += 1
        info['max_retries'] = retries
        
        return info
    
    def analyze_batch(self, step: int) -> Dict:
        """分析单个batch的工具调用情况"""
        if step not in self.step_data:
            return {}
        
        samples = self.step_data[step]
        batch_stats = {
            'step': step,
            'total_samples': len(samples),
            'samples_with_tools': 0,
            'total_tool_calls': 0,
            'avg_calls_per_sample': 0,
            'success_rate': 0,
            'tool_distribution': Counter(),
            'error_distribution': Counter(),
            'call_depth_stats': [],
            'retry_stats': [],
            'reward_stats': {
                'avg_total_score': 0,
                'avg_faithful_reward': 0,
                'avg_express_reward': 0
            }
        }
        
        all_tool_info = []
        reward_scores = []
        
        for sample in samples:
            tool_info = self.extract_tool_calls_info(sample)
            all_tool_info.append(tool_info)
            
            # 奖励统计
            reward_scores.append({
                'total_score': sample.get('score', 0),
                'faithful_reward': sample.get('faithful_reward', 0),
                'express_reward': sample.get('express_reward', 0)
            })
            
            if tool_info['total_calls'] > 0:
                batch_stats['samples_with_tools'] += 1
                batch_stats['total_tool_calls'] += tool_info['total_calls']
                batch_stats['tool_distribution'].update(tool_info['tools_used'])
                batch_stats['error_distribution'].update(tool_info['error_types'])
                batch_stats['call_depth_stats'].append(tool_info['call_depths'])
                batch_stats['retry_stats'].append(tool_info['max_retries'])
        
        # 计算统计指标
        if batch_stats['samples_with_tools'] > 0:
            batch_stats['avg_calls_per_sample'] = batch_stats['total_tool_calls'] / batch_stats['samples_with_tools']
        
        total_successful = sum(info['successful_calls'] for info in all_tool_info)
        total_calls = sum(info['total_calls'] for info in all_tool_info)
        if total_calls > 0:
            batch_stats['success_rate'] = total_successful / total_calls
        
        # 奖励统计
        if reward_scores:
            batch_stats['reward_stats']['avg_total_score'] = np.mean([r['total_score'] for r in reward_scores])
            batch_stats['reward_stats']['avg_faithful_reward'] = np.mean([r['faithful_reward'] for r in reward_scores])
            batch_stats['reward_stats']['avg_express_reward'] = np.mean([r['express_reward'] for r in reward_scores])
        
        return batch_stats
    
    def analyze_all_steps(self) -> None:
        """分析所有步骤"""
        for step in sorted(self.step_data.keys()):
            self.analysis_results[step] = self.analyze_batch(step)
            print(f"Analyzed step {step}")
    
    def create_visualizations(self, output_dir: str = "analysis_output") -> None:
        """生成可视化图表"""
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.analysis_results:
            print("No analysis results found. Run analyze_all_steps() first.")
            return
        
        # 设置matplotlib中文支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        self._plot_overview_metrics(output_dir)
        self._plot_tool_usage_trends(output_dir)
        self._plot_success_rate_trends(output_dir)
        self._plot_reward_correlation(output_dir)
        self._plot_error_analysis(output_dir)
        self._plot_call_depth_distribution(output_dir)
        
        print(f"All visualizations saved to {output_dir}/")
    
    def _plot_overview_metrics(self, output_dir: str) -> None:
        """概览指标图表"""
        steps = sorted(self.analysis_results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Tool Calls Overview Metrics', fontsize=16)
        
        # 1. 工具调用总数趋势
        total_calls = [self.analysis_results[s]['total_tool_calls'] for s in steps]
        axes[0, 0].plot(steps, total_calls, marker='o', linewidth=2)
        axes[0, 0].set_title('Total Tool Calls per Step')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Total Calls')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 成功率趋势
        success_rates = [self.analysis_results[s]['success_rate'] * 100 for s in steps]
        axes[0, 1].plot(steps, success_rates, marker='s', color='green', linewidth=2)
        axes[0, 1].set_title('Tool Call Success Rate')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Success Rate (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 平均调用次数
        avg_calls = [self.analysis_results[s]['avg_calls_per_sample'] for s in steps]
        axes[1, 0].plot(steps, avg_calls, marker='^', color='orange', linewidth=2)
        axes[1, 0].set_title('Average Calls per Sample')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Avg Calls')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 使用工具的样本比例
        tool_sample_ratio = [self.analysis_results[s]['samples_with_tools'] / self.analysis_results[s]['total_samples'] * 100 for s in steps]
        axes[1, 1].plot(steps, tool_sample_ratio, marker='d', color='purple', linewidth=2)
        axes[1, 1].set_title('Samples with Tool Calls (%)')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Percentage (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/overview_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_tool_usage_trends(self, output_dir: str) -> None:
        """工具使用趋势"""
        steps = sorted(self.analysis_results.keys())
        
        # 收集所有工具类型
        all_tools = set()
        for step_result in self.analysis_results.values():
            all_tools.update(step_result['tool_distribution'].keys())
        
        if not all_tools:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for tool in all_tools:
            tool_counts = []
            for step in steps:
                count = self.analysis_results[step]['tool_distribution'].get(tool, 0)
                tool_counts.append(count)
            ax.plot(steps, tool_counts, marker='o', label=tool, linewidth=2)
        
        ax.set_title('Tool Usage Trends Across Training Steps', fontsize=14)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Usage Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/tool_usage_trends.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_success_rate_trends(self, output_dir: str) -> None:
        """成功率详细分析"""
        steps = sorted(self.analysis_results.keys())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 成功率vs总奖励
        success_rates = [self.analysis_results[s]['success_rate'] for s in steps]
        total_scores = [self.analysis_results[s]['reward_stats']['avg_total_score'] for s in steps]
        
        ax1.scatter(success_rates, total_scores, c=steps, cmap='viridis', s=100)
        ax1.set_xlabel('Tool Call Success Rate')
        ax1.set_ylabel('Average Total Score')
        ax1.set_title('Success Rate vs Reward Score')
        ax1.grid(True, alpha=0.3)
        
        # 双轴图：成功率和奖励趋势
        ax2_twin = ax2.twinx()
        
        line1 = ax2.plot(steps, [r * 100 for r in success_rates], 'b-o', label='Success Rate (%)')
        line2 = ax2_twin.plot(steps, total_scores, 'r-s', label='Avg Total Score')
        
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Success Rate (%)', color='b')
        ax2_twin.set_ylabel('Average Total Score', color='r')
        ax2.set_title('Success Rate & Reward Trends')
        ax2.grid(True, alpha=0.3)
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/success_rate_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_reward_correlation(self, output_dir: str) -> None:
        """奖励相关性分析"""
        steps = sorted(self.analysis_results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Tool Calls vs Reward Correlation', fontsize=16)
        
        # 数据准备
        total_calls = [self.analysis_results[s]['total_tool_calls'] for s in steps]
        success_rates = [self.analysis_results[s]['success_rate'] for s in steps]
        total_scores = [self.analysis_results[s]['reward_stats']['avg_total_score'] for s in steps]
        faithful_scores = [self.analysis_results[s]['reward_stats']['avg_faithful_reward'] for s in steps]
        express_scores = [self.analysis_results[s]['reward_stats']['avg_express_reward'] for s in steps]
        
        # 1. 调用次数 vs 总分
        axes[0, 0].scatter(total_calls, total_scores, c=steps, cmap='viridis', s=80)
        axes[0, 0].set_xlabel('Total Tool Calls')
        axes[0, 0].set_ylabel('Avg Total Score')
        axes[0, 0].set_title('Tool Calls vs Total Score')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 成功率 vs 忠实度奖励
        axes[0, 1].scatter(success_rates, faithful_scores, c=steps, cmap='plasma', s=80)
        axes[0, 1].set_xlabel('Success Rate')
        axes[0, 1].set_ylabel('Avg Faithful Reward')
        axes[0, 1].set_title('Success Rate vs Faithful Reward')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 调用次数 vs 表达奖励
        axes[1, 0].scatter(total_calls, express_scores, c=steps, cmap='coolwarm', s=80)
        axes[1, 0].set_xlabel('Total Tool Calls')
        axes[1, 0].set_ylabel('Avg Express Reward')
        axes[1, 0].set_title('Tool Calls vs Express Reward')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 奖励成分对比
        x_pos = np.arange(len(steps))
        width = 0.35
        
        axes[1, 1].bar(x_pos - width/2, faithful_scores, width, label='Faithful', alpha=0.8)
        axes[1, 1].bar(x_pos + width/2, express_scores, width, label='Express', alpha=0.8)
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Reward Score')
        axes[1, 1].set_title('Reward Components Comparison')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(steps)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/reward_correlation.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_error_analysis(self, output_dir: str) -> None:
        """错误分析"""
        steps = sorted(self.analysis_results.keys())
        
        # 收集所有错误类型
        all_errors = set()
        for step_result in self.analysis_results.values():
            all_errors.update(step_result['error_distribution'].keys())
        
        if not all_errors:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 错误趋势图
        for error_type in all_errors:
            error_counts = []
            for step in steps:
                count = self.analysis_results[step]['error_distribution'].get(error_type, 0)
                error_counts.append(count)
            ax1.plot(steps, error_counts, marker='o', label=error_type, linewidth=2)
        
        ax1.set_title('Error Types Trends', fontsize=14)
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Error Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 错误率 vs 成功率
        error_rates = []
        success_rates = []
        for step in steps:
            total_errors = sum(self.analysis_results[step]['error_distribution'].values())
            total_calls = self.analysis_results[step]['total_tool_calls']
            error_rate = total_errors / total_calls if total_calls > 0 else 0
            error_rates.append(error_rate)
            success_rates.append(self.analysis_results[step]['success_rate'])
        
        ax2.scatter(error_rates, success_rates, c=steps, cmap='coolwarm', s=100)
        ax2.set_xlabel('Error Rate')
        ax2.set_ylabel('Success Rate')
        ax2.set_title('Error Rate vs Success Rate')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/error_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_call_depth_distribution(self, output_dir: str) -> None:
        """调用深度分布"""
        steps = sorted(self.analysis_results.keys())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 平均调用深度趋势
        avg_depths = []
        for step in steps:
            depths = self.analysis_results[step]['call_depth_stats']
            avg_depth = np.mean(depths) if depths else 0
            avg_depths.append(avg_depth)
        
        ax1.plot(steps, avg_depths, marker='o', color='darkblue', linewidth=2)
        ax1.set_title('Average Call Depth Trend', fontsize=14)
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Average Call Depth')
        ax1.grid(True, alpha=0.3)
        
        # 2. 调用深度分布热力图
        if len(steps) > 1:
            max_depth = max(max(self.analysis_results[s]['call_depth_stats'], default=0) for s in steps)
            if max_depth > 0:
                depth_matrix = []
                for step in steps:
                    depths = self.analysis_results[step]['call_depth_stats']
                    depth_dist = np.zeros(max_depth + 1)
                    for d in depths:
                        if d <= max_depth:
                            depth_dist[d] += 1
                    depth_matrix.append(depth_dist)
                
                depth_matrix = np.array(depth_matrix)
                im = ax2.imshow(depth_matrix.T, cmap='YlOrRd', aspect='auto')
                ax2.set_title('Call Depth Distribution', fontsize=14)
                ax2.set_xlabel('Training Step')
                ax2.set_ylabel('Call Depth')
                ax2.set_xticks(range(len(steps)))
                ax2.set_xticklabels(steps)
                ax2.set_yticks(range(max_depth + 1))
                plt.colorbar(im, ax=ax2, label='Count')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/call_depth_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self, output_dir: str = "analysis_output") -> None:
        """生成总结报告"""
        if not self.analysis_results:
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        steps = sorted(self.analysis_results.keys())
        
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_steps_analyzed': len(steps),
            'step_range': f"{min(steps)} - {max(steps)}",
            'key_metrics': {
                'avg_success_rate': float(np.mean([self.analysis_results[s]['success_rate'] for s in steps])),
                'avg_calls_per_sample': float(np.mean([self.analysis_results[s]['avg_calls_per_sample'] for s in steps])),
                'total_tool_calls': int(sum([self.analysis_results[s]['total_tool_calls'] for s in steps]))
            },
            'trends': {
                'success_rate_improving': bool(self._is_trend_improving([self.analysis_results[s]['success_rate'] for s in steps])),
                'call_volume_stable': bool(self._is_trend_stable([self.analysis_results[s]['total_tool_calls'] for s in steps]))
            }
        }
        
        with open(f"{output_dir}/analysis_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def _is_trend_improving(self, values: List[float]) -> bool:
        """判断趋势是否改善"""
        if len(values) < 2:
            return False
        return values[-1] > values[0]
    
    def _is_trend_stable(self, values: List[float]) -> bool:
        """判断趋势是否稳定"""
        if len(values) < 3:
            return True
        return bool(np.std(values) / np.mean(values) < 0.3)  # 变异系数小于30%


def main():
    """主程序入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch Tool Calls Analysis')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing jsonl files')
    parser.add_argument('--output_dir', type=str, default='analysis_output', help='Output directory for results')
    
    args = parser.parse_args()
    
    # 初始化分析器
    analyzer = BatchToolCallsAnalyzer(args.data_dir)
    
    # 执行分析流程
    print("Loading data...")
    analyzer.load_data()
    
    print("Analyzing all steps...")
    analyzer.analyze_all_steps()
    
    print("Creating visualizations...")
    analyzer.create_visualizations(args.output_dir)
    
    print("Generating summary report...")
    analyzer.generate_summary_report(args.output_dir)
    
    print(f"Analysis completed. Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
