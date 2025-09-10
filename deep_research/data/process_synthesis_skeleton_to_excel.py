import json
import pandas as pd
from pathlib import Path

skeleton_path_lst = [
    "/Users/chr/Dev/synthesis_skeleton/deepseek-v3.1-20250908-151402/skeletons/skeletons.jsonl",
    "/Users/chr/Dev/synthesis_skeleton/qwen3-max-preview-20250908-151327/skeletons/skeletons.jsonl"
]

# 初始化数据存储
all_data = []

# 读取所有skeleton数据
for skeleton_path in skeleton_path_lst:
    model_name = Path(skeleton_path).parent.parent.name  # 获取模型名称
    
    with open(skeleton_path, 'r', encoding='utf-8') as f:
        for line in f:
            skeleton = json.loads(line)
            
            # 提取基本字段
            row_data = {
                'uid': skeleton.get('uid', ''),
                'query': skeleton.get('query', ''),
                f'analysis_{model_name}': skeleton.get('analysis', ''),
                f'tool_results_{model_name}': json.dumps(skeleton.get('tool_results', []), ensure_ascii=False, indent=4),
            }
            all_data.append(row_data)

# 转换为DataFrame
df = pd.DataFrame(all_data)

# 按skeleton_id和uid分组，合并相同的记录
grouped_data = []
for (uid, query), group in df.groupby(['uid', 'query']):
    merged_row = {
        'uid': uid,
        'query': group['query'].iloc[0],  # 相同字段只取第一个
    }
    
    # 合并不同模型的特有字段
    for col in group.columns:
        if col.startswith(('analysis_', 'tool_calls_reference_', 'tool_results_', 'metadata_')):
            values = group[col].dropna()
            if len(values) > 0:
                merged_row[col] = values.iloc[0]
    
    grouped_data.append(merged_row)

# 创建最终的DataFrame
final_df = pd.DataFrame(grouped_data)

# 保存到Excel
output_path = "skeleton_comparison.xlsx"
final_df.to_excel(output_path, index=False, sheet_name='Skeleton对比')

print(f"数据已保存到 {output_path}")
print(f"共处理 {len(final_df)} 条记录")
