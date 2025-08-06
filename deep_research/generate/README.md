# Deep Research 数据合成Pipeline

## 概述

这是一个基于现有基建的Test-time Scaling数据合成Pipeline，用于解决Deep Research业务场景下缺乏verifiable标注数据的问题。

## 架构设计

### 两阶段架构

1. **阶段一：Skeleton生成**
   - 生成带有递进逻辑的agent tool calls skeleton
   - 输入：用户查询
   - 输出：包含分析思路和工具调用框架的JSON结构

2. **阶段二：Trajectory合成**
   - 基于skeleton通过PRM拒绝采样构造完整trajectory
   - 输入：Skeleton数据
   - 输出：高质量的完整对话轨迹

### 核心组件

```
DataSynthesisPipeline
├── SkeletonGenerator (阶段一)
├── TrajectoryGenerator (阶段二)
│   ├── SkeletonProcessor
│   └── AgentFramework
├── PRMSelector (拒绝采样)
└── ORMEvaluator (质量评估)
```


1. **数据读取和转换**
   - `load_skeleton_data_from_parquet()` - 从parquet文件读取skeleton数据
   - 支持解析`extra_info.skeleton`字段中的结构化数据
   - 自动转换为标准的`SkeletonData`格式

2. **工具管理系统**
   - `ToolManager` - 复用`deep_research/tool/dr_tool.py`中的所有工具
   - 支持异步并发工具调用
   - 集成现有的工具配置`dr_tool_config.yaml`

3. **对话管理系统**
   - `ConversationManager` - 集成verl的`async_rollout_manager`
   - 支持多轮对话生成
   - 包含fallback机制确保系统健壮性

4. **Trajectory生成器**
   - `TrajectoryGenerator` - 完整的轨迹生成逻辑
   - 支持基于skeleton指导的生成
   - 支持多样性参数调整

5. **专用CompletionCallback**
   - `DrSynthesisCompletionCallback` - 继承`DrCompletionCallback`
   - 增加skeleton引导功能
   - 支持占位符解析和工具调用引导



### **使用方法**

#### **方法1：使用示例脚本（推荐）**
```bash
cd deep_research/generate
python run_trajectory_generation.py
```

#### **方法2：使用配置文件**
```bash
cd deep_research/generate
python main_synthesis_clean.py \
  stage=trajectory \
  input.queries_file=deep_research/data/4k_w_skeleton/test_250731.parquet \
  model.path=your_model_path
```

### **配置要点**

1. **模型配置**：需要设置正确的模型路径
2. **工具配置**：自动使用`dr_tool_config.yaml`中的工具
3. **输出目录**：会自动创建`outputs/`目录保存结果
4. **GPU配置**：默认使用1个GPU，可根据需要调整


## 使用方式

### 1. 启动脚本

```bash
# 完整pipeline
./run_data_synthesis.sh both 10 best_of_n 3

# 仅生成skeleton
./run_data_synthesis.sh skeleton

# 仅生成trajectory
./run_data_synthesis.sh trajectory 10 threshold
```

### 2. 配置文件

主配置文件：`config/data_synthesis.yaml`

关键配置项：
- `stage`: 执行阶段 ("skeleton", "trajectory", "both")
- `generation.n_samples_per_skeleton`: 每个skeleton生成的trajectory数量
- `prm.selection_strategy`: PRM选择策略
- `orm.enabled`: 是否启用ORM评估

### 3. 输入数据格式

查询文件 (`.jsonl`):
```json
{"id": "query_1", "content": "帮我分析一下2024年Q2家电行业利润环比上升的公司有哪些，并给出投资建议"}
{"id": "query_2", "content": "最近哪些新能源汽车股票表现较好？请分析一下原因"}
```

### 4. 输出数据格式

Skeleton输出 (`skeletons.jsonl`):
```json
{
  "skeleton_id": "uuid",
  "query_id": "query_1", 
  "analysis_logic": "分析思路...",
  "multi_step": true,
  "tool_results": [
    {
      "step": 0,
      "tool_calls": [
        {
          "name": "search",
          "parameters": {
            "pool": ["金融指标"],
            "query": "2024年Q2家电行业利润环比上升公司",
            "entity": "家电行业"
          }
        }
      ]
    }
  ]
}
```

Trajectory输出 (`trajectories.jsonl`):
```json
{
  "trajectory_id": "uuid",
  "skeleton_id": "uuid",
  "messages": [...],
  "tool_calls": [...],
  "tool_responses": [...],
  "prm_score": 0.85,
  "orm_scores": {
    "faithful_reward": 0.8,
    "express_reward": 0.9
  }
}
```

## 与现有基建的集成

### 复用组件

1. **DrChatCompletionScheduler**: 多轮对话生成
2. **DrCompletionCallback**: 工具调用处理  
3. **AntfinRewardManager**: ORM评估
4. **dr_tool.py**: 工具实现

### 扩展组件

1. **新的CompletionCallback**: 支持skeleton引导
2. **增强的RewardManager**: 支持PRM评估
3. **新的ChatScheduler**: 支持skeleton初始化

## 特性

- ✅ 异步并行处理，支持高并发
- ✅ 模块化设计，易于扩展
- ✅ 完整的错误处理和日志
- ✅ 支持多种选择策略（PRM）
- ✅ 集成现有业务奖励模型（ORM）
- ✅ 支持批量处理和断点续传
- ✅ 丰富的统计信息和质量控制

## 配置示例

```yaml
# 数据合成配置
stage: "both"
generation:
  n_samples_per_skeleton: 10
  temperature: 0.7
  
prm:
  enabled: true
  selection_strategy: "best_of_n"
  top_k: 3
  
orm:
  enabled: true
  async_mode: true
  evaluation_metrics:
    - "faithful_reward"
    - "express_reward"
    
quality_control:
  quality_threshold: 0.3
  filter_invalid_trajectories: true
```

## 执行流程

1. **初始化**: 加载配置，初始化各个组件
2. **阶段一** (可选): 
   - 加载查询数据
   - 生成skeleton 
   - 验证和保存skeleton
3. **阶段二** (可选):
   - 加载skeleton数据
   - 生成候选trajectories
   - PRM筛选
   - ORM评估
   - 保存高质量trajectories
4. **清理**: 释放资源，保存统计信息

## 性能优化

- 异步并发执行，最大化GPU利用率
- 批量处理，减少网络开销
- 内存优化，支持大规模数据处理
- 缓存机制，避免重复计算

---

本framework提供了完整的数据合成解决方案，支持从skeleton生成到高质量trajectory产出的端到端流程。 