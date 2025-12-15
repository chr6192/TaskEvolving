# Reward Compute API 文档

高并发异步奖励计算API服务，基于 FastAPI 构建，支持同步/异步计算模式。

---

## 📋 目录

- [功能特性](#功能特性)
- [快速开始](#快速开始)
- [API使用指南](#api使用指南)
- [配置说明](#配置说明)
- [架构设计](#架构设计)
- [数据持久化](#数据持久化)
- [性能优化](#性能优化)
- [监控和日志](#监控和日志)
- [故障排查](#故障排查)
- [开发指南](#开发指南)

---

## 功能特性

- ✅ **同步计算接口**: 立即返回结果，适合小批量需要即时响应的场景
- ✅ **异步任务接口**: 任务入队后台处理，适合大批量可以等待的场景
- ✅ **批量计算**: 同时处理多个group，充分利用并发能力
- ✅ **结果缓存**: 支持内存缓存和Redis缓存，避免重复计算
- ✅ **任务队列**: 确保所有请求入队而不是直接失败
- ✅ **监控指标**: 完善的性能监控和统计信息
- ✅ **详细日志**: 可配置的日志级别和输出
- ✅ **并发控制**: 三层并发控制（请求级、group级、内部任务级）
- ✅ **数据持久化**: 自动保存计算前后的样本数据，支持离线分析

---

## 快速开始

### 5分钟快速部署

#### 1️⃣ 安装依赖（1分钟）

```bash
# 进入项目根目录
cd /Users/chr/Dev/agentRL

# 安装必需依赖
pip install fastapi uvicorn pydantic

# 可选：如果需要Redis缓存
# pip install redis
```

#### 2️⃣ 配置环境变量（1分钟）

```bash
# 复制配置文件
cp deep_research/reward/api/.env.example deep_research/reward/api/.env

# 编辑配置文件，至少设置以下内容
# vim deep_research/reward/api/.env
```

**最小配置**：
```bash
OPENAI_API_KEY=your_api_key_here
```

**推荐配置**：
```bash
OPENAI_API_KEY=your_api_key_here
MAX_CONCURRENT_GROUPS=10
NUM_TASK_PROCESSORS=4
UVICORN_WORKERS=1
CACHE_TYPE=memory
LOG_LEVEL=INFO
```

#### 3️⃣ 启动服务（1分钟）

```bash
# 方式1: 使用启动脚本（推荐）
bash deep_research/reward/api/start.sh

# 方式2: 直接运行
python -m deep_research.reward.api.main

# 方式3: 开发模式（支持热重载）
bash deep_research/reward/api/start_dev.sh
```

#### 4️⃣ 验证服务（1分钟）

**浏览器访问**：
- 打开 http://localhost:8000/docs 查看API文档
- 访问 http://localhost:8000/health 检查服务状态

**命令行测试**：
```bash
# 健康检查
curl http://localhost:8000/health

# 查看监控指标
curl http://localhost:8000/api/v1/metrics
```

#### 5️⃣ 测试API（1分钟）

```bash
# 运行示例客户端
python deep_research/reward/api/example_client.py
```

### ✅ 完成！

服务现在已经运行在 http://localhost:8000

---

## API使用指南

### 1. 同步计算（立即返回）

```python
import requests

url = "http://localhost:8000/api/v1/rewards/compute"
data = {
    "prompt": "如何购买基金？",
    "responses": [
        "您可以通过手机银行购买基金...",
        "建议您先了解基金知识..."
    ],
    "messages": [
        [
            {"role": "user", "content": "如何购买基金？"},
            {"role": "assistant", "content": "您可以通过手机银行购买基金..."}
        ],
        [
            {"role": "user", "content": "如何购买基金？"},
            {"role": "assistant", "content": "建议您先了解基金知识..."}
        ]
    ],
    "extras": [
        {
            "skeleton_id": "skel_001",
            "trajectory_id": "traj_001",
            "skeleton": {
                "query": "如何购买基金？",
                "tool_calls_reference": []
            },
            "tool_calls": [],
            "tool_responses": [],
            "rubrics": {}
        },
        {
            "skeleton_id": "skel_001",
            "trajectory_id": "traj_002",
            "skeleton": {
                "query": "如何购买基金？",
                "tool_calls_reference": []
            },
            "tool_calls": [],
            "tool_responses": [],
            "rubrics": {}
        }
    ],
    "config": {
        "return_details": True,
        "cache_ttl": 3600
    }
}

response = requests.post(url, json=data)
result = response.json()

print("总分:", result["data"]["total_scores"])
print("奖励详情:", result["data"]["reward_info"])
```

### 2. 异步计算（任务提交）

```python
import requests
import time

# 提交任务
url = "http://localhost:8000/api/v1/rewards/compute_async"
data = {
    "task_id": "my_task_123",  # 可选，自定义任务ID
    "data": {
        "prompt": "如何购买基金？",
        "responses": ["回复1", "回复2"],
        "messages": [[{"role": "user", "content": "问题"}]],
        "extras": [{"skeleton_id": "001"}]
    }
}

response = requests.post(url, json=data)
task_info = response.json()
task_id = task_info["task_id"]

print(f"任务已提交: {task_id}")
print(f"预估完成时间: {task_info['estimated_time']}秒")

# 轮询查询任务状态
while True:
    status_url = f"http://localhost:8000/api/v1/rewards/task/{task_id}"
    status_response = requests.get(status_url)
    status_data = status_response.json()
    
    print(f"状态: {status_data['status']}, 进度: {status_data['progress']:.0%}")
    
    if status_data["status"] == "completed":
        print("任务完成!")
        print("结果:", status_data["result"])
        break
    elif status_data["status"] == "failed":
        print("任务失败:", status_data["error"])
        break
    
    time.sleep(2)  # 等待2秒后再查询
```

### 3. 批量计算（多个group）

```python
import requests

url = "http://localhost:8000/api/v1/rewards/batch_compute"
data = {
    "groups": [
        {
            "group_id": "group_1",
            "prompt": "问题1",
            "responses": ["回复1_1", "回复1_2"],
            "messages": [
                [{"role": "user", "content": "问题1"}],
                [{"role": "user", "content": "问题1"}]
            ],
            "extras": [{"skeleton_id": "001"}, {"skeleton_id": "002"}]
        },
        {
            "group_id": "group_2",
            "prompt": "问题2",
            "responses": ["回复2_1", "回复2_2"],
            "messages": [
                [{"role": "user", "content": "问题2"}],
                [{"role": "user", "content": "问题2"}]
            ],
            "extras": [{"skeleton_id": "003"}, {"skeleton_id": "004"}]
        }
    ],
    "config": {
        "return_details": True
    }
}

response = requests.post(url, json=data)
result = response.json()

print(f"处理了 {result['meta']['total_groups']} 个groups")
print(f"总耗时: {result['meta']['total_processing_time']:.2f}秒")

for group_result in result["results"]:
    print(f"\nGroup {group_result['group_id']}:")
    print(f"  状态: {group_result['status']}")
    if group_result["status"] == "success":
        print(f"  分数: {group_result['total_scores']}")
```

### 4. 监控指标

```python
import requests

url = "http://localhost:8000/api/v1/metrics"
response = requests.get(url)
metrics = response.json()

print("服务指标:")
print(f"  总请求数: {metrics['metrics']['service']['total_requests']}")
print(f"  缓存命中率: {metrics['metrics']['service']['cache_hit_rate']:.2%}")
print(f"  平均计算时间: {metrics['metrics']['service']['avg_compute_time']:.2f}秒")

print("\n队列状态:")
print(f"  队列大小: {metrics['metrics']['queue']['queue_size']}")
print(f"  总任务数: {metrics['metrics']['queue']['total_tasks']}")
print(f"  状态分布: {metrics['metrics']['queue']['status_counts']}")

print("\n持久化状态:")
print(f"  已保存文件数: {metrics['metrics']['service']['persistence']['total_files']}")
print(f"  已保存样本数: {metrics['metrics']['service']['persistence']['total_samples']}")
```

### 5. 清空缓存

```python
import requests

url = "http://localhost:8000/api/v1/cache/clear"
response = requests.post(url)
result = response.json()

print(result["message"])
```

**curl命令**：
```bash
curl -X POST http://localhost:8000/api/v1/cache/clear | jq -r '.message'
```

或（在DSW中）：
```bash
curl -s -X POST http://localhost:8000/api/v1/cache/clear | python3 -c "import sys, json; print(json.load(sys.stdin)['message'])"
```

### 6. 查看持久化数据

```python
import json

# 读取某个样本的所有记录
sample_file = "./data/reward_samples/sample_traj_001.jsonl"
with open(sample_file, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        print(f"时间: {record['timestamp']}")
        print(f"分数: {record['output']['total_score']}")
        print(f"输入: {record['input']['response'][:100]}...")
        print("-" * 80)
```

---

## 配置说明

### 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `API_HOST` | `0.0.0.0` | 服务监听地址 |
| `API_PORT` | `8000` | 服务端口 |
| `MAX_CONCURRENT_GROUPS` | `10` | 最大并发group数量 |
| `NUM_TASK_PROCESSORS` | `4` | 进程内异步任务处理器数量 |
| `UVICORN_WORKERS` | `1` | Uvicorn多进程worker数量 |
| `MAX_QUEUE_SIZE` | `1000` | 最大队列大小 |
| `ENABLE_CACHE` | `true` | 是否启用缓存 |
| `CACHE_TYPE` | `memory` | 缓存类型 (memory/redis) |
| `CACHE_MAX_SIZE` | `10000` | 内存缓存最大条目数 |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis连接URL |
| `ENABLE_PERSISTENCE` | `true` | 是否启用数据持久化 |
| `PERSISTENCE_DIR` | `./data/reward_samples` | 数据持久化目录 |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `OPENAI_API_KEY` | - | OpenAI API Key (必填) |

### 并发控制策略

#### ⚠️ 重要概念区分

**两种"Worker"的区别**：

1. **Uvicorn Workers（多进程）**：
   - 通过 `--workers` 参数或 `UVICORN_WORKERS` 环境变量控制
   - 启动多个独立的进程，每个进程有独立的内存空间
   - **⚠️ 使用内存缓存时必须设为1**，否则任务状态在不同进程间无法共享
   - 使用Redis缓存时可以设为多个，提升并发能力

2. **Task Processors（进程内异步处理器）**：
   - 通过 `NUM_TASK_PROCESSORS` 环境变量控制
   - 进程内的异步协程，共享同一进程的内存
   - 负责从任务队列中取出任务并处理
   - 数量可以根据负载灵活调整

#### 三层并发控制

1. **进程级**: Uvicorn Workers 控制（多进程）
2. **Group级**: `MAX_CONCURRENT_GROUPS` 控制同时处理的group数量（单进程内）
3. **内部任务级**: `compute_score_async` 内部的并发控制

#### 推荐配置

**使用内存缓存时（开发/测试）**：
```bash
UVICORN_WORKERS=1  # 必须为1
NUM_TASK_PROCESSORS=4
MAX_CONCURRENT_GROUPS=10
CACHE_TYPE=memory
```

**使用Redis缓存时（生产环境）**：
- 轻负载: `UVICORN_WORKERS=2`, `NUM_TASK_PROCESSORS=2`, `MAX_CONCURRENT_GROUPS=5`
- 中负载: `UVICORN_WORKERS=4`, `NUM_TASK_PROCESSORS=4`, `MAX_CONCURRENT_GROUPS=10`
- 重负载: `UVICORN_WORKERS=8`, `NUM_TASK_PROCESSORS=8`, `MAX_CONCURRENT_GROUPS=20`

### 缓存策略

**内存缓存** (推荐单机部署):
- 优点: 速度快，无需额外依赖
- 缺点: 重启后丢失，不支持多实例共享
- 配置: `CACHE_TYPE=memory`

**Redis缓存** (可选):
- 优点: 持久化，支持多实例共享
- 缺点: 需要Redis服务，略慢
- 配置: `CACHE_TYPE=redis`, `REDIS_URL=redis://localhost:6379/0`

### 数据持久化配置

服务支持自动保存计算前后的样本数据到本地文件，用于离线分析和调试。

**启用持久化**：
```bash
ENABLE_PERSISTENCE=true
PERSISTENCE_DIR=./data/reward_samples
```

**禁用持久化**：
```bash
ENABLE_PERSISTENCE=false
```

**数据格式**：
- 文件格式: JSONL (每行一个JSON对象)
- 文件命名: `sample_{unique_id}.jsonl`（根据样本的unique_id/trajectory_id/skeleton_id命名）
- 保存方式: 追加写入（append mode）

---

## 架构设计

### 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        Client Layer                          │
│  (HTTP Requests: curl, Python requests, JavaScript, etc.)   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Application                      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Middleware Layer                         │  │
│  │  • Request Logger  • Error Handler  • CORS           │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                  API Routes                           │  │
│  │  • /api/v1/rewards/compute         (同步计算)        │  │
│  │  • /api/v1/rewards/compute_async   (异步提交)        │  │
│  │  • /api/v1/rewards/task/{id}       (任务查询)        │  │
│  │  • /api/v1/rewards/batch_compute   (批量计算)        │  │
│  │  • /api/v1/metrics                 (监控指标)        │  │
│  └───────────────────────────────────────────────────────┘  │
└──────────────────┬──────────────────┬──────────────────────┘
                   │                  │
                   ▼                  ▼
    ┌──────────────────────┐  ┌──────────────────────┐
    │ RewardComputeService │  │     TaskQueue        │
    │                      │  │                      │
    │ • 计算调度            │  │ • 任务管理            │
    │ • 缓存管理            │  │ • 异步任务处理器       │
    │ • 并发控制            │  │ • 状态追踪            │
    │ • 性能监控            │  │ • 队列调度            │
    └──────────┬───────────┘  └──────────┬───────────┘
               │                         │
               ▼                         ▼
    ┌──────────────────────┐  ┌──────────────────────┐
    │   Cache Backend      │  │  Task Processors     │
    │                      │  │  (进程内异步协程)      │
    │ • MemoryCache        │  │ • Processor 1        │
    │ • RedisCache         │  │ • Processor 2        │
    │                      │  │ • Processor N        │
    └──────────────────────┘  └──────────┬───────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │ compute_score_async  │
                              │                      │
                              │ • HybridRewardExecutor │
                              │ • 多维度奖励计算       │
                              │ • IO/CPU任务并发      │
                              └──────────────────────┘
```

### 核心组件

#### 1. FastAPI Application
**职责**：
- HTTP请求处理
- 路由分发
- 错误处理
- 生命周期管理

**特性**：
- 自动生成API文档
- 数据验证（Pydantic）
- 异步请求处理
- 中间件支持

#### 2. RewardComputeService
**职责**：
- 封装 `compute_score_async` 函数
- 管理计算任务的并发执行
- 实现缓存逻辑
- 收集性能指标

**并发控制**：
- 使用 `asyncio.Semaphore` 限制同时处理的group数量
- 默认值: `MAX_CONCURRENT_GROUPS=10`

#### 3. TaskQueue
**职责**：
- 异步任务提交和管理
- 后台工作进程调度
- 任务状态追踪
- 结果持久化

**状态管理**：
- `pending`: 等待处理
- `processing`: 处理中
- `completed`: 已完成
- `failed`: 失败

#### 4. Cache Backend
**职责**：
- 结果缓存
- 减少重复计算
- 提升响应速度

**实现方式**：
- **MemoryCache**: LRU策略，单机部署推荐
- **RedisCache**: 持久化，支持分布式（可选）

**缓存Key生成**：
```python
hash(prompt + responses + messages + extras + config) → cache_key
```

### 数据流

#### 同步计算流程
```
Client Request
    ↓
FastAPI Route (/compute)
    ↓
RewardComputeService.compute_single_group()
    ↓
检查缓存 → [命中] → 返回缓存结果
    ↓ [未命中]
并发控制 (Semaphore)
    ↓
compute_score_async()
    ↓
HybridRewardExecutor
    ↓
并发执行多个reward计算
    ↓
汇总结果
    ↓
保存到缓存
    ↓
返回给Client
```

#### 异步计算流程
```
Client Request
    ↓
FastAPI Route (/compute_async)
    ↓
TaskQueue.submit_task()
    ↓
任务入队 + 返回task_id
    ↓
Client收到task_id
    │
    └─→ 后台处理 ←──────┐
      TaskProcessor    │
           ↓           │
    RewardService      │
           ↓           │
    更新状态: processing
           ↓
    compute_score_async()
           ↓
    保存结果
           ↓
    更新状态: completed
           ↓
Client查询 (/task/{id})
    ↓
返回结果
```

---

## 数据持久化

### 功能说明

服务自动保存每次奖励计算的完整输入输出数据，用于：
- 离线分析和调试
- 模型训练数据收集
- 结果追溯和审计

### 数据结构

每个样本保存为一个JSON对象，包含以下字段：

```json
{
  "timestamp": "2025-12-15T10:30:00.123456",
  "unique_id": "traj_001",
  "input": {
    "prompt": "如何购买基金？",
    "response": "您可以通过手机银行购买基金...",
    "messages": [
      {"role": "user", "content": "如何购买基金？"},
      {"role": "assistant", "content": "您可以通过手机银行购买基金..."}
    ],
    "extra": {
      "skeleton_id": "skel_001",
      "trajectory_id": "traj_001",
      "skeleton": {...},
      "tool_calls": [...],
      "tool_responses": [...]
    }
  },
  "output": {
    "total_score": 0.85,
    "reward_info": {
      "faithful_num_exist_reward": 0.1,
      "map_format_reward": 0.3,
      "rubrics_reward": 0.45
    },
    "details": {
      "performance_stats": {...}
    }
  }
}
```

### 文件组织

```
./data/reward_samples/
├── sample_traj_001.jsonl    # trajectory_id = traj_001 的所有样本
├── sample_traj_002.jsonl    # trajectory_id = traj_002 的所有样本
├── sample_skel_003.jsonl    # skeleton_id = skel_003 的所有样本
└── ...
```

**命名规则**：
- 文件名格式: `sample_{unique_id}.jsonl`
- `unique_id` 按优先级从 extra 中提取: `unique_id` > `trajectory_id` > `skeleton_id`
- 相同 unique_id 的样本追加到同一文件中

### 使用示例

#### 读取单个样本文件

```python
import json

def read_sample_file(file_path):
    """读取样本文件"""
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples

# 读取特定样本
samples = read_sample_file("./data/reward_samples/sample_traj_001.jsonl")
print(f"共 {len(samples)} 条记录")

# 查看第一条记录
first = samples[0]
print(f"时间: {first['timestamp']}")
print(f"分数: {first['output']['total_score']}")
print(f"提示: {first['input']['prompt']}")
```

#### 统计分析

```python
import json
from pathlib import Path

def analyze_samples(data_dir="./data/reward_samples"):
    """分析所有样本数据"""
    total_samples = 0
    total_score = 0.0
    
    for file in Path(data_dir).glob("sample_*.jsonl"):
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line)
                total_samples += 1
                score = sample['output']['total_score']
                if score is not None:
                    total_score += score
    
    avg_score = total_score / total_samples if total_samples > 0 else 0
    print(f"总样本数: {total_samples}")
    print(f"平均分数: {avg_score:.4f}")

analyze_samples()
```

#### 过滤和导出

```python
import json
from pathlib import Path

def export_high_score_samples(data_dir="./data/reward_samples", min_score=0.8, output_file="high_scores.jsonl"):
    """导出高分样本"""
    with open(output_file, "w", encoding="utf-8") as out:
        for file in Path(data_dir).glob("sample_*.jsonl"):
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    sample = json.loads(line)
                    score = sample['output']['total_score']
                    if score is not None and score >= min_score:
                        out.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"高分样本已导出到: {output_file}")

export_high_score_samples()
```

### 配置选项

**启用/禁用持久化**：
```bash
# 启用（默认）
ENABLE_PERSISTENCE=true

# 禁用
ENABLE_PERSISTENCE=false
```

**自定义保存目录**：
```bash
# 相对路径
PERSISTENCE_DIR=./data/reward_samples

# 绝对路径
PERSISTENCE_DIR=/data/production/reward_samples
```

### 性能影响

- **写入方式**: 异步追加写入，不阻塞主计算流程
- **性能开销**: 约 1-5ms/样本（取决于磁盘IO性能）
- **存储空间**: 约 2-10KB/样本（取决于输入数据大小）

### 注意事项

1. **磁盘空间**: 定期清理旧数据避免磁盘满
2. **文件数量**: 大量unique_id会产生大量文件，建议定期归档
3. **并发安全**: 追加写入是原子操作，支持多进程/多线程
4. **错误处理**: 持久化失败不影响计算结果返回，只记录警告日志

---

## 性能优化

### 1. 缓存优化
- **缓存粒度**: 单个group为单位
- **缓存Key**: 输入数据的SHA256哈希
- **过期时间**: 可配置（默认3600秒）
- **淘汰策略**: LRU（内存缓存）

### 2. 异步处理
- **请求处理**: FastAPI原生异步支持
- **任务队列**: asyncio.Queue
- **并发控制**: asyncio.Semaphore
- **后台工作**: asyncio.create_task

### 3. 批量优化
- 相同prompt的requests可以合并处理
- 批量接口减少HTTP往返次数
- 并发执行多个group的计算

### 4. 调优建议

1. **调整并发参数**: 根据机器性能调整 `MAX_CONCURRENT_GROUPS`
2. **启用缓存**: 对于相同输入，缓存可以避免重复计算
3. **使用批量接口**: 多个group使用 `/batch_compute` 比多次调用 `/compute` 更高效
4. **异步任务**: 大批量任务使用 `/compute_async` 避免超时
5. **监控指标**: 定期检查 `/metrics` 了解服务状态

---

## 监控和日志

### 监控指标

**服务级别**：
- 总请求数
- 缓存命中率
- 平均计算时间
- 总样本处理数

**队列级别**：
- 队列大小
- 总任务数
- 各状态任务数量

**缓存级别**：
- 缓存条目数
- 使用率
- 命中/未命中统计

### 日志系统

**日志级别**：
- DEBUG: 详细调试信息
- INFO: 常规操作信息
- WARNING: 警告信息
- ERROR: 错误信息

**日志内容**：
- 请求开始/结束
- 处理时间
- 错误堆栈
- 缓存命中/未命中
- 任务状态变化

---

## 故障排查

### 常见问题

#### Q: 端口被占用怎么办？

修改 `.env` 文件中的 `API_PORT`：
```bash
API_PORT=8001
```

或者终止占用端口的进程：
```bash
# 查看端口占用
lsof -i :8000

# 终止进程
lsof -ti:8000 | xargs kill -9
```
或
```
ss -tulnp | grep ':8000 ' | grep -o 'pid=[0-9]*' | cut -d= -f2 | xargs kill -9
```

#### Q: 没有看到日志输出？

设置日志级别为 DEBUG：
```bash
LOG_LEVEL=DEBUG
```

#### Q: 服务启动很慢？

检查网络连接，确保可以访问 OpenAI API。或者设置更短的超时时间。

#### Q: 如何停止服务？

按 `Ctrl+C` 或者：
```bash
# 查找进程
ps aux | grep "uvicorn"

# 终止进程
kill <PID>
```

### 故障诊断

#### 服务无法启动

1. 检查端口是否被占用: `lsof -i :8000`
2. 检查环境变量是否正确配置
3. 查看日志输出找到具体错误
4. 确认 Python 版本 >= 3.8
5. 确认已安装所有必需依赖

#### 请求超时

1. 增加 `MAX_CONCURRENT_GROUPS` 提升并发能力
2. 检查 `compute_score_async` 内部是否有性能瓶颈
3. 考虑使用异步接口代替同步接口
4. 检查网络连接到 OpenAI API

#### 内存占用过高

1. 减小 `CACHE_MAX_SIZE` 降低内存缓存大小
2. 考虑使用 Redis 缓存代替内存缓存
3. 定期调用 `/cache/clear` 清空缓存

#### 任务堆积

1. 增加 `NUM_TASK_PROCESSORS` 提升处理能力
2. 检查单个任务的处理时间是否过长
3. 考虑提高 `MAX_QUEUE_SIZE` 允许更多任务排队
4. 如果使用Redis缓存，可以增加 `UVICORN_WORKERS` 多进程并发

### 依赖故障

- **OpenAI API不可用**: 任务失败，返回错误
- **Redis不可用**: 降级到内存缓存
- **compute_score_async异常**: 捕获并返回默认值

### 服务重启

1. 内存缓存会丢失（Redis缓存持久）
2. 队列中未完成的任务会丢失
3. 建议：关键任务使用重试机制

---

## 开发指南

### 项目结构

```
deep_research/reward/api/
├── main.py                 # FastAPI应用入口
├── schemas.py              # Pydantic数据模型
├── config.py               # 配置管理
├── middleware.py           # 中间件（日志、监控）
├── core/
│   ├── reward_service.py  # 奖励计算服务
│   ├── task_queue.py      # 任务队列管理
│   └── cache.py           # 缓存后端
├── start.sh               # 启动脚本
├── start_dev.sh           # 开发模式启动脚本
├── example_client.py      # 示例客户端
└── doc/
    └── README.md          # 本文档
```

### 扩展开发

如果需要添加新的奖励维度或修改计算逻辑：

1. 修改 `dr_reward_function_no_remote.py` 中的 `compute_score_async` 函数
2. API会自动使用更新后的计算逻辑
3. 无需修改API服务代码

### 测试

```bash
# 安装测试依赖
pip install pytest pytest-asyncio httpx

# 运行测试
python deep_research/reward/api/test_case.py

# 运行示例客户端
python deep_research/reward/api/example_client.py
```

### 扩展性

#### 水平扩展
虽然当前设计不需要分布式部署，但架构支持：

1. **多实例部署**: 使用负载均衡器
2. **共享缓存**: 切换到Redis缓存
3. **任务队列**: 可以迁移到Redis Queue或RabbitMQ

#### 垂直扩展
单机性能优化：

1. **增加Uvicorn Workers**: `UVICORN_WORKERS`（需使用Redis缓存）
2. **增加Task Processors**: `NUM_TASK_PROCESSORS`
3. **提高并发限制**: `MAX_CONCURRENT_GROUPS`
4. **优化缓存大小**: `CACHE_MAX_SIZE`
5. **调整队列大小**: `MAX_QUEUE_SIZE`

### 安全考虑

当前实现：
- ✅ 输入数据验证（Pydantic）
- ✅ 错误处理和日志
- ✅ 资源限制（队列大小、并发数）
- ⚠️ 无认证授权（根据需求）
- ⚠️ 无加密传输（可添加HTTPS）

如需生产环境部署，建议添加：
- API Key认证
- 请求签名
- HTTPS/TLS
- 速率限制（Rate Limiting）
- IP白名单

---