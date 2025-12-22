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
- [性能问题诊断](#性能问题诊断)
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
NUM_TASK_PROCESSORS=8
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

**开发模式 vs 生产模式对比**：

| 特性 | 开发模式 (start_dev.sh) | 生产模式 (start.sh) |
|------|------------------------|-------------------|
| 热重载 | ✅ `--reload` | ❌ 不启用 |
| Workers | 固定为 1 | 可配置多个 (默认1) |
| 日志级别 | DEBUG | INFO |
| 调试模式 | API_DEBUG=true | API_DEBUG=false |
| 适用场景 | 本地开发、快速迭代 | 生产部署、高并发 |
| 文件监控 | 自动重启 | 需手动重启 |

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
| `API_DEBUG` | `false` | 调试模式 |
| `MAX_CONCURRENT_GROUPS` | `10` | 最大并发group数量 |
| `NUM_TASK_PROCESSORS` | `8` | 进程内异步任务处理器数量 |
| `UVICORN_WORKERS` | `1` | Uvicorn多进程worker数量（仅生产模式） |
| `MAX_QUEUE_SIZE` | `1000` | 最大队列大小 |
| `ENABLE_CACHE` | `true` | 是否启用缓存 |
| `CACHE_TYPE` | `memory` | 缓存类型 (memory/redis) |
| `CACHE_MAX_SIZE` | `10000` | 内存缓存最大条目数 |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis连接URL |
| `ENABLE_PERSISTENCE` | `true` | 是否启用数据持久化 |
| `PERSISTENCE_DIR` | `./data/reward_samples` | 数据持久化目录 |
| `LOG_LEVEL` | `INFO` | 日志级别 (DEBUG/INFO/WARNING/ERROR) |
| `LOG_FILE` | - | 日志文件路径（可选） |
| `OPENAI_API_KEY` | - | OpenAI API Key (必填) |
| `OPENAI_BASE_URL` | `https://dashscope.aliyuncs.com/...` | OpenAI API Base URL |

### 配置默认值对照表

以下是所有配置项的详细默认值，来自 `config.py` 的定义：

#### 服务配置

| 配置项 | 默认值 | 来源 | 说明 |
|--------|--------|------|------|
| `API_HOST` | `0.0.0.0` | config.py | 监听所有网络接口 |
| `API_PORT` | `8000` | config.py | 标准HTTP端口 |
| `API_DEBUG` | `false` | config.py | 生产模式关闭调试 |
| `LOG_LEVEL` | `INFO` | config.py | 标准日志级别 |
| `LOG_FILE` | `None` | config.py | 不写入文件，输出到终端 |

#### 并发控制

| 配置项 | 默认值 | 来源 | 说明 |
|--------|--------|------|------|
| `MAX_CONCURRENT_GROUPS` | `10` | config.py | 单进程同时处理的group数 |
| `NUM_TASK_PROCESSORS` | `8` | config.py | 进程内异步协程数 |
| `MAX_QUEUE_SIZE` | `1000` | config.py | 异步任务队列容量 |
| `UVICORN_WORKERS` | `1` | start.sh | 仅生产模式使用，开发模式固定为1 |

#### 缓存配置

| 配置项 | 默认值 | 来源 | 说明 |
|--------|--------|------|------|
| `ENABLE_CACHE` | `true` | config.py | 启用结果缓存 |
| `CACHE_TYPE` | `memory` | config.py | 内存缓存（单机） |
| `CACHE_MAX_SIZE` | `10000` | config.py | LRU缓存容量 |
| `REDIS_URL` | `redis://localhost:6379/0` | config.py | Redis缓存连接 |

#### 持久化配置

| 配置项 | 默认值 | 来源 | 说明 |
|--------|--------|------|------|
| `ENABLE_PERSISTENCE` | `true` | config.py | 启用样本持久化 |
| `PERSISTENCE_DIR` | `./data/reward_samples` | config.py | 相对项目根目录 |

#### OpenAI配置

| 配置项 | 默认值 | 来源 | 说明 |
|--------|--------|------|------|
| `OPENAI_API_KEY` | `None` | config.py | **必须配置** |
| `OPENAI_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | config.py | 阿里云兼容接口 |

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
NUM_TASK_PROCESSORS=8
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

### 监控指标详解

#### 1. 基础指标

| 指标名称 | 说明 | 用途 |
|---------|------|------|
| `total_requests` | 总请求数（包括成功和失败） | 流量监控 |
| `success_requests` | 成功请求数 | 健康度评估 |
| `failed_requests` | 失败请求数 | 错误率监控 |
| `success_rate` | 成功率 = success_requests / total_requests | 服务质量指标 |
| `total_samples_processed` | 处理的样本总数 | 工作量统计 |
| `avg_samples_per_request` | 平均每请求样本数 | 请求复杂度 |

#### 2. 缓存指标

| 指标名称 | 说明 | 用途 |
|---------|------|------|
| `cache_hits` | 缓存命中次数 | 缓存效果评估 |
| `cache_misses` | 缓存未命中次数 | 实际计算量 |
| `cache_hit_rate` | 缓存命中率 | 缓存优化参考 |

#### 3. 性能指标

| 指标名称 | 说明 | 统计范围 |
|---------|------|---------|
| `total_response_time` | 总响应时间（包含缓存查询） | 所有请求 |
| `avg_response_time` | 平均响应时间 | 所有请求（包括缓存命中） |
| `total_compute_time` | 总计算耗时（仅实际计算） | 仅缓存未命中的请求 |
| `avg_compute_time` | 平均计算耗时 | 仅缓存未命中的请求 |
| `max_compute_time` | 最大计算耗时 | 历史极值 |
| `min_compute_time` | 最小计算耗时 | 历史极值 |

**响应时间 vs 计算时间**：
- **响应时间**：完整的请求处理时间（缓存查询 + 计算 + 持久化），统计所有请求
- **计算时间**：仅实际调用 `compute_score_async` 的时间，仅统计缓存未命中的请求
- **关系**：`avg_response_time < avg_compute_time`（因为缓存命中拉低了平均响应时间）

#### 4. 队列指标

| 指标名称 | 说明 |
|---------|------|
| `queue_size` | 当前等待处理的任务数 |
| `total_tasks` | 内存中的任务总数 |
| `status_counts` | 各状态任务数量（pending/processing/completed/failed） |
| `max_queue_size` | 最大队列容量 |

### 查询监控指标

```bash
# 查询所有指标
curl http://localhost:8000/api/v1/metrics | jq '.metrics.service'

# 查询特定指标
curl -s http://localhost:8000/api/v1/metrics | jq '.metrics.service | {
  total_requests,
  success_rate,
  cache_hit_rate,
  avg_response_time,
  avg_compute_time
}'
```

### 自动日志导出

服务在每次持久化保存后自动导出统计摘要到日志，格式如下：

```
📊 统计摘要 [总计] 请求: 100 (成功: 98, 失败: 2), 样本: 1000, 缓存命中率: 75.0%
📊 统计摘要 [性能] 平均响应: 156.3ms, 平均计算: 2.35s, 最大计算: 5.21s, 最小计算: 0.89s
📊 统计摘要 [本次] 样本: 10, 计算: 2.15s, 响应: 2.18s
```

**日志字段说明**：
- **[总计]**：累计统计，反映服务启动以来的整体情况
- **[性能]**：平均和极值性能指标
- **[本次]**：当前请求的统计，用于对比单次性能与平均性能

### 实际监控场景

#### 场景1：性能监控

```bash
# 持续监控平均响应时间（每30秒）
watch -n 30 'curl -s http://localhost:8000/api/v1/metrics | jq ".metrics.service.avg_response_time * 1000"'
```

#### 场景2：异常检测

```bash
# 检查是否有异常慢的请求
curl -s http://localhost:8000/api/v1/metrics | jq ".metrics.service | {
  max_compute_time,
  avg_compute_time,
  ratio: (.max_compute_time / .avg_compute_time)
}"
```

如果 `ratio > 3`，说明存在异常慢的请求，需要检查：
- LLM API 是否超时
- 是否有特别复杂的样本
- 系统资源是否充足

#### 场景3：健康度监控

```bash
# 监控成功率和失败请求
curl -s http://localhost:8000/api/v1/metrics | jq ".metrics.service | {
  success_rate,
  failed_requests,
  total_requests
}"
```

如果 `success_rate < 0.99`，需要检查日志找出失败原因。

#### 场景4：缓存效果评估

```bash
# 评估缓存效果
curl -s http://localhost:8000/api/v1/metrics | jq ".metrics.service | {
  cache_hit_rate,
  avg_response_time_ms: (.avg_response_time * 1000),
  avg_compute_time_s: .avg_compute_time,
  speedup: (if .avg_response_time > 0 then (.avg_compute_time / .avg_response_time) else 0 end)
}"
```

### 日志系统

**日志级别**：
- DEBUG: 详细调试信息（包括缓存key、锁获取等）
- INFO: 常规操作信息（请求完成、统计摘要等）
- WARNING: 警告信息（缓存失败、持久化失败等）
- ERROR: 错误信息（计算失败、异常堆栈等）

**关键日志示例**：

```
# 缓存命中
INFO: 缓存命中返回: 10 个样本, 响应耗时 2.34ms

# 计算完成
INFO: 计算完成: 10 个样本, 计算耗时 2.15s, 总响应耗时 2.18s

# 统计摘要（自动导出）
INFO: 📊 统计摘要 [总计] 请求: 100 (成功: 100, 失败: 0), 样本: 1000, 缓存命中率: 75.0%
INFO: 📊 统计摘要 [性能] 平均响应: 156.3ms, 平均计算: 2.35s, 最大计算: 5.21s, 最小计算: 0.89s
INFO: 📊 统计摘要 [本次] 样本: 10, 计算: 2.15s, 响应: 2.18s

# 计算失败
ERROR: 计算奖励时发生错误: ConnectionError...
```

### 统计准确性保证

✅ **已解决的统计问题**：

1. **缓存命中时统计遗漏** - 所有请求（包括缓存命中）都会被正确统计
2. **并发竞争导致统计丢失** - 使用 `asyncio.Lock` 保护所有统计更新
3. **失败请求未统计** - 新增 `failed_requests` 指标跟踪失败情况

**验证统计准确性**：

```bash
# 1. 清空缓存和统计
curl -X POST http://localhost:8000/api/v1/cache/clear

# 2. 发送3个相同请求（第2、3次应该缓存命中）
for i in {1..3}; do
  curl -X POST http://localhost:8000/api/v1/rewards/compute \
    -H "Content-Type: application/json" \
    -d '{
      "prompt": "测试",
      "response_list": ["A", "B"],
      "messages_list": [[], []],
      "extra_list": [{}, {}]
    }' > /dev/null 2>&1
done

# 3. 查看统计（应该显示 total_requests=3, cache_hits=2）
curl http://localhost:8000/api/v1/metrics | jq '.metrics.service | {
  total_requests,
  total_samples_processed,
  cache_hits,
  cache_misses
}'

# 预期输出：
# {
#   "total_requests": 3,
#   "total_samples_processed": 6,
#   "cache_hits": 2,
#   "cache_misses": 1
# }
```

---

## 测试和验证

### 自动化测试

#### 基础功能测试

```bash
# 运行完整测试套件
cd deep_research/reward/api
python test_metrics_fix.py
```

测试内容包括：
1. **缓存命中统计** - 验证缓存命中时统计是否准确
2. **并发请求统计** - 验证高并发场景下统计不丢失
3. **混合场景测试** - 验证部分缓存命中 + 并发的情况

#### 新增指标测试

```bash
# 测试新增统计维度
python test_new_metrics.py
```

测试内容包括：
1. **成功请求统计** - 验证 `success_requests` 和 `success_rate`
2. **响应时间统计** - 验证 `avg_response_time` 包含所有请求
3. **计算耗时范围** - 验证 `max_compute_time` 和 `min_compute_time`
4. **日志导出功能** - 验证统计摘要自动导出到日志

### 手动验证

#### 验证1：缓存命中统计准确性

```bash
# 步骤1：清空缓存
curl -X POST http://localhost:8000/api/v1/cache/clear

# 步骤2：发送相同请求2次
curl -X POST http://localhost:8000/api/v1/rewards/compute \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "测试问题",
    "response_list": ["响应1", "响应2"],
    "messages_list": [[], []],
    "extra_list": [{}, {}]
  }' | jq

# 再次发送相同请求
curl -X POST http://localhost:8000/api/v1/rewards/compute \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "测试问题",
    "response_list": ["响应1", "响应2"],
    "messages_list": [[], []],
    "extra_list": [{}, {}]
  }' | jq '.data.meta.cache_hit'

# 步骤3：查看统计
curl http://localhost:8000/api/v1/metrics | jq '.metrics.service | {
  total_requests,
  cache_hits,
  cache_misses,
  cache_hit_rate
}'

# 预期结果：
# {
#   "total_requests": 2,
#   "cache_hits": 1,
#   "cache_misses": 1,
#   "cache_hit_rate": 0.5
# }
```

#### 验证2：并发场景统计准确性

```python
# concurrent_test.py
import asyncio
import httpx

async def send_request(client, i):
    resp = await client.post(
        "http://localhost:8000/api/v1/rewards/compute",
        json={
            "prompt": f"并发测试 {i}",
            "response_list": ["响应1", "响应2"],
            "messages_list": [[], []],
            "extra_list": [{}, {}],
        }
    )
    return resp.status_code

async def main():
    # 先清空统计
    async with httpx.AsyncClient() as client:
        await client.post("http://localhost:8000/api/v1/cache/clear")
    
    # 并发发送10个不同请求
    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = [send_request(client, i) for i in range(10)]
        results = await asyncio.gather(*tasks)
    
    print(f"成功: {sum(1 for r in results if r == 200)}/10")
    
    # 查看统计
    async with httpx.AsyncClient() as client:
        resp = await client.get("http://localhost:8000/api/v1/metrics")
        metrics = resp.json()["metrics"]["service"]
        print(f"total_requests: {metrics['total_requests']}")
        print(f"total_samples_processed: {metrics['total_samples_processed']}")
        assert metrics["total_requests"] == 10, "统计不准确！"
        print("✅ 并发统计验证通过")

asyncio.run(main())
```

运行测试：
```bash
python concurrent_test.py
```

#### 验证3：新增指标功能

```bash
# 发送一些请求后查看新增指标
curl -s http://localhost:8000/api/v1/metrics | jq '.metrics.service | {
  # 成功率指标
  success_requests,
  failed_requests,
  success_rate,
  
  # 响应时间指标（包含所有请求）
  avg_response_time_ms: (.avg_response_time * 1000),
  total_response_time,
  
  # 计算耗时范围（仅缓存未命中）
  avg_compute_time,
  max_compute_time,
  min_compute_time
}'
```

### 性能基准测试

#### 吞吐量测试

```bash
# 使用 Apache Bench 测试
ab -n 100 -c 10 -p payload.json -T application/json \
  http://localhost:8000/api/v1/rewards/compute

# payload.json 内容：
# {
#   "prompt": "性能测试",
#   "response_list": ["响应1", "响应2"],
#   "messages_list": [[], []],
#   "extra_list": [{}, {}]
# }
```

#### 缓存效果测试

```bash
# 1. 清空缓存
curl -X POST http://localhost:8000/api/v1/cache/clear

# 2. 第一次请求（缓存未命中）
time curl -X POST http://localhost:8000/api/v1/rewards/compute \
  -H "Content-Type: application/json" \
  -d '{"prompt":"缓存测试","response_list":["A","B"],"messages_list":[[],[]],"extra_list":[{},{}]}' \
  > /dev/null 2>&1

# 3. 第二次请求（缓存命中）
time curl -X POST http://localhost:8000/api/v1/rewards/compute \
  -H "Content-Type: application/json" \
  -d '{"prompt":"缓存测试","response_list":["A","B"],"messages_list":[[],[]],"extra_list":[{},{}]}' \
  > /dev/null 2>&1

# 对比两次时间差异（缓存命中应该快很多）
```

### 日志验证

查看服务端日志验证统计摘要是否正确导出：

```bash
# 方法1：实时查看日志
tail -f <日志文件路径> | grep "统计摘要"

# 方法2：查看最近的统计日志
grep "📊 统计摘要" <日志文件路径> | tail -20

# 方法3：如果日志输出到终端，直接查看运行服务的终端
```

预期看到如下格式的日志：
```
INFO: 📊 统计摘要 [总计] 请求: 50 (成功: 49, 失败: 1), 样本: 500, 缓存命中率: 60.0%
INFO: 📊 统计摘要 [性能] 平均响应: 235.6ms, 平均计算: 2.15s, 最大计算: 5.32s, 最小计算: 1.02s
INFO: 📊 统计摘要 [本次] 样本: 10, 计算: 2.08s, 响应: 2.11s
```

### 常见测试问题

#### Q: 测试时统计不准确？

**可能原因**：
1. 服务代码未更新 - 检查是否包含 `metrics_lock`
2. 服务未重启 - 重启服务使更新生效
3. 缓存未清空 - 使用 `/api/v1/cache/clear` 清空

**解决方法**：
```bash
# 1. 确认代码已更新
grep "metrics_lock" deep_research/reward/api/core/reward_service.py

# 2. 重启服务
# 停止旧服务（Ctrl+C）
python -m uvicorn deep_research.reward.api.main:app --reload

# 3. 清空缓存和统计
curl -X POST http://localhost:8000/api/v1/cache/clear
```

#### Q: 并发测试失败？

可能是并发数过高，调整测试参数：
```python
# 降低并发数
tasks = [send_request(client, i) for i in range(5)]  # 从10降到5

# 增加超时时间
async with httpx.AsyncClient(timeout=120.0) as client:
```

#### Q: 日志中看不到统计摘要？

检查日志级别：
```bash
# 设置为 INFO 或 DEBUG
LOG_LEVEL=INFO

# 重启服务
```

---

## 性能问题诊断

### 异常长处理时间问题

#### 问题症状

部分样本处理时间出现异常：
- **正常处理时间**: ~48秒
- **异常处理时间**: ~5080秒 (约85分钟)
- **平均处理时间**: ~633秒 (约10.5分钟)

```json
{
  "max_compute_time": 5080.89,
  "min_compute_time": 48.67,
  "avg_compute_time": 633.37,
  "cache_hit_rate": 0.0
}
```

#### 根本原因分析

**主要原因: LLM API调用hang住/限流** 🔴

根据数据分析（min=48s, max=5080s, avg=633s），主要是LLM API问题：

1. **API限流/排队**: 部分请求遇到限流，需要等待很长时间
2. **API响应慢**: 某些LLM调用极慢或hang住
3. **重试机制**: 超时后的重试进一步延长处理时间
4. **并发冲击**: 多个请求同时发起，加剧API端压力

**证据**：
- 处理时间极不均匀（48s vs 5080s），如果是队列拥堵应该相对均匀
- 没有缓存命中，所有请求都需要调用LLM
- 每个group内部并行调用5个reward函数，实际LLM并发数高达50个

#### 已实施的性能改进

##### 1. ✅ 超时保护机制

**问题**: 原代码在调用 `compute_score_async` 时没有超时控制

**解决方案**:
```python
# 使用 asyncio.wait_for 添加超时保护（默认300秒）
timeout_seconds = config.get("timeout", 300)
result = await asyncio.wait_for(
    self.compute_score_async(...),
    timeout=timeout_seconds
)
```

**效果**:
- ✅ 防止LLM API hang住导致无限等待
- ✅ 最多等待5分钟（可配置）
- ✅ 超时后抛出 `TimeoutError` 异常
- ✅ 记录详细的超时日志

**日志示例**:
```
⏰ 计算超时: 样本数=1, 超时阈值=300s, 已耗时=301.45s
📊 性能分析: 等待槽位=0.23s, 实际计算=301.45s, 总耗时=301.68s
```

##### 2. ✅ API限流检测和告警（基于HTTP 429状态码）

**问题**: 无法识别API限流导致的延迟

**解决方案**:
```python
# 通过HTTP 429状态码检测限流
is_rate_limit = False
if hasattr(e, 'status_code') and e.status_code == 429:
    is_rate_limit = True
elif hasattr(e, 'response') and hasattr(e.response, 'status_code') and e.response.status_code == 429:
    is_rate_limit = True
elif type(e).__name__ in ['RateLimitError', 'RateLimitException']:
    is_rate_limit = True

if is_rate_limit:
    logger.warning("⚠️ API限流检测 (HTTP 429)")
```

**效果**:
- ✅ 精确识别HTTP 429响应码
- ✅ 支持常见SDK的限流异常类型
- ✅ 输出专门的warning日志
- ✅ 提供解决建议

**日志示例**:
```
⚠️ API限流检测 (HTTP 429): RateLimitError
  样本数: 1
  已耗时: 125.67s
  建议: 1) 降低并发数 2) 增加重试延迟 3) 使用API配额管理
```

##### 3. ✅ 随机抖动机制

**问题**: 所有并发请求同时发起，产生"惊群效应"

**解决方案**:
```python
# 在获取semaphore前添加50-300ms随机延迟
jitter_delay = random.uniform(0.05, 0.3)
await asyncio.sleep(jitter_delay)
logger.debug(f"随机抖动延迟: {jitter_delay*1000:.1f}ms，减少并发冲击")
```

**效果**:
- ✅ 将并发请求在时间轴上分散
- ✅ 减少同时到达API端的请求数
- ✅ 降低触发限流的概率
- ✅ 避免"惊群效应"

**原理示意**:
```
无抖动:  ||||||||||| (10个请求同时发起)
有抖动:  |  | |   ||  |  | |  (分散在300ms内)
```

##### 4. ✅ 详细性能监控

**新增监控点**:
```python
# 1. 记录队列等待时间
task_wait_time = (datetime.now() - task_created_at).total_seconds()
if task_wait_time > 10.0:
    logger.warning(f"任务在队列中等待了 {task_wait_time:.2f}s")

# 2. 记录并发槽位等待时间
semaphore_wait_time = time.perf_counter() - semaphore_wait_start
if semaphore_wait_time > 1.0:
    logger.warning(f"⏱️ 等待并发槽位耗时: {semaphore_wait_time:.2f}s")

# 3. 检测异常长的处理时间
if processing_time > 100.0:
    logger.warning(f"🐌 任务处理时间异常长! {processing_time:.2f}s")
```

**效果**:
- ✅ 区分"队列等待"和"实际计算"时间
- ✅ 自动检测异常长的处理时间
- ✅ 提供问题诊断建议

#### 性能问题快速诊断

##### 症状1: 日志显示 "⏰ 计算超时"
```
⏰ 计算超时: 样本数=1, 超时阈值=300s, 已耗时=301.45s
```

**原因**: LLM API响应慢或hang住

**解决方案**:
1. 检查LLM API服务状态
2. 降低并发数: `max_concurrent_groups=5`
3. 如果持续超时，考虑增加timeout: `timeout=600`

##### 症状2: 日志显示 "⚠️ API限流检测 (HTTP 429)"
```
⚠️ API限流检测 (HTTP 429): RateLimitError
```

**原因**: 触发了API限流

**解决方案**:
1. **立即**: 降低并发数 `max_concurrent_groups=5`
2. **短期**: 联系API提供商增加配额
3. **长期**: 使用多个API key轮询

##### 症状3: 日志显示 "⏱️ 等待并发槽位耗时"
```
⏱️ 等待并发槽位耗时: 45.23s (可能存在队列拥堵)
```

**原因**: 队列拥堵，并发槽位不足

**解决方案**:
1. 增加并发数: `max_concurrent_groups=15`
2. 增加处理器: `num_task_processors=6`
3. 检查是否有任务hang住占用槽位

##### 症状4: 日志显示 "🐌 任务处理时间异常长"
```
🐌 任务处理时间异常长!
  处理耗时: 5080.89s
  队列等待: 2.34s
```

**原因**: LLM API hang住

**解决方案**:
1. 降低timeout，更快检测失败: `timeout=180`
2. 降低并发，减少API压力: `max_concurrent_groups=5`
3. 检查特定reward函数是否有问题

#### 配置调优建议

##### 场景1: 频繁超时/限流（API压力大）
```python
# config.py 或环境变量
MAX_CONCURRENT_GROUPS=5      # 降低并发
TIMEOUT=180                  # 更快检测失败
NUM_TASK_PROCESSORS=2        # 减少任务数
```

##### 场景2: 队列拥堵（处理能力不足）
```python
MAX_CONCURRENT_GROUPS=15     # 增加并发
TIMEOUT=300                  # 保持默认
NUM_TASK_PROCESSORS=6        # 增加处理器
```

##### 场景3: API稳定，追求高吞吐
```python
MAX_CONCURRENT_GROUPS=20     # 大幅增加并发
TIMEOUT=300                  # 保持默认
NUM_TASK_PROCESSORS=8        # 增加处理器
ENABLE_CACHE=true            # 启用缓存
```

#### 监控指标解读

通过 `/api/v1/metrics` 接口监控关键指标：

| 指标 | 正常范围 | 异常值 | 说明 |
|-----|---------|--------|------|
| `max_compute_time` | < 100s | > 300s | 有任务hang住 |
| `avg_compute_time` | 40-60s | > 100s | 整体性能下降 |
| `failed_requests` | < 5% | > 20% | 大量失败 |
| `cache_hit_rate` | > 30% | < 10% | 缓存未生效 |

**查询命令**:
```bash
curl -s http://localhost:8000/api/v1/metrics | jq '.metrics.service | {
  max_compute_time,
  avg_compute_time,
  failed_requests,
  cache_hit_rate
}'
```

#### 日志关键字速查

| 日志关键字 | 含义 | 严重程度 |
|-----------|------|---------|
| `⏰ 计算超时` | LLM API超时 | 🔴 高 |
| `⚠️ API限流检测 (HTTP 429)` | 触发限流 | 🟡 中 |
| `⏱️ 等待并发槽位` | 队列拥堵 | 🟡 中 |
| `🐌 任务处理时间异常长` | 任务hang住 | 🔴 高 |
| `随机抖动延迟` | 正常运行 | 🟢 低 |
| `✅ 任务完成` | 正常完成 | 🟢 低 |

#### 性能测试脚本

运行性能测试验证改进效果：

```bash
# 确保服务已启动
python deep_research/reward/api/test_performance_improvements.py
```

测试脚本会：
1. ✅ 检查API健康状态
2. ✅ 获取初始指标
3. ✅ 发送20个并发请求
4. ✅ 统计超时、限流、成功率
5. ✅ 对比前后指标变化

**预期输出**:
```
📊 性能测试报告
================================================================================
总请求数: 20
成功: 18
失败: 2
  - 超时: 1
  - 限流: 1
成功率: 90.0%

⏱️ 响应时间:
  平均: 52.34s
  最小: 48.67s
  最大: 301.45s
================================================================================
```

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

#### 统计数据不准确

**问题描述**：`/api/v1/metrics` 返回的统计数据与实际发送的请求数不符。

**常见原因**：

1. **缓存命中导致的统计遗漏**（已修复）
   - 早期版本缓存命中时不统计请求数
   - 解决：更新到最新版本，统计已前置到函数入口

2. **并发竞争导致的统计丢失**（已修复）
   - 高并发下 `+=` 操作不是原子的
   - 解决：使用 `asyncio.Lock` 保护统计更新

3. **服务重启导致统计重置**
   - 统计数据存储在内存中，重启后清零
   - 这是正常行为，可以通过查看日志中的历史统计

**诊断步骤**：

```bash
# 1. 检查代码版本（应该包含修复）
grep "metrics_lock" deep_research/reward/api/core/reward_service.py
# 应该能找到这一行

# 2. 清空缓存并重置统计
curl -X POST http://localhost:8000/api/v1/cache/clear

# 3. 发送3个相同请求
for i in {1..3}; do
  curl -s -X POST http://localhost:8000/api/v1/rewards/compute \
    -H "Content-Type: application/json" \
    -d '{
      "prompt": "测试",
      "response_list": ["A", "B"],
      "messages_list": [[], []],
      "extra_list": [{}, {}]
    }' > /dev/null
done

# 4. 查看统计
curl -s http://localhost:8000/api/v1/metrics | jq '.metrics.service | {
  total_requests,
  cache_hits,
  cache_misses
}'

# 5. 验证结果
# 预期：total_requests=3, cache_hits=2, cache_misses=1
```

**如果统计仍然不准**：

```bash
# 1. 确认服务版本
git log --oneline deep_research/reward/api/core/reward_service.py | head -5

# 2. 完全重启服务
pkill -f "uvicorn.*reward.*api"
sleep 2
python -m uvicorn deep_research.reward.api.main:app --reload

# 3. 运行测试脚本验证
python deep_research/reward/api/test_metrics_fix.py
```

#### 统计指标理解错误

**问题：为什么 avg_compute_time > avg_response_time？**

这是**不可能**的，如果出现说明有bug。正确关系应该是：
```
avg_response_time < avg_compute_time
```

原因：
- `avg_response_time` 统计**所有请求**（包括快速的缓存命中）
- `avg_compute_time` 仅统计**缓存未命中**的请求（实际计算）
- 缓存命中响应时间极短（< 10ms），拉低了平均响应时间

**问题：min_compute_time 为什么是 0？**

两种情况：
1. 服务刚启动，还没有任何计算（初始值 `float('inf')` 转换为 `0.0`）
2. 某次计算确实非常快（如空样本）

验证方法：
```bash
curl -s http://localhost:8000/api/v1/metrics | jq '.metrics.service | {
  total_requests,
  cache_misses,
  min_compute_time
}'

# 如果 cache_misses = 0，说明还没有实际计算过
```

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

## 更新日志

### v1.1.0 (2025-12-18) - 统计系统改进

#### ✅ 修复的问题

1. **缓存命中时统计遗漏** 
   - **问题**：缓存命中的请求不计入 `total_requests` 和 `total_samples_processed`
   - **影响**：高缓存命中率时统计严重偏低（如75%命中率导致统计少75%）
   - **修复**：将统计更新前置到函数入口，所有请求都会被统计

2. **并发竞争导致统计丢失**
   - **问题**：高并发下 `+=` 操作不是原子的，导致统计值丢失
   - **影响**：并发场景下统计不准确，随机丢失更新
   - **修复**：使用 `asyncio.Lock` 保护所有统计更新操作

#### ✨ 新增功能

1. **失败请求统计**
   - 新增 `failed_requests` 指标跟踪计算失败的请求
   - 新增 `success_requests` 和 `success_rate` 评估服务健康度
   
2. **响应时间统计**
   - 新增 `total_response_time` 统计完整响应时间（包含缓存查询）
   - 新增 `avg_response_time` 反映真实用户体验
   - 区分响应时间（所有请求）和计算时间（仅缓存未命中）

3. **计算耗时范围**
   - 新增 `max_compute_time` 记录历史最长计算时间
   - 新增 `min_compute_time` 记录历史最短计算时间
   - 帮助识别性能异常和优化空间

4. **自动日志导出**
   - 在持久化保存后自动导出统计摘要到日志
   - 三段式日志格式：[总计]、[性能]、[本次]
   - 方便实时监控和问题排查

#### 🔧 改进的指标

| 指标 | 改进前 | 改进后 |
|-----|--------|--------|
| `total_requests` | 仅统计缓存未命中 | 统计所有请求 ✓ |
| `total_samples_processed` | 仅统计缓存未命中 | 统计所有样本 ✓ |
| `cache_hit_rate` | 可能不准确 | 准确率 100% ✓ |
| `avg_compute_time` | 含义不清 | 明确为缓存未命中平均 ✓ |
| `avg_response_time` | 不存在 | 新增，反映真实体验 ✨ |
| `success_rate` | 不存在 | 新增，健康度指标 ✨ |

#### 📊 统计对比

**改进前**（缓存命中率75%的场景）：
```json
{
  "total_requests": 25,           // ❌ 实际是100
  "total_samples_processed": 250, // ❌ 实际是1000
  "cache_hits": 75,
  "cache_misses": 25
}
```

**改进后**：
```json
{
  "total_requests": 100,           // ✓ 准确
  "success_requests": 98,          // ✨ 新增
  "failed_requests": 2,            // ✨ 新增
  "success_rate": 0.98,            // ✨ 新增
  "total_samples_processed": 1000, // ✓ 准确
  "cache_hits": 75,
  "cache_misses": 25,
  "cache_hit_rate": 0.75,
  "avg_response_time": 0.156,      // ✨ 新增（秒）
  "avg_compute_time": 2.35,        // ✓ 准确
  "max_compute_time": 5.21,        // ✨ 新增
  "min_compute_time": 0.89         // ✨ 新增
}
```

#### 🔄 迁移指南

**无需任何代码改动**，只需：

1. 重启服务使更新生效：
```bash
# 停止旧服务（Ctrl+C）
python -m uvicorn deep_research.reward.api.main:app --reload
```

2. 清空缓存重置统计（可选）：
```bash
curl -X POST http://localhost:8000/api/v1/cache/clear
```

3. 验证统计准确性：
```bash
python deep_research/reward/api/test_metrics_fix.py
```

**向后兼容**：
- 所有原有指标保持不变
- 仅新增指标，不删除或修改现有字段
- API接口完全兼容

#### 📝 相关文档

- `METRICS_FIX.md` - 问题详细分析和修复方案
- `NEW_METRICS_GUIDE.md` - 新增统计维度说明
- `TESTING_GUIDE.md` - 测试和验证指南
- `test_metrics_fix.py` - 自动化测试脚本
- `test_new_metrics.py` - 新功能测试脚本

---