# OpenAI API 集成使用说明

本项目现在支持两种执行模式：

1. **本地模式 (`local`)**: 使用 Ray + 本地 LLM 服务器
2. **OpenAI API 模式 (`openai_api`)**: 使用 OpenAI API 或兼容的 API 服务

## 使用方法

### 1. 本地模式（默认）

```bash
# 使用本地 Ray + LLM
export EXECUTION_MODE="local"
./run_trajectory_generation.sh
```

### 2. OpenAI API 模式

```bash
# 设置 OpenAI API 配置
export EXECUTION_MODE="openai_api"
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_MODEL_NAME="gpt-4"  # 可选，默认 gpt-4

# 运行生成
./run_trajectory_generation.sh

# 或者使用专用的 OpenAI 脚本
./run_trajectory_generation_openai.sh
```

### 3. 配置文件方式

也可以在配置文件 `config/data_synthesis.yaml` 中设置：

```yaml
# 执行模式配置
execution_mode: "openai_api"  # 或 "local"

# OpenAI API 配置
openai_api:
  api_key: "your-api-key"  # 或者使用环境变量 OPENAI_API_KEY
  base_url: "https://api.openai.com/v1"  # 可选，支持兼容的 API 服务
  organization: null  # 可选
  timeout: 60
  max_retries: 3

# 模型配置
model:
  path: "gpt-4"  # 对于 OpenAI API，这里是模型名称
  name: "gpt-4"
```

## 主要改动

### 1. 新增文件
- `deep_research/generate/core/oai_chat_scheduler.py`: OpenAI API 调度器
- `deep_research/generate/run_trajectory_generation_openai.sh`: OpenAI 模式示例脚本

### 2. 修改文件
- `deep_research/generate/utils/conversation_manager.py`: 添加 `OaiConversationManager`
- `deep_research/generate/core/agent_framework.py`: 支持两种对话管理器
- `deep_research/generate/main_synthesis.py`: 支持两种执行模式
- `deep_research/generate/config/data_synthesis.yaml`: 添加 OpenAI API 配置
- `deep_research/generate/run_trajectory_generation.sh`: 支持两种模式

### 3. 核心特性

#### OaiChatCompletionScheduler
- 继承自 `ChatCompletionScheduler`
- 支持 OpenAI API 的完整鉴权机制
- 自动过滤不支持的参数
- 支持兼容的 API 服务（如通过 `base_url` 配置）

#### 智能参数过滤
- 自动移除 Ray 相关的冗余参数
- 根据执行模式传递相应的配置参数
- 支持环境变量和配置文件两种配置方式

## 环境变量

### OpenAI API 模式支持的环境变量：
- `OPENAI_API_KEY`: API 密钥（必需）
- `OPENAI_MODEL_NAME`: 模型名称（可选，默认 gpt-4）
- `OPENAI_BASE_URL`: API 基础URL（可选，默认 OpenAI 官方）
- `OPENAI_ORG_ID`: 组织ID（可选）

### 执行模式控制：
- `EXECUTION_MODE`: `local` 或 `openai_api`

## 注意事项

1. **API 费用**: 使用 OpenAI API 会产生费用，请注意控制使用量
2. **速率限制**: OpenAI API 有速率限制，可能需要调整并发参数
3. **网络连接**: OpenAI API 模式需要稳定的网络连接
4. **兼容性**: 支持 OpenAI 兼容的 API 服务（如 Azure OpenAI, 自建服务等）

## 错误处理

- 如果 OpenAI API 密钥未设置，会在启动时报错
- 如果模型名称不正确，会在 API 调用时报错
- 网络错误会自动重试（可配置重试次数）

## 示例命令

```bash
# 使用 OpenAI GPT-4 生成 5 个样本
export EXECUTION_MODE="openai_api"
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL_NAME="gpt-4"
./run_trajectory_generation.sh 5

# 使用自定义 API 服务
export EXECUTION_MODE="openai_api"
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://your-api-server.com/v1"
export OPENAI_MODEL_NAME="your-model"
./run_trajectory_generation.sh
```