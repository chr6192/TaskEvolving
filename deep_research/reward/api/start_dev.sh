#!/bin/bash
# Reward API 开发模式启动脚本（支持热重载）

# 设置工作目录
cd "$(dirname "$0")/../../.." || exit 1

# 激活虚拟环境（如果存在）
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# 加载环境变量（优先级最高）
if [ -f "deep_research/reward/api/.env" ]; then
    echo "加载环境变量 (.env)..."
    export $(grep -v '^#' deep_research/reward/api/.env | xargs)
else
    echo "警告: .env 文件不存在，使用 config.py 中的默认配置"
fi



# 设置默认值（仅在环境变量未设置时生效，与 config.py 保持一致）
# 设置开发模式（强制覆盖）
export API_DEBUG=true
export LOG_LEVEL=DEBUG
# 服务配置
: ${API_HOST:=0.0.0.0}
: ${API_PORT:=8000}
: ${UVICORN_WORKERS:=1}  # Uvicorn多进程worker数量（生产环境建议设为1，避免内存缓存不共享问题）

# 并发控制（对应 config.py 中的默认值）
: ${MAX_CONCURRENT_GROUPS:=10}  # 最大并发group数量
: ${NUM_TASK_PROCESSORS:=8}  # 进程内异步任务处理器数量
: ${MAX_QUEUE_SIZE:=1000}  # 最大队列大小

# 缓存配置（对应 config.py 中的默认值）
: ${ENABLE_CACHE:=true}
: ${CACHE_TYPE:=memory}  # memory 或 redis
: ${CACHE_MAX_SIZE:=10000}
: ${REDIS_URL:=redis://localhost:6379/0}

# 持久化配置（对应 config.py 中的默认值）
: ${ENABLE_PERSISTENCE:=true}
: ${PERSISTENCE_DIR:=./data/reward_samples}


echo "=========================================="
echo "启动 Reward API 服务 (开发模式)"
echo "=========================================="
echo "服务配置:"
echo "  Host: ${API_HOST}"
echo "  Port: ${API_PORT}"
echo "  Debug: ${API_DEBUG}"
echo "  Log Level: ${LOG_LEVEL}"
echo "  热重载: 启用"
echo ""
echo "并发控制:"
echo "  Uvicorn Workers: 1 (开发模式固定为1，--reload不支持多worker)"
echo "  Max Concurrent Groups: ${MAX_CONCURRENT_GROUPS}"
echo "  Task Processors: ${NUM_TASK_PROCESSORS} (进程内异步处理器)"
echo "  Max Queue Size: ${MAX_QUEUE_SIZE}"
echo ""
echo "缓存配置:"
echo "  Enable Cache: ${ENABLE_CACHE}"
echo "  Cache Type: ${CACHE_TYPE}"
echo "  Cache Max Size: ${CACHE_MAX_SIZE}"
if [ "${CACHE_TYPE}" = "redis" ]; then
    echo "  Redis URL: ${REDIS_URL:-redis://localhost:6379/0}"
fi
echo ""
echo "持久化配置:"
echo "  Enable Persistence: ${ENABLE_PERSISTENCE}"
echo "  Persistence Dir: ${PERSISTENCE_DIR}"
echo "=========================================="
echo "⚠️  开发模式说明："
echo "  - 启用热重载 (--reload)"
echo "  - 单进程运行 (--reload 与 --workers 互斥)"
echo "  - 文件修改后自动重启服务"
echo "=========================================="

# 启动服务（开发模式：启用热重载）
uvicorn deep_research.reward.api.main:app \
    --host ${API_HOST} \
    --port ${API_PORT} \
    --reload \
    --log-level ${LOG_LEVEL,,}

