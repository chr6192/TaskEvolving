#!/bin/bash
# Reward API 启动脚本

# 设置工作目录
cd "$(dirname "$0")/../../.." || exit 1

# 激活虚拟环境（如果存在）
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# 检查必要的依赖
python -c "import fastapi" 2>/dev/null || {
    echo "错误: fastapi 未安装"
    echo "请执行: pip install fastapi uvicorn pydantic"
    exit 1
}

# 加载环境变量（优先级最高）
if [ -f "deep_research/reward/api/.env" ]; then
    echo "加载环境变量 (.env)..."
    export $(grep -v '^#' deep_research/reward/api/.env | xargs)
else
    echo "警告: .env 文件不存在，使用 config.py 中的默认配置"
fi



# 设置默认值（仅在环境变量未设置时生效，与 config.py 保持一致）
: ${API_DEBUG:=false}
: ${LOG_LEVEL:=INFO}

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
echo "启动 Reward API 服务"
echo "=========================================="
echo "服务配置:"
echo "  Host: ${API_HOST}"
echo "  Port: ${API_PORT}"
echo "  Debug: ${API_DEBUG}"
echo "  Log Level: ${LOG_LEVEL}"
echo ""
echo "并发控制:"
echo "  Uvicorn Workers: ${UVICORN_WORKERS} (多进程worker)"
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
echo "⚠️  注意："
echo "  - Uvicorn Workers > 1 时，需要使用Redis缓存"
echo "  - 使用内存缓存时，建议 Uvicorn Workers = 1"
echo "=========================================="

# 启动服务
uvicorn deep_research.reward.api.main:app \
    --host ${API_HOST} \
    --port ${API_PORT} \
    --log-level ${LOG_LEVEL,,} \
    --workers ${UVICORN_WORKERS}

