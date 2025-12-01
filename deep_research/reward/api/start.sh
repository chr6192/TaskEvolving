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

# 加载环境变量
if [ -f "deep_research/reward/api/.env" ]; then
    echo "加载环境变量..."
    export $(grep -v '^#' deep_research/reward/api/.env | xargs)
else
    echo "警告: .env 文件不存在，使用默认配置"
fi

# 设置默认值
: ${API_HOST:=0.0.0.0}
: ${API_PORT:=8000}
: ${LOG_LEVEL:=INFO}
: ${UVICORN_WORKERS:=1}  # Uvicorn多进程worker数量（生产环境建议设为1，避免内存缓存不共享问题）
: ${NUM_TASK_PROCESSORS:=4}  # 进程内异步任务处理器数量

echo "=========================================="
echo "启动 Reward API 服务"
echo "=========================================="
echo "Host: ${API_HOST}"
echo "Port: ${API_PORT}"
echo "Log Level: ${LOG_LEVEL}"
echo "Uvicorn Workers: ${UVICORN_WORKERS} (多进程worker)"
echo "Task Processors: ${NUM_TASK_PROCESSORS} (进程内异步处理器)"
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

