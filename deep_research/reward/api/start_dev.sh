#!/bin/bash
# Reward API 开发模式启动脚本（支持热重载）

# 设置工作目录
cd "$(dirname "$0")/../../.." || exit 1

# 激活虚拟环境（如果存在）
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# 加载环境变量
if [ -f "deep_research/reward/api/.env" ]; then
    echo "加载环境变量..."
    export $(grep -v '^#' deep_research/reward/api/.env | xargs)
fi

# 设置开发模式
export API_DEBUG=true
export LOG_LEVEL=DEBUG

echo "=========================================="
echo "启动 Reward API 服务 (开发模式)"
echo "=========================================="
echo "热重载: 启用"
echo "日志级别: DEBUG"
echo "=========================================="

# 启动服务（开发模式）
uvicorn deep_research.reward.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level debug

