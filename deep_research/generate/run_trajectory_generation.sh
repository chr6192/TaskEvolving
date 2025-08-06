#!/bin/bash
set -x

# Configuration
TASK_NAME="finance_trajectory_generation"
EXECUTION_MODE=${EXECUTION_MODE:-"openai_api"}  # "local" or "openai_api"
CURRENT_DATE=$(date +"%Y%m%d-%H%M%S")

MODEL_NAME="Qwen2.5-1.5B-Instruct"

# OpenAI API Configuration (只在openai_api模式下使用)
OPENAI_MODEL_NAME="qwen3-235b-a22b"  # 可以修改为其他OpenAI模型，如 "gpt-3.5-turbo", "gpt-4-turbo" 等
OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 兼容OpenAI API的服务地址

# 根据执行模式进行不同的初始化
if [ "$EXECUTION_MODE" = "local" ]; then
    GPUS=2
    echo "使用本地Ray+LLM模式"
    export VLLM_USE_V1=1
    ulimit -n 65535
    ray start --head --node-ip-address 127.0.0.1 --num-gpus $GPUS
elif [ "$EXECUTION_MODE" = "openai_api" ]; then
    echo "使用OpenAI API模式"
    # 检查环境变量
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "错误: 使用OpenAI API模式时需要设置环境变量 OPENAI_API_KEY"
        exit 1
    fi
    # 使用脚本中配置的OpenAI模型名称
    MODEL_NAME="$OPENAI_MODEL_NAME"
    echo "使用OpenAI模型: $MODEL_NAME"
    echo "使用OpenAI API地址: $OPENAI_BASE_URL"
else
    echo "错误: 不支持的执行模式: $EXECUTION_MODE"
    echo "支持的模式: local, openai_api"
    exit 1
fi

PATH_TO_PROJ=$(pwd)
PATH_TO_DR="${PATH_TO_PROJ}/deep_research"
PATH_TO_GENERATE="${PATH_TO_DR}/generate"

# Input and Output paths - 使用现有的skeleton数据作为输入
INPUT_QUERIES="${PATH_TO_DR}/data/4k_w_skeleton/test_250731.parquet"

# Config paths
CONFIG_PATH="$PATH_TO_GENERATE/config"
TOOL_CONFIG_PATH="${PATH_TO_DR}/config/tool_config/dr_tool_config.yaml"

# Model and function paths
PRETRAIN_MODEL="${MODEL_NAME}"
COMPLETION_CALLBACK="deep_research.generate.dr_synthesis_completion_callback.DrSynthesisCompletionCallback"

# Output directory setup
USER_DIR="$(dirname ${PATH_TO_PROJ})" 
OUTPUT_BASE_DIR="${USER_DIR}/${TASK_NAME}/${MODEL_NAME}-${CURRENT_DATE}"
TRAJECTORY_DIR="${OUTPUT_BASE_DIR}/trajectories"
LOG_DIR="${OUTPUT_BASE_DIR}/logs"

# Pipeline configuration - 只运行trajectory阶段
STAGE="trajectory"
N_SAMPLES_PER_SKELETON=${1:-1}  # 每个skeleton生成多少个trajectory

echo "Starting Trajectory Generation Pipeline"
echo "Stage: $STAGE (trajectory only)"
echo "Input file: $INPUT_QUERIES"
echo "N samples per skeleton: $N_SAMPLES_PER_SKELETON"
echo "Output directory: $OUTPUT_BASE_DIR"

# Create output directories
mkdir -p "$TRAJECTORY_DIR"
mkdir -p "$LOG_DIR"

# 检查输入文件是否存在
if [ ! -f "$INPUT_QUERIES" ]; then
    echo "错误: 输入文件不存在: $INPUT_QUERIES"
    echo "请确保skeleton数据文件存在"
    exit 1
fi

# 构建基础参数
BASE_ARGS=(
    --config-path="$CONFIG_PATH"
    --config-name='data_synthesis'
    stage="$STAGE"
    input.queries_file="$INPUT_QUERIES"
    output.trajectory_dir="$TRAJECTORY_DIR"
    output.log_dir="$LOG_DIR"
    model.path="$PRETRAIN_MODEL"
    model.name="$MODEL_NAME"
    generation.n_samples_per_skeleton=$N_SAMPLES_PER_SKELETON
    completion_callback="$COMPLETION_CALLBACK"
    tool_config_path="$TOOL_CONFIG_PATH"
    execution_mode="$EXECUTION_MODE"
    experiment.name="$TASK_NAME-$MODEL_NAME-$CURRENT_DATE"
    prm.enabled=false
    orm.enabled=false
)

# 根据执行模式添加特定参数
if [ "$EXECUTION_MODE" = "local" ]; then
    # 本地Ray+LLM模式的参数
    LOCAL_ARGS=(
        rollout.tensor_model_parallel_size=1
        rollout.gpu_memory_utilization=0.6
        rollout.multi_turn.completion_callback="$COMPLETION_CALLBACK"
        rollout.multi_turn.tool_config_path="$TOOL_CONFIG_PATH"
        experiment.gpus_per_node=$GPUS
        trainer.n_gpus_per_node=$GPUS
        trainer.nnodes=1
        actor_rollout_ref.model.path="$PRETRAIN_MODEL"
        actor_rollout_ref.model.name="$MODEL_NAME"
        actor_rollout_ref.rollout.tensor_model_parallel_size=1
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6
        actor_rollout_ref.rollout.multi_turn.completion_callback="$COMPLETION_CALLBACK"
        actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG_PATH"
    )
    
    # 使用ray job submit运行
    ray job submit --address="http://127.0.0.1:8265" \
        --runtime-env="$CONFIG_PATH/runtime_env.yaml" \
        -- \
        python3 -m deep_research.generate.main_synthesis \
        "${BASE_ARGS[@]}" \
        "${LOCAL_ARGS[@]}" \
        $@
        
elif [ "$EXECUTION_MODE" = "openai_api" ]; then
    # OpenAI API模式的参数
    OPENAI_ARGS=(
        rollout.multi_turn.completion_callback="$COMPLETION_CALLBACK"
        rollout.multi_turn.tool_config_path="$TOOL_CONFIG_PATH"
    )
    
    # 添加OpenAI API配置参数
    if [ -n "$OPENAI_API_KEY" ]; then
        OPENAI_ARGS+=(openai_api.api_key="$OPENAI_API_KEY")
    fi
    # 使用脚本中配置的base_url
    OPENAI_ARGS+=(openai_api.base_url="$OPENAI_BASE_URL")
    
    # 直接运行Python脚本（不需要Ray）
    python3 -m deep_research.generate.main_synthesis \
        "${BASE_ARGS[@]}" \
        "${OPENAI_ARGS[@]}" \
        $@
fi

echo "Trajectory Generation Pipeline completed. Results saved to: $OUTPUT_BASE_DIR"
echo "Generated trajectories can be found in: $TRAJECTORY_DIR"
echo "Logs can be found in: $LOG_DIR"