#!/bin/bash
set -x

export VLLM_USE_V1=1
ulimit -n 65535

GPUS=2
ray start --head --node-ip-address 127.0.0.1 --num-gpus $GPUS

TASK_NAME="finance_deep_researh_synthesis"
MODEL_NAME="Qwen3-0.6B"
CURRENT_DATE=$(date +"%Y%m%d-%H%M%S")
ROLLOUT_N=10

PATH_TO_PROJ=$(pwd)
PATH_TO_VERL="${PATH_TO_PROJ}/verl"
PATH_TO_DR="${PATH_TO_PROJ}/deep_research"
PATH_TO_GENERATE="${PATH_TO_DR}/generate"

# Input and Output paths
INPUT_QUERIES="${PATH_TO_DR}/data/synthesis_queries.jsonl"
SKELETON_OUTPUT_DIR="${PATH_TO_DR}/data/skeletons"
TRAJECTORY_OUTPUT_DIR="${PATH_TO_DR}/data/trajectories"

# Config paths
CONFIG_PATH="$PATH_TO_GENERATE/config"
TOOL_CONFIG_PATH="${PATH_TO_DR}/config/tool_config/dr_tool_config.yaml"

# Model and function paths
PRETRAIN_MODEL="${MODEL_ROOT:-/path/to/models}/${MODEL_NAME}"
REWARD_FUN="${PATH_TO_DR}/reward/dr_reward_function2.py"
COMPLETION_CALLBACK="deep_research.dr_completion_callback.DrCompletionCallback"

# Output directory setup
USER_DIR="$(dirname ${PATH_TO_PROJ})"
OUTPUT_BASE_DIR="${USER_DIR}/${TASK_NAME}/${MODEL_NAME}-${CURRENT_DATE}"
SKELETON_DIR="${OUTPUT_BASE_DIR}/skeletons"
TRAJECTORY_DIR="${OUTPUT_BASE_DIR}/trajectories"
LOG_DIR="${OUTPUT_BASE_DIR}/logs"

# Pipeline configuration
STAGE=${1:-"both"}  # "skeleton", "trajectory", or "both"
N_SAMPLES_PER_SKELETON=${2:-10}
PRM_SELECTION_STRATEGY=${3:-"best_of_n"}
PRM_TOP_K=${4:-3}

# For async rollout mode
rollout_mode="async"
rollout_name="vllm"
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

echo "Starting Data Synthesis Pipeline"
echo "Stage: $STAGE"
echo "N samples per skeleton: $N_SAMPLES_PER_SKELETON"
echo "PRM selection strategy: $PRM_SELECTION_STRATEGY"
echo "Output directory: $OUTPUT_BASE_DIR"

# Create output directories
mkdir -p "$SKELETON_DIR"
mkdir -p "$TRAJECTORY_DIR"
mkdir -p "$LOG_DIR"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env="$CONFIG_PATH/runtime_env.yaml" \
    -- \
    python3 -m deep_research.generate.main_synthesis \
    --config-path="$CONFIG_PATH" \
    --config-name='data_synthesis' \
    stage="$STAGE" \
    input.queries_file="$INPUT_QUERIES" \
    output.skeleton_dir="$SKELETON_DIR" \
    output.trajectory_dir="$TRAJECTORY_DIR" \
    output.log_dir="$LOG_DIR" \
    model.path="$PRETRAIN_MODEL" \
    model.name="$MODEL_NAME" \
    generation.n_samples_per_skeleton=$N_SAMPLES_PER_SKELETON \
    generation.rollout_n=$ROLLOUT_N \
    prm.selection_strategy="$PRM_SELECTION_STRATEGY" \
    prm.top_k=$PRM_TOP_K \
    reward_function.path="$REWARD_FUN" \
    completion_callback="$COMPLETION_CALLBACK" \
    tool_config_path="$TOOL_CONFIG_PATH" \
    rollout.name=$rollout_name \
    rollout.mode=$rollout_mode \
    rollout.tensor_model_parallel_size=2 \
    rollout.gpu_memory_utilization=0.6 \
    rollout.multi_turn.enable=True \
    rollout.multi_turn.max_turns=10 \
    rollout.multi_turn.format=hermes \
    rollout.multi_turn.completion_callback="$COMPLETION_CALLBACK" \
    rollout.multi_turn.tool_config_path="$TOOL_CONFIG_PATH" \
    experiment.name="$TASK_NAME-$MODEL_NAME-$CURRENT_DATE" \
    experiment.gpus_per_node=$GPUS \
    $@

echo "Data Synthesis Pipeline completed. Results saved to: $OUTPUT_BASE_DIR" 