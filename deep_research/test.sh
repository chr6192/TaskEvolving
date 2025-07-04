TASK_NAME="deep_research"
MODEL_NAME="Qwen__Qwen3-4B"
CURRENT_DATE=$(date +"%Y%m%d-%H%M%S")
ROLLOUT_N=4

PATH_TO_PROJ=$(pwd)
PATH_TO_VERL="${PATH_TO_PROJ}/verl"
PATH_TO_DR="${PATH_TO_PROJ}/deep_research"

TRAIN_DATA="${PATH_TO_DR}/data/train_250702.parquet"
EVAL_DATA="${PATH_TO_DR}/data/test_250702.parquet"

CONFIG_PATH="$PATH_TO_PROJ/deep_research/config"
TOOL_CONFIG_PATH="${PATH_TO_DR}/config/tool_config/dr_tool_config.yaml"

# RL_DATASET="${PATH_TO_DR}/dataset/custom_rl_dataset.py"
REWARD_FUN="${PATH_TO_DR}/reward/dr_reward_function.py"
# REWARD_FUN="${PATH_TO_DR}/reward/mock_reward_function.py"

COMPLETION_CALLBACK="deep_research.dr_completion_callback.DrCompletionCallback"
# CHAT_SCHEDULER="${PATH_TO_DR}/dr_chat_scheduler.py"

USER_DIR="$(dirname ${PATH_TO_PROJ})"
PRETRAIN_MODEL="/mnt4/data/open_source/${MODEL_NAME}"
CKPT_DIR="${USER_DIR}/${TASK_NAME}/${MODEL_NAME}-${CURRENT_DATE}/ckpt"
ROLLOUT_DATA_DIR="${USER_DIR}/${TASK_NAME}/${MODEL_NAME}-${CURRENT_DATE}/rollout_data"
VALIDATION_DATA_DIR="${USER_DIR}/${TASK_NAME}/${MODEL_NAME}-${CURRENT_DATE}/validation_data"

echo $ROLLOUT_DATA_DIR