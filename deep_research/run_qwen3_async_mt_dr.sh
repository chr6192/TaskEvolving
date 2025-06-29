set -x

export VLLM_USE_V1=1

ulimit -n 65535

GPUS=2
ray start --head --node-ip-address 127.0.0.1 --num-gpus $GPUS

TASK_NAME="deep_research"
MODEL_NAME="Qwen3-0.6B"
CURRENT_DATE=$(date +"%Y%m%d")
ROLLOUT_N=2

PATH_TO_PROJ=$(pwd)
PATH_TO_VERL="${PATH_TO_PROJ}/verl"
PATH_TO_DR="${PATH_TO_PROJ}/deep_research"

TRAIN_DATA="${PATH_TO_DR}/data/train_250626.parquet"
EVAL_DATA="${PATH_TO_DR}/data/test_250626.parquet"

CONFIG_PATH="$PATH_TO_PROJ/deep_research/config"
TOOL_CONFIG_PATH="${PATH_TO_DR}/config/tool_config/dr_tool_config.yaml"

# RL_DATASET="${PATH_TO_DR}/dataset/custom_rl_dataset.py"
REWARD_FUN="${PATH_TO_DR}/reward/dr_reward_function.py"

COMPLETION_CALLBACK="deep_research.dr_completion_callback.DrCompletionCallback"
# CHAT_SCHEDULER="${PATH_TO_DR}/dr_chat_scheduler.py"

USER_DIR="$(dirname ${PATH_TO_PROJ})"
PRETRAIN_MODEL="${MODEL_ROOT:-/path/to/models}/${MODEL_NAME}"
CKPT_DIR="${USER_DIR}/${TASK_NAME}/${MODEL_NAME}-${CURRENT_DATE}/ckpt"
ROLLOUT_DATA_DIR="${USER_DIR}/${TASK_NAME}/${MODEL_NAME}-${CURRENT_DATE}/rollout_data"
VALIDATION_DATA_DIR="${USER_DIR}/${TASK_NAME}/${MODEL_NAME}-${CURRENT_DATE}/validation_data"

# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS

# For async rollout mode, dataset should return raw chat.
rollout_mode="async"
rollout_name="vllm" # sglang or vllm
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi


ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env="$CONFIG_PATH/runtime_env.yaml" \
    -- \
    python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='dr_ppo_trainer' \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$EVAL_DATA" \
    data.return_raw_chat=$return_raw_chat \
    data.train_batch_size=16 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    custom_reward_function.path="$REWARD_FUN" \
    actor_rollout_ref.model.path="$PRETRAIN_MODEL" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$rollout_name \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_turns=10 \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.completion_callback="$COMPLETION_CALLBACK" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG_PATH" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="$TASK_NAME" \
    trainer.experiment_name="$MODEL_NAME-$CURRENT_DATE-n$ROLLOUT_N" \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.total_epochs=2 \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.rollout_data_dir="$ROLLOUT_DATA_DIR" \
    trainer.validation_data_dir="$VALIDATION_DATA_DIR" $@
