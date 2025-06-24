set -x

ulimit -n 65535

ray start --head --node-ip-address 127.0.0.1 --num-gpus 2

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/econml_bfcl/config"
# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS

# For async rollout mode, dataset should return raw chat.
rollout_mode="async"
rollout_name="sglang" # sglang or vllm
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env="$CONFIG_PATH/runtime_env.yaml" \
    -- \
    python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='custom_ppo_trainer' \
    algorithm.adv_estimator=grpo \
    data.train_files="$PROJECT_DIR/econml_bfcl/data/bfcl_multi_turn_base_train_verl.parquet" \
    data.val_files="$PROJECT_DIR/econml_bfcl/data/bfcl_multi_turn_base_test_verl.parquet" \
    data.return_raw_chat=$return_raw_chat \
    data.train_batch_size=16 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.custom_cls.path="$PROJECT_DIR/econml_bfcl/bfcl_rl_dataset.py" \
    data.custom_cls.name="BFCLDataset" \
    custom_reward_function.path="$PROJECT_DIR/econml_bfcl/reward/bfcl_reward_fn.py" \
    actor_rollout_ref.model.path=/redacted/internal-path \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$rollout_name \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.multi_turn.format=qwen \
    actor_rollout_ref.rollout.multi_turn.completion_callback="econml_bfcl.bfcl_completion_callback.CustomCompletionCallback" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/econml_bfcl/config/tool_config/bfcl_tool_config.yaml" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='deep_research' \
    trainer.experiment_name='qwen3_0.6b_multi_turn_n2' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.total_epochs=2 $@
