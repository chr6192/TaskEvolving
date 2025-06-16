set -x
TIMESTAMP=$(date "+%m%dT%H%M")

# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS

data_train_path=/redacted/internal-path
data_test_path=/redacted/internal-path

train_files="['$data_test_path']"
test_files="['$data_test_path']"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env=verl/trainer/runtime_env.yaml \
    -- \
    python3 -m recipe.antfin.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_prior_exp_replay=True \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=128 \
    data.val_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.prompt_key='x2' \
    data.custom_cls.path="recipe/antfin/chat_rl_dataset.py" \
    data.custom_cls.name="ChatRLDataset" \
    reward_model.reward_manager='antfin' \
    reward_model.reward_kwargs.use_group_reward=True \
    reward_model.reward_kwargs.return_details=False \
    reward_model.reward_kwargs.return_statistics=True \
    reward_model.reward_kwargs.comparison_score_threshold=0.99 \
    reward_model.reward_kwargs.use_dynamic_comparison=True \
    reward_model.reward_kwargs.save_path=null \
    custom_reward_function.path='recipe/antfin/antfin_grader_fn.py' \
    custom_reward_function.name='reward_func' \
    actor_rollout_ref.model.path=/redacted/internal-path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_antfin' \
    trainer.experiment_name=${TIMESTAMP} \
    trainer.wandb_host='http://REDACTED_INTERNAL_IP:8080' \
    trainer.wandb_api_key='***REMOVED***' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    trainer.default_local_dir=/redacted/internal-path