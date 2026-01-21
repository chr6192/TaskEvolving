"""
自定义 ChatCompletionScheduler，支持禁用 reasoning 模式

通过配置参数 enable_thinking 控制是否启用模型的 reasoning/thinking 模式
"""
import asyncio
import time
from typing import Any, Dict

from omegaconf import DictConfig
from verl.protocol import DataProto
from verl.workers.rollout.chat_scheduler import ChatCompletionScheduler


class DrChatSchedulerNoThinking(ChatCompletionScheduler):
    """
    支持禁用 reasoning 模式的 ChatCompletionScheduler
    
    通过配置 actor_rollout_ref.rollout.enable_thinking=False 来禁用 reasoning 模式
    默认行为：不传递 enable_thinking 参数（保持原有行为）
    """
    
    def __init__(self, config: DictConfig, server_addresses=None, max_cache_size: int = 10000):
        super().__init__(config, server_addresses, max_cache_size)
        
        # 从配置中读取 enable_thinking 参数（如果存在）
        self.enable_thinking = getattr(self.config, 'enable_thinking', None)
        
        if self.enable_thinking is not None:
            print(f"[DrChatSchedulerNoThinking] enable_thinking 设置为: {self.enable_thinking}")
        else:
            print("[DrChatSchedulerNoThinking] enable_thinking 未设置，将不传递该参数")
    
    async def generate_sequences(self, batch: DataProto) -> DataProto:
        t_start = time.time()
        
        # 基础采样参数
        kwargs = dict(
            model=self.model_name,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        
        # 添加 enable_thinking 参数（如果配置中指定了）
        if self.enable_thinking is not None:
            kwargs['enable_thinking'] = self.enable_thinking

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            kwargs["top_p"] = self.config.val_kwargs.top_p
            kwargs["temperature"] = self.config.val_kwargs.temperature

        print(f"[DrChatSchedulerNoThinking] generate_sequences sampling params: {kwargs}")

        # NOTE: For multi-turn rollout, repeat raw_prompt n times and process each prompt independently,
        # validation dataset has already been repeated in `PPOTrainer._validate`.
        n = 1 if batch.meta_info.get("validate", False) else self.config.n
        tasks, batch_conversations = [], [None] * len(batch) * n
        for batch_index, conversation in enumerate(batch.non_tensor_batch["raw_prompt"].repeat(n, axis=0)):
            # raw_prompt: [{"role": "user", "content": ""}, ["role": "assistant", "content"], ...]
            batch_conversations[batch_index] = conversation.tolist()

            tasks.append(
                asyncio.create_task(
                    self._submit_chat_completions_semaphore(
                        messages=batch_conversations[batch_index],
                        request_id=None,
                        sampling_params=kwargs,
                    )
                )
            )

        await asyncio.gather(*tasks)
        output_batch = self.completion_callback.postprocess(batch, batch_conversations, n=n)
        output_batch.meta_info["timing"] = {"generate_sequences": time.time() - t_start}
        print("[DrChatSchedulerNoThinking] generate_sequences done")
        return output_batch
