import asyncio
import heapq
import importlib
import itertools
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from uuid import uuid4

import aiohttp
import numpy as np
import torch
from cachetools import LRUCache
from omegaconf import DictConfig
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.tools.base_tool import initialize_tools_from_config
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.workers.rollout.chat_scheduler import CompletionCallback

from econml_bfcl.tool.env_handler import EnvHandler


logger = logging.getLogger(__file__)


class BfclCompletionCallback(CompletionCallback):
    def __init__(self, config: DictConfig, scheduler: "ChatCompletionScheduler"):
        super().__init__(config, scheduler)
        
        # 使用基类中已有的tools映射作为_tool_map
        self._tool_map = self.tools
        
        # TODO: add reward manager to calculate reward score once a sample finish

    @property
    def tool_schemas(self):
        """OpenAI JSON tool schemas."""
        return self._tool_schemas
    
    def get_tool_schemas_for_request(self, tools_kwargs=None):
        """获取请求特定的tool schemas.
        
        Args:
            tools_kwargs: Dict containing tool names as keys. If provided, only return schemas for these tools.
                         If None, return all available tool schemas.
        """
        if tools_kwargs is not None:
            # 动态返回指定工具的schemas，类似于sglang_rollout.py中的逻辑
            return [self._tool_map[k].get_openai_tool_schema().model_dump(exclude_unset=True, exclude_none=True) for k in tools_kwargs.keys()]
        else:
            # 返回所有工具的schemas（向后兼容）
            return self._tool_schemas

    async def __call__(self, messages: List[Dict[str, str]], completions: ChatCompletion, info: Dict[str, Any]):
        message = completions.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)
        if "content" not in message:
            message["content"] = ""
        messages.append(message)
        finish_reason = completions.choices[0].finish_reason

        # STEP 0: check if we reach max turns
        if self.max_turns and len(messages) >= self.max_turns:
            print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] Reach max turns, done!\n{message}")
            return

        # # STEP 1: check if the model called tools
        # if finish_reason != "tool_calls":
        #     print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] No tool called, done!\n{message}")
        #     return

        # STEP 2: call tools
        tool_calls = completions.choices[0].message.tool_calls
        print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] Call {len(tool_calls)} tools")

        # tasks = []
        # for tool_call in tool_calls:
        #     assert "original_test_entry" in test_entry, f"original_test_entry is not found for index {test_entry}"
        #     # 从info中获取test_entry，如果没有则使用空字典
        #     test_entry = info.get("original_test_entry", {})
        #     tasks.append(self._call_tool(tool_call, messages, test_entry))
        # tool_responses = await asyncio.gather(*tasks)

        test_entry = info.get("original_test_entry", {})

        env_messages = EnvHandler.interact_from_class(messages, test_entry)
        env_messages = env_messages["messages"]
        
        if any(isinstance(item, Exception) for item in env_messages):
            logger.error(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] Error when calling bfcl env, done!")
            return
        messages.extend(env_messages)

        if env_messages[-1]["role"] == "tool":
            print(f"[id={completions.id},turn={len(messages)}, tool_calls: {tool_calls}, tool_responses: {env_messages}]")
        elif env_messages[-1]["role"] == "user":
            print(f"[id={completions.id},turn={len(messages)}, tool_calls: {tool_calls}, user_responses: {env_messages}]")
        elif env_messages[-1]["role"] == "env":
            print(f"[id={completions.id},turn={len(messages)}, tool_calls: {tool_calls}, env_responses: {env_messages}]")
        else:
            logger.error(f"[Unknown env message, done!][id={completions.id},turn={len(messages)},finish_reason={finish_reason}] Unknown env message:{env_messages}")

        if finish_reason != "tool_calls" and env_messages[-1]["role"] == "env":
            print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] No tool called, and conversation done!\n{message}")
            return

        # STEP 3: resubmit completion request with tool responses
        self.scheduler.submit_chat_completions(messages=messages, request_id=completions.id, info=info)

    # async def _call_tool(self, tool_call, messages: List[Dict[str, str]], test_entry: Dict[str, Any]) -> Dict[str, str]:
    #     """Call tool and return tool response."""
    #     # tool_name = tool_call.function.name
    #     # tool_args = json.loads(tool_call.function.arguments)
    #     # tool = self.tools[tool_name]

    #     # instance_id = await tool.create()
    #     try:
    #         full_messages = messages.copy()

    #         tool_results = await tool.execute(instance_id, full_messages, test_entry)
            
    #         if isinstance(tool_results, list) and len(tool_results) > 0:
    #             tool_response, tool_reward_score, tool_metrics = tool_results[0]
    #         else:
    #             tool_response = "No response from tool"
    #             tool_reward_score = 0.0
    #             tool_metrics = {"error": "empty_response"}
                
    #     except Exception as e:
    #         logger.exception(f"Error when executing tool: {e}")
    #         return e
    #     finally:
    #         await tool.release(instance_id)

    #     return {
    #         "role": "tool",
    #         "content": tool_response,
    #         "tool_call_id": tool_call.id,
    #     }

    def postprocess(self, batch: DataProto, batch_conversations: List[List[Dict[str, str]]], n: int) -> DataProto:
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        # prompts: [prompt] from input dataset
        # 为每个样本动态获取对应的tool_schemas
        prompts = []
        for i, prompt in enumerate(batch.non_tensor_batch["raw_prompt"]):
            if "tools_kwargs" in batch.non_tensor_batch:
                tools_kwargs = batch.non_tensor_batch["tools_kwargs"][i]
                tool_schemas = self.get_tool_schemas_for_request(tools_kwargs)
            else:
                tool_schemas = self.get_tool_schemas_for_request()
            prompts.append(self.tokenizer.apply_chat_template(prompt, tools=tool_schemas, add_generation_prompt=True, tokenize=False))
        
        assert len(batch_conversations) == len(prompts) * n

        # sequences: [prompt + response]
        # 为每个对话动态获取对应的tool_schemas
        sequences = []
        for i, conversation in enumerate(batch_conversations):
            batch_idx = i // n  # 获取对应的batch索引
            if "tools_kwargs" in batch.non_tensor_batch:
                tools_kwargs = batch.non_tensor_batch["tools_kwargs"][batch_idx]
                tool_schemas = self.get_tool_schemas_for_request(tools_kwargs)
            else:
                tool_schemas = self.get_tool_schemas_for_request()
            sequences.append(self.tokenizer.apply_chat_template(conversation, tools=tool_schemas, add_generation_prompt=False, tokenize=False))

        # responses: [response]
        responses = [sequence[len(prompts[i // n]) :] for i, sequence in enumerate(sequences)]

        prompts = self.tokenizer(prompts, return_tensors="pt", padding="longest", padding_side="left")
        responses = self.tokenizer(responses, return_tensors="pt", padding="longest", padding_side="right")
        if n > 1:
            prompts["input_ids"] = prompts["input_ids"].repeat_interleave(n, dim=0)
            prompts["attention_mask"] = prompts["attention_mask"].repeat_interleave(n, dim=0)

        # 获取最大token长度限制
        max_token_len = self.config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu
        
        # 检查并截断超长的sequences - 需要真正截断tensor长度
        truncated_indices = []
        for i in range(prompts["input_ids"].size(0)):
            prompt_len = prompts["attention_mask"][i].sum().item()
            response_len = responses["attention_mask"][i].sum().item()
            total_len = prompt_len + response_len
            
            if total_len > max_token_len:
                # 计算需要截断的长度
                max_response_len = max_token_len - prompt_len
                if max_response_len <= 0:
                    # 如果prompt本身就超长，将response截断为1个token（通常是EOS）
                    max_response_len = 1
                
                truncated_indices.append((i, max_response_len))
                print(f"需要截断样本 {i}: 原始长度 {total_len} -> 截断后长度 {prompt_len + max_response_len}")
        
        # 如果有需要截断的样本，重新构造responses tensor
        if truncated_indices:
            # 计算新的最大response长度
            max_response_len_after_truncation = max_token_len - prompts["input_ids"].size(1)
            if max_response_len_after_truncation <= 0:
                max_response_len_after_truncation = 1
            
            # 如果当前response长度超过截断后的最大长度，需要缩小tensor
            if responses["input_ids"].size(1) > max_response_len_after_truncation:
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                
                # 创建新的截断后的responses tensor
                new_responses_input_ids = responses["input_ids"][:, :max_response_len_after_truncation].clone()
                new_responses_attention_mask = responses["attention_mask"][:, :max_response_len_after_truncation].clone()
                
                # 对需要截断的样本进行处理
                for i, max_resp_len in truncated_indices:
                    if max_resp_len < max_response_len_after_truncation:
                        # 截断并padding
                        new_responses_input_ids[i, max_resp_len:] = pad_token_id
                        new_responses_attention_mask[i, max_resp_len:] = 0
                
                responses["input_ids"] = new_responses_input_ids
                responses["attention_mask"] = new_responses_attention_mask
                
                print(f"截断responses tensor: {responses['input_ids'].size(1)} -> {max_response_len_after_truncation}")
            else:
                # 不需要缩小tensor，只需要修改需要截断的样本
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                for i, max_resp_len in truncated_indices:
                    responses["input_ids"][i, max_resp_len:] = pad_token_id
                    responses["attention_mask"][i, max_resp_len:] = 0

        # response_mask: response mask with tools calling masked out
        response_mask = self._mask_out_tools_calling_tokens(batch.non_tensor_batch["raw_prompt"].repeat(n, axis=0), batch_conversations, responses["input_ids"], responses["attention_mask"])

        input_ids = torch.cat([prompts["input_ids"], responses["input_ids"]], dim=1)
        attention_mask = torch.cat([prompts["attention_mask"], responses["attention_mask"]], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        # 构造与attention_mask长度一致的loss_mask
        # prompt部分设为0，response部分使用response_mask
        loss_mask = torch.zeros_like(attention_mask)
        prompt_length = prompts["input_ids"].size(1)
        loss_mask[:, prompt_length:] = response_mask

        batch = TensorDict(
            {
                "prompts": prompts["input_ids"],  # [bsz, prompt_length]
                "responses": responses["input_ids"],  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "loss_mask": loss_mask,  # [bsz, prompt_length + response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                "position_ids": position_ids,  # [bsz, prompt_length + response_length]
            },
            batch_size=len(input_ids),
        )

        num_turns = np.array([len(conversation) for conversation in batch_conversations], dtype=np.int32)
        messages = np.array(batch_conversations, dtype=object)
        return DataProto(batch=batch, non_tensor_batch={"__num_turns__": num_turns, "messages": messages})

    def _mask_out_tools_calling_tokens(
        self,
        raw_prompts: List[List[Dict[str, str]]],
        batch_conversations: List[List[Dict[str, str]]],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mask out tools calling tokens in the responses.

        Args:
            raw_prompts: [prompt] from input dataset
            batch_conversations: [prompt + response]
            input_ids: responses tokens
            attention_mask: responses attention mask

        Returns:
            mask: (batch_size, response_length)
        """
        batch_size = input_ids.size(0)
        assert len(raw_prompts) == batch_size, f"{len(raw_prompts)} != {batch_size}"
        assert len(batch_conversations) == batch_size, f"{len(batch_conversations)} != {batch_size}"

        # Deduplicate adjacent tool calls, since they're merged into one turn.
        # [user, assistant, tool, tool, assistant] -> [user, assistant, tool, assistant]
        # TODO: it's chat_template specific, find a more generic way to do this.
        def deduplicate_adjacent_tool_calls(roles):
            result = []
            for role, group in itertools.groupby(roles):
                if role == "tool":
                    result.append(role)
                else:
                    result.extend(group)
            return result

        loss_mask = attention_mask.clone()
        for i in range(batch_size):
            responses = batch_conversations[i][len(raw_prompts[i]) :]
            assert len(responses) > 0, f"responses is empty: {responses}"

            roles = deduplicate_adjacent_tool_calls([response["role"] for response in responses])
            # Each turn should be: [BOS]...[EOS]
            eos_indices = input_ids[i].eq(self.tokenizer.eos_token_id).nonzero().squeeze(1)
            
            # 防止因截断导致的索引越界：只处理实际存在的EOS tokens
            max_roles = min(len(roles), len(eos_indices))
            if max_roles < len(roles):
                print(f"警告: 样本 {i} 因截断缺少 {len(roles) - max_roles} 个EOS token，只处理前 {max_roles} 个roles")
            
            for j in range(max_roles):
                if roles[j] == "tool":
                    bos = eos_indices[j - 1] + 1 if j > 0 else 0
                    eos = eos_indices[j]
                    loss_mask[i, bos : eos + 1] = 0

        return loss_mask
