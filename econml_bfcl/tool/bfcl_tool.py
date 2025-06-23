import logging
import os
from typing import Any, Optional, Tuple
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema
from econml_bfcl.tool.env_handler import EnvHandler


class BfclTool(BaseTool):
    """A tool for interacting with BFCL environment using EnvHandler.

    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool via EnvHandler.interact.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        Initialize BfclTool with configuration and tool schema.
        
        Args:
            config: Tool configuration containing test_entry and other parameters
            tool_schema: OpenAI function tool schema defining the tool interface
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}
            
        # Initialize shared EnvHandler for all instances
        self._env_handler = EnvHandler()
        
        # # Store test_entry from config for environment interaction
        # self._test_entry = config.get("test_entry", {})
        # if not self._test_entry:
        #     raise ValueError("No test_entry provided in config. This may cause issues during execution.")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    # from typing import List, Dict, Any
    # async def execute(self, instance_id: str, messages: List[Dict[str, Any]], test_entry: Dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
    #     """Execute the tool by interacting with EnvHandler.
    #
    #     Args:
    #         instance_id: The instance id of the tool
    #         messages: The messages passed from the function call
    #         test_entry: The test entry passed from the function call
    #     """
    #     assert test_entry is not None, "test_entry is required"
    #     env_messages = self._env_handler.interact(messages, test_entry)
    #     env_messages = env_messages["messages"]
    #     # 提取最后连续的tool消息内容
    #     tool_contents = []
    #     try:
    #         for msg in reversed(env_messages):
    #             if msg.get("role") == "tool":
    #                 tool_contents.insert(0, msg.get("content", ""))
    #             else:
    #                 break
    #     except Exception as e:
    #         print(f"Error in execute: {str(e)}\nenv_messages: {env_messages}")
    #         raise e
    #
    #     metrics = {
    #         "status": "success",
    #         "message_count": len(env_messages)
    #     }
    #
    #     return [(tool_content, 0.0, metrics) for tool_content in tool_contents]
        
    # async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
    #     """Execute the tool by interacting with EnvHandler.
        
    #     Args:
    #         instance_id: The instance id of the tool
    #         parameters: The parameters passed from the function call
    #         **kwargs: Additional execution arguments
            
    #     Returns:
    #         tool_response: The response string from the environment
    #         tool_reward_score: The step reward score (0.0 for now)
    #         tool_metrics: The metrics dictionary
    #     """
    #     if instance_id not in self._instance_dict:
    #         raise ValueError(f"Instance {instance_id} not found. Please call create() first.")
            
    #     instance_state = self._instance_dict[instance_id]
        
    #     try:
    #         # Get current messages from instance state
    #         messages = instance_state["messages"].copy()
            
    #         # Create assistant message with tool call
    #         import json
    #         try:
    #             # Ensure arguments are properly JSON serialized
    #             if isinstance(parameters, dict):
    #                 arguments_str = json.dumps(parameters)
    #             elif isinstance(parameters, str):
    #                 # Try to parse as JSON to validate, then use as-is
    #                 try:
    #                     json.loads(parameters)
    #                     arguments_str = parameters
    #                 except json.JSONDecodeError:
    #                     # If not valid JSON, wrap as a simple string parameter
    #                     arguments_str = json.dumps({"value": parameters})
    #             else:
    #                 arguments_str = json.dumps({"value": str(parameters)})
    #         except Exception:
    #             arguments_str = json.dumps({"value": str(parameters)})
                
    #         tool_call_message = {
    #             "role": "assistant",
    #             "content": "",
    #             "tool_calls": [{
    #                 "id": f"call_{instance_id}",
    #                 "type": "function",
    #                 "function": {
    #                     "name": self.name,
    #                     "arguments": arguments_str
    #                 }
    #             }]
    #         }
            
    #         # Add the tool call message to conversation
    #         messages.append(tool_call_message)
            
    #         # Get test_entry for environment interaction
    #         test_entry = instance_state.get("test_entry", self._test_entry)
    #         if not test_entry:
    #             return "Error: No test_entry available for environment interaction", -1.0, {"error": "no_test_entry"}
            
    #         # Call EnvHandler.interact
    #         env_response = self._env_handler.interact(
    #             messages=messages,
    #             test_entry=test_entry
    #         )
            
    #         # Process environment response
    #         if "messages" in env_response:
    #             env_messages = env_response["messages"]
                
    #             # Update instance conversation state
    #             instance_state["messages"] = messages + env_messages
                
    #             # Extract response content for return
    #             if env_messages:
    #                 last_message = env_messages[-1]
    #                 if last_message.get("role") == "tool":
    #                     response_content = last_message.get("content", "")
    #                 elif last_message.get("role") == "user":
    #                     response_content = last_message.get("content", "")
    #                 elif last_message.get("role") == "env":
    #                     response_content = last_message.get("content", "")
    #                 else:
    #                     response_content = str(env_messages)
    #             else:
    #                 response_content = "No response from environment"
                    
    #             # Update metrics
    #             metrics = {
    #                 "status": "success",
    #                 "response_type": env_messages[-1].get("role", "unknown") if env_messages else "empty",
    #                 "message_count": len(env_messages)
    #             }
                
    #             return response_content, 0.0, metrics
                
    #         else:
    #             # Handle case where env_response doesn't contain messages
    #             error_msg = f"Invalid environment response format: {env_response}"
    #             logger.warning(error_msg)
    #             return error_msg, -0.1, {"error": "invalid_response_format"}
                
    #     except Exception as e:
    #         error_msg = f"Error during tool execution: {str(e)}"
    #         logger.error(error_msg)
    #         return error_msg, -0.1, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate the reward for the tool instance.
        
        Args:
            instance_id: The instance id of the tool
            **kwargs: Additional arguments for reward calculation
            
        Returns:
            The reward score (0.0 by default, can be extended)
        """
        if instance_id not in self._instance_dict:
            return 0.0
            
        # Basic reward calculation - can be extended based on conversation success
        instance_state = self._instance_dict[instance_id]
        return instance_state.get("reward", 0.0)

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance.
        
        Args:
            instance_id: The instance id of the tool
            **kwargs: Additional arguments for cleanup
        """
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
