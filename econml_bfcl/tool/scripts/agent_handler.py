import json
import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from bfcl.model_handler.env_handler import EnvHandler


class AgentHandler:
    """
    Agent that follows OpenAI interface and interacts with env_handler for multi-turn conversations
    """
    
    def __init__(
        self, 
        model_name: str = "qwen-max-2025-01-25",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0
    ):
        """
        Initialize OpenAI Agent
        
        Args:
            model_name: Name of OpenAI model
            api_key: OpenAI API key, if None will get from environment variables
            base_url: Base API URL, if None will use default
            temperature: Generation temperature
        """
        self.model_name = model_name
        self.temperature = temperature
        
        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        elif os.getenv("OPENAI_API_KEY"):
            client_kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError("OPENAI_API_KEY environment variable or api_key parameter is required")
            
        if base_url:
            client_kwargs["base_url"] = base_url
            
        self.client = OpenAI(**client_kwargs)
        
        self.env_handler = EnvHandler()
    
    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call OpenAI Chat Completions API
        
        Args:
            messages: List of conversation messages
            tools: List of available tools
            **kwargs: Additional parameters
            
        Returns:
            OpenAI API response
        """
        request_params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            **kwargs
        }
        
        if tools:
            request_params["tools"] = tools
            
        response = self.client.chat.completions.create(**request_params)
        return response
    
    def run_conversation(
        self, 
        test_entry: Dict[str, Any],
        max_turns: int = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run a complete multi-turn conversation
        
        Args:
            test_entry: Test entry containing messages, tools and original_test_entry info
            max_turns: Maximum number of turns, if None uses length of questions in original_test_entry
            verbose: Whether to print detailed information
            
        Returns:
            Conversation results including message history and statistics
        """
        conversation_history = []
        
        initial_messages = test_entry.get("messages", [])
        initial_tools = test_entry.get("tools", [])
        
        if verbose:
            print(f"Starting conversation - Test ID: {test_entry.get('id', 'unknown')}")
            print(f"Initial messages: {initial_messages}")
            print(f"Available tools: {len(initial_tools)}")
            print("-" * 50)
        
        conversation_history.extend(initial_messages)
        
        turn_count = 0
        total_input_tokens = 0
        total_output_tokens = 0
        max_turns = len(test_entry["original_test_entry"].get("question", []))
        
        while turn_count < max_turns:
            
            if verbose:
                print(f"\n=== Turn {turn_count} ===")
            
            if 1:
                response = self.chat_completion(
                    messages=conversation_history,
                    tools=initial_tools if initial_tools else None
                )
                
                if hasattr(response, 'usage') and response.usage:
                    total_input_tokens += response.usage.prompt_tokens
                    total_output_tokens += response.usage.completion_tokens
                
                assistant_message = response.choices[0].message
                print(f"Assistant response: {assistant_message}")
                if verbose:
                    print(f"Assistant response: {assistant_message.content}")
                    if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                        print(f"Tool calls: {len(assistant_message.tool_calls)}")
                        for i, tool_call in enumerate(assistant_message.tool_calls):
                            print(f"  {i+1}. {tool_call.function.name}({tool_call.function.arguments})")
                
                assistant_msg_dict = {
                    "role": "assistant",
                    "content": assistant_message.content
                }
                
                if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                    assistant_msg_dict["tool_calls"] = [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        }
                        for tool_call in assistant_message.tool_calls
                    ]
                
                conversation_history.append(assistant_msg_dict)
                
                env_response = self.env_handler.interact(
                    messages=conversation_history,
                    test_entry=test_entry["original_test_entry"]
                )
                
                if verbose:
                    print(f"Env response: {env_response}")
                
                if "messages" in env_response:
                    env_messages = env_response["messages"]

                    if env_messages[-1].get("role") == "env":
                        if env_messages[-1].get("content") == "[CONVERSATION_COMPLETED]":
                            if verbose:
                                print("Conversation completed!")
                            break
                        
                        if "[ERROR]" in str(env_messages[-1].get("content", "")):
                            if verbose:
                                print(f"Error occurred: {env_messages[-1].get('content')}")
                            break

                    elif env_messages[-1].get("role") == "user":
                        turn_count += 1
                    
                    conversation_history.extend(env_messages)
                    
                    if "tools" in env_response:
                        if initial_tools != env_response["tools"]:
                            initial_tools = env_response["tools"]
                            if verbose:
                                print(f"Tools list updated, new tool count: {len(initial_tools)}")
                        else:
                            if verbose:
                                print(f"Tools list unchanged")
                
                else:
                    if verbose:
                        print("Invalid environment response format, ending conversation")
                    break
        
        result = {
            "test_id": test_entry.get("id", "unknown"),
            "messages": conversation_history,
            "turn_count": turn_count,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "completed": turn_count < max_turns,
            "original_test_entry": test_entry["original_test_entry"]
        }
        
        if verbose:
            print(f"\nConversation statistics:")
            print(f"  Total turns: {turn_count}")
            print(f"  Input tokens: {total_input_tokens}")
            print(f"  Output tokens: {total_output_tokens}")
            print(f"  Completed: {result['completed']}")
        
        return result


def load_test_case_from_jsonl(file_path: str, test_id: str) -> Optional[Dict[str, Any]]:
    """
    Load test case with specified ID from JSONL file
    
    Args:
        file_path: Path to JSONL file
        test_id: Test case ID
        
    Returns:
        Test case data if found, None otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if data.get("id") == test_id:
                        return data
    except Exception as e:
        print(f"Error loading test case: {e}")
    
    return None