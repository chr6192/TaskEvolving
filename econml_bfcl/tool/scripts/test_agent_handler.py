import json
import os
from agent_handler import AgentHandler, load_test_case_from_jsonl


def create_mock_test_case():
    """
    创建一个简化的测试用例，用于测试基本功能
    """
    return {
        "id": "test_simple_case",
        "messages": [
            {
                "role": "user", 
                "content": "I heard that the technology sector is booming these days. Get me a list of potential tech stocks I could invest in."
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_available_stocks",
                    "description": "Get a list of stock symbols in the given sector.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sector": {
                                "type": "string",
                                "description": "The sector to retrieve stocks from (e.g., 'Technology')."
                            }
                        },
                        "required": ["sector"]
                    }
                }
            }
        ],
        "function": [
            {
                "name": "get_available_stocks",
                "description": "Get a list of stock symbols in the given sector.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "sector": {
                            "type": "string",
                            "description": "The sector to retrieve stocks from (e.g., 'Technology')."
                        }
                    },
                    "required": ["sector"]
                },
                "response": {
                    "type": "dict",
                    "properties": {
                        "stock_list": {
                            "type": "array",
                            "description": "List of stock symbols in the specified sector.",
                            "items": {"type": "string"}
                        }
                    }
                }
            }
        ],
        "question": [
            [{"role": "user", "content": "I heard that the technology sector is booming these days. Get me a list of potential tech stocks I could invest in."}],
            [{"role": "user", "content": "From the tech stocks available, let's dive into one that catches my eye. Grab the latest market data for this stock 'MSFT'. I'm thinking about acquiring it."}]
        ],
        "involved_classes": ["TradingBot"],
        "initial_config": {},
        "holdout_function": {},
        "original_test_entry": {
            "id": "test_simple_case",
            "question": [
                [{"role": "user", "content": "I heard that the technology sector is booming these days. Get me a list of potential tech stocks I could invest in."}],
                [{"role": "user", "content": "From the tech stocks available, let's dive into one that catches my eye. Grab the latest market data for this stock 'MSFT'. I'm thinking about acquiring it."}]
            ],
            "function": [
                {
                    "name": "get_available_stocks",
                    "description": "Get a list of stock symbols in the given sector.",
                    "parameters": {
                        "type": "dict",
                        "properties": {
                            "sector": {
                                "type": "string",
                                "description": "The sector to retrieve stocks from (e.g., 'Technology')."
                            }
                        },
                        "required": ["sector"]
                    },
                    "response": {
                        "type": "dict",
                        "properties": {
                            "stock_list": {
                                "type": "array",
                                "description": "List of stock symbols in the specified sector.",
                                "items": {"type": "string"}
                            }
                        }
                    }
                }
            ],
            "involved_classes": ["TradingBot"],
            "missed_function": {},
            "path": ["TradingBot.get_available_stocks"]
        }
    }

def test_with_real_case(test_file_path, test_id):
    """
    使用真实的测试用例进行测试
    """
    print("=== 使用真实测试用例 ===")    
    
    # 从JSONL文件加载测试用例
    test_case = load_test_case_from_jsonl(test_file_path, test_id)
    
    if not test_case:
        print(f"未找到测试用例: {test_id}")
        return
    
    print(f"成功加载测试用例: {test_id}")
    print(f"初始用户消息: {test_case['messages'][0]['content']}")
    print(f"可用工具数量: {len(test_case['tools'])}")
    
    # 检查是否有OpenAI API密钥
    if not os.getenv("OPENAI_API_KEY"):
        print("警告: 未设置OPENAI_API_KEY环境变量，将跳过OpenAI API调用测试")
        return
    
    # 创建agent
    agent = AgentHandler(
        model_name="qwen-max-2025-01-25",
        temperature=0.1,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    # 运行对话
    result = agent.run_conversation(
        test_entry=test_case,
        max_turns=None,
        verbose=True
    )
    
    # 保存结果
    output_file = f"conversation_result_{test_id}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n对话结果已保存到: {output_file}")

def test_env_handler_only():
    """
    仅测试env_handler的功能，不调用OpenAI API
    """
    print("=== 仅测试env_handler功能 ===")
    
    from bfcl.model_handler.env_handler import EnvHandler
    
    # 创建模拟测试用例
    test_case = create_mock_test_case()
    
    # 初始化环境处理器
    env_handler = EnvHandler()
    
    # 测试空消息列表
    print("\n1. 测试空消息列表:")
    response = env_handler.interact([], test_case)
    print(f"响应: {response}")
    
    # 测试带有assistant消息但无工具调用
    print("\n2. 测试assistant消息但无工具调用:")
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    response = env_handler.interact(messages, test_case)
    print(f"响应: {response}")
    
    # 测试带有工具调用的assistant消息
    print("\n3. 测试带有工具调用的assistant消息:")
    messages = [
        {"role": "user", "content": "Get me tech stocks"},
        {
            "role": "assistant",
            "content": "I'll get the tech stocks for you.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_available_stocks",
                        "arguments": '{"sector": "Technology"}'
                    }
                }
            ]
        }
    ]
    response = env_handler.interact(messages, test_case)
    print(f"响应: {response}")


def main():
    """
    主函数，运行所有测试
    """
    print("OpenAI Agent 与 env_handler 交互测试")
    print("=" * 50)
    
    # 首先测试env_handler的基本功能
    test_env_handler_only()
        
    print("\n" + "=" * 50)
    
    # 如果有真实数据文件，测试真实用例
    test_file_path = "../my_dataset/my_data_multi_turn_miss_func_train.jsonl"
    test_id = "multi_turn_miss_func_109"
    if os.path.exists(test_file_path):
        test_with_real_case(test_file_path, test_id)
    else:
        print("未找到真实测试数据文件，跳过真实用例测试")


if __name__ == "__main__":
    main() 