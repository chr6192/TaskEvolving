"""
工具加载工具函数
用于从YAML配置文件加载工具schemas
"""
import yaml
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


def load_tool_schemas_from_yaml(config_path: str = None) -> List[Dict[str, Any]]:
    """
    从YAML配置文件加载工具schemas，使用已有的initialize_tools_from_config函数
    
    Args:
        config_path: YAML配置文件路径，如果为None则抛出异常
        
    Returns:
        List[Dict]: 工具schema列表
    """
    if config_path is None:
        raise ValueError("config_path is required")
    
    try:
        from verl.tools.base_tool import initialize_tools_from_config
        
        tools = initialize_tools_from_config(config_path)
        
        tools_info = []
        for tool in tools:
            if hasattr(tool, 'tool_schema') and tool.tool_schema:
                function_info = tool.tool_schema.function
                tools_info.append({
                    'name': function_info.name,
                    'description': function_info.description,
                    'parameters': function_info.parameters.dict() if hasattr(function_info.parameters, 'dict') else function_info.parameters
                })
        
        logger.info(f"成功从 {config_path} 加载 {len(tools_info)} 个工具")
        return tools_info
    except Exception as e:
        logger.warning(f"使用initialize_tools_from_config加载失败: {e}，尝试回退方案")
        # 回退到原有逻辑
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            tools_info = []
            for tool in config.get('tools', []):
                tool_schema = tool.get('tool_schema', {})
                function_info = tool_schema.get('function', {})
                
                if function_info:
                    tools_info.append({
                        'name': function_info.get('name', ''),
                        'description': function_info.get('description', ''),
                        'parameters': function_info.get('parameters', {})
                    })
            
            logger.info(f"使用回退方案成功从 {config_path} 加载 {len(tools_info)} 个工具")
            return tools_info
        except Exception as fallback_e:
            logger.error(f"回退方案也失败: {fallback_e}")
            return []

