"""
日志配置模块
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    设置日志配置
    
    Args:
        level: 日志级别 ("DEBUG", "INFO", "WARNING", "ERROR")
        log_file: 可选的日志文件路径
    """
    # 设置日志级别
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # 创建根logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # 清除现有handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 文件handler（如果指定了日志文件）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # 配置第三方库的日志级别
    _configure_third_party_loggers()
    
    logging.info(f"日志系统初始化完成，级别: {level}, 文件: {log_file or 'None'}")


def _configure_third_party_loggers():
    """配置第三方库的日志级别"""
    # 设置第三方库的日志级别，避免过多输出
    third_party_loggers = {
        "transformers": logging.WARNING,
        "torch": logging.WARNING,
        "vllm": logging.INFO,
        "ray": logging.WARNING,
        "asyncio": logging.WARNING,
        "aiohttp": logging.WARNING,
        "urllib3": logging.WARNING,
        "requests": logging.WARNING,
        "httpx": logging.WARNING,
        "numpy": logging.WARNING,
        "pandas": logging.WARNING,
    }
    
    for logger_name, log_level in third_party_loggers.items():
        logging.getLogger(logger_name).setLevel(log_level)
    
    # 保持深度研究相关的日志为正常级别
    logging.getLogger("deep_research").setLevel(logging.INFO)
    logging.getLogger("verl").setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的logger
    
    Args:
        name: logger名称
        
    Returns:
        Logger实例
    """
    return logging.getLogger(name)