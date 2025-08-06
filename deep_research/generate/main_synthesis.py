"""
Deep Research 数据合成Pipeline主入口
"""

import asyncio
import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="data_synthesis")
def main(cfg: DictConfig) -> None:
    """
    主入口函数
    
    Args:
        cfg: Hydra配置对象
    """
    try:
        # 验证配置
        if not _validate_config(cfg):
            logger.error("配置验证失败")
            sys.exit(1)
        
        # 设置命令行参数
        _setup_config_from_args(cfg)
        
        # 运行异步pipeline
        asyncio.run(_run_pipeline_async(cfg))
        
        logger.info("数据合成pipeline成功完成")
        
    except KeyboardInterrupt:
        logger.info("用户中断执行")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline执行失败: {e}", exc_info=True)
        sys.exit(1)


async def _run_pipeline_async(cfg: DictConfig) -> None:
    """异步运行数据合成pipeline"""
    import ray
    
    from .core.data_synthesis_pipeline import DataSynthesisPipeline
    from .utils.data_loader import load_skeleton_data_from_parquet, load_queries_from_file
    from .utils.logger_setup import setup_logging
    
    # 设置日志
    setup_logging(cfg.experiment.logging_level, 
                  cfg.output.log_dir + "/synthesis.log" if cfg.output.log_dir else None)
    logger = logging.getLogger(__name__)
    
    logger.info("开始数据合成pipeline")
    logger.info(f"阶段: {cfg.stage}")
    logger.info(f"模型: {cfg.model.name}")
    logger.info(f"输入文件: {cfg.input.queries_file}")
    
    try:
        # 根据执行模式决定是否初始化Ray
        execution_mode = cfg.get('execution_mode', 'local')
        if execution_mode == 'local':
            # 初始化Ray（用于本地LLM）
            if not ray.is_initialized():
                ray.init(runtime_env={
                    "env_vars": {
                        "TOKENIZERS_PARALLELISM": "true",
                        "NCCL_DEBUG": "WARN", 
                        "VLLM_LOGGING_LEVEL": "INFO",
                        "VLLM_USE_V1": "1",
                    }
                })
                logger.info("Ray已初始化用于本地LLM")
        elif execution_mode == 'openai_api':
            logger.info("使用OpenAI API模式，跳过Ray初始化")
            # 验证OpenAI API配置
            if not _validate_openai_config(cfg):
                raise ValueError("OpenAI API配置验证失败")
        else:
            raise ValueError(f"不支持的执行模式: {execution_mode}")
        
        # 创建输出目录
        _create_output_directories(cfg)
        
        # 初始化pipeline
        pipeline = DataSynthesisPipeline(cfg)
        await pipeline.initialize()
        
        try:
            if cfg.stage == "trajectory":
                # 仅执行阶段二：trajectory生成
                logger.info("执行阶段二：从skeleton生成trajectory")
                
                # 从parquet文件读取skeleton数据
                skeletons = load_skeleton_data_from_parquet(cfg.input.queries_file)
                logger.info(f"从parquet文件加载了 {len(skeletons)} 个skeleton")
                
                trajectories = await pipeline.run_stage2(skeletons)
                logger.info(f"生成了 {len(trajectories)} 个trajectory")
                
            elif cfg.stage == "skeleton":
                # 仅执行阶段一：skeleton生成
                logger.info("执行阶段一：skeleton生成")
                
                if cfg.input.queries_file.endswith('.parquet'):
                    # 从parquet文件中提取queries
                    skeleton_data = load_skeleton_data_from_parquet(cfg.input.queries_file)
                    from .core.data_structures import QueryData
                    queries = [
                        QueryData(
                            query_id=s.query_id,
                            content=s.query_content,
                            metadata=s.metadata
                        ) for s in skeleton_data
                    ]
                else:
                    # 从标准查询文件加载
                    queries = load_queries_from_file(cfg.input.queries_file)
                
                skeletons = await pipeline.run_stage1(queries)
                logger.info(f"生成了 {len(skeletons)} 个skeleton")
                
            elif cfg.stage == "both":
                # 执行完整pipeline
                logger.info("执行完整pipeline")
                
                if cfg.input.queries_file.endswith('.parquet'):
                    # 从parquet文件中提取queries
                    skeleton_data = load_skeleton_data_from_parquet(cfg.input.queries_file)
                    from .core.data_structures import QueryData
                    queries = [
                        QueryData(
                            query_id=s.query_id,
                            content=s.query_content,
                            metadata=s.metadata
                        ) for s in skeleton_data
                    ]
                else:
                    # 从标准查询文件加载
                    queries = load_queries_from_file(cfg.input.queries_file)
                
                trajectories = await pipeline.run_full_pipeline(queries)
                logger.info(f"完整pipeline生成了 {len(trajectories)} 个trajectory")
                
            else:
                raise ValueError(f"未知的阶段: {cfg.stage}")
            
            # 输出统计信息
            stats = pipeline.get_stats()
            logger.info("=== Pipeline统计信息 ===")
            for key, value in stats.to_dict().items():
                logger.info(f"{key}: {value}")
            
        finally:
            # 清理资源
            await pipeline.cleanup()
            
    finally:
        # 关闭Ray
        if ray.is_initialized():
            ray.shutdown()
            
    logger.info("数据合成pipeline完成")


def _validate_config(cfg: DictConfig) -> bool:
    """验证配置参数"""
    required_fields = [
        "stage", "model.name", "model.path", "input.queries_file"
    ]
    
    for field in required_fields:
        try:
            value = cfg
            for key in field.split('.'):
                value = getattr(value, key)
            
            if value is None:
                logger.error(f"缺少必需配置项: {field}")
                return False
                
        except AttributeError:
            logger.error(f"缺少必需配置项: {field}")
            return False
    
    # 验证阶段参数
    if cfg.stage not in ["skeleton", "trajectory", "both"]:
        logger.error(f"无效的阶段参数: {cfg.stage}")
        return False
    
    # 验证输入文件存在
    input_file = Path(cfg.input.queries_file)
    if not input_file.exists():
        logger.error(f"输入文件不存在: {cfg.input.queries_file}")
        return False
        
    return True


def _validate_openai_config(cfg: DictConfig) -> bool:
    """验证OpenAI API配置"""
    import os
    
    openai_config = cfg.get('openai_api', {})
    
    # 检查API key
    api_key = openai_config.get('api_key') or os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OpenAI API key未设置。请在配置文件中设置 openai_api.api_key 或设置环境变量 OPENAI_API_KEY")
        return False
    
    # 检查模型名称
    model_path = cfg.model.path
    if not model_path or model_path == "/path/to/your/model":
        logger.error("使用OpenAI API时，请将 model.path 设置为有效的模型名称（如 'gpt-4', 'gpt-3.5-turbo'）")
        return False
    
    logger.info(f"OpenAI API配置验证通过，使用模型: {model_path}")
    return True


def _setup_config_from_args(cfg: DictConfig) -> None:
    """从命令行参数设置配置"""
    import sys
    
    # 解析命令行参数
    args = sys.argv[1:]
    for arg in args:
        if arg.startswith("input.queries_file="):
            cfg.input.queries_file = arg.split("=", 1)[1]
        elif arg.startswith("stage="):
            cfg.stage = arg.split("=", 1)[1]
        elif arg.startswith("model.name="):
            cfg.model.name = arg.split("=", 1)[1]
        elif arg.startswith("model.path="):
            cfg.model.path = arg.split("=", 1)[1]


def _create_output_directories(cfg: DictConfig) -> None:
    """创建输出目录"""
    from pathlib import Path
    
    if cfg.output.skeleton_dir:
        Path(cfg.output.skeleton_dir).mkdir(parents=True, exist_ok=True)
        
    if cfg.output.trajectory_dir:
        Path(cfg.output.trajectory_dir).mkdir(parents=True, exist_ok=True)
        
    if cfg.output.log_dir:
        Path(cfg.output.log_dir).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()