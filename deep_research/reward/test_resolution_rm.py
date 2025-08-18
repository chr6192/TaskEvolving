#!/usr/bin/env python3
"""
解决率RM接口测试脚本
"""

import asyncio
import logging
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from deep_research.reward.resolution_rm import ResolutionRMInterface

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_resolution_rm():
    """测试解决率RM接口"""
    
    # 测试数据
    from datetime import datetime
    test_samples = [
        {
            "query": "近十年中国CPI如何变化？",
            "answer": "根据国家统计局的数据，近十年中国CPI整体呈现温和上涨趋势。2014-2019年期间，CPI涨幅相对平稳，年同比涨幅多数在1.5%-2.5%之间。2020年受疫情影响，CPI出现较大波动，年初一度超过5%，但随后逐步回落。2021-2023年，CPI涨幅总体保持在合理区间，反映了宏观经济政策的有效性和经济运行的稳定性。",
            "chat_history": "",
            "chat_date": datetime.now().isoformat()  # 使用当前时间作为测试
        }
    ]
    
    try:
        # 创建解决率RM接口实例
        async with ResolutionRMInterface(
            nickname="测试用户",
            system_code="测试系统",
            timeout=60,
            poll_interval=5,
            max_poll_attempts=12  # 1分钟超时
        ) as rm_interface:
            
            logger.info("开始测试解决率RM接口...")
            
            # 执行批量评估
            results = await rm_interface.evaluate_batch(test_samples)
            
            logger.info(f"评估完成，结果数量: {len(results)}")
            
            for i, result in enumerate(results):
                logger.info(f"样本 {i+1} 解决率分数: {result.get('resolution_score', 0.0)}")
                logger.info(f"原始结果: {result.get('raw_result', {})}")
            
            return results
            
    except Exception as e:
        logger.error(f"测试失败: {e}")
        return []


if __name__ == "__main__":
    results = asyncio.run(test_resolution_rm())
    
    if results:
        print("\n测试成功完成！")
        for i, result in enumerate(results):
            print(f"样本 {i+1}: 解决率分数 = {result.get('resolution_score', 0.0)}")
    else:
        print("\n测试失败或无结果")
        sys.exit(1)
