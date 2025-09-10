"""
解决率RM接口封装
提供批量评测接口的自动化调用
"""

import asyncio
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv
import io

import aiohttp

try:
    import aiofiles  # type: ignore
except ImportError:
    # 如果aiofiles不可用，提供一个简单的替代方案
    aiofiles = None

try:
    import pandas as pd
except ImportError:
    # 如果pandas不可用，使用标准库csv
    pd = None

logger = logging.getLogger(__name__)


class ResolutionRMError(Exception):
    """解决率RM相关异常"""
    pass


class ResolutionRMInterface:
    """
    解决率RM接口封装类
    
    实现三个步骤的自动化串联：
    1. 上传跑批源数据文件
    2. 发起批量评测
    3. 查询任务进度和结果
    """
    
    def __init__(
        self,
        base_url: str = "https://internal.invalid",
        nickname: str = "数据合成系统",
        system_code: str = "蚂小财",
        timeout: int = 3000,
        poll_interval: int = 30,
        max_poll_attempts: int = 60  # 最大轮询次数，对应30分钟
    ):
        """
        初始化解决率RM接口
        
        Args:
            base_url: API基础URL
            nickname: 用户昵称
            system_code: 系统代码
            timeout: 单次请求超时时间（秒）
            poll_interval: 轮询间隔（秒）
            max_poll_attempts: 最大轮询次数
        """
        self.base_url = base_url.rstrip('/')
        self.nickname = nickname
        self.system_code = system_code
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.max_poll_attempts = max_poll_attempts
        
        # 固定的Cookie值（从示例中获取）
        self.headers = {
            'Accept': 'application/json',
            'Cookie': 'buservice_domain_id=KOUBEI_SALESCRM; spanner=XawBCI+toe6/yi4qEexPX8AZF7vhswrnXt2T4qEYgj0='
        }
        
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers=self.headers
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        if self.session:
            await self.session.close()
    
    async def evaluate_batch(
        self,
        data_samples: List[Dict[str, Any]],
        temp_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        批量评估解决率
        
        Args:
            data_samples: 待评估的数据样本列表，每个样本包含：
                - query: 用户问题
                - answer: 助手回答
                - chat_history: 对话历史（可选，默认为空）
                - chat_date: 对话日期（可选，默认为空）
                - trajectory_id: trajectory标识符（用于匹配结果）
            temp_dir: 临时文件目录
            
        Returns:
            评估结果列表，按照输入data_samples的顺序返回
        """
        if not self.session:
            raise ResolutionRMError("请在async with语句中使用此接口")
        
        if not data_samples:
            logger.warning("待评估数据为空")
            return []
        
        logger.info(f"开始批量评估解决率，样本数量: {len(data_samples)}")
        
        try:
            # Step 1: 准备并上传数据文件
            oss_url = await self._upload_data_file(data_samples, temp_dir)
            logger.info(f"数据文件上传成功: {oss_url}")
            
            # Step 2: 发起批量评测任务
            job_id = await self._start_batch_evaluation(oss_url)
            logger.info(f"批量评测任务已启动，任务ID: {job_id}")
            
            # Step 3: 轮询任务状态并获取结果
            csv_results = await self._poll_job_results(job_id)
            logger.info(f"批量评测完成，获得 {len(csv_results)} 个结果")
            
            # Step 4: 匹配输入数据和CSV结果
            matched_results = self._match_results_with_input(data_samples, csv_results)
            
            return matched_results
            
        except Exception as e:
            logger.error(f"批量评估解决率失败: {e}")
            raise ResolutionRMError(f"批量评估失败: {e}") from e
    
    async def _upload_data_file(
        self,
        data_samples: List[Dict[str, Any]],
        temp_dir: Optional[str] = None
    ) -> str:
        """
        Step 1: 上传跑批源数据文件
        
        Args:
            data_samples: 数据样本列表
            temp_dir: 临时文件目录
            
        Returns:
            上传后的OSS URL
        """
        # 准备数据格式
        formatted_data = []
        for sample in data_samples:
            formatted_sample = {
                "chat_history": sample.get("chat_history", ""),  # 可以为空
                "query": sample.get("query", ""),
                "answer": sample.get("answer", ""),
                "chat_date": sample.get("chat_date", "")  # 可以为空
            }
            formatted_data.append(formatted_sample)
        
        # 创建临时文件
        temp_dir_path = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
        temp_dir_path.mkdir(parents=True, exist_ok=True)
        
        temp_file = temp_dir_path / f"resolution_rm_data_{int(time.time())}.json"
        
        try:
            # 写入JSON文件
            if aiofiles:
                async with aiofiles.open(temp_file, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(formatted_data, ensure_ascii=False, indent=2))
            else:
                # 同步写入作为fallback
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(formatted_data, ensure_ascii=False, indent=2))
            
            logger.debug(f"临时数据文件已创建: {temp_file}")
            
            # 上传文件
            url = f"{self.base_url}/external/batch/upload"
            
            if aiofiles:
                async with aiofiles.open(temp_file, 'rb') as f:
                    file_content = await f.read()
            else:
                # 同步读取作为fallback
                with open(temp_file, 'rb') as f:
                    file_content = f.read()
            
            data = aiohttp.FormData()
            data.add_field('file', file_content, 
                          filename=temp_file.name, 
                          content_type='application/json')
            data.add_field('nickName', self.nickname)
            
            async with self.session.post(url, data=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ResolutionRMError(f"文件上传失败，状态码: {response.status}, 错误: {error_text}")
                
                result = await response.json()
                
                if not result.get('success', False):
                    raise ResolutionRMError(f"文件上传失败: {result}")
                
                oss_url = result['value']['ossUrl']
                return oss_url
        
        finally:
            # 清理临时文件
            if temp_file.exists():
                temp_file.unlink()
                logger.debug(f"临时文件已删除: {temp_file}")
    
    async def _start_batch_evaluation(self, oss_url: str) -> int:
        """
        Step 2: 发起批量评测
        
        Args:
            oss_url: 上传文件的OSS URL
            
        Returns:
            批量评测任务ID
        """
        url = f"{self.base_url}/external/batch/batchRunfinanceResolutionEvaluationJob"
        
        payload = {
            "nickName": self.nickname,
            "systemCode": self.system_code,
            "dataUrl": oss_url
        }
        
        headers = {
            **self.headers,
            'Content-Type': 'application/json'
        }
        
        async with self.session.post(url, json=payload, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise ResolutionRMError(f"启动批量评测失败，状态码: {response.status}, 错误: {error_text}")
            
            result = await response.json()
            
            if not result.get('success', False):
                raise ResolutionRMError(f"启动批量评测失败: {result}")
            
            job_id = result['value']
            return job_id
    
    async def _poll_job_results(self, job_id: int) -> List[Dict[str, Any]]:
        """
        Step 3: 轮询任务状态并获取结果
        
        Args:
            job_id: 批量评测任务ID
            
        Returns:
            评测结果列表
        """
        url = f"{self.base_url}/external/batch/queryGptJobReport/{job_id}"
        
        for attempt in range(self.max_poll_attempts):
            try:
                async with self.session.get(url) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.warning(f"查询任务状态失败，状态码: {response.status}, 错误: {error_text}")
                        
                        # 如果是临时错误，继续轮询
                        if response.status in [500, 502, 503, 504]:
                            logger.info(f"遇到临时错误，等待 {self.poll_interval} 秒后重试...")
                            await asyncio.sleep(self.poll_interval)
                            continue
                        else:
                            raise ResolutionRMError(f"查询任务状态失败，状态码: {response.status}")
                    
                    result = await response.json()
                    
                    # 检查任务状态
                    if 'status' in result:
                        status = result['status']
                        if status == 'COMPLETED':
                            # 任务完成，返回结果
                            return self._parse_evaluation_results(result)
                        elif status == 'FAILED':
                            raise ResolutionRMError(f"批量评测任务失败: {result.get('error', '未知错误')}")
                        elif status in ['RUNNING', 'PENDING']:
                            # 任务还在运行，继续轮询
                            logger.info(f"任务状态: {status}, 等待 {self.poll_interval} 秒后继续查询...")
                            await asyncio.sleep(self.poll_interval)
                            continue
                    
                    # 检查是否返回了CSV文件URL
                    if result.get('success') and 'value' in result and isinstance(result['value'], str) and result['value'].endswith('.csv'):
                        # 下载并解析CSV文件
                        csv_url = result['value']
                        logger.info(f"任务完成，开始下载CSV结果文件: {csv_url}")
                        return await self._download_and_parse_csv(csv_url)
                    
                    # 如果返回结果中包含数据，假设任务已完成
                    if 'data' in result or 'results' in result:
                        return self._parse_evaluation_results(result)
                    
                    # 继续等待
                    logger.info(f"第 {attempt + 1} 次查询url: {url}，任务可能仍在进行中，等待 {self.poll_interval} 秒后重试...,接口返回结果：{result}")
                    await asyncio.sleep(self.poll_interval)
            
            except aiohttp.ClientError as e:
                logger.warning(f"网络请求错误: {e}，等待 {self.poll_interval} 秒后重试...")
                await asyncio.sleep(self.poll_interval)
                continue
            except json.JSONDecodeError as e:
                logger.warning(f"JSON解析错误: {e}，等待 {self.poll_interval} 秒后重试...")
                await asyncio.sleep(self.poll_interval)
                continue
        
        # 超过最大轮询次数
        raise ResolutionRMError(f"任务查询超时，已轮询 {self.max_poll_attempts} 次")
    
    def _is_nan_value(self, value) -> bool:
        """
        检查值是否为NaN（各种形式）
        
        Args:
            value: 待检查的值
            
        Returns:
            如果是NaN值返回True，否则返回False
        """
        if value is None:
            return True
        
        # 检查字符串形式的NaN
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.lower() in ['nan', 'null', 'none'] or stripped == '':
                return True
            return False
        
        # 检查数值形式的NaN
        if isinstance(value, (int, float)):
            try:
                import math
                return math.isnan(float(value)) or math.isinf(float(value))
            except (ValueError, TypeError):
                return False
        
        return False
    
    async def _download_and_parse_csv(self, csv_url: str) -> List[Dict[str, Any]]:
        """
        下载并解析CSV结果文件
        
        Args:
            csv_url: CSV文件的URL
            
        Returns:
            解析后的评估结果列表
        """
        try:
            logger.info(f"开始下载CSV文件: {csv_url}")
            
            # 下载CSV文件
            async with self.session.get(csv_url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ResolutionRMError(f"下载CSV文件失败，状态码: {response.status}, 错误: {error_text}")
                
                csv_content = await response.text(encoding='utf-8')
                logger.info(f"CSV文件下载成功，大小: {len(csv_content)} 字符")
            
            # 解析CSV内容
            return self._parse_csv_content(csv_content)
            
        except Exception as e:
            logger.error(f"下载或解析CSV文件失败: {e}")
            raise ResolutionRMError(f"处理CSV结果文件失败: {e}") from e
    
    def _parse_csv_content(self, csv_content: str) -> List[Dict[str, Any]]:
        """
        解析CSV内容
        
        Args:
            csv_content: CSV文件内容
            
        Returns:
            解析后的评估结果列表
        """
        results = []
        
        try:
            # 使用pandas解析CSV（如果可用）
            if pd is not None:
                df = pd.read_csv(io.StringIO(csv_content))
                
                for _, row in df.iterrows():
                    result = self._parse_csv_row(row.to_dict())
                    if result:
                        results.append(result)
            else:
                # 使用标准库csv解析
                csv_reader = csv.DictReader(io.StringIO(csv_content))
                
                for row in csv_reader:
                    result = self._parse_csv_row(row)
                    if result:
                        results.append(result)
            
            logger.info(f"成功解析CSV，获得 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"解析CSV内容失败: {e}")
            logger.error(f"CSV内容预览: {csv_content[:500]}...")
            return []
    
    def _parse_csv_row(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        解析CSV中的单行数据
        
        Args:
            row: CSV行数据字典
            
        Returns:
            解析后的结果字典，如果解析失败返回None
        """
        try:
            # 提取必要字段
            ai_model_out_str = row.get('ai_model_out', '{}')
            answer = row.get('answer', '')
            query = row.get('query', '')
            chat_date = row.get('chat_date', '')
            index = row.get('index', '')
            chat_history = row.get('chat_history', '')
            
            # 解析ai_model_out JSON字符串
            try:
                ai_model_out = json.loads(ai_model_out_str) if ai_model_out_str else {}
            except json.JSONDecodeError as e:
                logger.warning(f"解析ai_model_out JSON失败: {e}, 原始内容: {ai_model_out_str[:200]}...")
                ai_model_out = {}
            
            # 提取总分并归一化到0-1
            total_score = ai_model_out.get('总分', 0)
            resolution_score = float(total_score) / 100.0 if total_score else 0.0
            resolution_score = max(0.0, min(1.0, resolution_score))  # 确保在0-1范围内
            
            result = {
                'resolution_score': resolution_score,
                'query': query,
                'answer': answer,
                'ai_model_out': ai_model_out,
                'chat_date': chat_date,
                'index': index,
                'chat_history': chat_history,
            }
            
            return result
            
        except Exception as e:
            logger.error(f"解析CSV行数据失败: {e}, 行数据: {row}")
            return None
    
    def _match_results_with_input(
        self, 
        data_samples: List[Dict[str, Any]], 
        csv_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        将CSV结果与输入数据进行匹配
        
        Args:
            data_samples: 输入的数据样本列表
            csv_results: 从CSV解析的结果列表
            
        Returns:
            按输入顺序排列的匹配结果列表
        """
        matched_results = []
        
        try:
            # 为CSV结果建立索引，使用query和answer的组合作为键
            csv_index = {}
            for csv_result in csv_results:
                query = csv_result.get('query', '').strip()
                answer = csv_result.get('answer', '').strip()
                key = f"{query}|||{answer}"  # 使用特殊分隔符避免冲突
                csv_index[key] = csv_result
            
            # 为每个输入样本查找对应的CSV结果
            for i, sample in enumerate(data_samples):
                input_query = sample.get('query', '').strip()
                input_answer = sample.get('answer', '').strip()
                search_key = f"{input_query}|||{input_answer}"
                
                if search_key in csv_index:
                    # 找到匹配的结果
                    matched_result = csv_index[search_key].copy()
                    
                    # 关键修复：如果CSV中的chat_history为NaN，使用输入数据中的chat_history
                    csv_chat_history = matched_result.get('chat_history', '')
                    if self._is_nan_value(csv_chat_history):
                        matched_result['chat_history'] = sample.get('chat_history', '')
                        logger.debug(f"CSV中chat_history为NaN，使用输入数据: {sample.get('chat_history', '')}")
                    
                    # 添加trajectory_id信息（如果有的话）
                    if 'trajectory_id' in sample:
                        matched_result['trajectory_id'] = sample['trajectory_id']
                    matched_results.append(matched_result)
                    logger.debug(f"成功匹配样本 {i}: trajectory_id={sample.get('trajectory_id', 'N/A')}")
                else:
                    # 未找到匹配结果，创建默认结果
                    logger.warning(f"未找到样本 {i} 的匹配结果: query={input_query[:50]}..., answer={input_answer[:50]}...")
                    default_result = {
                        'resolution_score': 0.0,
                        'query': input_query,
                        'answer': input_answer,
                        'ai_model_out': {},
                        'chat_date': sample.get('chat_date', ''),
                        'index': str(i),
                        'chat_history': sample.get('chat_history', '')
                    }
                    if 'trajectory_id' in sample:
                        default_result['trajectory_id'] = sample['trajectory_id']
                    matched_results.append(default_result)
            
            logger.info(f"结果匹配完成：{len(matched_results)} 个结果，其中 {len([r for r in matched_results if r.get('resolution_score', 0) > 0])} 个有有效分数")
            return matched_results
            
        except Exception as e:
            logger.error(f"结果匹配失败: {e}")
            # 返回默认结果列表
            return [
                {
                    'resolution_score': 0.0,
                    'query': sample.get('query', ''),
                    'answer': sample.get('answer', ''),
                    'ai_model_out': {},
                    'chat_date': sample.get('chat_date', ''),
                    'index': str(i),
                    'chat_history': sample.get('chat_history', ''),
                    'trajectory_id': sample.get('trajectory_id', '')
                }
                for i, sample in enumerate(data_samples)
            ]
    
    def _parse_evaluation_results(self, response_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        解析评估结果
        
        Args:
            response_data: API响应数据
            
        Returns:
            解析后的评估结果列表
        """
        results = []
        
        try:
            # 尝试不同的数据结构解析
            if 'data' in response_data:
                raw_results = response_data['data']
            elif 'results' in response_data:
                raw_results = response_data['results']
            elif 'value' in response_data:
                raw_results = response_data['value']
            else:
                # 如果没有明确的数据字段，尝试直接解析
                raw_results = response_data
            
            # 确保raw_results是列表
            if not isinstance(raw_results, list):
                if isinstance(raw_results, dict):
                    raw_results = [raw_results]
                else:
                    logger.warning(f"无法解析评估结果格式: {type(raw_results)}")
                    return []
            
            # 解析每个结果
            for item in raw_results:
                if isinstance(item, dict):
                    # 提取解决率分数
                    resolution_score = item.get('resolution_score', item.get('score', 0.0))
                    
                    # 构建标准化结果
                    result = {
                        'resolution_score': float(resolution_score),
                        'raw_result': item,  # 保留原始结果用于调试
                        'metadata': {
                            'evaluation_time': item.get('evaluation_time'),
                            'model_version': item.get('model_version'),
                            'confidence': item.get('confidence')
                        }
                    }
                    results.append(result)
                else:
                    logger.warning(f"跳过无效的评估结果项: {item}")
            
            logger.info(f"成功解析 {len(results)} 个评估结果")
            return results
        
        except Exception as e:
            logger.error(f"解析评估结果失败: {e}")
            logger.error(f"原始响应数据: {response_data}")
            
            # 返回默认结果，避免完全失败
            return [{'resolution_score': 0.0, 'raw_result': response_data, 'metadata': {}}]


async def evaluate_resolution_scores(
    data_samples: List[Dict[str, Any]],
    **kwargs
) -> List[float]:
    """
    便捷函数：评估解决率分数
    
    Args:
        data_samples: 数据样本列表
        **kwargs: 传递给ResolutionRMInterface的参数
        
    Returns:
        解决率分数列表
    """
    async with ResolutionRMInterface(**kwargs) as rm_interface:
        results = await rm_interface.evaluate_batch(data_samples)
        return [result.get('resolution_score', 0.0) for result in results]


# 同步包装函数（用于兼容性）
def evaluate_resolution_scores_sync(
    data_samples: List[Dict[str, Any]],
    **kwargs
) -> List[float]:
    """
    同步版本的解决率评估函数
    
    Args:
        data_samples: 数据样本列表
        **kwargs: 传递给ResolutionRMInterface的参数
        
    Returns:
        解决率分数列表
    """
    return asyncio.run(evaluate_resolution_scores(data_samples, **kwargs))
