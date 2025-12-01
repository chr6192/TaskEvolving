"""
Reward API 客户端
"""
import time
import requests
import json
from typing import Dict, Any, List


class RewardAPIClient:
    """Reward API 客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def compute_sync(
        self,
        prompt: str,
        response_list: List[str],
        messages_list: List[List[Dict]],
        extra_list: List[Dict],
        config: Dict = None
    ) -> Dict[str, Any]:
        """同步计算奖励"""
        url = f"{self.base_url}/api/v1/rewards/compute"
        data = {
            "prompt": prompt,
            "response_list": response_list,
            "messages_list": messages_list,
            "extra_list": extra_list,
            "config": config or {}
        }
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()
    
    def compute_async(
        self,
        prompt: str,
        response_list: List[str],
        messages_list: List[List[Dict]],
        extra_list: List[Dict],
        config: Dict = None,
        task_id: str = None
    ) -> str:
        """异步提交任务，返回任务ID"""
        url = f"{self.base_url}/api/v1/rewards/compute_async"
        data = {
            "task_id": task_id,
            "data": {
                "prompt": prompt,
                "response_list": response_list,
                "messages_list": messages_list,
                "extra_list": extra_list,
                "config": config or {}
            }
        }
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()["task_id"]
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """查询任务状态"""
        response = requests.get(f"{self.base_url}/api/v1/rewards/task/{task_id}")
        response.raise_for_status()
        return response.json()
    
    def wait_for_task(
        self,
        task_id: str,
        poll_interval: float = 2.0,
        timeout: float = 300.0
    ) -> Dict[str, Any]:
        """等待任务完成"""
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"任务超时: {task_id}")
            
            status = self.get_task_status(task_id)
            if status["status"] == "completed":
                return status["result"]
            elif status["status"] == "failed":
                raise RuntimeError(f"任务失败: {status['error']}")
            
            print(f"任务 {task_id}: {status['status']}, 进度 {status['progress']:.0%}")
            time.sleep(poll_interval)
    
    def compute_batch(
        self,
        groups: List[Dict[str, Any]],
        config: Dict = None
    ) -> Dict[str, Any]:
        """批量计算多个group"""
        url = f"{self.base_url}/api/v1/rewards/batch_compute"
        data = {"groups": groups, "config": config or {}}
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取服务监控指标"""
        response = requests.get(f"{self.base_url}/api/v1/metrics")
        response.raise_for_status()
        return response.json()
    
    def clear_cache(self) -> Dict[str, Any]:
        """清空缓存"""
        response = requests.post(f"{self.base_url}/api/v1/cache/clear")
        response.raise_for_status()
        return response.json()


def main():
    """运行示例"""
    client = RewardAPIClient()
    
    # 检查服务
    try:
        health = client.health_check()
        print(f"服务状态: {health['status']}")
    except Exception as e:
        print(f"无法连接到服务: {e}")
        print("请先启动服务: bash deep_research/reward/api/start.sh")
        return

if __name__ == "__main__":
    main()
