"""
Alternative.me 数据客户端
恐惧贪婪指数 - 完全免费，无限制
"""

import asyncio
import logging
import aiohttp
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ...input_types import (
    FearGreedIndex, SentimentData, DataSourceType
)

logger = logging.getLogger(__name__)


@dataclass
class AlternativeMeConfig:
    """Alternative.me配置"""
    base_url: str = "https://api.alternative.me/fng"
    rate_limit: float = 60.0  # 实际日更，但每小时检查
    timeout: int = 30


class AlternativeMeClient:
    """Alternative.me恐惧贪婪指数客户端"""
    
    # 情绪分类
    SENTIMENT_CLASSIFICATIONS = {
        (0, 24): "Extreme Fear",
        (25, 46): "Fear",
        (47, 54): "Neutral",
        (55, 75): "Greed",
        (76, 100): "Extreme Greed"
    }
    
    def __init__(self, config: Optional[AlternativeMeConfig] = None):
        self.config = config or AlternativeMeConfig()
        self.session: Optional[aiohttp.ClientSession] = None
        self._last_request_time: Optional[datetime] = None
        self._cache: Dict[str, Any] = {}
        self._request_count = 0
        self._error_count = 0
        
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
    
    async def start(self):
        """启动客户端"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        logger.info("Alternative.me客户端已启动")
    
    async def stop(self):
        """停止客户端"""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info(f"Alternative.me客户端已停止 (请求: {self._request_count}, 错误: {self._error_count})")
    
    async def _rate_limit(self):
        """速率限制"""
        if self._last_request_time:
            elapsed = (datetime.now() - self._last_request_time).total_seconds()
            if elapsed < self.config.rate_limit:
                wait_time = self.config.rate_limit - elapsed
                await asyncio.sleep(wait_time)
        self._last_request_time = datetime.now()
    
    def _get_classification(self, value: int) -> str:
        """根据数值获取情绪分类"""
        for (low, high), classification in self.SENTIMENT_CLASSIFICATIONS.items():
            if low <= value <= high:
                return classification
        return "Unknown"
    
    async def get_current_index(self, use_cache: bool = True) -> Optional[FearGreedIndex]:
        """获取当前恐惧贪婪指数"""
        cache_key = 'current_index'
        
        # 检查缓存
        if use_cache and cache_key in self._cache:
            cached_item = self._cache[cache_key]
            if (datetime.now() - cached_item['timestamp']).seconds < 3600:  # 1小时缓存
                return cached_item['data']
        
        await self._rate_limit()
        
        try:
            async with self.session.get(self.config.base_url, timeout=self.config.timeout) as response:
                self._request_count += 1
                
                if response.status == 200:
                    data = await response.json()
                    
                    if data and 'data' in data and len(data['data']) > 0:
                        item = data['data'][0]
                        value = int(item.get('value', 50))
                        
                        index = FearGreedIndex(
                            timestamp=datetime.now(),
                            value=value,
                            classification=self._get_classification(value),
                            source=DataSourceType.ALTERNATIVE_ME
                        )
                        
                        # 更新缓存
                        if use_cache:
                            self._cache[cache_key] = {
                                'data': index,
                                'timestamp': datetime.now()
                            }
                        
                        return index
                else:
                    error_text = await response.text()
                    logger.error(f"获取恐惧贪婪指数失败: {response.status}, {error_text}")
                    self._error_count += 1
                    return None
        
        except Exception as e:
            logger.error(f"请求错误: {e}")
            self._error_count += 1
            return None
    
    async def get_historical_data(
        self,
        limit: Optional[int] = None,
        use_cache: bool = True
    ) -> List[FearGreedIndex]:
        """
        获取历史恐惧贪婪指数
        
        Args:
            limit: 返回数量 (默认全部)
        """
        cache_key = f'historical_{limit}'
        
        # 检查缓存
        if use_cache and cache_key in self._cache:
            cached_item = self._cache[cache_key]
            if (datetime.now() - cached_item['timestamp']).seconds < 3600:
                return cached_item['data']
        
        await self._rate_limit()
        
        params = {}
        if limit:
            params['limit'] = limit
        
        try:
            async with self.session.get(
                self.config.base_url,
                params=params,
                timeout=self.config.timeout
            ) as response:
                self._request_count += 1
                
                if response.status == 200:
                    data = await response.json()
                    
                    indices = []
                    if data and 'data' in data:
                        for item in data['data']:
                            try:
                                value = int(item.get('value', 50))
                                timestamp = datetime.fromtimestamp(int(item.get('timestamp', 0)))
                                
                                indices.append(FearGreedIndex(
                                    timestamp=timestamp,
                                    value=value,
                                    classification=self._get_classification(value),
                                    source=DataSourceType.ALTERNATIVE_ME
                                ))
                            except Exception as e:
                                logger.error(f"解析历史数据失败: {e}")
                    
                    # 更新缓存
                    if use_cache:
                        self._cache[cache_key] = {
                            'data': indices,
                            'timestamp': datetime.now()
                        }
                    
                    return indices
                else:
                    error_text = await response.text()
                    logger.error(f"获取历史数据失败: {response.status}, {error_text}")
                    self._error_count += 1
                    return []
        
        except Exception as e:
            logger.error(f"请求错误: {e}")
            self._error_count += 1
            return []
    
    async def get_sentiment_signals(self) -> Dict[str, Any]:
        """获取情绪信号"""
        current = await self.get_current_index()
        historical = await get_historical_data(self, limit=30)
        
        if not current or not historical:
            return {}
        
        # 计算统计
        values = [h.value for h in historical]
        avg_7d = sum(values[:7]) / 7 if len(values) >= 7 else sum(values) / len(values)
        avg_30d = sum(values) / len(values)
        
        # 趋势判断
        trend = "neutral"
        if len(values) >= 2:
            if values[0] > values[1]:
                trend = "improving"
            elif values[0] < values[1]:
                trend = "worsening"
        
        # 信号生成
        signals = {
            'extreme_fear': current.value <= 20,
            'fear': 20 < current.value <= 45,
            'neutral': 45 < current.value <= 55,
            'greed': 55 < current.value <= 75,
            'extreme_greed': current.value > 75,
            'mean_reversion_opportunity': current.value <= 20 or current.value > 75,
        }
        
        return {
            'current': current,
            'average_7d': avg_7d,
            'average_30d': avg_30d,
            'trend': trend,
            'signals': signals,
            'interpretation': self._interpret_index(current.value)
        }
    
    def _interpret_index(self, value: int) -> str:
        """解读恐惧贪婪指数"""
        if value <= 20:
            return "极度恐惧 - 可能是买入机会"
        elif value <= 40:
            return "恐惧 - 市场悲观，谨慎买入"
        elif value <= 60:
            return "中性 - 市场情绪平衡"
        elif value <= 80:
            return "贪婪 - 市场乐观，注意风险"
        else:
            return "极度贪婪 - 可能是卖出信号"
    
    # ==================== 统计信息 ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取客户端统计信息"""
        return {
            'total_requests': self._request_count,
            'error_count': self._error_count,
            'error_rate': self._error_count / max(self._request_count, 1),
            'cache_size': len(self._cache)
        }
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        logger.info("Alternative.me缓存已清空")
