"""
Binance 期货情绪数据客户端
多空比、吃单量统计
使用 Binance 公开数据 API
"""

import asyncio
import logging
import aiohttp
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ...input_types import (
    LongShortRatio, DataSourceType
)

logger = logging.getLogger(__name__)


@dataclass
class BinanceSentimentConfig:
    """Binance情绪数据配置"""
    base_url: str = "https://fapi.binance.com/futures/data"
    rate_limit: float = 0.2  # 5 req/sec
    timeout: int = 30


class BinanceSentimentClient:
    """Binance期货情绪数据客户端"""
    
    def __init__(self, config: Optional[BinanceSentimentConfig] = None):
        self.config = config or BinanceSentimentConfig()
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
        logger.info("Binance情绪数据客户端已启动")
    
    async def stop(self):
        """停止客户端"""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info(f"Binance情绪数据客户端已停止 (请求: {self._request_count}, 错误: {self._error_count})")
    
    async def _rate_limit(self):
        """速率限制"""
        if self._last_request_time:
            elapsed = (datetime.now() - self._last_request_time).total_seconds()
            if elapsed < self.config.rate_limit:
                wait_time = self.config.rate_limit - elapsed
                await asyncio.sleep(wait_time)
        self._last_request_time = datetime.now()
    
    async def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        use_cache: bool = True,
        cache_ttl: int = 60
    ) -> Optional[Any]:
        """发送API请求"""
        cache_key = f"{endpoint}:{str(params)}"
        
        # 检查缓存
        if use_cache and cache_key in self._cache:
            cached_item = self._cache[cache_key]
            if (datetime.now() - cached_item['timestamp']).seconds < cache_ttl:
                return cached_item['data']
        
        await self._rate_limit()
        
        url = f"{self.config.base_url}{endpoint}"
        
        try:
            async with self.session.get(url, params=params, timeout=self.config.timeout) as response:
                self._request_count += 1
                
                if response.status == 200:
                    data = await response.json()
                    
                    # 更新缓存
                    if use_cache:
                        self._cache[cache_key] = {
                            'data': data,
                            'timestamp': datetime.now()
                        }
                    
                    return data
                
                elif response.status == 429:
                    logger.warning("Binance API速率限制，等待后重试...")
                    await asyncio.sleep(5)
                    return None
                
                else:
                    error_text = await response.text()
                    logger.error(f"API请求失败: {response.status}, {error_text}")
                    self._error_count += 1
                    return None
        
        except Exception as e:
            logger.error(f"请求错误: {e}")
            self._error_count += 1
            return None
    
    # ==================== 多空比 API ====================
    
    async def get_global_long_short_account_ratio(
        self,
        symbol: str,
        period: str = "5m",
        limit: int = 1,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[LongShortRatio]:
        """
        获取全账户多空比
        
        Args:
            symbol: 交易对，如 "BTCUSDT"
            period: 时间周期 (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)
            limit: 返回数量 (默认30，最大500)
        """
        params = {
            'symbol': symbol.upper(),
            'period': period,
            'limit': limit
        }
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        
        data = await self._make_request(
            '/globalLongShortAccountRatio',
            params,
            use_cache=True,
            cache_ttl=300
        )
        return self._parse_long_short_ratios(symbol, data or [])
    
    async def get_top_long_short_account_ratio(
        self,
        symbol: str,
        period: str = "5m",
        limit: int = 1
    ) -> List[LongShortRatio]:
        """获取大户账户多空比（前20%）"""
        params = {
            'symbol': symbol.upper(),
            'period': period,
            'limit': limit
        }
        
        data = await self._make_request(
            '/topLongShortAccountRatio',
            params,
            use_cache=True,
            cache_ttl=300
        )
        return self._parse_long_short_ratios(symbol, data or [], is_top_traders=True)
    
    async def get_top_long_short_position_ratio(
        self,
        symbol: str,
        period: str = "5m",
        limit: int = 1
    ) -> List[LongShortRatio]:
        """获取大户持仓多空比（前20%）"""
        params = {
            'symbol': symbol.upper(),
            'period': period,
            'limit': limit
        }
        
        data = await self._make_request(
            '/topLongShortPositionRatio',
            params,
            use_cache=True,
            cache_ttl=300
        )
        return self._parse_long_short_ratios(symbol, data or [], is_position=True)
    
    def _parse_long_short_ratios(
        self,
        symbol: str,
        data: List[Dict],
        is_top_traders: bool = False,
        is_position: bool = False
    ) -> List[LongShortRatio]:
        """解析多空比数据"""
        ratios = []
        for item in data:
            try:
                long_ratio = Decimal(str(item.get('longAccount', 0)))
                short_ratio = Decimal(str(item.get('shortAccount', 0)))
                
                ratio = LongShortRatio(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(item.get('timestamp', 0) / 1000),
                    long_account_ratio=long_ratio,
                    short_account_ratio=short_ratio,
                    long_short_ratio=long_ratio / short_ratio if short_ratio > 0 else Decimal('0'),
                    source=DataSourceType.BINANCE
                )
                
                # 添加元数据
                ratio.metadata = {
                    'is_top_traders': is_top_traders,
                    'is_position': is_position,
                    'long_short_ratio': item.get('longShortRatio')
                }
                
                ratios.append(ratio)
            except Exception as e:
                logger.error(f"解析多空比失败: {e}")
        return ratios
    
    # ==================== 吃单量 API ====================
    
    async def get_taker_long_short_ratio(
        self,
        symbol: str,
        period: str = "5m",
        limit: int = 1
    ) -> List[Dict[str, Any]]:
        """
        获取吃单买卖比
        
        Returns:
            [{
                'buySellRatio': str,
                'sellVol': str,
                'buyVol': str,
                'timestamp': int
            }]
        """
        params = {
            'symbol': symbol.upper(),
            'period': period,
            'limit': limit
        }
        
        return await self._make_request(
            '/takerlongshortRatio',
            params,
            use_cache=True,
            cache_ttl=300
        ) or []
    
    async def get_taker_volume(
        self,
        symbol: str,
        period: str = "5m",
        limit: int = 1
    ) -> List[Dict[str, Any]]:
        """获取吃单量统计"""
        # 与 get_taker_long_short_ratio 相同
        return await self.get_taker_long_short_ratio(symbol, period, limit)
    
    # ==================== 综合情绪分析 ====================
    
    async def get_sentiment_analysis(
        self,
        symbol: str,
        period: str = "1h"
    ) -> Dict[str, Any]:
        """获取综合情绪分析"""
        # 并行获取多个指标
        global_ratio_task = self.get_global_long_short_account_ratio(symbol, period, 30)
        top_ratio_task = self.get_top_long_short_account_ratio(symbol, period, 30)
        position_ratio_task = self.get_top_long_short_position_ratio(symbol, period, 30)
        taker_ratio_task = self.get_taker_long_short_ratio(symbol, period, 30)
        
        global_ratios, top_ratios, position_ratios, taker_ratios = await asyncio.gather(
            global_ratio_task, top_ratio_task, position_ratio_task, taker_ratio_task
        )
        
        analysis = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'period': period,
            'metrics': {}
        }
        
        # 全账户多空比分析
        if global_ratios:
            latest = global_ratios[-1]
            avg_ratio = sum(r.long_short_ratio for r in global_ratios) / len(global_ratios)
            
            analysis['metrics']['global_long_short'] = {
                'current': float(latest.long_short_ratio),
                'average': float(avg_ratio),
                'trend': 'increasing' if latest.long_short_ratio > avg_ratio else 'decreasing',
                'sentiment': 'bullish' if latest.long_short_ratio > 1.5 else 'neutral' if latest.long_short_ratio > 0.8 else 'bearish'
            }
        
        # 大户账户多空比分析
        if top_ratios:
            latest = top_ratios[-1]
            analysis['metrics']['top_traders_long_short'] = {
                'current': float(latest.long_short_ratio),
                'sentiment': 'bullish' if latest.long_short_ratio > 1.5 else 'neutral' if latest.long_short_ratio > 0.8 else 'bearish'
            }
        
        # 大户持仓多空比分析
        if position_ratios:
            latest = position_ratios[-1]
            analysis['metrics']['top_positions_long_short'] = {
                'current': float(latest.long_short_ratio),
                'sentiment': 'bullish' if latest.long_short_ratio > 1.5 else 'neutral' if latest.long_short_ratio > 0.8 else 'bearish'
            }
        
        # 吃单量分析
        if taker_ratios:
            latest = taker_ratios[-1]
            buy_sell_ratio = Decimal(latest.get('buySellRatio', '1'))
            
            analysis['metrics']['taker_volume'] = {
                'buy_sell_ratio': float(buy_sell_ratio),
                'buy_volume': latest.get('buyVol'),
                'sell_volume': latest.get('sellVol'),
                'sentiment': 'bullish' if buy_sell_ratio > 1.2 else 'neutral' if buy_sell_ratio > 0.8 else 'bearish'
            }
        
        # 综合判断
        sentiments = [
            analysis['metrics'].get('global_long_short', {}).get('sentiment', 'neutral'),
            analysis['metrics'].get('top_traders_long_short', {}).get('sentiment', 'neutral'),
            analysis['metrics'].get('top_positions_long_short', {}).get('sentiment', 'neutral'),
            analysis['metrics'].get('taker_volume', {}).get('sentiment', 'neutral')
        ]
        
        bullish_count = sentiments.count('bullish')
        bearish_count = sentiments.count('bearish')
        
        if bullish_count > bearish_count:
            analysis['overall_sentiment'] = 'bullish'
        elif bearish_count > bullish_count:
            analysis['overall_sentiment'] = 'bearish'
        else:
            analysis['overall_sentiment'] = 'neutral'
        
        analysis['sentiment_strength'] = abs(bullish_count - bearish_count) / len(sentiments)
        
        return analysis
    
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
        logger.info("Binance情绪数据缓存已清空")
