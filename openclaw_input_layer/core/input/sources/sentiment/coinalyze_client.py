"""
Coinalyze 数据客户端
资金费率、清算数据、多空比、持仓量
免费版 generous 限制
"""

import asyncio
import logging
import aiohttp
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ...input_types import (
    FundingRateData, LongShortRatio, LiquidationData,
    DataSourceType
)

logger = logging.getLogger(__name__)


@dataclass
class CoinalyzeConfig:
    """Coinalyze配置"""
    api_key: Optional[str] = None  # 免费版可能不需要
    base_url: str = "https://api.coinalyze.net/v1"
    rate_limit: float = 1.0  # generous 限制
    timeout: int = 30


class CoinalyzeClient:
    """Coinalyze数据客户端"""
    
    def __init__(self, config: Optional[CoinalyzeConfig] = None):
        self.config = config or CoinalyzeConfig()
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
        logger.info("Coinalyze客户端已启动")
    
    async def stop(self):
        """停止客户端"""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info(f"Coinalyze客户端已停止 (请求: {self._request_count}, 错误: {self._error_count})")
    
    async def _rate_limit(self):
        """速率限制"""
        if self._last_request_time:
            elapsed = (datetime.now() - self._last_request_time).total_seconds()
            if elapsed < self.config.rate_limit:
                wait_time = self.config.rate_limit - elapsed
                await asyncio.sleep(wait_time)
        self._last_request_time = datetime.now()
    
    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'OpenClaw-Trading-System/1.0'
        }
        if self.config.api_key:
            headers['Authorization'] = f'Bearer {self.config.api_key}'
        return headers
    
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
        headers = self._get_headers()
        
        try:
            async with self.session.get(url, headers=headers, params=params, timeout=self.config.timeout) as response:
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
                    logger.warning("Coinalyze API速率限制，等待后重试...")
                    await asyncio.sleep(10)
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
    
    # ==================== 资金费率 API ====================
    
    async def get_funding_rates(
        self,
        symbols: Optional[List[str]] = None
    ) -> List[FundingRateData]:
        """获取资金费率"""
        params = {}
        if symbols:
            params['symbols'] = ','.join(symbols)
        
        data = await self._make_request('/funding-rates', params, use_cache=True, cache_ttl=60)
        return self._parse_funding_rates(data or [])
    
    def _parse_funding_rates(self, data: List[Dict]) -> List[FundingRateData]:
        """解析资金费率数据"""
        rates = []
        for item in data:
            try:
                rates.append(FundingRateData(
                    symbol=item.get('symbol', ''),
                    timestamp=datetime.fromtimestamp(item.get('timestamp', 0)),
                    funding_rate=Decimal(str(item.get('fundingRate', 0))),
                    predicted_rate=Decimal(str(item.get('predictedFundingRate', 0))) if 'predictedFundingRate' in item else None,
                    source=DataSourceType.COINALYZE
                ))
            except Exception as e:
                logger.error(f"解析资金费率失败: {e}")
        return rates
    
    async def get_funding_rate_history(
        self,
        symbol: str,
        interval: str = "8h",
        limit: int = 100
    ) -> List[FundingRateData]:
        """获取资金费率历史"""
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        data = await self._make_request('/funding-rate-history', params, use_cache=True, cache_ttl=300)
        return self._parse_funding_rates(data or [])
    
    # ==================== 清算数据 API ====================
    
    async def get_liquidations(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "1h"
    ) -> List[LiquidationData]:
        """获取清算数据"""
        params = {'timeframe': timeframe}
        if symbols:
            params['symbols'] = ','.join(symbols)
        
        data = await self._make_request('/liquidations', params, use_cache=True, cache_ttl=60)
        return self._parse_liquidations(data or [])
    
    def _parse_liquidations(self, data: List[Dict]) -> List[LiquidationData]:
        """解析清算数据"""
        liquidations = []
        for item in data:
            try:
                long_liq = Decimal(str(item.get('longLiquidationUsd', 0)))
                short_liq = Decimal(str(item.get('shortLiquidationUsd', 0)))
                
                liquidations.append(LiquidationData(
                    symbol=item.get('symbol', ''),
                    timestamp=datetime.fromtimestamp(item.get('timestamp', 0)),
                    long_liquidation_usd=long_liq,
                    short_liquidation_usd=short_liq,
                    total_liquidation_usd=long_liq + short_liq,
                    source=DataSourceType.COINALYZE
                ))
            except Exception as e:
                logger.error(f"解析清算数据失败: {e}")
        return liquidations
    
    async def get_liquidation_heatmap(
        self,
        symbol: str,
        timeframe: str = "1d"
    ) -> Dict[str, Any]:
        """获取清算热力图数据"""
        params = {
            'symbol': symbol,
            'timeframe': timeframe
        }
        return await self._make_request('/liquidation-heatmap', params, use_cache=True, cache_ttl=300) or {}
    
    # ==================== 持仓量 API ====================
    
    async def get_open_interest(
        self,
        symbols: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """获取持仓量"""
        params = {}
        if symbols:
            params['symbols'] = ','.join(symbols)
        
        return await self._make_request('/open-interest', params, use_cache=True, cache_ttl=60) or []
    
    async def get_open_interest_history(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取持仓量历史"""
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        return await self._make_request('/open-interest-history', params, use_cache=True, cache_ttl=300) or []
    
    # ==================== 多空比 API ====================
    
    async def get_long_short_ratio(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "1h"
    ) -> List[LongShortRatio]:
        """获取多空比"""
        params = {'timeframe': timeframe}
        if symbols:
            params['symbols'] = ','.join(symbols)
        
        data = await self._make_request('/long-short-ratio', params, use_cache=True, cache_ttl=60)
        return self._parse_long_short_ratios(data or [])
    
    def _parse_long_short_ratios(self, data: List[Dict]) -> List[LongShortRatio]:
        """解析多空比数据"""
        ratios = []
        for item in data:
            try:
                long_ratio = Decimal(str(item.get('longAccountRatio', 0)))
                short_ratio = Decimal(str(item.get('shortAccountRatio', 0)))
                
                ratios.append(LongShortRatio(
                    symbol=item.get('symbol', ''),
                    timestamp=datetime.fromtimestamp(item.get('timestamp', 0)),
                    long_account_ratio=long_ratio,
                    short_account_ratio=short_ratio,
                    long_short_ratio=long_ratio / short_ratio if short_ratio > 0 else Decimal('0'),
                    source=DataSourceType.COINALYZE
                ))
            except Exception as e:
                logger.error(f"解析多空比失败: {e}")
        return ratios
    
    # ==================== 综合情绪指标 ====================
    
    async def get_composite_sentiment(
        self,
        symbol: str
    ) -> Dict[str, Any]:
        """获取综合情绪指标"""
        # 并行获取多个指标
        funding_task = self.get_funding_rates([symbol])
        liquidation_task = self.get_liquidations([symbol], timeframe="1h")
        ratio_task = self.get_long_short_ratio([symbol], timeframe="1h")
        oi_task = self.get_open_interest([symbol])
        
        funding_rates, liquidations, ratios, open_interests = await asyncio.gather(
            funding_task, liquidation_task, ratio_task, oi_task
        )
        
        # 计算综合情绪分数
        sentiment_components = {}
        
        # 资金费率成分 (0-100)
        if funding_rates:
            fr = funding_rates[0].funding_rate
            # 正费率 = 多头付空头 = 贪婪，负费率 = 恐惧
            sentiment_components['funding'] = 50 + float(fr) * 5000
        
        # 清算成分 (0-100)
        if liquidations:
            liq = liquidations[0]
            total_liq = float(liq.total_liquidation_usd)
            # 清算量越大越恐慌
            sentiment_components['liquidation'] = max(0, 100 - total_liq / 10000)
        
        # 多空比成分 (0-100)
        if ratios:
            ratio = ratios[0].long_short_ratio
            # 多空比越高越贪婪
            sentiment_components['long_short'] = min(100, float(ratio) * 33.33)
        
        # 持仓量成分 (0-100) - 需要历史数据对比
        if open_interests:
            oi = open_interests[0]
            # 简单处理：持仓量增长 = 贪婪
            sentiment_components['open_interest'] = 50  # 默认值
        
        # 加权平均
        weights = {
            'funding': 0.35,
            'liquidation': 0.25,
            'long_short': 0.25,
            'open_interest': 0.15
        }
        
        composite_score = sum(
            sentiment_components.get(k, 50) * weights.get(k, 0)
            for k in weights.keys()
        )
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'composite_score': composite_score,
            'components': sentiment_components,
            'interpretation': self._interpret_composite_score(composite_score),
            'raw_data': {
                'funding_rates': funding_rates,
                'liquidations': liquidations,
                'ratios': ratios,
                'open_interests': open_interests
            }
        }
    
    def _interpret_composite_score(self, score: float) -> str:
        """解读综合情绪分数"""
        if score <= 20:
            return "极度看空 - 强烈买入信号"
        elif score <= 40:
            return "看空 - 考虑买入"
        elif score <= 60:
            return "中性 - 观望"
        elif score <= 80:
            return "看多 - 注意回调风险"
        else:
            return "极度看多 - 考虑获利了结"
    
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
        logger.info("Coinalyze缓存已清空")
