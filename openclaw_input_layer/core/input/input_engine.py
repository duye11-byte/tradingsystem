"""
输入引擎主模块
第1层：多源数据融合与实时数据流处理

这是整个5层交易系统的数据入口，负责：
1. 从多个免费数据源获取实时数据
2. 数据验证和质量控制
3. 数据缓存和容错处理
4. 为第2层（推理层）提供标准化的市场数据
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator, Callable
from dataclasses import dataclass, field
from enum import Enum

from .input_types import (
    MarketData, PriceData, OrderBookData, TradeData,
    InputContext, InputResult, DataSourceType, DataFrequency
)
from .data_validator import DataValidator, DataCache
from .data_aggregator import DataAggregator

logger = logging.getLogger(__name__)


class InputMode(Enum):
    """输入模式"""
    REALTIME = "realtime"      # 实时模式（WebSocket）
    POLLING = "polling"        # 轮询模式（REST API）
    HYBRID = "hybrid"          # 混合模式
    BACKTEST = "backtest"      # 回测模式


@dataclass
class InputEngineConfig:
    """输入引擎配置"""
    mode: InputMode = InputMode.HYBRID
    default_symbols: List[str] = field(default_factory=lambda: ['BTCUSDT', 'ETHUSDT'])
    default_data_types: List[str] = field(default_factory=lambda: ['price', 'sentiment'])
    cache_enabled: bool = True
    cache_ttl: int = 60
    validation_enabled: bool = True
    fallback_enabled: bool = True
    websocket_auto_reconnect: bool = True
    max_concurrent_requests: int = 10
    request_timeout: int = 30


class InputEngine:
    """
    输入引擎主类
    
    提供统一的接口获取市场数据，整合多个免费数据源：
    - 价格数据：Binance, CoinGecko
    - 链上数据：Dune, DeFiLlama, Arkham
    - 情绪数据：Alternative.me, Coinalyze, Binance
    - 新闻数据：CryptoPanic, NewsData, Reddit
    """
    
    def __init__(self, config: Optional[InputEngineConfig] = None):
        self.config = config or InputEngineConfig()
        
        # 数据聚合器
        self.aggregator = DataAggregator({
            'validation': {'enabled': self.config.validation_enabled},
            'cache_ttl': self.config.cache_ttl
        })
        
        # 验证器
        self.validator = DataValidator()
        
        # 缓存
        self.cache = DataCache(default_ttl=self.config.cache_ttl)
        
        # 运行状态
        self._running = False
        self._subscribers: List[Callable] = []
        self._stream_tasks: List[asyncio.Task] = []
        
        # 统计
        self._stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
    
    async def start(self):
        """启动输入引擎"""
        if self._running:
            return
        
        self._running = True
        
        # 启动数据聚合器
        await self.aggregator.start()
        
        # 启动缓存
        await self.cache.start()
        
        logger.info(f"输入引擎已启动 (模式: {self.config.mode.value})")
    
    async def stop(self):
        """停止输入引擎"""
        if not self._running:
            return
        
        self._running = False
        
        # 取消所有流任务
        for task in self._stream_tasks:
            task.cancel()
        
        if self._stream_tasks:
            await asyncio.gather(*self._stream_tasks, return_exceptions=True)
        
        # 停止缓存
        await self.cache.stop()
        
        # 停止数据聚合器
        await self.aggregator.stop()
        
        logger.info("输入引擎已停止")
    
    # ==================== 核心数据获取接口 ====================
    
    async def get_market_data(
        self,
        symbol: str,
        data_types: Optional[List[str]] = None,
        use_cache: Optional[bool] = None
    ) -> InputResult:
        """
        获取综合市场数据
        
        Args:
            symbol: 交易对符号，如 "BTCUSDT"
            data_types: 数据类型 ['price', 'onchain', 'sentiment', 'news']
            use_cache: 是否使用缓存
        
        Returns:
            InputResult: 包含市场数据和元数据的结果
        """
        if not self._running:
            return InputResult(
                success=False,
                message="输入引擎未启动"
            )
        
        if use_cache is None:
            use_cache = self.config.cache_enabled
        
        if data_types is None:
            data_types = self.config.default_data_types
        
        self._stats['total_requests'] += 1
        
        try:
            result = await self.aggregator.get_market_data(
                symbol=symbol,
                data_types=data_types,
                use_cache=use_cache
            )
            
            if result.success:
                self._stats['successful_requests'] += 1
            else:
                self._stats['failed_requests'] += 1
            
            return result
        
        except Exception as e:
            logger.error(f"获取市场数据失败: {e}")
            self._stats['failed_requests'] += 1
            return InputResult(
                success=False,
                message=f"获取市场数据失败: {str(e)}"
            )
    
    async def get_price_data(self, symbol: str) -> Optional[PriceData]:
        """获取价格数据"""
        result = await self.get_market_data(symbol, data_types=['price'])
        if result.success and result.market_data:
            return result.market_data.price_data
        return None
    
    async def get_orderbook(self, symbol: str) -> Optional[OrderBookData]:
        """获取订单簿数据"""
        result = await self.get_market_data(symbol, data_types=['price'])
        if result.success and result.market_data:
            return result.market_data.orderbook_data
        return None
    
    async def get_recent_trades(self, symbol: str, limit: int = 50) -> List[TradeData]:
        """获取近期成交数据"""
        result = await self.get_market_data(symbol, data_types=['price'])
        if result.success and result.market_data:
            return result.market_data.recent_trades[:limit]
        return []
    
    # ==================== 批量数据获取 ====================
    
    async def get_multiple_market_data(
        self,
        symbols: List[str],
        data_types: Optional[List[str]] = None
    ) -> Dict[str, InputResult]:
        """批量获取多个交易对的市场数据"""
        tasks = [
            self.get_market_data(symbol, data_types)
            for symbol in symbols
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            symbol: result if not isinstance(result, Exception) else InputResult(
                success=False,
                message=f"获取数据失败: {str(result)}"
            )
            for symbol, result in zip(symbols, results)
        }
    
    # ==================== 实时数据流 ====================
    
    async def stream_prices(
        self,
        symbol: str,
        interval: str = "1m"
    ) -> AsyncGenerator[PriceData, None]:
        """
        实时价格数据流（WebSocket）
        
        Args:
            symbol: 交易对符号
            interval: K线间隔
        """
        if not self._running:
            raise RuntimeError("输入引擎未启动")
        
        if self.config.mode == InputMode.POLLING:
            # 轮询模式
            while self._running:
                price_data = await self.get_price_data(symbol)
                if price_data:
                    yield price_data
                await asyncio.sleep(60)  # 每分钟轮询
        else:
            # WebSocket模式
            async for price_data in self.aggregator.stream_price_data(symbol, interval):
                if not self._running:
                    break
                yield price_data
    
    async def stream_orderbook(
        self,
        symbol: str,
        level: int = 20
    ) -> AsyncGenerator[OrderBookData, None]:
        """实时订单簿数据流"""
        binance_client = self.aggregator.get_client(DataSourceType.BINANCE)
        if not binance_client:
            raise RuntimeError("Binance客户端不可用")
        
        async for orderbook in binance_client.stream_orderbook(symbol, level):
            if not self._running:
                break
            yield orderbook
    
    async def stream_trades(
        self,
        symbol: str
    ) -> AsyncGenerator[TradeData, None]:
        """实时成交数据流"""
        binance_client = self.aggregator.get_client(DataSourceType.BINANCE)
        if not binance_client:
            raise RuntimeError("Binance客户端不可用")
        
        async for trade in binance_client.stream_trades(symbol):
            if not self._running:
                break
            yield trade
    
    def subscribe(self, callback: Callable[[MarketData], None]):
        """订阅市场数据更新"""
        self._subscribers.append(callback)
        logger.info(f"新订阅者加入，当前订阅数: {len(self._subscribers)}")
    
    def unsubscribe(self, callback: Callable[[MarketData], None]):
        """取消订阅"""
        if callback in self._subscribers:
            self._subscribers.remove(callback)
            logger.info(f"订阅者离开，当前订阅数: {len(self._subscribers)}")
    
    async def _notify_subscribers(self, market_data: MarketData):
        """通知所有订阅者"""
        for callback in self._subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(market_data)
                else:
                    callback(market_data)
            except Exception as e:
                logger.error(f"通知订阅者失败: {e}")
    
    # ==================== 数据验证 ====================
    
    def validate_data(self, data: Any) -> Dict[str, Any]:
        """验证数据"""
        return self.validator.validate(data)
    
    def detect_anomaly(self, data: Any, method: str = "zscore") -> Dict[str, Any]:
        """检测异常"""
        return self.validator.detect_anomaly(data, method)
    
    # ==================== 缓存管理 ====================
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        self.aggregator.clear_cache()
        logger.info("输入引擎缓存已清空")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return self.cache.get_stats()
    
    # ==================== 数据源管理 ====================
    
    def get_data_source_status(self) -> Dict[str, Any]:
        """获取数据源状态"""
        return self.aggregator.get_source_status()
    
    def enable_data_source(self, source: DataSourceType):
        """启用数据源"""
        if source in self.aggregator.source_status:
            self.aggregator.source_status[source].is_active = True
            logger.info(f"数据源 {source.value} 已启用")
    
    def disable_data_source(self, source: DataSourceType):
        """禁用数据源"""
        if source in self.aggregator.source_status:
            self.aggregator.source_status[source].is_active = False
            logger.info(f"数据源 {source.value} 已禁用")
    
    # ==================== 统计信息 ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        return {
            **self._stats,
            'success_rate': self._stats['successful_requests'] / max(self._stats['total_requests'], 1),
            'cache_hit_rate': self._stats['cache_hits'] / max(self._stats['cache_hits'] + self._stats['cache_misses'], 1),
            'data_sources': self.get_data_source_status(),
            'config': {
                'mode': self.config.mode.value,
                'cache_enabled': self.config.cache_enabled,
                'validation_enabled': self.config.validation_enabled
            }
        }
    
    def reset_stats(self):
        """重置统计"""
        self._stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        logger.info("输入引擎统计已重置")
    
    # ==================== 健康检查 ====================
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'engine_running': self._running,
            'data_sources': {},
            'issues': []
        }
        
        # 检查数据源
        for source, source_status in self.aggregator.source_status.items():
            if not source_status.is_active:
                status['issues'].append(f"数据源 {source.value} 未激活")
            
            status['data_sources'][source.value] = {
                'active': source_status.is_active,
                'success_rate': source_status.success_count / max(source_status.error_count + source_status.success_count, 1)
            }
        
        # 判断整体状态
        active_sources = sum(1 for s in self.aggregator.source_status.values() if s.is_active)
        if active_sources == 0:
            status['status'] = 'critical'
            status['issues'].append("没有可用的数据源")
        elif active_sources < 3:
            status['status'] = 'degraded'
        
        return status
    
    # ==================== 便捷方法 ====================
    
    async def quick_price_check(self, symbols: List[str]) -> Dict[str, Optional[Decimal]]:
        """快速价格检查"""
        results = await self.get_multiple_market_data(symbols, data_types=['price'])
        
        return {
            symbol: result.market_data.price_data.close_price
            if result.success and result.market_data and result.market_data.price_data
            else None
            for symbol, result in results.items()
        }
    
    async def get_composite_sentiment(self, symbol: str) -> Dict[str, Any]:
        """获取综合情绪指标"""
        result = await self.get_market_data(symbol, data_types=['sentiment', 'price'])
        
        if not result.success or not result.market_data:
            return {'error': '无法获取情绪数据'}
        
        market_data = result.market_data
        
        sentiment_components = {}
        
        # 恐惧贪婪指数
        if market_data.fear_greed:
            sentiment_components['fear_greed'] = {
                'value': market_data.fear_greed.value,
                'classification': market_data.fear_greed.classification
            }
        
        # 资金费率
        if market_data.funding_rate:
            sentiment_components['funding_rate'] = {
                'rate': float(market_data.funding_rate.funding_rate),
                'interpretation': 'bullish' if market_data.funding_rate.funding_rate < 0 else 'bearish'
            }
        
        # 多空比
        if market_data.long_short_ratio:
            sentiment_components['long_short_ratio'] = {
                'ratio': float(market_data.long_short_ratio.long_short_ratio),
                'interpretation': 'bullish' if market_data.long_short_ratio.long_short_ratio > 1.5 else 'neutral' if market_data.long_short_ratio.long_short_ratio > 0.8 else 'bearish'
            }
        
        # 综合分数
        composite_score = market_data.composite_sentiment
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'composite_score': composite_score,
            'interpretation': self._interpret_composite_sentiment(composite_score),
            'components': sentiment_components
        }
    
    def _interpret_composite_sentiment(self, score: float) -> str:
        """解读综合情绪分数"""
        if score <= 20:
            return "极度恐慌 - 强烈买入信号"
        elif score <= 40:
            return "恐慌 - 考虑买入"
        elif score <= 60:
            return "中性 - 观望"
        elif score <= 80:
            return "贪婪 - 注意风险"
        else:
            return "极度贪婪 - 考虑获利了结"
