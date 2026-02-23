"""
数据聚合器模块
整合多个数据源，提供统一的数据接口
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field

from .input_types import (
    MarketData, PriceData, OrderBookData, TradeData,
    ExchangeFlowData, HolderBehaviorData, TVLData,
    FearGreedIndex, FundingRateData, LongShortRatio, LiquidationData,
    NewsData, SocialSentimentData,
    DataSourceType, InputContext, InputResult,
    DataQualityMetrics
)
from .data_validator import DataValidator, DataCache

# 导入所有数据源客户端
from .sources.price import BinanceClient, CoinGeckoClient
from .sources.onchain import DuneClient, DeFiLlamaClient, ArkhamClient
from .sources.sentiment import AlternativeMeClient, CoinalyzeClient, BinanceSentimentClient
from .sources.news import CryptoPanicClient, NewsDataClient, RedditClient

logger = logging.getLogger(__name__)


@dataclass
class DataSourceStatus:
    """数据源状态"""
    source: DataSourceType
    is_active: bool
    last_update: Optional[datetime]
    error_count: int
    success_count: int
    latency_ms: float


class DataAggregator:
    """数据聚合器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 初始化验证器
        self.validator = DataValidator(config.get('validation', {}))
        self.cache = DataCache(default_ttl=config.get('cache_ttl', 60))
        
        # 初始化数据源客户端
        self.clients: Dict[DataSourceType, Any] = {}
        self._init_clients()
        
        # 数据源状态
        self.source_status: Dict[DataSourceType, DataSourceStatus] = {}
        
        # 运行状态
        self._running = False
        self._aggregation_task: Optional[asyncio.Task] = None
        
    def _init_clients(self):
        """初始化数据源客户端"""
        # 价格数据源
        self.clients[DataSourceType.BINANCE] = BinanceClient()
        self.clients[DataSourceType.COINGECKO] = CoinGeckoClient()
        
        # 链上数据源
        self.clients[DataSourceType.DUNE] = DuneClient()
        self.clients[DataSourceType.DEFILLAMA] = DeFiLlamaClient()
        self.clients[DataSourceType.ARKHAM] = ArkhamClient()
        
        # 情绪数据源
        self.clients[DataSourceType.ALTERNATIVE_ME] = AlternativeMeClient()
        self.clients[DataSourceType.COINALYZE] = CoinalyzeClient()
        
        # 新闻数据源
        self.clients[DataSourceType.CRYPTOPANIC] = CryptoPanicClient()
        self.clients[DataSourceType.NEWSDATA] = NewsDataClient()
        self.clients[DataSourceType.REDDIT] = RedditClient()
    
    async def start(self):
        """启动聚合器"""
        self._running = True
        
        # 启动所有客户端
        for source, client in self.clients.items():
            try:
                if hasattr(client, 'start'):
                    await client.start()
                self.source_status[source] = DataSourceStatus(
                    source=source,
                    is_active=True,
                    last_update=None,
                    error_count=0,
                    success_count=0,
                    latency_ms=0
                )
            except Exception as e:
                logger.error(f"启动数据源 {source.value} 失败: {e}")
                self.source_status[source] = DataSourceStatus(
                    source=source,
                    is_active=False,
                    last_update=None,
                    error_count=1,
                    success_count=0,
                    latency_ms=0
                )
        
        # 启动缓存
        await self.cache.start()
        
        logger.info("数据聚合器已启动")
    
    async def stop(self):
        """停止聚合器"""
        self._running = False
        
        # 停止缓存
        await self.cache.stop()
        
        # 停止所有客户端
        for source, client in self.clients.items():
            try:
                if hasattr(client, 'stop'):
                    await client.stop()
            except Exception as e:
                logger.error(f"停止数据源 {source.value} 失败: {e}")
        
        logger.info("数据聚合器已停止")
    
    async def get_market_data(
        self,
        symbol: str,
        data_types: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> InputResult:
        """
        获取综合市场数据
        
        Args:
            symbol: 交易对符号，如 "BTCUSDT"
            data_types: 数据类型列表 ['price', 'onchain', 'sentiment', 'news']
            use_cache: 是否使用缓存
        """
        start_time = datetime.now()
        
        if data_types is None:
            data_types = ['price', 'onchain', 'sentiment', 'news']
        
        # 检查缓存
        cache_key = f"market_data:{symbol}:{','.join(sorted(data_types))}"
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached:
                return InputResult(
                    success=True,
                    message="从缓存获取数据",
                    market_data=cached,
                    processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000
                )
        
        market_data = MarketData(
            symbol=symbol,
            timestamp=datetime.now()
        )
        errors = []
        
        # 并行获取各类数据
        tasks = []
        
        if 'price' in data_types:
            tasks.append(self._fetch_price_data(symbol, market_data))
        
        if 'onchain' in data_types:
            tasks.append(self._fetch_onchain_data(symbol, market_data))
        
        if 'sentiment' in data_types:
            tasks.append(self._fetch_sentiment_data(symbol, market_data))
        
        if 'news' in data_types:
            tasks.append(self._fetch_news_data(symbol, market_data))
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                errors.append(str(result))
                logger.error(f"获取数据失败: {result}")
        
        # 验证数据
        if market_data.price_data:
            validation_result = self.validator.validate(market_data.price_data)
            if not validation_result['valid']:
                errors.extend(validation_result['critical_errors'])
        
        # 更新缓存
        if use_cache:
            self.cache.set(cache_key, market_data, ttl=60)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return InputResult(
            success=len(errors) == 0,
            message="数据获取成功" if len(errors) == 0 else "部分数据获取失败",
            market_data=market_data,
            errors=errors,
            processing_time_ms=processing_time
        )
    
    async def _fetch_price_data(self, symbol: str, market_data: MarketData):
        """获取价格数据"""
        try:
            # 优先使用Binance
            binance_client = self.clients.get(DataSourceType.BINANCE)
            if binance_client and self.source_status[DataSourceType.BINANCE].is_active:
                # 获取K线数据
                klines = await binance_client.get_klines(symbol, interval="1m", limit=1)
                if klines:
                    market_data.price_data = klines[0]
                
                # 获取订单簿
                orderbook = await binance_client.get_orderbook(symbol, limit=20)
                if orderbook:
                    market_data.orderbook_data = orderbook
                
                # 获取成交数据
                trades = await binance_client.get_recent_trades(symbol, limit=50)
                market_data.recent_trades = trades
                
                self._update_source_status(DataSourceType.BINANCE, True)
        
        except Exception as e:
            logger.error(f"获取Binance价格数据失败: {e}")
            self._update_source_status(DataSourceType.BINANCE, False)
            
            # 回退到CoinGecko
            try:
                coingecko_client = self.clients.get(DataSourceType.COINGECKO)
                if coingecko_client:
                    price = await coingecko_client.get_price_by_symbol(symbol)
                    if price:
                        # 创建简化的PriceData
                        market_data.price_data = PriceData(
                            symbol=symbol,
                            timestamp=datetime.now(),
                            open_price=price,
                            high_price=price,
                            low_price=price,
                            close_price=price,
                            volume=Decimal('0'),
                            quote_volume=Decimal('0'),
                            source=DataSourceType.COINGECKO
                        )
            except Exception as e2:
                logger.error(f"获取CoinGecko价格数据失败: {e2}")
    
    async def _fetch_onchain_data(self, symbol: str, market_data: MarketData):
        """获取链上数据"""
        # 获取交易所流向
        try:
            dune_client = self.clients.get(DataSourceType.DUNE)
            if dune_client and self.source_status[DataSourceType.DUNE].is_active:
                flows = await dune_client.get_exchange_flows(symbol.replace('USDT', '').replace('USD', ''))
                market_data.exchange_flows = flows
                self._update_source_status(DataSourceType.DUNE, True)
        except Exception as e:
            logger.error(f"获取Dune数据失败: {e}")
            self._update_source_status(DataSourceType.DUNE, False)
        
        # 获取TVL数据
        try:
            defillama_client = self.clients.get(DataSourceType.DEFILLAMA)
            if defillama_client:
                # 获取总TVL
                total_tvl = await defillama_client.get_total_tvl()
                market_data.metadata['total_tvl'] = total_tvl
        except Exception as e:
            logger.error(f"获取DeFiLlama数据失败: {e}")
    
    async def _fetch_sentiment_data(self, symbol: str, market_data: MarketData):
        """获取情绪数据"""
        # 获取恐惧贪婪指数
        try:
            altme_client = self.clients.get(DataSourceType.ALTERNATIVE_ME)
            if altme_client and self.source_status[DataSourceType.ALTERNATIVE_ME].is_active:
                fear_greed = await altme_client.get_current_index()
                if fear_greed:
                    market_data.fear_greed = fear_greed
                self._update_source_status(DataSourceType.ALTERNATIVE_ME, True)
        except Exception as e:
            logger.error(f"获取恐惧贪婪指数失败: {e}")
            self._update_source_status(DataSourceType.ALTERNATIVE_ME, False)
        
        # 获取资金费率和多空比
        try:
            coinalyze_client = self.clients.get(DataSourceType.COINALYZE)
            if coinalyze_client and self.source_status[DataSourceType.COINALYZE].is_active:
                # 获取资金费率
                funding_rates = await coinalyze_client.get_funding_rates([symbol])
                if funding_rates:
                    market_data.funding_rate = funding_rates[0]
                
                # 获取多空比
                long_short_ratios = await coinalyze_client.get_long_short_ratio([symbol])
                if long_short_ratios:
                    market_data.long_short_ratio = long_short_ratios[0]
                
                # 获取清算数据
                liquidations = await coinalyze_client.get_liquidations([symbol])
                if liquidations:
                    market_data.liquidation_data = liquidations[0]
                
                self._update_source_status(DataSourceType.COINALYZE, True)
        except Exception as e:
            logger.error(f"获取Coinalyze数据失败: {e}")
            self._update_source_status(DataSourceType.COINALYZE, False)
    
    async def _fetch_news_data(self, symbol: str, market_data: MarketData):
        """获取新闻数据"""
        # 获取CryptoPanic新闻
        try:
            cryptopanic_client = self.clients.get(DataSourceType.CRYPTOPANIC)
            if cryptopanic_client and self.source_status[DataSourceType.CRYPTOPANIC].is_active:
                base_symbol = symbol.replace('USDT', '').replace('USD', '')
                news = await cryptopanic_client.get_posts(
                    currencies=[base_symbol],
                    limit=10
                )
                market_data.recent_news = news
                self._update_source_status(DataSourceType.CRYPTOPANIC, True)
        except Exception as e:
            logger.error(f"获取CryptoPanic新闻失败: {e}")
            self._update_source_status(DataSourceType.CRYPTOPANIC, False)
        
        # 获取Reddit情绪
        try:
            reddit_client = self.clients.get(DataSourceType.REDDIT)
            if reddit_client and self.source_status[DataSourceType.REDDIT].is_active:
                sentiment = await reddit_client.analyze_sentiment(limit=50)
                market_data.social_sentiment = [sentiment]
                self._update_source_status(DataSourceType.REDDIT, True)
        except Exception as e:
            logger.error(f"获取Reddit情绪失败: {e}")
            self._update_source_status(DataSourceType.REDDIT, False)
    
    def _update_source_status(self, source: DataSourceType, success: bool):
        """更新数据源状态"""
        if source not in self.source_status:
            return
        
        status = self.source_status[source]
        status.last_update = datetime.now()
        
        if success:
            status.success_count += 1
        else:
            status.error_count += 1
    
    def get_source_status(self) -> Dict[str, Any]:
        """获取所有数据源状态"""
        return {
            source.value: {
                'is_active': status.is_active,
                'last_update': status.last_update.isoformat() if status.last_update else None,
                'error_count': status.error_count,
                'success_count': status.success_count,
                'total_requests': status.error_count + status.success_count,
                'success_rate': status.success_count / max(status.error_count + status.success_count, 1)
            }
            for source, status in self.source_status.items()
        }
    
    async def stream_price_data(
        self,
        symbol: str,
        interval: str = "1m"
    ):
        """流式获取价格数据（使用Binance WebSocket）"""
        binance_client = self.clients.get(DataSourceType.BINANCE)
        if not binance_client:
            raise RuntimeError("Binance客户端未初始化")
        
        async for price_data in binance_client.stream_klines(symbol, interval):
            yield price_data
    
    def get_client(self, source: DataSourceType) -> Optional[Any]:
        """获取指定数据源的客户端"""
        return self.clients.get(source)
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        for client in self.clients.values():
            if hasattr(client, 'clear_cache'):
                client.clear_cache()
        logger.info("数据聚合器缓存已清空")
