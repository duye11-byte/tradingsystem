"""
输入层类型定义
定义所有数据结构、枚举类型和配置类
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable, Union
import numpy as np


class DataSourceType(Enum):
    """数据源类型"""
    BINANCE = "binance"
    COINGECKO = "coingecko"
    COINBASE = "coinbase"
    DUNE = "dune"
    DEFILLAMA = "defillama"
    ARKHAM = "arkham"
    ALTERNATIVE_ME = "alternative_me"
    COINALYZE = "coinalyze"
    CRYPTOPANIC = "cryptopanic"
    NEWSDATA = "newsdata"
    REDDIT = "reddit"


class DataFrequency(Enum):
    """数据频率"""
    TICK = "tick"
    SECOND_1 = "1s"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"


@dataclass
class PriceData:
    """价格数据"""
    symbol: str
    timestamp: datetime
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: Decimal
    quote_volume: Decimal
    trades_count: int = 0
    source: DataSourceType = DataSourceType.BINANCE
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def price_change(self) -> Decimal:
        """价格变化"""
        return self.close_price - self.open_price
    
    @property
    def price_change_pct(self) -> Decimal:
        """价格变化百分比"""
        if self.open_price == 0:
            return Decimal('0')
        return (self.price_change / self.open_price) * 100


@dataclass
class OrderBookLevel:
    """订单簿层级"""
    price: Decimal
    quantity: Decimal
    order_count: int = 0


@dataclass
class OrderBookData:
    """订单簿数据"""
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    source: DataSourceType = DataSourceType.BINANCE
    
    @property
    def best_bid(self) -> Optional[OrderBookLevel]:
        """最优买价"""
        return self.bids[0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[OrderBookLevel]:
        """最优卖价"""
        return self.asks[0] if self.asks else None
    
    @property
    def spread(self) -> Decimal:
        """买卖价差"""
        if self.best_bid and self.best_ask:
            return self.best_ask.price - self.best_bid.price
        return Decimal('0')
    
    @property
    def spread_pct(self) -> Decimal:
        """价差百分比"""
        mid_price = self.mid_price
        if mid_price == 0:
            return Decimal('0')
        return (self.spread / mid_price) * 100
    
    @property
    def mid_price(self) -> Decimal:
        """中间价"""
        if self.best_bid and self.best_ask:
            return (self.best_bid.price + self.best_ask.price) / 2
        return Decimal('0')
    
    @property
    def bid_volume(self) -> Decimal:
        """买方总量"""
        return sum(level.quantity for level in self.bids)
    
    @property
    def ask_volume(self) -> Decimal:
        """卖方总量"""
        return sum(level.quantity for level in self.asks)
    
    @property
    def imbalance(self) -> Decimal:
        """订单簿不平衡度 (-1 到 1，正值表示买方占优)"""
        total_volume = self.bid_volume + self.ask_volume
        if total_volume == 0:
            return Decimal('0')
        return (self.bid_volume - self.ask_volume) / total_volume


@dataclass
class TradeData:
    """成交数据"""
    symbol: str
    timestamp: datetime
    price: Decimal
    quantity: Decimal
    is_buyer_maker: bool  # True: 买方是maker (卖压), False: 卖方是maker (买压)
    trade_id: str = ""
    source: DataSourceType = DataSourceType.BINANCE
    
    @property
    def side(self) -> str:
        """成交方向"""
        return "sell" if self.is_buyer_maker else "buy"


@dataclass
class OnChainData:
    """链上数据"""
    symbol: str
    timestamp: datetime
    metric_name: str
    value: Decimal
    blockchain: str = ""
    source: DataSourceType = DataSourceType.DUNE
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExchangeFlowData:
    """交易所资金流向数据"""
    symbol: str
    timestamp: datetime
    exchange: str
    inflow: Decimal  # 流入
    outflow: Decimal  # 流出
    netflow: Decimal  # 净流入
    source: DataSourceType = DataSourceType.DUNE
    
    @property
    def inflow_outflow_ratio(self) -> Decimal:
        """流入流出比"""
        if self.outflow == 0:
            return Decimal('0')
        return self.inflow / self.outflow


@dataclass
class HolderBehaviorData:
    """持有者行为数据"""
    symbol: str
    timestamp: datetime
    long_term_holders: Decimal  # 长期持有者数量/比例
    short_term_holders: Decimal  # 短期持有者数量/比例
    new_addresses: int  # 新增地址
    active_addresses: int  # 活跃地址
    source: DataSourceType = DataSourceType.DUNE


@dataclass
class TVLData:
    """TVL数据"""
    protocol: str
    timestamp: datetime
    tvl_usd: Decimal
    tvl_change_24h: Decimal
    tvl_change_7d: Decimal
    chain: str = ""
    source: DataSourceType = DataSourceType.DEFILLAMA


@dataclass
class SentimentData:
    """情绪数据基类"""
    timestamp: datetime
    metric_name: str
    value: float  # 通常 0-100
    source: DataSourceType = DataSourceType.ALTERNATIVE_ME
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FearGreedIndex:
    """恐惧贪婪指数"""
    timestamp: datetime
    value: int  # 0-100
    classification: str  # "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"
    source: DataSourceType = DataSourceType.ALTERNATIVE_ME


@dataclass
class FundingRateData:
    """资金费率数据"""
    symbol: str
    timestamp: datetime
    funding_rate: Decimal  # 正数表示多头付空头，负数相反
    predicted_rate: Optional[Decimal] = None
    source: DataSourceType = DataSourceType.COINALYZE


@dataclass
class LongShortRatio:
    """多空比数据"""
    symbol: str
    timestamp: datetime
    long_account_ratio: Decimal
    short_account_ratio: Decimal
    long_short_ratio: Decimal  # 多仓/空仓
    source: DataSourceType = DataSourceType.BINANCE


@dataclass
class LiquidationData:
    """清算数据"""
    symbol: str
    timestamp: datetime
    long_liquidation_usd: Decimal
    short_liquidation_usd: Decimal
    total_liquidation_usd: Decimal
    source: DataSourceType = DataSourceType.COINALYZE


@dataclass
class NewsData:
    """新闻数据"""
    title: str
    content: str
    timestamp: datetime
    source: str
    url: str = ""
    sentiment_score: Optional[float] = None  # -1 到 1
    keywords: List[str] = field(default_factory=list)
    related_symbols: List[str] = field(default_factory=list)
    source_type: DataSourceType = DataSourceType.CRYPTOPANIC


@dataclass
class SocialSentimentData:
    """社交情绪数据"""
    platform: str  # reddit, twitter, etc.
    timestamp: datetime
    mention_count: int
    sentiment_score: float  # -1 到 1
    trending_keywords: List[str] = field(default_factory=list)
    source: DataSourceType = DataSourceType.REDDIT


@dataclass
class MarketData:
    """综合市场数据"""
    """整合所有数据源的综合市场数据结构"""
    symbol: str
    timestamp: datetime
    
    # 价格数据
    price_data: Optional[PriceData] = None
    orderbook_data: Optional[OrderBookData] = None
    recent_trades: List[TradeData] = field(default_factory=list)
    
    # 链上数据
    exchange_flows: List[ExchangeFlowData] = field(default_factory=list)
    holder_behavior: Optional[HolderBehaviorData] = None
    tvl_data: List[TVLData] = field(default_factory=list)
    
    # 情绪数据
    fear_greed: Optional[FearGreedIndex] = None
    funding_rate: Optional[FundingRateData] = None
    long_short_ratio: Optional[LongShortRatio] = None
    liquidation_data: Optional[LiquidationData] = None
    
    # 新闻数据
    recent_news: List[NewsData] = field(default_factory=list)
    social_sentiment: List[SocialSentimentData] = field(default_factory=list)
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def composite_sentiment(self) -> float:
        """计算综合情绪指标 (0-100)"""
        scores = []
        weights = []
        
        if self.fear_greed:
            scores.append(self.fear_greed.value)
            weights.append(0.4)
        
        if self.funding_rate and self.funding_rate.funding_rate != 0:
            # 将资金费率转换为 0-100 分数
            rate = float(self.funding_rate.funding_rate)
            normalized = 50 + (rate * 5000)  # 放大并平移
            normalized = max(0, min(100, normalized))
            scores.append(normalized)
            weights.append(0.3)
        
        if self.liquidation_data and self.liquidation_data.total_liquidation_usd > 0:
            # 清算量越大，情绪越恐慌
            liq = float(self.liquidation_data.total_liquidation_usd)
            normalized = max(0, 100 - liq / 10000)
            scores.append(normalized)
            weights.append(0.2)
        
        if self.long_short_ratio:
            ratio = float(self.long_short_ratio.long_short_ratio)
            # 多空比越高，情绪越贪婪
            normalized = min(100, ratio * 33.33)
            scores.append(normalized)
            weights.append(0.1)
        
        if not scores:
            return 50.0
        
        # 加权平均
        total_weight = sum(weights)
        return sum(s * w for s, w in zip(scores, weights)) / total_weight


@dataclass
class DataSourceConfig:
    """数据源配置"""
    source_type: DataSourceType
    enabled: bool = True
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    base_url: str = ""
    websocket_url: str = ""
    rate_limit: float = 1.0  # 秒
    timeout: int = 30
    retry_count: int = 3
    cache_ttl: int = 60  # 秒
    symbols: List[str] = field(default_factory=list)
    endpoints: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InputContext:
    """输入上下文"""
    symbol: str
    timestamp: datetime
    data_types: List[str] = field(default_factory=list)  # price, onchain, sentiment, news
    frequency: DataFrequency = DataFrequency.MINUTE_1
    lookback_period: int = 100  # 回溯数据点数量
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InputResult:
    """输入层结果"""
    success: bool
    message: str
    market_data: Optional[MarketData] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0


@dataclass
class ValidationRule:
    """数据验证规则"""
    name: str
    check_func: Callable[[Any], bool]
    error_message: str
    is_critical: bool = True


@dataclass
class DataQualityMetrics:
    """数据质量指标"""
    source: DataSourceType
    timestamp: datetime
    completeness: float  # 完整度 0-1
    accuracy: float  # 准确度 0-1
    timeliness: float  # 及时性 0-1
    consistency: float  # 一致性 0-1
    error_rate: float  # 错误率 0-1
    latency_ms: float  # 延迟毫秒
