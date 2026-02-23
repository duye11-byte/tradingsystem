"""
OpenClaw 输入层 (Input Layer)
第1层：多源数据融合与实时数据流处理

提供功能：
- 实时价格数据 (Binance WebSocket/REST, CoinGecko)
- 链上数据分析 (Dune Analytics, DeFiLlama, Arkham)
- 市场情绪指标 (Alternative.me, Coinalyze, Binance)
- 新闻事件驱动 (CryptoPanic, NewsData.io, Reddit)
- 数据验证与容错机制
"""

from .input_types import (
    DataSourceType,
    DataFrequency,
    PriceData,
    OrderBookData,
    TradeData,
    OnChainData,
    SentimentData,
    NewsData,
    MarketData,
    DataSourceConfig,
    InputContext,
    InputResult,
)

from .data_validator import DataValidator, ValidationRule
from .data_aggregator import DataAggregator
from .input_engine import InputEngine

# 价格数据源
from .sources.price.binance_client import BinanceClient
from .sources.price.coingecko_client import CoinGeckoClient

# 链上数据源
from .sources.onchain.dune_client import DuneClient
from .sources.onchain.defillama_client import DeFiLlamaClient
from .sources.onchain.arkham_client import ArkhamClient

# 情绪数据源
from .sources.sentiment.alternative_me_client import AlternativeMeClient
from .sources.sentiment.coinalyze_client import CoinalyzeClient
from .sources.sentiment.binance_sentiment_client import BinanceSentimentClient

# 新闻数据源
from .sources.news.cryptopanic_client import CryptoPanicClient
from .sources.news.newsdata_client import NewsDataClient
from .sources.news.reddit_client import RedditClient

__all__ = [
    # 类型定义
    'DataSourceType',
    'DataFrequency',
    'PriceData',
    'OrderBookData',
    'TradeData',
    'OnChainData',
    'SentimentData',
    'NewsData',
    'MarketData',
    'DataSourceConfig',
    'InputContext',
    'InputResult',
    # 核心组件
    'DataValidator',
    'ValidationRule',
    'DataAggregator',
    'InputEngine',
    # 价格数据源
    'BinanceClient',
    'CoinGeckoClient',
    # 链上数据源
    'DuneClient',
    'DeFiLlamaClient',
    'ArkhamClient',
    # 情绪数据源
    'AlternativeMeClient',
    'CoinalyzeClient',
    'BinanceSentimentClient',
    # 新闻数据源
    'CryptoPanicClient',
    'NewsDataClient',
    'RedditClient',
]

__version__ = "1.0.0"
__author__ = "OpenClaw Team"
