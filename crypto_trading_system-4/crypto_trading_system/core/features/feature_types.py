"""
特征工程层类型定义
定义所有特征相关的数据结构和类型
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime
from enum import Enum
import numpy as np
import pandas as pd


class FeatureCategory(Enum):
    """特征类别"""
    TECHNICAL = "technical"
    ONCHAIN = "onchain"
    SENTIMENT = "sentiment"
    COMPOSITE = "composite"


class IndicatorType(Enum):
    """技术指标类型"""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"


@dataclass
class FeatureConfig:
    """特征配置"""
    # 技术特征配置
    technical_enabled: bool = True
    technical_indicators: List[str] = field(default_factory=lambda: [
        'rsi', 'macd', 'bollinger', 'ema', 'sma', 'atr', 'obv'
    ])
    technical_timeframes: List[str] = field(default_factory=lambda: ['1h', '4h', '1d'])
    
    # 链上特征配置
    onchain_enabled: bool = True
    onchain_metrics: List[str] = field(default_factory=lambda: [
        'exchange_flow', 'whale_activity', 'network_activity', 'supply_distribution'
    ])
    
    # 情绪特征配置
    sentiment_enabled: bool = True
    sentiment_sources: List[str] = field(default_factory=lambda: [
        'fear_greed', 'social_media', 'news', 'funding_rate'
    ])
    
    # 组合特征配置
    composite_enabled: bool = True
    composite_methods: List[str] = field(default_factory=lambda: [
        'pca', 'ts_decomposition', 'feature_interactions'
    ])
    
    # 通用配置
    lookback_periods: int = 100
    normalize: bool = True
    fill_missing: bool = True


@dataclass
class TechnicalFeatures:
    """技术特征集合"""
    # 趋势指标
    sma_20: float = 0.0
    sma_50: float = 0.0
    sma_200: float = 0.0
    ema_12: float = 0.0
    ema_26: float = 0.0
    ema_50: float = 0.0
    
    # 动量指标
    rsi_14: float = 50.0
    rsi_7: float = 50.0
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    stochastic_k: float = 50.0
    stochastic_d: float = 50.0
    williams_r: float = -50.0
    cci_20: float = 0.0
    
    # 波动率指标
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_width: float = 0.0
    bb_percent: float = 0.5
    atr_14: float = 0.0
    atr_7: float = 0.0
    
    # 成交量指标
    obv: float = 0.0
    volume_sma_20: float = 0.0
    volume_ratio: float = 1.0
    mfi_14: float = 50.0
    
    # 价格变化
    price_change_1h: float = 0.0
    price_change_4h: float = 0.0
    price_change_1d: float = 0.0
    price_change_7d: float = 0.0
    
    # 额外特征
    adx_14: float = 25.0
    plus_di: float = 25.0
    minus_di: float = 25.0
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'sma_20': self.sma_20,
            'sma_50': self.sma_50,
            'sma_200': self.sma_200,
            'ema_12': self.ema_12,
            'ema_26': self.ema_26,
            'ema_50': self.ema_50,
            'rsi_14': self.rsi_14,
            'rsi_7': self.rsi_7,
            'macd': self.macd,
            'macd_signal': self.macd_signal,
            'macd_histogram': self.macd_histogram,
            'stochastic_k': self.stochastic_k,
            'stochastic_d': self.stochastic_d,
            'williams_r': self.williams_r,
            'cci_20': self.cci_20,
            'bb_upper': self.bb_upper,
            'bb_middle': self.bb_middle,
            'bb_lower': self.bb_lower,
            'bb_width': self.bb_width,
            'bb_percent': self.bb_percent,
            'atr_14': self.atr_14,
            'atr_7': self.atr_7,
            'obv': self.obv,
            'volume_sma_20': self.volume_sma_20,
            'volume_ratio': self.volume_ratio,
            'mfi_14': self.mfi_14,
            'price_change_1h': self.price_change_1h,
            'price_change_4h': self.price_change_4h,
            'price_change_1d': self.price_change_1d,
            'price_change_7d': self.price_change_7d,
            'adx_14': self.adx_14,
            'plus_di': self.plus_di,
            'minus_di': self.minus_di
        }


@dataclass
class OnchainFeatures:
    """链上特征集合"""
    # 交易所流向
    exchange_inflow: float = 0.0
    exchange_outflow: float = 0.0
    exchange_netflow: float = 0.0
    exchange_inflow_change: float = 0.0
    exchange_outflow_change: float = 0.0
    
    # 鲸鱼活动
    whale_tx_count: int = 0
    whale_volume: float = 0.0
    whale_inflow: float = 0.0
    whale_outflow: float = 0.0
    whale_accumulation: float = 0.0
    
    # 网络活跃度
    active_addresses: int = 0
    active_addresses_change: float = 0.0
    transaction_count: int = 0
    transaction_count_change: float = 0.0
    avg_transaction_value: float = 0.0
    
    # 供应分布
    supply_on_exchanges: float = 0.0
    supply_on_exchanges_pct: float = 0.0
    long_term_holder_supply: float = 0.0
    short_term_holder_supply: float = 0.0
    
    # 矿工指标
    miner_revenue: float = 0.0
    miner_outflow: float = 0.0
    
    # 网络健康
    hash_rate: float = 0.0
    hash_rate_change: float = 0.0
    difficulty: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'exchange_inflow': self.exchange_inflow,
            'exchange_outflow': self.exchange_outflow,
            'exchange_netflow': self.exchange_netflow,
            'exchange_inflow_change': self.exchange_inflow_change,
            'exchange_outflow_change': self.exchange_outflow_change,
            'whale_tx_count': self.whale_tx_count,
            'whale_volume': self.whale_volume,
            'whale_inflow': self.whale_inflow,
            'whale_outflow': self.whale_outflow,
            'whale_accumulation': self.whale_accumulation,
            'active_addresses': self.active_addresses,
            'active_addresses_change': self.active_addresses_change,
            'transaction_count': self.transaction_count,
            'transaction_count_change': self.transaction_count_change,
            'avg_transaction_value': self.avg_transaction_value,
            'supply_on_exchanges': self.supply_on_exchanges,
            'supply_on_exchanges_pct': self.supply_on_exchanges_pct,
            'long_term_holder_supply': self.long_term_holder_supply,
            'short_term_holder_supply': self.short_term_holder_supply,
            'miner_revenue': self.miner_revenue,
            'miner_outflow': self.miner_outflow,
            'hash_rate': self.hash_rate,
            'hash_rate_change': self.hash_rate_change,
            'difficulty': self.difficulty
        }


@dataclass
class SentimentFeatures:
    """情绪特征集合"""
    # 恐惧贪婪指数
    fear_greed_index: int = 50
    fear_greed_classification: str = "Neutral"
    fear_greed_change: float = 0.0
    
    # 社交媒体情绪
    social_sentiment: float = 0.0
    social_sentiment_change: float = 0.0
    social_volume: int = 0
    twitter_sentiment: float = 0.0
    reddit_sentiment: float = 0.0
    
    # 新闻情绪
    news_sentiment: float = 0.0
    news_sentiment_change: float = 0.0
    news_volume: int = 0
    
    # 期货指标
    funding_rate: float = 0.0
    funding_rate_change: float = 0.0
    long_short_ratio: float = 1.0
    open_interest: float = 0.0
    open_interest_change: float = 0.0
    
    # 期权指标
    put_call_ratio: float = 1.0
    iv_skew: float = 0.0
    
    # 综合情绪
    composite_sentiment: float = 0.0
    sentiment_momentum: float = 0.0
    extreme_greed: bool = False
    extreme_fear: bool = False
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'fear_greed_index': self.fear_greed_index,
            'fear_greed_change': self.fear_greed_change,
            'social_sentiment': self.social_sentiment,
            'social_sentiment_change': self.social_sentiment_change,
            'social_volume': self.social_volume,
            'twitter_sentiment': self.twitter_sentiment,
            'reddit_sentiment': self.reddit_sentiment,
            'news_sentiment': self.news_sentiment,
            'news_sentiment_change': self.news_sentiment_change,
            'news_volume': self.news_volume,
            'funding_rate': self.funding_rate,
            'funding_rate_change': self.funding_rate_change,
            'long_short_ratio': self.long_short_ratio,
            'open_interest': self.open_interest,
            'open_interest_change': self.open_interest_change,
            'put_call_ratio': self.put_call_ratio,
            'iv_skew': self.iv_skew,
            'composite_sentiment': self.composite_sentiment,
            'sentiment_momentum': self.sentiment_momentum
        }


@dataclass
class CompositeFeatures:
    """组合特征集合"""
    # PCA 主成分
    pc1: float = 0.0
    pc2: float = 0.0
    pc3: float = 0.0
    explained_variance_ratio: float = 0.0
    
    # 时间序列分解
    trend_component: float = 0.0
    seasonal_component: float = 0.0
    residual_component: float = 0.0
    
    # 特征交互
    price_volume_interaction: float = 0.0
    momentum_sentiment_interaction: float = 0.0
    volatility_onchain_interaction: float = 0.0
    
    # 综合指标
    composite_momentum: float = 0.0
    composite_volatility: float = 0.0
    composite_liquidity: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'pc1': self.pc1,
            'pc2': self.pc2,
            'pc3': self.pc3,
            'explained_variance_ratio': self.explained_variance_ratio,
            'trend_component': self.trend_component,
            'seasonal_component': self.seasonal_component,
            'residual_component': self.residual_component,
            'price_volume_interaction': self.price_volume_interaction,
            'momentum_sentiment_interaction': self.momentum_sentiment_interaction,
            'volatility_onchain_interaction': self.volatility_onchain_interaction,
            'composite_momentum': self.composite_momentum,
            'composite_volatility': self.composite_volatility,
            'composite_liquidity': self.composite_liquidity
        }


@dataclass
class FeatureSet:
    """完整特征集合"""
    symbol: str
    timestamp: datetime
    
    # 原始价格数据
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    
    # 各类特征
    technical: TechnicalFeatures = field(default_factory=TechnicalFeatures)
    onchain: OnchainFeatures = field(default_factory=OnchainFeatures)
    sentiment: SentimentFeatures = field(default_factory=SentimentFeatures)
    composite: CompositeFeatures = field(default_factory=CompositeFeatures)
    
    # 元数据
    feature_version: str = "1.0"
    data_quality_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为完整字典"""
        result = {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'feature_version': self.feature_version,
            'data_quality_score': self.data_quality_score
        }
        
        # 合并所有特征
        result.update(self.technical.to_dict())
        result.update(self.onchain.to_dict())
        result.update(self.sentiment.to_dict())
        result.update(self.composite.to_dict())
        
        return result
    
    def to_feature_vector(self) -> np.ndarray:
        """转换为特征向量 (用于模型输入)"""
        feature_dict = self.to_dict()
        # 排除非数值字段
        exclude_keys = ['symbol', 'timestamp', 'feature_version', 'fear_greed_classification']
        numeric_values = [
            v for k, v in feature_dict.items() 
            if k not in exclude_keys and isinstance(v, (int, float))
        ]
        return np.array(numeric_values, dtype=np.float32)


@dataclass
class FeatureImportance:
    """特征重要性"""
    feature_name: str
    category: FeatureCategory
    importance_score: float
    correlation_with_target: float
    stability_score: float
    
    def __repr__(self):
        return f"{self.feature_name}: {self.importance_score:.3f} ({self.category.value})"


@dataclass
class FeatureExtractionResult:
    """特征提取结果"""
    success: bool
    feature_set: Optional[FeatureSet]
    error_message: Optional[str]
    extraction_time_ms: float
    features_extracted: int
    features_failed: List[str]
