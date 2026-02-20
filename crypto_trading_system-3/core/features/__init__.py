"""
特征工程层 (Feature Engineering Layer) - OpenClaw Crypto Trading System

将原始市场数据转换为高质量的特征向量，供推理层使用。

支持的特征类型：
- 技术特征: RSI, MACD, 布林带, EMA, ATR等
- 链上特征: 交易所流向, 鲸鱼活动, 网络活跃度等
- 情绪特征: 恐惧贪婪指数, 社交情绪, 新闻情绪等
- 组合特征: PCA, 时间序列分解等
"""

from .feature_types import (
    FeatureSet,
    FeatureConfig,
    TechnicalFeatures,
    OnchainFeatures,
    SentimentFeatures,
    CompositeFeatures,
    FeatureImportance
)

from .feature_engineering import FeatureEngineering
from .technical.technical_indicators import TechnicalIndicators
from .onchain.onchain_metrics import OnchainMetrics
from .sentiment.sentiment_analyzer import SentimentAnalyzer
from .composite.feature_composer import FeatureComposer

__all__ = [
    'FeatureSet',
    'FeatureConfig',
    'TechnicalFeatures',
    'OnchainFeatures',
    'SentimentFeatures',
    'CompositeFeatures',
    'FeatureImportance',
    'FeatureEngineering',
    'TechnicalIndicators',
    'OnchainMetrics',
    'SentimentAnalyzer',
    'FeatureComposer'
]
