"""
情绪分析模块
实现市场情绪指标的计算和分析
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from ..feature_types import SentimentFeatures

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    情绪分析器
    
    提供各种市场情绪指标的计算，包括：
    - 恐惧贪婪指数
    - 社交媒体情绪
    - 新闻情绪
    - 期货资金费率
    - 期权指标
    """
    
    def __init__(self):
        """初始化情绪分析器"""
        pass
    
    def calculate_all(self, data: Dict[str, any]) -> SentimentFeatures:
        """
        计算所有情绪指标
        
        Args:
            data: 包含情绪数据的字典
            
        Returns:
            SentimentFeatures: 所有情绪指标
        """
        features = SentimentFeatures()
        
        # 恐惧贪婪指数
        features = self._calculate_fear_greed(features, data)
        
        # 社交媒体情绪
        features = self._calculate_social_sentiment(features, data)
        
        # 新闻情绪
        features = self._calculate_news_sentiment(features, data)
        
        # 期货指标
        features = self._calculate_futures_metrics(features, data)
        
        # 期权指标
        features = self._calculate_options_metrics(features, data)
        
        # 综合情绪
        features = self._calculate_composite_sentiment(features)
        
        return features
    
    def _calculate_fear_greed(
        self, 
        features: SentimentFeatures, 
        data: Dict[str, any]
    ) -> SentimentFeatures:
        """计算恐惧贪婪指数"""
        features.fear_greed_index = data.get('fear_greed_index', 50)
        
        # 分类
        if features.fear_greed_index >= 75:
            features.fear_greed_classification = "Extreme Greed"
        elif features.fear_greed_index >= 55:
            features.fear_greed_classification = "Greed"
        elif features.fear_greed_index >= 45:
            features.fear_greed_classification = "Neutral"
        elif features.fear_greed_index >= 25:
            features.fear_greed_classification = "Fear"
        else:
            features.fear_greed_classification = "Extreme Fear"
        
        # 变化
        fg_history = data.get('fear_greed_history', [])
        if len(fg_history) >= 2:
            prev_fg = fg_history[-2]
            features.fear_greed_change = features.fear_greed_index - prev_fg
        
        # 极端情绪标记
        features.extreme_greed = features.fear_greed_index >= 80
        features.extreme_fear = features.fear_greed_index <= 20
        
        return features
    
    def _calculate_social_sentiment(
        self, 
        features: SentimentFeatures, 
        data: Dict[str, any]
    ) -> SentimentFeatures:
        """计算社交媒体情绪"""
        # 总体社交情绪
        features.social_sentiment = data.get('social_sentiment', 0.0)
        
        # 变化
        social_history = data.get('social_sentiment_history', [])
        if len(social_history) >= 2:
            prev_social = social_history[-2]
            features.social_sentiment_change = features.social_sentiment - prev_social
        
        # 社交量
        features.social_volume = data.get('social_volume', 0)
        
        # 平台细分
        features.twitter_sentiment = data.get('twitter_sentiment', 0.0)
        features.reddit_sentiment = data.get('reddit_sentiment', 0.0)
        
        return features
    
    def _calculate_news_sentiment(
        self, 
        features: SentimentFeatures, 
        data: Dict[str, any]
    ) -> SentimentFeatures:
        """计算新闻情绪"""
        features.news_sentiment = data.get('news_sentiment', 0.0)
        
        # 变化
        news_history = data.get('news_sentiment_history', [])
        if len(news_history) >= 2:
            prev_news = news_history[-2]
            features.news_sentiment_change = features.news_sentiment - prev_news
        
        # 新闻量
        features.news_volume = data.get('news_volume', 0)
        
        return features
    
    def _calculate_futures_metrics(
        self, 
        features: SentimentFeatures, 
        data: Dict[str, any]
    ) -> SentimentFeatures:
        """计算期货指标"""
        features.funding_rate = data.get('funding_rate', 0.0)
        
        # 资金费率变化
        funding_history = data.get('funding_rate_history', [])
        if len(funding_history) >= 2:
            prev_funding = funding_history[-2]
            features.funding_rate_change = features.funding_rate - prev_funding
        
        # 多空比
        features.long_short_ratio = data.get('long_short_ratio', 1.0)
        
        # 持仓量
        features.open_interest = data.get('open_interest', 0.0)
        
        oi_history = data.get('open_interest_history', [])
        if len(oi_history) >= 2:
            prev_oi = oi_history[-2]
            if prev_oi > 0:
                features.open_interest_change = (
                    (features.open_interest - prev_oi) / prev_oi * 100
                )
        
        return features
    
    def _calculate_options_metrics(
        self, 
        features: SentimentFeatures, 
        data: Dict[str, any]
    ) -> SentimentFeatures:
        """计算期权指标"""
        features.put_call_ratio = data.get('put_call_ratio', 1.0)
        features.iv_skew = data.get('iv_skew', 0.0)
        
        return features
    
    def _calculate_composite_sentiment(
        self, 
        features: SentimentFeatures
    ) -> SentimentFeatures:
        """计算综合情绪指标"""
        # 综合情绪分数 (加权平均)
        weights = {
            'fear_greed': 0.25,
            'social': 0.20,
            'news': 0.15,
            'funding': 0.25,
            'long_short': 0.15
        }
        
        # 标准化各指标到 -1 到 1 范围
        fg_normalized = (features.fear_greed_index - 50) / 50  # 0-100 -> -1 to 1
        social_normalized = features.social_sentiment  # 假设已经是 -1 to 1
        news_normalized = features.news_sentiment  # 假设已经是 -1 to 1
        funding_normalized = -np.sign(features.funding_rate) * min(abs(features.funding_rate) * 100, 1)  # 高资金费率 = 过度乐观
        ls_normalized = (features.long_short_ratio - 1) / 2  # 1 -> 0, >1 -> positive, <1 -> negative
        
        # 加权综合
        features.composite_sentiment = (
            fg_normalized * weights['fear_greed'] +
            social_normalized * weights['social'] +
            news_normalized * weights['news'] +
            funding_normalized * weights['funding'] +
            ls_normalized * weights['long_short']
        )
        
        # 情绪动量 (综合情绪的变化趋势)
        # 这里简化处理，实际应该使用历史数据计算
        features.sentiment_momentum = features.fear_greed_change / 10 if abs(features.fear_greed_change) <= 10 else np.sign(features.fear_greed_change)
        
        return features
    
    # ==================== 高级情绪分析 ====================
    
    def analyze_fear_greed_components(
        self, 
        components: Dict[str, float]
    ) -> Dict[str, any]:
        """
        分析恐惧贪婪指数的各个组成部分
        
        Args:
            components: 各组成部分分数
            
        Returns:
            Dict: 分析结果
        """
        analysis = {
            'dominant_factor': '',
            'dominant_score': 0.0,
            'fear_factors': [],
            'greed_factors': []
        }
        
        for factor, score in components.items():
            if score > 70:
                analysis['greed_factors'].append((factor, score))
            elif score < 30:
                analysis['fear_factors'].append((factor, score))
            
            if abs(score - 50) > abs(analysis['dominant_score'] - 50):
                analysis['dominant_factor'] = factor
                analysis['dominant_score'] = score
        
        return analysis
    
    def detect_sentiment_extremes(
        self, 
        sentiment_history: List[float], 
        threshold_std: float = 2.0
    ) -> Dict[str, any]:
        """
        检测情绪极端值
        
        Args:
            sentiment_history: 情绪历史数据
            threshold_std: 标准差阈值
            
        Returns:
            Dict: 极端值检测结果
        """
        if len(sentiment_history) < 10:
            return {'is_extreme': False, 'z_score': 0.0}
        
        mean = np.mean(sentiment_history)
        std = np.std(sentiment_history)
        
        if std == 0:
            return {'is_extreme': False, 'z_score': 0.0}
        
        current = sentiment_history[-1]
        z_score = (current - mean) / std
        
        is_extreme = abs(z_score) > threshold_std
        
        return {
            'is_extreme': is_extreme,
            'z_score': z_score,
            'current': current,
            'mean': mean,
            'std': std,
            'extreme_type': 'high' if z_score > threshold_std else 'low' if z_score < -threshold_std else 'normal'
        }
    
    def calculate_sentiment_divergence(
        self, 
        price_changes: List[float], 
        sentiment_changes: List[float]
    ) -> Dict[str, any]:
        """
        计算价格与情绪的背离
        
        Args:
            price_changes: 价格变化列表
            sentiment_changes: 情绪变化列表
            
        Returns:
            Dict: 背离分析结果
        """
        if len(price_changes) != len(sentiment_changes) or len(price_changes) < 5:
            return {'divergence_detected': False, 'correlation': 0.0}
        
        # 计算相关性
        correlation = np.corrcoef(price_changes, sentiment_changes)[0, 1]
        
        # 检测背离
        recent_price = price_changes[-5:]
        recent_sentiment = sentiment_changes[-5:]
        
        price_trend = np.mean(recent_price)
        sentiment_trend = np.mean(recent_sentiment)
        
        # 价格上升但情绪下降 = 看跌背离
        # 价格下降但情绪上升 = 看涨背离
        divergence_detected = (price_trend > 0 and sentiment_trend < 0) or (price_trend < 0 and sentiment_trend > 0)
        
        divergence_type = ''
        if divergence_detected:
            if price_trend > 0 and sentiment_trend < 0:
                divergence_type = 'bearish'  # 看跌背离
            else:
                divergence_type = 'bullish'  # 看涨背离
        
        return {
            'divergence_detected': divergence_detected,
            'divergence_type': divergence_type,
            'correlation': correlation,
            'price_trend': price_trend,
            'sentiment_trend': sentiment_trend
        }
    
    def analyze_funding_rate_regime(
        self, 
        funding_history: List[float], 
        lookback: int = 30
    ) -> Dict[str, any]:
        """
        分析资金费率制度
        
        Args:
            funding_history: 资金费率历史
            lookback: 回顾期
            
        Returns:
            Dict: 资金费率分析
        """
        if len(funding_history) < lookback:
            return {'regime': 'insufficient_data', 'avg_rate': 0.0}
        
        recent = funding_history[-lookback:]
        avg_rate = np.mean(recent)
        
        # 判断制度
        if avg_rate > 0.01:  # 1%
            regime = 'expensive_longs'  # 做多成本高，可能过度乐观
        elif avg_rate < -0.01:
            regime = 'expensive_shorts'  # 做空成本高，可能过度悲观
        else:
            regime = 'balanced'
        
        # 趋势
        first_half = np.mean(recent[:len(recent)//2])
        second_half = np.mean(recent[len(recent)//2:])
        
        if second_half > first_half * 1.2:
            trend = 'increasing'
        elif second_half < first_half * 0.8:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'regime': regime,
            'avg_rate': avg_rate,
            'trend': trend,
            'max_rate': max(recent),
            'min_rate': min(recent)
        }
    
    def calculate_social_volume_anomaly(
        self, 
        volume_history: List[int], 
        threshold_std: float = 2.5
    ) -> Dict[str, any]:
        """
        检测社交量异常
        
        Args:
            volume_history: 社交量历史
            threshold_std: 标准差阈值
            
        Returns:
            Dict: 异常检测结果
        """
        if len(volume_history) < 10:
            return {'is_anomaly': False, 'z_score': 0.0}
        
        mean = np.mean(volume_history)
        std = np.std(volume_history)
        
        if std == 0:
            return {'is_anomaly': False, 'z_score': 0.0}
        
        current = volume_history[-1]
        z_score = (current - mean) / std
        
        is_anomaly = z_score > threshold_std
        
        return {
            'is_anomaly': is_anomaly,
            'z_score': z_score,
            'current': current,
            'mean': mean,
            'std': std,
            'significance': 'high' if z_score > 3 else 'medium' if z_score > 2 else 'normal'
        }
    
    def generate_sentiment_signal(
        self, 
        features: SentimentFeatures
    ) -> Dict[str, any]:
        """
        基于情绪特征生成交易信号
        
        Args:
            features: 情绪特征
            
        Returns:
            Dict: 情绪信号
        """
        signal_strength = 0
        signal_direction = 'neutral'
        reasons = []
        
        # 恐惧贪婪指数信号
        if features.extreme_fear:
            signal_strength += 2
            signal_direction = 'bullish'
            reasons.append("极度恐惧 - 潜在买入机会")
        elif features.extreme_greed:
            signal_strength -= 2
            signal_direction = 'bearish'
            reasons.append("极度贪婪 - 潜在卖出信号")
        
        # 资金费率信号
        if features.funding_rate > 0.01:
            signal_strength -= 1
            reasons.append("高资金费率 - 多头过热")
        elif features.funding_rate < -0.01:
            signal_strength += 1
            reasons.append("负资金费率 - 空头过热")
        
        # 综合情绪信号
        if features.composite_sentiment < -0.5:
            signal_strength += 1
            if signal_direction == 'neutral':
                signal_direction = 'bullish'
            reasons.append("综合情绪悲观")
        elif features.composite_sentiment > 0.5:
            signal_strength -= 1
            if signal_direction == 'neutral':
                signal_direction = 'bearish'
            reasons.append("综合情绪乐观")
        
        return {
            'signal_direction': signal_direction,
            'signal_strength': min(abs(signal_strength), 3),
            'raw_score': signal_strength,
            'reasons': reasons,
            'confidence': min(abs(features.composite_sentiment) + 0.3, 1.0)
        }
