"""
特征组合模块
实现特征降维、组合和高级特征工程方法
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging

from ..feature_types import CompositeFeatures, FeatureSet

logger = logging.getLogger(__name__)


class FeatureComposer:
    """
    特征组合器
    
    提供高级特征工程方法：
    - PCA 主成分分析
    - 时间序列分解
    - 特征交互
    - 综合指标计算
    """
    
    def __init__(self, n_components: int = 3):
        """
        初始化特征组合器
        
        Args:
            n_components: PCA 主成分数量
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def calculate_all(
        self, 
        feature_set: FeatureSet,
        historical_features: Optional[List[FeatureSet]] = None
    ) -> CompositeFeatures:
        """
        计算所有组合特征
        
        Args:
            feature_set: 当前特征集
            historical_features: 历史特征集列表 (用于时间序列分解)
            
        Returns:
            CompositeFeatures: 组合特征
        """
        features = CompositeFeatures()
        
        # PCA 主成分
        features = self._calculate_pca_components(features, feature_set)
        
        # 时间序列分解
        if historical_features and len(historical_features) >= 30:
            features = self._calculate_ts_decomposition(features, historical_features)
        
        # 特征交互
        features = self._calculate_feature_interactions(features, feature_set)
        
        # 综合指标
        features = self._calculate_composite_indicators(features, feature_set)
        
        return features
    
    def _calculate_pca_components(
        self, 
        features: CompositeFeatures, 
        feature_set: FeatureSet
    ) -> CompositeFeatures:
        """计算 PCA 主成分"""
        try:
            # 提取数值特征
            feature_vector = feature_set.to_feature_vector()
            
            # 标准化
            if not self.is_fitted:
                # 第一次拟合
                scaled = self.scaler.fit_transform(feature_vector.reshape(1, -1))
                self.is_fitted = True
            else:
                scaled = self.scaler.transform(feature_vector.reshape(1, -1))
            
            # PCA 变换
            if not hasattr(self.pca, 'components_'):
                # 需要更多数据来拟合 PCA
                features.pc1 = 0.0
                features.pc2 = 0.0
                features.pc3 = 0.0
                features.explained_variance_ratio = 0.0
            else:
                pca_result = self.pca.transform(scaled)
                features.pc1 = float(pca_result[0, 0]) if pca_result.shape[1] > 0 else 0.0
                features.pc2 = float(pca_result[0, 1]) if pca_result.shape[1] > 1 else 0.0
                features.pc3 = float(pca_result[0, 2]) if pca_result.shape[1] > 2 else 0.0
                features.explained_variance_ratio = sum(self.pca.explained_variance_ratio_)
        
        except Exception as e:
            logger.warning(f"PCA calculation failed: {e}")
            features.pc1 = 0.0
            features.pc2 = 0.0
            features.pc3 = 0.0
        
        return features
    
    def fit_pca(self, feature_sets: List[FeatureSet]):
        """
        使用历史数据拟合 PCA
        
        Args:
            feature_sets: 历史特征集列表
        """
        if len(feature_sets) < 10:
            logger.warning("Insufficient data for PCA fitting")
            return
        
        try:
            # 构建特征矩阵
            feature_matrix = np.array([fs.to_feature_vector() for fs in feature_sets])
            
            # 检查并处理常数特征 (避免除以零)
            for i in range(feature_matrix.shape[1]):
                std = np.std(feature_matrix[:, i])
                if std == 0:
                    # 添加微小噪声
                    feature_matrix[:, i] += np.random.normal(0, 1e-10, len(feature_matrix))
            
            # 标准化
            scaled = self.scaler.fit_transform(feature_matrix)
            
            # 拟合 PCA
            self.pca.fit(scaled)
            self.is_fitted = True
            
            logger.info(f"PCA fitted with {len(feature_sets)} samples")
            logger.info(f"Explained variance ratio: {self.pca.explained_variance_ratio_}")
        
        except Exception as e:
            logger.error(f"PCA fitting failed: {e}")
    
    def _calculate_ts_decomposition(
        self, 
        features: CompositeFeatures, 
        historical_features: List[FeatureSet]
    ) -> CompositeFeatures:
        """计算时间序列分解"""
        try:
            # 提取收盘价序列
            prices = np.array([fs.close for fs in historical_features])
            
            if len(prices) < 30:
                return features
            
            # 简单移动平均作为趋势
            window = min(20, len(prices) // 3)
            trend = self._moving_average(prices, window)
            
            # 去趋势
            detrended = prices - trend
            
            # 季节性 (简化处理，假设周期为7天)
            seasonal = self._estimate_seasonality(detrended, period=7)
            
            # 残差
            residual = detrended - seasonal
            
            # 当前值
            features.trend_component = trend[-1] if len(trend) > 0 else 0.0
            features.seasonal_component = seasonal[-1] if len(seasonal) > 0 else 0.0
            features.residual_component = residual[-1] if len(residual) > 0 else 0.0
        
        except Exception as e:
            logger.warning(f"Time series decomposition failed: {e}")
        
        return features
    
    def _moving_average(self, data: np.ndarray, window: int) -> np.ndarray:
        """计算移动平均"""
        if len(data) < window:
            return np.array([np.mean(data)] * len(data))
        
        ma = np.convolve(data, np.ones(window)/window, mode='valid')
        # 补齐前面
        ma = np.concatenate([np.full(window-1, ma[0]), ma])
        return ma
    
    def _estimate_seasonality(self, data: np.ndarray, period: int) -> np.ndarray:
        """估计季节性成分"""
        if len(data) < period * 2:
            return np.zeros_like(data)
        
        # 计算每个周期的平均值
        seasonal_pattern = []
        for i in range(period):
            values = [data[j] for j in range(i, len(data), period) if j < len(data)]
            seasonal_pattern.append(np.mean(values) if values else 0)
        
        # 重复模式以匹配数据长度
        seasonal = np.tile(seasonal_pattern, (len(data) // period) + 1)[:len(data)]
        
        return seasonal
    
    def _calculate_feature_interactions(
        self, 
        features: CompositeFeatures, 
        feature_set: FeatureSet
    ) -> CompositeFeatures:
        """计算特征交互"""
        tech = feature_set.technical
        onchain = feature_set.onchain
        sentiment = feature_set.sentiment
        
        # 价格-成交量交互
        if tech.volume_sma_20 > 0:
            features.price_volume_interaction = (
                tech.price_change_1d * tech.volume_ratio
            )
        
        # 动量-情绪交互
        features.momentum_sentiment_interaction = (
            (tech.rsi_14 - 50) / 50 * sentiment.composite_sentiment
        )
        
        # 波动率-链上交互
        if onchain.exchange_netflow != 0:
            features.volatility_onchain_interaction = (
                tech.bb_width * np.sign(onchain.exchange_netflow)
            )
        
        return features
    
    def _calculate_composite_indicators(
        self, 
        features: CompositeFeatures, 
        feature_set: FeatureSet
    ) -> CompositeFeatures:
        """计算综合指标"""
        tech = feature_set.technical
        sentiment = feature_set.sentiment
        
        # 综合动量指标
        momentum_factors = [
            (tech.rsi_14 - 50) / 50,
            tech.macd_histogram / (abs(tech.macd) + 1e-10),
            (tech.stochastic_k - 50) / 50,
            sentiment.composite_sentiment
        ]
        features.composite_momentum = np.mean(momentum_factors)
        
        # 综合波动率指标
        volatility_factors = [
            tech.bb_width,
            tech.atr_14 / (tech.sma_20 + 1e-10),
            abs(tech.price_change_1d) / 100
        ]
        features.composite_volatility = np.mean(volatility_factors)
        
        # 综合流动性指标
        liquidity_factors = [
            tech.volume_ratio,
            sentiment.open_interest / (sentiment.open_interest + 1e-10),
            1.0 if sentiment.funding_rate < 0.01 else 0.5
        ]
        features.composite_liquidity = np.mean(liquidity_factors)
        
        return features
    
    # ==================== 高级特征工程方法 ====================
    
    def create_polynomial_features(
        self, 
        features: Dict[str, float], 
        degree: int = 2
    ) -> Dict[str, float]:
        """
        创建多项式特征
        
        Args:
            features: 原始特征字典
            degree: 多项式次数
            
        Returns:
            Dict: 包含多项式特征的字典
        """
        result = features.copy()
        
        if degree >= 2:
            keys = list(features.keys())
            for i, key1 in enumerate(keys):
                for key2 in keys[i:]:
                    new_key = f"{key1}_x_{key2}"
                    result[new_key] = features[key1] * features[key2]
        
        if degree >= 3:
            keys = list(features.keys())
            for key in keys:
                new_key = f"{key}_cubed"
                result[new_key] = features[key] ** 3
        
        return result
    
    def create_lag_features(
        self, 
        time_series: pd.Series, 
        lags: List[int] = [1, 3, 7, 14]
    ) -> pd.DataFrame:
        """
        创建滞后特征
        
        Args:
            time_series: 时间序列数据
            lags: 滞后周期列表
            
        Returns:
            pd.DataFrame: 滞后特征 DataFrame
        """
        df = pd.DataFrame({'original': time_series})
        
        for lag in lags:
            df[f'lag_{lag}'] = time_series.shift(lag)
        
        # 变化率特征
        for lag in lags:
            df[f'change_{lag}'] = (time_series - time_series.shift(lag)) / time_series.shift(lag)
        
        return df.dropna()
    
    def create_rolling_features(
        self, 
        time_series: pd.Series, 
        windows: List[int] = [7, 14, 30]
    ) -> pd.DataFrame:
        """
        创建滚动窗口特征
        
        Args:
            time_series: 时间序列数据
            windows: 窗口大小列表
            
        Returns:
            pd.DataFrame: 滚动特征 DataFrame
        """
        df = pd.DataFrame({'original': time_series})
        
        for window in windows:
            df[f'rolling_mean_{window}'] = time_series.rolling(window=window).mean()
            df[f'rolling_std_{window}'] = time_series.rolling(window=window).std()
            df[f'rolling_min_{window}'] = time_series.rolling(window=window).min()
            df[f'rolling_max_{window}'] = time_series.rolling(window=window).max()
        
        return df.dropna()
    
    def detect_regime_change(
        self, 
        feature_history: List[Dict[str, float]], 
        lookback: int = 30
    ) -> Dict[str, Any]:
        """
        检测市场制度变化
        
        Args:
            feature_history: 特征历史列表
            lookback: 回顾期
            
        Returns:
            Dict: 制度变化分析
        """
        if len(feature_history) < lookback * 2:
            return {'regime_change_detected': False}
        
        # 提取波动率和趋势特征
        recent = feature_history[-lookback:]
        past = feature_history[-lookback*2:-lookback]
        
        recent_volatility = np.mean([f.get('bb_width', 0) for f in recent])
        past_volatility = np.mean([f.get('bb_width', 0) for f in past])
        
        recent_trend = np.mean([f.get('price_change_1d', 0) for f in recent])
        past_trend = np.mean([f.get('price_change_1d', 0) for f in past])
        
        # 检测变化
        volatility_change = abs(recent_volatility - past_volatility) / (past_volatility + 1e-10)
        trend_change = abs(recent_trend - past_trend)
        
        regime_change = volatility_change > 0.5 or trend_change > 2.0
        
        return {
            'regime_change_detected': regime_change,
            'volatility_change': volatility_change,
            'trend_change': trend_change,
            'recent_regime': {
                'volatility': 'high' if recent_volatility > 0.1 else 'low',
                'trend': 'up' if recent_trend > 1 else 'down' if recent_trend < -1 else 'sideways'
            }
        }
    
    def calculate_feature_stability(
        self, 
        feature_history: List[float], 
        window: int = 20
    ) -> Dict[str, float]:
        """
        计算特征稳定性
        
        Args:
            feature_history: 特征历史值
            window: 窗口大小
            
        Returns:
            Dict: 稳定性指标
        """
        if len(feature_history) < window:
            return {'stability_score': 0.5, 'coefficient_of_variation': 1.0}
        
        recent = feature_history[-window:]
        
        mean = np.mean(recent)
        std = np.std(recent)
        
        # 变异系数
        cv = std / (abs(mean) + 1e-10)
        
        # 稳定性分数 (0-1, 越高越稳定)
        stability = 1 / (1 + cv)
        
        return {
            'stability_score': stability,
            'coefficient_of_variation': cv,
            'mean': mean,
            'std': std
        }
    
    def select_features_by_importance(
        self, 
        features: Dict[str, List[float]], 
        target: List[float], 
        top_k: int = 10
    ) -> List[str]:
        """
        基于相关性选择重要特征
        
        Args:
            features: 特征字典 (特征名 -> 历史值列表)
            target: 目标变量历史值
            top_k: 选择前k个特征
            
        Returns:
            List: 重要特征名列表
        """
        correlations = {}
        
        for feature_name, feature_values in features.items():
            if len(feature_values) != len(target):
                continue
            
            try:
                corr = np.corrcoef(feature_values, target)[0, 1]
                correlations[feature_name] = abs(corr)
            except:
                continue
        
        # 排序并选择 top k
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        return [name for name, _ in sorted_features[:top_k]]
