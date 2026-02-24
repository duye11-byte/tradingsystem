"""
OpenClaw 特征工程层 - 技术指标计算
第3层: 特征工程层
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """技术指标计算器"""
    
    @staticmethod
    def ema(data: np.ndarray, period: int) -> np.ndarray:
        """计算指数移动平均线"""
        if len(data) < period:
            return np.array([np.nan] * len(data))
        
        multiplier = 2 / (period + 1)
        ema_values = np.zeros(len(data))
        ema_values[:period-1] = np.nan
        ema_values[period-1] = np.mean(data[:period])
        
        for i in range(period, len(data)):
            ema_values[i] = (data[i] - ema_values[i-1]) * multiplier + ema_values[i-1]
        
        return ema_values
    
    @staticmethod
    def sma(data: np.ndarray, period: int) -> np.ndarray:
        """计算简单移动平均线"""
        if len(data) < period:
            return np.array([np.nan] * len(data))
        
        result = np.convolve(data, np.ones(period)/period, mode='valid')
        return np.concatenate([np.array([np.nan] * (period - 1)), result])
    
    @staticmethod
    def rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
        """计算RSI指标"""
        if len(data) < period + 1:
            return np.array([50.0] * len(data))
        
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.convolve(gains, np.ones(period)/period, mode='valid')
        avg_losses = np.convolve(losses, np.ones(period)/period, mode='valid')
        
        rs = np.where(avg_losses == 0, 100, avg_gains / avg_losses)
        rsi_values = 100 - (100 / (1 + rs))
        
        padding = len(data) - len(rsi_values)
        return np.concatenate([np.array([50.0] * padding), rsi_values])
    
    @staticmethod
    def macd(data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算MACD指标"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line[~np.isnan(macd_line)], signal)
        
        # 对齐长度
        min_len = min(len(macd_line), len(signal_line) + np.sum(np.isnan(macd_line)))
        macd_line = macd_line[-min_len:]
        
        # 重新计算signal line
        valid_macd = macd_line[~np.isnan(macd_line)]
        if len(valid_macd) >= signal:
            signal_line = TechnicalIndicators.ema(valid_macd, signal)
            histogram = valid_macd[-len(signal_line):] - signal_line
            
            # 填充nan
            nan_count = len(data) - len(signal_line)
            signal_line = np.concatenate([np.array([np.nan] * nan_count), signal_line])
            histogram = np.concatenate([np.array([np.nan] * nan_count), histogram])
            macd_line = np.concatenate([np.array([np.nan] * (len(data) - len(valid_macd))), valid_macd])
        else:
            signal_line = np.array([np.nan] * len(data))
            histogram = np.array([np.nan] * len(data))
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算ADX指标"""
        if len(close) < period + 1:
            return np.array([25.0] * len(close)), np.array([0.0] * len(close)), np.array([0.0] * len(close))
        
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        
        # +DM and -DM
        plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                          np.maximum(high[1:] - high[:-1], 0), 0)
        minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]),
                           np.maximum(low[:-1] - low[1:], 0), 0)
        
        # Smooth TR and DM
        atr = TechnicalIndicators.ema(np.concatenate([[tr[0]], tr]), period)[1:]
        plus_di = 100 * TechnicalIndicators.ema(np.concatenate([[plus_dm[0]], plus_dm]), period)[1:] / atr
        minus_di = 100 * TechnicalIndicators.ema(np.concatenate([[minus_dm[0]], minus_dm]), period)[1:] / atr
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx_values = TechnicalIndicators.ema(dx, period)
        
        # 填充
        padding = len(close) - len(adx_values)
        adx_values = np.concatenate([np.array([25.0] * padding), adx_values])
        plus_di = np.concatenate([np.array([0.0] * padding), plus_di])
        minus_di = np.concatenate([np.array([0.0] * padding), minus_di])
        
        return adx_values, plus_di, minus_di
    
    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """计算ATR指标"""
        if len(close) < 2:
            return np.array([0.0] * len(close))
        
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        
        atr_values = TechnicalIndicators.ema(np.concatenate([[tr[0]], tr]), period)
        return atr_values
    
    @staticmethod
    def bollinger_bands(data: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算布林带"""
        if len(data) < period:
            middle = np.array([np.nan] * len(data))
            upper = np.array([np.nan] * len(data))
            lower = np.array([np.nan] * len(data))
            return upper, middle, lower
        
        middle = TechnicalIndicators.sma(data, period)
        
        # 计算标准差
        std_values = np.array([np.nan] * len(data))
        for i in range(period - 1, len(data)):
            std_values[i] = np.std(data[i-period+1:i+1])
        
        upper = middle + std_dev * std_values
        lower = middle - std_dev * std_values
        
        return upper, middle, lower
    
    @staticmethod
    def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """计算OBV指标"""
        if len(close) < 2:
            return np.array([0.0] * len(close))
        
        obv_values = np.zeros(len(close))
        obv_values[0] = volume[0]
        
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv_values[i] = obv_values[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv_values[i] = obv_values[i-1] - volume[i]
            else:
                obv_values[i] = obv_values[i-1]
        
        return obv_values
    
    @staticmethod
    def pivot_points(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Tuple[float, float, float, float, float]:
        """计算枢轴点"""
        if len(close) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        
        pivot = (high[-1] + low[-1] + close[-1]) / 3
        r1 = 2 * pivot - low[-1]
        s1 = 2 * pivot - high[-1]
        r2 = pivot + (high[-1] - low[-1])
        s2 = pivot - (high[-1] - low[-1])
        
        return pivot, r1, s1, r2, s2
    
    @staticmethod
    def find_support_resistance(data: np.ndarray, window: int = 10) -> Tuple[float, float]:
        """寻找支撑阻力位"""
        if len(data) < window * 2:
            return data[-1] * 0.95, data[-1] * 1.05
        
        # 简单实现：找近期高低点
        recent_data = data[-window*2:]
        highs = []
        lows = []
        
        for i in range(window, len(recent_data) - window):
            # 局部高点
            if all(recent_data[i] >= recent_data[i-j] for j in range(1, window+1)) and \
               all(recent_data[i] >= recent_data[i+j] for j in range(1, window+1)):
                highs.append(recent_data[i])
            
            # 局部低点
            if all(recent_data[i] <= recent_data[i-j] for j in range(1, window+1)) and \
               all(recent_data[i] <= recent_data[i+j] for j in range(1, window+1)):
                lows.append(recent_data[i])
        
        support = np.mean(lows) if lows else data[-1] * 0.95
        resistance = np.mean(highs) if highs else data[-1] * 1.05
        
        return support, resistance
    
    @staticmethod
    def trend_strength(data: np.ndarray, period: int = 14) -> float:
        """计算趋势强度 (-1到1)"""
        if len(data) < period:
            return 0.0
        
        returns = np.diff(data[-period:]) / data[-period:-1]
        avg_return = np.mean(returns)
        std_return = np.std(returns) + 1e-10
        
        # 夏普风格的趋势强度
        strength = avg_return / std_return
        return np.clip(strength, -1, 1)
    
    @staticmethod
    def volatility_percentile(data: np.ndarray, lookback: int = 100) -> float:
        """计算波动率百分位"""
        if len(data) < lookback + 1:
            return 50.0
        
        returns = np.abs(np.diff(data[-lookback-1:]) / data[-lookback-1:-1])
        current_vol = np.std(returns[-20:])  # 最近20期波动率
        historical_vol = np.std(returns)
        
        if historical_vol == 0:
            return 50.0
        
        # 计算百分位
        percentile = (current_vol / historical_vol) * 50
        return np.clip(percentile, 0, 100)


class FeatureEngine:
    """特征工程引擎"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.cache: Dict[str, Dict] = {}
    
    def calculate_all_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """计算所有特征"""
        if len(df) < 50:
            logger.warning(f"数据不足: {len(df)} 条")
            return {}
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values if 'volume' in df.columns else np.ones(len(close))
        
        features = {}
        
        # 趋势指标
        features['ema_9'] = self.indicators.ema(close, 9)
        features['ema_21'] = self.indicators.ema(close, 21)
        features['ema_50'] = self.indicators.ema(close, 50)
        features['ema_200'] = self.indicators.ema(close, 200)
        
        # ADX
        features['adx'], features['di_plus'], features['di_minus'] = \
            self.indicators.adx(high, low, close, 14)
        
        # 动量指标
        features['rsi'] = self.indicators.rsi(close, 14)
        features['rsi_ema'] = self.indicators.ema(features['rsi'], 9)
        features['macd'], features['macd_signal'], features['macd_histogram'] = \
            self.indicators.macd(close, 12, 26, 9)
        
        # 波动率指标
        features['atr'] = self.indicators.atr(high, low, close, 14)
        features['atr_percent'] = features['atr'] / close * 100
        
        features['bb_upper'], features['bb_middle'], features['bb_lower'] = \
            self.indicators.bollinger_bands(close, 20, 2)
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
        
        # 成交量指标
        features['volume_sma'] = self.indicators.sma(volume, 20)
        features['volume_ratio'] = volume / (features['volume_sma'] + 1e-10)
        features['obv'] = self.indicators.obv(close, volume)
        
        # 支撑阻力
        features['support'], features['resistance'] = \
            self.indicators.find_support_resistance(close, 10)
        
        # 趋势强度
        features['trend_strength'] = np.array([
            self.indicators.trend_strength(close[:i+1], 14) if i >= 14 else 0
            for i in range(len(close))
        ])
        
        # 波动率百分位
        features['volatility_percentile'] = np.array([
            self.indicators.volatility_percentile(close[:i+1], 100) if i >= 100 else 50
            for i in range(len(close))
        ])
        
        return features
    
    def get_latest_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """获取最新特征值"""
        features = self.calculate_all_features(df)
        if not features:
            return {}
        
        latest = {}
        for key, values in features.items():
            if len(values) > 0 and not np.isnan(values[-1]):
                latest[key] = float(values[-1])
            else:
                latest[key] = 0.0
        
        return latest
    
    def detect_market_regime(self, df: pd.DataFrame) -> str:
        """检测市场状态"""
        features = self.calculate_all_features(df)
        if not features:
            return "UNKNOWN"
        
        adx = features['adx'][-1] if len(features['adx']) > 0 else 25
        bb_width = features['bb_width'][-1] if len(features['bb_width']) > 0 else 0.05
        trend = features['trend_strength'][-1] if len(features['trend_strength']) > 0 else 0
        
        if adx > 25:
            if trend > 0.3:
                return "TRENDING_UP"
            elif trend < -0.3:
                return "TRENDING_DOWN"
        
        if bb_width < 0.03:
            return "RANGING"
        
        if bb_width > 0.08:
            return "VOLATILE"
        
        return "UNKNOWN"
