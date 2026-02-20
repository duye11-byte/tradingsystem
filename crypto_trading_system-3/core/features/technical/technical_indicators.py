"""
技术指标计算模块
实现各种常用技术分析指标的计算
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

from ..feature_types import TechnicalFeatures

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    技术指标计算器
    
    提供各种技术分析指标的计算，包括：
    - 趋势指标: SMA, EMA
    - 动量指标: RSI, MACD, Stochastic
    - 波动率指标: Bollinger Bands, ATR
    - 成交量指标: OBV, MFI
    """
    
    def __init__(self):
        """初始化技术指标计算器"""
        pass
    
    def calculate_all(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        high_col: str = 'high',
        low_col: str = 'low',
        volume_col: str = 'volume'
    ) -> TechnicalFeatures:
        """
        计算所有技术指标
        
        Args:
            df: DataFrame 包含 OHLCV 数据
            price_col: 价格列名
            high_col: 最高价列名
            low_col: 最低价列名
            volume_col: 成交量列名
            
        Returns:
            TechnicalFeatures: 所有技术指标
        """
        features = TechnicalFeatures()
        
        # 确保数据足够
        if len(df) < 50:
            logger.warning(f"数据点不足 ({len(df)} < 50)，部分指标可能不准确")
        
        # 价格数据
        close = df[price_col].values
        high = df[high_col].values
        low = df[low_col].values
        volume = df[volume_col].values if volume_col in df.columns else None
        
        # 趋势指标
        features.sma_20 = self._get_last(self.sma(close, 20))
        features.sma_50 = self._get_last(self.sma(close, 50))
        features.sma_200 = self._get_last(self.sma(close, 200)) if len(close) >= 200 else features.sma_50
        features.ema_12 = self._get_last(self.ema(close, 12))
        features.ema_26 = self._get_last(self.ema(close, 26))
        features.ema_50 = self._get_last(self.ema(close, 50))
        
        # 动量指标
        features.rsi_14 = self._get_last(self.rsi(close, 14))
        features.rsi_7 = self._get_last(self.rsi(close, 7))
        
        macd_line, signal_line, histogram = self.macd(close)
        features.macd = self._get_last(macd_line)
        features.macd_signal = self._get_last(signal_line)
        features.macd_histogram = self._get_last(histogram)
        
        k, d = self.stochastic(high, low, close)
        features.stochastic_k = self._get_last(k)
        features.stochastic_d = self._get_last(d)
        
        features.williams_r = self._get_last(self.williams_r(high, low, close))
        features.cci_20 = self._get_last(self.cci(high, low, close, 20))
        
        # 波动率指标
        bb_upper, bb_middle, bb_lower = self.bollinger_bands(close)
        features.bb_upper = self._get_last(bb_upper)
        features.bb_middle = self._get_last(bb_middle)
        features.bb_lower = self._get_last(bb_lower)
        features.bb_width = self._get_last(self.bb_width(bb_upper, bb_lower, bb_middle))
        features.bb_percent = self._get_last(self.bb_percent(close, bb_upper, bb_lower))
        
        features.atr_14 = self._get_last(self.atr(high, low, close, 14))
        features.atr_7 = self._get_last(self.atr(high, low, close, 7))
        
        # 成交量指标
        if volume is not None:
            features.obv = self._get_last(self.obv(close, volume))
            features.volume_sma_20 = self._get_last(self.sma(volume, 20))
            features.volume_ratio = volume[-1] / features.volume_sma_20 if features.volume_sma_20 > 0 else 1.0
            features.mfi_14 = self._get_last(self.mfi(high, low, close, volume, 14))
        
        # 价格变化
        features.price_change_1h = self._calculate_price_change(close, 1)
        features.price_change_4h = self._calculate_price_change(close, 4)
        features.price_change_1d = self._calculate_price_change(close, 24)
        features.price_change_7d = self._calculate_price_change(close, 24 * 7)
        
        # ADX
        adx, plus_di, minus_di = self.adx(high, low, close, 14)
        features.adx_14 = self._get_last(adx)
        features.plus_di = self._get_last(plus_di)
        features.minus_di = self._get_last(minus_di)
        
        return features
    
    def _get_last(self, arr: np.ndarray) -> float:
        """获取数组最后一个值"""
        return float(arr[-1]) if len(arr) > 0 else 0.0
    
    def _calculate_price_change(self, close: np.ndarray, periods: int) -> float:
        """计算价格变化百分比"""
        if len(close) <= periods:
            return 0.0
        return (close[-1] - close[-periods-1]) / close[-periods-1] * 100
    
    # ==================== 趋势指标 ====================
    
    def sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        简单移动平均线 (Simple Moving Average)
        
        Args:
            data: 价格数据
            period: 周期
            
        Returns:
            np.ndarray: SMA 值
        """
        if len(data) < period:
            return np.array([np.nan] * len(data))
        
        weights = np.ones(period) / period
        sma = np.convolve(data, weights, mode='valid')
        # 补齐前面
        sma = np.concatenate([np.full(period - 1, np.nan), sma])
        return sma
    
    def ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        指数移动平均线 (Exponential Moving Average)
        
        Args:
            data: 价格数据
            period: 周期
            
        Returns:
            np.ndarray: EMA 值
        """
        if len(data) < period:
            return np.array([np.nan] * len(data))
        
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data, dtype=float)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    # ==================== 动量指标 ====================
    
    def rsi(self, close: np.ndarray, period: int = 14) -> np.ndarray:
        """
        相对强弱指数 (Relative Strength Index)
        
        Args:
            close: 收盘价
            period: 周期
            
        Returns:
            np.ndarray: RSI 值 (0-100)
        """
        if len(close) < period + 1:
            return np.array([50.0] * len(close))
        
        # 计算价格变化
        delta = np.diff(close)
        
        # 分离涨跌
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        # 计算平均涨跌
        avg_gains = np.concatenate([
            [np.nan] * period,
            [np.mean(gains[:period])]
        ])
        avg_losses = np.concatenate([
            [np.nan] * period,
            [np.mean(losses[:period])]
        ])
        
        for i in range(period + 1, len(gains) + 1):
            avg_gains = np.append(avg_gains, 
                (avg_gains[-1] * (period - 1) + gains[i-1]) / period)
            avg_losses = np.append(avg_losses,
                (avg_losses[-1] * (period - 1) + losses[i-1]) / period)
        
        # 计算 RS 和 RSI
        rs = avg_gains[period:] / (avg_losses[period:] + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # 补齐前面
        rsi = np.concatenate([[50.0] * period, rsi])
        
        return rsi
    
    def macd(
        self, 
        close: np.ndarray, 
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        MACD 指标
        
        Args:
            close: 收盘价
            fast_period: 快线周期
            slow_period: 慢线周期
            signal_period: 信号线周期
            
        Returns:
            Tuple: (MACD线, 信号线, 柱状图)
        """
        ema_fast = self.ema(close, fast_period)
        ema_slow = self.ema(close, slow_period)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def stochastic(
        self, 
        high: np.ndarray, 
        low: np.ndarray, 
        close: np.ndarray, 
        k_period: int = 14, 
        d_period: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        随机指标 (Stochastic Oscillator)
        
        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            k_period: %K 周期
            d_period: %D 周期
            
        Returns:
            Tuple: (%K, %D)
        """
        if len(close) < k_period:
            return np.array([50.0] * len(close)), np.array([50.0] * len(close))
        
        k_values = []
        for i in range(len(close)):
            if i < k_period - 1:
                k_values.append(50.0)
            else:
                highest_high = np.max(high[i-k_period+1:i+1])
                lowest_low = np.min(low[i-k_period+1:i+1])
                
                if highest_high == lowest_low:
                    k_values.append(50.0)
                else:
                    k = 100 * (close[i] - lowest_low) / (highest_high - lowest_low)
                    k_values.append(k)
        
        k = np.array(k_values)
        d = self.sma(k, d_period)
        
        return k, d
    
    def williams_r(
        self, 
        high: np.ndarray, 
        low: np.ndarray, 
        close: np.ndarray, 
        period: int = 14
    ) -> np.ndarray:
        """
        威廉指标 (Williams %R)
        
        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            period: 周期
            
        Returns:
            np.ndarray: Williams %R 值 (-100 to 0)
        """
        if len(close) < period:
            return np.array([-50.0] * len(close))
        
        wr_values = []
        for i in range(len(close)):
            if i < period - 1:
                wr_values.append(-50.0)
            else:
                highest_high = np.max(high[i-period+1:i+1])
                lowest_low = np.min(low[i-period+1:i+1])
                
                if highest_high == lowest_low:
                    wr_values.append(-50.0)
                else:
                    wr = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
                    wr_values.append(wr)
        
        return np.array(wr_values)
    
    def cci(
        self, 
        high: np.ndarray, 
        low: np.ndarray, 
        close: np.ndarray, 
        period: int = 20
    ) -> np.ndarray:
        """
        商品通道指数 (Commodity Channel Index)
        
        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            period: 周期
            
        Returns:
            np.ndarray: CCI 值
        """
        if len(close) < period:
            return np.array([0.0] * len(close))
        
        # 典型价格
        tp = (high + low + close) / 3
        
        # SMA of TP
        tp_sma = self.sma(tp, period)
        
        # 平均绝对偏差
        mad = np.array([np.nan] * len(tp))
        for i in range(period - 1, len(tp)):
            mad[i] = np.mean(np.abs(tp[i-period+1:i+1] - tp_sma[i]))
        
        # CCI
        cci = (tp - tp_sma) / (0.015 * mad + 1e-10)
        cci[:period-1] = 0
        
        return cci
    
    # ==================== 波动率指标 ====================
    
    def bollinger_bands(
        self, 
        close: np.ndarray, 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        布林带 (Bollinger Bands)
        
        Args:
            close: 收盘价
            period: 周期
            std_dev: 标准差倍数
            
        Returns:
            Tuple: (上轨, 中轨, 下轨)
        """
        middle = self.sma(close, period)
        
        # 计算标准差
        std = np.array([np.nan] * len(close))
        for i in range(period - 1, len(close)):
            std[i] = np.std(close[i-period+1:i+1])
        
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        
        return upper, middle, lower
    
    def bb_width(
        self, 
        upper: np.ndarray, 
        lower: np.ndarray, 
        middle: np.ndarray
    ) -> np.ndarray:
        """
        布林带宽度
        
        Args:
            upper: 上轨
            lower: 下轨
            middle: 中轨
            
        Returns:
            np.ndarray: 布林带宽度
        """
        width = (upper - lower) / (middle + 1e-10)
        return width
    
    def bb_percent(
        self, 
        close: np.ndarray, 
        upper: np.ndarray, 
        lower: np.ndarray
    ) -> np.ndarray:
        """
        %B 指标 (价格相对于布林带的位置)
        
        Args:
            close: 收盘价
            upper: 上轨
            lower: 下轨
            
        Returns:
            np.ndarray: %B 值 (0-1)
        """
        bb_range = upper - lower
        percent = (close - lower) / (bb_range + 1e-10)
        return np.clip(percent, 0, 1)
    
    def atr(
        self, 
        high: np.ndarray, 
        low: np.ndarray, 
        close: np.ndarray, 
        period: int = 14
    ) -> np.ndarray:
        """
        平均真实波幅 (Average True Range)
        
        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            period: 周期
            
        Returns:
            np.ndarray: ATR 值
        """
        if len(close) < 2:
            return np.array([0.0] * len(close))
        
        # 真实波幅
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.concatenate([[tr[0]], tr])
        
        # ATR
        atr = self.ema(tr, period)
        
        return atr
    
    # ==================== 成交量指标 ====================
    
    def obv(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """
        能量潮指标 (On-Balance Volume)
        
        Args:
            close: 收盘价
            volume: 成交量
            
        Returns:
            np.ndarray: OBV 值
        """
        if len(close) < 2:
            return np.array([0.0] * len(close))
        
        obv = np.zeros_like(close, dtype=float)
        obv[0] = volume[0]
        
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        
        return obv
    
    def mfi(
        self, 
        high: np.ndarray, 
        low: np.ndarray, 
        close: np.ndarray, 
        volume: np.ndarray, 
        period: int = 14
    ) -> np.ndarray:
        """
        资金流量指标 (Money Flow Index)
        
        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            volume: 成交量
            period: 周期
            
        Returns:
            np.ndarray: MFI 值 (0-100)
        """
        if len(close) < period + 1:
            return np.array([50.0] * len(close))
        
        # 典型价格
        tp = (high + low + close) / 3
        
        # 原始资金流量
        raw_mf = tp * volume
        
        # 资金流量变化
        mf_change = np.diff(tp)
        
        # 正向和负向资金流量
        pos_mf = np.where(mf_change > 0, raw_mf[1:], 0)
        neg_mf = np.where(mf_change < 0, raw_mf[1:], 0)
        
        mfi_values = [50.0]
        
        for i in range(period, len(pos_mf) + 1):
            pos_sum = np.sum(pos_mf[i-period:i])
            neg_sum = np.sum(neg_mf[i-period:i])
            
            if neg_sum == 0:
                mfi = 100.0
            else:
                mr = pos_sum / neg_sum
                mfi = 100 - (100 / (1 + mr))
            
            mfi_values.append(mfi)
        
        # 补齐前面
        mfi_values = [50.0] * (len(close) - len(mfi_values)) + mfi_values
        
        return np.array(mfi_values)
    
    # ==================== 趋势强度指标 ====================
    
    def adx(
        self, 
        high: np.ndarray, 
        low: np.ndarray, 
        close: np.ndarray, 
        period: int = 14
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        平均趋向指数 (Average Directional Index)
        
        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            period: 周期
            
        Returns:
            Tuple: (ADX, +DI, -DI)
        """
        if len(close) < period + 1:
            return (
                np.array([25.0] * len(close)),
                np.array([25.0] * len(close)),
                np.array([25.0] * len(close))
            )
        
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.concatenate([[tr[0]], tr])
        
        # +DM 和 -DM
        plus_dm = high[1:] - high[:-1]
        minus_dm = low[:-1] - low[1:]
        
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
        
        plus_dm = np.concatenate([[0], plus_dm])
        minus_dm = np.concatenate([[0], minus_dm])
        
        # 平滑
        atr = self.ema(tr, period)
        plus_di = 100 * self.ema(plus_dm, period) / (atr + 1e-10)
        minus_di = 100 * self.ema(minus_dm, period) / (atr + 1e-10)
        
        # DX 和 ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = self.ema(dx, period)
        
        return adx, plus_di, minus_di
    
    # ==================== 辅助方法 ====================
    
    def normalize(self, data: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        归一化数据
        
        Args:
            data: 输入数据
            method: 归一化方法 ('minmax', 'zscore', 'robust')
            
        Returns:
            np.ndarray: 归一化后的数据
        """
        if method == 'minmax':
            min_val = np.min(data)
            max_val = np.max(data)
            if max_val == min_val:
                return np.zeros_like(data)
            return (data - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return np.zeros_like(data)
            return (data - mean) / std
        
        elif method == 'robust':
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            if mad == 0:
                return np.zeros_like(data)
            return (data - median) / (1.4826 * mad)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
