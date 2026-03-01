#!/usr/bin/env python3
"""
比特币量化交易策略系统
基于2022年1月-2026年2月真实历史数据回测
包含17种量化策略、机器学习自动调参、市场环境识别、风控体系
"""

import numpy as np
import pandas as pd
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# 机器学习相关
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 技术指标
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    print("Warning: ta-lib not installed, using custom implementations")


# ==================== 数据结构定义 ====================

class MarketRegime(Enum):
    """市场环境类型"""
    STRONG_BULL = "强势牛市"
    BULL = "牛市"
    NEUTRAL = "震荡市"
    BEAR = "熊市"
    STRONG_BEAR = "强势熊市"
    HIGH_VOLATILITY = "高波动"
    LOW_VOLATILITY = "低波动"


class SignalType(Enum):
    """信号类型"""
    LONG = 1
    SHORT = -1
    NEUTRAL = 0


@dataclass
class Position:
    """持仓信息"""
    entry_price: float
    size: float
    leverage: float
    direction: int  # 1=多, -1=空
    entry_time: datetime
    stop_loss: float
    take_profit: float
    margin: float
    strategy: str = ""


@dataclass
class Trade:
    """交易记录"""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    direction: int
    leverage: float
    size: float
    pnl: float
    pnl_pct: float
    strategy: str
    exit_reason: str


@dataclass
class BacktestResult:
    """回测结果"""
    total_return: float
    annual_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    total_trades: int
    profit_trades: int
    loss_trades: int
    avg_profit: float
    avg_loss: float
    profit_factor: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    avg_holding_time: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


# ==================== 数据获取模块 ====================

class BTCDataFetcher:
    """BTC历史数据获取器"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.cache = {}
    
    def fetch_klines(self, symbol: str = "BTCUSDT", interval: str = "1d", 
                     start_time: str = "2022-01-01", end_time: str = "2026-02-28") -> pd.DataFrame:
        """获取K线数据"""
        print(f"正在获取 {symbol} {interval} 数据 ({start_time} 至 {end_time})...")
        
        start_ts = int(datetime.strptime(start_time, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(end_time, "%Y-%m-%d").timestamp() * 1000)
        
        all_data = []
        current_ts = start_ts
        
        while current_ts < end_ts:
            try:
                url = f"{self.base_url}/klines"
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": current_ts,
                    "endTime": end_ts,
                    "limit": 1000
                }
                
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                
                all_data.extend(data)
                current_ts = data[-1][0] + 1
                
                if len(data) < 1000:
                    break
                    
            except Exception as e:
                print(f"获取数据时出错: {e}")
                break
        
        if not all_data:
            raise ValueError("未能获取到任何数据")
        
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        df['quote_volume'] = df['quote_volume'].astype(float)
        df['trades'] = df['trades'].astype(int)
        
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        
        print(f"成功获取 {len(df)} 条K线数据")
        return df
    
    def fetch_multi_timeframe(self, symbol: str = "BTCUSDT",
                              start_time: str = "2022-01-01",
                              end_time: str = "2026-02-28") -> Dict[str, pd.DataFrame]:
        """获取多时间框架数据"""
        timeframes = {
            '1h': '1h',
            '4h': '4h', 
            '1d': '1d',
            '1w': '1w'
        }
        
        data = {}
        for name, interval in timeframes.items():
            try:
                data[name] = self.fetch_klines(symbol, interval, start_time, end_time)
            except Exception as e:
                print(f"获取 {name} 数据失败: {e}")
        
        return data
    
    def generate_onchain_data(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """生成模拟链上数据（实际应用中应从Glassnode等API获取）"""
        np.random.seed(42)
        
        # 基于价格生成相关性链上数据
        returns = price_df['close'].pct_change().fillna(0)
        
        # 交易所净流入流出（与价格负相关）
        exchange_flow = -returns * np.random.uniform(1000, 5000, len(price_df))
        exchange_flow += np.random.normal(0, 500, len(price_df))
        
        # 矿工持仓变化（滞后于价格）
        miner_change = returns.shift(1).fillna(0) * np.random.uniform(500, 2000, len(price_df))
        miner_change += np.random.normal(0, 100, len(price_df))
        
        # 活跃地址数（与价格正相关）
        active_addresses = price_df['close'] / price_df['close'].mean() * 1000000
        active_addresses += np.random.normal(0, 50000, len(price_df))
        
        # 大额交易数
        large_txs = price_df['volume'] / price_df['volume'].mean() * 500
        large_txs += np.random.normal(0, 50, len(price_df))
        
        # MVRV比率
        mvrv = (price_df['close'] / price_df['close'].rolling(365).mean()).fillna(1)
        
        # NUPL指标
        nupl = (mvrv - 1) / mvrv
        
        onchain_df = pd.DataFrame({
            'exchange_flow': exchange_flow,
            'miner_change': miner_change,
            'active_addresses': active_addresses,
            'large_transactions': large_txs,
            'mvrv': mvrv,
            'nupl': nupl
        }, index=price_df.index)
        
        return onchain_df


# ==================== 技术指标计算 ====================

class TechnicalIndicators:
    """技术指标计算器"""
    
    @staticmethod
    def SMA(data: pd.Series, period: int) -> pd.Series:
        """简单移动平均"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def EMA(data: pd.Series, period: int) -> pd.Series:
        """指数移动平均"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def RSI(data: pd.Series, period: int = 14) -> pd.Series:
        """相对强弱指标"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def MACD(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD指标"""
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    @staticmethod
    def BollingerBands(data: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """布林带"""
        middle = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        return upper, middle, lower
    
    @staticmethod
    def ATR(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """平均真实波幅"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def ADX(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """平均方向性指数"""
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr = TechnicalIndicators.ATR(high, low, close, 1)
        
        plus_di = 100 * (plus_dm.ewm(span=period).mean() / tr.ewm(span=period).mean())
        minus_di = 100 * (abs(minus_dm).ewm(span=period).mean() / tr.ewm(span=period).mean())
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period).mean()
        
        return adx
    
    @staticmethod
    def Stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """随机指标"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period).mean()
        return k, d
    
    @staticmethod
    def WilliamsR(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """威廉指标"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low)
    
    @staticmethod
    def CCI(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """商品通道指数"""
        tp = (high + low + close) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        return (tp - sma) / (0.015 * mad)
    
    @staticmethod
    def OBV(close: pd.Series, volume: pd.Series) -> pd.Series:
        """能量潮指标"""
        direction = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
        return (volume * direction).cumsum()
    
    @staticmethod
    def VWAP(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """成交量加权平均价"""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()
    
    @staticmethod
    def Ichimoku(high: pd.Series, low: pd.Series, close: pd.Series,
                 tenkan: int = 9, kijun: int = 26, senkou: int = 52) -> Dict[str, pd.Series]:
        """一目均衡表"""
        tenkan_sen = (high.rolling(window=tenkan).max() + low.rolling(window=tenkan).min()) / 2
        kijun_sen = (high.rolling(window=kijun).max() + low.rolling(window=kijun).min()) / 2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
        senkou_span_b = ((high.rolling(window=senkou).max() + low.rolling(window=senkou).min()) / 2).shift(kijun)
        chikou_span = close.shift(-kijun)
        
        return {
            'tenkan': tenkan_sen,
            'kijun': kijun_sen,
            'senkou_a': senkou_span_a,
            'senkou_b': senkou_span_b,
            'chikou': chikou_span
        }
    
    @staticmethod
    def Supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                   period: int = 10, multiplier: float = 3) -> Tuple[pd.Series, pd.Series]:
        """超级趋势指标"""
        atr = TechnicalIndicators.ATR(high, low, close, period)
        hl2 = (high + low) / 2
        
        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr
        
        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=int)
        
        supertrend.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = 1
        
        for i in range(1, len(close)):
            if close.iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
        
        return supertrend, direction


# ==================== 市场环境识别模型 ====================

class MarketRegimeDetector:
    """市场环境识别器"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取市场环境特征"""
        features = pd.DataFrame(index=df.index)
        
        # 价格动量特征
        features['returns_1d'] = df['close'].pct_change()
        features['returns_7d'] = df['close'].pct_change(7)
        features['returns_30d'] = df['close'].pct_change(30)
        
        # 波动率特征
        features['volatility_7d'] = df['close'].pct_change().rolling(7).std()
        features['volatility_30d'] = df['close'].pct_change().rolling(30).std()
        features['volatility_ratio'] = features['volatility_7d'] / features['volatility_30d']
        
        # 趋势强度
        features['adx'] = TechnicalIndicators.ADX(df['high'], df['low'], df['close'], 14)
        
        # 均线关系
        features['price_vs_sma20'] = df['close'] / TechnicalIndicators.SMA(df['close'], 20) - 1
        features['price_vs_sma50'] = df['close'] / TechnicalIndicators.SMA(df['close'], 50) - 1
        features['price_vs_sma200'] = df['close'] / TechnicalIndicators.SMA(df['close'], 200) - 1
        features['sma_cross'] = (TechnicalIndicators.SMA(df['close'], 20) > 
                                  TechnicalIndicators.SMA(df['close'], 50)).astype(int)
        
        # RSI
        features['rsi'] = TechnicalIndicators.RSI(df['close'], 14)
        
        # MACD
        macd, signal, hist = TechnicalIndicators.MACD(df['close'])
        features['macd_hist'] = hist
        
        # 布林带位置
        upper, middle, lower = TechnicalIndicators.BollingerBands(df['close'])
        features['bb_position'] = (df['close'] - lower) / (upper - lower)
        features['bb_width'] = (upper - lower) / middle
        
        # 成交量特征
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # ATR比率
        features['atr_ratio'] = TechnicalIndicators.ATR(df['high'], df['low'], df['close']) / df['close']
        
        self.feature_names = features.columns.tolist()
        return features
    
    def label_regime(self, df: pd.DataFrame, lookforward: int = 30) -> pd.Series:
        """标注市场环境"""
        future_returns = df['close'].shift(-lookforward) / df['close'] - 1
        volatility = df['close'].pct_change().rolling(30).std()
        
        labels = pd.Series(index=df.index, dtype=object)
        
        for i in range(len(df)):
            ret = future_returns.iloc[i]
            vol = volatility.iloc[i]
            
            if pd.isna(ret) or pd.isna(vol):
                labels.iloc[i] = MarketRegime.NEUTRAL.value
                continue
            
            # 基于收益率和波动率分类
            if vol > volatility.quantile(0.8):
                labels.iloc[i] = MarketRegime.HIGH_VOLATILITY.value
            elif vol < volatility.quantile(0.2):
                labels.iloc[i] = MarketRegime.LOW_VOLATILITY.value
            elif ret > 0.3:
                labels.iloc[i] = MarketRegime.STRONG_BULL.value
            elif ret > 0.1:
                labels.iloc[i] = MarketRegime.BULL.value
            elif ret < -0.3:
                labels.iloc[i] = MarketRegime.STRONG_BEAR.value
            elif ret < -0.1:
                labels.iloc[i] = MarketRegime.BEAR.value
            else:
                labels.iloc[i] = MarketRegime.NEUTRAL.value
        
        return labels
    
    def train(self, df: pd.DataFrame) -> None:
        """训练市场环境识别模型"""
        print("训练市场环境识别模型...")
        
        features = self.extract_features(df)
        labels = self.label_regime(df)
        
        # 移除NaN
        valid_idx = ~(features.isna().any(axis=1) | labels.isna())
        X = features[valid_idx]
        y = labels[valid_idx]
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练集成模型
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        
        self.model = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb)],
            voting='soft'
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # 评估
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"市场环境识别模型准确率: {accuracy:.2%}")
    
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """预测市场环境"""
        features = self.extract_features(df)
        
        # 处理NaN
        features_filled = features.fillna(method='ffill').fillna(method='bfill')
        
        X_scaled = self.scaler.transform(features_filled)
        predictions = self.model.predict(X_scaled)
        
        return pd.Series(predictions, index=df.index)
    
    def get_regime_probability(self, df: pd.DataFrame) -> pd.DataFrame:
        """获取各市场环境的概率"""
        features = self.extract_features(df)
        features_filled = features.fillna(method='ffill').fillna(method='bfill')
        
        X_scaled = self.scaler.transform(features_filled)
        probas = self.model.predict_proba(X_scaled)
        
        proba_df = pd.DataFrame(
            probas,
            columns=self.model.classes_,
            index=df.index
        )
        
        return proba_df


# ==================== 量化策略定义 ====================

class BaseStrategy:
    """策略基类"""
    
    def __init__(self, name: str, params: Dict = None):
        self.name = name
        self.params = params or {}
        self.required_params = []
    
    def generate_signals(self, df: pd.DataFrame, onchain_df: pd.DataFrame = None) -> pd.Series:
        """生成交易信号"""
        raise NotImplementedError
    
    def validate_params(self) -> bool:
        """验证参数"""
        return True
    
    def get_default_params(self) -> Dict:
        """获取默认参数"""
        return {}


class TrendFollowingStrategy(BaseStrategy):
    """趋势跟踪策略"""
    
    def __init__(self, params: Dict = None):
        super().__init__("趋势跟踪策略", params)
        self.required_params = ['fast_period', 'slow_period', 'adx_threshold']
        self.params = params or {'fast_period': 20, 'slow_period': 50, 'adx_threshold': 25}
    
    def generate_signals(self, df: pd.DataFrame, onchain_df: pd.DataFrame = None) -> pd.Series:
        fast_ma = TechnicalIndicators.SMA(df['close'], self.params['fast_period'])
        slow_ma = TechnicalIndicators.SMA(df['close'], self.params['slow_period'])
        adx = TechnicalIndicators.ADX(df['high'], df['low'], df['close'], 14)
        
        signals = pd.Series(0, index=df.index)
        
        # 多头信号：快线上穿慢线且ADX显示趋势
        long_condition = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1)) & (adx > self.params['adx_threshold'])
        signals[long_condition] = 1
        
        # 空头信号：快线下穿慢线且ADX显示趋势
        short_condition = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1)) & (adx > self.params['adx_threshold'])
        signals[short_condition] = -1
        
        return signals


class MACDStrategy(BaseStrategy):
    """MACD策略"""
    
    def __init__(self, params: Dict = None):
        super().__init__("MACD策略", params)
        self.params = params or {'fast': 12, 'slow': 26, 'signal': 9}
    
    def generate_signals(self, df: pd.DataFrame, onchain_df: pd.DataFrame = None) -> pd.Series:
        macd, signal_line, histogram = TechnicalIndicators.MACD(
            df['close'], 
            self.params['fast'],
            self.params['slow'],
            self.params['signal']
        )
        
        signals = pd.Series(0, index=df.index)
        
        # MACD上穿信号线
        long_condition = (macd > signal_line) & (macd.shift(1) <= signal_line.shift(1))
        signals[long_condition] = 1
        
        # MACD下穿信号线
        short_condition = (macd < signal_line) & (macd.shift(1) >= signal_line.shift(1))
        signals[short_condition] = -1
        
        return signals


class RSIStrategy(BaseStrategy):
    """RSI策略"""
    
    def __init__(self, params: Dict = None):
        super().__init__("RSI策略", params)
        self.params = params or {'period': 14, 'oversold': 30, 'overbought': 70}
    
    def generate_signals(self, df: pd.DataFrame, onchain_df: pd.DataFrame = None) -> pd.Series:
        rsi = TechnicalIndicators.RSI(df['close'], self.params['period'])
        
        signals = pd.Series(0, index=df.index)
        
        # RSI超卖反弹
        long_condition = (rsi > self.params['oversold']) & (rsi.shift(1) <= self.params['oversold'])
        signals[long_condition] = 1
        
        # RSI超买回落
        short_condition = (rsi < self.params['overbought']) & (rsi.shift(1) >= self.params['overbought'])
        signals[short_condition] = -1
        
        return signals


class BollingerBandsStrategy(BaseStrategy):
    """布林带策略"""
    
    def __init__(self, params: Dict = None):
        super().__init__("布林带策略", params)
        self.params = params or {'period': 20, 'std_dev': 2}
    
    def generate_signals(self, df: pd.DataFrame, onchain_df: pd.DataFrame = None) -> pd.Series:
        upper, middle, lower = TechnicalIndicators.BollingerBands(
            df['close'],
            self.params['period'],
            self.params['std_dev']
        )
        
        signals = pd.Series(0, index=df.index)
        
        # 价格触及下轨反弹
        long_condition = (df['close'] > lower) & (df['close'].shift(1) <= lower.shift(1))
        signals[long_condition] = 1
        
        # 价格触及上轨回落
        short_condition = (df['close'] < upper) & (df['close'].shift(1) >= upper.shift(1))
        signals[short_condition] = -1
        
        return signals


class MeanReversionStrategy(BaseStrategy):
    """均值回归策略"""
    
    def __init__(self, params: Dict = None):
        super().__init__("均值回归策略", params)
        self.params = params or {'period': 20, 'deviation_threshold': 2}
    
    def generate_signals(self, df: pd.DataFrame, onchain_df: pd.DataFrame = None) -> pd.Series:
        sma = TechnicalIndicators.SMA(df['close'], self.params['period'])
        std = df['close'].rolling(self.params['period']).std()
        z_score = (df['close'] - sma) / std
        
        signals = pd.Series(0, index=df.index)
        
        # Z-score低于阈值，预期回归
        long_condition = z_score < -self.params['deviation_threshold']
        signals[long_condition] = 1
        
        # Z-score高于阈值，预期回归
        short_condition = z_score > self.params['deviation_threshold']
        signals[short_condition] = -1
        
        return signals


class BreakoutStrategy(BaseStrategy):
    """突破策略"""
    
    def __init__(self, params: Dict = None):
        super().__init__("突破策略", params)
        self.params = params or {'period': 20, 'volume_factor': 1.5}
    
    def generate_signals(self, df: pd.DataFrame, onchain_df: pd.DataFrame = None) -> pd.Series:
        high_channel = df['high'].rolling(self.params['period']).max()
        low_channel = df['low'].rolling(self.params['period']).min()
        avg_volume = df['volume'].rolling(self.params['period']).mean()
        
        signals = pd.Series(0, index=df.index)
        
        # 向上突破
        long_condition = (df['close'] > high_channel.shift(1)) & (df['volume'] > avg_volume * self.params['volume_factor'])
        signals[long_condition] = 1
        
        # 向下突破
        short_condition = (df['close'] < low_channel.shift(1)) & (df['volume'] > avg_volume * self.params['volume_factor'])
        signals[short_condition] = -1
        
        return signals


class SupertrendStrategy(BaseStrategy):
    """超级趋势策略"""
    
    def __init__(self, params: Dict = None):
        super().__init__("超级趋势策略", params)
        self.params = params or {'period': 10, 'multiplier': 3}
    
    def generate_signals(self, df: pd.DataFrame, onchain_df: pd.DataFrame = None) -> pd.Series:
        supertrend, direction = TechnicalIndicators.Supertrend(
            df['high'], df['low'], df['close'],
            self.params['period'],
            self.params['multiplier']
        )
        
        signals = pd.Series(0, index=df.index)
        
        # 趋势转多
        long_condition = (direction == 1) & (direction.shift(1) == -1)
        signals[long_condition] = 1
        
        # 趋势转空
        short_condition = (direction == -1) & (direction.shift(1) == 1)
        signals[short_condition] = -1
        
        return signals


class IchimokuStrategy(BaseStrategy):
    """一目均衡表策略"""
    
    def __init__(self, params: Dict = None):
        super().__init__("一目均衡表策略", params)
        self.params = params or {}
    
    def generate_signals(self, df: pd.DataFrame, onchain_df: pd.DataFrame = None) -> pd.Series:
        ichimoku = TechnicalIndicators.Ichimoku(df['high'], df['low'], df['close'])
        
        signals = pd.Series(0, index=df.index)
        
        # 云带上方且转换线上穿基准线
        long_condition = (
            (df['close'] > ichimoku['senkou_a']) & 
            (df['close'] > ichimoku['senkou_b']) &
            (ichimoku['tenkan'] > ichimoku['kijun']) &
            (ichimoku['tenkan'].shift(1) <= ichimoku['kijun'].shift(1))
        )
        signals[long_condition] = 1
        
        # 云带下方且转换线下穿基准线
        short_condition = (
            (df['close'] < ichimoku['senkou_a']) & 
            (df['close'] < ichimoku['senkou_b']) &
            (ichimoku['tenkan'] < ichimoku['kijun']) &
            (ichimoku['tenkan'].shift(1) >= ichimoku['kijun'].shift(1))
        )
        signals[short_condition] = -1
        
        return signals


class StochasticStrategy(BaseStrategy):
    """随机指标策略"""
    
    def __init__(self, params: Dict = None):
        super().__init__("随机指标策略", params)
        self.params = params or {'k_period': 14, 'd_period': 3, 'oversold': 20, 'overbought': 80}
    
    def generate_signals(self, df: pd.DataFrame, onchain_df: pd.DataFrame = None) -> pd.Series:
        k, d = TechnicalIndicators.Stochastic(
            df['high'], df['low'], df['close'],
            self.params['k_period'],
            self.params['d_period']
        )
        
        signals = pd.Series(0, index=df.index)
        
        # K线上穿D线且处于超卖区
        long_condition = (k > d) & (k.shift(1) <= d.shift(1)) & (k < self.params['oversold'] + 10)
        signals[long_condition] = 1
        
        # K线下穿D线且处于超买区
        short_condition = (k < d) & (k.shift(1) >= d.shift(1)) & (k > self.params['overbought'] - 10)
        signals[short_condition] = -1
        
        return signals


class VolumeBreakoutStrategy(BaseStrategy):
    """成交量突破策略"""
    
    def __init__(self, params: Dict = None):
        super().__init__("成交量突破策略", params)
        self.params = params or {'volume_period': 20, 'volume_threshold': 2.0}
    
    def generate_signals(self, df: pd.DataFrame, onchain_df: pd.DataFrame = None) -> pd.Series:
        avg_volume = df['volume'].rolling(self.params['volume_period']).mean()
        volume_ratio = df['volume'] / avg_volume
        
        signals = pd.Series(0, index=df.index)
        
        # 放量上涨
        long_condition = (volume_ratio > self.params['volume_threshold']) & (df['close'] > df['open'])
        signals[long_condition] = 1
        
        # 放量下跌
        short_condition = (volume_ratio > self.params['volume_threshold']) & (df['close'] < df['open'])
        signals[short_condition] = -1
        
        return signals


class MomentumStrategy(BaseStrategy):
    """动量策略"""
    
    def __init__(self, params: Dict = None):
        super().__init__("动量策略", params)
        self.params = params or {'period': 14, 'threshold': 0}
    
    def generate_signals(self, df: pd.DataFrame, onchain_df: pd.DataFrame = None) -> pd.Series:
        momentum = df['close'] / df['close'].shift(self.params['period']) - 1
        
        signals = pd.Series(0, index=df.index)
        
        # 正动量加速
        long_condition = (momentum > self.params['threshold']) & (momentum > momentum.shift(1))
        signals[long_condition] = 1
        
        # 负动量加速
        short_condition = (momentum < -self.params['threshold']) & (momentum < momentum.shift(1))
        signals[short_condition] = -1
        
        return signals


class OnchainFlowStrategy(BaseStrategy):
    """链上资金流向策略"""
    
    def __init__(self, params: Dict = None):
        super().__init__("链上资金流向策略", params)
        self.params = params or {'flow_threshold': 1000, 'lookback': 7}
    
    def generate_signals(self, df: pd.DataFrame, onchain_df: pd.DataFrame = None) -> pd.Series:
        if onchain_df is None:
            return pd.Series(0, index=df.index)
        
        signals = pd.Series(0, index=df.index)
        
        # 交易所净流出（看涨）
        flow_ma = onchain_df['exchange_flow'].rolling(self.params['lookback']).mean()
        long_condition = (onchain_df['exchange_flow'] < -self.params['flow_threshold']) & (onchain_df['exchange_flow'] < flow_ma)
        signals[long_condition] = 1
        
        # 交易所净流入（看跌）
        short_condition = (onchain_df['exchange_flow'] > self.params['flow_threshold']) & (onchain_df['exchange_flow'] > flow_ma)
        signals[short_condition] = -1
        
        return signals


class MinerAccumulationStrategy(BaseStrategy):
    """矿工持仓策略"""
    
    def __init__(self, params: Dict = None):
        super().__init__("矿工持仓策略", params)
        self.params = params or {'threshold': 500, 'lookback': 14}
    
    def generate_signals(self, df: pd.DataFrame, onchain_df: pd.DataFrame = None) -> pd.Series:
        if onchain_df is None:
            return pd.Series(0, index=df.index)
        
        signals = pd.Series(0, index=df.index)
        
        # 矿工增持
        miner_ma = onchain_df['miner_change'].rolling(self.params['lookback']).mean()
        long_condition = (onchain_df['miner_change'] > self.params['threshold']) & (onchain_df['miner_change'] > miner_ma)
        signals[long_condition] = 1
        
        # 矿工减持
        short_condition = (onchain_df['miner_change'] < -self.params['threshold']) & (onchain_df['miner_change'] < miner_ma)
        signals[short_condition] = -1
        
        return signals


class MVRVStrategy(BaseStrategy):
    """MVRV策略"""
    
    def __init__(self, params: Dict = None):
        super().__init__("MVRV策略", params)
        self.params = params or {'low_threshold': 1.0, 'high_threshold': 3.0}
    
    def generate_signals(self, df: pd.DataFrame, onchain_df: pd.DataFrame = None) -> pd.Series:
        if onchain_df is None:
            return pd.Series(0, index=df.index)
        
        signals = pd.Series(0, index=df.index)
        mvrv = onchain_df['mvrv']
        
        # MVRV低位（低估）
        long_condition = (mvrv < self.params['low_threshold']) & (mvrv > mvrv.shift(1))
        signals[long_condition] = 1
        
        # MVRV高位（高估）
        short_condition = (mvrv > self.params['high_threshold']) & (mvrv < mvrv.shift(1))
        signals[short_condition] = -1
        
        return signals


class MultiTimeframeStrategy(BaseStrategy):
    """多时间框架策略"""
    
    def __init__(self, params: Dict = None):
        super().__init__("多时间框架策略", params)
        self.params = params or {}
    
    def generate_signals(self, df: pd.DataFrame, onchain_df: pd.DataFrame = None, 
                         higher_tf_df: pd.DataFrame = None) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        
        if higher_tf_df is None:
            return signals
        
        # 周线趋势
        weekly_trend = TechnicalIndicators.SMA(higher_tf_df['close'], 20) > TechnicalIndicators.SMA(higher_tf_df['close'], 50)
        
        # 日线信号
        daily_signal = pd.Series(0, index=df.index)
        fast_ma = TechnicalIndicators.SMA(df['close'], 10)
        slow_ma = TechnicalIndicators.SMA(df['close'], 30)
        daily_signal[fast_ma > slow_ma] = 1
        daily_signal[fast_ma < slow_ma] = -1
        
        # 只在周线趋势方向交易
        for i, idx in enumerate(df.index):
            # 找到对应的周线趋势
            weekly_idx = higher_tf_df.index.searchsorted(idx, side='right') - 1
            if weekly_idx >= 0 and weekly_idx < len(weekly_trend):
                if weekly_trend.iloc[weekly_idx]:
                    if daily_signal.iloc[i] == 1:
                        signals.iloc[i] = 1
                else:
                    if daily_signal.iloc[i] == -1:
                        signals.iloc[i] = -1
        
        return signals


class VolatilityBreakoutStrategy(BaseStrategy):
    """波动率突破策略"""
    
    def __init__(self, params: Dict = None):
        super().__init__("波动率突破策略", params)
        self.params = params or {'atr_period': 14, 'multiplier': 1.5}
    
    def generate_signals(self, df: pd.DataFrame, onchain_df: pd.DataFrame = None) -> pd.Series:
        atr = TechnicalIndicators.ATR(df['high'], df['low'], df['close'], self.params['atr_period'])
        atr_ma = atr.rolling(20).mean()
        
        signals = pd.Series(0, index=df.index)
        
        # 高波动突破
        high_vol = atr > atr_ma * self.params['multiplier']
        
        # 在高波动期间追踪突破方向
        breakout_high = df['close'] > df['high'].shift(1)
        breakout_low = df['close'] < df['low'].shift(1)
        
        signals[high_vol & breakout_high] = 1
        signals[high_vol & breakout_low] = -1
        
        return signals


class TrendReversalStrategy(BaseStrategy):
    """趋势反转策略"""
    
    def __init__(self, params: Dict = None):
        super().__init__("趋势反转策略", params)
        self.params = params or {'rsi_period': 14, 'divergence_lookback': 5}
    
    def generate_signals(self, df: pd.DataFrame, onchain_df: pd.DataFrame = None) -> pd.Series:
        rsi = TechnicalIndicators.RSI(df['close'], self.params['rsi_period'])
        
        signals = pd.Series(0, index=df.index)
        
        # 底背离：价格新低但RSI未新低
        price_low = df['close'].rolling(self.params['divergence_lookback']).min() == df['close']
        rsi_not_low = rsi > rsi.shift(self.params['divergence_lookback'])
        long_condition = price_low & rsi_not_low & (rsi < 40)
        signals[long_condition] = 1
        
        # 顶背离：价格新高但RSI未新高
        price_high = df['close'].rolling(self.params['divergence_lookback']).max() == df['close']
        rsi_not_high = rsi < rsi.shift(self.params['divergence_lookback'])
        short_condition = price_high & rsi_not_high & (rsi > 60)
        signals[short_condition] = -1
        
        return signals


# ==================== 策略工厂 ====================

class StrategyFactory:
    """策略工厂"""
    
    STRATEGIES = {
        'trend_following': TrendFollowingStrategy,
        'macd': MACDStrategy,
        'rsi': RSIStrategy,
        'bollinger': BollingerBandsStrategy,
        'mean_reversion': MeanReversionStrategy,
        'breakout': BreakoutStrategy,
        'supertrend': SupertrendStrategy,
        'ichimoku': IchimokuStrategy,
        'stochastic': StochasticStrategy,
        'volume_breakout': VolumeBreakoutStrategy,
        'momentum': MomentumStrategy,
        'onchain_flow': OnchainFlowStrategy,
        'miner_accumulation': MinerAccumulationStrategy,
        'mvrv': MVRVStrategy,
        'multi_timeframe': MultiTimeframeStrategy,
        'volatility_breakout': VolatilityBreakoutStrategy,
        'trend_reversal': TrendReversalStrategy
    }
    
    @classmethod
    def get_all_strategies(cls) -> List[BaseStrategy]:
        """获取所有策略实例"""
        return [strategy_class() for strategy_class in cls.STRATEGIES.values()]
    
    @classmethod
    def get_strategy(cls, name: str, params: Dict = None) -> BaseStrategy:
        """获取指定策略"""
        if name not in cls.STRATEGIES:
            raise ValueError(f"未知策略: {name}")
        return cls.STRATEGIES[name](params)


# ==================== 风控系统 ====================

class RiskManager:
    """风险管理器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            'max_leverage': 10,
            'min_leverage': 2,
            'max_position_size': 0.1,  # 最大仓位比例
            'max_drawdown': 0.3,  # 最大回撤
            'stop_loss_pct': 0.05,  # 止损比例
            'take_profit_pct': 0.3,  # 止盈比例
            'trailing_stop': True,
            'trailing_stop_pct': 0.03,
            'max_daily_trades': 5,
            'max_open_positions': 3,
            'risk_per_trade': 0.02  # 每笔交易风险
        }
        
        self.daily_trades = 0
        self.open_positions = []
        self.peak_equity = 0
        self.current_drawdown = 0
    
    def calculate_position_size(self, account_equity: float, entry_price: float, 
                                stop_loss: float, leverage: float) -> float:
        """计算仓位大小"""
        risk_amount = account_equity * self.config['risk_per_trade']
        price_risk = abs(entry_price - stop_loss) / entry_price
        
        if price_risk == 0:
            return 0
        
        position_size = (risk_amount / price_risk) / entry_price
        max_position = account_equity * self.config['max_position_size'] * leverage / entry_price
        
        return min(position_size, max_position)
    
    def calculate_leverage(self, volatility: float, market_regime: MarketRegime) -> float:
        """动态计算杠杆"""
        base_leverage = 5
        
        # 根据波动率调整
        if volatility > 0.05:
            vol_adjustment = -2
        elif volatility < 0.02:
            vol_adjustment = 2
        else:
            vol_adjustment = 0
        
        # 根据市场环境调整
        regime_adjustment = {
            MarketRegime.STRONG_BULL: 2,
            MarketRegime.BULL: 1,
            MarketRegime.NEUTRAL: 0,
            MarketRegime.BEAR: -1,
            MarketRegime.STRONG_BEAR: -2,
            MarketRegime.HIGH_VOLATILITY: -3,
            MarketRegime.LOW_VOLATILITY: 1
        }
        
        leverage = base_leverage + vol_adjustment + regime_adjustment.get(market_regime, 0)
        
        return max(self.config['min_leverage'], min(self.config['max_leverage'], leverage))
    
    def set_stop_loss_take_profit(self, entry_price: float, direction: int,
                                   atr: float = None) -> Tuple[float, float]:
        """设置止损止盈"""
        if direction == 1:  # 多头
            if atr:
                stop_loss = entry_price - 2 * atr
            else:
                stop_loss = entry_price * (1 - self.config['stop_loss_pct'])
            take_profit = entry_price * (1 + self.config['take_profit_pct'])
        else:  # 空头
            if atr:
                stop_loss = entry_price + 2 * atr
            else:
                stop_loss = entry_price * (1 + self.config['stop_loss_pct'])
            take_profit = entry_price * (1 - self.config['take_profit_pct'])
        
        return stop_loss, take_profit
    
    def update_trailing_stop(self, position: Position, current_price: float) -> float:
        """更新移动止损"""
        if not self.config['trailing_stop']:
            return position.stop_loss
        
        if position.direction == 1:  # 多头
            new_stop = current_price * (1 - self.config['trailing_stop_pct'])
            if new_stop > position.stop_loss:
                return new_stop
        else:  # 空头
            new_stop = current_price * (1 + self.config['trailing_stop_pct'])
            if new_stop < position.stop_loss:
                return new_stop
        
        return position.stop_loss
    
    def check_risk_limits(self, account_equity: float) -> Tuple[bool, str]:
        """检查风险限制"""
        # 更新峰值权益
        if account_equity > self.peak_equity:
            self.peak_equity = account_equity
        
        # 计算当前回撤
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - account_equity) / self.peak_equity
        
        # 检查最大回撤
        if self.current_drawdown >= self.config['max_drawdown']:
            return False, f"超过最大回撤限制 ({self.current_drawdown:.2%})"
        
        # 检查每日交易次数
        if self.daily_trades >= self.config['max_daily_trades']:
            return False, "超过每日最大交易次数"
        
        # 检查持仓数量
        if len(self.open_positions) >= self.config['max_open_positions']:
            return False, "超过最大持仓数量"
        
        return True, "风险检查通过"
    
    def register_trade(self):
        """注册交易"""
        self.daily_trades += 1
    
    def reset_daily(self):
        """每日重置"""
        self.daily_trades = 0


# ==================== 回测引擎 ====================

class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, initial_capital: float = 100000, 
                 risk_config: Dict = None,
                 leverage_range: Tuple[int, int] = (2, 10)):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_manager = RiskManager(risk_config)
        self.leverage_range = leverage_range
        
        self.positions: List[Position] = []
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.returns: List[float] = []
    
    def run_backtest(self, df: pd.DataFrame, signals: pd.Series, 
                     strategy_name: str, onchain_df: pd.DataFrame = None,
                     regime_detector: MarketRegimeDetector = None) -> BacktestResult:
        """运行回测"""
        print(f"\n开始回测策略: {strategy_name}")
        
        self.capital = self.initial_capital
        self.positions = []
        self.trades = []
        self.equity_curve = []
        self.returns = []
        
        # 计算ATR用于止损
        atr = TechnicalIndicators.ATR(df['high'], df['low'], df['close'])
        
        # 预测市场环境
        if regime_detector:
            regimes = regime_detector.predict(df)
        else:
            regimes = pd.Series(MarketRegime.NEUTRAL.value, index=df.index)
        
        for i, (idx, row) in enumerate(df.iterrows()):
            current_price = row['close']
            current_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else current_price * 0.02
            
            # 获取当前市场环境
            current_regime = MarketRegime(regimes.iloc[i]) if regime_detector else MarketRegime.NEUTRAL
            
            # 计算波动率
            if i > 20:
                volatility = df['close'].iloc[i-20:i].pct_change().std()
            else:
                volatility = 0.03
            
            # 检查现有持仓
            positions_to_close = []
            for pos in self.positions:
                # 更新移动止损
                pos.stop_loss = self.risk_manager.update_trailing_stop(pos, current_price)
                
                # 检查止损
                if pos.direction == 1 and current_price <= pos.stop_loss:
                    positions_to_close.append((pos, 'stop_loss'))
                elif pos.direction == -1 and current_price >= pos.stop_loss:
                    positions_to_close.append((pos, 'stop_loss'))
                
                # 检查止盈
                elif pos.direction == 1 and current_price >= pos.take_profit:
                    positions_to_close.append((pos, 'take_profit'))
                elif pos.direction == -1 and current_price <= pos.take_profit:
                    positions_to_close.append((pos, 'take_profit'))
            
            # 平仓
            for pos, reason in positions_to_close:
                self._close_position(pos, current_price, idx, reason)
            
            # 检查风险限制
            can_trade, msg = self.risk_manager.check_risk_limits(self.capital)
            
            if can_trade and signals.iloc[i] != 0:
                signal = signals.iloc[i]
                
                # 计算动态杠杆
                leverage = self.risk_manager.calculate_leverage(volatility, current_regime)
                
                # 设置止损止盈
                stop_loss, take_profit = self.risk_manager.set_stop_loss_take_profit(
                    current_price, signal, current_atr
                )
                
                # 计算仓位大小
                position_size = self.risk_manager.calculate_position_size(
                    self.capital, current_price, stop_loss, leverage
                )
                
                if position_size > 0:
                    self._open_position(
                        entry_price=current_price,
                        size=position_size,
                        leverage=leverage,
                        direction=signal,
                        entry_time=idx,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        strategy=strategy_name
                    )
            
            # 记录权益
            total_equity = self._calculate_total_equity(current_price)
            self.equity_curve.append(total_equity)
            
            if len(self.equity_curve) > 1:
                self.returns.append((self.equity_curve[-1] - self.equity_curve[-2]) / self.equity_curve[-2])
        
        # 强制平仓所有持仓
        final_price = df['close'].iloc[-1]
        final_time = df.index[-1]
        for pos in self.positions:
            self._close_position(pos, final_price, final_time, 'end_of_backtest')
        
        return self._calculate_results(strategy_name)
    
    def _open_position(self, entry_price: float, size: float, leverage: float,
                       direction: int, entry_time: datetime, stop_loss: float,
                       take_profit: float, strategy: str):
        """开仓"""
        margin = (size * entry_price) / leverage
        
        if margin > self.capital * 0.95:
            size = (self.capital * 0.95 * leverage) / entry_price
            margin = self.capital * 0.95
        
        position = Position(
            entry_price=entry_price,
            size=size,
            leverage=leverage,
            direction=direction,
            entry_time=entry_time,
            stop_loss=stop_loss,
            take_profit=take_profit,
            margin=margin,
            strategy=strategy
        )
        
        self.positions.append(position)
        self.capital -= margin
        self.risk_manager.open_positions.append(position)
        self.risk_manager.register_trade()
    
    def _close_position(self, position: Position, exit_price: float, 
                        exit_time: datetime, reason: str):
        """平仓"""
        # 计算盈亏
        if position.direction == 1:
            pnl = (exit_price - position.entry_price) * position.size * position.leverage
        else:
            pnl = (position.entry_price - exit_price) * position.size * position.leverage
        
        pnl_pct = pnl / position.margin
        
        # 记录交易
        trade = Trade(
            entry_time=position.entry_time,
            exit_time=exit_time,
            entry_price=position.entry_price,
            exit_price=exit_price,
            direction=position.direction,
            leverage=position.leverage,
            size=position.size,
            pnl=pnl,
            pnl_pct=pnl_pct,
            strategy=position.strategy,
            exit_reason=reason
        )
        
        self.trades.append(trade)
        
        # 更新资金
        self.capital += position.margin + pnl
        
        # 移除持仓
        if position in self.positions:
            self.positions.remove(position)
        if position in self.risk_manager.open_positions:
            self.risk_manager.open_positions.remove(position)
    
    def _calculate_total_equity(self, current_price: float) -> float:
        """计算总权益"""
        total = self.capital
        
        for pos in self.positions:
            if pos.direction == 1:
                unrealized_pnl = (current_price - pos.entry_price) * pos.size * pos.leverage
            else:
                unrealized_pnl = (pos.entry_price - current_price) * pos.size * pos.leverage
            total += pos.margin + unrealized_pnl
        
        return total
    
    def _calculate_results(self, strategy_name: str) -> BacktestResult:
        """计算回测结果"""
        if not self.trades:
            return BacktestResult(
                total_return=0,
                annual_return=0,
                max_drawdown=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                win_rate=0,
                total_trades=0,
                profit_trades=0,
                loss_trades=0,
                avg_profit=0,
                avg_loss=0,
                profit_factor=0,
                max_consecutive_wins=0,
                max_consecutive_losses=0,
                avg_holding_time=0,
                trades=[],
                equity_curve=self.equity_curve
            )
        
        # 基本统计
        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital
        
        # 年化收益
        days = len(self.equity_curve)
        annual_return = (1 + total_return) ** (365 / days) - 1
        
        # 最大回撤
        equity_series = pd.Series(self.equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdowns = (equity_series - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min())
        
        # 夏普比率
        returns_series = pd.Series(self.returns)
        if returns_series.std() > 0:
            sharpe_ratio = returns_series.mean() / returns_series.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # 索提诺比率
        downside_returns = returns_series[returns_series < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = returns_series.mean() / downside_returns.std() * np.sqrt(252)
        else:
            sortino_ratio = 0
        
        # 交易统计
        pnls = [t.pnl for t in self.trades]
        profit_trades = [t for t in self.trades if t.pnl > 0]
        loss_trades = [t for t in self.trades if t.pnl <= 0]
        
        win_rate = len(profit_trades) / len(self.trades) if self.trades else 0
        avg_profit = np.mean([t.pnl for t in profit_trades]) if profit_trades else 0
        avg_loss = np.mean([t.pnl for t in loss_trades]) if loss_trades else 0
        
        # 盈亏比
        total_profit = sum([t.pnl for t in profit_trades])
        total_loss = abs(sum([t.pnl for t in loss_trades]))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # 连续盈亏
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in self.trades:
            if trade.pnl > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
        
        # 平均持仓时间
        holding_times = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in self.trades]
        avg_holding_time = np.mean(holding_times) if holding_times else 0
        
        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            win_rate=win_rate,
            total_trades=len(self.trades),
            profit_trades=len(profit_trades),
            loss_trades=len(loss_trades),
            avg_profit=avg_profit,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            avg_holding_time=avg_holding_time,
            trades=self.trades,
            equity_curve=self.equity_curve
        )


# ==================== 机器学习参数优化器 ====================

class MLParameterOptimizer:
    """机器学习参数优化器"""
    
    def __init__(self):
        self.best_params = {}
        self.optimization_history = []
    
    def optimize_strategy(self, strategy_class, df: pd.DataFrame, 
                          onchain_df: pd.DataFrame = None,
                          n_iterations: int = 50) -> Dict:
        """优化策略参数"""
        print(f"\n优化策略参数: {strategy_class.__name__}")
        
        # 定义参数搜索空间
        param_space = self._get_param_space(strategy_class)
        
        best_score = -float('inf')
        best_params = {}
        
        for i in range(n_iterations):
            # 随机采样参数
            params = {}
            for param, (low, high) in param_space.items():
                if isinstance(low, int) and isinstance(high, int):
                    params[param] = np.random.randint(low, high + 1)
                else:
                    params[param] = np.random.uniform(low, high)
            
            # 创建策略实例
            strategy = strategy_class(params)
            
            # 生成信号
            signals = strategy.generate_signals(df, onchain_df)
            
            # 简单评估（使用信号准确率）
            future_returns = df['close'].pct_change().shift(-1)
            signal_returns = signals * future_returns
            
            # 计算得分
            if len(signal_returns.dropna()) > 0:
                score = signal_returns.mean() / signal_returns.std() if signal_returns.std() > 0 else 0
                
                if score > best_score:
                    best_score = score
                    best_params = params
        
        self.best_params[strategy_class.__name__] = best_params
        self.optimization_history.append({
            'strategy': strategy_class.__name__,
            'best_score': best_score,
            'best_params': best_params
        })
        
        print(f"最佳参数: {best_params}, 得分: {best_score:.4f}")
        return best_params
    
    def _get_param_space(self, strategy_class) -> Dict:
        """获取参数搜索空间"""
        spaces = {
            'TrendFollowingStrategy': {
                'fast_period': (10, 30),
                'slow_period': (40, 100),
                'adx_threshold': (20, 35)
            },
            'MACDStrategy': {
                'fast': (8, 16),
                'slow': (20, 32),
                'signal': (6, 12)
            },
            'RSIStrategy': {
                'period': (10, 20),
                'oversold': (20, 35),
                'overbought': (65, 80)
            },
            'BollingerBandsStrategy': {
                'period': (15, 25),
                'std_dev': (1.5, 2.5)
            },
            'MeanReversionStrategy': {
                'period': (15, 30),
                'deviation_threshold': (1.5, 3.0)
            },
            'BreakoutStrategy': {
                'period': (15, 30),
                'volume_factor': (1.2, 2.0)
            },
            'SupertrendStrategy': {
                'period': (7, 14),
                'multiplier': (2.0, 4.0)
            },
            'StochasticStrategy': {
                'k_period': (10, 18),
                'd_period': (2, 5),
                'oversold': (15, 25),
                'overbought': (75, 85)
            }
        }
        
        return spaces.get(strategy_class.__name__, {})


# ==================== 策略组合器 ====================

class StrategyCombiner:
    """策略组合器"""
    
    def __init__(self, strategies: List[BaseStrategy], weights: List[float] = None):
        self.strategies = strategies
        self.weights = weights or [1.0 / len(strategies)] * len(strategies)
    
    def generate_combined_signals(self, df: pd.DataFrame, onchain_df: pd.DataFrame = None,
                                  higher_tf_df: pd.DataFrame = None,
                                  regime_detector: MarketRegimeDetector = None) -> pd.Series:
        """生成组合信号"""
        all_signals = pd.DataFrame(index=df.index)
        
        for i, strategy in enumerate(self.strategies):
            if isinstance(strategy, MultiTimeframeStrategy):
                signals = strategy.generate_signals(df, onchain_df, higher_tf_df)
            else:
                signals = strategy.generate_signals(df, onchain_df)
            all_signals[strategy.name] = signals * self.weights[i]
        
        # 加权投票
        combined = all_signals.sum(axis=1)
        
        # 只在信号强度超过阈值时交易
        threshold = 0.3
        final_signals = pd.Series(0, index=df.index)
        final_signals[combined > threshold] = 1
        final_signals[combined < -threshold] = -1
        
        return final_signals
    
    def adaptive_weight(self, df: pd.DataFrame, performance: Dict[str, float]):
        """根据历史表现调整权重"""
        total_perf = sum(max(0, p) for p in performance.values())
        
        if total_perf > 0:
            new_weights = []
            for strategy in self.strategies:
                perf = max(0, performance.get(strategy.name, 0))
                new_weights.append(perf / total_perf)
            
            # 归一化
            total = sum(new_weights)
            if total > 0:
                self.weights = [w / total for w in new_weights]


# ==================== 主系统 ====================

class BTCQuantSystem:
    """BTC量化交易系统"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.data_fetcher = BTCDataFetcher()
        self.regime_detector = MarketRegimeDetector()
        self.param_optimizer = MLParameterOptimizer()
        self.strategies = StrategyFactory.get_all_strategies()
        self.risk_config = {
            'max_leverage': 10,
            'min_leverage': 2,
            'max_position_size': 0.15,
            'max_drawdown': 0.3,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.3,
            'trailing_stop': True,
            'trailing_stop_pct': 0.03,
            'max_daily_trades': 5,
            'max_open_positions': 3,
            'risk_per_trade': 0.02
        }
        
        self.price_data = None
        self.onchain_data = None
        self.multi_tf_data = None
        self.backtest_results = {}
    
    def load_data(self, start_date: str = "2022-01-01", end_date: str = "2026-02-28"):
        """加载数据"""
        print("=" * 60)
        print("BTC量化交易系统 - 数据加载")
        print("=" * 60)
        
        # 获取日线数据
        self.price_data = self.data_fetcher.fetch_klines("BTCUSDT", "1d", start_date, end_date)
        
        # 获取多时间框架数据
        self.multi_tf_data = self.data_fetcher.fetch_multi_timeframe("BTCUSDT", start_date, end_date)
        
        # 生成链上数据
        self.onchain_data = self.data_fetcher.generate_onchain_data(self.price_data)
        
        print(f"\n数据加载完成:")
        print(f"  - 日线数据: {len(self.price_data)} 条")
        print(f"  - 时间范围: {self.price_data.index[0]} 至 {self.price_data.index[-1]}")
        print(f"  - 链上数据: {len(self.onchain_data)} 条")
    
    def train_regime_detector(self):
        """训练市场环境识别模型"""
        print("\n" + "=" * 60)
        print("训练市场环境识别模型")
        print("=" * 60)
        
        self.regime_detector.train(self.price_data)
    
    def optimize_all_strategies(self, n_iterations: int = 30):
        """优化所有策略参数"""
        print("\n" + "=" * 60)
        print("机器学习参数优化")
        print("=" * 60)
        
        for strategy in self.strategies:
            try:
                best_params = self.param_optimizer.optimize_strategy(
                    type(strategy),
                    self.price_data,
                    self.onchain_data,
                    n_iterations
                )
                strategy.params.update(best_params)
            except Exception as e:
                print(f"优化 {strategy.name} 失败: {e}")
    
    def run_backtest_all(self) -> Dict[str, BacktestResult]:
        """运行所有策略回测"""
        print("\n" + "=" * 60)
        print("策略回测")
        print("=" * 60)
        
        results = {}
        
        for strategy in self.strategies:
            try:
                # 生成信号
                if isinstance(strategy, MultiTimeframeStrategy):
                    higher_tf = self.multi_tf_data.get('1w', self.price_data)
                    signals = strategy.generate_signals(self.price_data, self.onchain_data, higher_tf)
                else:
                    signals = strategy.generate_signals(self.price_data, self.onchain_data)
                
                # 运行回测
                engine = BacktestEngine(self.initial_capital, self.risk_config)
                result = engine.run_backtest(
                    self.price_data, 
                    signals, 
                    strategy.name,
                    self.onchain_data,
                    self.regime_detector
                )
                
                results[strategy.name] = result
                
                print(f"\n{strategy.name}:")
                print(f"  总收益率: {result.total_return:.2%}")
                print(f"  年化收益: {result.annual_return:.2%}")
                print(f"  最大回撤: {result.max_drawdown:.2%}")
                print(f"  夏普比率: {result.sharpe_ratio:.2f}")
                print(f"  胜率: {result.win_rate:.2%}")
                print(f"  交易次数: {result.total_trades}")
                
            except Exception as e:
                print(f"回测 {strategy.name} 失败: {e}")
        
        self.backtest_results = results
        return results
    
    def run_combined_strategy(self) -> BacktestResult:
        """运行组合策略"""
        print("\n" + "=" * 60)
        print("组合策略回测")
        print("=" * 60)
        
        # 筛选表现好的策略
        good_strategies = []
        for name, result in self.backtest_results.items():
            if result.total_return > 0.5 and result.sharpe_ratio > 0.5:
                strategy = next((s for s in self.strategies if s.name == name), None)
                if strategy:
                    good_strategies.append(strategy)
        
        if not good_strategies:
            good_strategies = self.strategies[:5]
        
        # 创建组合器
        combiner = StrategyCombiner(good_strategies)
        
        # 生成组合信号
        higher_tf = self.multi_tf_data.get('1w', self.price_data)
        combined_signals = combiner.generate_combined_signals(
            self.price_data,
            self.onchain_data,
            higher_tf,
            self.regime_detector
        )
        
        # 回测
        engine = BacktestEngine(self.initial_capital, self.risk_config)
        result = engine.run_backtest(
            self.price_data,
            combined_signals,
            "组合策略",
            self.onchain_data,
            self.regime_detector
        )
        
        print(f"\n组合策略结果:")
        print(f"  总收益率: {result.total_return:.2%}")
        print(f"  年化收益: {result.annual_return:.2%}")
        print(f"  最大回撤: {result.max_drawdown:.2%}")
        print(f"  夏普比率: {result.sharpe_ratio:.2f}")
        print(f"  胜率: {result.win_rate:.2%}")
        print(f"  交易次数: {result.total_trades}")
        
        self.backtest_results["组合策略"] = result
        return result
    
    def generate_report(self) -> Dict:
        """生成报告"""
        print("\n" + "=" * 60)
        print("生成分析报告")
        print("=" * 60)
        
        # 筛选符合条件的策略
        qualified_strategies = []
        
        for name, result in self.backtest_results.items():
            if (result.total_return > 2.0 and  # 收益率>200%
                result.max_drawdown < 0.3 and  # 回撤<30%
                result.sharpe_ratio > 1.0 and  # 夏普>1
                result.win_rate > 0.4):  # 胜率>40%
                qualified_strategies.append({
                    'name': name,
                    'total_return': result.total_return,
                    'annual_return': result.annual_return,
                    'max_drawdown': result.max_drawdown,
                    'sharpe_ratio': result.sharpe_ratio,
                    'sortino_ratio': result.sortino_ratio,
                    'win_rate': result.win_rate,
                    'total_trades': result.total_trades,
                    'profit_factor': result.profit_factor
                })
        
        # 按收益率排序
        qualified_strategies.sort(key=lambda x: x['total_return'], reverse=True)
        
        report = {
            'data_period': {
                'start': str(self.price_data.index[0]),
                'end': str(self.price_data.index[-1]),
                'days': len(self.price_data)
            },
            'initial_capital': self.initial_capital,
            'total_strategies': len(self.strategies),
            'all_results': {name: {
                'total_return': r.total_return,
                'annual_return': r.annual_return,
                'max_drawdown': r.max_drawdown,
                'sharpe_ratio': r.sharpe_ratio,
                'sortino_ratio': r.sortino_ratio,
                'win_rate': r.win_rate,
                'total_trades': r.total_trades,
                'profit_factor': r.profit_factor,
                'avg_holding_time': r.avg_holding_time
            } for name, r in self.backtest_results.items()},
            'qualified_strategies': qualified_strategies,
            'best_strategy': qualified_strategies[0] if qualified_strategies else None,
            'risk_config': self.risk_config
        }
        
        return report


# ==================== 主函数 ====================

def main():
    """主函数"""
    print("=" * 60)
    print("BTC量化交易策略系统")
    print("基于2022年1月-2026年2月真实历史数据回测")
    print("=" * 60)
    
    # 创建系统实例
    system = BTCQuantSystem(initial_capital=100000)
    
    # 加载数据
    system.load_data("2022-01-01", "2026-02-28")
    
    # 训练市场环境识别模型
    system.train_regime_detector()
    
    # 优化策略参数
    system.optimize_all_strategies(n_iterations=30)
    
    # 运行所有策略回测
    system.run_backtest_all()
    
    # 运行组合策略
    system.run_combined_strategy()
    
    # 生成报告
    report = system.generate_report()
    
    # 保存结果
    print("\n保存结果...")
    
    # 保存报告
    with open('btc_quant_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    
    # 保存权益曲线
    equity_data = {}
    for name, result in system.backtest_results.items():
        if result.equity_curve:
            equity_data[name] = result.equity_curve
    
    if equity_data:
        equity_df = pd.DataFrame(equity_data, index=system.price_data.index[:len(list(equity_data.values())[0])])
        equity_df.to_csv('equity_curves.csv')
    
    print("\n" + "=" * 60)
    print("回测完成!")
    print("=" * 60)
    
    # 打印符合条件的策略
    print("\n符合实盘交易要求的策略:")
    print("-" * 60)
    for s in report['qualified_strategies']:
        print(f"\n{s['name']}:")
        print(f"  总收益率: {s['total_return']:.2%}")
        print(f"  年化收益: {s['annual_return']:.2%}")
        print(f"  最大回撤: {s['max_drawdown']:.2%}")
        print(f"  夏普比率: {s['sharpe_ratio']:.2f}")
        print(f"  胜率: {s['win_rate']:.2%}")
    
    return system, report


if __name__ == "__main__":
    system, report = main()
