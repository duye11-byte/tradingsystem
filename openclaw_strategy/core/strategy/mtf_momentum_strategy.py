"""
OpenClaw 推理层 - 多时间框架趋势动量策略
第2层: 推理层

策略逻辑:
1. 趋势确认: EMA交叉 + ADX > 25
2. 动量确认: RSI在合理区间 + MACD同向
3. 波动率过滤: ATR计算仓位和止损
4. 多时间框架: 1H趋势 + 15M入场
5. 情绪过滤: 恐惧贪婪指数
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import uuid
import logging

from ..strategy.strategy_types import (
    Signal, SignalDirection, MarketRegime, MarketContext,
    TechnicalFeatures, StrategyConfig, StrategyState
)
from ..features.technical_indicators import FeatureEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MTFMomentumStrategy:
    """
    多时间框架趋势动量策略
    
    核心逻辑:
    - 大时间框架(1H)确定趋势方向
    - 中时间框架(15M)寻找入场点
    - 小时间框架(5M)精确执行
    - 多因子评分系统筛选高质量信号
    """
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()
        self.feature_engine = FeatureEngine()
        self.state = StrategyState.IDLE
        self.active_signals: List[Signal] = []
        self.signal_history: List[Signal] = []
        
        # 统计
        self.stats = {
            'total_signals': 0,
            'long_signals': 0,
            'short_signals': 0,
            'filtered_by_sentiment': 0,
            'filtered_by_score': 0
        }
    
    def analyze_market(self, 
                       df_1h: pd.DataFrame, 
                       df_15m: pd.DataFrame, 
                       df_5m: Optional[pd.DataFrame] = None,
                       sentiment_data: Optional[Dict] = None) -> MarketContext:
        """分析市场环境"""
        
        symbol = df_15m['symbol'].iloc[-1] if 'symbol' in df_15m.columns else "UNKNOWN"
        current_price = df_15m['close'].iloc[-1]
        
        context = MarketContext(
            symbol=symbol,
            current_price=current_price,
            timestamp=datetime.now()
        )
        
        # 计算各时间框架特征
        context.features_1h = self._extract_features(df_1h)
        context.features_15m = self._extract_features(df_15m)
        if df_5m is not None:
            context.features_5m = self._extract_features(df_5m)
        
        # 市场情绪
        if sentiment_data:
            context.fear_greed_index = sentiment_data.get('fear_greed', 50)
            context.funding_rate = sentiment_data.get('funding_rate', 0.0)
            context.long_short_ratio = sentiment_data.get('long_short_ratio', 1.0)
        
        # 检测市场状态
        context.regime = self._detect_regime(context)
        context.trend_strength = self._calculate_trend_strength(context)
        
        return context
    
    def _extract_features(self, df: pd.DataFrame) -> TechnicalFeatures:
        """提取技术特征"""
        features = TechnicalFeatures()
        
        if len(df) < 50:
            return features
        
        latest = self.feature_engine.get_latest_features(df)
        
        features.ema_9 = latest.get('ema_9', 0)
        features.ema_21 = latest.get('ema_21', 0)
        features.ema_50 = latest.get('ema_50', 0)
        features.ema_200 = latest.get('ema_200', 0)
        
        features.adx = latest.get('adx', 0)
        features.adx_di_plus = latest.get('di_plus', 0)
        features.adx_di_minus = latest.get('di_minus', 0)
        
        features.rsi = latest.get('rsi', 50)
        features.rsi_ema = latest.get('rsi_ema', 50)
        features.macd_line = latest.get('macd', 0)
        features.macd_signal = latest.get('macd_signal', 0)
        features.macd_histogram = latest.get('macd_histogram', 0)
        
        features.atr = latest.get('atr', 0)
        features.atr_percent = latest.get('atr_percent', 0)
        features.bollinger_upper = latest.get('bb_upper', 0)
        features.bollinger_lower = latest.get('bb_lower', 0)
        features.bollinger_width = latest.get('bb_width', 0)
        
        features.volume_sma = latest.get('volume_sma', 0)
        features.volume_ratio = latest.get('volume_ratio', 1)
        features.obv = latest.get('obv', 0)
        
        features.support_level = latest.get('support', 0)
        features.resistance_level = latest.get('resistance', 0)
        
        features.timestamp = datetime.now()
        
        return features
    
    def _detect_regime(self, context: MarketContext) -> MarketRegime:
        """检测市场状态"""
        f_1h = context.features_1h
        
        # 使用1H时间框架判断趋势
        if f_1h.adx > self.config.adx_threshold:
            if f_1h.ema_9 > f_1h.ema_21 > f_1h.ema_50:
                return MarketRegime.TRENDING_UP
            elif f_1h.ema_9 < f_1h.ema_21 < f_1h.ema_50:
                return MarketRegime.TRENDING_DOWN
        
        # 震荡区间
        if f_1h.bollinger_width < 0.03:
            return MarketRegime.RANGING
        
        # 高波动
        if f_1h.atr_percent > 5:
            return MarketRegime.VOLATILE
        
        return MarketRegime.UNKNOWN
    
    def _calculate_trend_strength(self, context: MarketContext) -> float:
        """计算趋势强度"""
        f_1h = context.features_1h
        
        strength = 0.0
        
        # EMA排列
        if f_1h.ema_9 > f_1h.ema_21 > f_1h.ema_50:
            strength += 0.3
        elif f_1h.ema_9 < f_1h.ema_21 < f_1h.ema_50:
            strength -= 0.3
        
        # ADX
        if f_1h.adx > 25:
            strength += 0.2 if f_1h.adx_di_plus > f_1h.adx_di_minus else -0.2
        
        # MACD
        if f_1h.macd_histogram > 0:
            strength += 0.2
        else:
            strength -= 0.2
        
        # 价格相对EMA位置
        current_price = context.current_price
        if current_price > f_1h.ema_21:
            strength += 0.1
        else:
            strength -= 0.1
        
        return np.clip(strength, -1, 1)
    
    def generate_signal(self, context: MarketContext) -> Optional[Signal]:
        """生成交易信号"""
        
        self.stats['total_signals'] += 1
        
        # 情绪过滤
        if self.config.use_sentiment_filter:
            if context.fear_greed_index < self.config.fear_greed_threshold:
                self.stats['filtered_by_sentiment'] += 1
                logger.info(f"信号被过滤: 恐惧贪婪指数 {context.fear_greed_index} < {self.config.fear_greed_threshold}")
                return None
        
        # 判断信号方向
        direction = self._determine_direction(context)
        
        if direction == SignalDirection.NEUTRAL:
            return None
        
        # 计算信号评分
        score, reasons = self._calculate_signal_score(context, direction)
        
        if score < self.config.min_score:
            self.stats['filtered_by_score'] += 1
            logger.info(f"信号被过滤: 评分 {score:.2f} < {self.config.min_score}")
            return None
        
        # 计算入场和止损止盈价格
        entry, stop_loss, tp1, tp2, tp3 = self._calculate_prices(context, direction)
        
        # 计算置信度
        confidence = min(100, score * 10 + 20)
        
        # 创建信号
        signal = Signal(
            id=str(uuid.uuid4())[:8],
            symbol=context.symbol,
            direction=direction,
            timestamp=datetime.now(),
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            confidence=confidence,
            score=score,
            timeframe=self.config.primary_timeframe,
            strategy_name=self.config.name,
            reasons=reasons
        )
        
        # 更新统计
        if direction == SignalDirection.LONG:
            self.stats['long_signals'] += 1
        else:
            self.stats['short_signals'] += 1
        
        self.active_signals.append(signal)
        self.signal_history.append(signal)
        
        logger.info(f"生成信号: {direction.name} {context.symbol} 评分:{score:.2f} 置信度:{confidence:.1f}%")
        
        return signal
    
    def _determine_direction(self, context: MarketContext) -> SignalDirection:
        """确定信号方向"""
        f_1h = context.features_1h
        f_15m = context.features_15m
        
        # 多头条件
        long_conditions = 0
        
        # 1H趋势向上
        if f_1h.ema_9 > f_1h.ema_21 and f_1h.adx > 25 and f_1h.adx_di_plus > f_1h.adx_di_minus:
            long_conditions += 2
        
        # 15M动量向上
        if f_15m.macd_histogram > 0 and f_15m.macd_line > f_15m.macd_signal:
            long_conditions += 1
        
        # RSI不在超买区
        if 30 < f_15m.rsi < 65:
            long_conditions += 1
        
        # 价格在EMA上方
        if context.current_price > f_15m.ema_21:
            long_conditions += 1
        
        # 成交量确认
        if f_15m.volume_ratio > 1.2:
            long_conditions += 1
        
        # 空头条件
        short_conditions = 0
        
        # 1H趋势向下
        if f_1h.ema_9 < f_1h.ema_21 and f_1h.adx > 25 and f_1h.adx_di_plus < f_1h.adx_di_minus:
            short_conditions += 2
        
        # 15M动量向下
        if f_15m.macd_histogram < 0 and f_15m.macd_line < f_15m.macd_signal:
            short_conditions += 1
        
        # RSI不在超卖区
        if 35 < f_15m.rsi < 70:
            short_conditions += 1
        
        # 价格在EMA下方
        if context.current_price < f_15m.ema_21:
            short_conditions += 1
        
        # 成交量确认
        if f_15m.volume_ratio > 1.2:
            short_conditions += 1
        
        # 判断方向
        if long_conditions >= 4 and long_conditions > short_conditions:
            return SignalDirection.LONG
        elif short_conditions >= 4 and short_conditions > long_conditions:
            return SignalDirection.SHORT
        
        return SignalDirection.NEUTRAL
    
    def _calculate_signal_score(self, context: MarketContext, 
                                direction: SignalDirection) -> Tuple[float, List[str]]:
        """计算信号评分 (0-10)"""
        score = 0.0
        reasons = []
        
        f_1h = context.features_1h
        f_15m = context.features_15m
        
        # 趋势评分 (0-2.5)
        if direction == SignalDirection.LONG:
            if f_1h.ema_9 > f_1h.ema_21 > f_1h.ema_50:
                score += 2.5
                reasons.append("强上升趋势 (EMA多头排列)")
            elif f_1h.ema_9 > f_1h.ema_21:
                score += 1.5
                reasons.append("中等上升趋势")
        else:
            if f_1h.ema_9 < f_1h.ema_21 < f_1h.ema_50:
                score += 2.5
                reasons.append("强下降趋势 (EMA空头排列)")
            elif f_1h.ema_9 < f_1h.ema_21:
                score += 1.5
                reasons.append("中等下降趋势")
        
        # ADX评分 (0-2)
        if f_1h.adx > 30:
            score += 2.0
            reasons.append(f"强趋势 (ADX={f_1h.adx:.1f})")
        elif f_1h.adx > 25:
            score += 1.5
            reasons.append(f"中等趋势 (ADX={f_1h.adx:.1f})")
        elif f_1h.adx > 20:
            score += 0.5
            reasons.append(f"弱趋势 (ADX={f_1h.adx:.1f})")
        
        # 动量评分 (0-2)
        if direction == SignalDirection.LONG:
            if f_15m.macd_histogram > 0 and f_15m.macd_line > 0:
                score += 2.0
                reasons.append("强多头动量 (MACD)")
            elif f_15m.macd_histogram > 0:
                score += 1.0
                reasons.append("多头动量")
        else:
            if f_15m.macd_histogram < 0 and f_15m.macd_line < 0:
                score += 2.0
                reasons.append("强空头动量 (MACD)")
            elif f_15m.macd_histogram < 0:
                score += 1.0
                reasons.append("空头动量")
        
        # RSI评分 (0-1.5)
        if 40 <= f_15m.rsi <= 60:
            score += 1.5
            reasons.append(f"RSI中性区域 ({f_15m.rsi:.1f})")
        elif direction == SignalDirection.LONG and 30 < f_15m.rsi < 50:
            score += 1.0
            reasons.append(f"RSI低位反弹 ({f_15m.rsi:.1f})")
        elif direction == SignalDirection.SHORT and 50 < f_15m.rsi < 70:
            score += 1.0
            reasons.append(f"RSI高位回落 ({f_15m.rsi:.1f})")
        
        # 成交量评分 (0-1)
        if f_15m.volume_ratio > 1.5:
            score += 1.0
            reasons.append(f"成交量放大 ({f_15m.volume_ratio:.2f}x)")
        elif f_15m.volume_ratio > 1.0:
            score += 0.5
            reasons.append(f"正常成交量 ({f_15m.volume_ratio:.2f}x)")
        
        # 支撑阻力评分 (0-1)
        if direction == SignalDirection.LONG:
            if abs(context.current_price - f_15m.support_level) / context.current_price < 0.01:
                score += 1.0
                reasons.append("价格接近支撑位")
        else:
            if abs(context.current_price - f_15m.resistance_level) / context.current_price < 0.01:
                score += 1.0
                reasons.append("价格接近阻力位")
        
        return score, reasons
    
    def _calculate_prices(self, context: MarketContext, 
                          direction: SignalDirection) -> Tuple[float, float, float, float, float]:
        """计算入场、止损和止盈价格"""
        f_15m = context.features_15m
        current_price = context.current_price
        atr = f_15m.atr
        
        if direction == SignalDirection.LONG:
            entry = current_price
            stop_loss = entry - atr * self.config.atr_multiplier_sl
            tp1 = entry + atr * self.config.atr_multiplier_tp1
            tp2 = entry + atr * self.config.atr_multiplier_tp2
            tp3 = entry + atr * self.config.atr_multiplier_tp3
        else:
            entry = current_price
            stop_loss = entry + atr * self.config.atr_multiplier_sl
            tp1 = entry - atr * self.config.atr_multiplier_tp1
            tp2 = entry - atr * self.config.atr_multiplier_tp2
            tp3 = entry - atr * self.config.atr_multiplier_tp3
        
        return entry, stop_loss, tp1, tp2, tp3
    
    def check_exit_conditions(self, position: Dict, context: MarketContext) -> Optional[SignalDirection]:
        """检查出场条件"""
        current_price = context.current_price
        entry_price = position.get('entry_price', 0)
        direction = position.get('direction', '')
        stop_loss = position.get('stop_loss', 0)
        take_profit = position.get('take_profit', 0)
        
        if direction == 'LONG':
            # 止损
            if current_price <= stop_loss:
                return SignalDirection.CLOSE_LONG
            # 止盈
            if take_profit > 0 and current_price >= take_profit:
                return SignalDirection.CLOSE_LONG
            # 趋势反转
            f_15m = context.features_15m
            if f_15m.ema_9 < f_15m.ema_21 and f_15m.macd_histogram < 0:
                return SignalDirection.CLOSE_LONG
                
        elif direction == 'SHORT':
            # 止损
            if current_price >= stop_loss:
                return SignalDirection.CLOSE_SHORT
            # 止盈
            if take_profit > 0 and current_price <= take_profit:
                return SignalDirection.CLOSE_SHORT
            # 趋势反转
            f_15m = context.features_15m
            if f_15m.ema_9 > f_15m.ema_21 and f_15m.macd_histogram > 0:
                return SignalDirection.CLOSE_SHORT
        
        return None
    
    def get_stats(self) -> Dict:
        """获取策略统计"""
        return {
            **self.stats,
            'active_signals': len(self.active_signals),
            'signal_history': len(self.signal_history),
            'state': self.state.name
        }
    
    def reset(self):
        """重置策略状态"""
        self.state = StrategyState.IDLE
        self.active_signals = []
        self.signal_history = []
        self.stats = {
            'total_signals': 0,
            'long_signals': 0,
            'short_signals': 0,
            'filtered_by_sentiment': 0,
            'filtered_by_score': 0
        }
