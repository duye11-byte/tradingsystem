"""
OpenClaw 策略系统 - 类型定义
多时间框架趋势动量策略 (MTF-Momentum Strategy)
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Callable, Any, Tuple
from datetime import datetime
import numpy as np


class SignalDirection(Enum):
    """信号方向"""
    LONG = auto()
    SHORT = auto()
    NEUTRAL = auto()
    CLOSE_LONG = auto()
    CLOSE_SHORT = auto()


class MarketRegime(Enum):
    """市场状态"""
    TRENDING_UP = auto()
    TRENDING_DOWN = auto()
    RANGING = auto()
    VOLATILE = auto()
    UNKNOWN = auto()


class StrategyState(Enum):
    """策略状态"""
    IDLE = auto()
    IN_LONG = auto()
    IN_SHORT = auto()
    PENDING_ENTRY = auto()
    PENDING_EXIT = auto()


@dataclass
class TimeframeConfig:
    """时间框架配置"""
    name: str
    interval: str
    minutes: int
    weight: float = 1.0


@dataclass
class IndicatorValue:
    """指标值"""
    name: str
    value: float
    timestamp: datetime
    timeframe: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TechnicalFeatures:
    """技术特征集合"""
    # 趋势指标
    ema_9: float = 0.0
    ema_21: float = 0.0
    ema_50: float = 0.0
    ema_200: float = 0.0
    adx: float = 0.0
    adx_di_plus: float = 0.0
    adx_di_minus: float = 0.0
    
    # 动量指标
    rsi: float = 50.0
    rsi_ema: float = 50.0
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    
    # 波动率指标
    atr: float = 0.0
    atr_percent: float = 0.0
    bollinger_upper: float = 0.0
    bollinger_lower: float = 0.0
    bollinger_width: float = 0.0
    
    # 成交量指标
    volume_sma: float = 0.0
    volume_ratio: float = 1.0
    obv: float = 0.0
    
    # 支撑阻力
    pivot_high: float = 0.0
    pivot_low: float = 0.0
    support_level: float = 0.0
    resistance_level: float = 0.0
    
    # 多时间框架确认
    higher_tf_trend: int = 0  # 1=up, -1=down, 0=neutral
    higher_tf_strength: float = 0.0
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MarketContext:
    """市场环境上下文"""
    symbol: str
    current_price: float
    timestamp: datetime
    
    # 各时间框架特征
    features_1h: TechnicalFeatures = field(default_factory=TechnicalFeatures)
    features_15m: TechnicalFeatures = field(default_factory=TechnicalFeatures)
    features_5m: TechnicalFeatures = field(default_factory=TechnicalFeatures)
    
    # 市场情绪
    fear_greed_index: int = 50
    funding_rate: float = 0.0
    long_short_ratio: float = 1.0
    
    # 市场状态
    regime: MarketRegime = MarketRegime.UNKNOWN
    volatility_percentile: float = 50.0
    trend_strength: float = 0.0
    
    # 链上数据
    exchange_inflow: float = 0.0
    exchange_outflow: float = 0.0
    whale_activity: float = 0.0


@dataclass
class Signal:
    """交易信号"""
    id: str
    symbol: str
    direction: SignalDirection
    timestamp: datetime
    
    # 价格信息
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit_1: float = 0.0
    take_profit_2: float = 0.0
    take_profit_3: float = 0.0
    
    # 信号质量
    confidence: float = 0.0  # 0-100
    score: float = 0.0  # 综合评分
    
    # 信号来源
    timeframe: str = ""
    strategy_name: str = ""
    
    # 理由
    reasons: List[str] = field(default_factory=list)
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """检查信号是否有效"""
        return self.confidence >= 60 and self.score >= 5.0


@dataclass
class Position:
    """持仓信息"""
    id: str
    symbol: str
    direction: SignalDirection
    entry_price: float
    size: float
    leverage: float = 1.0
    
    entry_time: datetime = field(default_factory=datetime.now)
    
    # 止损止盈
    stop_loss: float = 0.0
    take_profit_1: float = 0.0
    take_profit_2: float = 0.0
    take_profit_3: float = 0.0
    
    # 部分止盈跟踪
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    
    # 动态止损
    trailing_stop: float = 0.0
    trailing_activated: bool = False
    highest_price: float = 0.0
    lowest_price: float = float('inf')
    
    # 盈亏
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    def update_pnl(self, current_price: float) -> float:
        """更新未实现盈亏"""
        if self.direction == SignalDirection.LONG:
            self.unrealized_pnl = (current_price - self.entry_price) * self.size
            self.highest_price = max(self.highest_price, current_price)
            self.lowest_price = min(self.lowest_price, current_price)
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.size
            self.highest_price = max(self.highest_price, self.entry_price * 2 - current_price)
            self.lowest_price = min(self.lowest_price, self.entry_price * 2 - current_price)
        return self.unrealized_pnl
    
    def pnl_percent(self) -> float:
        """盈亏百分比"""
        if self.entry_price == 0:
            return 0.0
        if self.direction == SignalDirection.LONG:
            return (self.current_price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - self.current_price) / self.entry_price * 100
    
    @property
    def current_price(self) -> float:
        """获取当前价格（用于计算）"""
        if self.direction == SignalDirection.LONG:
            return self.entry_price + self.unrealized_pnl / self.size if self.size != 0 else self.entry_price
        else:
            return self.entry_price - self.unrealized_pnl / self.size if self.size != 0 else self.entry_price


@dataclass
class StrategyConfig:
    """策略配置"""
    name: str = "MTF_Momentum_Strategy"
    version: str = "1.0.0"
    
    # 交易对
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    
    # 时间框架
    primary_timeframe: str = "15m"
    confirmation_timeframe: str = "1h"
    entry_timeframe: str = "5m"
    
    # 趋势参数
    ema_fast: int = 9
    ema_slow: int = 21
    ema_trend: int = 50
    adx_period: int = 14
    adx_threshold: float = 25.0
    
    # 动量参数
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # 波动率参数
    atr_period: int = 14
    atr_multiplier_sl: float = 2.0
    atr_multiplier_tp1: float = 2.5
    atr_multiplier_tp2: float = 4.0
    atr_multiplier_tp3: float = 6.0
    
    # 信号阈值
    min_confidence: float = 65.0
    min_score: float = 6.0
    
    # 风险管理
    max_position_size: float = 0.1  # 账户的10%
    max_leverage: float = 5.0
    risk_per_trade: float = 0.01  # 账户的1%
    max_daily_loss: float = 0.03  # 账户的3%
    
    # 情绪过滤
    use_sentiment_filter: bool = True
    fear_greed_threshold: int = 20  # 低于20不交易
    
    # 追踪止损
    use_trailing_stop: bool = True
    trailing_activation: float = 1.5  # 盈利1.5%激活
    trailing_distance: float = 1.0  # 距离最高点1%
    
    # 部分止盈
    partial_tp1_size: float = 0.3  # 30%仓位
    partial_tp2_size: float = 0.3  # 30%仓位
    partial_tp3_size: float = 0.4  # 40%仓位


@dataclass
class StrategyPerformance:
    """策略绩效"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    total_pnl: float = 0.0
    total_pnl_percent: float = 0.0
    
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_win: float = 0.0
    max_loss: float = 0.0
    
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    
    avg_trade_duration: float = 0.0  # 分钟
    
    # 按方向统计
    long_trades: int = 0
    long_win_rate: float = 0.0
    short_trades: int = 0
    short_win_rate: float = 0.0
    
    # 按时间框架统计
    performance_by_timeframe: Dict[str, Dict] = field(default_factory=dict)
    
    # 最近交易
    recent_trades: List[Dict] = field(default_factory=list)
    
    def calculate_metrics(self):
        """计算绩效指标"""
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades * 100
            
        if self.losing_trades > 0 and self.avg_loss != 0:
            self.profit_factor = abs(self.avg_win * self.winning_trades / (self.avg_loss * self.losing_trades))
        
        if self.long_trades > 0:
            long_wins = sum(1 for t in self.recent_trades if t.get('direction') == 'LONG' and t.get('pnl', 0) > 0)
            self.long_win_rate = long_wins / self.long_trades * 100
            
        if self.short_trades > 0:
            short_wins = sum(1 for t in self.recent_trades if t.get('direction') == 'SHORT' and t.get('pnl', 0) > 0)
            self.short_win_rate = short_wins / self.short_trades * 100


@dataclass
class BacktestResult:
    """回测结果"""
    config: StrategyConfig
    start_date: datetime
    end_date: datetime
    
    performance: StrategyPerformance
    equity_curve: List[Tuple[datetime, float]]
    trades: List[Dict]
    signals: List[Signal]
    
    # 统计
    total_signals: int = 0
    executed_signals: int = 0
    skipped_signals: int = 0
    
    # 按月统计
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    
    def summary(self) -> str:
        """生成回测摘要"""
        p = self.performance
        return f"""
====================================
回测结果: {self.config.name}
====================================
回测周期: {self.start_date.strftime('%Y-%m-%d')} ~ {self.end_date.strftime('%Y-%m-%d')}

交易统计:
- 总交易次数: {p.total_trades}
- 盈利次数: {p.winning_trades}
- 亏损次数: {p.losing_trades}
- 胜率: {p.win_rate:.2f}%

盈亏统计:
- 总盈亏: {p.total_pnl:.2f} USDT ({p.total_pnl_percent:.2f}%)
- 平均盈利: {p.avg_win:.2f} USDT
- 平均亏损: {p.avg_loss:.2f} USDT
- 最大盈利: {p.max_win:.2f} USDT
- 最大亏损: {p.max_loss:.2f} USDT
- 盈亏比: {p.profit_factor:.2f}

风险指标:
- 最大回撤: {p.max_drawdown_percent:.2f}%
- 夏普比率: {p.sharpe_ratio:.2f}
- 索提诺比率: {p.sortino_ratio:.2f}

方向统计:
- 做多胜率: {p.long_win_rate:.2f}% ({p.long_trades}笔)
- 做空胜率: {p.short_win_rate:.2f}% ({p.short_trades}笔)
====================================
"""
