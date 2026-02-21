"""
反馈层类型定义
定义所有反馈相关的数据结构和类型
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import numpy as np


class TradeResult(Enum):
    """交易结果"""
    WIN = "win"
    LOSS = "loss"
    BREAK_EVEN = "break_even"
    OPEN = "open"
    CANCELLED = "cancelled"


class FeedbackType(Enum):
    """反馈类型"""
    PERFORMANCE = "performance"
    HUMAN_RATING = "human_rating"
    MARKET_CONDITION = "market_condition"
    SELF_EVALUATION = "self_evaluation"


class LearningStatus(Enum):
    """学习状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class FeedbackConfig:
    """反馈配置"""
    # 性能分析配置
    performance_window_days: int = 30
    min_trades_for_analysis: int = 10
    
    # 在线学习配置
    learning_enabled: bool = True
    learning_batch_size: int = 32
    learning_rate: float = 0.001
    min_samples_for_learning: int = 50
    
    # RLHF配置
    rlhf_enabled: bool = True
    human_feedback_weight: float = 0.3
    auto_feedback_weight: float = 0.7
    
    # 存储配置
    max_trade_history: int = 10000
    max_feedback_history: int = 5000
    
    # 监控配置
    report_interval_hours: int = 24
    alert_on_drawdown: float = 0.10


@dataclass
class TradeRecord:
    """交易记录"""
    id: str
    symbol: str
    
    # 入场信息
    entry_time: datetime
    entry_price: float
    entry_side: str  # long / short
    entry_quantity: float
    
    # 出场信息
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # stop_loss, take_profit, manual, signal
    
    # 盈亏
    realized_pnl: float = 0.0
    realized_pnl_pct: float = 0.0
    
    # 费用
    entry_fee: float = 0.0
    exit_fee: float = 0.0
    total_fee: float = 0.0
    
    # 风险
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_amount: float = 0.0
    risk_reward_ratio: float = 0.0
    
    # 信号信息
    signal_confidence: float = 0.0
    signal_consistency: float = 0.0
    
    # 状态
    result: TradeResult = TradeResult.OPEN
    is_closed: bool = False
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def close_trade(
        self,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str
    ):
        """关闭交易"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = exit_reason
        self.is_closed = True
        
        # 计算盈亏
        if self.entry_side == 'long':
            self.realized_pnl = (exit_price - self.entry_price) * self.entry_quantity - self.total_fee
        else:
            self.realized_pnl = (self.entry_price - exit_price) * self.entry_quantity - self.total_fee
        
        if self.entry_price > 0:
            self.realized_pnl_pct = self.realized_pnl / (self.entry_price * self.entry_quantity)
        
        # 确定结果
        if self.realized_pnl > 0:
            self.result = TradeResult.WIN
        elif self.realized_pnl < 0:
            self.result = TradeResult.LOSS
        else:
            self.result = TradeResult.BREAK_EVEN
    
    def get_holding_period(self) -> Optional[float]:
        """获取持仓时间 (小时)"""
        if self.exit_time is None:
            return None
        return (self.exit_time - self.entry_time).total_seconds() / 3600


@dataclass
class PerformanceMetrics:
    """性能指标"""
    # 基本统计
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    break_even_trades: int = 0
    
    # 盈亏
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    avg_pnl_per_trade: float = 0.0
    
    # 胜率
    win_rate: float = 0.0
    loss_rate: float = 0.0
    
    # 盈亏比
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_win_loss_ratio: float = 0.0
    
    # 风险调整收益
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # 回撤
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    current_drawdown: float = 0.0
    
    # 其他
    avg_holding_period_hours: float = 0.0
    expectancy: float = 0.0
    
    # 时间范围
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'total_pnl': self.total_pnl,
            'total_pnl_pct': self.total_pnl_pct,
            'profit_factor': self.profit_factor,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown_pct,
            'expectancy': self.expectancy
        }


@dataclass
class LearningSample:
    """学习样本"""
    id: str
    
    # 输入特征
    features: Dict[str, float]
    
    # 决策信息
    predicted_signal: str
    predicted_confidence: float
    
    # 实际结果
    actual_result: str
    actual_pnl: float
    
    # 反馈
    reward: float
    human_rating: Optional[float] = None
    
    # 时间
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 状态
    status: LearningStatus = LearningStatus.PENDING
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HumanFeedback:
    """人类反馈"""
    id: str
    
    # 反馈内容
    rating: int = 3  # 1-5 评分，默认3分
    
    # 反馈对象
    trade_id: Optional[str] = None
    decision_id: Optional[str] = None
    signal_id: Optional[str] = None
    
    # 评论
    comment: Optional[str] = None
    
    # 反馈类型
    feedback_type: FeedbackType = FeedbackType.HUMAN_RATING
    
    # 时间
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelUpdate:
    """模型更新"""
    id: str
    
    # 更新信息
    component: str  # reasoning, decision, etc.
    update_type: str  # weight_adjustment, parameter_update, etc.
    
    # 更新内容
    changes: Dict[str, Any]
    
    # 验证结果
    validation_score: float = 0.0
    is_validated: bool = False
    
    # 时间
    created_at: datetime = field(default_factory=datetime.now)
    applied_at: Optional[datetime] = None
    
    # 状态
    status: str = "pending"  # pending, validated, applied, rolled_back


@dataclass
class FeedbackSummary:
    """反馈摘要"""
    # 性能摘要
    performance: PerformanceMetrics
    
    # 学习摘要
    total_samples: int
    processed_samples: int
    pending_samples: int
    
    # 人类反馈摘要
    total_human_feedback: int
    avg_human_rating: float
    
    # 模型更新摘要
    total_updates: int
    applied_updates: int
    pending_updates: int
    
    # 建议
    recommendations: List[str]
    
    # 时间
    generated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'performance': self.performance.to_dict(),
            'total_samples': self.total_samples,
            'processed_samples': self.processed_samples,
            'total_human_feedback': self.total_human_feedback,
            'avg_human_rating': self.avg_human_rating,
            'total_updates': self.total_updates,
            'applied_updates': self.applied_updates,
            'recommendations': self.recommendations
        }


@dataclass
class Alert:
    """告警"""
    id: str
    
    # 告警信息
    alert_type: str
    severity: str  # info, warning, critical
    message: str
    
    # 触发条件
    trigger_value: float
    threshold: float
    
    # 时间
    triggered_at: datetime = field(default_factory=datetime.now)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # 状态
    is_acknowledged: bool = False
    is_resolved: bool = False
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentPerformance:
    """代理性能"""
    agent_name: str
    
    # 交易统计
    trade_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # 性能指标
    win_rate: float = 0.0
    avg_reward: float = 0.0
    total_pnl: float = 0.0
    
    # 时间
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update(self, reward: float, is_win: bool):
        """更新性能"""
        self.trade_count += 1
        if is_win:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        self.total_pnl += reward
        self.win_rate = self.winning_trades / self.trade_count if self.trade_count > 0 else 0.0
        
        # 更新平均奖励
        self.avg_reward = ((self.avg_reward * (self.trade_count - 1)) + reward) / self.trade_count
        self.last_updated = datetime.now()


@dataclass
class StrategyInsight:
    """策略洞察"""
    category: str
    message: str
    confidence: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """性能告警"""
    alert_type: str
    message: str
    severity: str = "warning"  # info, warning, critical
    metric_value: float = 0.0
    threshold: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
