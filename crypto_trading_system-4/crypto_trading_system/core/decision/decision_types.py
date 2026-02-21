"""
决策层类型定义
定义所有决策相关的数据结构和类型
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum
import numpy as np


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionSide(Enum):
    """持仓方向"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class ExecutionStrategy(Enum):
    """执行策略"""
    IMMEDIATE = "immediate"  # 立即执行
    TWAP = "twap"  # 时间加权平均价格
    VWAP = "vwap"  # 成交量加权平均价格
    ICEBERG = "iceberg"  # 冰山订单
    SMART = "smart"  # 智能执行


@dataclass
class DecisionConfig:
    """决策配置"""
    # 信号配置
    min_confidence: float = 0.6
    min_consistency_score: float = 0.7
    signal_validity_minutes: int = 60
    
    # 仓位配置
    max_position_size: float = 1.0  # 最大仓位比例
    default_position_size: float = 0.1  # 默认仓位比例
    max_concurrent_positions: int = 5  # 最大同时持仓数
    
    # 风险管理配置
    max_risk_per_trade: float = 0.02  # 单笔交易最大风险 (2%)
    max_daily_risk: float = 0.05  # 日最大风险 (5%)
    max_drawdown: float = 0.15  # 最大回撤 (15%)
    
    # 止损止盈配置
    default_stop_loss_pct: float = 0.02  # 默认止损比例 (2%)
    default_take_profit_pct: float = 0.04  # 默认止盈比例 (4%)
    trailing_stop_enabled: bool = True
    trailing_stop_distance: float = 0.015  # 追踪止损距离 (1.5%)
    
    # 执行配置
    execution_strategy: ExecutionStrategy = ExecutionStrategy.SMART
    max_slippage: float = 0.001  # 最大滑点 (0.1%)
    order_timeout_seconds: int = 300  # 订单超时时间
    
    # 手续费配置
    maker_fee: float = 0.001  # Maker 手续费 (0.1%)
    taker_fee: float = 0.001  # Taker 手续费 (0.1%)


@dataclass
class Order:
    """订单"""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_filled(self) -> bool:
        """是否已完全成交"""
        return self.status == OrderStatus.FILLED
    
    def is_active(self) -> bool:
        """是否活跃"""
        return self.status in [OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]
    
    def get_remaining_quantity(self) -> float:
        """获取剩余未成交数量"""
        return self.quantity - self.filled_quantity


@dataclass
class Position:
    """持仓"""
    symbol: str
    side: PositionSide
    quantity: float
    average_entry_price: float
    
    # 当前状态
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    
    # 止损止盈
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    trailing_stop_price: Optional[float] = None
    
    # 时间
    opened_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_pnl(self, current_price: float) -> tuple:
        """计算盈亏"""
        if self.side == PositionSide.LONG:
            pnl = (current_price - self.average_entry_price) * self.quantity
            pnl_pct = (current_price - self.average_entry_price) / self.average_entry_price
        elif self.side == PositionSide.SHORT:
            pnl = (self.average_entry_price - current_price) * self.quantity
            pnl_pct = (self.average_entry_price - current_price) / self.average_entry_price
        else:
            pnl = 0.0
            pnl_pct = 0.0
        
        return pnl, pnl_pct
    
    def get_notional_value(self) -> float:
        """获取名义价值"""
        return self.quantity * self.current_price
    
    def is_stop_triggered(self, price: float) -> bool:
        """检查是否触发止损"""
        if self.stop_loss_price is None:
            return False
        
        if self.side == PositionSide.LONG:
            return price <= self.stop_loss_price
        elif self.side == PositionSide.SHORT:
            return price >= self.stop_loss_price
        return False
    
    def is_take_profit_triggered(self, price: float) -> bool:
        """检查是否触发止盈"""
        if self.take_profit_price is None:
            return False
        
        if self.side == PositionSide.LONG:
            return price >= self.take_profit_price
        elif self.side == PositionSide.SHORT:
            return price <= self.take_profit_price
        return False


@dataclass
class RiskProfile:
    """风险画像"""
    # 当前风险状态
    total_exposure: float = 0.0  # 总敞口
    total_risk: float = 0.0  # 总风险
    daily_pnl: float = 0.0  # 日盈亏
    daily_risk_used: float = 0.0  # 日风险使用
    
    # 风险指标
    current_drawdown: float = 0.0  # 当前回撤
    max_drawdown_reached: float = 0.0  # 达到的最大回撤
    
    # 持仓统计
    position_count: int = 0
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    
    # 风险限制
    risk_limit_reached: bool = False
    daily_limit_reached: bool = False
    drawdown_limit_reached: bool = False
    
    def can_take_new_position(self, size: float) -> bool:
        """是否可以开新仓位"""
        if self.risk_limit_reached or self.daily_limit_reached or self.drawdown_limit_reached:
            return False
        return True


@dataclass
class ExecutionPlan:
    """执行计划"""
    strategy: ExecutionStrategy
    orders: List[Order]
    
    # 执行参数
    total_quantity: float
    executed_quantity: float = 0.0
    
    # 时间参数
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    # 价格参数
    target_price: Optional[float] = None
    price_tolerance: float = 0.001  # 价格容忍度
    
    # 状态
    is_complete: bool = False
    average_execution_price: float = 0.0
    
    def get_progress(self) -> float:
        """获取执行进度"""
        if self.total_quantity == 0:
            return 0.0
        return self.executed_quantity / self.total_quantity


@dataclass
class SignalValidationResult:
    """信号验证结果"""
    is_valid: bool
    confidence_score: float
    risk_score: float
    
    # 验证详情
    passed_filters: List[str]
    failed_filters: List[str]
    
    # 建议
    recommended_position_size: float
    recommended_leverage: float
    
    # 警告
    warnings: List[str]
    
    def __repr__(self):
        return f"SignalValidation(valid={self.is_valid}, confidence={self.confidence_score:.2%})"


@dataclass
class TradingDecision:
    """交易决策"""
    # 基本信息
    id: str
    symbol: str
    timestamp: datetime
    
    # 决策类型
    action: str  # OPEN_LONG, OPEN_SHORT, CLOSE, HOLD, INCREASE, DECREASE
    
    # 数量
    quantity: float
    position_size_ratio: float
    
    # 价格
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # 订单
    orders: List[Order] = field(default_factory=list)
    execution_plan: Optional[ExecutionPlan] = None
    
    # 风险
    risk_amount: float = 0.0
    risk_reward_ratio: float = 0.0
    
    # 推理信息
    signal_confidence: float = 0.0
    consistency_score: float = 0.0
    reasoning_summary: str = ""
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_executable(self) -> bool:
        """是否可执行"""
        return self.action not in ['HOLD'] and len(self.orders) > 0
    
    def get_expected_pnl(self) -> tuple:
        """获取预期盈亏"""
        if self.entry_price is None or self.take_profit is None:
            return 0.0, 0.0
        
        if self.action in ['OPEN_LONG', 'INCREASE']:
            profit = (self.take_profit - self.entry_price) * self.quantity
            profit_pct = (self.take_profit - self.entry_price) / self.entry_price
        elif self.action in ['OPEN_SHORT']:
            profit = (self.entry_price - self.take_profit) * self.quantity
            profit_pct = (self.entry_price - self.take_profit) / self.entry_price
        else:
            return 0.0, 0.0
        
        return profit, profit_pct


@dataclass
class PortfolioState:
    """组合状态"""
    # 资金
    total_equity: float
    available_balance: float
    frozen_balance: float
    
    # 持仓
    positions: Dict[str, Position]
    
    # 盈亏
    total_unrealized_pnl: float
    total_realized_pnl: float
    today_realized_pnl: float
    
    # 风险
    margin_used: float
    margin_ratio: float
    
    # 时间
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_position_value(self) -> float:
        """获取持仓总价值"""
        return sum(pos.get_notional_value() for pos in self.positions.values())
    
    def get_total_pnl(self) -> float:
        """获取总盈亏"""
        return self.total_unrealized_pnl + self.total_realized_pnl
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total_equity': self.total_equity,
            'available_balance': self.available_balance,
            'frozen_balance': self.frozen_balance,
            'total_unrealized_pnl': self.total_unrealized_pnl,
            'total_realized_pnl': self.total_realized_pnl,
            'today_realized_pnl': self.today_realized_pnl,
            'margin_used': self.margin_used,
            'margin_ratio': self.margin_ratio
        }
