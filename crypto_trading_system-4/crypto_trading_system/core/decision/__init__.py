"""
决策层 (Decision Layer) - OpenClaw Crypto Trading System

将推理层的信号转换为实际的交易决策和执行。

核心组件：
- SignalGenerator: 信号生成和过滤
- PositionManager: 仓位管理
- ExecutionOptimizer: 执行优化
- RiskManager: 风险管理
- OrderManager: 订单管理
"""

from .decision_types import (
    DecisionConfig,
    TradingDecision,
    Order,
    OrderType,
    OrderSide,
    OrderStatus,
    Position,
    PositionSide,
    RiskProfile,
    ExecutionPlan,
    SignalValidationResult,
    PortfolioState,
    ExecutionStrategy
)

from .signal_generator import SignalGenerator
from .position_manager import PositionManager
from .execution_optimizer import ExecutionOptimizer
from .risk_manager import RiskManager
from .order_manager import OrderManager
from .decision_engine import DecisionEngine

__all__ = [
    'DecisionConfig',
    'TradingDecision',
    'Order',
    'OrderType',
    'OrderSide',
    'OrderStatus',
    'Position',
    'PositionSide',
    'RiskProfile',
    'ExecutionPlan',
    'SignalValidationResult',
    'PortfolioState',
    'ExecutionStrategy',
    'SignalGenerator',
    'PositionManager',
    'ExecutionOptimizer',
    'RiskManager',
    'OrderManager',
    'DecisionEngine'
]
