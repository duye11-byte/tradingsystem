"""
决策引擎主入口
整合信号生成、仓位管理、风险管理和执行优化
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import asyncio

from .decision_types import (
    DecisionConfig,
    TradingDecision,
    PortfolioState,
    Position,
    PositionSide,
    Order,
    ExecutionPlan
)
from .signal_generator import SignalGenerator
from .position_manager import PositionManager
from .risk_manager import RiskManager
from .execution_optimizer import ExecutionOptimizer
from .order_manager import OrderManager
from ..reasoning import TradingSignal

logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    决策引擎
    
    整合决策层的所有组件，提供统一的决策接口：
    1. 接收推理层的交易信号
    2. 验证信号并生成交易决策
    3. 检查风险限制
    4. 创建执行计划
    5. 提交订单
    """
    
    def __init__(self, config: Optional[DecisionConfig] = None):
        """
        初始化决策引擎
        
        Args:
            config: 决策配置 (可选)
        """
        self.config = config or DecisionConfig()
        
        # 初始化各组件
        self.signal_generator = SignalGenerator(self.config)
        self.position_manager = PositionManager(self.config)
        self.risk_manager = RiskManager(self.config)
        self.execution_optimizer = ExecutionOptimizer(self.config)
        self.order_manager = OrderManager(self.config)
        
        # 组合状态
        self.portfolio_state = PortfolioState(
            total_equity=100000.0,  # 默认初始资金
            available_balance=100000.0,
            frozen_balance=0.0,
            positions={},
            total_unrealized_pnl=0.0,
            total_realized_pnl=0.0,
            today_realized_pnl=0.0,
            margin_used=0.0,
            margin_ratio=0.0
        )
        
        # 性能统计
        self.stats = {
            'total_decisions': 0,
            'executed_decisions': 0,
            'rejected_decisions': 0,
            'total_pnl': 0.0
        }
        
        logger.info("DecisionEngine initialized")
    
    async def process_signal(
        self,
        signal: TradingSignal,
        current_price: Optional[float] = None
    ) -> Optional[TradingDecision]:
        """
        处理交易信号
        
        Args:
            signal: 交易信号
            current_price: 当前价格 (可选)
            
        Returns:
            TradingDecision: 交易决策，如果信号被拒绝则返回 None
        """
        self.stats['total_decisions'] += 1
        
        logger.info(
            f"Processing signal for {signal.symbol}: "
            f"{signal.signal.value} (confidence={signal.confidence:.1%})"
        )
        
        # 1. 生成交易决策
        decision = self.signal_generator.generate_decision(
            signal,
            self.portfolio_state.to_dict() if hasattr(self.portfolio_state, 'to_dict') else None,
            current_price
        )
        
        if not decision:
            logger.info(f"Signal converted to HOLD for {signal.symbol}")
            self.stats['rejected_decisions'] += 1
            return None
        
        # 2. 风险评估
        positions = self.position_manager.get_all_positions()
        passed, risk_reasons = self.risk_manager.validate_decision(
            decision,
            self.portfolio_state,
            positions
        )
        
        if not passed:
            logger.warning(
                f"Decision rejected by risk manager for {signal.symbol}: {risk_reasons}"
            )
            self.stats['rejected_decisions'] += 1
            return None
        
        # 3. 检查仓位限制
        can_open, reason = self.position_manager.can_open_new_position(
            signal.symbol,
            self.portfolio_state
        )
        
        if not can_open:
            logger.warning(
                f"Cannot open position for {signal.symbol}: {reason}"
            )
            self.stats['rejected_decisions'] += 1
            return None
        
        # 4. 创建执行计划
        execution_plan = self.execution_optimizer.create_execution_plan(
            symbol=decision.symbol,
            side=decision.orders[0].side if decision.orders else None,
            total_quantity=decision.quantity,
            target_price=decision.entry_price
        )
        
        decision.execution_plan = execution_plan
        
        # 5. 提交订单
        await self._execute_decision(decision)
        
        self.stats['executed_decisions'] += 1
        
        logger.info(
            f"Decision executed for {signal.symbol}: "
            f"{decision.action} qty={decision.quantity:.4f}"
        )
        
        return decision
    
    async def _execute_decision(self, decision: TradingDecision):
        """执行交易决策"""
        # 提交所有订单
        for order in decision.orders:
            self.order_manager.submit_order(order)
        
        # 如果是开仓决策，更新持仓
        if decision.action in ['OPEN_LONG', 'OPEN_SHORT']:
            side = PositionSide.LONG if decision.action == 'OPEN_LONG' else PositionSide.SHORT
            
            self.position_manager.open_position(
                symbol=decision.symbol,
                side=side,
                quantity=decision.quantity,
                entry_price=decision.entry_price or 0,
                stop_loss=decision.stop_loss,
                take_profit=decision.take_profit,
                metadata={
                    'decision_id': decision.id,
                    'signal_confidence': decision.signal_confidence
                }
            )
        
        # 更新组合状态
        self._update_portfolio_state()
    
    def _update_portfolio_state(self):
        """更新组合状态"""
        # 更新持仓
        self.portfolio_state.positions = self.position_manager.get_all_positions()
        
        # 计算未实现盈亏
        total_unrealized = sum(
            pos.unrealized_pnl for pos in self.portfolio_state.positions.values()
        )
        self.portfolio_state.total_unrealized_pnl = total_unrealized
        
        # 更新可用资金 (简化计算)
        position_value = sum(
            pos.get_notional_value() for pos in self.portfolio_state.positions.values()
        )
        self.portfolio_state.frozen_balance = position_value
        self.portfolio_state.available_balance = (
            self.portfolio_state.total_equity - position_value
        )
        
        # 记录权益
        self.risk_manager.record_equity(self.portfolio_state.total_equity)
    
    def update_position_prices(self, prices: Dict[str, float]):
        """
        更新持仓价格
        
        Args:
            prices: 价格字典 {symbol: price}
        """
        for symbol, price in prices.items():
            self.position_manager.update_position_price(symbol, price)
            
            # 检查止损止盈
            trigger = self.position_manager.check_stop_conditions(symbol, price)
            
            if trigger:
                logger.info(f"Stop condition triggered for {symbol}: {trigger}")
                # 这里可以自动触发平仓
                # asyncio.create_task(self._close_position(symbol, price, trigger))
        
        # 更新组合状态
        self._update_portfolio_state()
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        获取组合摘要
        
        Returns:
            Dict: 组合摘要
        """
        position_summary = self.position_manager.get_portfolio_summary()
        risk_profile = self.risk_manager.assess_risk(
            self.portfolio_state,
            self.position_manager.get_all_positions()
        )
        order_stats = self.order_manager.get_order_statistics()
        
        return {
            'portfolio': {
                'total_equity': self.portfolio_state.total_equity,
                'available_balance': self.portfolio_state.available_balance,
                'frozen_balance': self.portfolio_state.frozen_balance,
                'total_unrealized_pnl': self.portfolio_state.total_unrealized_pnl,
                'total_realized_pnl': self.portfolio_state.total_realized_pnl
            },
            'positions': position_summary,
            'risk': {
                'current_drawdown': risk_profile.current_drawdown,
                'total_exposure': risk_profile.total_exposure,
                'position_count': risk_profile.position_count,
                'risk_limit_reached': risk_profile.risk_limit_reached
            },
            'orders': order_stats,
            'performance': self.stats
        }
    
    def get_decision_stats(self) -> Dict[str, Any]:
        """
        获取决策统计
        
        Returns:
            Dict: 决策统计
        """
        signal_stats = self.signal_generator.get_signal_stats()
        
        return {
            **self.stats,
            'success_rate': (
                self.stats['executed_decisions'] / self.stats['total_decisions']
                if self.stats['total_decisions'] > 0 else 0
            ),
            'signal_stats': signal_stats
        }
    
    def set_portfolio_state(self, state: PortfolioState):
        """
        设置组合状态
        
        Args:
            state: 组合状态
        """
        self.portfolio_state = state
        logger.info(f"Portfolio state updated: equity={state.total_equity:.2f}")
    
    def close_all_positions(self, reason: str = 'manual') -> List[Dict[str, Any]]:
        """
        平掉所有持仓
        
        Args:
            reason: 平仓原因
            
        Returns:
            List[Dict]: 平仓结果列表
        """
        results = []
        positions = self.position_manager.get_open_positions()
        
        for symbol in list(positions.keys()):
            # 获取当前价格
            current_price = positions[symbol].current_price
            
            # 平仓
            result = self.position_manager.close_position(symbol, current_price, reason)
            
            if result:
                results.append(result)
                
                # 更新实现盈亏
                self.portfolio_state.total_realized_pnl += result['realized_pnl']
                self.stats['total_pnl'] += result['realized_pnl']
        
        # 更新组合状态
        self._update_portfolio_state()
        
        logger.info(f"Closed {len(results)} positions")
        
        return results
    
    def get_risk_report(self) -> Dict[str, Any]:
        """
        获取风险报告
        
        Returns:
            Dict: 风险报告
        """
        return self.risk_manager.get_risk_report()
