"""
仓位管理器
管理交易仓位，包括开仓、平仓、加仓、减仓等操作
"""

import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from .decision_types import (
    Position,
    PositionSide,
    PortfolioState,
    DecisionConfig,
    TradingDecision
)

logger = logging.getLogger(__name__)


class PositionManager:
    """
    仓位管理器
    
    管理交易仓位，提供以下功能：
    1. 仓位跟踪
    2. 仓位调整
    3. 止损止盈管理
    4. 仓位限制检查
    """
    
    def __init__(self, config: DecisionConfig):
        """
        初始化仓位管理器
        
        Args:
            config: 决策配置
        """
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.position_history: List[Dict[str, Any]] = []
        
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        获取指定交易对的持仓
        
        Args:
            symbol: 交易对符号
            
        Returns:
            Position: 持仓信息，如果没有持仓则返回 None
        """
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """
        获取所有持仓
        
        Returns:
            Dict[str, Position]: 所有持仓
        """
        return self.positions.copy()
    
    def get_open_positions(self) -> Dict[str, Position]:
        """
        获取所有开仓的持仓
        
        Returns:
            Dict[str, Position]: 开仓的持仓
        """
        return {
            symbol: pos for symbol, pos in self.positions.items()
            if pos.side != PositionSide.FLAT
        }
    
    def open_position(
        self,
        symbol: str,
        side: PositionSide,
        quantity: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Position:
        """
        开新仓位
        
        Args:
            symbol: 交易对符号
            side: 持仓方向
            quantity: 数量
            entry_price: 入场价格
            stop_loss: 止损价格 (可选)
            take_profit: 止盈价格 (可选)
            metadata: 元数据 (可选)
            
        Returns:
            Position: 新创建的持仓
        """
        # 检查是否已有持仓
        existing = self.positions.get(symbol)
        if existing and existing.side != PositionSide.FLAT:
            logger.warning(f"Position already exists for {symbol}, consider using add_to_position")
            return existing
        
        # 创建新持仓
        position = Position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            average_entry_price=entry_price,
            current_price=entry_price,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            metadata=metadata or {}
        )
        
        # 计算初始盈亏
        position.unrealized_pnl, position.unrealized_pnl_pct = position.calculate_pnl(entry_price)
        
        # 保存持仓
        self.positions[symbol] = position
        
        # 记录历史
        self._record_position_change('OPEN', position)
        
        logger.info(
            f"Position opened: {symbol} {side.value} "
            f"qty={quantity:.4f} price={entry_price:.2f}"
        )
        
        return position
    
    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str = 'manual'
    ) -> Optional[Dict[str, Any]]:
        """
        平仓
        
        Args:
            symbol: 交易对符号
            exit_price: 出场价格
            reason: 平仓原因
            
        Returns:
            Dict: 平仓结果，包含盈亏信息
        """
        position = self.positions.get(symbol)
        if not position or position.side == PositionSide.FLAT:
            logger.warning(f"No position to close for {symbol}")
            return None
        
        # 计算实现盈亏
        realized_pnl, realized_pnl_pct = position.calculate_pnl(exit_price)
        
        # 记录平仓
        result = {
            'symbol': symbol,
            'side': position.side.value,
            'quantity': position.quantity,
            'entry_price': position.average_entry_price,
            'exit_price': exit_price,
            'realized_pnl': realized_pnl,
            'realized_pnl_pct': realized_pnl_pct,
            'reason': reason,
            'opened_at': position.opened_at,
            'closed_at': datetime.now()
        }
        
        # 记录历史
        self._record_position_change('CLOSE', position, result)
        
        # 移除持仓
        del self.positions[symbol]
        
        logger.info(
            f"Position closed: {symbol} "
            f"pnl={realized_pnl:.2f} ({realized_pnl_pct:.2%}) reason={reason}"
        )
        
        return result
    
    def add_to_position(
        self,
        symbol: str,
        additional_quantity: float,
        price: float
    ) -> Optional[Position]:
        """
        加仓
        
        Args:
            symbol: 交易对符号
            additional_quantity: 增加的数量
            price: 加仓价格
            
        Returns:
            Position: 更新后的持仓
        """
        position = self.positions.get(symbol)
        if not position or position.side == PositionSide.FLAT:
            logger.warning(f"No existing position to add for {symbol}")
            return None
        
        # 计算新的平均入场价格
        total_value = (position.quantity * position.average_entry_price + 
                      additional_quantity * price)
        total_quantity = position.quantity + additional_quantity
        
        position.average_entry_price = total_value / total_quantity
        position.quantity = total_quantity
        position.current_price = price
        position.updated_at = datetime.now()
        
        # 重新计算盈亏
        position.unrealized_pnl, position.unrealized_pnl_pct = position.calculate_pnl(price)
        
        # 记录历史
        self._record_position_change('INCREASE', position)
        
        logger.info(
            f"Position increased: {symbol} "
            f"added={additional_quantity:.4f} new_avg={position.average_entry_price:.2f}"
        )
        
        return position
    
    def reduce_position(
        self,
        symbol: str,
        reduce_quantity: float,
        price: float
    ) -> Optional[Position]:
        """
        减仓
        
        Args:
            symbol: 交易对符号
            reduce_quantity: 减少的数量
            price: 减仓价格
            
        Returns:
            Position: 更新后的持仓，如果完全平仓则返回 None
        """
        position = self.positions.get(symbol)
        if not position or position.side == PositionSide.FLAT:
            logger.warning(f"No position to reduce for {symbol}")
            return None
        
        # 检查减仓数量
        if reduce_quantity >= position.quantity:
            # 完全平仓
            return self.close_position(symbol, price, reason='reduce_to_zero')
        
        # 计算实现盈亏
        realized_pnl, _ = position.calculate_pnl(price)
        realized_pnl *= (reduce_quantity / position.quantity)
        
        # 减仓
        position.quantity -= reduce_quantity
        position.current_price = price
        position.updated_at = datetime.now()
        
        # 重新计算盈亏
        position.unrealized_pnl, position.unrealized_pnl_pct = position.calculate_pnl(price)
        
        # 记录历史
        self._record_position_change('DECREASE', position, {'realized_pnl': realized_pnl})
        
        logger.info(
            f"Position decreased: {symbol} "
            f"reduced={reduce_quantity:.4f} remaining={position.quantity:.4f}"
        )
        
        return position
    
    def update_position_price(self, symbol: str, current_price: float):
        """
        更新持仓当前价格
        
        Args:
            symbol: 交易对符号
            current_price: 当前价格
        """
        position = self.positions.get(symbol)
        if not position or position.side == PositionSide.FLAT:
            return
        
        position.current_price = current_price
        position.updated_at = datetime.now()
        
        # 更新盈亏
        position.unrealized_pnl, position.unrealized_pnl_pct = position.calculate_pnl(current_price)
        
        # 更新追踪止损
        self._update_trailing_stop(position)
    
    def _update_trailing_stop(self, position: Position):
        """更新追踪止损价格"""
        if not self.config.trailing_stop_enabled:
            return
        
        if position.stop_loss_price is None:
            return
        
        # 计算新的追踪止损价格
        if position.side == PositionSide.LONG:
            # 多头：价格上涨时，止损上移
            new_stop = position.current_price * (1 - self.config.trailing_stop_distance)
            if new_stop > position.stop_loss_price:
                position.stop_loss_price = new_stop
                position.trailing_stop_price = new_stop
        
        elif position.side == PositionSide.SHORT:
            # 空头：价格下跌时，止损下移
            new_stop = position.current_price * (1 + self.config.trailing_stop_distance)
            if new_stop < position.stop_loss_price:
                position.stop_loss_price = new_stop
                position.trailing_stop_price = new_stop
    
    def check_stop_conditions(self, symbol: str, price: float) -> Optional[str]:
        """
        检查止损止盈条件
        
        Args:
            symbol: 交易对符号
            price: 当前价格
            
        Returns:
            str: 触发条件 ('stop_loss', 'take_profit', 'trailing_stop', None)
        """
        position = self.positions.get(symbol)
        if not position or position.side == PositionSide.FLAT:
            return None
        
        # 检查止损
        if position.is_stop_triggered(price):
            return 'stop_loss'
        
        # 检查止盈
        if position.is_take_profit_triggered(price):
            return 'take_profit'
        
        return None
    
    def can_open_new_position(
        self,
        symbol: str,
        portfolio_state: PortfolioState
    ) -> tuple:
        """
        检查是否可以开新仓位
        
        Args:
            symbol: 交易对符号
            portfolio_state: 组合状态
            
        Returns:
            tuple: (是否可以, 原因)
        """
        # 检查是否已有持仓
        if symbol in self.positions and self.positions[symbol].side != PositionSide.FLAT:
            return False, "position_already_exists"
        
        # 检查持仓数量限制
        open_positions = len(self.get_open_positions())
        if open_positions >= self.config.max_concurrent_positions:
            return False, f"max_positions_reached ({open_positions}/{self.config.max_concurrent_positions})"
        
        # 检查资金
        if portfolio_state.available_balance <= 0:
            return False, "insufficient_balance"
        
        return True, "ok"
    
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        portfolio_state: PortfolioState,
        risk_per_trade: Optional[float] = None
    ) -> float:
        """
        计算仓位大小 (基于风险)
        
        Args:
            symbol: 交易对符号
            entry_price: 入场价格
            stop_loss: 止损价格
            portfolio_state: 组合状态
            risk_per_trade: 单笔交易风险比例 (可选)
            
        Returns:
            float: 建议的仓位数量
        """
        if risk_per_trade is None:
            risk_per_trade = self.config.max_risk_per_trade
        
        # 计算风险金额
        risk_amount = portfolio_state.total_equity * risk_per_trade
        
        # 计算每单位风险
        if entry_price <= 0 or stop_loss <= 0:
            return 0.0
        
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit == 0:
            return 0.0
        
        # 计算数量
        quantity = risk_amount / risk_per_unit
        
        # 检查可用资金
        max_quantity = portfolio_state.available_balance / entry_price * 0.95  # 留5%缓冲
        quantity = min(quantity, max_quantity)
        
        return quantity
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        获取组合摘要
        
        Returns:
            Dict: 组合摘要
        """
        positions = self.get_open_positions()
        
        if not positions:
            return {
                'position_count': 0,
                'total_exposure': 0.0,
                'total_unrealized_pnl': 0.0,
                'long_count': 0,
                'short_count': 0
            }
        
        total_exposure = sum(pos.get_notional_value() for pos in positions.values())
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in positions.values())
        
        long_count = sum(1 for pos in positions.values() if pos.side == PositionSide.LONG)
        short_count = sum(1 for pos in positions.values() if pos.side == PositionSide.SHORT)
        
        return {
            'position_count': len(positions),
            'total_exposure': total_exposure,
            'total_unrealized_pnl': total_unrealized_pnl,
            'long_count': long_count,
            'short_count': short_count,
            'positions': [
                {
                    'symbol': pos.symbol,
                    'side': pos.side.value,
                    'quantity': pos.quantity,
                    'entry_price': pos.average_entry_price,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'unrealized_pnl_pct': pos.unrealized_pnl_pct
                }
                for pos in positions.values()
            ]
        }
    
    def _record_position_change(
        self, 
        action: str, 
        position: Position,
        extra_data: Optional[Dict] = None
    ):
        """记录持仓变化"""
        record = {
            'action': action,
            'symbol': position.symbol,
            'side': position.side.value,
            'quantity': position.quantity,
            'average_entry_price': position.average_entry_price,
            'current_price': position.current_price,
            'unrealized_pnl': position.unrealized_pnl,
            'timestamp': datetime.now()
        }
        
        if extra_data:
            record.update(extra_data)
        
        self.position_history.append(record)
    
    def get_position_history(self) -> List[Dict[str, Any]]:
        """获取持仓历史"""
        return self.position_history.copy()
