"""
OpenClaw 决策层 - 交易决策引擎
第4层: 决策层

功能:
1. 信号过滤和验证
2. 仓位管理
3. 风险管理
4. 订单生成
5. 执行决策
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import uuid
import logging

from ..strategy.strategy_types import (
    Signal, SignalDirection, Position, StrategyConfig, StrategyPerformance
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskManager:
    """风险管理器"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.daily_pnl = 0.0
        self.daily_loss = 0.0
        self.last_reset = datetime.now().date()
        self.position_count = 0
        self.max_positions = 5  # 最大同时持仓数
    
    def reset_daily(self):
        """重置每日统计"""
        today = datetime.now().date()
        if today != self.last_reset:
            self.daily_pnl = 0.0
            self.daily_loss = 0.0
            self.last_reset = today
            self.position_count = 0
    
    def check_risk_limits(self, account_balance: float, 
                          positions: List[Position]) -> Tuple[bool, str]:
        """检查风险限制"""
        self.reset_daily()
        
        # 检查每日最大亏损
        if self.daily_loss <= -account_balance * self.config.max_daily_loss:
            return False, f"每日最大亏损限制: {self.daily_loss:.2f}"
        
        # 检查最大持仓数
        if len(positions) >= self.max_positions:
            return False, f"最大持仓数限制: {len(positions)}"
        
        # 检查总风险敞口
        total_risk = sum(p.size * abs(p.entry_price - p.stop_loss) for p in positions)
        max_risk = account_balance * self.config.risk_per_trade * self.max_positions
        if total_risk > max_risk:
            return False, f"总风险敞口超限: {total_risk:.2f} > {max_risk:.2f}"
        
        return True, "通过"
    
    def calculate_position_size(self, signal: Signal, 
                                account_balance: float,
                                current_positions: List[Position]) -> float:
        """计算仓位大小"""
        
        # 基于风险计算仓位
        risk_amount = account_balance * self.config.risk_per_trade
        price_risk = abs(signal.entry_price - signal.stop_loss)
        
        if price_risk == 0:
            price_risk = signal.entry_price * 0.01  # 默认1%风险
        
        position_size = risk_amount / price_risk
        
        # 限制最大仓位
        max_size = account_balance * self.config.max_position_size / signal.entry_price
        position_size = min(position_size, max_size)
        
        # 根据信号质量调整
        if signal.score >= 8:
            position_size *= 1.0  # 高质量信号满仓
        elif signal.score >= 6.5:
            position_size *= 0.7  # 中等质量70%
        else:
            position_size *= 0.5  # 低质量50%
        
        # 根据置信度调整
        position_size *= (signal.confidence / 100)
        
        return position_size
    
    def calculate_leverage(self, signal: Signal, 
                          volatility_percentile: float) -> float:
        """计算杠杆倍数"""
        base_leverage = 2.0
        
        # 根据波动率调整
        if volatility_percentile > 80:
            leverage = base_leverage * 0.5  # 高波动降低杠杆
        elif volatility_percentile < 30:
            leverage = base_leverage * 1.5  # 低波动增加杠杆
        else:
            leverage = base_leverage
        
        # 根据信号质量调整
        if signal.score >= 8:
            leverage *= 1.2
        elif signal.score < 6:
            leverage *= 0.8
        
        return min(leverage, self.config.max_leverage)


class DecisionEngine:
    """决策引擎"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.risk_manager = RiskManager(config)
        self.positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self.pending_orders: List[Dict] = []
        self.performance = StrategyPerformance()
        
        # 账户信息
        self.account_balance = 10000.0  # 默认初始资金
        self.total_equity = self.account_balance
    
    def process_signal(self, signal: Signal, 
                       market_data: Dict) -> Optional[Dict]:
        """处理交易信号"""
        
        # 1. 风险检查
        can_trade, reason = self.risk_manager.check_risk_limits(
            self.account_balance, self.positions
        )
        if not can_trade:
            logger.warning(f"信号被拒绝 - 风险限制: {reason}")
            return None
        
        # 2. 信号验证
        if not signal.is_valid():
            logger.warning(f"信号被拒绝 - 无效信号: 置信度{signal.confidence}")
            return None
        
        # 3. 检查是否有相反方向持仓
        for pos in self.positions:
            if pos.symbol == signal.symbol:
                if (signal.direction == SignalDirection.LONG and pos.direction == SignalDirection.SHORT) or \
                   (signal.direction == SignalDirection.SHORT and pos.direction == SignalDirection.LONG):
                    logger.info(f"发现相反持仓，先平仓: {pos.symbol}")
                    self.close_position(pos.id, signal.entry_price, "反向信号")
        
        # 4. 计算仓位
        position_size = self.risk_manager.calculate_position_size(
            signal, self.account_balance, self.positions
        )
        
        # 5. 计算杠杆
        volatility = market_data.get('volatility_percentile', 50)
        leverage = self.risk_manager.calculate_leverage(signal, volatility)
        
        # 6. 生成订单
        order = self._create_order(signal, position_size, leverage)
        
        logger.info(f"生成订单: {signal.symbol} {signal.direction.name} "
                   f"数量:{position_size:.4f} 杠杆:{leverage:.1f}x")
        
        return order
    
    def _create_order(self, signal: Signal, 
                      size: float, 
                      leverage: float) -> Dict:
        """创建订单"""
        return {
            'id': str(uuid.uuid4())[:8],
            'signal_id': signal.id,
            'symbol': signal.symbol,
            'direction': signal.direction.name,
            'type': 'MARKET',  # 或 LIMIT
            'entry_price': signal.entry_price,
            'size': size,
            'leverage': leverage,
            'stop_loss': signal.stop_loss,
            'take_profit_1': signal.take_profit_1,
            'take_profit_2': signal.take_profit_2,
            'take_profit_3': signal.take_profit_3,
            'timestamp': datetime.now(),
            'status': 'PENDING',
            'reasons': signal.reasons,
            'confidence': signal.confidence,
            'score': signal.score
        }
    
    def execute_order(self, order: Dict, execution_price: float) -> Optional[Position]:
        """执行订单"""
        
        direction = SignalDirection.LONG if order['direction'] == 'LONG' else SignalDirection.SHORT
        
        position = Position(
            id=str(uuid.uuid4())[:8],
            symbol=order['symbol'],
            direction=direction,
            entry_price=execution_price,
            size=order['size'],
            leverage=order['leverage'],
            stop_loss=order['stop_loss'],
            take_profit_1=order['take_profit_1'],
            take_profit_2=order['take_profit_2'],
            take_profit_3=order['take_profit_3'],
            trailing_stop=order['stop_loss'],
            highest_price=execution_price,
            lowest_price=execution_price
        )
        
        self.positions.append(position)
        self.risk_manager.position_count += 1
        
        logger.info(f"订单执行成功: {position.symbol} {position.direction.name} "
                   f"@{execution_price:.2f}")
        
        return position
    
    def update_positions(self, current_prices: Dict[str, float]):
        """更新持仓状态"""
        for position in self.positions:
            if position.symbol in current_prices:
                current_price = current_prices[position.symbol]
                position.update_pnl(current_price)
                
                # 检查追踪止损
                if self.config.use_trailing_stop:
                    self._update_trailing_stop(position, current_price)
                
                # 检查部分止盈
                self._check_partial_tp(position, current_price)
    
    def _update_trailing_stop(self, position: Position, current_price: float):
        """更新追踪止损"""
        entry = position.entry_price
        
        if position.direction == SignalDirection.LONG:
            pnl_percent = (current_price - entry) / entry * 100
            
            # 激活追踪止损
            if pnl_percent >= self.config.trailing_activation and not position.trailing_activated:
                position.trailing_activated = True
                position.trailing_stop = current_price * (1 - self.config.trailing_distance / 100)
                logger.info(f"追踪止损激活: {position.symbol} @ {position.trailing_stop:.2f}")
            
            # 更新追踪止损
            elif position.trailing_activated:
                new_stop = current_price * (1 - self.config.trailing_distance / 100)
                if new_stop > position.trailing_stop:
                    position.trailing_stop = new_stop
        
        else:  # SHORT
            pnl_percent = (entry - current_price) / entry * 100
            
            if pnl_percent >= self.config.trailing_activation and not position.trailing_activated:
                position.trailing_activated = True
                position.trailing_stop = current_price * (1 + self.config.trailing_distance / 100)
                logger.info(f"追踪止损激活: {position.symbol} @ {position.trailing_stop:.2f}")
            
            elif position.trailing_activated:
                new_stop = current_price * (1 + self.config.trailing_distance / 100)
                if new_stop < position.trailing_stop:
                    position.trailing_stop = new_stop
    
    def _check_partial_tp(self, position: Position, current_price: float):
        """检查部分止盈"""
        if position.direction == SignalDirection.LONG:
            if not position.tp1_hit and current_price >= position.take_profit_1:
                position.tp1_hit = True
                logger.info(f"TP1 触发: {position.symbol} @ {current_price:.2f}")
            if not position.tp2_hit and current_price >= position.take_profit_2:
                position.tp2_hit = True
                logger.info(f"TP2 触发: {position.symbol} @ {current_price:.2f}")
            if not position.tp3_hit and current_price >= position.take_profit_3:
                position.tp3_hit = True
                logger.info(f"TP3 触发: {position.symbol} @ {current_price:.2f}")
        else:
            if not position.tp1_hit and current_price <= position.take_profit_1:
                position.tp1_hit = True
                logger.info(f"TP1 触发: {position.symbol} @ {current_price:.2f}")
            if not position.tp2_hit and current_price <= position.take_profit_2:
                position.tp2_hit = True
                logger.info(f"TP2 触发: {position.symbol} @ {current_price:.2f}")
            if not position.tp3_hit and current_price <= position.take_profit_3:
                position.tp3_hit = True
                logger.info(f"TP3 触发: {position.symbol} @ {current_price:.2f}")
    
    def check_exit_signals(self, current_prices: Dict[str, float]) -> List[Dict]:
        """检查出场信号"""
        exit_orders = []
        
        for position in self.positions[:]:
            if position.symbol not in current_prices:
                continue
            
            current_price = current_prices[position.symbol]
            exit_reason = None
            
            if position.direction == SignalDirection.LONG:
                # 止损
                if current_price <= position.stop_loss:
                    exit_reason = "STOP_LOSS"
                # 追踪止损
                elif position.trailing_activated and current_price <= position.trailing_stop:
                    exit_reason = "TRAILING_STOP"
                # 止盈
                elif position.tp3_hit and current_price <= position.take_profit_3 * 0.99:
                    exit_reason = "TAKE_PROFIT_3"
            
            else:  # SHORT
                if current_price >= position.stop_loss:
                    exit_reason = "STOP_LOSS"
                elif position.trailing_activated and current_price >= position.trailing_stop:
                    exit_reason = "TRAILING_STOP"
                elif position.tp3_hit and current_price >= position.take_profit_3 * 1.01:
                    exit_reason = "TAKE_PROFIT_3"
            
            if exit_reason:
                exit_orders.append({
                    'position_id': position.id,
                    'symbol': position.symbol,
                    'direction': 'CLOSE_' + position.direction.name,
                    'price': current_price,
                    'reason': exit_reason
                })
        
        return exit_orders
    
    def close_position(self, position_id: str, 
                       exit_price: float, 
                       reason: str) -> Optional[Dict]:
        """平仓"""
        position = None
        for p in self.positions:
            if p.id == position_id:
                position = p
                break
        
        if not position:
            return None
        
        # 计算盈亏
        position.update_pnl(exit_price)
        pnl = position.unrealized_pnl
        pnl_percent = pnl / (position.entry_price * position.size) * 100
        
        # 更新绩效
        self._update_performance(position, pnl, pnl_percent)
        
        # 移除持仓
        self.positions.remove(position)
        position.realized_pnl = pnl
        self.closed_positions.append(position)
        
        # 更新账户
        self.account_balance += pnl
        
        trade_record = {
            'position_id': position.id,
            'symbol': position.symbol,
            'direction': position.direction.name,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'size': position.size,
            'leverage': position.leverage,
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'exit_reason': reason,
            'duration_minutes': (datetime.now() - position.entry_time).total_seconds() / 60,
            'timestamp': datetime.now()
        }
        
        logger.info(f"平仓: {position.symbol} {reason} 盈亏:{pnl:.2f} ({pnl_percent:.2f}%)")
        
        return trade_record
    
    def _update_performance(self, position: Position, pnl: float, pnl_percent: float):
        """更新绩效统计"""
        p = self.performance
        p.total_trades += 1
        p.total_pnl += pnl
        
        if position.direction == SignalDirection.LONG:
            p.long_trades += 1
        else:
            p.short_trades += 1
        
        if pnl > 0:
            p.winning_trades += 1
            p.avg_win = (p.avg_win * (p.winning_trades - 1) + pnl) / p.winning_trades
            p.max_win = max(p.max_win, pnl)
        else:
            p.losing_trades += 1
            p.avg_loss = (p.avg_loss * (p.losing_trades - 1) + pnl) / p.losing_trades
            p.max_loss = min(p.max_loss, pnl)
        
        # 更新风险统计
        if pnl < 0:
            self.risk_manager.daily_loss += pnl
        else:
            self.risk_manager.daily_pnl += pnl
        
        p.calculate_metrics()
    
    def get_portfolio_summary(self) -> Dict:
        """获取投资组合摘要"""
        total_unrealized = sum(p.unrealized_pnl for p in self.positions)
        
        return {
            'account_balance': self.account_balance,
            'total_equity': self.account_balance + total_unrealized,
            'unrealized_pnl': total_unrealized,
            'open_positions': len(self.positions),
            'closed_trades': len(self.closed_positions),
            'performance': {
                'total_trades': self.performance.total_trades,
                'win_rate': self.performance.win_rate,
                'profit_factor': self.performance.profit_factor,
                'total_pnl': self.performance.total_pnl,
                'max_drawdown': self.performance.max_drawdown
            }
        }
    
    def get_position_summary(self) -> List[Dict]:
        """获取持仓摘要"""
        return [{
            'id': p.id,
            'symbol': p.symbol,
            'direction': p.direction.name,
            'entry_price': p.entry_price,
            'size': p.size,
            'leverage': p.leverage,
            'unrealized_pnl': p.unrealized_pnl,
            'stop_loss': p.stop_loss,
            'take_profit_1': p.take_profit_1,
            'tp1_hit': p.tp1_hit,
            'tp2_hit': p.tp2_hit,
            'trailing_activated': p.trailing_activated
        } for p in self.positions]
