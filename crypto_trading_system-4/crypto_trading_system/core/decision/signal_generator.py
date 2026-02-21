"""
信号生成器
将推理层的交易信号转换为可执行的交易决策
"""

import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from .decision_types import (
    TradingDecision,
    SignalValidationResult,
    DecisionConfig,
    Order,
    OrderType,
    OrderSide
)
from ..reasoning import TradingSignal, SignalType

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    信号生成器
    
    将推理层的交易信号转换为实际的交易决策，包括：
    1. 信号验证
    2. 信号过滤
    3. 决策生成
    4. 订单创建
    """
    
    def __init__(self, config: DecisionConfig):
        """
        初始化信号生成器
        
        Args:
            config: 决策配置
        """
        self.config = config
        self.signal_history: List[TradingSignal] = []
        self.max_history_size = 100
        
    def validate_signal(
        self,
        signal: TradingSignal,
        portfolio_state: Optional[Dict[str, Any]] = None
    ) -> SignalValidationResult:
        """
        验证交易信号
        
        Args:
            signal: 交易信号
            portfolio_state: 组合状态 (可选)
            
        Returns:
            SignalValidationResult: 验证结果
        """
        passed_filters = []
        failed_filters = []
        warnings = []
        
        # 1. 置信度检查
        if signal.confidence >= self.config.min_confidence:
            passed_filters.append(f"confidence >= {self.config.min_confidence}")
        else:
            failed_filters.append(f"confidence < {self.config.min_confidence}")
        
        # 2. 一致性检查
        if signal.consistency_score >= self.config.min_consistency_score:
            passed_filters.append(f"consistency >= {self.config.min_consistency_score}")
        else:
            failed_filters.append(f"consistency < {self.config.min_consistency_score}")
        
        # 3. 信号有效性检查
        if signal.valid_until and signal.valid_until < datetime.now():
            failed_filters.append("signal_expired")
        else:
            passed_filters.append("signal_valid")
        
        # 4. 信号类型检查
        if signal.signal in [SignalType.HOLD]:
            failed_filters.append("signal_is_hold")
        else:
            passed_filters.append("signal_is_tradable")
        
        # 5. 组合状态检查
        if portfolio_state:
            # 检查是否有足够资金
            available = portfolio_state.get('available_balance', 0)
            required = signal.entry_price * signal.position_size_ratio * 1000 if signal.entry_price else 0
            
            if available >= required:
                passed_filters.append("sufficient_balance")
            else:
                failed_filters.append("insufficient_balance")
                warnings.append(f"可用资金不足: {available} < {required}")
        
        # 计算综合分数
        total_filters = len(passed_filters) + len(failed_filters)
        confidence_score = len(passed_filters) / total_filters if total_filters > 0 else 0
        
        # 风险分数 (基于信号类型和置信度)
        risk_score = self._calculate_risk_score(signal)
        
        # 推荐仓位大小
        recommended_size = self._calculate_recommended_size(signal, confidence_score)
        
        # 是否有效
        is_valid = len(failed_filters) == 0 or confidence_score >= 0.7
        
        return SignalValidationResult(
            is_valid=is_valid,
            confidence_score=confidence_score,
            risk_score=risk_score,
            passed_filters=passed_filters,
            failed_filters=failed_filters,
            recommended_position_size=recommended_size,
            recommended_leverage=1.0,
            warnings=warnings
        )
    
    def generate_decision(
        self,
        signal: TradingSignal,
        portfolio_state: Optional[Dict[str, Any]] = None,
        current_price: Optional[float] = None
    ) -> Optional[TradingDecision]:
        """
        生成交易决策
        
        Args:
            signal: 交易信号
            portfolio_state: 组合状态
            current_price: 当前价格
            
        Returns:
            TradingDecision: 交易决策，如果信号无效则返回 None
        """
        # 验证信号
        validation = self.validate_signal(signal, portfolio_state)
        
        if not validation.is_valid:
            logger.warning(
                f"Signal rejected for {signal.symbol}: "
                f"failed filters: {validation.failed_filters}"
            )
            return None
        
        # 确定行动类型
        action = self._determine_action(signal)
        
        if action == 'HOLD':
            logger.info(f"HOLD decision for {signal.symbol}")
            return None
        
        # 获取价格
        entry_price = current_price or signal.entry_price
        if entry_price is None:
            logger.error(f"No price available for {signal.symbol}")
            return None
        
        # 计算数量
        quantity = self._calculate_quantity(
            signal, 
            validation.recommended_position_size,
            portfolio_state
        )
        
        if quantity <= 0:
            logger.warning(f"Calculated quantity is zero or negative for {signal.symbol}")
            return None
        
        # 创建订单
        orders = self._create_orders(signal, action, entry_price, quantity)
        
        # 计算风险
        risk_amount = self._calculate_risk_amount(
            entry_price, 
            signal.stop_loss, 
            quantity
        )
        
        risk_reward = self._calculate_risk_reward_ratio(
            entry_price,
            signal.stop_loss,
            signal.take_profit
        )
        
        # 创建决策
        decision = TradingDecision(
            id=str(uuid.uuid4()),
            symbol=signal.symbol,
            timestamp=datetime.now(),
            action=action,
            quantity=quantity,
            position_size_ratio=validation.recommended_position_size,
            entry_price=entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            orders=orders,
            risk_amount=risk_amount,
            risk_reward_ratio=risk_reward,
            signal_confidence=signal.confidence,
            consistency_score=signal.consistency_score,
            reasoning_summary=self._summarize_reasoning(signal),
            metadata={
                'validation': validation,
                'signal_type': signal.signal.value
            }
        )
        
        # 保存信号历史
        self._add_to_history(signal)
        
        logger.info(
            f"Decision generated for {signal.symbol}: "
            f"action={action}, quantity={quantity:.4f}, "
            f"confidence={signal.confidence:.1%}"
        )
        
        return decision
    
    def _determine_action(self, signal: TradingSignal) -> str:
        """确定行动类型"""
        action_map = {
            SignalType.BUY: 'OPEN_LONG',
            SignalType.STRONG_BUY: 'OPEN_LONG',
            SignalType.SELL: 'OPEN_SHORT',
            SignalType.STRONG_SELL: 'OPEN_SHORT',
            SignalType.HOLD: 'HOLD'
        }
        
        return action_map.get(signal.signal, 'HOLD')
    
    def _calculate_risk_score(self, signal: TradingSignal) -> float:
        """计算风险分数"""
        score = 0.0
        
        # 基于信号类型的风险
        if signal.signal in [SignalType.STRONG_BUY, SignalType.STRONG_SELL]:
            score += 0.3
        elif signal.signal in [SignalType.BUY, SignalType.SELL]:
            score += 0.2
        
        # 基于置信度的风险调整
        score += (1 - signal.confidence) * 0.3
        
        # 基于一致性的风险调整
        score += (1 - signal.consistency_score) * 0.2
        
        # 如果有止损，降低风险分数
        if signal.stop_loss and signal.entry_price:
            stop_distance = abs(signal.stop_loss - signal.entry_price) / signal.entry_price
            if stop_distance > 0:
                score -= 0.1  # 有止损，风险降低
        
        return max(0.0, min(1.0, score))
    
    def _calculate_recommended_size(
        self, 
        signal: TradingSignal,
        confidence_score: float
    ) -> float:
        """计算推荐仓位大小"""
        # 基础仓位
        base_size = signal.position_size_ratio
        
        # 根据置信度调整
        confidence_adjustment = confidence_score
        
        # 根据一致性调整
        consistency_adjustment = signal.consistency_score
        
        # 综合调整
        adjusted_size = base_size * confidence_adjustment * consistency_adjustment
        
        # 限制最大仓位
        return min(adjusted_size, self.config.max_position_size)
    
    def _calculate_quantity(
        self,
        signal: TradingSignal,
        position_size_ratio: float,
        portfolio_state: Optional[Dict[str, Any]]
    ) -> float:
        """计算交易数量"""
        if portfolio_state is None:
            # 默认数量
            return 0.1
        
        # 获取可用资金
        available = portfolio_state.get('available_balance', 0)
        total_equity = portfolio_state.get('total_equity', available)
        
        # 计算投入金额
        position_value = total_equity * position_size_ratio
        
        # 限制在可用资金范围内
        position_value = min(position_value, available * 0.95)  # 留5%缓冲
        
        # 计算数量
        if signal.entry_price and signal.entry_price > 0:
            quantity = position_value / signal.entry_price
        else:
            quantity = 0.1  # 默认数量
        
        return quantity
    
    def _create_orders(
        self,
        signal: TradingSignal,
        action: str,
        entry_price: float,
        quantity: float
    ) -> List[Order]:
        """创建订单"""
        orders = []
        
        # 确定订单方向
        side = OrderSide.BUY if action in ['OPEN_LONG', 'INCREASE'] else OrderSide.SELL
        
        # 创建主订单 (市价单或限价单)
        main_order = Order(
            id=str(uuid.uuid4()),
            symbol=signal.symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=None,  # 市价单
            metadata={'purpose': 'entry'}
        )
        orders.append(main_order)
        
        # 创建止损订单
        if signal.stop_loss:
            stop_order = Order(
                id=str(uuid.uuid4()),
                symbol=signal.symbol,
                side=OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY,
                order_type=OrderType.STOP_MARKET,
                quantity=quantity,
                stop_price=signal.stop_loss,
                metadata={'purpose': 'stop_loss', 'linked_to': main_order.id}
            )
            orders.append(stop_order)
        
        # 创建止盈订单
        if signal.take_profit:
            tp_order = Order(
                id=str(uuid.uuid4()),
                symbol=signal.symbol,
                side=OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                price=signal.take_profit,
                metadata={'purpose': 'take_profit', 'linked_to': main_order.id}
            )
            orders.append(tp_order)
        
        return orders
    
    def _calculate_risk_amount(
        self,
        entry_price: float,
        stop_loss: Optional[float],
        quantity: float
    ) -> float:
        """计算风险金额"""
        if stop_loss is None or entry_price <= 0:
            return 0.0
        
        risk_per_unit = abs(entry_price - stop_loss)
        return risk_per_unit * quantity
    
    def _calculate_risk_reward_ratio(
        self,
        entry_price: float,
        stop_loss: Optional[float],
        take_profit: Optional[float]
    ) -> float:
        """计算风险收益比"""
        if stop_loss is None or take_profit is None or entry_price <= 0:
            return 0.0
        
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if risk == 0:
            return 0.0
        
        return reward / risk
    
    def _summarize_reasoning(self, signal: TradingSignal) -> str:
        """总结推理过程"""
        if not signal.reasoning_chain:
            return "No reasoning available"
        
        summary_parts = []
        for step in signal.reasoning_chain:
            summary_parts.append(f"{step.step_number}. {step.title}: {step.intermediate_conclusion}")
        
        return "\n".join(summary_parts)
    
    def _add_to_history(self, signal: TradingSignal):
        """添加信号到历史"""
        self.signal_history.append(signal)
        
        if len(self.signal_history) > self.max_history_size:
            self.signal_history = self.signal_history[-self.max_history_size:]
    
    def get_signal_stats(self) -> Dict[str, Any]:
        """获取信号统计"""
        if not self.signal_history:
            return {'total_signals': 0}
        
        signals_by_type = {}
        for signal in self.signal_history:
            signal_type = signal.signal.value
            signals_by_type[signal_type] = signals_by_type.get(signal_type, 0) + 1
        
        avg_confidence = sum(s.confidence for s in self.signal_history) / len(self.signal_history)
        avg_consistency = sum(s.consistency_score for s in self.signal_history) / len(self.signal_history)
        
        return {
            'total_signals': len(self.signal_history),
            'signals_by_type': signals_by_type,
            'avg_confidence': avg_confidence,
            'avg_consistency': avg_consistency
        }
