"""
执行优化器
优化订单执行，包括TWAP、VWAP、冰山订单等策略
"""

import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from .decision_types import (
    Order,
    OrderType,
    OrderSide,
    ExecutionPlan,
    ExecutionStrategy,
    DecisionConfig
)

logger = logging.getLogger(__name__)


class ExecutionOptimizer:
    """
    执行优化器
    
    提供订单执行优化策略：
    1. 立即执行 (Immediate)
    2. TWAP (时间加权平均价格)
    3. VWAP (成交量加权平均价格)
    4. 冰山订单 (Iceberg)
    5. 智能执行 (Smart)
    """
    
    def __init__(self, config: DecisionConfig):
        """
        初始化执行优化器
        
        Args:
            config: 决策配置
        """
        self.config = config
        self.execution_plans: Dict[str, ExecutionPlan] = {}
        
    def create_execution_plan(
        self,
        symbol: str,
        side: OrderSide,
        total_quantity: float,
        target_price: Optional[float] = None,
        strategy: Optional[ExecutionStrategy] = None,
        market_data: Optional[Dict[str, Any]] = None
    ) -> ExecutionPlan:
        """
        创建执行计划
        
        Args:
            symbol: 交易对符号
            side: 订单方向
            total_quantity: 总数量
            target_price: 目标价格 (可选)
            strategy: 执行策略 (可选)
            market_data: 市场数据 (可选)
            
        Returns:
            ExecutionPlan: 执行计划
        """
        if strategy is None:
            strategy = self.config.execution_strategy
        
        # 根据策略创建订单
        if strategy == ExecutionStrategy.IMMEDIATE:
            orders = self._create_immediate_orders(symbol, side, total_quantity)
        
        elif strategy == ExecutionStrategy.TWAP:
            orders = self._create_twap_orders(symbol, side, total_quantity, target_price)
        
        elif strategy == ExecutionStrategy.VWAP:
            orders = self._create_vwap_orders(symbol, side, total_quantity, market_data)
        
        elif strategy == ExecutionStrategy.ICEBERG:
            orders = self._create_iceberg_orders(symbol, side, total_quantity, target_price)
        
        elif strategy == ExecutionStrategy.SMART:
            orders = self._create_smart_orders(symbol, side, total_quantity, target_price, market_data)
        
        else:
            # 默认立即执行
            orders = self._create_immediate_orders(symbol, side, total_quantity)
        
        # 创建执行计划
        plan = ExecutionPlan(
            strategy=strategy,
            orders=orders,
            total_quantity=total_quantity,
            target_price=target_price,
            start_time=datetime.now()
        )
        
        # 保存计划
        self.execution_plans[symbol] = plan
        
        logger.info(
            f"Execution plan created for {symbol}: "
            f"strategy={strategy.value}, orders={len(orders)}, "
            f"quantity={total_quantity:.4f}"
        )
        
        return plan
    
    def _create_immediate_orders(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float
    ) -> List[Order]:
        """创建立即执行的订单"""
        order = Order(
            id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            metadata={'strategy': 'immediate'}
        )
        return [order]
    
    def _create_twap_orders(
        self,
        symbol: str,
        side: OrderSide,
        total_quantity: float,
        target_price: Optional[float],
        num_slices: int = 5,
        duration_minutes: int = 30
    ) -> List[Order]:
        """
        创建TWAP订单
        
        将大订单分割成多个小订单，在指定时间内均匀执行
        """
        orders = []
        slice_quantity = total_quantity / num_slices
        interval_seconds = (duration_minutes * 60) / num_slices
        
        for i in range(num_slices):
            order = Order(
                id=str(uuid.uuid4()),
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT if target_price else OrderType.MARKET,
                quantity=slice_quantity,
                price=target_price,
                metadata={
                    'strategy': 'twap',
                    'slice_index': i,
                    'total_slices': num_slices,
                    'execute_after_seconds': i * interval_seconds
                }
            )
            orders.append(order)
        
        return orders
    
    def _create_vwap_orders(
        self,
        symbol: str,
        side: OrderSide,
        total_quantity: float,
        market_data: Optional[Dict[str, Any]],
        num_slices: int = 5
    ) -> List[Order]:
        """
        创建VWAP订单
        
        根据历史成交量分布来分配订单
        """
        orders = []
        
        # 如果没有市场数据，使用均匀分布
        if not market_data or 'volume_profile' not in market_data:
            volume_weights = [1.0 / num_slices] * num_slices
        else:
            # 使用成交量分布
            volume_profile = market_data['volume_profile']
            total_volume = sum(volume_profile)
            volume_weights = [v / total_volume for v in volume_profile]
        
        # 创建订单
        for i, weight in enumerate(volume_weights[:num_slices]):
            slice_quantity = total_quantity * weight
            
            order = Order(
                id=str(uuid.uuid4()),
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=slice_quantity,
                metadata={
                    'strategy': 'vwap',
                    'slice_index': i,
                    'volume_weight': weight
                }
            )
            orders.append(order)
        
        return orders
    
    def _create_iceberg_orders(
        self,
        symbol: str,
        side: OrderSide,
        total_quantity: float,
        target_price: Optional[float],
        visible_quantity: Optional[float] = None
    ) -> List[Order]:
        """
        创建冰山订单
        
        只显示部分数量，隐藏真实订单规模
        """
        if visible_quantity is None:
            visible_quantity = total_quantity / 10  # 默认显示10%
        
        orders = []
        remaining = total_quantity
        
        while remaining > 0:
            slice_quantity = min(visible_quantity, remaining)
            
            order = Order(
                id=str(uuid.uuid4()),
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT if target_price else OrderType.MARKET,
                quantity=slice_quantity,
                price=target_price,
                metadata={
                    'strategy': 'iceberg',
                    'visible_quantity': visible_quantity,
                    'total_quantity': total_quantity
                }
            )
            orders.append(order)
            
            remaining -= slice_quantity
        
        return orders
    
    def _create_smart_orders(
        self,
        symbol: str,
        side: OrderSide,
        total_quantity: float,
        target_price: Optional[float],
        market_data: Optional[Dict[str, Any]]
    ) -> List[Order]:
        """
        创建智能执行订单
        
        根据市场条件动态选择最优执行策略
        """
        # 分析市场条件
        market_condition = self._analyze_market_condition(market_data)
        
        # 根据市场条件选择策略
        if market_condition == 'low_liquidity':
            # 低流动性：使用冰山订单
            return self._create_iceberg_orders(symbol, side, total_quantity, target_price)
        
        elif market_condition == 'high_volatility':
            # 高波动：使用TWAP分散风险
            return self._create_twap_orders(symbol, side, total_quantity, target_price, num_slices=10)
        
        elif market_condition == 'trending':
            # 趋势市场：立即执行
            return self._create_immediate_orders(symbol, side, total_quantity)
        
        else:
            # 默认：使用VWAP
            return self._create_vwap_orders(symbol, side, total_quantity, market_data)
    
    def _analyze_market_condition(
        self,
        market_data: Optional[Dict[str, Any]]
    ) -> str:
        """分析市场条件"""
        if not market_data:
            return 'normal'
        
        # 检查流动性
        if market_data.get('spread_pct', 0) > 0.005:  # 价差大于0.5%
            return 'low_liquidity'
        
        # 检查波动率
        if market_data.get('volatility', 0) > 0.05:  # 波动率大于5%
            return 'high_volatility'
        
        # 检查趋势
        if market_data.get('trend_strength', 0) > 0.7:
            return 'trending'
        
        return 'normal'
    
    def optimize_order(
        self,
        order: Order,
        market_data: Dict[str, Any]
    ) -> Order:
        """
        优化单个订单
        
        Args:
            order: 原始订单
            market_data: 市场数据
            
        Returns:
            Order: 优化后的订单
        """
        # 获取市场数据
        bid = market_data.get('bid', 0)
        ask = market_data.get('ask', 0)
        spread = ask - bid
        
        # 根据订单类型优化
        if order.order_type == OrderType.LIMIT and order.price:
            # 限价单：优化价格
            if order.side == OrderSide.BUY:
                # 买单：略高于买一价
                optimized_price = bid + spread * 0.1
                order.price = min(order.price, optimized_price)
            else:
                # 卖单：略低于卖一价
                optimized_price = ask - spread * 0.1
                order.price = max(order.price, optimized_price)
        
        elif order.order_type == OrderType.MARKET:
            # 市价单：考虑滑点
            estimated_slippage = self._estimate_slippage(order.quantity, market_data)
            
            if estimated_slippage > self.config.max_slippage:
                logger.warning(
                    f"High slippage expected: {estimated_slippage:.2%} > "
                    f"{self.config.max_slippage:.2%}"
                )
                # 考虑拆分成多个小订单
        
        return order
    
    def _estimate_slippage(
        self,
        quantity: float,
        market_data: Dict[str, Any]
    ) -> float:
        """估计滑点"""
        # 简化模型：基于订单规模和市场深度
        orderbook_depth = market_data.get('orderbook_depth', 1000)
        
        if orderbook_depth <= 0:
            return 0.001  # 默认0.1%滑点
        
        # 订单规模相对于市场深度的比例
        size_ratio = quantity / orderbook_depth
        
        # 估计滑点
        estimated_slippage = min(size_ratio * 0.001, 0.01)  # 最大1%
        
        return estimated_slippage
    
    def get_execution_summary(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取执行摘要
        
        Args:
            symbol: 交易对符号
            
        Returns:
            Dict: 执行摘要
        """
        plan = self.execution_plans.get(symbol)
        if not plan:
            return None
        
        return {
            'symbol': symbol,
            'strategy': plan.strategy.value,
            'total_quantity': plan.total_quantity,
            'executed_quantity': plan.executed_quantity,
            'progress': plan.get_progress(),
            'is_complete': plan.is_complete,
            'average_execution_price': plan.average_execution_price,
            'orders': [
                {
                    'id': order.id,
                    'type': order.order_type.value,
                    'status': order.status.value,
                    'quantity': order.quantity,
                    'filled': order.filled_quantity
                }
                for order in plan.orders
            ]
        }
    
    def cancel_execution(self, symbol: str) -> bool:
        """
        取消执行计划
        
        Args:
            symbol: 交易对符号
            
        Returns:
            bool: 是否成功取消
        """
        plan = self.execution_plans.get(symbol)
        if not plan:
            return False
        
        # 取消所有未成交订单
        for order in plan.orders:
            if order.is_active():
                order.status = OrderStatus.CANCELLED
        
        plan.is_complete = True
        
        logger.info(f"Execution cancelled for {symbol}")
        
        return True
