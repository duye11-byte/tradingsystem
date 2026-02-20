"""
订单管理器
管理订单生命周期，包括提交、跟踪、取消等
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import logging
import asyncio

from .decision_types import (
    Order,
    OrderStatus,
    DecisionConfig
)

logger = logging.getLogger(__name__)


class OrderManager:
    """
    订单管理器
    
    管理订单的生命周期：
    1. 订单提交
    2. 订单跟踪
    3. 订单取消
    4. 订单状态更新
    """
    
    def __init__(self, config: DecisionConfig):
        """
        初始化订单管理器
        
        Args:
            config: 决策配置
        """
        self.config = config
        
        # 订单存储
        self.orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        
        # 订单回调
        self.status_callbacks: List[Callable[[Order], None]] = []
        
        # 模拟模式 (用于测试)
        self.simulation_mode = True
        
    def submit_order(self, order: Order) -> bool:
        """
        提交订单
        
        Args:
            order: 订单
            
        Returns:
            bool: 是否成功提交
        """
        # 检查订单
        if order.quantity <= 0:
            logger.error(f"Invalid order quantity: {order.quantity}")
            return False
        
        # 设置初始状态
        order.status = OrderStatus.PENDING
        order.created_at = datetime.now()
        
        # 保存订单
        self.orders[order.id] = order
        
        logger.info(
            f"Order submitted: {order.id} {order.symbol} "
            f"{order.side.value} {order.quantity:.4f}"
        )
        
        # 触发回调
        self._notify_status_change(order)
        
        # 如果是模拟模式，立即处理
        if self.simulation_mode:
            asyncio.create_task(self._simulate_order_execution(order))
        
        return True
    
    def submit_orders(self, orders: List[Order]) -> List[bool]:
        """
        批量提交订单
        
        Args:
            orders: 订单列表
            
        Returns:
            List[bool]: 每个订单的提交结果
        """
        return [self.submit_order(order) for order in orders]
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        获取订单
        
        Args:
            order_id: 订单ID
            
        Returns:
            Order: 订单，如果不存在则返回 None
        """
        return self.orders.get(order_id)
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """
        获取指定交易对的所有订单
        
        Args:
            symbol: 交易对符号
            
        Returns:
            List[Order]: 订单列表
        """
        return [
            order for order in self.orders.values()
            if order.symbol == symbol
        ]
    
    def get_active_orders(self) -> List[Order]:
        """
        获取所有活跃订单
        
        Returns:
            List[Order]: 活跃订单列表
        """
        return [
            order for order in self.orders.values()
            if order.is_active()
        ]
    
    def cancel_order(self, order_id: str) -> bool:
        """
        取消订单
        
        Args:
            order_id: 订单ID
            
        Returns:
            bool: 是否成功取消
        """
        order = self.orders.get(order_id)
        if not order:
            logger.warning(f"Order not found: {order_id}")
            return False
        
        if not order.is_active():
            logger.warning(f"Order not active: {order_id}")
            return False
        
        # 更新状态
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now()
        
        # 触发回调
        self._notify_status_change(order)
        
        logger.info(f"Order cancelled: {order_id}")
        
        return True
    
    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        取消所有订单
        
        Args:
            symbol: 交易对符号 (可选，如果指定则只取消该交易对的订单)
            
        Returns:
            int: 取消的订单数量
        """
        cancelled_count = 0
        
        for order in list(self.orders.values()):
            if symbol and order.symbol != symbol:
                continue
            
            if order.is_active():
                if self.cancel_order(order.id):
                    cancelled_count += 1
        
        logger.info(f"Cancelled {cancelled_count} orders")
        
        return cancelled_count
    
    def update_order_status(
        self,
        order_id: str,
        status: OrderStatus,
        filled_quantity: Optional[float] = None,
        average_price: Optional[float] = None
    ) -> bool:
        """
        更新订单状态
        
        Args:
            order_id: 订单ID
            status: 新状态
            filled_quantity: 已成交数量 (可选)
            average_price: 成交均价 (可选)
            
        Returns:
            bool: 是否成功更新
        """
        order = self.orders.get(order_id)
        if not order:
            logger.warning(f"Order not found: {order_id}")
            return False
        
        # 更新状态
        old_status = order.status
        order.status = status
        order.updated_at = datetime.now()
        
        # 更新成交信息
        if filled_quantity is not None:
            order.filled_quantity = filled_quantity
        
        if average_price is not None:
            order.average_fill_price = average_price
        
        # 如果完全成交，记录成交时间
        if status == OrderStatus.FILLED:
            order.filled_at = datetime.now()
            self.order_history.append(order)
        
        # 触发回调
        if old_status != status:
            self._notify_status_change(order)
        
        logger.debug(
            f"Order status updated: {order_id} {old_status.value} -> {status.value}"
        )
        
        return True
    
    def register_status_callback(self, callback: Callable[[Order], None]):
        """
        注册状态变更回调
        
        Args:
            callback: 回调函数
        """
        self.status_callbacks.append(callback)
    
    def _notify_status_change(self, order: Order):
        """通知状态变更"""
        for callback in self.status_callbacks:
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")
    
    async def _simulate_order_execution(self, order: Order):
        """模拟订单执行 (用于测试)"""
        # 模拟延迟
        await asyncio.sleep(0.1)
        
        # 更新为开放状态
        self.update_order_status(order.id, OrderStatus.OPEN)
        
        # 模拟部分成交
        await asyncio.sleep(0.2)
        
        # 计算模拟成交价格
        if order.price:
            fill_price = order.price
        else:
            # 市价单：模拟市场价格
            fill_price = 45000.0  # 模拟价格
        
        # 完全成交
        self.update_order_status(
            order.id,
            OrderStatus.FILLED,
            filled_quantity=order.quantity,
            average_price=fill_price
        )
        
        logger.info(
            f"Order simulated fill: {order.id} "
            f"qty={order.quantity:.4f} price={fill_price:.2f}"
        )
    
    def get_order_statistics(self) -> Dict[str, Any]:
        """
        获取订单统计
        
        Returns:
            Dict: 订单统计
        """
        all_orders = list(self.orders.values())
        
        if not all_orders:
            return {
                'total_orders': 0,
                'active_orders': 0,
                'filled_orders': 0,
                'cancelled_orders': 0
            }
        
        active = sum(1 for o in all_orders if o.is_active())
        filled = sum(1 for o in all_orders if o.is_filled())
        cancelled = sum(1 for o in all_orders if o.status == OrderStatus.CANCELLED)
        
        # 计算平均成交时间
        fill_times = []
        for order in all_orders:
            if order.filled_at and order.created_at:
                fill_time = (order.filled_at - order.created_at).total_seconds()
                fill_times.append(fill_time)
        
        avg_fill_time = sum(fill_times) / len(fill_times) if fill_times else 0
        
        return {
            'total_orders': len(all_orders),
            'active_orders': active,
            'filled_orders': filled,
            'cancelled_orders': cancelled,
            'avg_fill_time_seconds': avg_fill_time,
            'fill_rate': filled / len(all_orders) if all_orders else 0
        }
    
    def get_pending_orders(self, timeout_seconds: Optional[int] = None) -> List[Order]:
        """
        获取待处理订单
        
        Args:
            timeout_seconds: 超时时间 (可选)
            
        Returns:
            List[Order]: 待处理订单列表
        """
        pending = []
        
        for order in self.orders.values():
            if order.status == OrderStatus.PENDING:
                if timeout_seconds:
                    elapsed = (datetime.now() - order.created_at).total_seconds()
                    if elapsed > timeout_seconds:
                        pending.append(order)
                else:
                    pending.append(order)
        
        return pending
    
    def cleanup_old_orders(self, max_age_hours: int = 24) -> int:
        """
        清理旧订单
        
        Args:
            max_age_hours: 最大保留时间 (小时)
            
        Returns:
            int: 清理的订单数量
        """
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        to_remove = []
        
        for order_id, order in self.orders.items():
            if order.created_at < cutoff and not order.is_active():
                to_remove.append(order_id)
        
        for order_id in to_remove:
            del self.orders[order_id]
        
        logger.info(f"Cleaned up {len(to_remove)} old orders")
        
        return len(to_remove)
