"""
数据验证器模块
提供数据质量检查、异常值检测和容错机制
"""

import logging
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
import numpy as np
from collections import deque

from .input_types import (
    PriceData, OrderBookData, TradeData, ValidationRule,
    DataSourceType, DataQualityMetrics
)

logger = logging.getLogger(__name__)


class DataValidator:
    """数据验证器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.price_history: Dict[str, deque] = {}
        self.validation_rules: List[ValidationRule] = []
        self.quality_metrics: Dict[DataSourceType, DataQualityMetrics] = {}
        self._setup_default_rules()
        
    def _setup_default_rules(self):
        """设置默认验证规则"""
        self.validation_rules = [
            ValidationRule(
                name="price_not_zero",
                check_func=lambda x: hasattr(x, 'close_price') and x.close_price > 0,
                error_message="价格不能为零或负数",
                is_critical=True
            ),
            ValidationRule(
                name="price_within_range",
                check_func=self._check_price_range,
                error_message="价格超出合理范围",
                is_critical=True
            ),
            ValidationRule(
                name="timestamp_valid",
                check_func=lambda x: hasattr(x, 'timestamp') and self._is_timestamp_valid(x.timestamp),
                error_message="时间戳无效或过于陈旧",
                is_critical=True
            ),
            ValidationRule(
                name="volume_positive",
                check_func=lambda x: hasattr(x, 'volume') and x.volume >= 0,
                error_message="成交量不能为负数",
                is_critical=False
            ),
            ValidationRule(
                name="ohlc_logic",
                check_func=self._check_ohlc_logic,
                error_message="OHLC数据逻辑错误",
                is_critical=True
            ),
            ValidationRule(
                name="orderbook_balance",
                check_func=self._check_orderbook_balance,
                error_message="订单簿数据不平衡",
                is_critical=False
            ),
        ]
    
    def _is_timestamp_valid(self, timestamp: datetime) -> bool:
        """检查时间戳是否有效"""
        now = datetime.now()
        # 时间戳不能是未来时间
        if timestamp > now + timedelta(minutes=1):
            return False
        # 时间戳不能过于陈旧（超过1小时）
        if timestamp < now - timedelta(hours=1):
            return False
        return True
    
    def _check_price_range(self, data: Any) -> bool:
        """检查价格是否在合理范围内（基于历史数据）"""
        if not hasattr(data, 'symbol') or not hasattr(data, 'close_price'):
            return True
        
        symbol = data.symbol
        price = float(data.close_price)
        
        # 获取历史价格统计
        if symbol in self.price_history and len(self.price_history[symbol]) >= 20:
            history = list(self.price_history[symbol])
            prices = [p for p in history if isinstance(p, (int, float, Decimal))]
            if prices:
                prices = [float(p) for p in prices]
                mean_price = np.mean(prices)
                std_price = np.std(prices)
                
                # 价格超出3个标准差视为异常
                if std_price > 0:
                    z_score = abs(price - mean_price) / std_price
                    if z_score > 3:
                        logger.warning(f"价格异常: {symbol} 当前价格 {price}, 均值 {mean_price:.2f}, Z-score {z_score:.2f}")
                        return False
        
        return True
    
    def _check_ohlc_logic(self, data: Any) -> bool:
        """检查OHLC数据逻辑"""
        if not all(hasattr(data, attr) for attr in ['open_price', 'high_price', 'low_price', 'close_price']):
            return True
        
        o, h, l, c = data.open_price, data.high_price, data.low_price, data.close_price
        
        # 最高价应该大于等于开盘价和收盘价
        # 最低价应该小于等于开盘价和收盘价
        # 最高价应该大于等于最低价
        if h < max(o, c) or l > min(o, c) or h < l:
            return False
        
        return True
    
    def _check_orderbook_balance(self, data: Any) -> bool:
        """检查订单簿数据平衡性"""
        if not isinstance(data, OrderBookData):
            return True
        
        # 检查买卖盘是否都有数据
        if not data.bids or not data.asks:
            return False
        
        # 检查最优买价是否小于最优卖价
        if data.best_bid and data.best_ask:
            if data.best_bid.price >= data.best_ask.price:
                return False
        
        return True
    
    def add_rule(self, rule: ValidationRule):
        """添加验证规则"""
        self.validation_rules.append(rule)
    
    def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        验证数据
        
        Returns:
            {
                'valid': bool,
                'errors': List[str],
                'warnings': List[str],
                'critical_errors': List[str]
            }
        """
        errors = []
        warnings = []
        critical_errors = []
        
        for rule in self.validation_rules:
            try:
                if not rule.check_func(data):
                    if rule.is_critical:
                        critical_errors.append(rule.error_message)
                    else:
                        warnings.append(rule.error_message)
            except Exception as e:
                error_msg = f"验证规则 '{rule.name}' 执行失败: {str(e)}"
                logger.error(error_msg)
                if rule.is_critical:
                    critical_errors.append(error_msg)
                else:
                    errors.append(error_msg)
        
        # 更新价格历史
        if isinstance(data, PriceData):
            self._update_price_history(data)
        
        return {
            'valid': len(critical_errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'critical_errors': critical_errors
        }
    
    def _update_price_history(self, data: PriceData):
        """更新价格历史"""
        symbol = data.symbol
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=100)
        self.price_history[symbol].append(data.close_price)
    
    def detect_anomaly(self, data: Any, method: str = "zscore") -> Dict[str, Any]:
        """
        异常检测
        
        Args:
            data: 待检测数据
            method: 检测方法 (zscore, iqr, mad)
        
        Returns:
            {
                'is_anomaly': bool,
                'score': float,
                'method': str,
                'details': Dict
            }
        """
        if not isinstance(data, PriceData):
            return {'is_anomaly': False, 'score': 0, 'method': method, 'details': {}}
        
        symbol = data.symbol
        price = float(data.close_price)
        
        if symbol not in self.price_history or len(self.price_history[symbol]) < 20:
            return {'is_anomaly': False, 'score': 0, 'method': method, 'details': {'insufficient_data': True}}
        
        history = list(self.price_history[symbol])
        prices = [float(p) for p in history]
        
        if method == "zscore":
            return self._zscore_detection(price, prices)
        elif method == "iqr":
            return self._iqr_detection(price, prices)
        elif method == "mad":
            return self._mad_detection(price, prices)
        else:
            return {'is_anomaly': False, 'score': 0, 'method': method, 'details': {'unknown_method': True}}
    
    def _zscore_detection(self, price: float, prices: List[float]) -> Dict[str, Any]:
        """Z-score异常检测"""
        mean = np.mean(prices)
        std = np.std(prices)
        
        if std == 0:
            return {'is_anomaly': False, 'score': 0, 'method': 'zscore', 'details': {}}
        
        z_score = abs(price - mean) / std
        threshold = self.config.get('zscore_threshold', 3.0)
        
        return {
            'is_anomaly': z_score > threshold,
            'score': z_score,
            'method': 'zscore',
            'details': {
                'mean': mean,
                'std': std,
                'threshold': threshold,
                'current_price': price
            }
        }
    
    def _iqr_detection(self, price: float, prices: List[float]) -> Dict[str, Any]:
        """IQR异常检测"""
        q1 = np.percentile(prices, 25)
        q3 = np.percentile(prices, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        is_anomaly = price < lower_bound or price > upper_bound
        
        # 计算异常分数 (0-1)
        if price < lower_bound:
            score = (lower_bound - price) / iqr if iqr > 0 else 0
        elif price > upper_bound:
            score = (price - upper_bound) / iqr if iqr > 0 else 0
        else:
            score = 0
        
        return {
            'is_anomaly': is_anomaly,
            'score': min(score, 10),  # 限制最大分数
            'method': 'iqr',
            'details': {
                'q1': q1,
                'q3': q3,
                'iqr': iqr,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'current_price': price
            }
        }
    
    def _mad_detection(self, price: float, prices: List[float]) -> Dict[str, Any]:
        """MAD异常检测"""
        median = np.median(prices)
        mad = np.median([abs(p - median) for p in prices])
        
        if mad == 0:
            return {'is_anomaly': False, 'score': 0, 'method': 'mad', 'details': {}}
        
        modified_z_score = 0.6745 * (price - median) / mad
        threshold = self.config.get('mad_threshold', 3.5)
        
        return {
            'is_anomaly': abs(modified_z_score) > threshold,
            'score': abs(modified_z_score),
            'method': 'mad',
            'details': {
                'median': median,
                'mad': mad,
                'threshold': threshold,
                'current_price': price
            }
        }
    
    def update_quality_metrics(self, source: DataSourceType, metrics: DataQualityMetrics):
        """更新数据质量指标"""
        self.quality_metrics[source] = metrics
    
    def get_quality_report(self) -> Dict[str, Any]:
        """获取数据质量报告"""
        report = {
            'timestamp': datetime.now(),
            'sources': {},
            'overall': {
                'completeness': 0.0,
                'accuracy': 0.0,
                'timeliness': 0.0,
                'consistency': 0.0
            }
        }
        
        if not self.quality_metrics:
            return report
        
        for source, metrics in self.quality_metrics.items():
            report['sources'][source.value] = {
                'completeness': metrics.completeness,
                'accuracy': metrics.accuracy,
                'timeliness': metrics.timeliness,
                'consistency': metrics.consistency,
                'error_rate': metrics.error_rate,
                'latency_ms': metrics.latency_ms
            }
        
        # 计算整体指标
        sources_count = len(self.quality_metrics)
        if sources_count > 0:
            report['overall']['completeness'] = sum(m.completeness for m in self.quality_metrics.values()) / sources_count
            report['overall']['accuracy'] = sum(m.accuracy for m in self.quality_metrics.values()) / sources_count
            report['overall']['timeliness'] = sum(m.timeliness for m in self.quality_metrics.values()) / sources_count
            report['overall']['consistency'] = sum(m.consistency for m in self.quality_metrics.values()) / sources_count
        
        return report


class DataCache:
    """数据缓存管理器"""
    
    def __init__(self, default_ttl: int = 60):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """启动缓存清理任务"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop(self):
        """停止缓存清理任务"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    async def _cleanup_loop(self):
        """定期清理过期缓存"""
        while True:
            try:
                await asyncio.sleep(30)  # 每30秒清理一次
                self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"缓存清理错误: {e}")
    
    def _cleanup_expired(self):
        """清理过期数据"""
        now = datetime.now()
        expired_keys = [
            key for key, item in self.cache.items()
            if now > item['expires_at']
        ]
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.debug(f"清理 {len(expired_keys)} 个过期缓存项")
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存数据"""
        if key in self.cache:
            item = self.cache[key]
            if datetime.now() <= item['expires_at']:
                return item['data']
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, data: Any, ttl: Optional[int] = None):
        """设置缓存数据"""
        ttl = ttl or self.default_ttl
        self.cache[key] = {
            'data': data,
            'expires_at': datetime.now() + timedelta(seconds=ttl),
            'created_at': datetime.now()
        }
    
    def delete(self, key: str):
        """删除缓存数据"""
        if key in self.cache:
            del self.cache[key]
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        now = datetime.now()
        total = len(self.cache)
        expired = sum(1 for item in self.cache.values() if now > item['expires_at'])
        
        return {
            'total_items': total,
            'expired_items': expired,
            'valid_items': total - expired
        }
