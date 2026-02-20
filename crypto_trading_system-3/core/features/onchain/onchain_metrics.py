"""
链上指标计算模块
实现各种链上数据分析指标的计算
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from ..feature_types import OnchainFeatures

logger = logging.getLogger(__name__)


class OnchainMetrics:
    """
    链上指标计算器
    
    提供各种链上数据分析指标的计算，包括：
    - 交易所资金流向
    - 鲸鱼活动监控
    - 网络活跃度
    - 供应分布
    - 矿工指标
    """
    
    def __init__(self):
        """初始化链上指标计算器"""
        pass
    
    def calculate_all(self, data: Dict[str, any]) -> OnchainFeatures:
        """
        计算所有链上指标
        
        Args:
            data: 包含链上数据的字典
            
        Returns:
            OnchainFeatures: 所有链上指标
        """
        features = OnchainFeatures()
        
        # 交易所流向
        features = self._calculate_exchange_flows(features, data)
        
        # 鲸鱼活动
        features = self._calculate_whale_activity(features, data)
        
        # 网络活跃度
        features = self._calculate_network_activity(features, data)
        
        # 供应分布
        features = self._calculate_supply_distribution(features, data)
        
        # 矿工指标
        features = self._calculate_miner_metrics(features, data)
        
        # 网络健康
        features = self._calculate_network_health(features, data)
        
        return features
    
    def _calculate_exchange_flows(
        self, 
        features: OnchainFeatures, 
        data: Dict[str, any]
    ) -> OnchainFeatures:
        """计算交易所流向指标"""
        # 当前流入流出
        features.exchange_inflow = data.get('exchange_inflow', 0.0)
        features.exchange_outflow = data.get('exchange_outflow', 0.0)
        features.exchange_netflow = features.exchange_outflow - features.exchange_inflow
        
        # 计算变化率 (与24小时前比较)
        inflow_history = data.get('exchange_inflow_history', [])
        outflow_history = data.get('exchange_outflow_history', [])
        
        if len(inflow_history) >= 2:
            prev_inflow = inflow_history[-2] if len(inflow_history) > 1 else inflow_history[0]
            if prev_inflow > 0:
                features.exchange_inflow_change = (
                    (features.exchange_inflow - prev_inflow) / prev_inflow * 100
                )
        
        if len(outflow_history) >= 2:
            prev_outflow = outflow_history[-2] if len(outflow_history) > 1 else outflow_history[0]
            if prev_outflow > 0:
                features.exchange_outflow_change = (
                    (features.exchange_outflow - prev_outflow) / prev_outflow * 100
                )
        
        return features
    
    def _calculate_whale_activity(
        self, 
        features: OnchainFeatures, 
        data: Dict[str, any]
    ) -> OnchainFeatures:
        """计算鲸鱼活动指标"""
        # 大额交易统计
        whale_transactions = data.get('whale_transactions', [])
        
        features.whale_tx_count = len(whale_transactions)
        features.whale_volume = sum(tx.get('amount', 0) for tx in whale_transactions)
        
        # 鲸鱼流入流出 (假设有方向信息)
        whale_in = sum(
            tx.get('amount', 0) for tx in whale_transactions 
            if tx.get('direction') == 'in'
        )
        whale_out = sum(
            tx.get('amount', 0) for tx in whale_transactions 
            if tx.get('direction') == 'out'
        )
        
        features.whale_inflow = whale_in
        features.whale_outflow = whale_out
        
        # 鲸鱼积累指标
        total_whale_volume = whale_in + whale_out
        if total_whale_volume > 0:
            features.whale_accumulation = (whale_in - whale_out) / total_whale_volume
        
        return features
    
    def _calculate_network_activity(
        self, 
        features: OnchainFeatures, 
        data: Dict[str, any]
    ) -> OnchainFeatures:
        """计算网络活跃度指标"""
        # 活跃地址
        features.active_addresses = data.get('active_addresses', 0)
        
        active_addresses_history = data.get('active_addresses_history', [])
        if len(active_addresses_history) >= 2:
            prev_active = active_addresses_history[-2]
            if prev_active > 0:
                features.active_addresses_change = (
                    (features.active_addresses - prev_active) / prev_active * 100
                )
        
        # 交易数量
        features.transaction_count = data.get('transaction_count', 0)
        
        tx_count_history = data.get('transaction_count_history', [])
        if len(tx_count_history) >= 2:
            prev_tx_count = tx_count_history[-2]
            if prev_tx_count > 0:
                features.transaction_count_change = (
                    (features.transaction_count - prev_tx_count) / prev_tx_count * 100
                )
        
        # 平均交易价值
        if features.transaction_count > 0:
            total_volume = data.get('total_transfer_volume', 0)
            features.avg_transaction_value = total_volume / features.transaction_count
        
        return features
    
    def _calculate_supply_distribution(
        self, 
        features: OnchainFeatures, 
        data: Dict[str, any]
    ) -> OnchainFeatures:
        """计算供应分布指标"""
        # 交易所持仓
        features.supply_on_exchanges = data.get('supply_on_exchanges', 0.0)
        
        total_supply = data.get('total_supply', 0.0)
        if total_supply > 0:
            features.supply_on_exchanges_pct = (
                features.supply_on_exchanges / total_supply * 100
            )
        
        # 长期/短期持有者供应
        features.long_term_holder_supply = data.get('long_term_holder_supply', 0.0)
        features.short_term_holder_supply = data.get('short_term_holder_supply', 0.0)
        
        return features
    
    def _calculate_miner_metrics(
        self, 
        features: OnchainFeatures, 
        data: Dict[str, any]
    ) -> OnchainFeatures:
        """计算矿工指标"""
        features.miner_revenue = data.get('miner_revenue', 0.0)
        features.miner_outflow = data.get('miner_outflow', 0.0)
        
        return features
    
    def _calculate_network_health(
        self, 
        features: OnchainFeatures, 
        data: Dict[str, any]
    ) -> OnchainFeatures:
        """计算网络健康指标"""
        features.hash_rate = data.get('hash_rate', 0.0)
        features.difficulty = data.get('difficulty', 0.0)
        
        # 算力变化
        hash_rate_history = data.get('hash_rate_history', [])
        if len(hash_rate_history) >= 2:
            prev_hash_rate = hash_rate_history[-2]
            if prev_hash_rate > 0:
                features.hash_rate_change = (
                    (features.hash_rate - prev_hash_rate) / prev_hash_rate * 100
                )
        
        return features
    
    # ==================== 高级指标计算 ====================
    
    def calculate_exchange_flow_ratio(
        self, 
        inflow: float, 
        outflow: float
    ) -> float:
        """
        计算交易所流向比率
        
        Args:
            inflow: 流入量
            outflow: 流出量
            
        Returns:
            float: 流向比率 (>0 表示净流出，<0 表示净流入)
        """
        total = inflow + outflow
        if total == 0:
            return 0.0
        return (outflow - inflow) / total
    
    def calculate_whale_concentration(
        self, 
        whale_holdings: List[float], 
        total_supply: float
    ) -> float:
        """
        计算鲸鱼集中度 (Herfindahl-Hirschman Index 简化版)
        
        Args:
            whale_holdings: 鲸鱼持仓列表
            total_supply: 总供应量
            
        Returns:
            float: 集中度指数 (0-1)
        """
        if total_supply == 0 or not whale_holdings:
            return 0.0
        
        # 计算每个鲸鱼的持仓比例
        shares = [h / total_supply for h in whale_holdings]
        
        # HHI
        hhi = sum(s ** 2 for s in shares)
        
        return hhi
    
    def calculate_network_value_ratio(
        self, 
        market_cap: float, 
        transaction_volume: float
    ) -> float:
        """
        计算网络价值比率 (NVT Ratio)
        
        Args:
            market_cap: 市值
            transaction_volume: 链上交易量
            
        Returns:
            float: NVT 比率
        """
        if transaction_volume == 0:
            return 0.0
        return market_cap / transaction_volume
    
    def calculate_realized_cap(
        self, 
        utxo_data: List[Dict]
    ) -> float:
        """
        计算已实现市值 (Realized Cap)
        
        Args:
            utxo_data: UTXO 数据列表，包含价格和数量
            
        Returns:
            float: 已实现市值
        """
        realized_cap = 0.0
        for utxo in utxo_data:
            amount = utxo.get('amount', 0)
            price_at_creation = utxo.get('price_at_creation', 0)
            realized_cap += amount * price_at_creation
        
        return realized_cap
    
    def calculate_mvrv_ratio(
        self, 
        market_cap: float, 
        realized_cap: float
    ) -> float:
        """
        计算 MVRV 比率
        
        Args:
            market_cap: 市值
            realized_cap: 已实现市值
            
        Returns:
            float: MVRV 比率
        """
        if realized_cap == 0:
            return 0.0
        return market_cap / realized_cap
    
    def calculate_sopr(
        self, 
        spent_outputs: List[Dict]
    ) -> float:
        """
        计算 SOPR (Spent Output Profit Ratio)
        
        Args:
            spent_outputs: 已花费输出列表
            
        Returns:
            float: SOPR 值
        """
        total_spent = 0.0
        total_created = 0.0
        
        for output in spent_outputs:
            amount = output.get('amount', 0)
            price_at_creation = output.get('price_at_creation', 0)
            price_at_spending = output.get('price_at_spending', 0)
            
            total_created += amount * price_at_creation
            total_spent += amount * price_at_spending
        
        if total_created == 0:
            return 1.0
        
        return total_spent / total_created
    
    def calculate_cdd(
        self, 
        spent_outputs: List[Dict]
    ) -> float:
        """
        计算 CDD (Coin Days Destroyed)
        
        Args:
            spent_outputs: 已花费输出列表
            
        Returns:
            float: CDD 值
        """
        total_cdd = 0.0
        
        for output in spent_outputs:
            amount = output.get('amount', 0)
            age_days = output.get('age_days', 0)
            total_cdd += amount * age_days
        
        return total_cdd
    
    def calculate_liveliness(
        self, 
        total_cdd: float, 
        total_supply: float, 
        chain_age_days: float
    ) -> float:
        """
        计算 Liveliness 指标
        
        Args:
            total_cdd: 总 CDD
            total_supply: 总供应量
            chain_age_days: 链龄(天)
            
        Returns:
            float: Liveliness (0-1)
        """
        if total_supply == 0 or chain_age_days == 0:
            return 0.0
        
        max_possible_cdd = total_supply * chain_age_days
        return total_cdd / max_possible_cdd
    
    def detect_whale_accumulation(
        self, 
        whale_transactions: List[Dict], 
        threshold_btc: float = 1000
    ) -> Dict[str, any]:
        """
        检测鲸鱼积累/分发模式
        
        Args:
            whale_transactions: 鲸鱼交易列表
            threshold_btc: 鲸鱼阈值 (BTC)
            
        Returns:
            Dict: 积累/分发分析结果
        """
        accumulation = 0.0
        distribution = 0.0
        
        for tx in whale_transactions:
            amount = tx.get('amount', 0)
            if amount < threshold_btc:
                continue
            
            direction = tx.get('direction', '')
            if direction == 'in':
                accumulation += amount
            elif direction == 'out':
                distribution += amount
        
        net_flow = accumulation - distribution
        total_volume = accumulation + distribution
        
        if total_volume == 0:
            accumulation_ratio = 0.5
        else:
            accumulation_ratio = accumulation / total_volume
        
        # 判断模式
        if accumulation_ratio > 0.6:
            pattern = 'accumulation'
        elif accumulation_ratio < 0.4:
            pattern = 'distribution'
        else:
            pattern = 'neutral'
        
        return {
            'pattern': pattern,
            'accumulation_volume': accumulation,
            'distribution_volume': distribution,
            'net_flow': net_flow,
            'accumulation_ratio': accumulation_ratio
        }
    
    def analyze_exchange_reserves(
        self, 
        reserves_history: List[float], 
        lookback: int = 30
    ) -> Dict[str, any]:
        """
        分析交易所储备变化
        
        Args:
            reserves_history: 储备历史数据
            lookback: 回顾期
            
        Returns:
            Dict: 储备分析结果
        """
        if len(reserves_history) < lookback:
            return {
                'trend': 'insufficient_data',
                'change_pct': 0.0,
                'current': reserves_history[-1] if reserves_history else 0
            }
        
        current = reserves_history[-1]
        past = reserves_history[-lookback]
        
        change_pct = (current - past) / past * 100 if past > 0 else 0.0
        
        # 判断趋势
        if change_pct < -5:
            trend = 'decreasing'  # 储备减少，可能看涨
        elif change_pct > 5:
            trend = 'increasing'  # 储备增加，可能看跌
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'change_pct': change_pct,
            'current': current,
            'past': past
        }
