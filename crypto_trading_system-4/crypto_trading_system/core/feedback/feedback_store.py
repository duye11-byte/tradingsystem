"""
反馈存储模块
存储和管理所有反馈数据
"""

import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import logging

from .feedback_types import (
    TradeRecord, LearningSample, HumanFeedback, ModelUpdate, Alert
)

logger = logging.getLogger(__name__)


class FeedbackStore:
    """
    反馈存储
    
    存储和管理所有反馈数据：
    1. 交易记录
    2. 学习样本
    3. 人类反馈
    4. 模型更新
    5. 告警
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        初始化反馈存储
        
        Args:
            storage_path: 存储路径 (可选)
        """
        if storage_path is None:
            storage_path = os.path.join(
                os.path.dirname(__file__), 
                '../../data/feedback'
            )
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 内存缓存
        self.trades: Dict[str, TradeRecord] = {}
        self.samples: Dict[str, LearningSample] = {}
        self.human_feedback: Dict[str, HumanFeedback] = {}
        self.model_updates: Dict[str, ModelUpdate] = {}
        self.alerts: Dict[str, Alert] = {}
        
        # 加载已有数据
        self._load_all()
    
    def save_trade(self, trade: TradeRecord) -> bool:
        """
        保存交易记录
        
        Args:
            trade: 交易记录
            
        Returns:
            bool: 是否成功
        """
        self.trades[trade.id] = trade
        return self._save_to_file('trades', trade.id, self._trade_to_dict(trade))
    
    def get_trade(self, trade_id: str) -> Optional[TradeRecord]:
        """
        获取交易记录
        
        Args:
            trade_id: 交易ID
            
        Returns:
            TradeRecord: 交易记录
        """
        return self.trades.get(trade_id)
    
    def get_all_trades(self) -> List[TradeRecord]:
        """
        获取所有交易记录
        
        Returns:
            List[TradeRecord]: 交易记录列表
        """
        return list(self.trades.values())
    
    def get_trades_by_symbol(self, symbol: str) -> List[TradeRecord]:
        """
        获取指定交易对的交易记录
        
        Args:
            symbol: 交易对符号
            
        Returns:
            List[TradeRecord]: 交易记录列表
        """
        return [t for t in self.trades.values() if t.symbol == symbol]
    
    def save_sample(self, sample: LearningSample) -> bool:
        """
        保存学习样本
        
        Args:
            sample: 学习样本
            
        Returns:
            bool: 是否成功
        """
        self.samples[sample.id] = sample
        return self._save_to_file('samples', sample.id, self._sample_to_dict(sample))
    
    def get_sample(self, sample_id: str) -> Optional[LearningSample]:
        """
        获取学习样本
        
        Args:
            sample_id: 样本ID
            
        Returns:
            LearningSample: 学习样本
        """
        return self.samples.get(sample_id)
    
    def get_all_samples(self) -> List[LearningSample]:
        """
        获取所有学习样本
        
        Returns:
            List[LearningSample]: 学习样本列表
        """
        return list(self.samples.values())
    
    def save_human_feedback(self, feedback: HumanFeedback) -> bool:
        """
        保存人类反馈
        
        Args:
            feedback: 人类反馈
            
        Returns:
            bool: 是否成功
        """
        self.human_feedback[feedback.id] = feedback
        return self._save_to_file('human_feedback', feedback.id, self._feedback_to_dict(feedback))
    
    def get_human_feedback(self, feedback_id: str) -> Optional[HumanFeedback]:
        """
        获取人类反馈
        
        Args:
            feedback_id: 反馈ID
            
        Returns:
            HumanFeedback: 人类反馈
        """
        return self.human_feedback.get(feedback_id)
    
    def get_all_human_feedback(self) -> List[HumanFeedback]:
        """
        获取所有人类反馈
        
        Returns:
            List[HumanFeedback]: 人类反馈列表
        """
        return list(self.human_feedback.values())
    
    def save_model_update(self, update: ModelUpdate) -> bool:
        """
        保存模型更新
        
        Args:
            update: 模型更新
            
        Returns:
            bool: 是否成功
        """
        self.model_updates[update.id] = update
        return self._save_to_file('model_updates', update.id, self._update_to_dict(update))
    
    def get_model_update(self, update_id: str) -> Optional[ModelUpdate]:
        """
        获取模型更新
        
        Args:
            update_id: 更新ID
            
        Returns:
            ModelUpdate: 模型更新
        """
        return self.model_updates.get(update_id)
    
    def get_all_model_updates(self) -> List[ModelUpdate]:
        """
        获取所有模型更新
        
        Returns:
            List[ModelUpdate]: 模型更新列表
        """
        return list(self.model_updates.values())
    
    def save_alert(self, alert: Alert) -> bool:
        """
        保存告警
        
        Args:
            alert: 告警
            
        Returns:
            bool: 是否成功
        """
        self.alerts[alert.id] = alert
        return self._save_to_file('alerts', alert.id, self._alert_to_dict(alert))
    
    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """
        获取告警
        
        Args:
            alert_id: 告警ID
            
        Returns:
            Alert: 告警
        """
        return self.alerts.get(alert_id)
    
    def get_all_alerts(self) -> List[Alert]:
        """
        获取所有告警
        
        Returns:
            List[Alert]: 告警列表
        """
        return list(self.alerts.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取存储统计
        
        Returns:
            Dict: 统计信息
        """
        return {
            'trades': len(self.trades),
            'samples': len(self.samples),
            'human_feedback': len(self.human_feedback),
            'model_updates': len(self.model_updates),
            'alerts': len(self.alerts),
            'storage_path': str(self.storage_path)
        }
    
    def clear_all(self):
        """清空所有数据"""
        self.trades.clear()
        self.samples.clear()
        self.human_feedback.clear()
        self.model_updates.clear()
        self.alerts.clear()
        
        # 删除文件
        for subdir in ['trades', 'samples', 'human_feedback', 'model_updates', 'alerts']:
            path = self.storage_path / subdir
            if path.exists():
                for file in path.glob('*.json'):
                    file.unlink()
        
        logger.info("All feedback data cleared")
    
    def _save_to_file(self, subdir: str, item_id: str, data: Dict) -> bool:
        """保存到文件"""
        try:
            path = self.storage_path / subdir
            path.mkdir(exist_ok=True)
            
            file_path = path / f"{item_id}.json"
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            return True
        except Exception as e:
            logger.error(f"Failed to save to file: {e}")
            return False
    
    def _load_all(self):
        """加载所有数据"""
        # 加载交易记录
        trades_path = self.storage_path / 'trades'
        if trades_path.exists():
            for file in trades_path.glob('*.json'):
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                        trade = self._dict_to_trade(data)
                        self.trades[trade.id] = trade
                except Exception as e:
                    logger.error(f"Failed to load trade: {e}")
        
        logger.info(f"Loaded {len(self.trades)} trades from storage")
    
    def _trade_to_dict(self, trade: TradeRecord) -> Dict:
        """交易记录转字典"""
        return {
            'id': trade.id,
            'symbol': trade.symbol,
            'entry_time': trade.entry_time.isoformat(),
            'entry_price': trade.entry_price,
            'entry_side': trade.entry_side,
            'entry_quantity': trade.entry_quantity,
            'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
            'exit_price': trade.exit_price,
            'exit_reason': trade.exit_reason,
            'realized_pnl': trade.realized_pnl,
            'realized_pnl_pct': trade.realized_pnl_pct,
            'entry_fee': trade.entry_fee,
            'exit_fee': trade.exit_fee,
            'total_fee': trade.total_fee,
            'stop_loss': trade.stop_loss,
            'take_profit': trade.take_profit,
            'risk_amount': trade.risk_amount,
            'risk_reward_ratio': trade.risk_reward_ratio,
            'signal_confidence': trade.signal_confidence,
            'signal_consistency': trade.signal_consistency,
            'result': trade.result.value,
            'is_closed': trade.is_closed,
            'metadata': trade.metadata
        }
    
    def _dict_to_trade(self, data: Dict) -> TradeRecord:
        """字典转交易记录"""
        from .feedback_types import TradeResult
        
        return TradeRecord(
            id=data['id'],
            symbol=data['symbol'],
            entry_time=datetime.fromisoformat(data['entry_time']),
            entry_price=data['entry_price'],
            entry_side=data['entry_side'],
            entry_quantity=data['entry_quantity'],
            exit_time=datetime.fromisoformat(data['exit_time']) if data.get('exit_time') else None,
            exit_price=data.get('exit_price'),
            exit_reason=data.get('exit_reason'),
            realized_pnl=data.get('realized_pnl', 0.0),
            realized_pnl_pct=data.get('realized_pnl_pct', 0.0),
            entry_fee=data.get('entry_fee', 0.0),
            exit_fee=data.get('exit_fee', 0.0),
            total_fee=data.get('total_fee', 0.0),
            stop_loss=data.get('stop_loss'),
            take_profit=data.get('take_profit'),
            risk_amount=data.get('risk_amount', 0.0),
            risk_reward_ratio=data.get('risk_reward_ratio', 0.0),
            signal_confidence=data.get('signal_confidence', 0.0),
            signal_consistency=data.get('signal_consistency', 0.0),
            result=TradeResult(data.get('result', 'open')),
            is_closed=data.get('is_closed', False),
            metadata=data.get('metadata', {})
        )
    
    def _sample_to_dict(self, sample: LearningSample) -> Dict:
        """学习样本转字典"""
        return {
            'id': sample.id,
            'features': sample.features,
            'predicted_signal': sample.predicted_signal,
            'predicted_confidence': sample.predicted_confidence,
            'actual_result': sample.actual_result,
            'actual_pnl': sample.actual_pnl,
            'reward': sample.reward,
            'human_rating': sample.human_rating,
            'timestamp': sample.timestamp.isoformat(),
            'status': sample.status.value,
            'metadata': sample.metadata
        }
    
    def _feedback_to_dict(self, feedback: HumanFeedback) -> Dict:
        """人类反馈转字典"""
        return {
            'id': feedback.id,
            'trade_id': feedback.trade_id,
            'decision_id': feedback.decision_id,
            'signal_id': feedback.signal_id,
            'rating': feedback.rating,
            'comment': feedback.comment,
            'feedback_type': feedback.feedback_type.value,
            'timestamp': feedback.timestamp.isoformat(),
            'metadata': feedback.metadata
        }
    
    def _update_to_dict(self, update: ModelUpdate) -> Dict:
        """模型更新转字典"""
        return {
            'id': update.id,
            'component': update.component,
            'update_type': update.update_type,
            'changes': update.changes,
            'validation_score': update.validation_score,
            'is_validated': update.is_validated,
            'created_at': update.created_at.isoformat(),
            'applied_at': update.applied_at.isoformat() if update.applied_at else None,
            'status': update.status
        }
    
    def _alert_to_dict(self, alert: Alert) -> Dict:
        """告警转字典"""
        return {
            'id': alert.id,
            'alert_type': alert.alert_type,
            'severity': alert.severity,
            'message': alert.message,
            'trigger_value': alert.trigger_value,
            'threshold': alert.threshold,
            'triggered_at': alert.triggered_at.isoformat(),
            'acknowledged_at': alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
            'resolved_at': alert.resolved_at.isoformat() if alert.resolved_at else None,
            'is_acknowledged': alert.is_acknowledged,
            'is_resolved': alert.is_resolved,
            'metadata': alert.metadata
        }
