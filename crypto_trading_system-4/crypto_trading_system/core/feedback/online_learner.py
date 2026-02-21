"""
在线学习模块
根据交易结果持续优化模型参数
"""

import numpy as np
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from collections import deque
import logging

from .feedback_types import LearningSample, LearningStatus, ModelUpdate

logger = logging.getLogger(__name__)


class OnlineLearner:
    """
    在线学习模块
    
    根据实时交易结果持续优化系统：
    1. 收集学习样本
    2. 计算奖励信号
    3. 更新模型权重
    4. 验证更新效果
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        min_samples: int = 50
    ):
        """
        初始化在线学习模块
        
        Args:
            learning_rate: 学习率
            batch_size: 批次大小
            min_samples: 最小样本数
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.min_samples = min_samples
        
        # 学习样本
        self.samples: List[LearningSample] = []
        self.pending_samples: deque = deque(maxlen=1000)
        
        # 模型权重
        self.agent_weights: Dict[str, float] = {
            'technical_analyst': 0.35,
            'onchain_analyst': 0.25,
            'sentiment_analyst': 0.20,
            'macro_analyst': 0.20
        }
        
        # 学习统计
        self.learning_stats = {
            'total_samples': 0,
            'processed_samples': 0,
            'total_updates': 0,
            'last_update': None
        }
        
        # 回调函数
        self.update_callbacks: List[Callable[[ModelUpdate], None]] = []
    
    def add_sample(self, sample: LearningSample):
        """
        添加学习样本
        
        Args:
            sample: 学习样本
        """
        self.samples.append(sample)
        self.pending_samples.append(sample)
        
        self.learning_stats['total_samples'] += 1
        
        logger.debug(f"Learning sample added: {sample.id}")
    
    def create_sample(
        self,
        features: Dict[str, float],
        predicted_signal: str,
        predicted_confidence: float,
        actual_result: str,
        actual_pnl: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LearningSample:
        """
        创建学习样本
        
        Args:
            features: 特征
            predicted_signal: 预测信号
            predicted_confidence: 预测置信度
            actual_result: 实际结果
            actual_pnl: 实际盈亏
            metadata: 元数据
            
        Returns:
            LearningSample: 学习样本
        """
        # 计算奖励
        reward = self._calculate_reward(
            predicted_signal,
            actual_result,
            actual_pnl,
            predicted_confidence
        )
        
        sample = LearningSample(
            id=f"sample_{datetime.now().timestamp()}",
            features=features,
            predicted_signal=predicted_signal,
            predicted_confidence=predicted_confidence,
            actual_result=actual_result,
            actual_pnl=actual_pnl,
            reward=reward,
            metadata=metadata or {}
        )
        
        return sample
    
    def learn(self) -> Optional[ModelUpdate]:
        """
        执行学习
        
        Returns:
            ModelUpdate: 模型更新，如果没有足够样本则返回 None
        """
        # 检查样本数量
        if len(self.pending_samples) < self.min_samples:
            logger.debug(f"Not enough samples: {len(self.pending_samples)} < {self.min_samples}")
            return None
        
        # 获取批次样本
        batch = list(self.pending_samples)[:self.batch_size]
        
        logger.info(f"Learning from {len(batch)} samples")
        
        # 更新代理权重
        weight_update = self._update_agent_weights(batch)
        
        # 更新置信度阈值
        confidence_update = self._update_confidence_threshold(batch)
        
        # 创建模型更新
        update = ModelUpdate(
            id=f"update_{datetime.now().timestamp()}",
            component="ensemble",
            update_type="weight_adjustment",
            changes={
                'agent_weights': weight_update,
                'confidence_threshold': confidence_update
            }
        )
        
        # 验证更新
        validation_score = self._validate_update(batch, update)
        update.validation_score = validation_score
        update.is_validated = validation_score > 0.5
        
        # 应用更新
        if update.is_validated:
            self._apply_update(update)
            update.status = "applied"
            update.applied_at = datetime.now()
            
            self.learning_stats['total_updates'] += 1
            self.learning_stats['last_update'] = datetime.now()
            
            # 清空已处理样本
            for sample in batch:
                sample.status = LearningStatus.COMPLETED
                self.learning_stats['processed_samples'] += 1
            
            # 触发回调
            self._notify_update(update)
            
            logger.info(f"Model update applied: score={validation_score:.2%}")
        else:
            update.status = "rejected"
            logger.warning(f"Model update rejected: score={validation_score:.2%}")
        
        # 清空待处理队列
        self.pending_samples.clear()
        
        return update
    
    def _calculate_reward(
        self,
        predicted_signal: str,
        actual_result: str,
        actual_pnl: float,
        confidence: float
    ) -> float:
        """
        计算奖励信号
        
        Args:
            predicted_signal: 预测信号
            actual_result: 实际结果
            actual_pnl: 实际盈亏
            confidence: 置信度
            
        Returns:
            float: 奖励值 (-1 到 1)
        """
        # 基础奖励基于盈亏
        if actual_pnl > 0:
            base_reward = 1.0
        elif actual_pnl < 0:
            base_reward = -1.0
        else:
            base_reward = 0.0
        
        # 根据盈亏大小调整
        pnl_scale = min(abs(actual_pnl) / 1000, 1.0)  # 假设1000为基准
        scaled_reward = base_reward * (0.5 + 0.5 * pnl_scale)
        
        # 置信度奖励/惩罚
        if base_reward > 0 and confidence > 0.7:
            # 高置信度正确预测，额外奖励
            confidence_bonus = 0.2
        elif base_reward < 0 and confidence > 0.7:
            # 高置信度错误预测，额外惩罚
            confidence_bonus = -0.2
        else:
            confidence_bonus = 0.0
        
        # 预测准确性奖励
        prediction_correct = (
            (predicted_signal in ['buy', 'strong_buy'] and actual_result in ['win', 'break_even']) or
            (predicted_signal in ['sell', 'strong_sell'] and actual_result in ['win', 'break_even'])
        )
        
        accuracy_reward = 0.3 if prediction_correct else -0.3
        
        # 综合奖励
        total_reward = scaled_reward + confidence_bonus + accuracy_reward
        
        # 限制在 -1 到 1
        return max(-1.0, min(1.0, total_reward))
    
    def _update_agent_weights(
        self,
        samples: List[LearningSample]
    ) -> Dict[str, float]:
        """
        更新代理权重
        
        Args:
            samples: 学习样本
            
        Returns:
            Dict[str, float]: 新的权重
        """
        # 按代理分组计算平均奖励
        agent_rewards: Dict[str, List[float]] = {}
        
        for sample in samples:
            # 从元数据获取参与的代理
            agents = sample.metadata.get('participating_agents', list(self.agent_weights.keys()))
            
            for agent in agents:
                if agent not in agent_rewards:
                    agent_rewards[agent] = []
                agent_rewards[agent].append(sample.reward)
        
        # 计算每个代理的平均奖励
        agent_avg_rewards = {
            agent: np.mean(rewards) if rewards else 0.0
            for agent, rewards in agent_rewards.items()
        }
        
        # 更新权重
        new_weights = self.agent_weights.copy()
        
        for agent, avg_reward in agent_avg_rewards.items():
            if agent in new_weights:
                # 根据奖励调整权重
                adjustment = avg_reward * self.learning_rate
                new_weights[agent] = max(0.05, min(0.5, new_weights[agent] + adjustment))
        
        # 归一化权重
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            new_weights = {k: v / total_weight for k, v in new_weights.items()}
        
        return new_weights
    
    def _update_confidence_threshold(
        self,
        samples: List[LearningSample]
    ) -> float:
        """
        更新置信度阈值
        
        Args:
            samples: 学习样本
            
        Returns:
            float: 新的置信度阈值
        """
        # 分析高置信度预测的准确性
        high_conf_samples = [s for s in samples if s.predicted_confidence > 0.7]
        
        if len(high_conf_samples) < 10:
            return 0.6  # 默认阈值
        
        # 计算高置信度预测的准确率
        correct_predictions = sum(
            1 for s in high_conf_samples
            if (s.predicted_signal in ['buy', 'strong_buy'] and s.actual_result == 'win') or
               (s.predicted_signal in ['sell', 'strong_sell'] and s.actual_result == 'win')
        )
        
        accuracy = correct_predictions / len(high_conf_samples)
        
        # 根据准确率调整阈值
        if accuracy > 0.7:
            # 高置信度预测准确，可以降低阈值
            new_threshold = 0.55
        elif accuracy < 0.5:
            # 高置信度预测不准确，提高阈值
            new_threshold = 0.7
        else:
            new_threshold = 0.6
        
        return new_threshold
    
    def _validate_update(
        self,
        samples: List[LearningSample],
        update: ModelUpdate
    ) -> float:
        """
        验证更新效果
        
        Args:
            samples: 验证样本
            update: 模型更新
            
        Returns:
            float: 验证分数 (0-1)
        """
        # 使用留一法验证
        validation_samples = samples[-10:]  # 最后10个样本用于验证
        
        # 模拟使用新权重的预测
        correct_predictions = 0
        
        for sample in validation_samples:
            # 简化验证：检查奖励是否改善
            if sample.reward > 0:
                correct_predictions += 1
        
        if len(validation_samples) > 0:
            validation_score = correct_predictions / len(validation_samples)
        else:
            validation_score = 0.5
        
        return validation_score
    
    def _apply_update(self, update: ModelUpdate):
        """应用模型更新"""
        changes = update.changes
        
        # 更新代理权重
        if 'agent_weights' in changes:
            self.agent_weights = changes['agent_weights']
            logger.info(f"Agent weights updated: {self.agent_weights}")
        
        # 更新其他参数
        # ...
    
    def get_agent_weights(self) -> Dict[str, float]:
        """获取当前代理权重"""
        return self.agent_weights.copy()
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """获取学习统计"""
        return {
            **self.learning_stats,
            'pending_samples': len(self.pending_samples),
            'agent_weights': self.agent_weights
        }
    
    def register_update_callback(self, callback: Callable[[ModelUpdate], None]):
        """注册更新回调"""
        self.update_callbacks.append(callback)
    
    def _notify_update(self, update: ModelUpdate):
        """通知更新"""
        for callback in self.update_callbacks:
            try:
                callback(update)
            except Exception as e:
                logger.error(f"Error in update callback: {e}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性
        
        Returns:
            Dict[str, float]: 特征重要性
        """
        if not self.samples:
            return {}
        
        # 计算每个特征与奖励的相关性
        feature_rewards: Dict[str, List[tuple]] = {}
        
        for sample in self.samples:
            for feature, value in sample.features.items():
                if feature not in feature_rewards:
                    feature_rewards[feature] = []
                feature_rewards[feature].append((value, sample.reward))
        
        # 计算相关性
        importance = {}
        
        for feature, values in feature_rewards.items():
            if len(values) < 10:
                continue
            
            feature_values = [v[0] for v in values]
            rewards = [v[1] for v in values]
            
            # 计算相关系数
            if len(set(feature_values)) > 1:
                correlation = np.corrcoef(feature_values, rewards)[0, 1]
                importance[feature] = abs(correlation)
        
        # 归一化
        if importance:
            max_importance = max(importance.values())
            if max_importance > 0:
                importance = {k: v / max_importance for k, v in importance.items()}
        
        # 排序
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20])
