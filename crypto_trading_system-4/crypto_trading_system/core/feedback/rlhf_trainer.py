"""
RLHF (基于人类反馈的强化学习) 模块
使用人类反馈来优化决策策略
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

from .feedback_types import HumanFeedback, LearningSample, ModelUpdate, FeedbackType

logger = logging.getLogger(__name__)


class RLHFTrainer:
    """
    RLHF 训练器
    
    使用人类反馈来优化决策策略：
    1. 收集人类反馈
    2. 训练奖励模型
    3. 优化决策策略
    4. 验证改进效果
    """
    
    def __init__(
        self,
        human_feedback_weight: float = 0.3,
        auto_feedback_weight: float = 0.7
    ):
        """
        初始化 RLHF 训练器
        
        Args:
            human_feedback_weight: 人类反馈权重
            auto_feedback_weight: 自动反馈权重
        """
        self.human_feedback_weight = human_feedback_weight
        self.auto_feedback_weight = auto_feedback_weight
        
        # 人类反馈历史
        self.human_feedback: List[HumanFeedback] = []
        
        # 偏好对 (用于训练奖励模型)
        self.preference_pairs: List[Tuple[LearningSample, LearningSample, int]] = []
        
        # 奖励模型参数 (简化版)
        self.reward_model = {
            'confidence_weight': 0.3,
            'consistency_weight': 0.2,
            'risk_reward_weight': 0.3,
            'market_condition_weight': 0.2
        }
        
        # 统计
        self.stats = {
            'total_human_feedback': 0,
            'avg_human_rating': 0.0,
            'preference_pairs': 0
        }
    
    def add_human_feedback(self, feedback: HumanFeedback):
        """
        添加人类反馈
        
        Args:
            feedback: 人类反馈
        """
        self.human_feedback.append(feedback)
        
        self.stats['total_human_feedback'] += 1
        
        # 更新平均评分
        ratings = [f.rating for f in self.human_feedback]
        self.stats['avg_human_rating'] = np.mean(ratings)
        
        logger.info(
            f"Human feedback added: {feedback.feedback_type.value} "
            f"rating={feedback.rating}"
        )
    
    def create_human_feedback(
        self,
        trade_id: Optional[str] = None,
        decision_id: Optional[str] = None,
        rating: int = 3,
        comment: Optional[str] = None
    ) -> HumanFeedback:
        """
        创建人类反馈
        
        Args:
            trade_id: 交易ID
            decision_id: 决策ID
            rating: 评分 (1-5)
            comment: 评论
            
        Returns:
            HumanFeedback: 人类反馈
        """
        feedback = HumanFeedback(
            id=f"hf_{datetime.now().timestamp()}",
            trade_id=trade_id,
            decision_id=decision_id,
            rating=rating,
            comment=comment,
            feedback_type=FeedbackType.HUMAN_RATING
        )
        
        return feedback
    
    def build_preference_pairs(self, samples: List[LearningSample]):
        """
        构建偏好对
        
        Args:
            samples: 学习样本
        """
        # 按人类反馈分组（支持trade_id和decision_id）
        feedback_by_id: Dict[str, HumanFeedback] = {}
        
        for feedback in self.human_feedback:
            if feedback.trade_id:
                feedback_by_id[feedback.trade_id] = feedback
            if feedback.decision_id:
                feedback_by_id[feedback.decision_id] = feedback
        
        # 构建偏好对
        for i, sample1 in enumerate(samples):
            # 尝试多种ID匹配
            sample_id1 = (sample1.metadata.get('trade_id') or 
                         sample1.metadata.get('decision_id') or 
                         sample1.id)
            
            if sample_id1 not in feedback_by_id:
                continue
            
            feedback1 = feedback_by_id[sample_id1]
            
            for sample2 in samples[i+1:]:
                sample_id2 = (sample2.metadata.get('trade_id') or 
                             sample2.metadata.get('decision_id') or 
                             sample2.id)
                
                if sample_id2 not in feedback_by_id:
                    continue
                
                feedback2 = feedback_by_id[sample_id2]
                
                # 如果评分不同，构建偏好对
                if feedback1.rating != feedback2.rating:
                    # 评分高的为偏好
                    if feedback1.rating > feedback2.rating:
                        preferred = sample1
                        less_preferred = sample2
                    else:
                        preferred = sample2
                        less_preferred = sample1
                    
                    self.preference_pairs.append((preferred, less_preferred, 1))
        
        self.stats['preference_pairs'] = len(self.preference_pairs)
        
        logger.info(f"Built {len(self.preference_pairs)} preference pairs")
    
    def train_reward_model(self) -> Dict[str, float]:
        """
        训练奖励模型
        
        Returns:
            Dict[str, float]: 更新后的奖励模型参数
        """
        if len(self.preference_pairs) < 10:
            logger.warning("Not enough preference pairs for training")
            return self.reward_model
        
        logger.info(f"Training reward model with {len(self.preference_pairs)} pairs")
        
        # 简化版：根据偏好对调整权重
        # 实际应用中应该使用更复杂的优化算法
        
        # 计算每个特征在偏好对中的表现
        feature_performance = {
            'confidence': [],
            'consistency': [],
            'risk_reward': [],
            'market_condition': []
        }
        
        for preferred, less_preferred, _ in self.preference_pairs:
            # 置信度
            if preferred.predicted_confidence > less_preferred.predicted_confidence:
                feature_performance['confidence'].append(1)
            else:
                feature_performance['confidence'].append(-1)
            
            # 一致性 (从元数据获取)
            pref_consistency = preferred.metadata.get('consistency_score', 0.5)
            less_consistency = less_preferred.metadata.get('consistency_score', 0.5)
            
            if pref_consistency > less_consistency:
                feature_performance['consistency'].append(1)
            else:
                feature_performance['consistency'].append(-1)
            
            # 风险收益比
            pref_rr = preferred.metadata.get('risk_reward_ratio', 1.0)
            less_rr = less_preferred.metadata.get('risk_reward_ratio', 1.0)
            
            if pref_rr > less_rr:
                feature_performance['risk_reward'].append(1)
            else:
                feature_performance['risk_reward'].append(-1)
        
        # 根据表现调整权重
        new_model = self.reward_model.copy()
        
        for feature, performances in feature_performance.items():
            if performances:
                avg_performance = np.mean(performances)
                
                # 调整权重
                if feature == 'confidence':
                    new_model['confidence_weight'] = max(0.1, min(0.5,
                        new_model['confidence_weight'] + avg_performance * 0.05))
                elif feature == 'consistency':
                    new_model['consistency_weight'] = max(0.1, min(0.5,
                        new_model['consistency_weight'] + avg_performance * 0.05))
                elif feature == 'risk_reward':
                    new_model['risk_reward_weight'] = max(0.1, min(0.5,
                        new_model['risk_reward_weight'] + avg_performance * 0.05))
        
        # 归一化权重
        total_weight = sum(new_model.values())
        if total_weight > 0:
            new_model = {k: v / total_weight for k, v in new_model.items()}
        
        self.reward_model = new_model
        
        logger.info(f"Reward model updated: {self.reward_model}")
        
        return self.reward_model
    
    def compute_reward(self, sample: LearningSample) -> float:
        """
        计算综合奖励 (人类反馈 + 自动反馈)
        
        Args:
            sample: 学习样本
            
        Returns:
            float: 综合奖励
        """
        # 自动奖励
        auto_reward = sample.reward
        
        # 人类反馈奖励
        human_reward = 0.0
        
        # 查找相关的人类反馈
        trade_id = sample.metadata.get('trade_id')
        if trade_id:
            for feedback in self.human_feedback:
                if feedback.trade_id == trade_id:
                    # 将 1-5 评分转换为 -1 到 1
                    human_reward = (feedback.rating - 3) / 2
                    break
        
        # 综合奖励
        combined_reward = (
            self.auto_feedback_weight * auto_reward +
            self.human_feedback_weight * human_reward
        )
        
        return combined_reward
    
    def optimize_policy(self, samples: List[LearningSample]) -> ModelUpdate:
        """
        优化决策策略
        
        Args:
            samples: 学习样本
            
        Returns:
            ModelUpdate: 策略更新
        """
        logger.info(f"Optimizing policy with {len(samples)} samples")
        
        # 计算每个决策特征的综合奖励
        feature_rewards: Dict[str, List[float]] = {}
        
        for sample in samples:
            reward = self.compute_reward(sample)
            
            # 按特征分组
            for feature in ['signal_confidence', 'consistency_score', 'risk_reward_ratio']:
                value = sample.metadata.get(feature, 0.5)
                
                if feature not in feature_rewards:
                    feature_rewards[feature] = []
                
                feature_rewards[feature].append((value, reward))
        
        # 分析最优特征范围
        optimal_ranges = {}
        
        for feature, values in feature_rewards.items():
            if len(values) < 10:
                continue
            
            # 按特征值排序
            sorted_values = sorted(values, key=lambda x: x[0])
            
            # 找到奖励最高的范围
            best_range = None
            best_avg_reward = -float('inf')
            
            for i in range(len(sorted_values) - 5):
                window = sorted_values[i:i+5]
                avg_reward = np.mean([v[1] for v in window])
                
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    best_range = (window[0][0], window[-1][0])
            
            if best_range:
                optimal_ranges[feature] = {
                    'min': best_range[0],
                    'max': best_range[1],
                    'avg_reward': best_avg_reward
                }
        
        # 创建策略更新
        update = ModelUpdate(
            id=f"rlhf_update_{datetime.now().timestamp()}",
            component="decision_policy",
            update_type="policy_optimization",
            changes={
                'reward_model': self.reward_model,
                'optimal_ranges': optimal_ranges,
                'human_feedback_stats': {
                    'total': self.stats['total_human_feedback'],
                    'avg_rating': self.stats['avg_human_rating']
                }
            }
        )
        
        # 验证更新
        validation_score = self._validate_policy_update(samples, update)
        update.validation_score = validation_score
        update.is_validated = validation_score > 0.5
        
        if update.is_validated:
            update.status = "validated"
            logger.info(f"Policy update validated: score={validation_score:.2%}")
        else:
            update.status = "rejected"
            logger.warning(f"Policy update rejected: score={validation_score:.2%}")
        
        return update
    
    def _validate_policy_update(
        self,
        samples: List[LearningSample],
        update: ModelUpdate
    ) -> float:
        """
        验证策略更新
        
        Args:
            samples: 验证样本
            update: 策略更新
            
        Returns:
            float: 验证分数
        """
        # 使用新奖励模型计算奖励
        new_rewards = [self.compute_reward(s) for s in samples]
        old_rewards = [s.reward for s in samples]
        
        # 比较平均奖励
        new_avg = np.mean(new_rewards)
        old_avg = np.mean(old_rewards)
        
        # 如果新奖励更高，验证通过
        if new_avg > old_avg:
            return min(1.0, 0.5 + (new_avg - old_avg))
        else:
            return max(0.0, 0.5 + (new_avg - old_avg))
    
    def get_human_feedback_summary(self) -> Dict[str, Any]:
        """
        获取人类反馈摘要
        
        Returns:
            Dict: 人类反馈摘要
        """
        if not self.human_feedback:
            return {
                'total': 0,
                'avg_rating': 0.0,
                'rating_distribution': {}
            }
        
        # 评分分布
        rating_dist = {}
        for feedback in self.human_feedback:
            rating = feedback.rating
            rating_dist[rating] = rating_dist.get(rating, 0) + 1
        
        # 按类型统计
        type_stats = {}
        for feedback in self.human_feedback:
            ftype = feedback.feedback_type.value
            if ftype not in type_stats:
                type_stats[ftype] = {'count': 0, 'avg_rating': 0.0}
            type_stats[ftype]['count'] += 1
        
        for ftype in type_stats:
            ratings = [f.rating for f in self.human_feedback 
                      if f.feedback_type.value == ftype]
            type_stats[ftype]['avg_rating'] = np.mean(ratings)
        
        return {
            'total': self.stats['total_human_feedback'],
            'avg_rating': self.stats['avg_human_rating'],
            'rating_distribution': rating_dist,
            'by_type': type_stats,
            'preference_pairs': self.stats['preference_pairs']
        }
    
    def get_reward_model(self) -> Dict[str, float]:
        """获取当前奖励模型"""
        return self.reward_model.copy()
    
    def reset(self):
        """重置训练器"""
        self.human_feedback.clear()
        self.preference_pairs.clear()
        self.stats = {
            'total_human_feedback': 0,
            'avg_human_rating': 0.0,
            'preference_pairs': 0
        }
        logger.info("RLHF trainer reset")
