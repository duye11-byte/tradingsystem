"""
反馈层模块

提供交易性能分析、在线学习、RLHF训练和反馈存储功能
"""

from .feedback_types import (
    TradeRecord, TradeResult, PerformanceMetrics, LearningSample,
    AgentPerformance, FeedbackConfig, HumanFeedback, FeedbackType,
    ModelUpdate, StrategyInsight, PerformanceAlert
)

from .performance_analyzer import PerformanceAnalyzer
from .online_learner import OnlineLearner
from .rlhf_trainer import RLHFTrainer
from .feedback_store import FeedbackStore
from .feedback_engine import FeedbackEngine, FeedbackMode, FeedbackContext, FeedbackResult

__all__ = [
    # 类型
    'TradeRecord', 'TradeResult', 'PerformanceMetrics', 'LearningSample',
    'AgentPerformance', 'FeedbackConfig', 'HumanFeedback', 'FeedbackType',
    'ModelUpdate', 'StrategyInsight', 'PerformanceAlert',
    # 组件
    'PerformanceAnalyzer', 'OnlineLearner', 'RLHFTrainer', 'FeedbackStore',
    # 引擎
    'FeedbackEngine', 'FeedbackMode', 'FeedbackContext', 'FeedbackResult'
]
