"""
反馈层 (Feedback Layer) - OpenClaw Crypto Trading System

负责性能监控、学习和持续优化。

核心组件：
- PerformanceAnalyzer: 性能分析器
- OnlineLearner: 在线学习模块
- RLHFTrainer: 人类反馈强化学习
- FeedbackStore: 反馈存储
- FeedbackEngine: 反馈引擎主入口
"""

from .feedback_types import (
    FeedbackConfig,
    TradeRecord,
    TradeResult,
    PerformanceMetrics,
    LearningSample,
    LearningStatus,
    HumanFeedback,
    FeedbackType,
    ModelUpdate,
    FeedbackSummary,
    AgentPerformance,
    StrategyInsight,
    PerformanceAlert,
    Alert
)

from .performance_analyzer import PerformanceAnalyzer
from .online_learner import OnlineLearner
from .rlhf_trainer import RLHFTrainer
from .feedback_store import FeedbackStore
from .feedback_engine import FeedbackEngine, FeedbackMode, FeedbackContext, FeedbackResult

__all__ = [
    # 类型
    'FeedbackConfig',
    'TradeRecord',
    'TradeResult',
    'PerformanceMetrics',
    'LearningSample',
    'LearningStatus',
    'HumanFeedback',
    'FeedbackType',
    'ModelUpdate',
    'FeedbackSummary',
    'AgentPerformance',
    'StrategyInsight',
    'PerformanceAlert',
    'Alert',
    # 组件
    'PerformanceAnalyzer',
    'OnlineLearner',
    'RLHFTrainer',
    'FeedbackStore',
    # 引擎
    'FeedbackEngine',
    'FeedbackMode',
    'FeedbackContext',
    'FeedbackResult'
]
