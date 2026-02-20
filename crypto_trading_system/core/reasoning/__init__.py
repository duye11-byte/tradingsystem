"""
推理层 (Reasoning Layer) - OpenClaw Crypto Trading System

提供基于 Chain-of-Thought 的推理能力、多模型集成和自我一致性验证
"""

from .reasoning_engine import ReasoningEngine
from .cot_engine import ChainOfThoughtEngine
from .ensemble_manager import EnsembleManager
from .consistency_validator import ConsistencyValidator
from .reasoning_types import (
    ReasoningResult,
    CoTStep,
    ModelPrediction,
    ConsensusResult,
    MarketAnalysis,
    TradingSignal,
    SignalType,
    ValidationResult
)

__all__ = [
    'ReasoningEngine',
    'ChainOfThoughtEngine', 
    'EnsembleManager',
    'ConsistencyValidator',
    'ReasoningResult',
    'CoTStep',
    'ModelPrediction',
    'ConsensusResult',
    'MarketAnalysis',
    'TradingSignal',
    'SignalType',
    'ValidationResult'
]
