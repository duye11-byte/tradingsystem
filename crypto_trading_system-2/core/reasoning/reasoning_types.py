"""
推理层类型定义
定义推理层使用的所有数据结构和类型
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum
import numpy as np


class SignalType(Enum):
    """交易信号类型"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


class ConfidenceLevel(Enum):
    """置信度等级"""
    VERY_LOW = 0.0
    LOW = 0.25
    MEDIUM = 0.5
    HIGH = 0.75
    VERY_HIGH = 0.9


@dataclass
class CoTStep:
    """思维链单步推理"""
    step_number: int
    title: str
    reasoning: str
    evidence: Dict[str, Any]
    intermediate_conclusion: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ModelPrediction:
    """单个模型的预测结果"""
    model_name: str
    model_type: str
    signal: SignalType
    confidence: float
    reasoning: str
    features_used: List[str]
    raw_output: Dict[str, Any]
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConsensusResult:
    """多模型共识结果"""
    final_signal: SignalType
    consensus_confidence: float
    agreement_ratio: float
    participating_models: List[str]
    predictions: List[ModelPrediction]
    dissensus_analysis: str
    weighted_score: float


@dataclass
class MarketAnalysis:
    """市场分析结果"""
    symbol: str
    timeframe: str
    analysis_timestamp: datetime
    
    # 技术分析
    technical_summary: str
    trend_direction: str
    support_levels: List[float]
    resistance_levels: List[float]
    key_indicators: Dict[str, float]
    
    # 链上分析
    onchain_summary: str
    exchange_flows: Dict[str, float]
    whale_activity: str
    network_health: str
    
    # 情绪分析
    sentiment_summary: str
    fear_greed_index: int
    social_sentiment: float
    news_sentiment: float
    
    # 综合评估
    overall_assessment: str
    risk_level: str
    opportunity_score: float


@dataclass
class TradingSignal:
    """最终交易信号"""
    symbol: str
    signal: SignalType
    confidence: float
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    position_size_ratio: float
    
    # 推理信息
    reasoning_chain: List[CoTStep]
    market_analysis: MarketAnalysis
    consensus_result: ConsensusResult
    
    # 验证信息
    consistency_check_passed: bool
    consistency_score: float
    
    # 元数据
    generated_at: datetime = field(default_factory=datetime.now)
    valid_until: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningResult:
    """推理层完整结果"""
    success: bool
    signal: Optional[TradingSignal]
    error_message: Optional[str]
    
    # 性能指标
    total_latency_ms: float
    cot_latency_ms: float
    ensemble_latency_ms: float
    consistency_latency_ms: float
    
    # 调试信息
    raw_reasoning: Dict[str, Any]
    debug_info: Dict[str, Any]


@dataclass
class ValidationResult:
    """自我一致性验证结果"""
    is_consistent: bool
    consistency_score: float
    sample_count: int
    agreement_count: int
    variance: float
    confidence_interval: tuple
    detailed_analysis: str


@dataclass
class ExpertAgent:
    """专家代理定义"""
    name: str
    specialization: str
    model_config: Dict[str, Any]
    prompt_template: str
    weight: float
    confidence_threshold: float
    validator: Optional[Callable] = None
