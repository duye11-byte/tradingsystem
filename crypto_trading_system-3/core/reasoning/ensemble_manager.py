"""
多模型集成管理器
实现多专家模型的集成推理和共识决策
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass
import logging

from .reasoning_types import (
    ModelPrediction, 
    ConsensusResult, 
    SignalType,
    ExpertAgent
)

logger = logging.getLogger(__name__)


class EnsembleManager:
    """
    多模型集成管理器
    
    管理多个专家代理（技术分析、基本面分析、情绪分析等），
    通过加权投票或共识机制生成最终决策。
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化集成管理器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.agents: List[ExpertAgent] = []
        self.consensus_threshold = self.config.get('consensus_threshold', 0.6)
        self.min_agreement_ratio = self.config.get('min_agreement_ratio', 0.5)
        self.voting_method = self.config.get('voting_method', 'weighted')  # weighted, majority, average
        
        # 初始化专家代理
        self._initialize_agents()
        
    def _initialize_agents(self):
        """初始化默认的专家代理"""
        default_agents = [
            ExpertAgent(
                name="technical_analyst",
                specialization="技术分析",
                model_config={
                    'type': 'technical',
                    'indicators': ['rsi', 'macd', 'bollinger', 'ema', 'volume'],
                    'timeframes': ['1h', '4h', '1d']
                },
                prompt_template=self._get_technical_prompt(),
                weight=0.35,
                confidence_threshold=0.65
            ),
            ExpertAgent(
                name="onchain_analyst",
                specialization="链上分析",
                model_config={
                    'type': 'onchain',
                    'metrics': ['exchange_flow', 'whale_activity', 'network_activity']
                },
                prompt_template=self._get_onchain_prompt(),
                weight=0.25,
                confidence_threshold=0.60
            ),
            ExpertAgent(
                name="sentiment_analyst",
                specialization="情绪分析",
                model_config={
                    'type': 'sentiment',
                    'sources': ['social_media', 'news', 'fear_greed_index']
                },
                prompt_template=self._get_sentiment_prompt(),
                weight=0.20,
                confidence_threshold=0.55
            ),
            ExpertAgent(
                name="macro_analyst",
                specialization="宏观分析",
                model_config={
                    'type': 'macro',
                    'factors': ['btc_correlation', 'market_breadth', 'liquidity']
                },
                prompt_template=self._get_macro_prompt(),
                weight=0.20,
                confidence_threshold=0.55
            )
        ]
        
        self.agents = default_agents
        logger.info(f"Initialized {len(self.agents)} expert agents")
    
    def _get_technical_prompt(self) -> str:
        """获取技术分析提示词"""
        return """你是一位专业的技术分析师，专注于加密货币的技术分析。

你的任务是分析提供的技术指标数据，给出明确的交易信号。

分析维度：
1. 趋势分析：识别当前趋势方向和强度
2. 支撑阻力：识别关键价格水平
3. 动量指标：RSI, MACD, 成交量分析
4. 形态识别：识别常见图表形态

输出格式：
- 信号：BUY/SELL/HOLD/STRONG_BUY/STRONG_SELL
- 置信度：0-1之间的数值
- 推理：简要说明分析依据
- 关键指标：使用的关键指标及其数值"""

    def _get_onchain_prompt(self) -> str:
        """获取链上分析提示词"""
        return """你是一位链上数据分析师，专注于加密货币的链上指标分析。

你的任务是分析链上数据，识别资金流向和鲸鱼活动。

分析维度：
1. 交易所资金流向：净流入/流出
2. 鲸鱼活动：大额转账监控
3. 网络活跃度：活跃地址，交易数量
4. 持仓分布：长期持有者行为

输出格式：
- 信号：BUY/SELL/HOLD/STRONG_BUY/STRONG_SELL
- 置信度：0-1之间的数值
- 推理：简要说明分析依据
- 关键指标：使用的关键链上指标"""

    def _get_sentiment_prompt(self) -> str:
        """获取情绪分析提示词"""
        return """你是一位市场情绪分析师，专注于加密货币的市场情绪分析。

你的任务是分析市场情绪指标，识别极端情绪和潜在反转信号。

分析维度：
1. 恐惧贪婪指数：当前市场情绪状态
2. 社交媒体情绪：Twitter, Reddit等平台情绪
3. 新闻情绪：主流媒体报道情绪
4. 期货资金费率：多空情绪指标

输出格式：
- 信号：BUY/SELL/HOLD/STRONG_BUY/STRONG_SELL
- 置信度：0-1之间的数值
- 推理：简要说明分析依据
- 关键指标：使用的关键情绪指标"""

    def _get_macro_prompt(self) -> str:
        """获取宏观分析提示词"""
        return """你是一位宏观分析师，专注于加密货币的宏观市场分析。

你的任务是分析宏观市场因素，评估整体市场环境。

分析维度：
1. BTC相关性：与比特币的价格相关性
2. 市场广度：整体市场健康状况
3. 流动性状况：市场流动性分析
4. 风险资产表现：与传统风险资产的关系

输出格式：
- 信号：BUY/SELL/HOLD/STRONG_BUY/STRONG_SELL
- 置信度：0-1之间的数值
- 推理：简要说明分析依据
- 关键指标：使用的关键宏观指标"""

    async def predict(
        self,
        symbol: str,
        features: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> ConsensusResult:
        """
        执行多模型预测并生成共识结果
        
        Args:
            symbol: 交易对符号
            features: 特征数据
            market_data: 原始市场数据
            
        Returns:
            ConsensusResult: 共识结果
        """
        # 并行执行所有代理的预测
        predictions = await self._gather_predictions(symbol, features, market_data)
        
        # 计算共识
        consensus = self._calculate_consensus(predictions)
        
        logger.info(
            f"Ensemble prediction completed for {symbol}: "
            f"signal={consensus.final_signal.value}, "
            f"confidence={consensus.consensus_confidence:.2%}, "
            f"agreement={consensus.agreement_ratio:.2%}"
        )
        
        return consensus
    
    async def _gather_predictions(
        self,
        symbol: str,
        features: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> List[ModelPrediction]:
        """并行收集所有代理的预测"""
        tasks = [
            self._run_agent(agent, symbol, features, market_data)
            for agent in self.agents
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        predictions = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Agent {self.agents[i].name} failed: {result}")
            else:
                predictions.append(result)
        
        return predictions
    
    async def _run_agent(
        self,
        agent: ExpertAgent,
        symbol: str,
        features: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> ModelPrediction:
        """运行单个代理的预测"""
        start_time = datetime.now()
        
        # 根据代理类型提取相关特征
        agent_features = self._extract_agent_features(agent, features)
        
        # 执行预测逻辑（这里使用模拟实现，实际应调用LLM或ML模型）
        prediction = await self._execute_agent_prediction(
            agent, symbol, agent_features, market_data
        )
        
        latency = (datetime.now() - start_time).total_seconds() * 1000
        
        return ModelPrediction(
            model_name=agent.name,
            model_type=agent.specialization,
            signal=prediction['signal'],
            confidence=prediction['confidence'],
            reasoning=prediction['reasoning'],
            features_used=list(agent_features.keys()),
            raw_output=prediction,
            latency_ms=latency
        )
    
    def _extract_agent_features(
        self,
        agent: ExpertAgent,
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """提取代理相关的特征"""
        agent_type = agent.model_config.get('type', '')
        
        if agent_type == 'technical':
            return self._extract_technical_features(features)
        elif agent_type == 'onchain':
            return self._extract_onchain_features(features)
        elif agent_type == 'sentiment':
            return self._extract_sentiment_features(features)
        elif agent_type == 'macro':
            return self._extract_macro_features(features)
        else:
            return features
    
    def _extract_technical_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """提取技术特征"""
        technical_keys = [
            'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_lower', 'bb_middle', 'bb_width',
            'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'volume', 'volume_ratio', 'atr_14',
            'close', 'high', 'low', 'open'
        ]
        return {k: features.get(k, 0) for k in technical_keys}
    
    def _extract_onchain_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """提取链上特征"""
        onchain_keys = [
            'exchange_inflow', 'exchange_outflow',
            'whale_tx_count', 'whale_volume',
            'active_addresses_change', 'transaction_count',
            'network_hash_rate', 'mining_difficulty'
        ]
        return {k: features.get(k, 0) for k in onchain_keys}
    
    def _extract_sentiment_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """提取情绪特征"""
        sentiment_keys = [
            'fear_greed_index', 'social_sentiment',
            'news_sentiment', 'funding_rate',
            'long_short_ratio', 'open_interest'
        ]
        return {k: features.get(k, 0) for k in sentiment_keys}
    
    def _extract_macro_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """提取宏观特征"""
        macro_keys = [
            'btc_correlation', 'eth_correlation',
            'market_breadth', 'liquidity_index',
            'dxy_change', 'sp500_change'
        ]
        return {k: features.get(k, 0) for k in macro_keys}
    
    async def _execute_agent_prediction(
        self,
        agent: ExpertAgent,
        symbol: str,
        features: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行代理预测逻辑"""
        # 这里实现基于规则的预测逻辑
        # 实际生产环境应该调用LLM或预训练的ML模型
        
        agent_type = agent.model_config.get('type', '')
        
        if agent_type == 'technical':
            return self._technical_prediction(features)
        elif agent_type == 'onchain':
            return self._onchain_prediction(features)
        elif agent_type == 'sentiment':
            return self._sentiment_prediction(features)
        elif agent_type == 'macro':
            return self._macro_prediction(features)
        else:
            return self._default_prediction(features)
    
    def _technical_prediction(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """技术分析预测"""
        rsi = features.get('rsi_14', 50)
        macd = features.get('macd', 0)
        macd_signal = features.get('macd_signal', 0)
        price = features.get('close', 0)
        sma20 = features.get('sma_20', price)
        
        score = 0
        reasons = []
        
        # RSI 信号
        if rsi < 30:
            score += 2
            reasons.append("RSI超卖")
        elif rsi > 70:
            score -= 2
            reasons.append("RSI超买")
        
        # MACD 信号
        if macd > macd_signal:
            score += 1
            reasons.append("MACD金叉")
        else:
            score -= 1
            reasons.append("MACD死叉")
        
        # 均线信号
        if price > sma20:
            score += 1
            reasons.append("价格在均线上方")
        else:
            score -= 1
            reasons.append("价格在均线下方")
        
        # 确定信号
        if score >= 3:
            signal = SignalType.STRONG_BUY
        elif score >= 1:
            signal = SignalType.BUY
        elif score <= -3:
            signal = SignalType.STRONG_SELL
        elif score <= -1:
            signal = SignalType.SELL
        else:
            signal = SignalType.HOLD
        
        confidence = min(0.95, 0.5 + abs(score) * 0.1)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reasoning': "; ".join(reasons),
            'score': score,
            'indicators': {
                'rsi': rsi,
                'macd': macd,
                'price_vs_sma20': price / sma20 if sma20 > 0 else 1
            }
        }
    
    def _onchain_prediction(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """链上分析预测"""
        net_flow = features.get('exchange_outflow', 0) - features.get('exchange_inflow', 0)
        whale_volume = features.get('whale_volume', 0)
        
        score = 0
        reasons = []
        
        # 资金流向
        if net_flow > 1000000:  # 大额流出
            score += 2
            reasons.append("大额资金流出交易所")
        elif net_flow < -1000000:  # 大额流入
            score -= 2
            reasons.append("大额资金流入交易所")
        
        # 鲸鱼活动
        if whale_volume > 10000000:
            score += 1
            reasons.append("鲸鱼活跃")
        
        if score >= 2:
            signal = SignalType.BUY
        elif score <= -2:
            signal = SignalType.SELL
        else:
            signal = SignalType.HOLD
        
        confidence = min(0.85, 0.5 + abs(score) * 0.15)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reasoning': "; ".join(reasons) if reasons else "链上信号中性",
            'score': score,
            'net_flow': net_flow
        }
    
    def _sentiment_prediction(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """情绪分析预测"""
        fear_greed = features.get('fear_greed_index', 50)
        social = features.get('social_sentiment', 0)
        funding = features.get('funding_rate', 0)
        
        score = 0
        reasons = []
        
        # 恐惧贪婪指数（反向指标）
        if fear_greed < 20:
            score += 2
            reasons.append("极度恐惧，可能反弹")
        elif fear_greed > 80:
            score -= 2
            reasons.append("极度贪婪，可能回调")
        
        # 社交情绪
        if social > 0.3:
            score += 1
            reasons.append("社交情绪积极")
        elif social < -0.3:
            score -= 1
            reasons.append("社交情绪消极")
        
        # 资金费率（反向指标）
        if funding > 0.01:
            score -= 1
            reasons.append("多头支付过高资金费")
        elif funding < -0.01:
            score += 1
            reasons.append("空头支付过高资金费")
        
        if score >= 2:
            signal = SignalType.BUY
        elif score <= -2:
            signal = SignalType.SELL
        else:
            signal = SignalType.HOLD
        
        confidence = min(0.80, 0.5 + abs(score) * 0.12)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reasoning': "; ".join(reasons) if reasons else "情绪中性",
            'score': score,
            'fear_greed': fear_greed
        }
    
    def _macro_prediction(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """宏观分析预测"""
        btc_corr = features.get('btc_correlation', 0.8)
        market_breadth = features.get('market_breadth', 0.5)
        
        score = 0
        reasons = []
        
        # 市场广度
        if market_breadth > 0.7:
            score += 1
            reasons.append("市场广度健康")
        elif market_breadth < 0.3:
            score -= 1
            reasons.append("市场广度恶化")
        
        # BTC相关性（假设分析altcoin）
        if btc_corr > 0.8:
            reasons.append("高度跟随BTC走势")
        
        if score > 0:
            signal = SignalType.BUY
        elif score < 0:
            signal = SignalType.SELL
        else:
            signal = SignalType.HOLD
        
        confidence = min(0.70, 0.5 + abs(score) * 0.15)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reasoning': "; ".join(reasons) if reasons else "宏观环境中性",
            'score': score,
            'market_breadth': market_breadth
        }
    
    def _default_prediction(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """默认预测"""
        return {
            'signal': SignalType.HOLD,
            'confidence': 0.5,
            'reasoning': "默认观望",
            'score': 0
        }
    
    def _calculate_consensus(self, predictions: List[ModelPrediction]) -> ConsensusResult:
        """计算共识结果"""
        if not predictions:
            return ConsensusResult(
                final_signal=SignalType.HOLD,
                consensus_confidence=0.0,
                agreement_ratio=0.0,
                participating_models=[],
                predictions=[],
                dissensus_analysis="无可用预测",
                weighted_score=0.0
            )
        
        # 获取代理权重映射
        agent_weights = {agent.name: agent.weight for agent in self.agents}
        
        # 计算加权得分
        weighted_scores = {}
        total_weight = 0
        
        for pred in predictions:
            weight = agent_weights.get(pred.model_name, 0.25)
            signal_value = self._signal_to_value(pred.signal)
            
            weighted_score = signal_value * pred.confidence * weight
            weighted_scores[pred.model_name] = weighted_score
            total_weight += weight
        
        # 计算最终加权得分
        final_score = sum(weighted_scores.values()) / total_weight if total_weight > 0 else 0
        
        # 确定最终信号
        final_signal = self._value_to_signal(final_score)
        
        # 计算共识置信度
        consensus_confidence = self._calculate_consensus_confidence(predictions, final_signal)
        
        # 计算一致率
        agreement_count = sum(1 for p in predictions if p.signal == final_signal)
        agreement_ratio = agreement_count / len(predictions)
        
        # 分析分歧
        dissensus_analysis = self._analyze_dissensus(predictions, final_signal)
        
        return ConsensusResult(
            final_signal=final_signal,
            consensus_confidence=consensus_confidence,
            agreement_ratio=agreement_ratio,
            participating_models=[p.model_name for p in predictions],
            predictions=predictions,
            dissensus_analysis=dissensus_analysis,
            weighted_score=final_score
        )
    
    def _signal_to_value(self, signal: SignalType) -> float:
        """将信号转换为数值"""
        mapping = {
            SignalType.STRONG_SELL: -2,
            SignalType.SELL: -1,
            SignalType.HOLD: 0,
            SignalType.BUY: 1,
            SignalType.STRONG_BUY: 2
        }
        return mapping.get(signal, 0)
    
    def _value_to_signal(self, value: float) -> SignalType:
        """将数值转换为信号"""
        if value >= 1.5:
            return SignalType.STRONG_BUY
        elif value >= 0.5:
            return SignalType.BUY
        elif value <= -1.5:
            return SignalType.STRONG_SELL
        elif value <= -0.5:
            return SignalType.SELL
        else:
            return SignalType.HOLD
    
    def _calculate_consensus_confidence(
        self,
        predictions: List[ModelPrediction],
        final_signal: SignalType
    ) -> float:
        """计算共识置信度"""
        # 基于预测置信度和一致性的综合计算
        avg_confidence = np.mean([p.confidence for p in predictions])
        
        # 信号一致性
        signal_values = [self._signal_to_value(p.signal) for p in predictions]
        signal_variance = np.var(signal_values)
        consistency = max(0, 1 - signal_variance / 4)  # 归一化到0-1
        
        # 加权组合
        consensus_confidence = avg_confidence * 0.6 + consistency * 0.4
        
        return min(0.99, consensus_confidence)
    
    def _analyze_dissensus(
        self,
        predictions: List[ModelPrediction],
        final_signal: SignalType
    ) -> str:
        """分析分歧原因"""
        buy_signals = [p for p in predictions if p.signal in [SignalType.BUY, SignalType.STRONG_BUY]]
        sell_signals = [p for p in predictions if p.signal in [SignalType.SELL, SignalType.STRONG_SELL]]
        hold_signals = [p for p in predictions if p.signal == SignalType.HOLD]
        
        if len(buy_signals) > 0 and len(sell_signals) > 0:
            return f"存在多空分歧：{len(buy_signals)}个看涨信号 vs {len(sell_signals)}个看跌信号"
        elif len(hold_signals) > len(predictions) / 2:
            return "多数代理建议观望，市场信号不明确"
        else:
            return "信号相对一致"
    
    def add_agent(self, agent: ExpertAgent):
        """添加新的专家代理"""
        self.agents.append(agent)
        logger.info(f"Added new agent: {agent.name}")
    
    def remove_agent(self, agent_name: str):
        """移除专家代理"""
        self.agents = [a for a in self.agents if a.name != agent_name]
        logger.info(f"Removed agent: {agent_name}")
    
    def update_agent_weight(self, agent_name: str, new_weight: float):
        """更新代理权重"""
        for agent in self.agents:
            if agent.name == agent_name:
                agent.weight = new_weight
                logger.info(f"Updated weight for {agent_name}: {new_weight}")
                break
