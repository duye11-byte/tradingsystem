"""
Chain-of-Thought 推理引擎
实现逐步推理能力，用于分析复杂的市场状况
"""

import json
import asyncio
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime
import logging

from .reasoning_types import CoTStep, MarketAnalysis, SignalType

logger = logging.getLogger(__name__)


class ChainOfThoughtEngine:
    """
    思维链推理引擎
    
    实现基于 Chain-of-Thought 的逐步推理，将复杂的市场分析
    分解为多个逻辑步骤，每个步骤都有明确的推理和证据支持。
    """
    
    def __init__(self, llm_client=None, config: Dict[str, Any] = None):
        """
        初始化 CoT 引擎
        
        Args:
            llm_client: LLM 客户端（OpenAI, Anthropic 等）
            config: 配置参数
        """
        self.llm_client = llm_client
        self.config = config or {}
        self.max_steps = self.config.get('max_steps', 7)
        self.min_confidence = self.config.get('min_confidence', 0.6)
        self.temperature = self.config.get('temperature', 0.3)
        
        # 加载提示词模板
        self.prompts = self._load_prompts()
        
    def _load_prompts(self) -> Dict[str, str]:
        """加载提示词模板"""
        return {
            'system_prompt': """你是一位专业的加密货币交易分析师，擅长通过逐步推理分析市场状况。
你的任务是将复杂的市场分析分解为清晰的逻辑步骤，每个步骤都要有充分的证据支持。

分析原则：
1. 客观性：基于数据和事实，避免主观臆断
2. 逻辑性：每一步推理都要有清晰的逻辑链条
3. 证据驱动：每个结论都要有具体的数据支持
4. 风险意识：始终考虑潜在风险和不确定性

请按照以下结构进行逐步分析：""",

            'step_1_trend': """步骤 1: 趋势识别
分析当前价格趋势：
- 短期趋势（1-7天）
- 中期趋势（1-4周）
- 长期趋势（1-6个月）

使用提供的技术指标数据，识别：
1. 主要趋势方向（上升/下降/横盘）
2. 趋势强度
3. 关键转折点信号

请提供具体的指标数值和解读。""",

            'step_2_support_resistance': """步骤 2: 支撑与阻力分析
基于历史价格数据和技术指标：
1. 识别关键支撑位（至少3个）
2. 识别关键阻力位（至少3个）
3. 分析价格与这些水平的关系
4. 评估突破或反弹的可能性

请提供具体的价位和概率评估。""",

            'step_3_momentum': """步骤 3: 动量分析
分析市场动量：
1. RSI 指标解读（超买/超卖状态）
2. MACD 信号（金叉/死叉，柱状图趋势）
3. 成交量分析（放量/缩量，异常波动）
4. 布林带状态（收窄/扩张，价格位置）

评估动量是否支持当前趋势持续。""",

            'step_4_onchain': """步骤 4: 链上数据分析
分析链上指标：
1. 交易所资金流向（流入/流出趋势）
2. 大额转账活动（鲸鱼行为）
3. 网络活跃度（活跃地址数，交易数）
4. 持仓分布变化

评估这些信号对价格的潜在影响。""",

            'step_5_sentiment': """步骤 5: 市场情绪分析
综合分析市场情绪：
1. 恐惧贪婪指数解读
2. 社交媒体情绪趋势
3. 新闻情绪分析
4. 期货资金费率（多空情绪）

评估情绪是否极端或存在反转信号。""",

            'step_6_risk': """步骤 6: 风险评估
全面评估交易风险：
1. 波动性分析（ATR, 历史波动率）
2. 相关性风险（与BTC/ETH的相关性）
3. 流动性风险（成交量，买卖价差）
4. 宏观风险（监管，市场情绪）

给出风险等级和具体风险因素。""",

            'step_7_synthesis': """步骤 7: 综合判断与信号生成
综合以上所有分析：
1. 总结主要看涨因素
2. 总结主要看跌因素
3. 权衡多空力量对比
4. 生成最终交易信号（强烈买入/买入/持有/卖出/强烈卖出）
5. 给出置信度评分（0-100%）
6. 建议仓位大小（0-100%）

必须提供清晰的推理过程和最终结论。"""
        }
    
    async def analyze_market(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        features: Dict[str, Any]
    ) -> List[CoTStep]:
        """
        执行完整的市场分析思维链
        
        Args:
            symbol: 交易对符号
            market_data: 原始市场数据
            features: 特征工程层输出的特征
            
        Returns:
            List[CoTStep]: 完整的推理步骤列表
        """
        reasoning_chain = []
        
        try:
            # 步骤 1: 趋势识别
            step1 = await self._analyze_trend(symbol, features)
            reasoning_chain.append(step1)
            
            # 步骤 2: 支撑阻力分析
            step2 = await self._analyze_support_resistance(symbol, features)
            reasoning_chain.append(step2)
            
            # 步骤 3: 动量分析
            step3 = await self._analyze_momentum(symbol, features)
            reasoning_chain.append(step3)
            
            # 步骤 4: 链上数据分析
            step4 = await self._analyze_onchain(symbol, features)
            reasoning_chain.append(step4)
            
            # 步骤 5: 情绪分析
            step5 = await self._analyze_sentiment(symbol, features)
            reasoning_chain.append(step5)
            
            # 步骤 6: 风险评估
            step6 = await self._analyze_risk(symbol, features, reasoning_chain)
            reasoning_chain.append(step6)
            
            # 步骤 7: 综合判断
            step7 = await self._synthesize_analysis(symbol, features, reasoning_chain)
            reasoning_chain.append(step7)
            
            logger.info(f"CoT analysis completed for {symbol} with {len(reasoning_chain)} steps")
            return reasoning_chain
            
        except Exception as e:
            logger.error(f"Error in CoT analysis for {symbol}: {e}")
            raise
    
    async def _analyze_trend(self, symbol: str, features: Dict[str, Any]) -> CoTStep:
        """分析趋势"""
        # 提取趋势相关特征
        trend_features = {
            'sma_20': features.get('sma_20', 0),
            'sma_50': features.get('sma_50', 0),
            'ema_12': features.get('ema_12', 0),
            'ema_26': features.get('ema_26', 0),
            'price': features.get('close', 0),
            'price_change_1d': features.get('price_change_1d', 0),
            'price_change_7d': features.get('price_change_7d', 0),
            'price_change_30d': features.get('price_change_30d', 0),
        }
        
        # 计算趋势信号
        short_trend = "上升" if trend_features['price_change_1d'] > 0 else "下降"
        medium_trend = "上升" if trend_features['price_change_7d'] > 0 else "下降"
        long_trend = "上升" if trend_features['price_change_30d'] > 0 else "下降"
        
        # 判断均线排列
        golden_cross = trend_features['sma_20'] > trend_features['sma_50']
        
        reasoning = f"""
基于技术指标分析：

短期趋势（1天）：{short_trend}，价格变化 {trend_features['price_change_1d']:.2f}%
中期趋势（7天）：{medium_trend}，价格变化 {trend_features['price_change_7d']:.2f}%
长期趋势（30天）：{long_trend}，价格变化 {trend_features['price_change_30d']:.2f}%

移动平均线分析：
- SMA20: {trend_features['sma_20']:.2f}
- SMA50: {trend_features['sma_50']:.2f}
- 均线排列: {'多头排列' if golden_cross else '空头排列'}

当前价格 {trend_features['price']:.2f} 相对于SMA20 {'上方' if trend_features['price'] > trend_features['sma_20'] else '下方'}。
"""
        
        # 计算置信度
        confidence = 0.7 if abs(trend_features['price_change_7d']) > 5 else 0.5
        
        return CoTStep(
            step_number=1,
            title="趋势识别",
            reasoning=reasoning,
            evidence=trend_features,
            intermediate_conclusion=f"{medium_trend}趋势，{'多头' if golden_cross else '空头'}排列",
            confidence=confidence
        )
    
    async def _analyze_support_resistance(self, symbol: str, features: Dict[str, Any]) -> CoTStep:
        """分析支撑与阻力"""
        price = features.get('close', 0)
        
        # 计算支撑阻力位（基于布林带和近期高低点）
        bb_upper = features.get('bb_upper', price * 1.05)
        bb_lower = features.get('bb_lower', price * 0.95)
        bb_middle = features.get('bb_middle', price)
        
        # 模拟支撑阻力位计算
        resistance_levels = sorted([
            bb_upper,
            price * 1.03,
            price * 1.08
        ], reverse=True)
        
        support_levels = sorted([
            bb_lower,
            price * 0.97,
            price * 0.92
        ])
        
        # 计算距离最近支撑阻力的距离
        nearest_resistance = min([r for r in resistance_levels if r > price], default=price * 1.1)
        nearest_support = max([s for s in support_levels if s < price], default=price * 0.9)
        
        distance_to_resistance = (nearest_resistance - price) / price * 100
        distance_to_support = (price - nearest_support) / price * 100
        
        reasoning = f"""
支撑与阻力分析：

当前价格: {price:.2f}

关键阻力位：
- R1: {resistance_levels[0]:.2f} (距离 +{distance_to_resistance:.2f}%)
- R2: {resistance_levels[1]:.2f}
- R3: {resistance_levels[2]:.2f}

关键支撑位：
- S1: {support_levels[0]:.2f} (距离 -{distance_to_support:.2f}%)
- S2: {support_levels[1]:.2f}
- S3: {support_levels[2]:.2f}

布林带分析：
- 上轨: {bb_upper:.2f}
- 中轨: {bb_middle:.2f}
- 下轨: {bb_lower:.2f}

价格位置: {'接近上轨，可能面临阻力' if price > bb_middle else '接近下轨，可能获得支撑'}
"""
        
        confidence = 0.65
        
        return CoTStep(
            step_number=2,
            title="支撑与阻力分析",
            reasoning=reasoning,
            evidence={
                'resistance_levels': resistance_levels,
                'support_levels': support_levels,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'bb_middle': bb_middle
            },
            intermediate_conclusion=f"上方阻力在 {nearest_resistance:.2f}，下方支撑在 {nearest_support:.2f}",
            confidence=confidence
        )
    
    async def _analyze_momentum(self, symbol: str, features: Dict[str, Any]) -> CoTStep:
        """分析动量"""
        rsi = features.get('rsi_14', 50)
        macd = features.get('macd', 0)
        macd_signal = features.get('macd_signal', 0)
        macd_histogram = features.get('macd_histogram', 0)
        volume_ratio = features.get('volume_ratio', 1.0)
        
        # RSI 解读
        if rsi > 70:
            rsi_state = "超买区域"
        elif rsi < 30:
            rsi_state = "超卖区域"
        else:
            rsi_state = "中性区域"
        
        # MACD 解读
        macd_cross = "金叉" if macd > macd_signal and macd_histogram > 0 else "死叉" if macd < macd_signal else "交叉"
        
        reasoning = f"""
动量指标分析：

RSI (14): {rsi:.2f}
状态: {rsi_state}
解读: {'价格可能回调' if rsi > 70 else '可能存在反弹机会' if rsi < 30 else '动量中性'}

MACD 分析：
- MACD 线: {macd:.4f}
- 信号线: {macd_signal:.4f}
- 柱状图: {macd_histogram:.4f}
- 信号: {macd_cross}

成交量分析：
- 成交量比率: {volume_ratio:.2f}
- 状态: {'放量' if volume_ratio > 1.2 else '缩量' if volume_ratio < 0.8 else '正常'}

综合动量评估：
{'动量强劲，趋势可能延续' if macd_histogram > 0 and volume_ratio > 1 else '动量减弱，需警惕反转' if macd_histogram < 0 else '动量中性，观望为主'}
"""
        
        confidence = 0.7 if abs(rsi - 50) > 20 else 0.55
        
        return CoTStep(
            step_number=3,
            title="动量分析",
            reasoning=reasoning,
            evidence={
                'rsi': rsi,
                'macd': macd,
                'macd_signal': macd_signal,
                'macd_histogram': macd_histogram,
                'volume_ratio': volume_ratio
            },
            intermediate_conclusion=f"RSI处于{rsi_state}，MACD显示{macd_cross}信号",
            confidence=confidence
        )
    
    async def _analyze_onchain(self, symbol: str, features: Dict[str, Any]) -> CoTStep:
        """分析链上数据"""
        # 提取链上特征
        exchange_inflow = features.get('exchange_inflow', 0)
        exchange_outflow = features.get('exchange_outflow', 0)
        net_flow = exchange_outflow - exchange_inflow
        
        whale_tx_count = features.get('whale_tx_count', 0)
        whale_volume = features.get('whale_volume', 0)
        
        active_addresses = features.get('active_addresses_change', 0)
        
        reasoning = f"""
链上数据分析：

交易所资金流向：
- 流入: {exchange_inflow:,.0f} USD
- 流出: {exchange_outflow:,.0f} USD
- 净流入: {net_flow:,.0f} USD
- 解读: {'资金流出，看涨信号' if net_flow > 0 else '资金流入，看跌信号'}

鲸鱼活动：
- 大额交易数: {whale_tx_count}
- 鲸鱼交易量: {whale_volume:,.0f} USD
- 活动水平: {'活跃' if whale_tx_count > 10 else '正常' if whale_tx_count > 5 else '低迷'}

网络活跃度：
- 活跃地址变化: {active_addresses:.2f}%
- 趋势: {'增长' if active_addresses > 0 else '下降'}

链上信号总结：
{'链上数据显示积累迹象' if net_flow > 0 and whale_volume > 0 else '链上数据显示分发迹象' if net_flow < 0 else '链上信号中性'}
"""
        
        confidence = 0.6
        
        return CoTStep(
            step_number=4,
            title="链上数据分析",
            reasoning=reasoning,
            evidence={
                'exchange_inflow': exchange_inflow,
                'exchange_outflow': exchange_outflow,
                'net_flow': net_flow,
                'whale_tx_count': whale_tx_count,
                'whale_volume': whale_volume,
                'active_addresses_change': active_addresses
            },
            intermediate_conclusion=f"链上{'资金流出' if net_flow > 0 else '资金流入'}，鲸鱼活动{'活跃' if whale_tx_count > 10 else '正常'}",
            confidence=confidence
        )
    
    async def _analyze_sentiment(self, symbol: str, features: Dict[str, Any]) -> CoTStep:
        """分析市场情绪"""
        fear_greed = features.get('fear_greed_index', 50)
        social_sentiment = features.get('social_sentiment', 0)
        news_sentiment = features.get('news_sentiment', 0)
        funding_rate = features.get('funding_rate', 0)
        
        # 恐惧贪婪指数解读
        if fear_greed >= 75:
            fg_state = "极度贪婪"
        elif fear_greed >= 55:
            fg_state = "贪婪"
        elif fear_greed >= 45:
            fg_state = "中性"
        elif fear_greed >= 25:
            fg_state = "恐惧"
        else:
            fg_state = "极度恐惧"
        
        reasoning = f"""
市场情绪分析：

恐惧贪婪指数: {fear_greed}/100 ({fg_state})
解读: {'市场过热，警惕回调' if fear_greed > 75 else '市场情绪悲观，可能存在机会' if fear_greed < 25 else '情绪中性'}

社交媒体情绪: {social_sentiment:.2f}
新闻情绪: {news_sentiment:.2f}
综合情绪: {'积极' if social_sentiment > 0.2 and news_sentiment > 0.2 else '消极' if social_sentiment < -0.2 and news_sentiment < -0.2 else '中性'}

期货资金费率: {funding_rate:.4f}%
解读: {'多头支付空头，看涨情绪浓厚' if funding_rate > 0.01 else '空头支付多头，看跌情绪浓厚' if funding_rate < -0.01 else '多空平衡'}

情绪信号总结：
{'情绪极端，注意反转风险' if fear_greed > 80 or fear_greed < 20 else '情绪相对理性，可按技术面操作'}
"""
        
        confidence = 0.65
        
        return CoTStep(
            step_number=5,
            title="市场情绪分析",
            reasoning=reasoning,
            evidence={
                'fear_greed_index': fear_greed,
                'fear_greed_state': fg_state,
                'social_sentiment': social_sentiment,
                'news_sentiment': news_sentiment,
                'funding_rate': funding_rate
            },
            intermediate_conclusion=f"市场情绪{fg_state}，{'注意反转' if fear_greed > 80 or fear_greed < 20 else '情绪正常'}",
            confidence=confidence
        )
    
    async def _analyze_risk(
        self,
        symbol: str,
        features: Dict[str, Any],
        previous_steps: List[CoTStep]
    ) -> CoTStep:
        """风险评估"""
        atr = features.get('atr_14', 0)
        price = features.get('close', 1)
        volatility = (atr / price) * 100 if price > 0 else 0
        
        bb_width = features.get('bb_width', 0)
        volume_profile = features.get('volume_profile', 0)
        
        # 综合风险评分
        risk_factors = []
        if volatility > 5:
            risk_factors.append("高波动性")
        if bb_width > 0.1:
            risk_factors.append("布林带扩张")
        if volume_profile < 0.3:
            risk_factors.append("低流动性")
            
        risk_score = min(100, volatility * 5 + len(risk_factors) * 10)
        
        if risk_score >= 70:
            risk_level = "高风险"
        elif risk_score >= 40:
            risk_level = "中等风险"
        else:
            risk_level = "低风险"
        
        reasoning = f"""
风险评估：

波动性分析：
- ATR (14): {atr:.4f}
- 波动率: {volatility:.2f}%
- 评估: {'高波动' if volatility > 5 else '正常波动' if volatility > 2 else '低波动'}

布林带宽度: {bb_width:.4f}
解读: {'波动加剧' if bb_width > 0.1 else '波动收窄'}

流动性风险：
- 成交量分布: {volume_profile:.2f}
- 评估: {'流动性充足' if volume_profile > 0.5 else '流动性一般' if volume_profile > 0.3 else '流动性不足'}

识别的风险因素：
{chr(10).join(['- ' + f for f in risk_factors]) if risk_factors else '- 无明显风险因素'}

综合风险评分: {risk_score:.0f}/100
风险等级: {risk_level}

建议：
{'建议降低仓位，严格止损' if risk_score >= 70 else '可适当参与，控制仓位' if risk_score >= 40 else '风险可控，可按计划操作'}
"""
        
        confidence = 0.7
        
        return CoTStep(
            step_number=6,
            title="风险评估",
            reasoning=reasoning,
            evidence={
                'atr': atr,
                'volatility': volatility,
                'bb_width': bb_width,
                'volume_profile': volume_profile,
                'risk_score': risk_score,
                'risk_level': risk_level,
                'risk_factors': risk_factors
            },
            intermediate_conclusion=f"风险等级：{risk_level}，评分 {risk_score:.0f}/100",
            confidence=confidence
        )
    
    async def _synthesize_analysis(
        self,
        symbol: str,
        features: Dict[str, Any],
        previous_steps: List[CoTStep]
    ) -> CoTStep:
        """综合分析并生成信号"""
        # 汇总前面步骤的信息
        trend_step = previous_steps[0]
        momentum_step = previous_steps[2]
        sentiment_step = previous_steps[4]
        risk_step = previous_steps[5]
        
        # 提取关键信息
        trend_signal = 1 if "上升" in trend_step.intermediate_conclusion else -1
        momentum_signal = 1 if "金叉" in momentum_step.intermediate_conclusion else -1
        sentiment_score = sentiment_step.evidence.get('fear_greed_index', 50)
        risk_score = risk_step.evidence.get('risk_score', 50)
        
        # 计算综合得分
        # 趋势权重 30%，动量权重 25%，情绪权重 20%，风险权重 25%
        composite_score = (
            trend_signal * 0.30 +
            momentum_signal * 0.25 +
            (sentiment_score - 50) / 50 * 0.20 -
            risk_score / 100 * 0.25
        )
        
        # 确定信号类型
        if composite_score > 0.3:
            signal = SignalType.STRONG_BUY
        elif composite_score > 0.1:
            signal = SignalType.BUY
        elif composite_score < -0.3:
            signal = SignalType.STRONG_SELL
        elif composite_score < -0.1:
            signal = SignalType.SELL
        else:
            signal = SignalType.HOLD
            
        # 计算置信度
        avg_confidence = sum(step.confidence for step in previous_steps) / len(previous_steps)
        confidence = avg_confidence * (1 - abs(composite_score) * 0.3)
        
        # 建议仓位
        position_size = min(1.0, abs(composite_score) * 2 * (1 - risk_score / 100))
        
        reasoning = f"""
综合分析与信号生成：

各维度评估汇总：
1. 趋势分析: {trend_step.intermediate_conclusion} (置信度: {trend_step.confidence:.0%})
2. 动量分析: {momentum_step.intermediate_conclusion} (置信度: {momentum_step.confidence:.0%})
3. 情绪分析: {sentiment_step.intermediate_conclusion} (置信度: {sentiment_step.confidence:.0%})
4. 风险评估: {risk_step.intermediate_conclusion} (置信度: {risk_step.confidence:.0%})

综合评分计算：
- 趋势贡献: {trend_signal * 0.30:.2f}
- 动量贡献: {momentum_signal * 0.25:.2f}
- 情绪贡献: {(sentiment_score - 50) / 50 * 0.20:.2f}
- 风险调整: {-risk_score / 100 * 0.25:.2f}
- 综合得分: {composite_score:.3f}

最终交易信号: {signal.value.upper()}
置信度: {confidence:.1%}
建议仓位: {position_size:.0%}

关键理由：
{'技术面和情绪面均支持做多' if composite_score > 0.2 else '技术面和情绪面均支持做空' if composite_score < -0.2 else '信号不明确，建议观望'}
{'风险较高，建议降低仓位' if risk_score > 60 else '风险可控'}
"""
        
        return CoTStep(
            step_number=7,
            title="综合判断与信号生成",
            reasoning=reasoning,
            evidence={
                'composite_score': composite_score,
                'signal': signal.value,
                'confidence': confidence,
                'position_size': position_size,
                'trend_contribution': trend_signal * 0.30,
                'momentum_contribution': momentum_signal * 0.25,
                'sentiment_contribution': (sentiment_score - 50) / 50 * 0.20,
                'risk_contribution': -risk_score / 100 * 0.25
            },
            intermediate_conclusion=f"信号: {signal.value.upper()}，置信度 {confidence:.1%}，建议仓位 {position_size:.0%}",
            confidence=confidence
        )
    
    async def stream_reasoning(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        features: Dict[str, Any]
    ) -> AsyncGenerator[CoTStep, None]:
        """
        流式推理，逐步返回分析结果
        
        Args:
            symbol: 交易对符号
            market_data: 原始市场数据
            features: 特征工程层输出的特征
            
        Yields:
            CoTStep: 每个推理步骤
        """
        steps = [
            ("趋势识别", self._analyze_trend),
            ("支撑阻力分析", self._analyze_support_resistance),
            ("动量分析", self._analyze_momentum),
            ("链上数据分析", self._analyze_onchain),
            ("市场情绪分析", self._analyze_sentiment),
            ("风险评估", self._analyze_risk),
            ("综合判断", self._synthesize_analysis)
        ]
        
        previous_steps = []
        for i, (name, func) in enumerate(steps):
            try:
                if i < 5:
                    step = await func(symbol, features)
                else:
                    step = await func(symbol, features, previous_steps)
                previous_steps.append(step)
                yield step
            except Exception as e:
                logger.error(f"Error in step {i+1} ({name}): {e}")
                raise
