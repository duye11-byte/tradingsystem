"""
推理引擎主入口
整合 Chain-of-Thought、多模型集成和自我一致性验证
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime, timedelta
import logging

from .reasoning_types import (
    ReasoningResult,
    TradingSignal,
    MarketAnalysis,
    SignalType,
    CoTStep,
    ConsensusResult,
    ValidationResult
)
from .cot_engine import ChainOfThoughtEngine
from .ensemble_manager import EnsembleManager
from .consistency_validator import ConsistencyValidator

logger = logging.getLogger(__name__)


class ReasoningEngine:
    """
    推理引擎主类
    
    整合三个核心组件：
    1. ChainOfThoughtEngine: 执行逐步推理分析
    2. EnsembleManager: 管理多模型集成和共识
    3. ConsistencyValidator: 验证推理结果的一致性
    
    提供统一的推理接口，生成最终的交易信号。
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化推理引擎
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        
        # 初始化各个组件
        self.cot_engine = ChainOfThoughtEngine(
            llm_client=self.config.get('llm_client'),
            config=self.config.get('cot_config', {})
        )
        
        self.ensemble_manager = EnsembleManager(
            config=self.config.get('ensemble_config', {})
        )
        
        self.consistency_validator = ConsistencyValidator(
            config=self.config.get('consistency_config', {})
        )
        
        # 性能监控
        self.performance_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_latency_ms': 0,
            'consistency_pass_rate': 0
        }
        
        logger.info("ReasoningEngine initialized successfully")
    
    async def reason(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        features: Dict[str, Any],
        skip_validation: bool = False
    ) -> ReasoningResult:
        """
        执行完整推理流程
        
        Args:
            symbol: 交易对符号
            market_data: 原始市场数据
            features: 特征工程层输出的特征
            skip_validation: 是否跳过一致性验证（用于快速推理）
            
        Returns:
            ReasoningResult: 推理结果
        """
        start_time = time.time()
        self.performance_stats['total_requests'] += 1
        
        try:
            # 1. 执行 Chain-of-Thought 推理
            cot_start = time.time()
            reasoning_chain = await self.cot_engine.analyze_market(
                symbol, market_data, features
            )
            cot_latency = (time.time() - cot_start) * 1000
            
            # 2. 执行多模型集成预测
            ensemble_start = time.time()
            consensus_result = await self.ensemble_manager.predict(
                symbol, features, market_data
            )
            ensemble_latency = (time.time() - ensemble_start) * 1000
            
            # 3. 构建市场分析对象
            market_analysis = self._build_market_analysis(
                symbol, features, reasoning_chain
            )
            
            # 4. 生成交易信号
            trading_signal = self._build_trading_signal(
                symbol,
                reasoning_chain,
                consensus_result,
                market_analysis,
                features
            )
            
            # 5. 执行一致性验证（除非跳过）
            consistency_latency = 0
            if not skip_validation:
                consistency_start = time.time()
                passed, consistency_score, validation_result = await self._validate_signal(
                    trading_signal, symbol, features, market_data
                )
                consistency_latency = (time.time() - consistency_start) * 1000
                
                # 更新信号验证信息
                trading_signal.consistency_check_passed = passed
                trading_signal.consistency_score = consistency_score
                
                if not passed:
                    logger.warning(
                        f"Signal for {symbol} failed consistency check: "
                        f"score={consistency_score:.2%}"
                    )
            else:
                trading_signal.consistency_check_passed = True
                trading_signal.consistency_score = 0.5
                validation_result = None
            
            # 6. 计算总延迟
            total_latency = (time.time() - start_time) * 1000
            
            # 7. 更新性能统计
            self.performance_stats['successful_requests'] += 1
            self._update_avg_latency(total_latency)
            
            # 8. 构建结果
            result = ReasoningResult(
                success=True,
                signal=trading_signal,
                error_message=None,
                total_latency_ms=total_latency,
                cot_latency_ms=cot_latency,
                ensemble_latency_ms=ensemble_latency,
                consistency_latency_ms=consistency_latency,
                raw_reasoning={
                    'cot_steps': len(reasoning_chain),
                    'ensemble_models': len(consensus_result.participating_models),
                    'validation_passed': trading_signal.consistency_check_passed
                },
                debug_info={
                    'validation_result': validation_result.detailed_analysis if validation_result else None,
                    'consensus_details': {
                        'agreement_ratio': consensus_result.agreement_ratio,
                        'dissensus_analysis': consensus_result.dissensus_analysis
                    }
                }
            )
            
            logger.info(
                f"Reasoning completed for {symbol}: "
                f"signal={trading_signal.signal.value}, "
                f"confidence={trading_signal.confidence:.2%}, "
                f"latency={total_latency:.0f}ms"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in reasoning for {symbol}: {e}")
            self.performance_stats['failed_requests'] += 1
            
            return ReasoningResult(
                success=False,
                signal=None,
                error_message=str(e),
                total_latency_ms=(time.time() - start_time) * 1000,
                cot_latency_ms=0,
                ensemble_latency_ms=0,
                consistency_latency_ms=0,
                raw_reasoning={},
                debug_info={'error': str(e)}
            )
    
    async def reason_stream(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        features: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式推理，逐步返回结果
        
        Args:
            symbol: 交易对符号
            market_data: 市场数据
            features: 特征数据
            
        Yields:
            Dict[str, Any]: 推理进度和中间结果
        """
        yield {
            'stage': 'started',
            'message': f'开始推理分析: {symbol}',
            'progress': 0
        }
        
        try:
            # 1. 流式 CoT 推理
            yield {'stage': 'cot', 'message': '执行思维链推理...', 'progress': 10}
            
            cot_steps = []
            async for step in self.cot_engine.stream_reasoning(symbol, market_data, features):
                cot_steps.append(step)
                yield {
                    'stage': 'cot',
                    'message': f'完成步骤 {step.step_number}: {step.title}',
                    'progress': 10 + step.step_number * 10,
                    'step': step
                }
            
            # 2. 集成预测
            yield {'stage': 'ensemble', 'message': '执行多模型集成预测...', 'progress': 80}
            consensus_result = await self.ensemble_manager.predict(symbol, features, market_data)
            
            yield {
                'stage': 'ensemble',
                'message': f'集成预测完成: {consensus_result.final_signal.value}',
                'progress': 90,
                'consensus': consensus_result
            }
            
            # 3. 生成最终信号
            yield {'stage': 'finalizing', 'message': '生成最终交易信号...', 'progress': 95}
            
            market_analysis = self._build_market_analysis(symbol, features, cot_steps)
            trading_signal = self._build_trading_signal(
                symbol, cot_steps, consensus_result, market_analysis, features
            )
            
            # 4. 完成
            yield {
                'stage': 'completed',
                'message': '推理完成',
                'progress': 100,
                'signal': trading_signal
            }
            
        except Exception as e:
            yield {
                'stage': 'error',
                'message': f'推理出错: {str(e)}',
                'progress': 0,
                'error': str(e)
            }
    
    async def batch_reason(
        self,
        tasks: List[Dict[str, Any]],
        max_concurrency: int = 5
    ) -> List[ReasoningResult]:
        """
        批量推理
        
        Args:
            tasks: 任务列表，每个任务包含 symbol, market_data, features
            max_concurrency: 最大并发数
            
        Returns:
            List[ReasoningResult]: 推理结果列表
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def _reason_with_semaphore(task):
            async with semaphore:
                return await self.reason(
                    task['symbol'],
                    task['market_data'],
                    task['features'],
                    task.get('skip_validation', False)
                )
        
        results = await asyncio.gather(
            *[_reason_with_semaphore(task) for task in tasks],
            return_exceptions=True
        )
        
        # 处理异常
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ReasoningResult(
                    success=False,
                    signal=None,
                    error_message=str(result),
                    total_latency_ms=0,
                    cot_latency_ms=0,
                    ensemble_latency_ms=0,
                    consistency_latency_ms=0,
                    raw_reasoning={},
                    debug_info={'task_index': i, 'error': str(result)}
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def _build_market_analysis(
        self,
        symbol: str,
        features: Dict[str, Any],
        reasoning_chain: List[CoTStep]
    ) -> MarketAnalysis:
        """构建市场分析对象"""
        # 从推理链中提取信息
        trend_step = reasoning_chain[0] if len(reasoning_chain) > 0 else None
        support_resistance_step = reasoning_chain[1] if len(reasoning_chain) > 1 else None
        momentum_step = reasoning_chain[2] if len(reasoning_chain) > 2 else None
        onchain_step = reasoning_chain[3] if len(reasoning_chain) > 3 else None
        sentiment_step = reasoning_chain[4] if len(reasoning_chain) > 4 else None
        risk_step = reasoning_chain[5] if len(reasoning_chain) > 5 else None
        
        return MarketAnalysis(
            symbol=symbol,
            timeframe='1h',  # 默认时间框架
            analysis_timestamp=datetime.now(),
            
            # 技术分析
            technical_summary=trend_step.intermediate_conclusion if trend_step else "",
            trend_direction="上升" if features.get('price_change_7d', 0) > 0 else "下降",
            support_levels=support_resistance_step.evidence.get('support_levels', []) if support_resistance_step else [],
            resistance_levels=support_resistance_step.evidence.get('resistance_levels', []) if support_resistance_step else [],
            key_indicators={
                'rsi': features.get('rsi_14', 50),
                'macd': features.get('macd', 0),
                'sma20': features.get('sma_20', 0),
                'sma50': features.get('sma_50', 0)
            },
            
            # 链上分析
            onchain_summary=onchain_step.intermediate_conclusion if onchain_step else "",
            exchange_flows={
                'inflow': features.get('exchange_inflow', 0),
                'outflow': features.get('exchange_outflow', 0)
            },
            whale_activity=onchain_step.evidence.get('whale_tx_count', 0) if onchain_step else 0,
            network_health="正常",  # 简化处理
            
            # 情绪分析
            sentiment_summary=sentiment_step.intermediate_conclusion if sentiment_step else "",
            fear_greed_index=features.get('fear_greed_index', 50),
            social_sentiment=features.get('social_sentiment', 0),
            news_sentiment=features.get('news_sentiment', 0),
            
            # 综合评估
            overall_assessment=reasoning_chain[-1].intermediate_conclusion if reasoning_chain else "",
            risk_level=risk_step.evidence.get('risk_level', '未知') if risk_step else "未知",
            opportunity_score=0.5  # 简化处理
        )
    
    def _build_trading_signal(
        self,
        symbol: str,
        reasoning_chain: List[CoTStep],
        consensus_result: ConsensusResult,
        market_analysis: MarketAnalysis,
        features: Dict[str, Any]
    ) -> TradingSignal:
        """构建交易信号"""
        price = features.get('close', 0)
        
        # 从共识结果获取信号
        signal = consensus_result.final_signal
        confidence = consensus_result.consensus_confidence
        
        # 计算止损止盈（基于ATR）
        atr = features.get('atr_14', price * 0.02)
        
        if signal in [SignalType.BUY, SignalType.STRONG_BUY]:
            stop_loss = price - atr * 2
            take_profit = price + atr * 3
        elif signal in [SignalType.SELL, SignalType.STRONG_SELL]:
            stop_loss = price + atr * 2
            take_profit = price - atr * 3
        else:
            stop_loss = None
            take_profit = None
        
        # 仓位大小基于置信度和风险
        risk_adjustment = 0.7 if market_analysis.risk_level == '高风险' else 1.0
        position_size = min(1.0, confidence * risk_adjustment)
        
        return TradingSignal(
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_ratio=position_size,
            reasoning_chain=reasoning_chain,
            market_analysis=market_analysis,
            consensus_result=consensus_result,
            consistency_check_passed=False,  # 将在验证后更新
            consistency_score=0.0,  # 将在验证后更新
            valid_until=datetime.now() + timedelta(hours=1),
            metadata={
                'generated_by': 'ReasoningEngine',
                'version': '1.0'
            }
        )
    
    async def _validate_signal(
        self,
        signal: TradingSignal,
        symbol: str,
        features: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> tuple:
        """验证信号一致性"""
        return await self.consistency_validator.validate_signal(
            signal,
            self.cot_engine,
            self.ensemble_manager,
            symbol,
            features,
            market_data
        )
    
    def _update_avg_latency(self, new_latency: float):
        """更新平均延迟"""
        n = self.performance_stats['successful_requests']
        current_avg = self.performance_stats['avg_latency_ms']
        self.performance_stats['avg_latency_ms'] = (
            (current_avg * (n - 1) + new_latency) / n
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        total = self.performance_stats['total_requests']
        success = self.performance_stats['successful_requests']
        
        return {
            **self.performance_stats,
            'success_rate': success / total if total > 0 else 0,
            'consistency_stats': self.consistency_validator.get_validation_statistics(),
            'consistency_trend': self.consistency_validator.get_consistency_trend()
        }
    
    def get_agent_weights(self) -> Dict[str, float]:
        """获取当前代理权重"""
        return {agent.name: agent.weight for agent in self.ensemble_manager.agents}
    
    def update_agent_weight(self, agent_name: str, new_weight: float):
        """更新代理权重"""
        self.ensemble_manager.update_agent_weight(agent_name, new_weight)
    
    def add_custom_agent(self, agent_config: Dict[str, Any]):
        """添加自定义代理"""
        from .reasoning_types import ExpertAgent
        
        agent = ExpertAgent(
            name=agent_config['name'],
            specialization=agent_config['specialization'],
            model_config=agent_config.get('model_config', {}),
            prompt_template=agent_config.get('prompt_template', ''),
            weight=agent_config.get('weight', 0.25),
            confidence_threshold=agent_config.get('confidence_threshold', 0.6)
        )
        
        self.ensemble_manager.add_agent(agent)
