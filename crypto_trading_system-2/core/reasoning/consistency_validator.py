"""
自我一致性验证器
实现 Self-Consistency 验证机制，通过多次采样验证推理结果的一致性
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
import logging
from collections import Counter

from .reasoning_types import (
    ValidationResult,
    CoTStep,
    SignalType,
    TradingSignal,
    ConsensusResult
)

logger = logging.getLogger(__name__)


class ConsistencyValidator:
    """
    自我一致性验证器
    
    通过多次采样和验证，确保推理结果的可靠性和一致性。
    实现以下验证机制：
    1. 多次推理采样
    2. 结果一致性检查
    3. 置信度评估
    4. 异常检测
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化验证器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.num_samples = self.config.get('num_samples', 5)
        self.consistency_threshold = self.config.get('consistency_threshold', 0.7)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        self.variance_threshold = self.config.get('variance_threshold', 0.3)
        
        # 验证历史，用于跟踪一致性趋势
        self.validation_history: List[ValidationResult] = []
        self.max_history_size = 100
        
    async def validate_signal(
        self,
        signal: TradingSignal,
        cot_engine,
        ensemble_manager,
        symbol: str,
        features: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Tuple[bool, float, ValidationResult]:
        """
        验证交易信号的一致性
        
        Args:
            signal: 待验证的交易信号
            cot_engine: CoT引擎实例
            ensemble_manager: 集成管理器实例
            symbol: 交易对符号
            features: 特征数据
            market_data: 市场数据
            
        Returns:
            Tuple[bool, float, ValidationResult]: (是否通过, 一致性分数, 详细结果)
        """
        logger.info(f"Starting consistency validation for {symbol}")
        
        # 1. 多次采样 CoT 推理
        cot_samples = await self._sample_cot_reasoning(
            cot_engine, symbol, market_data, features
        )
        
        # 2. 多次采样集成预测
        ensemble_samples = await self._sample_ensemble_predictions(
            ensemble_manager, symbol, features, market_data
        )
        
        # 3. 验证 CoT 一致性
        cot_validation = self._validate_cot_consistency(cot_samples)
        
        # 4. 验证集成预测一致性
        ensemble_validation = self._validate_ensemble_consistency(ensemble_samples)
        
        # 5. 验证最终信号一致性
        signal_validation = self._validate_signal_consistency(
            signal, cot_samples, ensemble_samples
        )
        
        # 6. 综合验证结果
        final_validation = self._combine_validations(
            cot_validation, ensemble_validation, signal_validation
        )
        
        # 7. 存储验证历史
        self._add_to_history(final_validation)
        
        # 8. 判断是否通过验证
        passed = self._check_passed(final_validation)
        
        logger.info(
            f"Validation completed for {symbol}: "
            f"passed={passed}, score={final_validation.consistency_score:.2%}"
        )
        
        return passed, final_validation.consistency_score, final_validation
    
    async def _sample_cot_reasoning(
        self,
        cot_engine,
        symbol: str,
        market_data: Dict[str, Any],
        features: Dict[str, Any]
    ) -> List[List[CoTStep]]:
        """多次采样 CoT 推理"""
        samples = []
        
        for i in range(self.num_samples):
            try:
                # 添加微小扰动到特征，模拟不同的推理路径
                perturbed_features = self._perturb_features(features, seed=i)
                
                # 执行推理
                reasoning_chain = await cot_engine.analyze_market(
                    symbol, market_data, perturbed_features
                )
                samples.append(reasoning_chain)
                
            except Exception as e:
                logger.error(f"Error in CoT sample {i}: {e}")
        
        return samples
    
    async def _sample_ensemble_predictions(
        self,
        ensemble_manager,
        symbol: str,
        features: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> List[ConsensusResult]:
        """多次采样集成预测"""
        samples = []
        
        for i in range(self.num_samples):
            try:
                # 添加微小扰动
                perturbed_features = self._perturb_features(features, seed=i)
                
                # 执行预测
                consensus = await ensemble_manager.predict(
                    symbol, perturbed_features, market_data
                )
                samples.append(consensus)
                
            except Exception as e:
                logger.error(f"Error in ensemble sample {i}: {e}")
        
        return samples
    
    def _perturb_features(
        self,
        features: Dict[str, Any],
        seed: int,
        perturbation_scale: float = 0.02
    ) -> Dict[str, Any]:
        """
        对特征添加微小扰动，用于生成不同的推理路径
        
        Args:
            features: 原始特征
            seed: 随机种子
            perturbation_scale: 扰动幅度
            
        Returns:
            Dict[str, Any]: 扰动后的特征
        """
        np.random.seed(seed)
        
        perturbed = {}
        for key, value in features.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                # 对数值特征添加高斯噪声
                noise = np.random.normal(0, abs(value) * perturbation_scale)
                perturbed[key] = value + noise
            else:
                perturbed[key] = value
        
        return perturbed
    
    def _validate_cot_consistency(self, samples: List[List[CoTStep]]) -> ValidationResult:
        """验证 CoT 推理的一致性"""
        if not samples or len(samples) < 2:
            return ValidationResult(
                is_consistent=False,
                consistency_score=0.0,
                sample_count=len(samples),
                agreement_count=0,
                variance=1.0,
                confidence_interval=(0, 0),
                detailed_analysis="样本不足，无法验证"
            )
        
        # 提取最终信号
        final_signals = []
        for sample in samples:
            if sample and len(sample) > 0:
                # 获取最后一步的结论
                last_step = sample[-1]
                signal = self._extract_signal_from_text(last_step.intermediate_conclusion)
                final_signals.append(signal)
        
        # 计算信号一致性
        signal_consistency = self._calculate_signal_consistency(final_signals)
        
        # 计算置信度一致性
        confidence_values = [sample[-1].confidence for sample in samples if sample]
        confidence_variance = np.var(confidence_values) if confidence_values else 1.0
        confidence_consistency = max(0, 1 - confidence_variance)
        
        # 综合一致性分数
        overall_score = signal_consistency * 0.7 + confidence_consistency * 0.3
        
        # 计算置信区间
        signal_values = [self._signal_to_value(s) for s in final_signals]
        mean_signal = np.mean(signal_values) if signal_values else 0
        std_signal = np.std(signal_values) if signal_values else 0
        confidence_interval = (mean_signal - 1.96 * std_signal, mean_signal + 1.96 * std_signal)
        
        # 统计最频繁的信号
        signal_counts = Counter(final_signals)
        most_common_signal, agreement_count = signal_counts.most_common(1)[0]
        
        is_consistent = overall_score >= self.consistency_threshold
        
        analysis = f"""
CoT推理一致性分析：
- 样本数: {len(samples)}
- 信号一致性: {signal_consistency:.1%}
- 置信度一致性: {confidence_consistency:.1%}
- 最频繁信号: {most_common_signal.value if hasattr(most_common_signal, 'value') else most_common_signal}
- 一致样本数: {agreement_count}/{len(samples)}
- 综合评分: {overall_score:.1%}
"""
        
        return ValidationResult(
            is_consistent=is_consistent,
            consistency_score=overall_score,
            sample_count=len(samples),
            agreement_count=agreement_count,
            variance=confidence_variance,
            confidence_interval=confidence_interval,
            detailed_analysis=analysis
        )
    
    def _validate_ensemble_consistency(
        self,
        samples: List[ConsensusResult]
    ) -> ValidationResult:
        """验证集成预测的一致性"""
        if not samples or len(samples) < 2:
            return ValidationResult(
                is_consistent=False,
                consistency_score=0.0,
                sample_count=len(samples),
                agreement_count=0,
                variance=1.0,
                confidence_interval=(0, 0),
                detailed_analysis="样本不足，无法验证"
            )
        
        # 提取信号和置信度
        signals = [s.final_signal for s in samples]
        confidences = [s.consensus_confidence for s in samples]
        weighted_scores = [s.weighted_score for s in samples]
        
        # 计算信号一致性
        signal_consistency = self._calculate_signal_consistency(signals)
        
        # 计算加权分数的一致性
        score_variance = np.var(weighted_scores)
        score_consistency = max(0, 1 - score_variance / 4)
        
        # 计算置信度一致性
        confidence_variance = np.var(confidences)
        confidence_consistency = max(0, 1 - confidence_variance)
        
        # 综合一致性
        overall_score = signal_consistency * 0.5 + score_consistency * 0.3 + confidence_consistency * 0.2
        
        # 置信区间
        mean_score = np.mean(weighted_scores)
        std_score = np.std(weighted_scores)
        confidence_interval = (mean_score - 1.96 * std_score, mean_score + 1.96 * std_score)
        
        # 统计
        signal_counts = Counter(signals)
        most_common_signal, agreement_count = signal_counts.most_common(1)[0]
        
        is_consistent = overall_score >= self.consistency_threshold
        
        analysis = f"""
集成预测一致性分析：
- 样本数: {len(samples)}
- 信号一致性: {signal_consistency:.1%}
- 分数一致性: {score_consistency:.1%}
- 置信度一致性: {confidence_consistency:.1%}
- 最频繁信号: {most_common_signal.value}
- 一致样本数: {agreement_count}/{len(samples)}
- 综合评分: {overall_score:.1%}
"""
        
        return ValidationResult(
            is_consistent=is_consistent,
            consistency_score=overall_score,
            sample_count=len(samples),
            agreement_count=agreement_count,
            variance=score_variance,
            confidence_interval=confidence_interval,
            detailed_analysis=analysis
        )
    
    def _validate_signal_consistency(
        self,
        signal: TradingSignal,
        cot_samples: List[List[CoTStep]],
        ensemble_samples: List[ConsensusResult]
    ) -> ValidationResult:
        """验证最终信号与采样结果的一致性"""
        # 提取采样中的信号
        cot_signals = []
        for sample in cot_samples:
            if sample and len(sample) > 0:
                s = self._extract_signal_from_text(sample[-1].intermediate_conclusion)
                cot_signals.append(s)
        
        ensemble_signals = [s.final_signal for s in ensemble_samples]
        
        # 检查原始信号是否在采样中出现
        cot_match = sum(1 for s in cot_signals if s == signal.signal)
        ensemble_match = sum(1 for s in ensemble_signals if s == signal.signal)
        
        # 计算匹配率
        cot_match_rate = cot_match / len(cot_signals) if cot_signals else 0
        ensemble_match_rate = ensemble_match / len(ensemble_signals) if ensemble_signals else 0
        
        # 综合匹配率
        overall_match_rate = (cot_match_rate + ensemble_match_rate) / 2
        
        # 检查置信度范围
        cot_confidences = [sample[-1].confidence for sample in cot_samples if sample]
        ensemble_confidences = [s.consensus_confidence for s in ensemble_samples]
        
        all_confidences = cot_confidences + ensemble_confidences
        if all_confidences:
            mean_conf = np.mean(all_confidences)
            std_conf = np.std(all_confidences)
            
            # 检查原始信号置信度是否在合理范围内
            conf_in_range = mean_conf - 2 * std_conf <= signal.confidence <= mean_conf + 2 * std_conf
            conf_score = 1.0 if conf_in_range else 0.5
        else:
            conf_score = 0.5
        
        overall_score = overall_match_rate * 0.7 + conf_score * 0.3
        
        is_consistent = overall_score >= self.consistency_threshold
        
        analysis = f"""
信号一致性验证：
- 原始信号: {signal.signal.value}
- CoT匹配率: {cot_match_rate:.1%} ({cot_match}/{len(cot_signals)})
- 集成匹配率: {ensemble_match_rate:.1%} ({ensemble_match}/{len(ensemble_signals)})
- 置信度合理性: {'是' if conf_score > 0.5 else '否'}
- 综合评分: {overall_score:.1%}
"""
        
        return ValidationResult(
            is_consistent=is_consistent,
            consistency_score=overall_score,
            sample_count=len(cot_signals) + len(ensemble_signals),
            agreement_count=cot_match + ensemble_match,
            variance=0,
            confidence_interval=(0, 0),
            detailed_analysis=analysis
        )
    
    def _combine_validations(
        self,
        cot_validation: ValidationResult,
        ensemble_validation: ValidationResult,
        signal_validation: ValidationResult
    ) -> ValidationResult:
        """综合多个验证结果"""
        # 加权平均
        weights = {
            'cot': 0.3,
            'ensemble': 0.4,
            'signal': 0.3
        }
        
        overall_score = (
            cot_validation.consistency_score * weights['cot'] +
            ensemble_validation.consistency_score * weights['ensemble'] +
            signal_validation.consistency_score * weights['signal']
        )
        
        total_samples = cot_validation.sample_count + ensemble_validation.sample_count + signal_validation.sample_count
        total_agreements = cot_validation.agreement_count + ensemble_validation.agreement_count + signal_validation.agreement_count
        
        # 综合方差
        overall_variance = np.mean([
            cot_validation.variance,
            ensemble_validation.variance,
            signal_validation.variance
        ])
        
        is_consistent = overall_score >= self.consistency_threshold
        
        combined_analysis = f"""
=== 综合一致性验证报告 ===

1. CoT推理一致性:
{cot_validation.detailed_analysis}

2. 集成预测一致性:
{ensemble_validation.detailed_analysis}

3. 信号验证:
{signal_validation.detailed_analysis}

=== 综合评估 ===
- 总体一致性评分: {overall_score:.1%}
- 阈值: {self.consistency_threshold:.1%}
- 验证结果: {'通过' if is_consistent else '未通过'}
- 建议: {'信号可靠，可以执行' if is_consistent else '信号不一致，建议观望或降低仓位'}
"""
        
        return ValidationResult(
            is_consistent=is_consistent,
            consistency_score=overall_score,
            sample_count=total_samples,
            agreement_count=total_agreements,
            variance=overall_variance,
            confidence_interval=(0, 0),
            detailed_analysis=combined_analysis
        )
    
    def _calculate_signal_consistency(self, signals: List[SignalType]) -> float:
        """计算信号一致性"""
        if not signals:
            return 0.0
        
        # 将信号转换为数值
        values = [self._signal_to_value(s) for s in signals]
        
        # 计算方差
        variance = np.var(values)
        
        # 转换为一致性分数（方差越小，一致性越高）
        max_variance = 4  # 信号值范围是 -2 到 2
        consistency = max(0, 1 - variance / max_variance)
        
        return consistency
    
    def _extract_signal_from_text(self, text: str) -> SignalType:
        """从文本中提取信号"""
        text_lower = text.lower()
        
        if 'strong_buy' in text_lower or '强烈买入' in text:
            return SignalType.STRONG_BUY
        elif 'buy' in text_lower or '买入' in text:
            return SignalType.BUY
        elif 'strong_sell' in text_lower or '强烈卖出' in text:
            return SignalType.STRONG_SELL
        elif 'sell' in text_lower or '卖出' in text:
            return SignalType.SELL
        else:
            return SignalType.HOLD
    
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
    
    def _check_passed(self, validation: ValidationResult) -> bool:
        """检查是否通过验证"""
        return (
            validation.consistency_score >= self.consistency_threshold and
            validation.consistency_score >= self.confidence_threshold
        )
    
    def _add_to_history(self, validation: ValidationResult):
        """添加验证结果到历史"""
        self.validation_history.append(validation)
        
        # 限制历史大小
        if len(self.validation_history) > self.max_history_size:
            self.validation_history = self.validation_history[-self.max_history_size:]
    
    def get_consistency_trend(self, window: int = 10) -> Dict[str, Any]:
        """
        获取一致性趋势
        
        Args:
            window: 时间窗口大小
            
        Returns:
            Dict[str, Any]: 趋势分析结果
        """
        if len(self.validation_history) < 2:
            return {
                'trend': 'insufficient_data',
                'avg_consistency': 0,
                'trend_direction': 'unknown'
            }
        
        recent = self.validation_history[-window:]
        scores = [v.consistency_score for v in recent]
        
        avg_score = np.mean(scores)
        
        # 判断趋势
        if len(scores) >= 3:
            first_half = np.mean(scores[:len(scores)//2])
            second_half = np.mean(scores[len(scores)//2:])
            
            if second_half > first_half * 1.05:
                trend = 'improving'
            elif second_half < first_half * 0.95:
                trend = 'degrading'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'trend': trend,
            'avg_consistency': avg_score,
            'trend_direction': 'up' if trend == 'improving' else 'down' if trend == 'degrading' else 'flat',
            'sample_count': len(recent),
            'pass_rate': sum(1 for v in recent if v.is_consistent) / len(recent)
        }
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """获取验证统计信息"""
        if not self.validation_history:
            return {
                'total_validations': 0,
                'passed': 0,
                'failed': 0,
                'pass_rate': 0,
                'avg_consistency': 0,
                'min_consistency': 0,
                'max_consistency': 0,
                'std_consistency': 0
            }
        
        total = len(self.validation_history)
        passed = sum(1 for v in self.validation_history if v.is_consistent)
        scores = [v.consistency_score for v in self.validation_history]
        
        return {
            'total_validations': total,
            'passed': passed,
            'failed': total - passed,
            'pass_rate': passed / total,
            'avg_consistency': np.mean(scores),
            'min_consistency': np.min(scores),
            'max_consistency': np.max(scores),
            'std_consistency': np.std(scores)
        }
