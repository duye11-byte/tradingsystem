"""
反馈引擎主入口
整合性能分析、在线学习、RLHF训练和反馈存储，提供统一的反馈接口
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

from .feedback_types import (
    TradeRecord, TradeResult, PerformanceMetrics, LearningSample,
    AgentPerformance, FeedbackConfig, HumanFeedback, FeedbackType,
    ModelUpdate, StrategyInsight, PerformanceAlert
)
from .performance_analyzer import PerformanceAnalyzer
from .online_learner import OnlineLearner
from .rlhf_trainer import RLHFTrainer
from .feedback_store import FeedbackStore

logger = logging.getLogger(__name__)


class FeedbackMode(Enum):
    """反馈模式"""
    AUTO = "auto"              # 全自动模式
    SEMI_AUTO = "semi_auto"    # 半自动模式（需要人工确认）
    MANUAL = "manual"          # 手动模式


@dataclass
class FeedbackContext:
    """反馈上下文"""
    symbol: str
    timestamp: datetime
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    decision_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackResult:
    """反馈结果"""
    success: bool
    message: str
    updates: List[ModelUpdate] = field(default_factory=list)
    insights: List[StrategyInsight] = field(default_factory=list)
    alerts: List[PerformanceAlert] = field(default_factory=list)
    metrics: Optional[PerformanceMetrics] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeedbackEngine:
    """
    反馈引擎主类
    
    整合所有反馈组件，提供统一的反馈接口，实现:
    - 交易记录和性能跟踪
    - 自动性能分析和报告
    - 在线学习和模型优化
    - RLHF训练和策略改进
    - 反馈存储和历史管理
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化反馈引擎
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.mode = FeedbackMode(self.config.get('mode', 'auto'))
        
        # 初始化组件
        self.performance_analyzer = PerformanceAnalyzer()
        self.online_learner = OnlineLearner()
        self.rlhf_trainer = RLHFTrainer()
        self.feedback_store = FeedbackStore()
        
        # 回调函数
        self._alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        self._update_callbacks: List[Callable[[ModelUpdate], None]] = []
        
        # 运行状态
        self._running = False
        self._analysis_task: Optional[asyncio.Task] = None
        self._learning_task: Optional[asyncio.Task] = None
        
        logger.info("反馈引擎初始化完成")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载配置文件"""
        default_config = {
            'mode': 'auto',
            'analysis_interval': 3600,  # 性能分析间隔（秒）
            'learning_interval': 1800,  # 学习间隔（秒）
            'min_samples_for_learning': 10,
            'auto_apply_updates': True,
            'alert_thresholds': {
                'max_drawdown': 0.15,
                'min_sharpe': 1.0,
                'min_win_rate': 0.45
            },
            'storage': {
                'data_dir': './feedback_data',
                'auto_save': True,
                'save_interval': 300
            }
        }
        
        if config_path:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    import yaml
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"加载配置文件失败，使用默认配置: {e}")
        
        return default_config
    
    # ==================== 回调注册 ====================
    
    def register_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """注册告警回调"""
        self._alert_callbacks.append(callback)
    
    def register_update_callback(self, callback: Callable[[ModelUpdate], None]):
        """注册更新回调"""
        self._update_callbacks.append(callback)
    
    def _notify_alert(self, alert: PerformanceAlert):
        """通知告警"""
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"告警回调执行失败: {e}")
    
    def _notify_update(self, update: ModelUpdate):
        """通知更新"""
        for callback in self._update_callbacks:
            try:
                callback(update)
            except Exception as e:
                logger.error(f"更新回调执行失败: {e}")
    
    # ==================== 交易记录 ====================
    
    def record_trade(self, trade: TradeRecord) -> bool:
        """
        记录交易
        
        Args:
            trade: 交易记录
            
        Returns:
            是否成功记录
        """
        try:
            self.performance_analyzer.add_trade(trade)
            
            # 检查是否需要触发分析
            if len(self.performance_analyzer.trade_history) % 10 == 0:
                asyncio.create_task(self._quick_analysis())
            
            logger.debug(f"交易记录已添加: {trade.id}")
            return True
        except Exception as e:
            logger.error(f"记录交易失败: {e}")
            return False
    
    def record_trade_from_decision(self, 
                                   decision_id: str,
                                   symbol: str,
                                   side: str,
                                   price: float,
                                   size: float,
                                   timestamp: Optional[datetime] = None) -> str:
        """
        从决策记录交易
        
        Args:
            decision_id: 决策ID
            symbol: 交易对
            side: 方向 (buy/sell)
            price: 价格
            size: 数量
            timestamp: 时间戳
            
        Returns:
            交易记录ID
        """
        trade = TradeRecord(
            id=decision_id,
            symbol=symbol,
            entry_time=timestamp or datetime.now(),
            entry_price=price,
            entry_side=side,
            entry_quantity=size
        )
        
        self.record_trade(trade)
        return trade.id
    
    def close_trade(self, 
                   trade_id: str,
                   exit_price: float,
                   exit_time: Optional[datetime] = None) -> bool:
        """
        关闭交易
        
        Args:
            trade_id: 交易ID
            exit_price: 出场价格
            exit_time: 出场时间
            
        Returns:
            是否成功关闭
        """
        try:
            # 查找交易
            trade = next(
                (t for t in self.performance_analyzer.trade_history if t.id == trade_id),
                None
            )
            
            if trade and not trade.is_closed:
                trade.close_trade(exit_price, exit_time or datetime.now(), "manual")
                
                # 创建学习样本
                if trade.result != TradeResult.OPEN:
                    sample = LearningSample(
                        id=f"sample_{trade_id}",
                        features={},  # 可以从决策上下文获取
                        predicted_signal=trade.entry_side,
                        predicted_confidence=trade.signal_confidence,
                        actual_result=trade.result.value,
                        actual_pnl=trade.realized_pnl,
                        reward=trade.realized_pnl,
                        timestamp=trade.exit_time,
                        metadata={
                            'trade_id': trade_id,
                            'symbol': trade.symbol,
                            'entry_price': trade.entry_price,
                            'exit_price': trade.exit_price,
                            'pnl': trade.realized_pnl
                        }
                    )
                    self.online_learner.add_sample(sample)
                
                return True
            
            return False
        except Exception as e:
            logger.error(f"关闭交易失败: {e}")
            return False
    
    # ==================== 性能分析 ====================
    
    async def analyze_performance(self, 
                                 symbol: Optional[str] = None,
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None) -> FeedbackResult:
        """
        分析性能
        
        Args:
            symbol: 交易对（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）
            
        Returns:
            反馈结果
        """
        try:
            # 执行性能分析
            metrics = self.performance_analyzer.analyze_performance(
                symbol, start_date, end_date
            )
            
            # 检查告警
            alerts = self._check_performance_alerts(metrics)
            for alert in alerts:
                self._notify_alert(alert)
            
            # 生成洞察
            insights = self._generate_insights(metrics)
            
            return FeedbackResult(
                success=True,
                message="性能分析完成",
                insights=insights,
                alerts=alerts,
                metrics=metrics
            )
        except Exception as e:
            logger.error(f"性能分析失败: {e}")
            return FeedbackResult(
                success=False,
                message=f"性能分析失败: {str(e)}"
            )
    
    def _generate_insights(self, metrics: PerformanceMetrics) -> List[StrategyInsight]:
        """生成策略洞察"""
        insights = []
        
        # 胜率洞察
        if metrics.win_rate > 0.6:
            insights.append(StrategyInsight(
                category="win_rate",
                message=f"胜率优秀: {metrics.win_rate:.1%}",
                confidence=0.8
            ))
        elif metrics.win_rate < 0.4:
            insights.append(StrategyInsight(
                category="win_rate",
                message=f"胜率偏低: {metrics.win_rate:.1%}，建议优化策略",
                confidence=0.7
            ))
        
        # 夏普比率洞察
        if metrics.sharpe_ratio > 2:
            insights.append(StrategyInsight(
                category="risk_adjusted_return",
                message=f"夏普比率优秀: {metrics.sharpe_ratio:.2f}",
                confidence=0.85
            ))
        elif metrics.sharpe_ratio < 0.5:
            insights.append(StrategyInsight(
                category="risk_adjusted_return",
                message=f"夏普比率偏低: {metrics.sharpe_ratio:.2f}",
                confidence=0.7
            ))
        
        # 回撤洞察
        if metrics.max_drawdown > 0.2:
            insights.append(StrategyInsight(
                category="drawdown",
                message=f"最大回撤较大: {metrics.max_drawdown:.1%}，建议加强风险控制",
                confidence=0.75
            ))
        
        # 盈亏比洞察
        if metrics.profit_factor > 2:
            insights.append(StrategyInsight(
                category="profit_factor",
                message=f"盈亏比优秀: {metrics.profit_factor:.2f}",
                confidence=0.8
            ))
        
        return insights
    
    def _check_performance_alerts(self, metrics: PerformanceMetrics) -> List[PerformanceAlert]:
        """检查性能告警"""
        alerts = []
        thresholds = self.config['alert_thresholds']
        
        if metrics.max_drawdown > thresholds['max_drawdown']:
            alerts.append(PerformanceAlert(
                alert_type='max_drawdown',
                message=f"最大回撤超过阈值: {metrics.max_drawdown:.2%} > {thresholds['max_drawdown']:.2%}",
                severity='high',
                metric_value=metrics.max_drawdown,
                threshold=thresholds['max_drawdown'],
                timestamp=datetime.now()
            ))
        
        if metrics.sharpe_ratio < thresholds['min_sharpe']:
            alerts.append(PerformanceAlert(
                alert_type='low_sharpe',
                message=f"夏普比率低于阈值: {metrics.sharpe_ratio:.2f} < {thresholds['min_sharpe']}",
                severity='medium',
                metric_value=metrics.sharpe_ratio,
                threshold=thresholds['min_sharpe'],
                timestamp=datetime.now()
            ))
        
        if metrics.win_rate < thresholds['min_win_rate']:
            alerts.append(PerformanceAlert(
                alert_type='low_win_rate',
                message=f"胜率低于阈值: {metrics.win_rate:.2%} < {thresholds['min_win_rate']:.2%}",
                severity='medium',
                metric_value=metrics.win_rate,
                threshold=thresholds['min_win_rate'],
                timestamp=datetime.now()
            ))
        
        return alerts
    
    async def _quick_analysis(self):
        """快速分析（异步）"""
        try:
            result = await self.analyze_performance()
            if result.alerts:
                logger.warning(f"性能告警: {len(result.alerts)} 个")
        except Exception as e:
            logger.error(f"快速分析失败: {e}")
    
    # ==================== 在线学习 ====================
    
    async def run_online_learning(self) -> FeedbackResult:
        """
        运行在线学习
        
        Returns:
            反馈结果
        """
        try:
            # 检查样本数量
            if len(self.online_learner.samples) < self.config['min_samples_for_learning']:
                return FeedbackResult(
                    success=False,
                    message=f"样本不足，当前 {len(self.online_learner.samples)}，需要 {self.config['min_samples_for_learning']}"
                )
            
            # 执行学习
            update = self.online_learner.learn()
            
            # 应用更新
            applied_updates = []
            if update and self.config['auto_apply_updates'] and self.mode == FeedbackMode.AUTO:
                if self._apply_model_update(update):
                    applied_updates.append(update)
                    self._notify_update(update)
            
            return FeedbackResult(
                success=True,
                message=f"在线学习完成，生成 {len(updates)} 个更新，应用 {len(applied_updates)} 个",
                updates=applied_updates
            )
        except Exception as e:
            logger.error(f"在线学习失败: {e}")
            return FeedbackResult(
                success=False,
                message=f"在线学习失败: {str(e)}"
            )
    
    def _apply_model_update(self, update: ModelUpdate) -> bool:
        """应用模型更新"""
        try:
            # 这里可以实现具体的更新应用逻辑
            logger.info(f"应用模型更新: {update.component} -> {update.update_type}")
            return True
        except Exception as e:
            logger.error(f"应用模型更新失败: {e}")
            return False
    
    # ==================== RLHF ====================
    
    def add_human_feedback(self, feedback: HumanFeedback) -> bool:
        """
        添加人类反馈
        
        Args:
            feedback: 人类反馈
            
        Returns:
            是否成功添加
        """
        try:
            self.rlhf_trainer.add_human_feedback(feedback)
            self.feedback_store.save_human_feedback(feedback)
            logger.info(f"人类反馈已添加: {feedback.id}")
            return True
        except Exception as e:
            logger.error(f"添加人类反馈失败: {e}")
            return False
    
    async def train_rlhf(self) -> FeedbackResult:
        """
        训练RLHF模型
        
        Returns:
            反馈结果
        """
        try:
            # 构建偏好对
            self.rlhf_trainer.build_preference_pairs(self.online_learner.samples)
            
            if len(self.rlhf_trainer.preference_pairs) < 5:
                return FeedbackResult(
                    success=False,
                    message=f"偏好对不足，当前 {len(self.rlhf_trainer.preference_pairs)}，需要 5"
                )
            
            # 训练奖励模型
            reward_model = self.rlhf_trainer.train_reward_model()
            
            # 优化策略
            policy_update = self.rlhf_trainer.optimize_policy(self.online_learner.samples)
            
            # 保存模型更新
            if policy_update:
                self.feedback_store.save_model_update(policy_update)
            
            return FeedbackResult(
                success=True,
                message="RLHF训练完成",
                metadata={
                    'reward_model': reward_model,
                    'policy_update': policy_update.changes if policy_update else None
                }
            )
        except Exception as e:
            logger.error(f"RLHF训练失败: {e}")
            return FeedbackResult(
                success=False,
                message=f"RLHF训练失败: {str(e)}"
            )
    
    # ==================== 后台任务 ====================
    
    async def start(self):
        """启动反馈引擎"""
        if self._running:
            logger.warning("反馈引擎已在运行")
            return
        
        self._running = True
        
        # 启动后台任务
        self._analysis_task = asyncio.create_task(self._analysis_loop())
        self._learning_task = asyncio.create_task(self._learning_loop())
        
        logger.info("反馈引擎已启动")
    
    async def stop(self):
        """停止反馈引擎"""
        if not self._running:
            return
        
        self._running = False
        
        # 取消任务
        if self._analysis_task:
            self._analysis_task.cancel()
        if self._learning_task:
            self._learning_task.cancel()
        
        # 保存状态 (通过FeedbackStore持久化)
        for sample in self.online_learner.samples:
            self.feedback_store.save_sample(sample)
        for feedback in self.rlhf_trainer.human_feedback:
            self.feedback_store.save_human_feedback(feedback)
        
        logger.info("反馈引擎已停止")
    
    async def _analysis_loop(self):
        """分析循环"""
        interval = self.config['analysis_interval']
        
        while self._running:
            try:
                await asyncio.sleep(interval)
                await self.analyze_performance()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"分析循环错误: {e}")
    
    async def _learning_loop(self):
        """学习循环"""
        interval = self.config['learning_interval']
        
        while self._running:
            try:
                await asyncio.sleep(interval)
                await self.run_online_learning()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"学习循环错误: {e}")
    
    # ==================== 报告和导出 ====================
    
    def generate_report(self, 
                       days: int = 30,
                       format: str = 'json') -> Dict[str, Any]:
        """
        生成反馈报告
        
        Args:
            days: 报告天数
            format: 格式 (json, dict)
            
        Returns:
            报告数据
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # 性能指标
        metrics = self.performance_analyzer.analyze_performance(
            start_date=start_date,
            end_date=end_date
        )
        
        # 学习统计
        learning_stats = self.online_learner.get_learning_stats()
        
        report = {
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
                'days': days
            },
            'performance': {
                'total_trades': metrics.total_trades,
                'win_rate': metrics.win_rate,
                'profit_factor': metrics.profit_factor,
                'sharpe_ratio': metrics.sharpe_ratio,
                'sortino_ratio': metrics.sortino_ratio,
                'max_drawdown': metrics.max_drawdown,
                'total_pnl': metrics.total_pnl,
                'avg_pnl_per_trade': metrics.avg_pnl_per_trade
            },
            'agents': {
                name: {
                    'current_weight': weight
                }
                for name, weight in self.online_learner.agent_weights.items()
            },
            'learning': learning_stats,
            'rlhf': {
                'feedback_count': len(self.rlhf_trainer.human_feedback),
                'preference_pairs': len(self.rlhf_trainer.preference_pairs)
            }
        }
        
        return report
    
    def export_data(self, filepath: str, format: str = 'json'):
        """
        导出反馈数据
        
        Args:
            filepath: 文件路径
            format: 格式
        """
        data = {
            'trades': [
                {
                    'id': t.id,
                    'symbol': t.symbol,
                    'entry_time': t.entry_time.isoformat(),
                    'entry_price': t.entry_price,
                    'entry_side': t.entry_side,
                    'exit_time': t.exit_time.isoformat() if t.exit_time else None,
                    'exit_price': t.exit_price,
                    'realized_pnl': t.realized_pnl,
                    'result': t.result.value
                }
                for t in self.performance_analyzer.trade_history
            ],
            'samples': [
                {
                    'predicted_signal': s.predicted_signal,
                    'reward': s.reward,
                    'timestamp': s.timestamp.isoformat(),
                    'metadata': s.metadata
                }
                for s in self.online_learner.samples
            ],
            'human_feedback': [
                {
                    'id': f.id,
                    'feedback_type': f.feedback_type.value,
                    'rating': f.rating,
                    'comment': f.comment,
                    'timestamp': f.timestamp.isoformat()
                }
                for f in self.rlhf_trainer.human_feedback
            ]
        }
        
        if format == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"数据已导出: {filepath}")
    
    # ==================== 状态管理 ====================
    
    def get_status(self) -> Dict[str, Any]:
        """获取引擎状态"""
        return {
            'running': self._running,
            'mode': self.mode.value,
            'trades_count': len(self.performance_analyzer.trade_history),
            'samples_count': len(self.online_learner.samples),
            'feedback_count': len(self.rlhf_trainer.human_feedback),
            'agent_weights': self.online_learner.agent_weights
        }


# ==================== 便捷函数 ====================

async def create_feedback_engine(config_path: Optional[str] = None) -> FeedbackEngine:
    """
    创建反馈引擎实例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        反馈引擎实例
    """
    engine = FeedbackEngine(config_path)
    await engine.start()
    return engine
