"""
性能分析器
分析交易性能，计算各种性能指标
"""

import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import logging

from .feedback_types import TradeRecord, PerformanceMetrics, TradeResult, Alert

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    性能分析器
    
    分析交易性能，计算各种指标：
    - 基本统计 (胜率、盈亏比等)
    - 风险调整收益 (Sharpe, Sortino, Calmar)
    - 回撤分析
    - 交易模式分析
    """
    
    def __init__(self, window_days: int = 30):
        """
        初始化性能分析器
        
        Args:
            window_days: 分析窗口 (天)
        """
        self.window_days = window_days
        self.trade_history: List[TradeRecord] = []
        self.equity_curve: List[Dict[str, Any]] = []
        self.alerts: List[Alert] = []
        
    def add_trade(self, trade: TradeRecord):
        """
        添加交易记录
        
        Args:
            trade: 交易记录
        """
        self.trade_history.append(trade)
        
        # 更新权益曲线
        self._update_equity_curve(trade)
        
        logger.debug(f"Trade added: {trade.id} {trade.symbol} {trade.result.value}")
    
    def analyze_performance(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> PerformanceMetrics:
        """
        分析性能
        
        Args:
            symbol: 交易对符号 (可选)
            start_date: 开始日期 (可选)
            end_date: 结束日期 (可选)
            
        Returns:
            PerformanceMetrics: 性能指标
        """
        # 筛选交易
        trades = self._filter_trades(symbol, start_date, end_date)
        
        if not trades:
            return PerformanceMetrics()
        
        metrics = PerformanceMetrics()
        
        # 基本统计
        closed_trades = [t for t in trades if t.is_closed]
        metrics.total_trades = len(closed_trades)
        metrics.winning_trades = sum(1 for t in closed_trades if t.result == TradeResult.WIN)
        metrics.losing_trades = sum(1 for t in closed_trades if t.result == TradeResult.LOSS)
        metrics.break_even_trades = sum(1 for t in closed_trades if t.result == TradeResult.BREAK_EVEN)
        
        # 胜率
        if metrics.total_trades > 0:
            metrics.win_rate = metrics.winning_trades / metrics.total_trades
            metrics.loss_rate = metrics.losing_trades / metrics.total_trades
        
        # 盈亏
        metrics.total_pnl = sum(t.realized_pnl for t in closed_trades)
        if metrics.total_trades > 0:
            metrics.avg_pnl_per_trade = metrics.total_pnl / metrics.total_trades
        
        # 计算盈亏百分比 (基于初始资金假设)
        initial_capital = 100000  # 假设初始资金
        if initial_capital > 0:
            metrics.total_pnl_pct = metrics.total_pnl / initial_capital
        
        # 盈亏比
        wins = [t.realized_pnl for t in closed_trades if t.result == TradeResult.WIN]
        losses = [t.realized_pnl for t in closed_trades if t.result == TradeResult.LOSS]
        
        metrics.avg_win = np.mean(wins) if wins else 0.0
        metrics.avg_loss = np.mean(losses) if losses else 0.0
        
        if metrics.avg_loss != 0:
            metrics.avg_win_loss_ratio = abs(metrics.avg_win / metrics.avg_loss)
        
        # 利润因子
        total_wins = sum(wins) if wins else 0.0
        total_losses = abs(sum(losses)) if losses else 0.0
        
        if total_losses > 0:
            metrics.profit_factor = total_wins / total_losses
        elif total_wins > 0:
            metrics.profit_factor = float('inf')
        
        # 风险调整收益
        returns = self._calculate_returns(closed_trades)
        metrics.sharpe_ratio = self._calculate_sharpe_ratio(returns)
        metrics.sortino_ratio = self._calculate_sortino_ratio(returns)
        
        # 回撤
        drawdowns = self._calculate_drawdowns()
        if drawdowns:
            metrics.max_drawdown = max(drawdowns)
            metrics.max_drawdown_pct = metrics.max_drawdown
            metrics.current_drawdown = drawdowns[-1] if drawdowns else 0.0
        
        # Calmar 比率
        if metrics.max_drawdown_pct > 0:
            annual_return = metrics.total_pnl_pct * (365 / self.window_days) if self.window_days > 0 else 0
            metrics.calmar_ratio = annual_return / metrics.max_drawdown_pct
        
        # 平均持仓时间
        holding_periods = [t.get_holding_period() for t in closed_trades if t.get_holding_period()]
        if holding_periods:
            metrics.avg_holding_period_hours = np.mean(holding_periods)
        
        # 期望值
        metrics.expectancy = self._calculate_expectancy(metrics)
        
        # 时间范围
        metrics.start_date = min(t.entry_time for t in trades)
        metrics.end_date = max(t.exit_time or t.entry_time for t in trades)
        
        return metrics
    
    def analyze_by_symbol(self) -> Dict[str, PerformanceMetrics]:
        """
        按交易对分析性能
        
        Returns:
            Dict[str, PerformanceMetrics]: 每个交易对的性能指标
        """
        symbols = set(t.symbol for t in self.trade_history)
        
        results = {}
        for symbol in symbols:
            results[symbol] = self.analyze_performance(symbol=symbol)
        
        return results
    
    def analyze_by_signal_type(self) -> Dict[str, PerformanceMetrics]:
        """
        按信号类型分析性能
        
        Returns:
            Dict[str, PerformanceMetrics]: 每种信号类型的性能指标
        """
        signal_types = defaultdict(list)
        
        for trade in self.trade_history:
            signal_type = trade.metadata.get('signal_type', 'unknown')
            signal_types[signal_type].append(trade)
        
        results = {}
        for signal_type, trades in signal_types.items():
            # 临时替换交易历史
            original_history = self.trade_history
            self.trade_history = trades
            
            results[signal_type] = self.analyze_performance()
            
            # 恢复
            self.trade_history = original_history
        
        return results
    
    def get_recent_trades(self, n: int = 10) -> List[TradeRecord]:
        """
        获取最近的交易
        
        Args:
            n: 数量
            
        Returns:
            List[TradeRecord]: 最近的交易记录
        """
        return sorted(
            self.trade_history,
            key=lambda t: t.entry_time,
            reverse=True
        )[:n]
    
    def get_winning_streak(self) -> int:
        """获取当前连胜次数"""
        streak = 0
        for trade in reversed(self.trade_history):
            if not trade.is_closed:
                continue
            if trade.result == TradeResult.WIN:
                streak += 1
            elif trade.result == TradeResult.LOSS:
                break
        return streak
    
    def get_losing_streak(self) -> int:
        """获取当前连败次数"""
        streak = 0
        for trade in reversed(self.trade_history):
            if not trade.is_closed:
                continue
            if trade.result == TradeResult.LOSS:
                streak += 1
            elif trade.result == TradeResult.WIN:
                break
        return streak
    
    def check_alerts(self) -> List[Alert]:
        """
        检查告警条件
        
        Returns:
            List[Alert]: 触发的告警
        """
        new_alerts = []
        
        # 检查回撤
        metrics = self.analyze_performance()
        
        if metrics.current_drawdown > 0.10:  # 10% 回撤
            alert = Alert(
                id=f"drawdown_{datetime.now().timestamp()}",
                alert_type="high_drawdown",
                severity="warning" if metrics.current_drawdown < 0.15 else "critical",
                message=f"Current drawdown is {metrics.current_drawdown:.1%}",
                trigger_value=metrics.current_drawdown,
                threshold=0.10
            )
            new_alerts.append(alert)
        
        # 检查连败
        losing_streak = self.get_losing_streak()
        if losing_streak >= 5:
            alert = Alert(
                id=f"losing_streak_{datetime.now().timestamp()}",
                alert_type="losing_streak",
                severity="warning",
                message=f"Losing streak of {losing_streak} trades",
                trigger_value=losing_streak,
                threshold=5
            )
            new_alerts.append(alert)
        
        # 检查胜率
        if metrics.total_trades >= 20 and metrics.win_rate < 0.40:
            alert = Alert(
                id=f"low_win_rate_{datetime.now().timestamp()}",
                alert_type="low_win_rate",
                severity="warning",
                message=f"Win rate is {metrics.win_rate:.1%}",
                trigger_value=metrics.win_rate,
                threshold=0.40
            )
            new_alerts.append(alert)
        
        self.alerts.extend(new_alerts)
        
        return new_alerts
    
    def generate_report(self) -> Dict[str, Any]:
        """
        生成性能报告
        
        Returns:
            Dict: 性能报告
        """
        metrics = self.analyze_performance()
        by_symbol = self.analyze_by_symbol()
        
        return {
            'summary': {
                'total_trades': metrics.total_trades,
                'win_rate': f"{metrics.win_rate:.1%}",
                'total_pnl': f"${metrics.total_pnl:.2f}",
                'total_pnl_pct': f"{metrics.total_pnl_pct:.2%}",
                'profit_factor': f"{metrics.profit_factor:.2f}",
                'sharpe_ratio': f"{metrics.sharpe_ratio:.2f}",
                'max_drawdown': f"{metrics.max_drawdown_pct:.2%}",
                'expectancy': f"${metrics.expectancy:.2f}"
            },
            'by_symbol': {
                symbol: {
                    'trades': m.total_trades,
                    'win_rate': f"{m.win_rate:.1%}",
                    'pnl': f"${m.total_pnl:.2f}"
                }
                for symbol, m in by_symbol.items()
            },
            'recent_trades': [
                {
                    'symbol': t.symbol,
                    'result': t.result.value,
                    'pnl': f"${t.realized_pnl:.2f}",
                    'pnl_pct': f"{t.realized_pnl_pct:.2%}"
                }
                for t in self.get_recent_trades(5)
            ],
            'streaks': {
                'winning_streak': self.get_winning_streak(),
                'losing_streak': self.get_losing_streak()
            },
            'alerts': [
                {
                    'type': a.alert_type,
                    'severity': a.severity,
                    'message': a.message
                }
                for a in self.alerts[-5:]  # 最近5个告警
            ]
        }
    
    def _filter_trades(
        self,
        symbol: Optional[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> List[TradeRecord]:
        """筛选交易"""
        trades = self.trade_history
        
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        
        if start_date:
            trades = [t for t in trades if t.entry_time >= start_date]
        
        if end_date:
            trades = [t for t in trades if t.entry_time <= end_date]
        
        return trades
    
    def _update_equity_curve(self, trade: TradeRecord):
        """更新权益曲线"""
        if not trade.is_closed:
            return
        
        # 计算累计盈亏
        cumulative_pnl = sum(t.realized_pnl for t in self.trade_history if t.is_closed)
        
        self.equity_curve.append({
            'timestamp': trade.exit_time,
            'trade_id': trade.id,
            'pnl': trade.realized_pnl,
            'cumulative_pnl': cumulative_pnl
        })
    
    def _calculate_returns(self, trades: List[TradeRecord]) -> List[float]:
        """计算收益率序列"""
        returns = []
        
        for trade in trades:
            if trade.entry_price > 0:
                ret = trade.realized_pnl / (trade.entry_price * trade.entry_quantity)
                returns.append(ret)
        
        return returns
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """计算 Sharpe 比率"""
        if not returns:
            return 0.0
        
        returns_array = np.array(returns)
        avg_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return == 0:
            return 0.0
        
        return (avg_return - risk_free_rate) / std_return
    
    def _calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """计算 Sortino 比率"""
        if not returns:
            return 0.0
        
        returns_array = np.array(returns)
        avg_return = np.mean(returns_array)
        
        # 下行标准差
        downside_returns = returns_array[returns_array < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
        
        if downside_std == 0:
            return 0.0
        
        return (avg_return - risk_free_rate) / downside_std
    
    def _calculate_drawdowns(self) -> List[float]:
        """计算回撤序列"""
        if not self.equity_curve:
            return []
        
        drawdowns = []
        peak = 0
        
        for point in self.equity_curve:
            equity = point['cumulative_pnl']
            
            if equity > peak:
                peak = equity
            
            if peak > 0:
                drawdown = (peak - equity) / peak if peak > 0 else 0
                drawdowns.append(drawdown)
        
        return drawdowns
    
    def _calculate_expectancy(self, metrics: PerformanceMetrics) -> float:
        """计算期望值"""
        if metrics.total_trades == 0:
            return 0.0
        
        return (metrics.win_rate * metrics.avg_win) + (metrics.loss_rate * metrics.avg_loss)
