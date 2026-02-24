"""
OpenClaw 反馈层 - 绩效跟踪与策略优化
第5层: 反馈层

功能:
1. 绩效指标计算
2. 交易分析
3. 策略优化建议
4. 风险监控
5. 报告生成
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """交易记录"""
    id: str
    symbol: str
    direction: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    size: float
    leverage: float
    pnl: float
    pnl_percent: float
    exit_reason: str
    signal_score: float
    signal_confidence: float
    market_regime: str


@dataclass
class DailyPerformance:
    """每日绩效"""
    date: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0


class PerformanceTracker:
    """绩效跟踪器"""
    
    def __init__(self):
        self.trades: List[TradeRecord] = []
        self.daily_performance: Dict[str, DailyPerformance] = {}
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.initial_capital = 10000.0
        self.current_equity = self.initial_capital
        
        # 统计缓存
        self._cache = {}
        self._cache_time = None
    
    def add_trade(self, trade: Dict):
        """添加交易记录"""
        record = TradeRecord(
            id=trade.get('position_id', ''),
            symbol=trade.get('symbol', ''),
            direction=trade.get('direction', ''),
            entry_time=trade.get('entry_time', datetime.now()),
            exit_time=trade.get('timestamp', datetime.now()),
            entry_price=trade.get('entry_price', 0),
            exit_price=trade.get('exit_price', 0),
            size=trade.get('size', 0),
            leverage=trade.get('leverage', 1),
            pnl=trade.get('pnl', 0),
            pnl_percent=trade.get('pnl_percent', 0),
            exit_reason=trade.get('exit_reason', ''),
            signal_score=trade.get('signal_score', 0),
            signal_confidence=trade.get('signal_confidence', 0),
            market_regime=trade.get('market_regime', 'UNKNOWN')
        )
        
        self.trades.append(record)
        self.current_equity += record.pnl
        self.equity_curve.append((record.exit_time, self.current_equity))
        
        # 更新每日绩效
        date_str = record.exit_time.strftime('%Y-%m-%d')
        if date_str not in self.daily_performance:
            self.daily_performance[date_str] = DailyPerformance(date=date_str)
        
        daily = self.daily_performance[date_str]
        daily.trades += 1
        daily.gross_pnl += record.pnl
        
        if record.pnl > 0:
            daily.wins += 1
        else:
            daily.losses += 1
        
        daily.win_rate = daily.wins / daily.trades * 100 if daily.trades > 0 else 0
        
        # 清空缓存
        self._cache = {}
        
        logger.info(f"交易记录已添加: {record.symbol} {record.exit_reason} PnL:{record.pnl:.2f}")
    
    def calculate_metrics(self) -> Dict:
        """计算绩效指标"""
        if not self.trades:
            return self._empty_metrics()
        
        # 检查缓存
        if self._cache and self._cache_time == len(self.trades):
            return self._cache
        
        pnls = [t.pnl for t in self.trades]
        pnl_percents = [t.pnl_percent for t in self.trades]
        
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        metrics = {
            'total_trades': len(self.trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / len(pnls) * 100 if pnls else 0,
            
            'total_pnl': sum(pnls),
            'total_pnl_percent': sum(pnl_percents),
            'avg_pnl': np.mean(pnls) if pnls else 0,
            'avg_pnl_percent': np.mean(pnl_percents) if pnl_percents else 0,
            
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'max_win': max(wins) if wins else 0,
            'max_loss': min(losses) if losses else 0,
            
            'profit_factor': abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf'),
            
            'sharpe_ratio': self._calculate_sharpe(),
            'sortino_ratio': self._calculate_sortino(),
            'max_drawdown': self._calculate_max_drawdown(),
            
            'avg_trade_duration': self._calculate_avg_duration(),
            
            # 按方向统计
            'long_trades': len([t for t in self.trades if t.direction == 'LONG']),
            'short_trades': len([t for t in self.trades if t.direction == 'SHORT']),
            'long_win_rate': self._calculate_direction_win_rate('LONG'),
            'short_win_rate': self._calculate_direction_win_rate('SHORT'),
            
            # 按出场原因统计
            'exit_analysis': self._analyze_exit_reasons(),
            
            # 按信号质量统计
            'score_analysis': self._analyze_by_score(),
            
            # 按市场状态统计
            'regime_analysis': self._analyze_by_regime(),
        }
        
        self._cache = metrics
        self._cache_time = len(self.trades)
        
        return metrics
    
    def _empty_metrics(self) -> Dict:
        """空指标"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'total_pnl_percent': 0,
            'avg_pnl': 0,
            'avg_pnl_percent': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'max_win': 0,
            'max_loss': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'max_drawdown': 0,
            'avg_trade_duration': 0,
            'long_trades': 0,
            'short_trades': 0,
            'long_win_rate': 0,
            'short_win_rate': 0,
            'exit_analysis': {},
            'score_analysis': {},
            'regime_analysis': {}
        }
    
    def _calculate_sharpe(self, risk_free_rate: float = 0.0) -> float:
        """计算夏普比率"""
        if len(self.trades) < 2:
            return 0.0
        
        returns = [t.pnl_percent for t in self.trades]
        excess_returns = [r - risk_free_rate for r in returns]
        
        std = np.std(excess_returns)
        if std == 0:
            return 0.0
        
        return np.mean(excess_returns) / std * np.sqrt(365)  # 年化
    
    def _calculate_sortino(self, risk_free_rate: float = 0.0) -> float:
        """计算索提诺比率"""
        if len(self.trades) < 2:
            return 0.0
        
        returns = [t.pnl_percent for t in self.trades]
        excess_returns = [r - risk_free_rate for r in returns]
        
        downside_returns = [r for r in excess_returns if r < 0]
        downside_std = np.std(downside_returns) if downside_returns else 1e-10
        
        return np.mean(excess_returns) / downside_std * np.sqrt(365)
    
    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        if not self.equity_curve:
            return 0.0
        
        equity_values = [e[1] for e in self.equity_curve]
        peak = equity_values[0]
        max_dd = 0.0
        
        for equity in equity_values:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd * 100
    
    def _calculate_avg_duration(self) -> float:
        """计算平均持仓时间（分钟）"""
        if not self.trades:
            return 0.0
        
        durations = [(t.exit_time - t.entry_time).total_seconds() / 60 for t in self.trades]
        return np.mean(durations)
    
    def _calculate_direction_win_rate(self, direction: str) -> float:
        """计算方向胜率"""
        trades = [t for t in self.trades if t.direction == direction]
        if not trades:
            return 0.0
        
        wins = len([t for t in trades if t.pnl > 0])
        return wins / len(trades) * 100
    
    def _analyze_exit_reasons(self) -> Dict:
        """分析出场原因"""
        analysis = {}
        
        for reason in set(t.exit_reason for t in self.trades):
            trades = [t for t in self.trades if t.exit_reason == reason]
            wins = len([t for t in trades if t.pnl > 0])
            
            analysis[reason] = {
                'count': len(trades),
                'win_rate': wins / len(trades) * 100 if trades else 0,
                'avg_pnl': np.mean([t.pnl for t in trades]) if trades else 0,
                'total_pnl': sum(t.pnl for t in trades)
            }
        
        return analysis
    
    def _analyze_by_score(self) -> Dict:
        """按信号评分分析"""
        analysis = {'high': {'count': 0, 'win_rate': 0, 'avg_pnl': 0},
                    'medium': {'count': 0, 'win_rate': 0, 'avg_pnl': 0},
                    'low': {'count': 0, 'win_rate': 0, 'avg_pnl': 0}}
        
        for trade in self.trades:
            score = trade.signal_score
            
            if score >= 8:
                key = 'high'
            elif score >= 6.5:
                key = 'medium'
            else:
                key = 'low'
            
            analysis[key]['count'] += 1
            if trade.pnl > 0:
                analysis[key]['wins'] = analysis[key].get('wins', 0) + 1
        
        for key in analysis:
            if analysis[key]['count'] > 0:
                analysis[key]['win_rate'] = analysis[key].get('wins', 0) / analysis[key]['count'] * 100
        
        return analysis
    
    def _analyze_by_regime(self) -> Dict:
        """按市场状态分析"""
        analysis = {}
        
        for regime in set(t.market_regime for t in self.trades):
            trades = [t for t in self.trades if t.market_regime == regime]
            wins = len([t for t in trades if t.pnl > 0])
            
            analysis[regime] = {
                'count': len(trades),
                'win_rate': wins / len(trades) * 100 if trades else 0,
                'avg_pnl': np.mean([t.pnl for t in trades]) if trades else 0,
                'total_pnl': sum(t.pnl for t in trades)
            }
        
        return analysis
    
    def generate_optimization_suggestions(self) -> List[str]:
        """生成优化建议"""
        suggestions = []
        metrics = self.calculate_metrics()
        
        if metrics['total_trades'] < 30:
            suggestions.append("交易样本不足，建议继续收集数据后再进行优化")
            return suggestions
        
        # 胜率分析
        if metrics['win_rate'] < 50:
            suggestions.append(f"胜率偏低 ({metrics['win_rate']:.1f}%)，建议提高信号过滤条件")
        elif metrics['win_rate'] > 70:
            suggestions.append(f"胜率优秀 ({metrics['win_rate']:.1f}%)，可考虑适当增加仓位")
        
        # 盈亏比分析
        profit_factor = metrics['profit_factor']
        if profit_factor < 1.5:
            suggestions.append(f"盈亏比偏低 ({profit_factor:.2f})，建议优化止盈止损比例")
        
        # 最大回撤
        if metrics['max_drawdown'] > 20:
            suggestions.append(f"最大回撤较大 ({metrics['max_drawdown']:.1f}%)，建议加强风险管理")
        
        # 方向分析
        long_wr = metrics['long_win_rate']
        short_wr = metrics['short_win_rate']
        if long_wr > short_wr + 15:
            suggestions.append(f"做多胜率 ({long_wr:.1f}%) 显著高于做空 ({short_wr:.1f}%)，可考虑偏向多头策略")
        elif short_wr > long_wr + 15:
            suggestions.append(f"做空胜率 ({short_wr:.1f}%) 显著高于做多 ({long_wr:.1f}%)，可考虑偏向空头策略")
        
        # 信号质量分析
        score_analysis = metrics['score_analysis']
        if 'high' in score_analysis and 'low' in score_analysis:
            high_wr = score_analysis['high'].get('win_rate', 0)
            low_wr = score_analysis['low'].get('win_rate', 0)
            if high_wr > low_wr + 20:
                suggestions.append(f"高质量信号胜率 ({high_wr:.1f}%) 显著高于低质量 ({low_wr:.1f}%)，建议提高入场门槛")
        
        # 出场原因分析
        exit_analysis = metrics['exit_analysis']
        if 'STOP_LOSS' in exit_analysis:
            sl_stats = exit_analysis['STOP_LOSS']
            if sl_stats['count'] / metrics['total_trades'] > 0.5:
                suggestions.append(f"止损出场比例较高 ({sl_stats['count']}/{metrics['total_trades']})，建议放宽止损或优化入场时机")
        
        return suggestions
    
    def generate_report(self) -> str:
        """生成绩效报告"""
        metrics = self.calculate_metrics()
        suggestions = self.generate_optimization_suggestions()
        
        report = f"""
{'='*60}
OpenClaw 策略绩效报告
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

【交易统计】
总交易次数: {metrics['total_trades']}
盈利次数: {metrics['winning_trades']}
亏损次数: {metrics['losing_trades']}
胜率: {metrics['win_rate']:.2f}%

【盈亏统计】
总盈亏: {metrics['total_pnl']:.2f} USDT ({metrics['total_pnl_percent']:.2f}%)
平均盈亏: {metrics['avg_pnl']:.2f} USDT ({metrics['avg_pnl_percent']:.2f}%)
平均盈利: {metrics['avg_win']:.2f} USDT
平均亏损: {metrics['avg_loss']:.2f} USDT
最大盈利: {metrics['max_win']:.2f} USDT
最大亏损: {metrics['max_loss']:.2f} USDT
盈亏比: {metrics['profit_factor']:.2f}

【风险指标】
夏普比率: {metrics['sharpe_ratio']:.2f}
索提诺比率: {metrics['sortino_ratio']:.2f}
最大回撤: {metrics['max_drawdown']:.2f}%
平均持仓时间: {metrics['avg_trade_duration']:.1f} 分钟

【方向统计】
做多交易: {metrics['long_trades']} 笔 (胜率: {metrics['long_win_rate']:.1f}%)
做空交易: {metrics['short_trades']} 笔 (胜率: {metrics['short_win_rate']:.1f}%)

【优化建议】
"""
        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                report += f"{i}. {suggestion}\n"
        else:
            report += "暂无优化建议\n"
        
        report += f"{'='*60}\n"
        
        return report
    
    def export_trades(self, filepath: str):
        """导出交易记录"""
        if not self.trades:
            logger.warning("没有交易记录可导出")
            return
        
        data = []
        for t in self.trades:
            data.append({
                'id': t.id,
                'symbol': t.symbol,
                'direction': t.direction,
                'entry_time': t.entry_time.isoformat(),
                'exit_time': t.exit_time.isoformat(),
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'size': t.size,
                'leverage': t.leverage,
                'pnl': t.pnl,
                'pnl_percent': t.pnl_percent,
                'exit_reason': t.exit_reason,
                'signal_score': t.signal_score,
                'signal_confidence': t.signal_confidence,
                'market_regime': t.market_regime
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        logger.info(f"交易记录已导出: {filepath}")
    
    def get_equity_curve_data(self) -> List[Dict]:
        """获取权益曲线数据"""
        return [{'time': t.isoformat(), 'equity': e} for t, e in self.equity_curve]
