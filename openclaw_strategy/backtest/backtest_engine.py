"""
OpenClaw 回测引擎
用于验证策略历史表现
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from ..core.strategy import MTFMomentumStrategy, StrategyConfig, Signal, SignalDirection
from ..core.decision.decision_engine import DecisionEngine
from ..core.feedback.performance_tracker import PerformanceTracker
from ..core.features.technical_indicators import FeatureEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.strategy = MTFMomentumStrategy(config)
        self.decision = DecisionEngine(config)
        self.tracker = PerformanceTracker()
        self.feature_engine = FeatureEngine()
        
        # 回测状态
        self.current_time = None
        self.equity_history = []
        self.signals_history = []
    
    def run(self, 
            data_1h: pd.DataFrame,
            data_15m: pd.DataFrame,
            data_5m: Optional[pd.DataFrame] = None,
            sentiment_data: Optional[pd.DataFrame] = None,
            initial_capital: float = 10000.0) -> Dict:
        """
        运行回测
        
        Args:
            data_1h: 1小时K线数据
            data_15m: 15分钟K线数据
            data_5m: 5分钟K线数据 (可选)
            sentiment_data: 情绪数据 (可选)
            initial_capital: 初始资金
        """
        
        logger.info(f"开始回测: {self.config.name}")
        logger.info(f"初始资金: {initial_capital} USDT")
        
        self.decision.account_balance = initial_capital
        self.tracker.initial_capital = initial_capital
        self.tracker.current_equity = initial_capital
        
        # 确保数据按时间排序
        data_1h = data_1h.sort_values('timestamp').reset_index(drop=True)
        data_15m = data_15m.sort_values('timestamp').reset_index(drop=True)
        
        # 按15分钟时间框架遍历
        for i in range(200, len(data_15m)):
            current_time = data_15m['timestamp'].iloc[i]
            self.current_time = current_time
            
            # 获取当前窗口数据
            df_15m = data_15m.iloc[:i+1]
            
            # 获取对应的1H数据
            df_1h = data_1h[data_1h['timestamp'] <= current_time]
            if len(df_1h) < 50:
                continue
            
            # 获取5M数据 (可选)
            df_5m = None
            if data_5m is not None:
                df_5m = data_5m[data_5m['timestamp'] <= current_time]
            
            # 获取情绪数据
            sentiment = None
            if sentiment_data is not None:
                sentiment_row = sentiment_data[sentiment_data['timestamp'] <= current_time]
                if len(sentiment_row) > 0:
                    sentiment = {
                        'fear_greed': sentiment_row['fear_greed'].iloc[-1],
                        'funding_rate': sentiment_row['funding_rate'].iloc[-1],
                        'long_short_ratio': sentiment_row['long_short_ratio'].iloc[-1]
                    }
            
            # 分析市场
            context = self.strategy.analyze_market(df_1h, df_15m, df_5m, sentiment)
            
            # 更新持仓
            current_price = df_15m['close'].iloc[-1]
            self.decision.update_positions({context.symbol: current_price})
            
            # 检查出场条件
            exit_orders = self.decision.check_exit_signals({context.symbol: current_price})
            for exit_order in exit_orders:
                trade = self.decision.close_position(
                    exit_order['position_id'],
                    exit_order['price'],
                    exit_order['reason']
                )
                if trade:
                    self.tracker.add_trade(trade)
            
            # 生成信号
            signal = self.strategy.generate_signal(context)
            
            if signal:
                self.signals_history.append({
                    'time': current_time,
                    'signal': signal
                })
                
                # 处理信号
                market_data = {
                    'volatility_percentile': context.volatility_percentile,
                    'current_price': current_price
                }
                
                order = self.decision.process_signal(signal, market_data)
                
                if order:
                    # 模拟执行 (使用下一根K线开盘价或当前收盘价)
                    execution_price = current_price
                    position = self.decision.execute_order(order, execution_price)
            
            # 记录权益
            summary = self.decision.get_portfolio_summary()
            self.equity_history.append({
                'time': current_time,
                'equity': summary['total_equity'],
                'balance': summary['account_balance'],
                'unrealized': summary['unrealized_pnl']
            })
            
            # 每100根K线输出进度
            if i % 100 == 0:
                logger.info(f"回测进度: {i}/{len(data_15m)} 权益: {summary['total_equity']:.2f}")
        
        # 生成回测结果
        return self._generate_results()
    
    def _generate_results(self) -> Dict:
        """生成回测结果"""
        metrics = self.tracker.calculate_metrics()
        portfolio = self.decision.get_portfolio_summary()
        
        results = {
            'config': {
                'name': self.config.name,
                'symbols': self.config.symbols,
                'timeframes': {
                    'primary': self.config.primary_timeframe,
                    'confirmation': self.config.confirmation_timeframe
                }
            },
            'performance': metrics,
            'portfolio': portfolio,
            'equity_curve': self.equity_history,
            'signals': len(self.signals_history),
            'trades': len(self.tracker.trades),
            'strategy_stats': self.strategy.get_stats()
        }
        
        # 打印摘要
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict):
        """打印回测摘要"""
        p = results['performance']
        portfolio = results['portfolio']
        
        print("\n" + "="*60)
        print(f"回测完成: {self.config.name}")
        print("="*60)
        print(f"\n【交易统计】")
        print(f"总交易次数: {p['total_trades']}")
        print(f"盈利次数: {p['winning_trades']}")
        print(f"亏损次数: {p['losing_trades']}")
        print(f"胜率: {p['win_rate']:.2f}%")
        
        print(f"\n【盈亏统计】")
        print(f"总盈亏: {p['total_pnl']:.2f} USDT ({p['total_pnl_percent']:.2f}%)")
        print(f"平均盈利: {p['avg_win']:.2f} USDT")
        print(f"平均亏损: {p['avg_loss']:.2f} USDT")
        print(f"盈亏比: {p['profit_factor']:.2f}")
        
        print(f"\n【风险指标】")
        print(f"最大回撤: {p['max_drawdown']:.2f}%")
        print(f"夏普比率: {p['sharpe_ratio']:.2f}")
        print(f"索提诺比率: {p['sortino_ratio']:.2f}")
        
        print(f"\n【方向统计】")
        print(f"做多胜率: {p['long_win_rate']:.1f}% ({p['long_trades']}笔)")
        print(f"做空胜率: {p['short_win_rate']:.1f}% ({p['short_trades']}笔)")
        
        print(f"\n【最终权益】")
        print(f"账户余额: {portfolio['account_balance']:.2f} USDT")
        print(f"未实现盈亏: {portfolio['unrealized_pnl']:.2f} USDT")
        print(f"总权益: {portfolio['total_equity']:.2f} USDT")
        print("="*60 + "\n")
    
    def generate_report(self) -> str:
        """生成详细报告"""
        return self.tracker.generate_report()
    
    def export_results(self, filepath: str):
        """导出回测结果"""
        results = self._generate_results()
        
        # 导出权益曲线
        equity_df = pd.DataFrame(self.equity_history)
        equity_df.to_csv(f"{filepath}_equity.csv", index=False)
        
        # 导出交易记录
        self.tracker.export_trades(f"{filepath}_trades.csv")
        
        # 导出汇总
        with open(f"{filepath}_summary.json", 'w') as f:
            import json
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"回测结果已导出: {filepath}")


def create_sample_data(days: int = 30, symbol: str = "BTCUSDT") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    创建示例数据用于测试回测
    
    生成模拟的1H和15M K线数据
    """
    np.random.seed(42)
    
    # 生成1H数据
    hours = days * 24
    timestamps_1h = pd.date_range(end=datetime.now(), periods=hours, freq='1H')
    
    # 生成价格序列 (带趋势的随机游走)
    returns = np.random.normal(0.0002, 0.015, hours)  # 轻微正收益，1.5%波动
    
    # 添加趋势
    trend = np.sin(np.linspace(0, 4*np.pi, hours)) * 0.005
    returns += trend
    
    prices = 40000 * np.exp(np.cumsum(returns))
    
    # 生成OHLCV
    data_1h = pd.DataFrame({
        'timestamp': timestamps_1h,
        'symbol': symbol,
        'open': prices * (1 + np.random.normal(0, 0.001, hours)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.008, hours))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.008, hours))),
        'close': prices,
        'volume': np.random.uniform(100, 1000, hours) * prices
    })
    
    # 确保OHLC关系正确
    data_1h['high'] = np.maximum(data_1h[['open', 'close', 'high']].max(axis=1), data_1h['high'])
    data_1h['low'] = np.minimum(data_1h[['open', 'close', 'low']].min(axis=1), data_1h['low'])
    
    # 生成15M数据 (通过插值)
    minutes_15 = days * 24 * 4
    timestamps_15m = pd.date_range(end=datetime.now(), periods=minutes_15, freq='15min')
    
    # 插值价格
    from scipy import interpolate
    x_1h = np.arange(len(data_1h))
    x_15m = np.linspace(0, len(data_1h)-1, minutes_15)
    
    f_close = interpolate.interp1d(x_1h, data_1h['close'], kind='linear')
    prices_15m = f_close(x_15m)
    
    # 添加噪声
    prices_15m *= (1 + np.random.normal(0, 0.003, minutes_15))
    
    data_15m = pd.DataFrame({
        'timestamp': timestamps_15m,
        'symbol': symbol,
        'open': prices_15m * (1 + np.random.normal(0, 0.001, minutes_15)),
        'high': prices_15m * (1 + np.abs(np.random.normal(0, 0.005, minutes_15))),
        'low': prices_15m * (1 - np.abs(np.random.normal(0, 0.005, minutes_15))),
        'close': prices_15m,
        'volume': np.random.uniform(20, 200, minutes_15) * prices_15m
    })
    
    data_15m['high'] = np.maximum(data_15m[['open', 'close', 'high']].max(axis=1), data_15m['high'])
    data_15m['low'] = np.minimum(data_15m[['open', 'close', 'low']].min(axis=1), data_15m['low'])
    
    return data_1h, data_15m


if __name__ == "__main__":
    # 测试回测
    config = StrategyConfig(
        name="MTF_Momentum_BTC",
        symbols=["BTCUSDT"],
        min_confidence=60,
        min_score=5.0
    )
    
    # 创建示例数据
    data_1h, data_15m = create_sample_data(days=30)
    
    # 运行回测
    engine = BacktestEngine(config)
    results = engine.run(data_1h, data_15m, initial_capital=10000.0)
    
    # 生成报告
    print(engine.generate_report())
