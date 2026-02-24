"""
OpenClaw 策略交易系统 - 主入口

多时间框架趋势动量策略 (MTF-Momentum Strategy)

策略特点:
1. 多时间框架确认 - 1H定趋势，15M找入场点
2. 多因子评分系统 - 趋势、动量、成交量、支撑阻力
3. 严格风险管理 - ATR止损、追踪止损、部分止盈
4. 情绪过滤 - 恐惧贪婪指数过滤极端行情
5. 自适应仓位 - 根据信号质量和波动率调整

使用方法:
    python main.py --mode backtest --symbol BTCUSDT --days 30
    python main.py --mode live --config config/strategy_config.yaml
"""
import argparse
import yaml
import logging
from datetime import datetime
from typing import Optional

from core.strategy import StrategyConfig, MTFMomentumStrategy
from core.decision.decision_engine import DecisionEngine
from core.feedback.performance_tracker import PerformanceTracker
from backtest.backtest_engine import BacktestEngine, create_sample_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> StrategyConfig:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    strategy_config = config_dict.get('strategy', {})
    
    return StrategyConfig(
        name=strategy_config.get('name', 'MTF_Momentum_Strategy'),
        symbols=config_dict.get('symbols', ['BTCUSDT']),
        primary_timeframe=config_dict.get('timeframes', {}).get('primary', '15m'),
        confirmation_timeframe=config_dict.get('timeframes', {}).get('confirmation', '1h'),
        entry_timeframe=config_dict.get('timeframes', {}).get('entry', '5m'),
        ema_fast=config_dict.get('trend', {}).get('ema_fast', 9),
        ema_slow=config_dict.get('trend', {}).get('ema_slow', 21),
        adx_threshold=config_dict.get('trend', {}).get('adx_threshold', 25.0),
        rsi_period=config_dict.get('momentum', {}).get('rsi_period', 14),
        rsi_overbought=config_dict.get('momentum', {}).get('rsi_overbought', 70.0),
        rsi_oversold=config_dict.get('momentum', {}).get('rsi_oversold', 30.0),
        macd_fast=config_dict.get('momentum', {}).get('macd_fast', 12),
        macd_slow=config_dict.get('momentum', {}).get('macd_slow', 26),
        macd_signal=config_dict.get('momentum', {}).get('macd_signal', 9),
        atr_period=config_dict.get('volatility', {}).get('atr_period', 14),
        atr_multiplier_sl=config_dict.get('volatility', {}).get('atr_multiplier_sl', 2.0),
        atr_multiplier_tp1=config_dict.get('volatility', {}).get('atr_multiplier_tp1', 2.5),
        atr_multiplier_tp2=config_dict.get('volatility', {}).get('atr_multiplier_tp2', 4.0),
        atr_multiplier_tp3=config_dict.get('volatility', {}).get('atr_multiplier_tp3', 6.0),
        min_confidence=config_dict.get('signal_filter', {}).get('min_confidence', 65.0),
        min_score=config_dict.get('signal_filter', {}).get('min_score', 6.0),
        use_sentiment_filter=config_dict.get('signal_filter', {}).get('use_sentiment_filter', True),
        fear_greed_threshold=config_dict.get('signal_filter', {}).get('fear_greed_threshold', 20),
        max_position_size=config_dict.get('risk_management', {}).get('max_position_size', 0.1),
        max_leverage=config_dict.get('risk_management', {}).get('max_leverage', 5.0),
        risk_per_trade=config_dict.get('risk_management', {}).get('risk_per_trade', 0.01),
        max_daily_loss=config_dict.get('risk_management', {}).get('max_daily_loss', 0.03),
        use_trailing_stop=config_dict.get('trailing_stop', {}).get('enabled', True),
        trailing_activation=config_dict.get('trailing_stop', {}).get('activation_percent', 1.5),
        trailing_distance=config_dict.get('trailing_stop', {}).get('trailing_distance', 1.0),
        partial_tp1_size=config_dict.get('partial_take_profit', {}).get('tp1_size', 0.3),
        partial_tp2_size=config_dict.get('partial_take_profit', {}).get('tp2_size', 0.3),
        partial_tp3_size=config_dict.get('partial_take_profit', {}).get('tp3_size', 0.4)
    )


def run_backtest(args):
    """运行回测"""
    logger.info("="*60)
    logger.info("OpenClaw 策略回测")
    logger.info("="*60)
    
    # 加载配置
    if args.config:
        config = load_config(args.config)
    else:
        config = StrategyConfig(
            name=f"MTF_Momentum_{args.symbol}",
            symbols=[args.symbol],
            min_confidence=args.min_confidence or 65.0,
            min_score=args.min_score or 6.0
        )
    
    logger.info(f"策略: {config.name}")
    logger.info(f"交易对: {config.symbols}")
    logger.info(f"回测天数: {args.days}")
    
    # 创建示例数据
    logger.info("生成回测数据...")
    data_1h, data_15m = create_sample_data(days=args.days, symbol=args.symbol)
    
    # 运行回测
    engine = BacktestEngine(config)
    results = engine.run(
        data_1h=data_1h,
        data_15m=data_15m,
        initial_capital=args.capital or 10000.0
    )
    
    # 生成报告
    report = engine.generate_report()
    print(report)
    
    # 导出结果
    if args.output:
        engine.export_results(args.output)
        logger.info(f"结果已导出到: {args.output}")
    
    return results


def run_live(args):
    """运行实盘/模拟交易"""
    logger.info("="*60)
    logger.info("OpenClaw 实盘交易")
    logger.info("="*60)
    
    # 加载配置
    if args.config:
        config = load_config(args.config)
    else:
        config = StrategyConfig()
    
    logger.info(f"策略: {config.name}")
    logger.info(f"交易对: {config.symbols}")
    logger.info(f"模式: {'模拟交易' if args.paper else '实盘交易'}")
    
    # 初始化组件
    strategy = MTFMomentumStrategy(config)
    decision = DecisionEngine(config)
    tracker = PerformanceTracker()
    
    logger.info("系统初始化完成，等待市场数据...")
    
    # 这里需要接入实际的数据源
    # 可以使用第1层输入层的数据
    
    return {
        'status': 'initialized',
        'config': config,
        'strategy': strategy,
        'decision': decision,
        'tracker': tracker
    }


def run_analysis(args):
    """运行策略分析"""
    logger.info("="*60)
    logger.info("OpenClaw 策略分析")
    logger.info("="*60)
    
    # 加载历史交易数据进行分析
    tracker = PerformanceTracker()
    
    # 这里可以从文件加载历史交易数据
    # tracker.load_trades(args.trades_file)
    
    # 生成报告
    report = tracker.generate_report()
    print(report)
    
    # 生成优化建议
    suggestions = tracker.generate_optimization_suggestions()
    print("\n【优化建议】")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='OpenClaw 多时间框架趋势动量策略交易系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 回测模式
  python main.py --mode backtest --symbol BTCUSDT --days 30 --capital 10000
  
  # 模拟交易模式
  python main.py --mode live --config config/strategy_config.yaml --paper
  
  # 分析模式
  python main.py --mode analysis --trades-file trades.csv
        """
    )
    
    parser.add_argument('--mode', '-m',
                       choices=['backtest', 'live', 'analysis'],
                       default='backtest',
                       help='运行模式 (默认: backtest)')
    
    parser.add_argument('--config', '-c',
                       type=str,
                       help='配置文件路径')
    
    parser.add_argument('--symbol', '-s',
                       type=str,
                       default='BTCUSDT',
                       help='交易对 (默认: BTCUSDT)')
    
    parser.add_argument('--days', '-d',
                       type=int,
                       default=30,
                       help='回测天数 (默认: 30)')
    
    parser.add_argument('--capital',
                       type=float,
                       default=10000.0,
                       help='初始资金 (默认: 10000)')
    
    parser.add_argument('--min-confidence',
                       type=float,
                       help='最小置信度')
    
    parser.add_argument('--min-score',
                       type=float,
                       help='最小评分')
    
    parser.add_argument('--paper',
                       action='store_true',
                       help='模拟交易模式')
    
    parser.add_argument('--output', '-o',
                       type=str,
                       help='输出文件路径')
    
    parser.add_argument('--trades-file',
                       type=str,
                       help='交易记录文件路径')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'backtest':
            run_backtest(args)
        elif args.mode == 'live':
            run_live(args)
        elif args.mode == 'analysis':
            run_analysis(args)
    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.error(f"错误: {e}", exc_info=True)


if __name__ == "__main__":
    main()
