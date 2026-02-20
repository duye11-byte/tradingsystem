"""
推理层测试脚本
演示如何使用推理层进行加密货币交易决策
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.reasoning import ReasoningEngine, TradingSignal, SignalType
import yaml
import time


def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(__file__), '../config/reasoning_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_mock_features(symbol: str = "BTC/USDT") -> dict:
    """创建模拟特征数据"""
    return {
        # 价格数据
        'open': 45000.0,
        'high': 46500.0,
        'low': 44500.0,
        'close': 46000.0,
        'volume': 1500000000,
        
        # 技术指标
        'rsi_14': 62.5,
        'macd': 150.5,
        'macd_signal': 120.3,
        'macd_histogram': 30.2,
        'bb_upper': 48000.0,
        'bb_lower': 44000.0,
        'bb_middle': 46000.0,
        'bb_width': 0.087,
        'sma_20': 45500.0,
        'sma_50': 44500.0,
        'ema_12': 45800.0,
        'ema_26': 45200.0,
        'atr_14': 1200.0,
        'volume_ratio': 1.25,
        
        # 价格变化
        'price_change_1d': 2.5,
        'price_change_7d': 8.3,
        'price_change_30d': 15.2,
        
        # 链上数据
        'exchange_inflow': 50000000,
        'exchange_outflow': 80000000,
        'whale_tx_count': 15,
        'whale_volume': 250000000,
        'active_addresses_change': 5.2,
        'transaction_count': 350000,
        
        # 情绪数据
        'fear_greed_index': 65,
        'social_sentiment': 0.35,
        'news_sentiment': 0.28,
        'funding_rate': 0.008,
        'long_short_ratio': 1.35,
        'open_interest': 12000000000,
        
        # 宏观数据
        'btc_correlation': 0.85,
        'eth_correlation': 0.78,
        'market_breadth': 0.62,
        'liquidity_index': 0.75,
        'dxy_change': -0.3,
        'sp500_change': 0.8
    }


def create_mock_market_data(symbol: str = "BTC/USDT") -> dict:
    """创建模拟市场数据"""
    return {
        'symbol': symbol,
        'timestamp': int(time.time()),
        'ohlcv': {
            'open': 45000.0,
            'high': 46500.0,
            'low': 44500.0,
            'close': 46000.0,
            'volume': 1500000000
        },
        'orderbook': {
            'bids': [[45900, 1.5], [45800, 2.3], [45700, 3.1]],
            'asks': [[46100, 1.2], [46200, 2.1], [46300, 2.8]]
        },
        'trades': []
    }


async def test_basic_reasoning():
    """测试基本推理功能"""
    print("=" * 60)
    print("测试 1: 基本推理功能")
    print("=" * 60)
    
    # 加载配置
    config = load_config()
    
    # 创建推理引擎
    engine = ReasoningEngine(config.get('reasoning_engine', {}))
    
    # 准备数据
    symbol = "BTC/USDT"
    features = create_mock_features(symbol)
    market_data = create_mock_market_data(symbol)
    
    # 执行推理
    print(f"\n开始推理分析: {symbol}")
    print("-" * 40)
    
    result = await engine.reason(symbol, market_data, features)
    
    if result.success:
        signal = result.signal
        print(f"\n✅ 推理成功!")
        print(f"\n交易信号:")
        print(f"  信号类型: {signal.signal.value.upper()}")
        print(f"  置信度: {signal.confidence:.1%}")
        print(f"  入场价格: ${signal.entry_price:,.2f}")
        print(f"  止损价格: ${signal.stop_loss:,.2f}" if signal.stop_loss else "  止损价格: N/A")
        print(f"  止盈价格: ${signal.take_profit:,.2f}" if signal.take_profit else "  止盈价格: N/A")
        print(f"  建议仓位: {signal.position_size_ratio:.0%}")
        
        print(f"\n推理链 ({len(signal.reasoning_chain)} 步):")
        for step in signal.reasoning_chain:
            print(f"  步骤 {step.step_number}: {step.title}")
            print(f"    结论: {step.intermediate_conclusion}")
            print(f"    置信度: {step.confidence:.0%}")
        
        print(f"\n共识结果:")
        consensus = signal.consensus_result
        print(f"  参与模型: {', '.join(consensus.participating_models)}")
        print(f"  一致率: {consensus.agreement_ratio:.1%}")
        print(f"  分歧分析: {consensus.dissensus_analysis}")
        
        print(f"\n一致性验证:")
        print(f"  通过验证: {'✅ 是' if signal.consistency_check_passed else '❌ 否'}")
        print(f"  一致性分数: {signal.consistency_score:.1%}")
        
        print(f"\n性能指标:")
        print(f"  总延迟: {result.total_latency_ms:.0f}ms")
        print(f"  CoT延迟: {result.cot_latency_ms:.0f}ms")
        print(f"  集成延迟: {result.ensemble_latency_ms:.0f}ms")
        print(f"  验证延迟: {result.consistency_latency_ms:.0f}ms")
    else:
        print(f"\n❌ 推理失败: {result.error_message}")
    
    return result


async def test_streaming_reasoning():
    """测试流式推理"""
    print("\n" + "=" * 60)
    print("测试 2: 流式推理")
    print("=" * 60)
    
    config = load_config()
    engine = ReasoningEngine(config.get('reasoning_engine', {}))
    
    symbol = "ETH/USDT"
    features = create_mock_features(symbol)
    market_data = create_mock_market_data(symbol)
    
    print(f"\n开始流式推理: {symbol}")
    print("-" * 40)
    
    async for update in engine.reason_stream(symbol, market_data, features):
        print(f"[{update['progress']:3d}%] {update['stage']}: {update['message']}")
        
        if update['stage'] == 'completed' and 'signal' in update:
            signal = update['signal']
            print(f"\n✅ 流式推理完成!")
            print(f"最终信号: {signal.signal.value.upper()} (置信度: {signal.confidence:.1%})")


async def test_batch_reasoning():
    """测试批量推理"""
    print("\n" + "=" * 60)
    print("测试 3: 批量推理")
    print("=" * 60)
    
    config = load_config()
    engine = ReasoningEngine(config.get('reasoning_engine', {}))
    
    # 准备多个任务
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "DOT/USDT"]
    tasks = []
    
    for symbol in symbols:
        features = create_mock_features(symbol)
        # 为每个币种稍微调整特征
        features['close'] *= (1 + hash(symbol) % 100 / 1000)
        features['rsi_14'] = 40 + hash(symbol) % 40
        
        tasks.append({
            'symbol': symbol,
            'market_data': create_mock_market_data(symbol),
            'features': features,
            'skip_validation': True  # 批量推理跳过验证以提高速度
        })
    
    print(f"\n开始批量推理: {len(tasks)} 个币种")
    print("-" * 40)
    
    start_time = time.time()
    results = await engine.batch_reason(tasks, max_concurrency=3)
    elapsed = time.time() - start_time
    
    print(f"\n批量推理完成，耗时: {elapsed:.2f}s")
    print("\n结果汇总:")
    print("-" * 40)
    
    for i, result in enumerate(results):
        symbol = tasks[i]['symbol']
        if result.success:
            signal = result.signal
            status = "✅"
            print(f"{status} {symbol}: {signal.signal.value.upper():12s} "
                  f"(置信度: {signal.confidence:.1%}, 延迟: {result.total_latency_ms:.0f}ms)")
        else:
            print(f"❌ {symbol}: 失败 - {result.error_message}")


async def test_consistency_validation():
    """测试一致性验证"""
    print("\n" + "=" * 60)
    print("测试 4: 一致性验证")
    print("=" * 60)
    
    config = load_config()
    engine = ReasoningEngine(config.get('reasoning_engine', {}))
    
    symbol = "BTC/USDT"
    features = create_mock_features(symbol)
    market_data = create_mock_market_data(symbol)
    
    print(f"\n执行多次推理以测试一致性...")
    print("-" * 40)
    
    # 执行多次推理
    results = []
    for i in range(5):
        result = await engine.reason(symbol, market_data, features)
        results.append(result)
        if result.success:
            print(f"  运行 {i+1}: {result.signal.signal.value.upper()} "
                  f"(置信度: {result.signal.confidence:.1%})")
    
    # 获取验证统计
    stats = engine.consistency_validator.get_validation_statistics()
    trend = engine.consistency_validator.get_consistency_trend()
    
    print(f"\n一致性统计:")
    print(f"  总验证次数: {stats['total_validations']}")
    print(f"  通过次数: {stats['passed']}")
    print(f"  失败次数: {stats['failed']}")
    print(f"  通过率: {stats['pass_rate']:.1%}")
    print(f"  平均一致性: {stats['avg_consistency']:.1%}")
    
    print(f"\n一致性趋势:")
    print(f"  趋势: {trend['trend']}")
    print(f"  平均一致性: {trend['avg_consistency']:.1%}")
    print(f"  趋势方向: {trend['trend_direction']}")


async def test_performance_monitoring():
    """测试性能监控"""
    print("\n" + "=" * 60)
    print("测试 5: 性能监控")
    print("=" * 60)
    
    config = load_config()
    engine = ReasoningEngine(config.get('reasoning_engine', {}))
    
    symbol = "BTC/USDT"
    features = create_mock_features(symbol)
    market_data = create_mock_market_data(symbol)
    
    print(f"\n执行多次推理以收集性能数据...")
    
    # 执行多次推理
    for i in range(10):
        await engine.reason(symbol, market_data, features)
    
    # 获取性能统计
    stats = engine.get_performance_stats()
    
    print(f"\n性能统计:")
    print(f"  总请求数: {stats['total_requests']}")
    print(f"  成功请求: {stats['successful_requests']}")
    print(f"  失败请求: {stats['failed_requests']}")
    print(f"  成功率: {stats['success_rate']:.1%}")
    print(f"  平均延迟: {stats['avg_latency_ms']:.0f}ms")


async def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("推理层测试套件")
    print("OpenClaw Crypto Trading System - Reasoning Layer")
    print("=" * 60)
    
    try:
        # 运行所有测试
        await test_basic_reasoning()
        await test_streaming_reasoning()
        await test_batch_reasoning()
        await test_consistency_validation()
        await test_performance_monitoring()
        
        print("\n" + "=" * 60)
        print("所有测试完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
