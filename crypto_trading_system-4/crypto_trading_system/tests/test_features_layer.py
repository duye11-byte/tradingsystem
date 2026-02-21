"""
特征工程层测试脚本
测试所有特征提取功能
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from core.features import (
    FeatureEngineering,
    TechnicalIndicators,
    OnchainMetrics,
    SentimentAnalyzer,
    FeatureComposer,
    FeatureConfig
)


def create_mock_ohlcv_data(n: int = 100) -> pd.DataFrame:
    """创建模拟 OHLCV 数据"""
    np.random.seed(42)
    
    # 生成随机价格数据
    base_price = 45000
    returns = np.random.normal(0.001, 0.02, n)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # 生成 OHLC
    data = {
        'timestamp': pd.date_range(end=datetime.now(), periods=n, freq='1H'),
        'open': prices * (1 + np.random.normal(0, 0.005, n)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n))),
        'close': prices,
        'volume': np.random.lognormal(20, 1, n)
    }
    
    df = pd.DataFrame(data)
    
    # 确保 high >= open, close, low
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    return df


def create_mock_onchain_data() -> dict:
    """创建模拟链上数据"""
    return {
        'exchange_inflow': 50000000,
        'exchange_outflow': 80000000,
        'exchange_inflow_history': [45000000, 48000000, 50000000],
        'exchange_outflow_history': [70000000, 75000000, 80000000],
        
        'whale_transactions': [
            {'amount': 1500, 'direction': 'in'},
            {'amount': 2300, 'direction': 'out'},
            {'amount': 1800, 'direction': 'in'},
        ],
        
        'active_addresses': 850000,
        'active_addresses_history': [820000, 835000, 850000],
        'transaction_count': 350000,
        'transaction_count_history': [340000, 345000, 350000],
        'total_transfer_volume': 25000000000,
        
        'supply_on_exchanges': 2500000,
        'total_supply': 19000000,
        'long_term_holder_supply': 12000000,
        'short_term_holder_supply': 4500000,
        
        'miner_revenue': 15000000,
        'miner_outflow': 5000000,
        
        'hash_rate': 350000000,
        'hash_rate_history': [340000000, 345000000, 350000000],
        'difficulty': 55000000000000
    }


def create_mock_sentiment_data() -> dict:
    """创建模拟情绪数据"""
    return {
        'fear_greed_index': 65,
        'fear_greed_history': [55, 60, 65],
        
        'social_sentiment': 0.35,
        'social_sentiment_history': [0.25, 0.30, 0.35],
        'social_volume': 150000,
        'twitter_sentiment': 0.40,
        'reddit_sentiment': 0.30,
        
        'news_sentiment': 0.28,
        'news_sentiment_history': [0.20, 0.25, 0.28],
        'news_volume': 500,
        
        'funding_rate': 0.008,
        'funding_rate_history': [0.005, 0.006, 0.008],
        'long_short_ratio': 1.35,
        'open_interest': 12000000000,
        'open_interest_history': [11500000000, 11800000000, 12000000000],
        
        'put_call_ratio': 0.85,
        'iv_skew': -0.05
    }


async def test_technical_indicators():
    """测试技术指标"""
    print("=" * 60)
    print("测试 1: 技术指标")
    print("=" * 60)
    
    # 创建数据
    df = create_mock_ohlcv_data(100)
    
    # 计算指标
    calculator = TechnicalIndicators()
    features = calculator.calculate_all(df)
    
    print(f"\n✅ 技术指标计算完成")
    print(f"\n趋势指标:")
    print(f"  SMA20: {features.sma_20:,.2f}")
    print(f"  SMA50: {features.sma_50:,.2f}")
    print(f"  EMA12: {features.ema_12:,.2f}")
    print(f"  EMA26: {features.ema_26:,.2f}")
    
    print(f"\n动量指标:")
    print(f"  RSI14: {features.rsi_14:.2f}")
    print(f"  MACD: {features.macd:.2f}")
    print(f"  MACD Signal: {features.macd_signal:.2f}")
    print(f"  Stochastic K: {features.stochastic_k:.2f}")
    
    print(f"\n波动率指标:")
    print(f"  BB Upper: {features.bb_upper:,.2f}")
    print(f"  BB Lower: {features.bb_lower:,.2f}")
    print(f"  BB Width: {features.bb_width:.4f}")
    print(f"  ATR14: {features.atr_14:.2f}")
    
    print(f"\n成交量指标:")
    print(f"  OBV: {features.obv:,.0f}")
    print(f"  Volume Ratio: {features.volume_ratio:.2f}")
    print(f"  MFI14: {features.mfi_14:.2f}")
    
    print(f"\n价格变化:")
    print(f"  1H Change: {features.price_change_1h:.2f}%")
    print(f"  1D Change: {features.price_change_1d:.2f}%")
    print(f"  7D Change: {features.price_change_7d:.2f}%")


async def test_onchain_metrics():
    """测试链上指标"""
    print("\n" + "=" * 60)
    print("测试 2: 链上指标")
    print("=" * 60)
    
    data = create_mock_onchain_data()
    
    calculator = OnchainMetrics()
    features = calculator.calculate_all(data)
    
    print(f"\n✅ 链上指标计算完成")
    print(f"\n交易所流向:")
    print(f"  Inflow: {features.exchange_inflow:,.0f}")
    print(f"  Outflow: {features.exchange_outflow:,.0f}")
    print(f"  Netflow: {features.exchange_netflow:,.0f}")
    print(f"  Inflow Change: {features.exchange_inflow_change:.2f}%")
    
    print(f"\n鲸鱼活动:")
    print(f"  TX Count: {features.whale_tx_count}")
    print(f"  Volume: {features.whale_volume:,.0f}")
    print(f"  Accumulation: {features.whale_accumulation:.2%}")
    
    print(f"\n网络活跃度:")
    print(f"  Active Addresses: {features.active_addresses:,}")
    print(f"  Active Change: {features.active_addresses_change:.2f}%")
    print(f"  TX Count: {features.transaction_count:,}")
    
    print(f"\n供应分布:")
    print(f"  On Exchanges: {features.supply_on_exchanges_pct:.2f}%")
    print(f"  LTH Supply: {features.long_term_holder_supply:,.0f}")
    print(f"  STH Supply: {features.short_term_holder_supply:,.0f}")


async def test_sentiment_analyzer():
    """测试情绪分析"""
    print("\n" + "=" * 60)
    print("测试 3: 情绪指标")
    print("=" * 60)
    
    data = create_mock_sentiment_data()
    
    analyzer = SentimentAnalyzer()
    features = analyzer.calculate_all(data)
    
    print(f"\n✅ 情绪指标计算完成")
    print(f"\n恐惧贪婪指数:")
    print(f"  Index: {features.fear_greed_index}/100")
    print(f"  Classification: {features.fear_greed_classification}")
    print(f"  Change: {features.fear_greed_change:+.1f}")
    print(f"  Extreme Greed: {features.extreme_greed}")
    print(f"  Extreme Fear: {features.extreme_fear}")
    
    print(f"\n社交媒体情绪:")
    print(f"  Social Sentiment: {features.social_sentiment:+.2f}")
    print(f"  Twitter: {features.twitter_sentiment:+.2f}")
    print(f"  Reddit: {features.reddit_sentiment:+.2f}")
    
    print(f"\n期货指标:")
    print(f"  Funding Rate: {features.funding_rate:.4f}")
    print(f"  Long/Short Ratio: {features.long_short_ratio:.2f}")
    print(f"  Open Interest: {features.open_interest:,.0f}")
    
    print(f"\n综合情绪:")
    print(f"  Composite: {features.composite_sentiment:+.2f}")
    print(f"  Momentum: {features.sentiment_momentum:+.2f}")
    
    # 生成情绪信号
    signal = analyzer.generate_sentiment_signal(features)
    print(f"\n情绪信号:")
    print(f"  Direction: {signal['signal_direction']}")
    print(f"  Strength: {signal['signal_strength']}")
    print(f"  Reasons: {signal['reasons']}")


async def test_feature_composer():
    """测试特征组合"""
    print("\n" + "=" * 60)
    print("测试 4: 组合特征")
    print("=" * 60)
    
    # 创建模拟特征集历史
    from core.features.feature_types import FeatureSet, TechnicalFeatures
    
    historical = []
    for i in range(50):
        fs = FeatureSet(
            symbol="BTC/USDT",
            timestamp=datetime.now() - timedelta(hours=i)
        )
        fs.close = 45000 + np.random.normal(0, 500)
        fs.technical = TechnicalFeatures()
        fs.technical.rsi_14 = 50 + np.random.normal(0, 10)
        fs.technical.bb_width = 0.05 + np.random.normal(0, 0.01)
        historical.append(fs)
    
    historical.reverse()
    
    # 当前特征集
    current = historical[-1]
    
    # 计算组合特征
    composer = FeatureComposer(n_components=3)
    composer.fit_pca(historical)
    
    composite = composer.calculate_all(current, historical)
    
    print(f"\n✅ 组合特征计算完成")
    print(f"\nPCA 主成分:")
    print(f"  PC1: {composite.pc1:.4f}")
    print(f"  PC2: {composite.pc2:.4f}")
    print(f"  PC3: {composite.pc3:.4f}")
    
    print(f"\n时间序列分解:")
    print(f"  Trend: {composite.trend_component:.2f}")
    print(f"  Seasonal: {composite.seasonal_component:.2f}")
    print(f"  Residual: {composite.residual_component:.2f}")
    
    print(f"\n特征交互:")
    print(f"  Price-Volume: {composite.price_volume_interaction:.4f}")
    print(f"  Momentum-Sentiment: {composite.momentum_sentiment_interaction:.4f}")
    
    print(f"\n综合指标:")
    print(f"  Composite Momentum: {composite.composite_momentum:.4f}")
    print(f"  Composite Volatility: {composite.composite_volatility:.4f}")
    print(f"  Composite Liquidity: {composite.composite_liquidity:.4f}")


async def test_feature_engineering():
    """测试完整特征工程"""
    print("\n" + "=" * 60)
    print("测试 5: 完整特征工程")
    print("=" * 60)
    
    # 创建数据
    ohlcv = create_mock_ohlcv_data(100)
    onchain = create_mock_onchain_data()
    sentiment = create_mock_sentiment_data()
    
    # 创建特征工程
    config = FeatureConfig()
    fe = FeatureEngineering(config)
    
    # 提取特征
    result = await fe.extract_features(
        symbol="BTC/USDT",
        ohlcv_data=ohlcv,
        onchain_data=onchain,
        sentiment_data=sentiment
    )
    
    if result.success:
        print(f"\n✅ 特征提取成功!")
        print(f"  提取时间: {result.extraction_time_ms:.1f}ms")
        print(f"  特征数量: {result.features_extracted}")
        
        fs = result.feature_set
        
        print(f"\n价格数据:")
        print(f"  Open: {fs.open:,.2f}")
        print(f"  High: {fs.high:,.2f}")
        print(f"  Low: {fs.low:,.2f}")
        print(f"  Close: {fs.close:,.2f}")
        print(f"  Volume: {fs.volume:,.0f}")
        
        print(f"\n技术特征数: {len(fs.technical.to_dict())}")
        print(f"链上特征数: {len(fs.onchain.to_dict())}")
        print(f"情绪特征数: {len(fs.sentiment.to_dict())}")
        print(f"组合特征数: {len(fs.composite.to_dict())}")
        
        # 转换为字典
        feature_dict = fs.to_dict()
        print(f"\n总特征数: {len(feature_dict)}")
        
        # 性能统计
        stats = fe.get_stats()
        print(f"\n性能统计:")
        print(f"  成功率: {stats['success_rate']:.1%}")
        print(f"  平均提取时间: {stats['avg_extraction_time_ms']:.1f}ms")
    else:
        print(f"\n❌ 特征提取失败: {result.error_message}")


async def test_batch_extraction():
    """测试批量特征提取"""
    print("\n" + "=" * 60)
    print("测试 6: 批量特征提取")
    print("=" * 60)
    
    fe = FeatureEngineering()
    
    # 创建批量任务
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "DOT/USDT"]
    tasks = []
    
    for symbol in symbols:
        tasks.append({
            'symbol': symbol,
            'ohlcv_data': create_mock_ohlcv_data(100),
            'onchain_data': create_mock_onchain_data(),
            'sentiment_data': create_mock_sentiment_data()
        })
    
    print(f"\n开始批量提取 {len(tasks)} 个币种...")
    
    results = await fe.extract_batch(tasks, max_concurrency=3)
    
    print(f"\n✅ 批量提取完成")
    print(f"\n结果汇总:")
    for i, result in enumerate(results):
        symbol = tasks[i]['symbol']
        if result.success:
            print(f"  {symbol}: ✅ {result.features_extracted} features ({result.extraction_time_ms:.1f}ms)")
        else:
            print(f"  {symbol}: ❌ {result.error_message}")


async def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("特征工程层测试套件")
    print("=" * 60)
    
    try:
        await test_technical_indicators()
        await test_onchain_metrics()
        await test_sentiment_analyzer()
        await test_feature_composer()
        await test_feature_engineering()
        await test_batch_extraction()
        
        print("\n" + "=" * 60)
        print("所有测试完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
