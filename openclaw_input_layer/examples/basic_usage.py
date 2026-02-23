"""
OpenClaw 输入层基础使用示例
演示如何获取市场数据、实时流和情绪指标
"""

import asyncio
import logging
from decimal import Decimal

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入输入层
import sys
sys.path.insert(0, '/mnt/okcomputer/output/openclaw_input_layer')

from core.input import InputEngine, InputEngineConfig, InputMode


async def example_basic_market_data():
    """示例1：获取基础市场数据"""
    logger.info("=" * 50)
    logger.info("示例1：获取基础市场数据")
    logger.info("=" * 50)
    
    config = InputEngineConfig(
        mode=InputMode.POLLING,
        default_symbols=['BTCUSDT'],
        default_data_types=['price', 'sentiment']
    )
    
    async with InputEngine(config) as engine:
        # 获取BTC市场数据
        result = await engine.get_market_data('BTCUSDT')
        
        if result.success:
            market_data = result.market_data
            
            logger.info(f"数据获取成功，处理时间: {result.processing_time_ms:.2f}ms")
            
            if market_data.price_data:
                price = market_data.price_data
                logger.info(f"BTC价格: ${price.close_price:,.2f}")
                logger.info(f"24h涨跌: {price.price_change_pct:.2f}%")
            
            if market_data.fear_greed:
                fg = market_data.fear_greed
                logger.info(f"恐惧贪婪指数: {fg.value} ({fg.classification})")
            
            if market_data.funding_rate:
                fr = market_data.funding_rate
                logger.info(f"资金费率: {fr.funding_rate:.4%}")
        else:
            logger.error(f"获取数据失败: {result.message}")
            if result.errors:
                for error in result.errors:
                    logger.error(f"  - {error}")


async def example_multiple_symbols():
    """示例2：批量获取多个交易对"""
    logger.info("\n" + "=" * 50)
    logger.info("示例2：批量获取多个交易对")
    logger.info("=" * 50)
    
    async with InputEngine() as engine:
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        
        results = await engine.get_multiple_market_data(symbols)
        
        for symbol, result in results.items():
            if result.success and result.market_data and result.market_data.price_data:
                price = result.market_data.price_data.close_price
                logger.info(f"{symbol}: ${float(price):,.2f}")
            else:
                logger.warning(f"{symbol}: 获取失败")


async def example_streaming_data():
    """示例3：实时数据流"""
    logger.info("\n" + "=" * 50)
    logger.info("示例3：实时数据流（获取5个数据点）")
    logger.info("=" * 50)
    
    config = InputEngineConfig(mode=InputMode.HYBRID)
    
    async with InputEngine(config) as engine:
        count = 0
        async for price_data in engine.stream_prices('BTCUSDT', interval='1m'):
            logger.info(f"[{count+1}] BTC: ${float(price_data.close_price):,.2f} "
                       f"Vol: {float(price_data.volume):.4f}")
            
            count += 1
            if count >= 5:
                break


async def example_sentiment_analysis():
    """示例4：情绪分析"""
    logger.info("\n" + "=" * 50)
    logger.info("示例4：综合情绪分析")
    logger.info("=" * 50)
    
    async with InputEngine() as engine:
        sentiment = await engine.get_composite_sentiment('BTCUSDT')
        
        logger.info(f"综合情绪分数: {sentiment.get('composite_score', 'N/A'):.1f}/100")
        logger.info(f"解读: {sentiment.get('interpretation', 'N/A')}")
        
        components = sentiment.get('components', {})
        
        if 'fear_greed' in components:
            fg = components['fear_greed']
            logger.info(f"恐惧贪婪指数: {fg['value']} ({fg['classification']})")
        
        if 'funding_rate' in components:
            fr = components['funding_rate']
            logger.info(f"资金费率: {fr['rate']:.4%} ({fr['interpretation']})")
        
        if 'long_short_ratio' in components:
            ls = components['long_short_ratio']
            logger.info(f"多空比: {ls['ratio']:.2f} ({ls['interpretation']})")


async def example_data_validation():
    """示例5：数据验证和异常检测"""
    logger.info("\n" + "=" * 50)
    logger.info("示例5：数据验证和异常检测")
    logger.info("=" * 50)
    
    async with InputEngine() as engine:
        # 获取价格数据
        result = await engine.get_market_data('BTCUSDT', data_types=['price'])
        
        if result.success and result.market_data and result.market_data.price_data:
            price_data = result.market_data.price_data
            
            # 验证数据
            validation = engine.validate_data(price_data)
            logger.info(f"数据验证结果: {'通过' if validation['valid'] else '失败'}")
            
            if validation['warnings']:
                logger.warning(f"警告: {validation['warnings']}")
            
            if validation['critical_errors']:
                logger.error(f"严重错误: {validation['critical_errors']}")
            
            # 异常检测
            anomaly = engine.detect_anomaly(price_data, method='zscore')
            logger.info(f"异常检测结果: {'异常' if anomaly['is_anomaly'] else '正常'} "
                       f"(Z-score: {anomaly['score']:.2f})")


async def example_health_check():
    """示例6：健康检查"""
    logger.info("\n" + "=" * 50)
    logger.info("示例6：健康检查")
    logger.info("=" * 50)
    
    async with InputEngine() as engine:
        health = await engine.health_check()
        
        logger.info(f"整体状态: {health['status']}")
        logger.info(f"引擎运行: {health['engine_running']}")
        
        if health['issues']:
            logger.warning("发现问题:")
            for issue in health['issues']:
                logger.warning(f"  - {issue}")
        
        logger.info("数据源状态:")
        for source, status in health['data_sources'].items():
            logger.info(f"  {source}: {'活跃' if status['active'] else '不活跃'} "
                       f"(成功率: {status['success_rate']:.1%})")


async def example_stats():
    """示例7：统计信息"""
    logger.info("\n" + "=" * 50)
    logger.info("示例7：统计信息")
    logger.info("=" * 50)
    
    async with InputEngine() as engine:
        # 执行一些请求
        for _ in range(3):
            await engine.get_market_data('BTCUSDT')
        
        # 获取统计
        stats = engine.get_stats()
        
        logger.info(f"总请求数: {stats['total_requests']}")
        logger.info(f"成功请求: {stats['successful_requests']}")
        logger.info(f"失败请求: {stats['failed_requests']}")
        logger.info(f"成功率: {stats['success_rate']:.1%}")
        logger.info(f"缓存命中率: {stats['cache_hit_rate']:.1%}")


async def main():
    """主函数"""
    logger.info("OpenClaw 输入层使用示例")
    logger.info("=" * 50)
    
    try:
        # 运行所有示例
        await example_basic_market_data()
        await example_multiple_symbols()
        # await example_streaming_data()  # 取消注释以测试实时流
        await example_sentiment_analysis()
        await example_data_validation()
        await example_health_check()
        await example_stats()
        
    except Exception as e:
        logger.error(f"示例运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
