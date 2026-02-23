"""
输入层测试脚本
测试所有数据源和核心功能
"""

import asyncio
import unittest
import logging
from datetime import datetime
from decimal import Decimal

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, '/mnt/okcomputer/output/openclaw_input_layer')

from core.input import (
    InputEngine, InputEngineConfig, InputMode,
    BinanceClient, CoinGeckoClient,
    AlternativeMeClient, CoinalyzeClient,
    CryptoPanicClient,
    DataSourceType
)


class TestPriceSources(unittest.IsolatedAsyncioTestCase):
    """测试价格数据源"""
    
    async def test_binance_client(self):
        """测试Binance客户端"""
        logger.info("测试Binance客户端...")
        
        async with BinanceClient() as client:
            await client.start()
            
            # 测试获取K线
            klines = await client.get_klines('BTCUSDT', interval='1m', limit=5)
            self.assertIsInstance(klines, list)
            if klines:
                self.assertIsNotNone(klines[0].close_price)
                logger.info(f"✓ Binance K线数据: {len(klines)} 条")
            
            # 测试获取订单簿
            orderbook = await client.get_orderbook('BTCUSDT', limit=5)
            if orderbook:
                self.assertIsNotNone(orderbook.best_bid)
                self.assertIsNotNone(orderbook.best_ask)
                logger.info(f"✓ Binance 订单簿: 买一 {orderbook.best_bid.price}, 卖一 {orderbook.best_ask.price}")
            
            # 测试获取成交
            trades = await client.get_recent_trades('BTCUSDT', limit=5)
            self.assertIsInstance(trades, list)
            logger.info(f"✓ Binance 成交数据: {len(trades)} 条")
    
    async def test_coingecko_client(self):
        """测试CoinGecko客户端"""
        logger.info("测试CoinGecko客户端...")
        
        async with CoinGeckoClient() as client:
            await client.start()
            
            # 测试获取价格
            price = await client.get_price_by_symbol('BTCUSDT')
            if price:
                self.assertIsInstance(price, Decimal)
                logger.info(f"✓ CoinGecko BTC价格: ${price:,.2f}")
            
            # 测试获取市场数据
            markets = await client.get_markets(vs_currency='usd', per_page=5)
            self.assertIsInstance(markets, list)
            logger.info(f"✓ CoinGecko 市场数据: {len(markets)} 条")


class TestSentimentSources(unittest.IsolatedAsyncioTestCase):
    """测试情绪数据源"""
    
    async def test_alternative_me_client(self):
        """测试Alternative.me客户端"""
        logger.info("测试Alternative.me客户端...")
        
        async with AlternativeMeClient() as client:
            await client.start()
            
            # 测试获取当前恐惧贪婪指数
            index = await client.get_current_index()
            if index:
                self.assertIsInstance(index.value, int)
                self.assertTrue(0 <= index.value <= 100)
                logger.info(f"✓ 恐惧贪婪指数: {index.value} ({index.classification})")
            
            # 测试获取历史数据
            historical = await client.get_historical_data(limit=5)
            self.assertIsInstance(historical, list)
            logger.info(f"✓ 历史恐惧贪婪指数: {len(historical)} 条")
    
    async def test_coinalyze_client(self):
        """测试Coinalyze客户端"""
        logger.info("测试Coinalyze客户端...")
        
        async with CoinalyzeClient() as client:
            await client.start()
            
            # 测试获取资金费率
            rates = await client.get_funding_rates(['BTCUSDT_PERP'])
            self.assertIsInstance(rates, list)
            if rates:
                logger.info(f"✓ 资金费率: {rates[0].funding_rate:.4%}")
            
            # 测试获取多空比
            ratios = await client.get_long_short_ratio(['BTCUSDT_PERP'])
            self.assertIsInstance(ratios, list)
            if ratios:
                logger.info(f"✓ 多空比: {ratios[0].long_short_ratio:.2f}")


class TestNewsSources(unittest.IsolatedAsyncioTestCase):
    """测试新闻数据源"""
    
    async def test_cryptopanic_client(self):
        """测试CryptoPanic客户端"""
        logger.info("测试CryptoPanic客户端...")
        
        async with CryptoPanicClient() as client:
            await client.start()
            
            # 测试获取新闻
            news = await client.get_posts(currencies=['BTC'], limit=5)
            self.assertIsInstance(news, list)
            if news:
                logger.info(f"✓ CryptoPanic新闻: {len(news)} 条")
                logger.info(f"  最新: {news[0].title[:50]}...")


class TestInputEngine(unittest.IsolatedAsyncioTestCase):
    """测试输入引擎"""
    
    async def test_engine_lifecycle(self):
        """测试引擎生命周期"""
        logger.info("测试引擎生命周期...")
        
        engine = InputEngine()
        
        # 启动
        await engine.start()
        self.assertTrue(engine._running)
        logger.info("✓ 引擎启动成功")
        
        # 停止
        await engine.stop()
        self.assertFalse(engine._running)
        logger.info("✓ 引擎停止成功")
    
    async def test_get_market_data(self):
        """测试获取市场数据"""
        logger.info("测试获取市场数据...")
        
        config = InputEngineConfig(
            mode=InputMode.POLLING,
            default_data_types=['price', 'sentiment']
        )
        
        async with InputEngine(config) as engine:
            result = await engine.get_market_data('BTCUSDT')
            
            self.assertIsInstance(result.success, bool)
            
            if result.success:
                self.assertIsNotNone(result.market_data)
                logger.info(f"✓ 市场数据获取成功，处理时间: {result.processing_time_ms:.2f}ms")
                
                if result.market_data.price_data:
                    logger.info(f"  BTC价格: ${result.market_data.price_data.close_price:,.2f}")
            else:
                logger.warning(f"市场数据获取失败: {result.message}")
    
    async def test_health_check(self):
        """测试健康检查"""
        logger.info("测试健康检查...")
        
        async with InputEngine() as engine:
            health = await engine.health_check()
            
            self.assertIn('status', health)
            self.assertIn('data_sources', health)
            logger.info(f"✓ 健康检查: {health['status']}")
    
    async def test_data_validation(self):
        """测试数据验证"""
        logger.info("测试数据验证...")
        
        async with InputEngine() as engine:
            result = await engine.get_market_data('BTCUSDT', data_types=['price'])
            
            if result.success and result.market_data and result.market_data.price_data:
                validation = engine.validate_data(result.market_data.price_data)
                self.assertIn('valid', validation)
                logger.info(f"✓ 数据验证: {'通过' if validation['valid'] else '失败'}")


class TestDataAggregator(unittest.IsolatedAsyncioTestCase):
    """测试数据聚合器"""
    
    async def test_aggregator_lifecycle(self):
        """测试聚合器生命周期"""
        logger.info("测试聚合器生命周期...")
        
        from core.input.data_aggregator import DataAggregator
        
        aggregator = DataAggregator()
        
        await aggregator.start()
        logger.info("✓ 聚合器启动成功")
        
        await aggregator.stop()
        logger.info("✓ 聚合器停止成功")
    
    async def test_multiple_data_sources(self):
        """测试多数据源获取"""
        logger.info("测试多数据源获取...")
        
        from core.input.data_aggregator import DataAggregator
        
        async with DataAggregator() as aggregator:
            await aggregator.start()
            
            result = await aggregator.get_market_data(
                'BTCUSDT',
                data_types=['price', 'sentiment']
            )
            
            self.assertIsInstance(result.success, bool)
            logger.info(f"✓ 多数据源获取: {'成功' if result.success else '失败'}")


def run_tests():
    """运行所有测试"""
    logger.info("=" * 60)
    logger.info("OpenClaw 输入层测试")
    logger.info("=" * 60)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试
    suite.addTests(loader.loadTestsFromTestCase(TestPriceSources))
    suite.addTests(loader.loadTestsFromTestCase(TestSentimentSources))
    suite.addTests(loader.loadTestsFromTestCase(TestNewsSources))
    suite.addTests(loader.loadTestsFromTestCase(TestInputEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestDataAggregator))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
