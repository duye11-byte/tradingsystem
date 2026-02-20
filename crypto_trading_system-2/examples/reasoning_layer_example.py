"""
推理层使用示例
展示如何在实际交易中集成推理层
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.reasoning import ReasoningEngine, SignalType
import yaml


class TradingStrategy:
    """
    交易策略类 - 集成推理层
    
    演示如何在实际交易策略中使用推理层生成交易信号
    """
    
    def __init__(self, config_path: str = None):
        """初始化策略"""
        # 加载配置
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {}
        
        # 创建推理引擎
        self.reasoning_engine = ReasoningEngine(config.get('reasoning_engine', {}))
        
        # 策略状态
        self.active_signals = {}
        self.trade_history = []
        
    async def analyze_opportunity(
        self,
        symbol: str,
        market_data: dict,
        features: dict
    ) -> dict:
        """
        分析交易机会
        
        Args:
            symbol: 交易对符号
            market_data: 市场数据
            features: 特征数据
            
        Returns:
            dict: 分析结果和交易建议
        """
        # 执行推理
        result = await self.reasoning_engine.reason(
            symbol=symbol,
            market_data=market_data,
            features=features
        )
        
        if not result.success:
            return {
                'success': False,
                'error': result.error_message,
                'recommendation': 'HOLD'
            }
        
        signal = result.signal
        
        # 根据信号生成交易建议
        recommendation = self._generate_recommendation(signal)
        
        # 记录信号
        self.active_signals[symbol] = {
            'signal': signal,
            'timestamp': signal.generated_at,
            'result': result
        }
        
        return {
            'success': True,
            'symbol': symbol,
            'signal': signal.signal.value,
            'confidence': signal.consistency_score,
            'entry_price': signal.entry_price,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'position_size': signal.position_size_ratio,
            'recommendation': recommendation,
            'reasoning_summary': self._summarize_reasoning(signal),
            'latency_ms': result.total_latency_ms
        }
    
    def _generate_recommendation(self, signal) -> str:
        """生成交易建议"""
        if not signal.consistency_check_passed:
            return 'HOLD'
        
        if signal.consistency_score < 0.6:
            return 'HOLD'
        
        if signal.signal == SignalType.STRONG_BUY:
            return 'OPEN_LONG'
        elif signal.signal == SignalType.BUY:
            return 'OPEN_LONG_SMALL'
        elif signal.signal == SignalType.STRONG_SELL:
            return 'OPEN_SHORT'
        elif signal.signal == SignalType.SELL:
            return 'OPEN_SHORT_SMALL'
        else:
            return 'HOLD'
    
    def _summarize_reasoning(self, signal) -> str:
        """总结推理过程"""
        steps = signal.reasoning_chain
        if not steps:
            return "无推理数据"
        
        summary = []
        for step in steps:
            summary.append(f"{step.step_number}. {step.title}: {step.intermediate_conclusion}")
        
        return "\n".join(summary)
    
    async def monitor_positions(self, symbols: list):
        """监控持仓状态"""
        for symbol in symbols:
            if symbol in self.active_signals:
                signal_info = self.active_signals[symbol]
                signal = signal_info['signal']
                
                # 检查信号是否过期
                if signal.valid_until and signal.valid_until < datetime.now():
                    print(f"[{symbol}] 信号已过期，需要重新分析")
                    del self.active_signals[symbol]
    
    def get_performance_report(self) -> dict:
        """获取性能报告"""
        stats = self.reasoning_engine.get_performance_stats()
        
        return {
            'engine_stats': stats,
            'active_signals_count': len(self.active_signals),
            'total_signals_generated': len(self.trade_history)
        }


async def example_basic_usage():
    """基本使用示例"""
    print("=" * 60)
    print("示例 1: 基本使用")
    print("=" * 60)
    
    # 创建策略实例
    strategy = TradingStrategy()
    
    # 模拟市场数据
    symbol = "BTC/USDT"
    market_data = {
        'symbol': symbol,
        'ohlcv': {
            'open': 45000.0,
            'high': 46500.0,
            'low': 44500.0,
            'close': 46000.0,
            'volume': 1500000000
        }
    }
    
    # 模拟特征数据
    features = {
        'close': 46000.0,
        'rsi_14': 62.5,
        'macd': 150.5,
        'macd_signal': 120.3,
        'sma_20': 45500.0,
        'sma_50': 44500.0,
        'atr_14': 1200.0,
        'price_change_7d': 8.3,
        'fear_greed_index': 65,
        'exchange_outflow': 80000000,
        'exchange_inflow': 50000000,
        'social_sentiment': 0.35
    }
    
    # 分析交易机会
    result = await strategy.analyze_opportunity(symbol, market_data, features)
    
    if result['success']:
        print(f"\n✅ 分析成功!")
        print(f"交易对: {result['symbol']}")
        print(f"信号: {result['signal'].upper()}")
        print(f"置信度: {result['confidence']:.1%}")
        print(f"建议: {result['recommendation']}")
        print(f"\n推理总结:\n{result['reasoning_summary']}")
    else:
        print(f"❌ 分析失败: {result['error']}")
    
    return result


async def example_multi_symbol_analysis():
    """多币种分析示例"""
    print("\n" + "=" * 60)
    print("示例 2: 多币种分析")
    print("=" * 60)
    
    strategy = TradingStrategy()
    
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    
    # 基础特征模板
    base_features = {
        'close': 46000.0,
        'rsi_14': 55.0,
        'macd': 100.0,
        'macd_signal': 80.0,
        'sma_20': 45000.0,
        'sma_50': 44000.0,
        'atr_14': 1000.0,
        'price_change_7d': 5.0,
        'fear_greed_index': 50,
        'exchange_outflow': 60000000,
        'exchange_inflow': 40000000,
        'social_sentiment': 0.2
    }
    
    results = []
    for symbol in symbols:
        # 为每个币种生成略有不同的特征
        features = base_features.copy()
        features['close'] *= (1 + hash(symbol) % 50 / 1000)
        features['rsi_14'] = 45 + hash(symbol) % 30
        
        market_data = {
            'symbol': symbol,
            'ohlcv': {
                'open': features['close'] * 0.98,
                'high': features['close'] * 1.02,
                'low': features['close'] * 0.97,
                'close': features['close'],
                'volume': 1000000000
            }
        }
        
        result = await strategy.analyze_opportunity(symbol, market_data, features)
        results.append(result)
        
        if result['success']:
            print(f"\n{symbol}: {result['signal'].upper()} (置信度: {result['confidence']:.1%})")
    
    # 汇总结果
    buy_signals = sum(1 for r in results if r['success'] and 'BUY' in r['signal'])
    sell_signals = sum(1 for r in results if r['success'] and 'SELL' in r['signal'])
    hold_signals = sum(1 for r in results if r['success'] and r['signal'] == 'hold')
    
    print(f"\n信号汇总:")
    print(f"  买入信号: {buy_signals}")
    print(f"  卖出信号: {sell_signals}")
    print(f"  观望信号: {hold_signals}")


async def example_risk_management():
    """风险管理示例"""
    print("\n" + "=" * 60)
    print("示例 3: 风险管理集成")
    print("=" * 60)
    
    strategy = TradingStrategy()
    
    # 高风险场景
    high_risk_features = {
        'close': 46000.0,
        'rsi_14': 85.0,  # 超买
        'macd': 200.0,
        'macd_signal': 150.0,
        'sma_20': 44000.0,
        'sma_50': 42000.0,
        'atr_14': 2500.0,  # 高波动
        'price_change_7d': 25.0,  # 大涨
        'fear_greed_index': 90,  # 极度贪婪
        'exchange_outflow': 20000000,
        'exchange_inflow': 100000000,  # 大量流入
        'social_sentiment': 0.8  # 过度乐观
    }
    
    market_data = {
        'symbol': 'BTC/USDT',
        'ohlcv': {
            'open': 45000.0,
            'high': 47000.0,
            'low': 44000.0,
            'close': 46000.0,
            'volume': 3000000000
        }
    }
    
    result = await strategy.analyze_opportunity('BTC/USDT', market_data, high_risk_features)
    
    if result['success']:
        print(f"\n高风险场景分析:")
        print(f"  信号: {result['signal'].upper()}")
        print(f"  置信度: {result['confidence']:.1%}")
        print(f"  建议仓位: {result['position_size']:.0%}")
        print(f"  建议操作: {result['recommendation']}")
        
        if result['confidence'] < 0.6:
            print(f"\n⚠️  风险提示: 置信度较低，建议观望或降低仓位")


async def example_custom_agent():
    """自定义代理示例"""
    print("\n" + "=" * 60)
    print("示例 4: 添加自定义专家代理")
    print("=" * 60)
    
    strategy = TradingStrategy()
    
    # 添加自定义代理配置
    custom_agent_config = {
        'name': 'pattern_recognizer',
        'specialization': '形态识别',
        'model_config': {
            'type': 'pattern',
            'patterns': ['head_and_shoulders', 'double_top', 'triangle']
        },
        'prompt_template': '识别图表形态并提供交易信号',
        'weight': 0.15,
        'confidence_threshold': 0.6
    }
    
    strategy.reasoning_engine.add_custom_agent(custom_agent_config)
    
    print(f"\n✅ 已添加自定义代理: {custom_agent_config['name']}")
    
    # 查看当前代理权重
    weights = strategy.reasoning_engine.get_agent_weights()
    print(f"\n当前代理权重:")
    for name, weight in weights.items():
        print(f"  {name}: {weight:.0%}")


async def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("推理层使用示例")
    print("OpenClaw Crypto Trading System")
    print("=" * 60)
    
    try:
        await example_basic_usage()
        await example_multi_symbol_analysis()
        await example_risk_management()
        await example_custom_agent()
        
        print("\n" + "=" * 60)
        print("所有示例执行完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 示例执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    from datetime import datetime
    asyncio.run(main())
