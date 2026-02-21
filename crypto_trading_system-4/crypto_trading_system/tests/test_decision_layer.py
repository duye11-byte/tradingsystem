"""
决策层测试脚本
测试决策层的所有功能
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime, timedelta
from core.decision import (
    DecisionEngine,
    DecisionConfig,
    Position,
    PositionSide,
    PortfolioState
)
from core.reasoning import TradingSignal, SignalType, ReasoningEngine
from core.features import FeatureEngineering, FeatureConfig
import pandas as pd
import numpy as np


def create_mock_trading_signal(
    symbol: str = "BTC/USDT",
    signal_type: SignalType = SignalType.BUY,
    confidence: float = 0.75
) -> TradingSignal:
    """创建模拟交易信号"""
    from core.reasoning import MarketAnalysis, ConsensusResult, CoTStep
    
    return TradingSignal(
        symbol=symbol,
        signal=signal_type,
        confidence=confidence,
        entry_price=45000.0,
        stop_loss=44100.0,
        take_profit=46800.0,
        position_size_ratio=0.1,
        reasoning_chain=[
            CoTStep(
                step_number=1,
                title="趋势识别",
                reasoning="上升趋势",
                evidence={},
                intermediate_conclusion="上升趋势",
                confidence=0.8
            ),
            CoTStep(
                step_number=7,
                title="综合判断",
                reasoning="买入信号",
                evidence={},
                intermediate_conclusion="BUY 信号",
                confidence=confidence
            )
        ],
        market_analysis=MarketAnalysis(
            symbol=symbol,
            timeframe="1h",
            analysis_timestamp=datetime.now(),
            technical_summary="看涨",
            trend_direction="上升",
            support_levels=[44000, 43000],
            resistance_levels=[46000, 47000],
            key_indicators={},
            onchain_summary="资金流出",
            exchange_flows={},
            whale_activity="活跃",
            network_health="健康",
            sentiment_summary="贪婪",
            fear_greed_index=65,
            social_sentiment=0.35,
            news_sentiment=0.28,
            overall_assessment="看涨",
            risk_level="中等",
            opportunity_score=0.7
        ),
        consensus_result=ConsensusResult(
            final_signal=signal_type,
            consensus_confidence=confidence,
            agreement_ratio=0.8,
            participating_models=["technical", "sentiment"],
            predictions=[],
            dissensus_analysis="一致",
            weighted_score=0.75
        ),
        consistency_check_passed=True,
        consistency_score=0.85,
        valid_until=datetime.now() + timedelta(hours=1)
    )


def create_mock_portfolio_state() -> PortfolioState:
    """创建模拟组合状态"""
    return PortfolioState(
        total_equity=100000.0,
        available_balance=100000.0,
        frozen_balance=0.0,
        positions={},
        total_unrealized_pnl=0.0,
        total_realized_pnl=0.0,
        today_realized_pnl=0.0,
        margin_used=0.0,
        margin_ratio=0.0
    )


async def test_signal_generator():
    """测试信号生成器"""
    print("=" * 60)
    print("测试 1: 信号生成器")
    print("=" * 60)
    
    from core.decision import SignalGenerator
    
    config = DecisionConfig()
    generator = SignalGenerator(config)
    
    # 创建买入信号
    signal = create_mock_trading_signal(
        symbol="BTC/USDT",
        signal_type=SignalType.BUY,
        confidence=0.75
    )
    
    portfolio = create_mock_portfolio_state()
    
    # 验证信号
    validation = generator.validate_signal(signal, portfolio.to_dict())
    print(f"\n信号验证结果:")
    print(f"  是否有效: {validation.is_valid}")
    print(f"  置信度分数: {validation.confidence_score:.1%}")
    print(f"  风险分数: {validation.risk_score:.1%}")
    print(f"  推荐仓位: {validation.recommended_position_size:.1%}")
    print(f"  通过的检查: {validation.passed_filters}")
    
    # 生成决策
    decision = generator.generate_decision(signal, portfolio.to_dict())
    
    if decision:
        print(f"\n✅ 决策生成成功!")
        print(f"  行动: {decision.action}")
        print(f"  数量: {decision.quantity:.4f}")
        print(f"  入场价格: {decision.entry_price:,.2f}")
        print(f"  止损: {decision.stop_loss:,.2f}")
        print(f"  止盈: {decision.take_profit:,.2f}")
        print(f"  风险金额: {decision.risk_amount:.2f}")
        print(f"  风险收益比: {decision.risk_reward_ratio:.2f}")
        print(f"  订单数量: {len(decision.orders)}")
    else:
        print("\n❌ 决策生成失败")


async def test_position_manager():
    """测试仓位管理器"""
    print("\n" + "=" * 60)
    print("测试 2: 仓位管理器")
    print("=" * 60)
    
    from core.decision import PositionManager
    
    config = DecisionConfig()
    pm = PositionManager(config)
    
    # 开仓
    position = pm.open_position(
        symbol="BTC/USDT",
        side=PositionSide.LONG,
        quantity=0.5,
        entry_price=45000.0,
        stop_loss=44100.0,
        take_profit=46800.0
    )
    
    print(f"\n✅ 开仓成功!")
    print(f"  交易对: {position.symbol}")
    print(f"  方向: {position.side.value}")
    print(f"  数量: {position.quantity:.4f}")
    print(f"  入场价格: {position.average_entry_price:,.2f}")
    
    # 更新价格
    pm.update_position_price("BTC/USDT", 46000.0)
    position = pm.get_position("BTC/USDT")
    
    print(f"\n价格更新后:")
    print(f"  当前价格: {position.current_price:,.2f}")
    print(f"  未实现盈亏: {position.unrealized_pnl:.2f}")
    print(f"  未实现盈亏%: {position.unrealized_pnl_pct:.2%}")
    
    # 加仓
    pm.add_to_position("BTC/USDT", 0.3, 46000.0)
    position = pm.get_position("BTC/USDT")
    
    print(f"\n加仓后:")
    print(f"  新数量: {position.quantity:.4f}")
    print(f"  新平均入场价: {position.average_entry_price:,.2f}")
    
    # 获取组合摘要
    summary = pm.get_portfolio_summary()
    print(f"\n组合摘要:")
    print(f"  持仓数量: {summary['position_count']}")
    print(f"  总敞口: {summary['total_exposure']:,.2f}")
    print(f"  总未实现盈亏: {summary['total_unrealized_pnl']:.2f}")


async def test_risk_manager():
    """测试风险管理器"""
    print("\n" + "=" * 60)
    print("测试 3: 风险管理器")
    print("=" * 60)
    
    from core.decision import RiskManager
    
    config = DecisionConfig()
    rm = RiskManager(config)
    
    # 创建组合状态
    portfolio = create_mock_portfolio_state()
    
    # 创建持仓
    from core.decision import PositionManager
    pm = PositionManager(config)
    pm.open_position(
        symbol="BTC/USDT",
        side=PositionSide.LONG,
        quantity=0.5,
        entry_price=45000.0,
        stop_loss=44100.0
    )
    pm.update_position_price("BTC/USDT", 46000.0)
    
    # 评估风险
    positions = pm.get_all_positions()
    risk_profile = rm.assess_risk(portfolio, positions)
    
    print(f"\n✅ 风险评估完成!")
    print(f"  总敞口: {risk_profile.total_exposure:,.2f}")
    print(f"  总风险: {risk_profile.total_risk:,.2f}")
    print(f"  当前回撤: {risk_profile.current_drawdown:.2%}")
    print(f"  持仓数量: {risk_profile.position_count}")
    print(f"  风险限制: {'已触发' if risk_profile.risk_limit_reached else '正常'}")
    
    # 获取风险报告
    report = rm.get_risk_report()
    print(f"\n风险报告:")
    print(f"  最大回撤: {report['drawdown_statistics']['max_drawdown']:.2%}")
    print(f"  日风险限制: {report['risk_limits']['max_daily_risk']:.2%}")


async def test_execution_optimizer():
    """测试执行优化器"""
    print("\n" + "=" * 60)
    print("测试 4: 执行优化器")
    print("=" * 60)
    
    from core.decision import ExecutionOptimizer, ExecutionStrategy, OrderSide
    
    config = DecisionConfig()
    optimizer = ExecutionOptimizer(config)
    
    # 测试不同执行策略
    strategies = [
        ExecutionStrategy.IMMEDIATE,
        ExecutionStrategy.TWAP,
        ExecutionStrategy.ICEBERG
    ]
    
    for strategy in strategies:
        plan = optimizer.create_execution_plan(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            total_quantity=1.0,
            target_price=45000.0,
            strategy=strategy
        )
        
        print(f"\n{strategy.value.upper()} 策略:")
        print(f"  订单数量: {len(plan.orders)}")
        print(f"  总数量: {plan.total_quantity:.4f}")
        
        for i, order in enumerate(plan.orders[:3]):  # 只显示前3个
            print(f"  订单 {i+1}: {order.order_type.value} {order.quantity:.4f}")


async def test_order_manager():
    """测试订单管理器"""
    print("\n" + "=" * 60)
    print("测试 5: 订单管理器")
    print("=" * 60)
    
    from core.decision import OrderManager, Order, OrderType, OrderSide
    
    config = DecisionConfig()
    om = OrderManager(config)
    
    # 创建订单
    order = Order(
        id="test-order-1",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.5
    )
    
    # 提交订单
    result = om.submit_order(order)
    print(f"\n✅ 订单提交: {result}")
    
    # 获取订单
    retrieved = om.get_order("test-order-1")
    print(f"  订单状态: {retrieved.status.value}")
    
    # 等待模拟执行
    await asyncio.sleep(0.5)
    
    # 更新后查看
    retrieved = om.get_order("test-order-1")
    print(f"  执行后状态: {retrieved.status.value}")
    print(f"  成交数量: {retrieved.filled_quantity:.4f}")
    print(f"  成交均价: {retrieved.average_fill_price:.2f}")
    
    # 获取统计
    stats = om.get_order_statistics()
    print(f"\n订单统计:")
    print(f"  总订单: {stats['total_orders']}")
    print(f"  已成交: {stats['filled_orders']}")
    print(f"  成交率: {stats['fill_rate']:.1%}")


async def test_decision_engine():
    """测试完整决策引擎"""
    print("\n" + "=" * 60)
    print("测试 6: 完整决策引擎")
    print("=" * 60)
    
    # 创建决策引擎
    config = DecisionConfig()
    engine = DecisionEngine(config)
    
    # 设置组合状态
    portfolio = create_mock_portfolio_state()
    engine.set_portfolio_state(portfolio)
    
    # 处理买入信号
    buy_signal = create_mock_trading_signal(
        symbol="BTC/USDT",
        signal_type=SignalType.BUY,
        confidence=0.75
    )
    
    decision = await engine.process_signal(buy_signal, current_price=45000.0)
    
    if decision:
        print(f"\n✅ 买入决策执行成功!")
        print(f"  行动: {decision.action}")
        print(f"  数量: {decision.quantity:.4f}")
        print(f"  订单: {len(decision.orders)} 个")
    
    # 等待订单执行
    await asyncio.sleep(0.5)
    
    # 获取组合摘要
    summary = engine.get_portfolio_summary()
    print(f"\n组合状态:")
    print(f"  总权益: {summary['portfolio']['total_equity']:,.2f}")
    print(f"  可用资金: {summary['portfolio']['available_balance']:,.2f}")
    print(f"  持仓数量: {summary['positions']['position_count']}")
    
    # 处理卖出信号
    sell_signal = create_mock_trading_signal(
        symbol="BTC/USDT",
        signal_type=SignalType.SELL,
        confidence=0.70
    )
    
    # 更新价格
    engine.update_position_prices({"BTC/USDT": 46500.0})
    
    # 获取决策统计
    stats = engine.get_decision_stats()
    print(f"\n决策统计:")
    print(f"  总决策: {stats['total_decisions']}")
    print(f"  执行决策: {stats['executed_decisions']}")
    print(f"  成功率: {stats['success_rate']:.1%}")


async def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("决策层测试套件")
    print("=" * 60)
    
    try:
        await test_signal_generator()
        await test_position_manager()
        await test_risk_manager()
        await test_execution_optimizer()
        await test_order_manager()
        await test_decision_engine()
        
        print("\n" + "=" * 60)
        print("所有测试完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
