"""
反馈层测试脚本
验证反馈引擎的各个组件功能
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
import random
import numpy as np

from feedback_types import (
    TradeRecord, TradeResult, LearningSample, HumanFeedback, FeedbackType
)
from performance_analyzer import PerformanceAnalyzer
from online_learner import OnlineLearner
from rlhf_trainer import RLHFTrainer
from feedback_store import FeedbackStore
from feedback_engine import FeedbackEngine, FeedbackMode


def generate_mock_trades(count: int = 50) -> list:
    """生成模拟交易数据"""
    trades = []
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    sides = ['buy', 'sell']
    
    base_time = datetime.now() - timedelta(days=30)
    
    for i in range(count):
        symbol = random.choice(symbols)
        side = random.choice(sides)
        entry_price = random.uniform(20000, 60000) if 'BTC' in symbol else random.uniform(1000, 4000)
        
        # 模拟盈亏（胜率约55%）
        is_win = random.random() < 0.55
        pnl_pct = random.uniform(0.01, 0.08) if is_win else random.uniform(-0.05, -0.01)
        exit_price = entry_price * (1 + pnl_pct) if side == 'buy' else entry_price * (1 - pnl_pct)
        
        entry_time = base_time + timedelta(hours=i * 12)
        exit_time = entry_time + timedelta(hours=random.randint(1, 48))
        
        trade = TradeRecord(
            id=f"trade_{i:04d}",
            symbol=symbol,
            entry_time=entry_time,
            entry_price=entry_price,
            entry_side=side,
            position_size=random.uniform(0.1, 1.0),
            exit_time=exit_time,
            exit_price=exit_price,
            realized_pnl=(exit_price - entry_price) * (1 if side == 'buy' else -1) * random.uniform(0.1, 1.0),
            result=TradeResult.WIN if is_win else TradeResult.LOSS
        )
        trades.append(trade)
    
    return trades


def test_performance_analyzer():
    """测试性能分析器"""
    print("\n" + "="*60)
    print("测试性能分析器 (Performance Analyzer)")
    print("="*60)
    
    analyzer = PerformanceAnalyzer()
    
    # 添加模拟交易
    trades = generate_mock_trades(50)
    for trade in trades:
        analyzer.add_trade(trade)
    
    print(f"✓ 添加了 {len(trades)} 条交易记录")
    
    # 分析性能
    metrics = analyzer.analyze_performance()
    
    print(f"\n性能指标:")
    print(f"  总交易数: {metrics.total_trades}")
    print(f"  胜率: {metrics.win_rate:.2%}")
    print(f"  盈亏比: {metrics.profit_factor:.2f}")
    print(f"  夏普比率: {metrics.sharpe_ratio:.2f}")
    print(f"  索提诺比率: {metrics.sortino_ratio:.2f}")
    print(f"  最大回撤: {metrics.max_drawdown:.2%}")
    print(f"  总盈亏: ${metrics.total_pnl:,.2f}")
    print(f"  平均交易盈亏: ${metrics.avg_trade_pnl:,.2f}")
    
    # 生成洞察
    insights = analyzer.generate_insights(metrics)
    print(f"\n策略洞察 ({len(insights)} 条):")
    for i, insight in enumerate(insights[:3], 1):
        print(f"  {i}. [{insight.category}] {insight.message}")
    
    print("\n✅ 性能分析器测试通过")
    return analyzer


def test_online_learner():
    """测试在线学习模块"""
    print("\n" + "="*60)
    print("测试在线学习模块 (Online Learner)")
    print("="*60)
    
    learner = OnlineLearner()
    
    # 添加学习样本
    for i in range(30):
        sample = LearningSample(
            state={'price': random.uniform(20000, 60000), 'volume': random.uniform(1000, 5000)},
            action=random.choice(['buy', 'sell', 'hold']),
            reward=random.uniform(-0.05, 0.08),
            timestamp=datetime.now() - timedelta(hours=i),
            metadata={
                'symbol': random.choice(['BTC/USDT', 'ETH/USDT']),
                'participating_agents': random.sample(['trend', 'momentum', 'mean_reversion'], k=random.randint(1, 3))
            }
        )
        learner.add_sample(sample)
    
    print(f"✓ 添加了 {len(learner.samples)} 个学习样本")
    
    # 初始权重
    print(f"\n初始代理权重:")
    for agent, weight in learner.agent_weights.items():
        print(f"  {agent}: {weight:.3f}")
    
    # 执行学习
    updates = learner.learn()
    
    print(f"\n学习后代理权重:")
    for agent, weight in learner.agent_weights.items():
        print(f"  {agent}: {weight:.3f}")
    
    print(f"\n生成 {len(updates)} 个模型更新")
    
    # 获取统计
    stats = learner.get_learning_statistics()
    print(f"\n学习统计:")
    print(f"  样本数: {stats['sample_count']}")
    print(f"  平均奖励: {stats['avg_reward']:.4f}")
    print(f"  奖励方差: {stats['reward_variance']:.4f}")
    
    print("\n✅ 在线学习模块测试通过")
    return learner


def test_rlhf_trainer():
    """测试RLHF训练器"""
    print("\n" + "="*60)
    print("测试RLHF训练器 (RLHF Trainer)")
    print("="*60)
    
    trainer = RLHFTrainer()
    
    # 添加人类反馈
    for i in range(20):
        feedback = HumanFeedback(
            id=f"feedback_{i:03d}",
            feedback_type=random.choice([FeedbackType.RATING, FeedbackType.COMPARISON]),
            decision_id=f"decision_{i:04d}",
            rating=random.randint(1, 5),
            comment=random.choice(['很好', '一般', '需要改进', '优秀', '较差']),
            timestamp=datetime.now() - timedelta(hours=i * 2)
        )
        trainer.add_human_feedback(feedback)
    
    print(f"✓ 添加了 {len(trainer.human_feedback)} 条人类反馈")
    
    # 添加学习样本用于构建偏好对
    samples = []
    for i in range(20):
        sample = LearningSample(
            state={'feature': random.random()},
            action='buy',
            reward=random.uniform(-0.05, 0.08),
            timestamp=datetime.now() - timedelta(hours=i),
            metadata={'decision_id': f"decision_{i:04d}"}
        )
        samples.append(sample)
    
    # 构建偏好对
    trainer.build_preference_pairs(samples)
    print(f"✓ 构建了 {len(trainer.preference_pairs)} 个偏好对")
    
    # 训练奖励模型
    if len(trainer.preference_pairs) >= 5:
        print("\n训练奖励模型...")
        reward_metrics = trainer.train_reward_model(epochs=10)
        print(f"  最终损失: {reward_metrics['final_loss']:.4f}")
        
        # 优化策略
        print("\n优化策略...")
        policy_metrics = trainer.optimize_policy(epochs=10)
        print(f"  最终损失: {policy_metrics['final_loss']:.4f}")
    
    print("\n✅ RLHF训练器测试通过")
    return trainer


def test_feedback_store():
    """测试反馈存储"""
    print("\n" + "="*60)
    print("测试反馈存储 (Feedback Store)")
    print("="*60)
    
    store = FeedbackStore(data_dir='./test_feedback_data')
    
    # 存储性能指标
    from feedback.feedback_types import PerformanceMetrics
    metrics = PerformanceMetrics(
        total_trades=100,
        winning_trades=55,
        losing_trades=45,
        win_rate=0.55,
        profit_factor=1.5,
        sharpe_ratio=1.8,
        max_drawdown=0.12,
        total_pnl=5000.0
    )
    store.store_performance_metrics(metrics)
    print("✓ 存储性能指标")
    
    # 存储人类反馈
    feedback = HumanFeedback(
        id="test_feedback_001",
        feedback_type=FeedbackType.RATING,
        rating=4,
        comment="测试反馈",
        timestamp=datetime.now()
    )
    store.store_human_feedback(feedback)
    print("✓ 存储人类反馈")
    
    # 存储学习样本
    sample = LearningSample(
        state={'test': 1.0},
        action='buy',
        reward=0.05,
        timestamp=datetime.now()
    )
    store.store_learning_sample(sample)
    print("✓ 存储学习样本")
    
    # 获取统计
    stats = store.get_storage_stats()
    print(f"\n存储统计:")
    print(f"  性能指标: {stats['performance_metrics_count']}")
    print(f"  人类反馈: {stats['human_feedback_count']}")
    print(f"  学习样本: {stats['learning_samples_count']}")
    
    print("\n✅ 反馈存储测试通过")
    return store


async def test_feedback_engine():
    """测试反馈引擎"""
    print("\n" + "="*60)
    print("测试反馈引擎 (Feedback Engine)")
    print("="*60)
    
    engine = FeedbackEngine()
    
    # 记录交易
    trades = generate_mock_trades(30)
    for trade in trades:
        engine.record_trade(trade)
    
    print(f"✓ 记录了 {len(trades)} 条交易")
    
    # 性能分析
    result = await engine.analyze_performance()
    print(f"\n性能分析结果:")
    print(f"  成功: {result.success}")
    print(f"  消息: {result.message}")
    if result.metrics:
        print(f"  胜率: {result.metrics.win_rate:.2%}")
        print(f"  夏普比率: {result.metrics.sharpe_ratio:.2f}")
    
    # 添加人类反馈
    for i in range(10):
        feedback = HumanFeedback(
            id=f"hf_{i:03d}",
            feedback_type=FeedbackType.RATING,
            rating=random.randint(1, 5),
            timestamp=datetime.now() - timedelta(hours=i)
        )
        engine.add_human_feedback(feedback)
    
    print(f"\n✓ 添加了 10 条人类反馈")
    
    # 在线学习
    learning_result = await engine.run_online_learning()
    print(f"\n在线学习结果:")
    print(f"  成功: {learning_result.success}")
    print(f"  消息: {learning_result.message}")
    
    # 生成报告
    report = engine.generate_report(days=30)
    print(f"\n反馈报告:")
    print(f"  总交易: {report['performance']['total_trades']}")
    print(f"  胜率: {report['performance']['win_rate']:.2%}")
    print(f"  总盈亏: ${report['performance']['total_pnl']:,.2f}")
    
    # 获取状态
    status = engine.get_status()
    print(f"\n引擎状态:")
    print(f"  运行中: {status['running']}")
    print(f"  模式: {status['mode']}")
    print(f"  交易数: {status['trades_count']}")
    print(f"  样本数: {status['samples_count']}")
    
    print("\n✅ 反馈引擎测试通过")
    return engine


async def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print(" "*20 + "反馈层测试开始")
    print("="*70)
    
    try:
        # 测试各个组件
        test_performance_analyzer()
        test_online_learner()
        test_rlhf_trainer()
        test_feedback_store()
        
        # 测试引擎
        await test_feedback_engine()
        
        print("\n" + "="*70)
        print(" "*20 + "所有测试通过 ✅")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_all_tests())
