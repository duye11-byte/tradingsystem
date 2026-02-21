"""
反馈层测试脚本
验证反馈引擎的各个组件功能
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datetime import datetime, timedelta
import random
import numpy as np

from core.feedback.feedback_types import (
    TradeRecord, TradeResult, LearningSample, HumanFeedback, FeedbackType, PerformanceMetrics
)
from core.feedback.performance_analyzer import PerformanceAnalyzer
from core.feedback.online_learner import OnlineLearner
from core.feedback.rlhf_trainer import RLHFTrainer
from core.feedback.feedback_store import FeedbackStore
from core.feedback.feedback_engine import FeedbackEngine, FeedbackMode


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
            entry_quantity=random.uniform(0.1, 1.0),
            exit_time=exit_time,
            exit_price=exit_price,
            realized_pnl=(exit_price - entry_price) * (1 if side == 'buy' else -1) * random.uniform(0.1, 1.0),
            result=TradeResult.WIN if is_win else TradeResult.LOSS,
            is_closed=True
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
    print(f"  平均交易盈亏: ${metrics.avg_pnl_per_trade:,.2f}")
    
    # 生成报告
    report = analyzer.generate_report()
    print(f"\n交易报告摘要:")
    print(f"  当前连赢: {analyzer.get_winning_streak()}")
    print(f"  当前连亏: {analyzer.get_losing_streak()}")
    
    print("\n✅ 性能分析器测试通过")
    return analyzer


def test_online_learner():
    """测试在线学习模块"""
    print("\n" + "="*60)
    print("测试在线学习模块 (Online Learner)")
    print("="*60)
    
    learner = OnlineLearner(learning_rate=0.01, min_samples=30)
    
    # 添加学习样本 (添加更多样本以触发学习)
    for i in range(60):
        is_win = random.random() < 0.55
        reward = random.uniform(0.02, 0.08) if is_win else random.uniform(-0.05, -0.01)
        sample = LearningSample(
            id=f"sample_{i:03d}",
            features={'price': random.uniform(20000, 60000), 'volume': random.uniform(1000, 5000)},
            predicted_signal=random.choice(['buy', 'sell', 'hold']),
            predicted_confidence=random.uniform(0.5, 0.95),
            actual_result='win' if is_win else 'loss',
            actual_pnl=reward,
            reward=reward,
            timestamp=datetime.now() - timedelta(hours=i),
            metadata={
                'symbol': random.choice(['BTC/USDT', 'ETH/USDT']),
                'participating_agents': ['technical_analyst', 'onchain_analyst'] if random.random() > 0.5 else ['sentiment_analyst', 'macro_analyst']
            }
        )
        learner.add_sample(sample)
    
    print(f"✓ 添加了 {len(learner.samples)} 个学习样本")
    
    # 初始权重
    print(f"\n初始代理权重:")
    for agent, weight in learner.agent_weights.items():
        print(f"  {agent}: {weight:.3f}")
    
    # 执行学习
    update = learner.learn()
    
    print(f"\n学习后代理权重:")
    for agent, weight in learner.agent_weights.items():
        print(f"  {agent}: {weight:.3f}")
    
    if update:
        print(f"\n生成模型更新: {update.id}")
        print(f"  组件: {update.component}")
        print(f"  验证分数: {update.validation_score:.3f}")
        print(f"  状态: {update.status}")
    else:
        print(f"\n未生成模型更新（样本不足 {learner.min_samples}）")
    
    # 获取特征重要性
    importance = learner.get_feature_importance()
    if importance:
        print(f"\n特征重要性 (Top 5):")
        for feature, score in list(importance.items())[:5]:
            print(f"  {feature}: {score:.3f}")
    
    # 获取统计
    stats = learner.get_learning_stats()
    print(f"\n学习统计:")
    print(f"  总样本数: {stats['total_samples']}")
    print(f"  已处理样本: {stats['processed_samples']}")
    print(f"  总更新数: {stats['total_updates']}")
    
    print("\n✅ 在线学习模块测试通过")
    return learner


def test_rlhf_trainer():
    """测试RLHF训练器"""
    print("\n" + "="*60)
    print("测试RLHF训练器 (RLHF Trainer)")
    print("="*60)
    
    trainer = RLHFTrainer()
    
    # 添加人类反馈（使用decision_id）
    for i in range(20):
        feedback = HumanFeedback(
            id=f"feedback_{i:03d}",
            feedback_type=FeedbackType.HUMAN_RATING,
            decision_id=f"decision_{i:04d}",
            rating=random.randint(1, 5),
            comment=random.choice(['很好', '一般', '需要改进', '优秀', '较差']),
            timestamp=datetime.now() - timedelta(hours=i * 2)
        )
        trainer.add_human_feedback(feedback)
    
    print(f"✓ 添加了 {len(trainer.human_feedback)} 条人类反馈")
    print(f"  平均评分: {trainer.stats['avg_human_rating']:.2f}")
    
    # 添加学习样本用于构建偏好对（使用decision_id匹配）
    samples = []
    for i in range(20):
        sample = LearningSample(
            id=f"sample_{i:03d}",
            features={'feature': random.random(), 'confidence': random.uniform(0.5, 0.95)},
            predicted_signal='buy',
            predicted_confidence=random.uniform(0.5, 0.95),
            actual_result='win' if random.random() > 0.5 else 'loss',
            actual_pnl=random.uniform(-0.05, 0.08),
            reward=random.uniform(-0.05, 0.08),
            timestamp=datetime.now() - timedelta(hours=i),
            metadata={
                'decision_id': f"decision_{i:04d}",
                'consistency_score': random.uniform(0.5, 0.9),
                'risk_reward_ratio': random.uniform(1.0, 3.0)
            }
        )
        samples.append(sample)
    
    # 构建偏好对
    trainer.build_preference_pairs(samples)
    print(f"✓ 构建了 {len(trainer.preference_pairs)} 个偏好对")
    
    # 训练奖励模型
    if len(trainer.preference_pairs) >= 5:
        print("\n训练奖励模型...")
        reward_model = trainer.train_reward_model()
        print(f"  奖励模型权重:")
        for key, value in reward_model.items():
            print(f"    {key}: {value:.3f}")
        
        # 优化策略
        print("\n优化策略...")
        policy_update = trainer.optimize_policy(samples)
        if policy_update:
            print(f"  策略更新: {policy_update.id}")
            print(f"  验证分数: {policy_update.validation_score:.3f}")
            print(f"  状态: {policy_update.status}")
    
    # 获取人类反馈摘要
    summary = trainer.get_human_feedback_summary()
    print(f"\n人类反馈摘要:")
    print(f"  总反馈数: {summary['total']}")
    print(f"  评分分布: {summary['rating_distribution']}")
    
    print("\n✅ RLHF训练器测试通过")
    return trainer


def test_feedback_store():
    """测试反馈存储"""
    print("\n" + "="*60)
    print("测试反馈存储 (Feedback Store)")
    print("="*60)
    
    store = FeedbackStore(storage_path='./test_feedback_data')
    
    # 存储交易记录
    trade = TradeRecord(
        id="test_trade_001",
        symbol="BTC/USDT",
        entry_time=datetime.now(),
        entry_price=50000.0,
        entry_side="buy",
        entry_quantity=1.0,
        is_closed=True,
        realized_pnl=1000.0
    )
    store.save_trade(trade)
    print("✓ 存储交易记录")
    
    # 存储人类反馈
    feedback = HumanFeedback(
        id="test_feedback_001",
        feedback_type=FeedbackType.HUMAN_RATING,
        rating=4,
        comment="测试反馈",
        timestamp=datetime.now()
    )
    store.save_human_feedback(feedback)
    print("✓ 存储人类反馈")
    
    # 存储学习样本
    sample = LearningSample(
        id="test_sample_001",
        features={'test': 1.0},
        predicted_signal='buy',
        predicted_confidence=0.8,
        actual_result='win',
        actual_pnl=0.05,
        reward=0.05,
        timestamp=datetime.now()
    )
    store.save_sample(sample)
    print("✓ 存储学习样本")
    
    # 获取统计
    stats = store.get_statistics()
    print(f"\n存储统计:")
    print(f"  交易记录: {stats['trades']}")
    print(f"  人类反馈: {stats['human_feedback']}")
    print(f"  学习样本: {stats['samples']}")
    
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
    
    # 生成洞察
    if result.insights:
        print(f"\n策略洞察:")
        for insight in result.insights[:3]:
            print(f"  - {insight.message} (置信度: {insight.confidence:.2f})")
    
    # 添加人类反馈
    for i in range(15):
        feedback = HumanFeedback(
            id=f"hf_{i:03d}",
            feedback_type=FeedbackType.HUMAN_RATING,
            rating=random.randint(1, 5),
            timestamp=datetime.now() - timedelta(hours=i)
        )
        engine.add_human_feedback(feedback)
    
    print(f"\n✓ 添加了 15 条人类反馈")
    
    # 添加学习样本（通过关闭交易自动创建）
    for i, trade in enumerate(trades[:20]):
        if trade.result == TradeResult.OPEN:
            trade.close_trade(
                exit_price=trade.entry_price * (1 + random.uniform(-0.05, 0.08)),
                exit_time=trade.entry_time + timedelta(hours=random.randint(1, 24)),
                exit_reason="take_profit" if random.random() > 0.5 else "stop_loss"
            )
            # 创建学习样本
            sample = LearningSample(
                id=f"sample_{i:03d}",
                features={'price': trade.entry_price, 'volume': random.uniform(1000, 5000)},
                predicted_signal=trade.entry_side,
                predicted_confidence=random.uniform(0.6, 0.9),
                actual_result=trade.result.value,
                actual_pnl=trade.realized_pnl,
                reward=1.0 if trade.result == TradeResult.WIN else -1.0,
                timestamp=trade.exit_time,
                metadata={
                    'trade_id': trade.id,
                    'symbol': trade.symbol,
                    'participating_agents': random.sample(['technical_analyst', 'onchain_analyst', 'sentiment_analyst'], k=2)
                }
            )
            engine.online_learner.add_sample(sample)
    
    print(f"✓ 添加了 {len(engine.online_learner.samples)} 个学习样本")
    
    # 在线学习
    learning_result = await engine.run_online_learning()
    print(f"\n在线学习结果:")
    print(f"  成功: {learning_result.success}")
    print(f"  消息: {learning_result.message}")
    if learning_result.updates:
        print(f"  应用更新数: {len(learning_result.updates)}")
    
    # 生成报告
    report = engine.generate_report(days=30)
    print(f"\n反馈报告:")
    print(f"  总交易: {report['performance']['total_trades']}")
    print(f"  胜率: {report['performance']['win_rate']:.2%}")
    print(f"  总盈亏: ${report['performance']['total_pnl']:,.2f}")
    print(f"  夏普比率: {report['performance']['sharpe_ratio']:.2f}")
    print(f"  最大回撤: {report['performance']['max_drawdown']:.2%}")
    
    # 获取状态
    status = engine.get_status()
    print(f"\n引擎状态:")
    print(f"  运行中: {status['running']}")
    print(f"  模式: {status['mode']}")
    print(f"  交易数: {status['trades_count']}")
    print(f"  样本数: {status['samples_count']}")
    print(f"  反馈数: {status['feedback_count']}")
    
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
