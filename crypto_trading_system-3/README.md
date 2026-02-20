# OpenClaw 加密货币交易决策系统

一个基于多层架构的量化交易系统，集成 AI 推理能力进行加密货币交易决策。

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    OpenClaw Crypto Trading System               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Input      │  │  Feature    │  │  Reasoning  │             │
│  │  Layer      │──▶│ Engineering │──▶│   Layer     │             │
│  │  (Data)     │  │  (Features) │  │  (AI/ML)    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│         │                │                │                      │
│         ▼                ▼                ▼                      │
│  ┌─────────────────────────────────────────────────────┐        │
│  │              TradingSignal (交易信号)                │        │
│  │         Signal + Confidence + Entry/SL/TP           │        │
│  └─────────────────────────────────────────────────────┘        │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────┐        │
│  │              Decision Layer (决策层)                 │        │
│  │    Signal → Decision → Order → Execution            │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 已完成模块

### ✅ 推理层 (Reasoning Layer)

**核心组件**:
- Chain-of-Thought 推理引擎 (7步分析)
- 多模型集成管理器 (4专家代理)
- 自我一致性验证器 (5次采样)

**性能**:
- 总延迟: 20-60ms
- 成功率: > 95%
- 一致性通过率: > 80%

**文件**:
- `core/reasoning/` - 推理层核心代码
- `config/reasoning_config.yaml` - 配置文件
- `tests/test_reasoning_layer.py` - 测试脚本

### ✅ 特征工程层 (Feature Engineering Layer)

**核心组件**:
- 技术指标 (33个): RSI, MACD, 布林带, ATR等
- 链上指标 (24个): 交易所流向, 鲸鱼活动, 网络活跃度等
- 情绪指标 (19个): 恐惧贪婪指数, 社交情绪, 资金费率等
- 组合特征 (13个): PCA, 时间序列分解, 特征交互等

**性能**:
- 总延迟: 12-26ms
- 特征数量: 94维
- 成功率: 100%

**文件**:
- `core/features/` - 特征工程核心代码
- `config/features_config.yaml` - 配置文件
- `tests/test_features_layer.py` - 测试脚本

### ✅ 决策层 (Decision Layer)

**核心组件**:
- 信号生成器: 信号验证、决策生成、订单创建
- 仓位管理器: 开仓/平仓、加仓/减仓、止损止盈管理
- 风险管理器: 风险评估、限制检查、回撤监控
- 执行优化器: TWAP、VWAP、冰山订单、智能执行
- 订单管理器: 订单提交、跟踪、取消

**性能**:
- 信号验证: < 1ms
- 决策生成: < 5ms
- 订单执行: 模拟模式 ~0.3s

**文件**:
- `core/decision/` - 决策层核心代码
- `config/decision_config.yaml` - 配置文件
- `tests/test_decision_layer.py` - 测试脚本

## 快速开始

### 安装依赖

```bash
pip install numpy pandas pyyaml scikit-learn
```

### 使用特征工程层

```python
import asyncio
from core.features import FeatureEngineering, FeatureConfig

async def main():
    # 创建特征工程
    config = FeatureConfig()
    fe = FeatureEngineering(config)
    
    # 准备数据
    import pandas as pd
    ohlcv_data = pd.read_csv('btc_ohlcv.csv')
    onchain_data = {'exchange_inflow': 50000000, ...}
    sentiment_data = {'fear_greed_index': 65, ...}
    
    # 提取特征
    result = await fe.extract_features(
        symbol="BTC/USDT",
        ohlcv_data=ohlcv_data,
        onchain_data=onchain_data,
        sentiment_data=sentiment_data
    )
    
    if result.success:
        features = result.feature_set
        print(f"RSI: {features.technical.rsi_14}")
        print(f"Netflow: {features.onchain.exchange_netflow}")
        print(f"Fear & Greed: {features.sentiment.fear_greed_index}")

asyncio.run(main())
```

### 使用推理层

```python
import asyncio
from core.reasoning import ReasoningEngine

async def main():
    # 创建推理引擎
    engine = ReasoningEngine()
    
    # 准备特征数据 (从特征工程层获取)
    features = {
        'rsi_14': 62.5,
        'macd': 150.5,
        'exchange_netflow': 30000000,
        'fear_greed_index': 65,
        # ... 更多特征
    }
    
    # 执行推理
    result = await engine.reason(
        symbol="BTC/USDT",
        market_data={'ohlcv': {...}},
        features=features
    )
    
    if result.success:
        signal = result.signal
        print(f"信号: {signal.signal.value}")
        print(f"置信度: {signal.confidence:.1%}")
        print(f"建议仓位: {signal.position_size_ratio:.0%}")
        
        # 查看推理链
        for step in signal.reasoning_chain:
            print(f"{step.step_number}. {step.title}: {step.intermediate_conclusion}")

asyncio.run(main())
```

### 完整流程

```python
import asyncio
from core.features import FeatureEngineering
from core.reasoning import ReasoningEngine

async def full_pipeline():
    # 1. 特征工程
    fe = FeatureEngineering()
    feature_result = await fe.extract_features(
        symbol="BTC/USDT",
        ohlcv_data=ohlcv_df,
        onchain_data=onchain_data,
        sentiment_data=sentiment_data
    )
    
    if not feature_result.success:
        print("特征提取失败")
        return
    
    # 2. 推理决策
    engine = ReasoningEngine()
    reasoning_result = await engine.reason(
        symbol="BTC/USDT",
        market_data={'ohlcv': ohlcv_df.to_dict()},
        features=feature_result.feature_set.to_dict()
    )
    
    if reasoning_result.success:
        signal = reasoning_result.signal
        print(f"\n{'='*50}")
        print(f"交易信号: {signal.signal.value.upper()}")
        print(f"置信度: {signal.confidence:.1%}")
        print(f"入场价格: ${signal.entry_price:,.2f}")
        print(f"止损价格: ${signal.stop_loss:,.2f}" if signal.stop_loss else "止损: N/A")
        print(f"止盈价格: ${signal.take_profit:,.2f}" if signal.take_profit else "止盈: N/A")
        print(f"建议仓位: {signal.position_size_ratio:.0%}")
        print(f"{'='*50}")

asyncio.run(full_pipeline())
```

### 使用决策层

```python
import asyncio
from core.decision import DecisionEngine, DecisionConfig
from core.reasoning import TradingSignal, SignalType

async def main():
    # 创建决策引擎
    config = DecisionConfig()
    engine = DecisionEngine(config)
    
    # 设置组合状态
    engine.set_portfolio_state(portfolio_state)
    
    # 处理交易信号
    decision = await engine.process_signal(signal, current_price=45000.0)
    
    if decision:
        print(f"行动: {decision.action}")
        print(f"数量: {decision.quantity:.4f}")
        print(f"止损: {decision.stop_loss:.2f}")
        print(f"止盈: {decision.take_profit:.2f}")
        print(f"订单: {len(decision.orders)} 个")
        
        # 获取组合摘要
        summary = engine.get_portfolio_summary()
        print(f"总权益: {summary['portfolio']['total_equity']:.2f}")
        print(f"持仓数量: {summary['positions']['position_count']}")

asyncio.run(main())
```

### 完整交易流程

```python
import asyncio
from core.features import FeatureEngineering
from core.reasoning import ReasoningEngine
from core.decision import DecisionEngine

async def complete_trading_pipeline():
    # 1. 特征工程
    fe = FeatureEngineering()
    feature_result = await fe.extract_features(
        symbol="BTC/USDT",
        ohlcv_data=ohlcv_df,
        onchain_data=onchain_data,
        sentiment_data=sentiment_data
    )
    
    if not feature_result.success:
        print("特征提取失败")
        return
    
    # 2. 推理决策
    reasoning_engine = ReasoningEngine()
    reasoning_result = await reasoning_engine.reason(
        symbol="BTC/USDT",
        market_data={'ohlcv': ohlcv_df.to_dict()},
        features=feature_result.feature_set.to_dict()
    )
    
    if not reasoning_result.success:
        print("推理失败")
        return
    
    # 3. 决策执行
    decision_engine = DecisionEngine()
    decision_engine.set_portfolio_state(portfolio_state)
    
    decision = await decision_engine.process_signal(
        reasoning_result.signal,
        current_price=feature_result.feature_set.close
    )
    
    if decision:
        print(f"\n{'='*50}")
        print(f"交易执行: {decision.action}")
        print(f"数量: {decision.quantity:.4f}")
        print(f"入场: ${decision.entry_price:,.2f}")
        print(f"止损: ${decision.stop_loss:,.2f}")
        print(f"止盈: ${decision.take_profit:,.2f}")
        print(f"风险收益比: {decision.risk_reward_ratio:.2f}")
        print(f"{'='*50}")

asyncio.run(complete_trading_pipeline())
```

## 项目结构

```
crypto_trading_system/
├── core/
│   ├── features/              # 特征工程层
│   │   ├── technical/         # 技术指标
│   │   ├── onchain/           # 链上指标
│   │   ├── sentiment/         # 情绪指标
│   │   └── composite/         # 组合特征
│   ├── reasoning/             # 推理层
│   │   ├── cot_engine.py      # 思维链引擎
│   │   ├── ensemble_manager.py # 多模型集成
│   │   └── consistency_validator.py # 一致性验证
│   └── decision/              # 决策层
│       ├── decision_engine.py # 决策引擎主入口
│       ├── signal_generator.py # 信号生成器
│       ├── position_manager.py # 仓位管理器
│       ├── risk_manager.py    # 风险管理器
│       ├── execution_optimizer.py # 执行优化器
│       └── order_manager.py   # 订单管理器
├── config/
│   ├── features_config.yaml   # 特征工程配置
│   ├── reasoning_config.yaml  # 推理层配置
│   └── decision_config.yaml   # 决策层配置
├── tests/
│   ├── test_features_layer.py # 特征工程测试
│   ├── test_reasoning_layer.py # 推理层测试
│   └── test_decision_layer.py # 决策层测试
├── docs/
│   ├── system_architecture.md # 系统架构文档
│   └── reasoning_layer_architecture.md # 推理层文档
├── examples/
│   └── reasoning_layer_example.py # 使用示例
├── FEATURE_LAYER_SUMMARY.md   # 特征工程层总结
├── REASONING_LAYER_SUMMARY.md # 推理层总结
├── DECISION_LAYER_SUMMARY.md  # 决策层总结
└── README.md                  # 本文件
```

## 性能指标

| 模块 | 延迟 | 特征/输出 |
|------|------|----------|
| 特征工程层 | 12-26ms | 94维特征 |
| 推理层 | 20-60ms | 交易信号 |
| 决策层 | 5-10ms | 交易决策 |
| **总计** | **37-96ms** | 完整决策 |

## 测试

```bash
# 运行特征工程层测试
python tests/test_features_layer.py

# 运行推理层测试
python tests/test_reasoning_layer.py

# 运行决策层测试
python tests/test_decision_layer.py
```

## 文档

- [系统架构文档](docs/system_architecture.md)
- [推理层架构文档](docs/reasoning_layer_architecture.md)
- [特征工程层总结](FEATURE_LAYER_SUMMARY.md)
- [推理层总结](REASONING_LAYER_SUMMARY.md)
- [决策层总结](DECISION_LAYER_SUMMARY.md)

## 下一步开发

### 待完成模块

1. **反馈层 (Feedback Layer)**
   - 性能分析
   - 在线学习
   - RLHF (基于人类反馈的强化学习)

2. **输入层优化**
   - 实时数据流
   - 数据质量检查
   - 异常处理

3. **生产环境部署**
   - 交易所API集成
   - 实时监控
   - 容错处理

## 贡献

欢迎提交 Issue 和 Pull Request!

## 许可证

MIT License

---

**版本**: 1.0.0  
**最后更新**: 2026-02-19
