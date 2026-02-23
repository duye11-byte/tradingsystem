# OpenClaw 5层交易系统架构

## 系统概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        OpenClaw Crypto Trading System                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Layer 5: 反馈层 (Feedback Layer)                                  │   │
│  │  - 性能分析、在线学习、RLHF训练、反馈存储                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    ▲                                     │
│                                    │                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Layer 4: 决策层 (Decision Layer)                                  │   │
│  │  - 信号生成、仓位管理、风险管理、执行优化、订单管理                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    ▲                                     │
│                                    │                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Layer 3: 特征工程层 (Feature Engineering Layer)                   │   │
│  │  - 技术指标、链上数据、情绪分析、复合特征                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    ▲                                     │
│                                    │                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Layer 2: 推理层 (Reasoning Layer)                                 │   │
│  │  - CoT推理、多模型集成、一致性验证                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    ▲                                     │
│                                    │                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Layer 1: 输入层 (Input Layer)  ⭐ 当前实现                         │   │
│  │  - 多源数据融合、实时数据流、数据验证、容错机制                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## 各层职责

### Layer 1: 输入层 (Input Layer)
**职责**: 数据采集与预处理

**输入**: 多个免费数据源的原始数据
**输出**: 标准化的 `MarketData` 对象

**核心组件**:
- `InputEngine`: 输入引擎主入口
- `DataAggregator`: 数据聚合器
- `DataValidator`: 数据验证器
- 多个数据源客户端

**数据流**:
```
[Binance] ──┐
[CoinGecko]─┼──> DataAggregator ──> DataValidator ──> MarketData
[Dune]     ─┤                                    │
[DeFiLlama]─┤                                    └──> Layer 2
[...]       ──┘
```

### Layer 2: 推理层 (Reasoning Layer)
**职责**: AI 推理与信号生成

**输入**: `MarketData` (来自 Layer 1)
**输出**: `TradingSignal` (交易信号)

**核心组件**:
- `CoTEngine`: Chain-of-Thought 推理引擎
- `EnsembleManager`: 多模型集成管理器
- `ConsistencyValidator`: 一致性验证器

**数据流**:
```
MarketData ──> CoTEngine ──> TradingSignal
                    │
                    ├──> 趋势识别
                    ├──> 支撑阻力分析
                    ├──> 动量分析
                    ├──> 链上数据分析
                    ├──> 市场情绪分析
                    ├──> 风险评估
                    └──> 综合判断
```

### Layer 3: 特征工程层 (Feature Engineering Layer)
**职责**: 特征提取与工程

**输入**: `MarketData` (来自 Layer 1)
**输出**: `FeatureVector` (特征向量)

**核心组件**:
- `TechnicalIndicators`: 技术指标
- `OnChainMetrics`: 链上指标
- `SentimentFeatures`: 情绪特征
- `CompositeFeatures`: 复合特征

**数据流**:
```
MarketData ──┬──> TechnicalIndicators ──┐
             ├──> OnChainMetrics ───────┼──> FeatureVector
             ├──> SentimentFeatures ────┤
             └──> CompositeFeatures ────┘
```

### Layer 4: 决策层 (Decision Layer)
**职责**: 交易决策与执行

**输入**: `TradingSignal` (来自 Layer 2)
**输出**: `Order` (订单)

**核心组件**:
- `SignalGenerator`: 信号生成器
- `PositionManager`: 仓位管理器
- `RiskManager`: 风险管理器
- `ExecutionOptimizer`: 执行优化器
- `OrderManager`: 订单管理器

**数据流**:
```
TradingSignal ──> SignalGenerator ──> Decision
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
            PositionManager ──> RiskManager ──> ExecutionOptimizer
                                                        │
                                                        ▼
                                                  OrderManager
                                                        │
                                                        ▼
                                                      Order
```

### Layer 5: 反馈层 (Feedback Layer)
**职责**: 性能分析与模型优化

**输入**: `TradeResult` (交易结果)
**输出**: `ModelUpdate` (模型更新)

**核心组件**:
- `PerformanceAnalyzer`: 性能分析器
- `OnlineLearner`: 在线学习器
- `RLHFTrainer`: RLHF训练器
- `FeedbackStore`: 反馈存储

**数据流**:
```
TradeResult ──┬──> PerformanceAnalyzer ──┐
              ├──> OnlineLearner ────────┼──> ModelUpdate
              └──> RLHFTrainer ──────────┘
```

## 层间接口

### Layer 1 → Layer 2
```python
# Layer 1 输出
market_data = MarketData(
    symbol="BTCUSDT",
    timestamp=datetime.now(),
    price_data=PriceData(...),
    orderbook_data=OrderBookData(...),
    exchange_flows=[ExchangeFlowData(...)],
    fear_greed=FearGreedIndex(...),
    funding_rate=FundingRateData(...),
    recent_news=[NewsData(...)],
    # ...
)

# Layer 2 接收
class ReasoningEngine:
    async def process(self, market_data: MarketData) -> TradingSignal:
        # 推理逻辑
        pass
```

### Layer 2 → Layer 4
```python
# Layer 2 输出
trading_signal = TradingSignal(
    symbol="BTCUSDT",
    signal_type=SignalType.BUY,
    confidence=0.85,
    entry_price=Decimal("45000"),
    stop_loss=Decimal("43000"),
    take_profit=Decimal("50000"),
    reasoning="...",
    # ...
)

# Layer 4 接收
class DecisionEngine:
    async def process_signal(self, signal: TradingSignal) -> Decision:
        # 决策逻辑
        pass
```

### Layer 4 → Layer 5
```python
# Layer 4 输出
trade_result = TradeResult(
    trade_id="...",
    symbol="BTCUSDT",
    entry_price=Decimal("45000"),
    exit_price=Decimal("48000"),
    pnl=Decimal("3000"),
    pnl_percent=Decimal("0.067"),
    # ...
)

# Layer 5 接收
class FeedbackEngine:
    async def process_trade(self, trade_result: TradeResult):
        # 反馈逻辑
        pass
```

## 数据流转

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  Input  │────>│ Reason  │────>│ Feature │────>│ Decision│────>│ Feedback│
│  Layer  │     │  Layer  │     │  Layer  │     │  Layer  │     │  Layer  │
└─────────┘     └─────────┘     └─────────┘     └─────────┘     └─────────┘
     │                │                │                │                │
     │                │                │                │                │
     ▼                ▼                ▼                ▼                ▼
MarketData    TradingSignal   FeatureVector      Order         ModelUpdate
```

## 集成示例

### 完整交易流程

```python
import asyncio
from datetime import datetime

# 导入各层
from openclaw_input_layer.core.input import InputEngine
from crypto_trading_system.core.reasoning import ReasoningEngine
from crypto_trading_system_2.core.features import FeatureEngine
from crypto_trading_system_3.core.decision import DecisionEngine
from crypto_trading_system_4.core.feedback import FeedbackEngine

async def trading_loop():
    """完整交易循环"""
    
    # 初始化各层
    input_engine = InputEngine()
    reasoning_engine = ReasoningEngine()
    feature_engine = FeatureEngine()
    decision_engine = DecisionEngine()
    feedback_engine = FeedbackEngine()
    
    # 启动
    await input_engine.start()
    
    symbol = "BTCUSDT"
    
    while True:
        try:
            # Layer 1: 获取市场数据
            input_result = await input_engine.get_market_data(symbol)
            if not input_result.success:
                logger.error(f"获取数据失败: {input_result.message}")
                await asyncio.sleep(60)
                continue
            
            market_data = input_result.market_data
            
            # Layer 3: 特征工程
            features = await feature_engine.extract_features(market_data)
            
            # Layer 2: AI 推理
            signal = await reasoning_engine.process(market_data, features)
            
            if signal.confidence < 0.6:
                logger.info(f"信号置信度不足: {signal.confidence}")
                await asyncio.sleep(60)
                continue
            
            # Layer 4: 决策与执行
            decision = await decision_engine.process_signal(signal)
            
            if decision.action == Action.EXECUTE:
                order = await decision_engine.create_order(decision)
                execution_result = await decision_engine.execute_order(order)
                
                # Layer 5: 反馈
                if execution_result:
                    await feedback_engine.process_trade(execution_result)
            
            await asyncio.sleep(60)
        
        except Exception as e:
            logger.error(f"交易循环错误: {e}")
            await asyncio.sleep(60)

# 运行
asyncio.run(trading_loop())
```

## 扩展性设计

### 添加新数据源

1. 在 `core/input/sources/` 下创建新的客户端
2. 实现标准接口（`start`, `stop`, 数据获取方法）
3. 在 `DataAggregator` 中注册新客户端

### 添加新特征

1. 在 `core/features/` 下创建新的特征提取器
2. 实现 `extract` 方法
3. 在 `FeatureEngine` 中注册

### 添加新策略

1. 在 `core/decision/strategies/` 下创建新的策略
2. 实现 `evaluate` 方法
3. 在 `DecisionEngine` 中注册

## 性能考虑

### 延迟优化
- WebSocket 连接优先于 REST API
- 本地缓存减少重复请求
- 异步并行数据获取

### 可靠性保障
- 多数据源回退机制
- 自动重连和错误恢复
- 数据验证和异常检测

### 可扩展性
- 模块化设计，各层独立
- 插件化数据源和策略
- 配置驱动，无需修改代码
