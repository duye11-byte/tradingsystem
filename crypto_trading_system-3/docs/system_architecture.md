# 系统架构文档

## 概述

OpenClaw 加密货币交易决策系统是一个多层架构的量化交易系统，包含以下核心层次：

1. **输入层 (Input Layer)** - 数据采集和整合
2. **特征工程层 (Feature Engineering Layer)** - 特征提取和转换
3. **推理层 (Reasoning Layer)** - AI 推理和决策
4. **决策层 (Decision Layer)** - 交易信号生成和执行
5. **反馈层 (Feedback Layer)** - 性能监控和学习

## 系统架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OpenClaw Crypto Trading System                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Layer 1: Input Layer (输入层)                                              │
│  ─────────────────────────────                                              │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │  Exchange   │  │  Onchain    │  │  Sentiment  │  │   Macro     │       │
│  │    APIs     │  │   APIs      │  │   APIs      │  │   APIs      │       │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘       │
│         │                │                │                │               │
│         └────────────────┴────────────────┴────────────────┘               │
│                                   │                                         │
│                          ┌────────┴────────┐                               │
│                          │  Data Fetcher   │                               │
│                          │  (数据获取器)    │                               │
│                          └────────┬────────┘                               │
│                                   │                                         │
│                          ┌────────┴────────┐                               │
│                          │Input Integrator │                               │
│                          │  (输入整合器)    │                               │
│                          └────────┬────────┘                               │
└───────────────────────────────────┼─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Layer 2: Feature Engineering Layer (特征工程层)                            │
│  ───────────────────────────────────────────────                            │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     FeatureEngineering (特征工程主入口)              │   │
│  └───────────────────────────────┬─────────────────────────────────────┘   │
│                                  │                                          │
│      ┌─────────────┬─────────────┼─────────────┬─────────────┐             │
│      │             │             │             │             │             │
│      ▼             ▼             ▼             ▼             ▼             │
│  ┌────────┐  ┌────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐     │
│  │Technical│  │Onchain │  │Sentiment │  │Composite │  │   Feature    │     │
│  │Indicators│  │Metrics │  │Analyzer │  │Composer │  │   Storage    │     │
│  │  (33)   │  │  (24)  │  │  (19)   │  │  (13)   │  │              │     │
│  └────┬────┘  └────┬───┘  └────┬─────┘  └────┬─────┘  └──────┬───────┘     │
│       │            │           │             │               │             │
│       └────────────┴───────────┴─────────────┘               │             │
│                              │                               │             │
│                              ▼                               │             │
│                    ┌──────────────────┐                      │             │
│                    │   FeatureSet     │──────────────────────┘             │
│                    │   (94 features)  │                                    │
│                    └────────┬─────────┘                                    │
└─────────────────────────────┼───────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Layer 3: Reasoning Layer (推理层)                                          │
│  ─────────────────────────────────                                          │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      ReasoningEngine (推理引擎主入口)                │   │
│  └───────────────────────────────┬─────────────────────────────────────┘   │
│                                  │                                          │
│      ┌───────────────────────────┼───────────────────────────┐             │
│      │                           │                           │             │
│      ▼                           ▼                           ▼             │
│  ┌──────────────┐      ┌──────────────────┐      ┌──────────────────────┐  │
│  │  CoT Engine  │      │ Ensemble Manager │      │Consistency Validator │  │
│  │  (思维链)    │      │   (多模型集成)    │      │   (一致性验证)        │  │
│  │              │      │                  │      │                      │  │
│  │ 7-step       │      │ 4 Expert Agents  │      │ Multi-sampling       │  │
│  │ reasoning    │      │ Weighted Voting  │      │ Verification         │  │
│  └──────┬───────┘      └────────┬─────────┘      └──────────┬───────────┘  │
│         │                       │                           │              │
│         └───────────────────────┼───────────────────────────┘              │
│                                 │                                          │
│                                 ▼                                          │
│                    ┌────────────────────┐                                  │
│                    │   TradingSignal    │                                  │
│                    │   (交易信号)        │                                  │
│                    │                    │                                  │
│                    │ - Signal Type      │                                  │
│                    │ - Confidence       │                                  │
│                    │ - Entry/SL/TP      │                                  │
│                    │ - Position Size    │                                  │
│                    └────────┬───────────┘                                  │
└─────────────────────────────┼───────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Layer 4: Decision Layer (决策层) - 待开发                                  │
│  ─────────────────────────────────────────                                  │
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                      │
│  │Signal Gen.   │  │Position Mgmt │  │Execution Opt │                      │
│  │(信号生成)    │  │(仓位管理)    │  │(执行优化)    │                      │
│  └──────────────┘  └──────────────┘  └──────────────┘                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Layer 5: Feedback Layer (反馈层) - 待开发                                  │
│  ─────────────────────────────────────────                                  │
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                      │
│  │Performance   │  │Online Learn. │  │    RLHF      │                      │
│  │Analysis      │  │              │  │              │                      │
│  └──────────────┘  └──────────────┘  └──────────────┘                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 数据流

```
┌─────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Data   │────▶│   Feature   │────▶│  Reasoning  │────▶│  Decision   │
│ Sources │     │ Engineering │     │   Engine    │     │   Engine    │
└─────────┘     └─────────────┘     └─────────────┘     └─────────────┘
     │                 │                   │                   │
     │                 │                   │                   │
     ▼                 ▼                   ▼                   ▼
  Raw Data        Features           Trading Signal      Executed Trade
  (OHLCV)         (94 dims)          (Signal + Conf.)    (Order + Result)
```

## 模块关系

### 输入层 → 特征工程层

```
InputIntegrator.output ──▶ FeatureEngineering.extract_features()
    │
    ├── OHLCV DataFrame ──▶ TechnicalIndicators.calculate_all()
    ├── Onchain Data ─────▶ OnchainMetrics.calculate_all()
    └── Sentiment Data ───▶ SentimentAnalyzer.calculate_all()
```

### 特征工程层 → 推理层

```
FeatureSet.to_dict() ──▶ ReasoningEngine.reason()
    │
    ├── technical.rsi_14 ──▶ CoT Step 3: Momentum Analysis
    ├── onchain.exchange_netflow ──▶ CoT Step 4: Onchain Analysis
    ├── sentiment.fear_greed_index ──▶ CoT Step 5: Sentiment Analysis
    └── composite.composite_momentum ──▶ Ensemble Voting
```

## 性能指标

| 层次 | 延迟 | 输出 |
|------|------|------|
| 输入层 | 50-200ms | 原始数据 |
| 特征工程层 | 12-26ms | 94维特征 |
| 推理层 | 20-60ms | 交易信号 |
| **总计** | **82-286ms** | 完整决策 |

## 特征维度

```
Feature Vector (94 dimensions):
├─ OHLCV (5)
│  ├─ open
│  ├─ high
│  ├─ low
│  ├─ close
│  └─ volume
│
├─ Technical (33)
│  ├─ Trend: sma_20, sma_50, ema_12, ema_26, ...
│  ├─ Momentum: rsi_14, macd, stochastic_k, ...
│  ├─ Volatility: bb_upper, bb_lower, atr_14, ...
│  ├─ Volume: obv, volume_ratio, mfi_14, ...
│  └─ Price Change: price_change_1h, price_change_1d, ...
│
├─ Onchain (24)
│  ├─ Exchange Flow: inflow, outflow, netflow, ...
│  ├─ Whale Activity: tx_count, volume, accumulation, ...
│  ├─ Network: active_addresses, tx_count, ...
│  ├─ Supply: on_exchanges_pct, lth_supply, ...
│  └─ Network Health: hash_rate, difficulty, ...
│
├─ Sentiment (19)
│  ├─ Fear & Greed: index, classification, change, ...
│  ├─ Social: sentiment, twitter, reddit, ...
│  ├─ News: sentiment, volume, ...
│  ├─ Futures: funding_rate, long_short_ratio, ...
│  └─ Composite: composite_sentiment, momentum, ...
│
└─ Composite (13)
   ├─ PCA: pc1, pc2, pc3, ...
   ├─ TS Decomposition: trend, seasonal, residual, ...
   ├─ Interactions: price_volume, momentum_sentiment, ...
   └─ Composite Indicators: momentum, volatility, liquidity
```

## 扩展性

### 添加新的技术指标

```python
# 在 technical_indicators.py 中添加新方法
def custom_indicator(self, close: np.ndarray, period: int) -> np.ndarray:
    """自定义指标"""
    # 实现逻辑
    return result
```

### 添加新的特征类别

```python
# 在 feature_types.py 中添加新的特征数据类
@dataclass
class NewFeatures:
    """新特征集合"""
    feature1: float = 0.0
    feature2: float = 0.0
```

## 部署架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Production Environment                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Data       │  │  Feature    │  │  Reasoning  │        │
│  │  Collector  │──▶│  Engine     │──▶│  Engine     │        │
│  │  (Cron)     │  │  (Service)  │  │  (Service)  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│         │                │                │                │
│         ▼                ▼                ▼                │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              Redis Cache / Feature Store             │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 监控和日志

```
┌─────────────────────────────────────────────────────────────┐
│                      Monitoring Stack                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Prometheus │  │  Grafana    │  │   ELK       │        │
│  │  (Metrics)  │  │ (Dashboard) │  │   (Logs)    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                             │
│  Metrics:                                                   │
│  - Feature extraction latency                               │
│  - Reasoning latency                                        │
│  - Signal accuracy                                          │
│  - System throughput                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

**版本**: 1.0.0  
**最后更新**: 2026-02-19
