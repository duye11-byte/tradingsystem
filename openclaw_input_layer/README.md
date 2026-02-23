# OpenClaw 输入层 (Input Layer)

## 第1层：多源数据融合与实时数据流处理

OpenClaw 交易系统的数据入口层，整合多个免费数据源，为上层提供标准化、高质量的市场数据。

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    OpenClaw 决策中枢                             │
└──────────────────┬──────────────────────────────────────────────┘
                   │
    ┌──────────────┼──────────────┬──────────────┐
    ▼              ▼              ▼              ▼
┌─────────┐  ┌─────────┐   ┌─────────┐   ┌─────────┐
│ 价格数据 │  │ 链上数据 │   │ 情绪指标 │   │ 事件驱动 │
│(Tier 1) │  │(Tier 2) │   │(Tier 3) │   │(Tier 4) │
└────┬────┘  └────┬────┘   └────┬────┘   └────┬────┘
     │            │             │             │
CoinGecko    Dune Analytics   Alternative.me  CryptoPanic
Binance      DeFiLlama       Santiment(部分)  NewsData.io
Coinbase     Arkham(免费)    Coinalyze       Reddit
```

## 功能特性

### 1. 实时价格数据
- **Binance API**: WebSocket 实时流 + REST API
  - 实时 K 线、订单簿、成交数据
  - 无限制 WebSocket，IP 限流 1200 req/min
- **CoinGecko**: 低频市场数据
  - 市值、完全稀释估值、开发活动
  - 免费档 10-30 calls/min

### 2. 链上数据分析
- **Dune Analytics**: SQL 查询链上数据
  - 交易所资金流向
  - 持有者行为分析
  - 稳定币储备
- **DeFiLlama**: TVL 和收益率数据
  - 完全免费，无 key 限制
  - 多链 TVL 追踪
- **Arkham**: 实体标记和聪明钱追踪
  - 免费版有限查询
  - 鲸鱼钱包监控

### 3. 市场情绪指标
- **Alternative.me**: 恐惧贪婪指数
  - 完全免费，无限制
  - 日更数据
- **Coinalyze**: 资金费率和清算数据
  - 实时资金费率
  - 多空比、持仓量
- **Binance**: 期货情绪数据
  - 多空账户比
  - 吃单量统计

### 4. 新闻事件驱动
- **CryptoPanic**: 新闻聚合
  - 免费 API key
  - 情绪标签（看涨/看跌）
- **NewsData.io**: 新闻 API
  - 免费版 200 req/day
- **Reddit**: 社区情绪
  - 60 req/min
  - 多 subreddit 监控

## 快速开始

### 安装依赖

```bash
pip install aiohttp websockets pyyaml
# 可选
pip install praw  # Reddit 支持
```

### 基础使用

```python
import asyncio
from core.input import InputEngine, InputEngineConfig

async def main():
    # 创建引擎
    config = InputEngineConfig(
        default_symbols=['BTCUSDT', 'ETHUSDT'],
        default_data_types=['price', 'sentiment']
    )
    
    async with InputEngine(config) as engine:
        # 获取市场数据
        result = await engine.get_market_data('BTCUSDT')
        
        if result.success:
            data = result.market_data
            print(f"BTC价格: ${data.price_data.close_price}")
            print(f"恐惧贪婪指数: {data.fear_greed.value}")
            print(f"资金费率: {data.funding_rate.funding_rate}")
        
        # 获取综合情绪
        sentiment = await engine.get_composite_sentiment('BTCUSDT')
        print(f"综合情绪: {sentiment['composite_score']:.1f}/100")

asyncio.run(main())
```

### 实时数据流

```python
async with InputEngine() as engine:
    # 实时价格流
    async for price in engine.stream_prices('BTCUSDT', interval='1m'):
        print(f"价格更新: ${price.close_price}")
```

## 项目结构

```
openclaw_input_layer/
├── core/
│   └── input/
│       ├── __init__.py           # 模块导出
│       ├── input_types.py        # 类型定义
│       ├── input_engine.py       # 输入引擎主类
│       ├── data_aggregator.py    # 数据聚合器
│       ├── data_validator.py     # 数据验证器
│       └── sources/
│           ├── price/            # 价格数据源
│           │   ├── binance_client.py
│           │   └── coingecko_client.py
│           ├── onchain/          # 链上数据源
│           │   ├── dune_client.py
│           │   ├── defillama_client.py
│           │   └── arkham_client.py
│           ├── sentiment/        # 情绪数据源
│           │   ├── alternative_me_client.py
│           │   ├── coinalyze_client.py
│           │   └── binance_sentiment_client.py
│           └── news/             # 新闻数据源
│               ├── cryptopanic_client.py
│               ├── newsdata_client.py
│               └── reddit_client.py
├── config/
│   └── input_config.yaml         # 配置文件
├── examples/
│   └── basic_usage.py            # 使用示例
├── tests/
│   └── test_input_layer.py       # 测试脚本
└── README.md                     # 本文件
```

## 配置说明

编辑 `config/input_config.yaml` 配置数据源：

```yaml
engine:
  mode: hybrid  # realtime, polling, hybrid, backtest
  default_symbols: ['BTCUSDT', 'ETHUSDT']
  cache:
    enabled: true
    ttl: 60

price_sources:
  binance:
    enabled: true
    api_key: ""  # 可选
    api_secret: ""  # 可选
  
  coingecko:
    enabled: true
    # 免费版无需 API key

sentiment_sources:
  alternative_me:
    enabled: true
  
  coinalyze:
    enabled: true

news_sources:
  cryptopanic:
    enabled: true
    api_key: "your_api_key"  # 免费注册获取
```

## 数据质量保障

### 1. 数据验证
- 价格范围检查
- 时间戳有效性
- OHLC 逻辑验证
- 订单簿平衡性

### 2. 异常检测
- Z-score 异常检测
- IQR 异常检测
- MAD 异常检测

### 3. 容错机制
- 多数据源回退
- 自动重连
- 缓存机制

## 免费数据源限制与应对

| 限制 | 解决方案 |
|------|----------|
| Binance IP 限流 | WebSocket 为主，REST 为辅 |
| CoinGecko 频率限制 | 本地缓存 + 保守请求间隔 |
| Dune 查询次数 | 缓存结果，定时刷新 |
| 新闻 API 限额 | 优先级：CryptoPanic > RSS |

## 测试

```bash
# 运行所有测试
python tests/test_input_layer.py

# 运行示例
python examples/basic_usage.py
```

## 与上层集成

输入层为第2层（推理层）提供标准化的 `MarketData`：

```python
# 第2层接收的数据格式
market_data = MarketData(
    symbol="BTCUSDT",
    timestamp=datetime.now(),
    price_data=PriceData(...),
    orderbook_data=OrderBookData(...),
    fear_greed=FearGreedIndex(...),
    funding_rate=FundingRateData(...),
    recent_news=[NewsData(...)],
    # ... 更多数据
)
```

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 PR！
