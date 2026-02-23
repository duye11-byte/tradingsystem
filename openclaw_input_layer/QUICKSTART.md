# OpenClaw 输入层快速启动指南

## 1. 安装依赖

```bash
cd /mnt/okcomputer/output/openclaw_input_layer
pip install -r requirements.txt
```

## 2. 运行测试

```bash
# 运行所有测试
python tests/test_input_layer.py

# 运行单个测试
python -m pytest tests/test_input_layer.py::TestPriceSources::test_binance_client -v
```

## 3. 运行示例

```bash
# 基础使用示例
python examples/basic_usage.py

# 5层集成示例
python examples/integration_with_upper_layers.py
```

## 4. 快速代码示例

### 获取市场数据

```python
import asyncio
from core.input import InputEngine

async def main():
    async with InputEngine() as engine:
        # 获取BTC市场数据
        result = await engine.get_market_data('BTCUSDT')
        
        if result.success:
            data = result.market_data
            print(f"BTC价格: ${data.price_data.close_price}")
            print(f"恐惧贪婪指数: {data.fear_greed.value}")
        else:
            print(f"获取失败: {result.message}")

asyncio.run(main())
```

### 实时数据流

```python
async with InputEngine() as engine:
    async for price in engine.stream_prices('BTCUSDT'):
        print(f"价格: ${price.close_price}")
```

### 综合情绪分析

```python
async with InputEngine() as engine:
    sentiment = await engine.get_composite_sentiment('BTCUSDT')
    print(f"情绪分数: {sentiment['composite_score']:.1f}/100")
    print(f"解读: {sentiment['interpretation']}")
```

## 5. 配置数据源

编辑 `config/input_config.yaml`：

```yaml
# 启用/禁用数据源
price_sources:
  binance:
    enabled: true
  coingecko:
    enabled: true

sentiment_sources:
  alternative_me:
    enabled: true
  coinalyze:
    enabled: true

news_sources:
  cryptopanic:
    enabled: true
    api_key: "your_api_key_here"  # 免费注册获取
```

## 6. 常见问题

### Q: 某些数据源返回空数据？
A: 检查网络连接和数据源状态：
```python
health = await engine.health_check()
print(health['data_sources'])
```

### Q: 如何减少API请求？
A: 启用缓存：
```python
config = InputEngineConfig(cache_enabled=True, cache_ttl=120)
engine = InputEngine(config)
```

### Q: 如何处理速率限制？
A: 系统已内置速率限制处理，可以通过配置调整：
```yaml
price_sources:
  binance:
    rate_limit: 0.1  # 秒
```

## 7. 下一步

- 阅读 [完整文档](README.md)
- 查看 [架构设计](docs/ARCHITECTURE.md)
- 探索更多 [示例代码](examples/)
