# 推理层 (Reasoning Layer)

## 简介

推理层是加密货币交易决策系统的核心智能组件，负责将原始市场数据和特征转换为可操作的交易信号。该层采用先进的 AI 推理技术，包括 Chain-of-Thought 推理、多模型集成和自我一致性验证。

## 核心特性

- ✅ **Chain-of-Thought 推理** - 7步逐步分析市场状况
- ✅ **多模型集成** - 4个专家代理协同决策
- ✅ **自我一致性验证** - 多次采样确保结果可靠
- ✅ **流式推理** - 实时查看推理进度
- ✅ **批量推理** - 高效处理多个交易对
- ✅ **性能监控** - 完整的性能指标和统计

## 快速开始

### 安装依赖

```bash
pip install numpy pyyaml
```

### 基本使用

```python
import asyncio
from core.reasoning import ReasoningEngine

async def main():
    # 创建推理引擎
    engine = ReasoningEngine()
    
    # 准备数据
    symbol = "BTC/USDT"
    features = {
        'close': 46000.0,
        'rsi_14': 62.5,
        'macd': 150.5,
        # ... 更多特征
    }
    market_data = {...}
    
    # 执行推理
    result = await engine.reason(symbol, market_data, features)
    
    if result.success:
        signal = result.signal
        print(f"信号: {signal.signal.value}")
        print(f"置信度: {signal.confidence:.1%}")
        print(f"建议仓位: {signal.position_size_ratio:.0%}")

asyncio.run(main())
```

## 文件结构

```
core/reasoning/
├── __init__.py              # 模块入口
├── reasoning_engine.py      # 推理引擎主类
├── cot_engine.py            # Chain-of-Thought 引擎
├── ensemble_manager.py      # 多模型集成管理器
├── consistency_validator.py # 一致性验证器
├── reasoning_types.py       # 类型定义
└── README.md                # 本文件
```

## 架构概览

```
┌─────────────────────────────────────────┐
│         ReasoningEngine                 │
│         (推理引擎主入口)                 │
└────────────┬────────────────────────────┘
             │
    ┌────────┼────────┐
    │        │        │
    ▼        ▼        ▼
┌───────┐ ┌───────┐ ┌───────────┐
│  CoT  │ │Ensemble│ │Consistency│
│Engine │ │Manager │ │ Validator │
└───┬───┘ └───┬───┘ └─────┬─────┘
    │         │           │
    └─────────┴───────────┘
              │
              ▼
    ┌──────────────────┐
    │  TradingSignal   │
    │   (交易信号)      │
    └──────────────────┘
```

## 推理流程

1. **特征输入** - 接收特征工程层输出的特征数据
2. **CoT 推理** - 执行7步逐步分析
3. **集成预测** - 4个专家代理并行预测
4. **共识计算** - 加权投票生成共识结果
5. **一致性验证** - 多次采样验证结果可靠性
6. **信号生成** - 输出最终交易信号

## 配置示例

```yaml
reasoning_engine:
  cot_config:
    max_steps: 7
    min_confidence: 0.6
    
  ensemble_config:
    consensus_threshold: 0.6
    agents:
      - name: technical_analyst
        weight: 0.35
      - name: onchain_analyst
        weight: 0.25
      - name: sentiment_analyst
        weight: 0.20
      - name: macro_analyst
        weight: 0.20
  
  consistency_config:
    num_samples: 5
    consistency_threshold: 0.7
```

## 性能指标

| 指标 | 数值 |
|------|------|
| 总延迟 | 20-60ms |
| 成功率 | > 95% |
| 一致性通过率 | > 80% |
| 平均置信度 | > 70% |

## API 参考

### ReasoningEngine

#### `async reason(symbol, market_data, features, skip_validation=False)`

执行完整推理流程。

**参数**:
- `symbol` (str): 交易对符号
- `market_data` (dict): 市场数据
- `features` (dict): 特征数据
- `skip_validation` (bool): 是否跳过一致性验证

**返回**: `ReasoningResult`

#### `async reason_stream(symbol, market_data, features)`

流式推理，逐步返回结果。

**返回**: `AsyncGenerator[Dict, None]`

#### `async batch_reason(tasks, max_concurrency=5)`

批量推理。

**参数**:
- `tasks` (List[dict]): 任务列表
- `max_concurrency` (int): 最大并发数

**返回**: `List[ReasoningResult]`

#### `get_performance_stats()`

获取性能统计。

**返回**: `Dict[str, Any]`

## 测试

```bash
# 运行测试
python tests/test_reasoning_layer.py

# 运行示例
python examples/reasoning_layer_example.py
```

## 扩展开发

### 添加自定义代理

```python
custom_agent = {
    'name': 'my_agent',
    'specialization': '自定义分析',
    'model_config': {...},
    'prompt_template': '...',
    'weight': 0.15,
    'confidence_threshold': 0.6
}

engine.add_custom_agent(custom_agent)
```

### 调整代理权重

```python
engine.update_agent_weight('technical_analyst', 0.40)
```

## 文档

- [架构文档](../../docs/reasoning_layer_architecture.md) - 详细架构设计
- [使用示例](../../examples/reasoning_layer_example.py) - 代码示例

## 贡献

欢迎提交 Issue 和 Pull Request!

## 许可证

MIT License
