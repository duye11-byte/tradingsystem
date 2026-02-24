# OpenClaw Trading System Web UI

OpenClaw 5层交易系统的Web管理界面，提供完整的交易监控和管理功能。

## 在线演示

**访问地址**: https://6ffndykqj24qc.ok.kimi.link

## 功能特性

### 1. 仪表盘 (Dashboard)
- 实时盈亏统计
- 胜率、夏普比率等关键指标
- 盈亏趋势图表
- 市场行情概览
- 最新交易信号
- 当前持仓列表

### 2. 市场行情 (Market)
- 多交易对实时监控
- 实时价格图表
- 订单簿深度展示
- 市场情绪指标（恐惧贪婪指数、资金费率、多空比）

### 3. 交易信号 (Signals)
- AI生成的交易信号列表
- 信号详情（置信度、入场价、止损止盈）
- AI推理依据展示
- 信号状态管理

### 4. 订单管理 (Orders)
- 所有订单列表
- 订单状态筛选（待执行、已成交、已取消）
- 订单详情查看
- 成交进度追踪

### 5. 仓位管理 (Positions)
- 当前持仓列表
- 盈亏统计
- 资金分配图表
- 强平风险提示
- 一键平仓功能

### 6. 风险管理 (Risk)
- 最大回撤限制
- 单笔交易风险设置
- 仓位大小限制
- 杠杆倍数限制
- 止损止盈默认设置
- 风险警报配置

### 7. 数据源 (DataSources)
- 多数据源管理（价格、链上、情绪、新闻）
- 数据源健康状态监控
- 延迟和错误率统计
- 数据源启用/停用控制

### 8. 系统日志 (Logs)
- 实时日志查看
- 日志级别筛选（信息、警告、错误、调试）
- 日志导出功能
- 按来源筛选

### 9. 设置 (Settings)
- Binance API 配置
- 默认交易设置
- 通知设置
- 外观主题设置

## 技术栈

- **框架**: React + TypeScript
- **构建工具**: Vite
- **UI组件**: shadcn/ui
- **样式**: Tailwind CSS
- **图表**: Recharts
- **路由**: React Router

## 本地开发

```bash
# 安装依赖
npm install

# 启动开发服务器
npm run dev

# 构建生产版本
npm run build
```

## 项目结构

```
src/
├── components/
│   ├── layout/
│   │   ├── Sidebar.tsx    # 侧边导航
│   │   └── Header.tsx     # 顶部导航
│   └── ui/                # shadcn/ui 组件
├── pages/
│   ├── Dashboard.tsx      # 仪表盘
│   ├── Market.tsx         # 市场行情
│   ├── Signals.tsx        # 交易信号
│   ├── Orders.tsx         # 订单管理
│   ├── Positions.tsx      # 仓位管理
│   ├── Risk.tsx           # 风险管理
│   ├── DataSources.tsx    # 数据源
│   ├── Logs.tsx           # 系统日志
│   └── Settings.tsx       # 设置
├── lib/
│   ├── mockData.ts        # 模拟数据
│   └── utils.ts           # 工具函数
├── types/
│   └── index.ts           # TypeScript类型定义
└── App.tsx                # 主应用组件
```

## 与5层系统集成

本Web UI与OpenClaw 5层交易系统集成：

- **Layer 1 (输入层)**: 数据源配置页面管理所有数据源
- **Layer 2 (推理层)**: 交易信号页面展示AI推理结果
- **Layer 3 (特征层)**: 市场情绪指标展示
- **Layer 4 (决策层)**: 订单和仓位管理
- **Layer 5 (反馈层)**: 盈亏统计和性能监控

## 许可证

MIT License
