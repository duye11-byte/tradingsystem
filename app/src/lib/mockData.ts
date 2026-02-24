import type { 
  MarketData, 
  TradingSignal, 
  Order, 
  Position, 
  RiskSettings, 
  DataSource, 
  SystemLog,
  PerformanceMetrics,
  SentimentData,
  OrderBook
} from '@/types';

export const mockMarketData: MarketData[] = [
  { symbol: 'BTCUSDT', price: 67432.50, priceChange24h: 1234.20, priceChangePercent24h: 1.86, volume24h: 28500000000, high24h: 68100.00, low24h: 65800.00, timestamp: new Date().toISOString() },
  { symbol: 'ETHUSDT', price: 3521.80, priceChange24h: 45.30, priceChangePercent24h: 1.30, volume24h: 15200000000, high24h: 3580.00, low24h: 3450.00, timestamp: new Date().toISOString() },
  { symbol: 'SOLUSDT', price: 178.45, priceChange24h: -2.15, priceChangePercent24h: -1.19, volume24h: 3200000000, high24h: 182.00, low24h: 175.50, timestamp: new Date().toISOString() },
  { symbol: 'BNBUSDT', price: 612.30, priceChange24h: 8.70, priceChangePercent24h: 1.44, volume24h: 1800000000, high24h: 618.00, low24h: 600.00, timestamp: new Date().toISOString() },
  { symbol: 'ADAUSDT', price: 0.4850, priceChange24h: 0.012, priceChangePercent24h: 2.54, volume24h: 890000000, high24h: 0.492, low24h: 0.468, timestamp: new Date().toISOString() },
  { symbol: 'DOTUSDT', price: 7.82, priceChange24h: 0.15, priceChangePercent24h: 1.96, volume24h: 450000000, high24h: 7.95, low24h: 7.62, timestamp: new Date().toISOString() },
  { symbol: 'MATICUSDT', price: 0.652, priceChange24h: -0.008, priceChangePercent24h: -1.21, volume24h: 380000000, high24h: 0.668, low24h: 0.645, timestamp: new Date().toISOString() },
  { symbol: 'LINKUSDT', price: 18.45, priceChange24h: 0.32, priceChangePercent24h: 1.77, volume24h: 520000000, high24h: 18.80, low24h: 18.00, timestamp: new Date().toISOString() },
];

export const mockTradingSignals: TradingSignal[] = [
  { id: 'sig-001', symbol: 'BTCUSDT', type: 'BUY', confidence: 0.85, entryPrice: 67200, stopLoss: 65000, takeProfit: 72000, reasoning: '极度恐慌指数(22) + 负资金费率，逆向买入机会', timestamp: new Date(Date.now() - 3600000).toISOString(), status: 'active' },
  { id: 'sig-002', symbol: 'ETHUSDT', type: 'BUY', confidence: 0.72, entryPrice: 3500, stopLoss: 3350, takeProfit: 3800, reasoning: '突破EMA20 + 稳定币流入增加', timestamp: new Date(Date.now() - 7200000).toISOString(), status: 'active' },
  { id: 'sig-003', symbol: 'SOLUSDT', type: 'SELL', confidence: 0.68, entryPrice: 180, stopLoss: 188, takeProfit: 165, reasoning: '极度贪婪(82) + 多空比过高，回调风险', timestamp: new Date(Date.now() - 10800000).toISOString(), status: 'executed' },
  { id: 'sig-004', symbol: 'BTCUSDT', type: 'HOLD', confidence: 0.55, entryPrice: 67400, stopLoss: 66000, takeProfit: 70000, reasoning: '市场情绪中性，观望', timestamp: new Date(Date.now() - 14400000).toISOString(), status: 'expired' },
  { id: 'sig-005', symbol: 'LINKUSDT', type: 'BUY', confidence: 0.78, entryPrice: 18.20, stopLoss: 17.50, takeProfit: 20.00, reasoning: '链上数据显示聪明钱流入', timestamp: new Date(Date.now() - 18000000).toISOString(), status: 'active' },
];

export const mockOrders: Order[] = [
  { id: 'ord-001', symbol: 'BTCUSDT', side: 'buy', type: 'limit', status: 'filled', price: 67200, quantity: 0.15, filledQuantity: 0.15, remainingQuantity: 0, total: 10080, timestamp: new Date(Date.now() - 7200000).toISOString(), updatedAt: new Date(Date.now() - 7100000).toISOString() },
  { id: 'ord-002', symbol: 'ETHUSDT', side: 'buy', type: 'limit', status: 'partially_filled', price: 3500, quantity: 2.5, filledQuantity: 1.2, remainingQuantity: 1.3, total: 8750, timestamp: new Date(Date.now() - 3600000).toISOString(), updatedAt: new Date(Date.now() - 1800000).toISOString() },
  { id: 'ord-003', symbol: 'SOLUSDT', side: 'sell', type: 'market', status: 'filled', price: 179.50, quantity: 15, filledQuantity: 15, remainingQuantity: 0, total: 2692.50, timestamp: new Date(Date.now() - 1800000).toISOString(), updatedAt: new Date(Date.now() - 1795000).toISOString() },
  { id: 'ord-004', symbol: 'BTCUSDT', side: 'sell', type: 'stop_loss', status: 'pending', price: 65000, quantity: 0.15, filledQuantity: 0, remainingQuantity: 0.15, total: 9750, timestamp: new Date(Date.now() - 7200000).toISOString(), updatedAt: new Date(Date.now() - 7200000).toISOString() },
  { id: 'ord-005', symbol: 'BTCUSDT', side: 'sell', type: 'take_profit', status: 'pending', price: 72000, quantity: 0.15, filledQuantity: 0, remainingQuantity: 0.15, total: 10800, timestamp: new Date(Date.now() - 7200000).toISOString(), updatedAt: new Date(Date.now() - 7200000).toISOString() },
  { id: 'ord-006', symbol: 'ADAUSDT', side: 'buy', type: 'limit', status: 'cancelled', price: 0.48, quantity: 5000, filledQuantity: 0, remainingQuantity: 5000, total: 2400, timestamp: new Date(Date.now() - 14400000).toISOString(), updatedAt: new Date(Date.now() - 14000000).toISOString() },
];

export const mockPositions: Position[] = [
  { id: 'pos-001', symbol: 'BTCUSDT', side: 'long', entryPrice: 67200, currentPrice: 67432.50, quantity: 0.15, leverage: 5, margin: 2016, pnl: 34.88, pnlPercent: 1.73, liquidationPrice: 53760, timestamp: new Date(Date.now() - 7200000).toISOString() },
  { id: 'pos-002', symbol: 'ETHUSDT', side: 'long', entryPrice: 3500, currentPrice: 3521.80, quantity: 1.2, leverage: 3, margin: 1400, pnl: 26.16, pnlPercent: 1.87, liquidationPrice: 2450, timestamp: new Date(Date.now() - 3600000).toISOString() },
  { id: 'pos-003', symbol: 'LINKUSDT', side: 'long', entryPrice: 18.20, currentPrice: 18.45, quantity: 100, leverage: 2, margin: 910, pnl: 25.00, pnlPercent: 2.75, liquidationPrice: 9.10, timestamp: new Date(Date.now() - 18000000).toISOString() },
];

export const mockRiskSettings: RiskSettings = {
  maxPositionSize: 0.15,
  maxLeverage: 10,
  stopLossPercent: 3,
  takeProfitPercent: 6,
  maxDailyLoss: 1000,
  maxDrawdown: 15,
  riskPerTrade: 2,
};

export const mockDataSources: DataSource[] = [
  { id: 'ds-001', name: 'Binance', type: 'price', status: 'active', lastUpdate: new Date(Date.now() - 5000).toISOString(), latency: 45, errorRate: 0.001, config: { websocket: true, rest: true } },
  { id: 'ds-002', name: 'CoinGecko', type: 'price', status: 'active', lastUpdate: new Date(Date.now() - 25000).toISOString(), latency: 120, errorRate: 0.005, config: { apiKey: '***', rateLimit: 25 } },
  { id: 'ds-003', name: 'Dune Analytics', type: 'onchain', status: 'active', lastUpdate: new Date(Date.now() - 300000).toISOString(), latency: 850, errorRate: 0.02, config: { queries: ['exchange_flows', 'holder_behavior'] } },
  { id: 'ds-004', name: 'DeFiLlama', type: 'onchain', status: 'active', lastUpdate: new Date(Date.now() - 180000).toISOString(), latency: 200, errorRate: 0.001, config: { endpoints: ['tvl', 'yields'] } },
  { id: 'ds-005', name: 'Alternative.me', type: 'sentiment', status: 'active', lastUpdate: new Date(Date.now() - 3600000).toISOString(), latency: 300, errorRate: 0, config: { indicator: 'fear_greed' } },
  { id: 'ds-006', name: 'Coinalyze', type: 'sentiment', status: 'active', lastUpdate: new Date(Date.now() - 60000).toISOString(), latency: 150, errorRate: 0.003, config: { metrics: ['funding', 'liquidations', 'long_short'] } },
  { id: 'ds-007', name: 'CryptoPanic', type: 'news', status: 'active', lastUpdate: new Date(Date.now() - 120000).toISOString(), latency: 400, errorRate: 0.01, config: { currencies: ['BTC', 'ETH', 'SOL'] } },
  { id: 'ds-008', name: 'Reddit', type: 'news', status: 'inactive', lastUpdate: new Date(Date.now() - 86400000).toISOString(), latency: 0, errorRate: 0, config: { subreddits: ['BitcoinMarkets', 'CryptoCurrency'] } },
];

export const mockSystemLogs: SystemLog[] = [
  { id: 'log-001', level: 'info', message: '系统启动成功', source: 'System', timestamp: new Date(Date.now() - 3600000).toISOString() },
  { id: 'log-002', level: 'info', message: 'Binance WebSocket连接成功', source: 'DataSource', timestamp: new Date(Date.now() - 3595000).toISOString() },
  { id: 'log-003', level: 'info', message: '交易信号生成: BTCUSDT BUY (置信度: 0.85)', source: 'ReasoningEngine', timestamp: new Date(Date.now() - 3600000).toISOString() },
  { id: 'log-004', level: 'warning', message: 'CoinGecko API速率限制，切换到缓存数据', source: 'DataSource', timestamp: new Date(Date.now() - 1800000).toISOString() },
  { id: 'log-005', level: 'info', message: '订单执行成功: ord-001', source: 'OrderManager', timestamp: new Date(Date.now() - 7100000).toISOString() },
  { id: 'log-006', level: 'error', message: 'Dune Analytics查询超时', source: 'DataSource', timestamp: new Date(Date.now() - 900000).toISOString() },
  { id: 'log-007', level: 'info', message: '仓位更新: BTCUSDT 多头 +$34.88', source: 'PositionManager', timestamp: new Date(Date.now() - 300000).toISOString() },
  { id: 'log-008', level: 'warning', message: '风险警报: BTCUSDT 仓位接近止损线', source: 'RiskManager', timestamp: new Date(Date.now() - 120000).toISOString() },
  { id: 'log-009', level: 'info', message: '数据缓存清理完成', source: 'CacheManager', timestamp: new Date(Date.now() - 60000).toISOString() },
  { id: 'log-010', level: 'debug', message: '心跳检测正常', source: 'System', timestamp: new Date(Date.now() - 30000).toISOString() },
];

export const mockPerformanceMetrics: PerformanceMetrics = {
  totalTrades: 156,
  winningTrades: 98,
  losingTrades: 58,
  winRate: 62.8,
  totalPnl: 12580.50,
  dailyPnl: 342.80,
  weeklyPnl: 1856.40,
  monthlyPnl: 5234.60,
  sharpeRatio: 1.85,
  maxDrawdown: 8.5,
  avgTradeDuration: 4.2,
};

export const mockSentimentData: SentimentData[] = [
  { symbol: 'BTCUSDT', fearGreedIndex: 42, fearGreedClassification: 'Fear', fundingRate: 0.0001, longShortRatio: 1.85, liquidationVolume: 1250000, timestamp: new Date().toISOString() },
  { symbol: 'ETHUSDT', fearGreedIndex: 48, fearGreedClassification: 'Neutral', fundingRate: 0.0002, longShortRatio: 2.12, liquidationVolume: 890000, timestamp: new Date().toISOString() },
  { symbol: 'SOLUSDT', fearGreedIndex: 65, fearGreedClassification: 'Greed', fundingRate: 0.0005, longShortRatio: 2.85, liquidationVolume: 450000, timestamp: new Date().toISOString() },
  { symbol: 'BNBUSDT', fearGreedIndex: 55, fearGreedClassification: 'Greed', fundingRate: 0.0003, longShortRatio: 1.95, liquidationVolume: 320000, timestamp: new Date().toISOString() },
];

export const mockOrderBook: OrderBook = {
  symbol: 'BTCUSDT',
  bids: [
    [67430, 1.25], [67425, 2.50], [67420, 3.80], [67415, 5.20], [67410, 7.50],
    [67405, 9.00], [67400, 12.50], [67395, 15.00], [67390, 18.50], [67385, 22.00],
  ],
  asks: [
    [67435, 1.10], [67440, 2.30], [67445, 3.60], [67450, 4.90], [67455, 6.80],
    [67460, 8.50], [67465, 11.00], [67470, 14.50], [67475, 17.00], [67480, 20.50],
  ],
  timestamp: new Date().toISOString(),
};

// Generate price history for charts
export function generatePriceHistory(basePrice: number, points: number = 100): { time: string; price: number }[] {
  const history: { time: string; price: number }[] = [];
  let currentPrice = basePrice * 0.95;
  const now = Date.now();
  
  for (let i = 0; i < points; i++) {
    const change = (Math.random() - 0.48) * basePrice * 0.02;
    currentPrice += change;
    history.push({
      time: new Date(now - (points - i) * 60000).toISOString(),
      price: Math.max(currentPrice, basePrice * 0.8),
    });
  }
  
  return history;
}

// Generate PnL history
export function generatePnlHistory(days: number = 30): { date: string; pnl: number; cumulative: number }[] {
  const history: { date: string; pnl: number; cumulative: number }[] = [];
  let cumulative = 0;
  const now = Date.now();
  
  for (let i = 0; i < days; i++) {
    const dailyPnl = (Math.random() - 0.4) * 500;
    cumulative += dailyPnl;
    history.push({
      date: new Date(now - (days - i) * 86400000).toISOString().split('T')[0],
      pnl: dailyPnl,
      cumulative: cumulative,
    });
  }
  
  return history;
}
