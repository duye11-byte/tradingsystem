// OpenClaw Trading System Types

export interface MarketData {
  symbol: string;
  price: number;
  priceChange24h: number;
  priceChangePercent24h: number;
  volume24h: number;
  high24h: number;
  low24h: number;
  timestamp: string;
}

export interface OrderBook {
  symbol: string;
  bids: [number, number][]; // [price, quantity]
  asks: [number, number][];
  timestamp: string;
}

export interface Trade {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  price: number;
  quantity: number;
  total: number;
  timestamp: string;
}

export interface TradingSignal {
  id: string;
  symbol: string;
  type: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  entryPrice: number;
  stopLoss: number;
  takeProfit: number;
  reasoning: string;
  timestamp: string;
  status: 'active' | 'executed' | 'expired' | 'cancelled';
}

export interface Order {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit' | 'stop_loss' | 'take_profit';
  status: 'pending' | 'filled' | 'partially_filled' | 'cancelled' | 'rejected';
  price: number;
  quantity: number;
  filledQuantity: number;
  remainingQuantity: number;
  total: number;
  timestamp: string;
  updatedAt: string;
}

export interface Position {
  id: string;
  symbol: string;
  side: 'long' | 'short';
  entryPrice: number;
  currentPrice: number;
  quantity: number;
  leverage: number;
  margin: number;
  pnl: number;
  pnlPercent: number;
  liquidationPrice: number;
  timestamp: string;
}

export interface RiskSettings {
  maxPositionSize: number;
  maxLeverage: number;
  stopLossPercent: number;
  takeProfitPercent: number;
  maxDailyLoss: number;
  maxDrawdown: number;
  riskPerTrade: number;
}

export interface DataSource {
  id: string;
  name: string;
  type: 'price' | 'onchain' | 'sentiment' | 'news';
  status: 'active' | 'inactive' | 'error';
  lastUpdate: string;
  latency: number;
  errorRate: number;
  config: Record<string, any>;
}

export interface SystemLog {
  id: string;
  level: 'info' | 'warning' | 'error' | 'debug';
  message: string;
  source: string;
  timestamp: string;
  metadata?: Record<string, any>;
}

export interface PerformanceMetrics {
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  totalPnl: number;
  dailyPnl: number;
  weeklyPnl: number;
  monthlyPnl: number;
  sharpeRatio: number;
  maxDrawdown: number;
  avgTradeDuration: number;
}

export interface SentimentData {
  symbol: string;
  fearGreedIndex: number;
  fearGreedClassification: string;
  fundingRate: number;
  longShortRatio: number;
  liquidationVolume: number;
  timestamp: string;
}

export interface UserSettings {
  apiKey: string;
  apiSecret: string;
  testnet: boolean;
  defaultLeverage: number;
  notifications: boolean;
  theme: 'light' | 'dark';
}
