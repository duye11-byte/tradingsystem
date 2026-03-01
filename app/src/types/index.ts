// 策略相关类型
export interface Strategy {
  id: string;
  name: string;
  description: string;
  type: 'trend_following' | 'mean_reversion' | 'breakout' | 'scalping' | 'custom';
  createdAt: Date;
  updatedAt: Date;
  isActive: boolean;
  parameters: StrategyParameters;
  rules: TradeRules;
  performance?: StrategyPerformance;
  source: 'imported' | 'manual' | 'ai_generated';
}

export interface StrategyParameters {
  entryConditions: Condition[];
  exitConditions: Condition[];
  riskManagement: RiskParameters;
  indicators: IndicatorConfig[];
}

export interface Condition {
  id: string;
  indicator: string;
  operator: '>' | '<' | '==' | '>=' | '<=' | 'crosses_above' | 'crosses_below';
  value: number | string;
  timeframe: string;
}

export interface RiskParameters {
  maxPositionSize: number; // 百分比
  stopLossPercent: number;
  takeProfitPercent: number;
  maxDailyLoss: number;
  maxDrawdownPercent: number;
  useTrailingStop: boolean;
  trailingStopPercent?: number;
}

export interface IndicatorConfig {
  name: string;
  parameters: Record<string, number>;
  timeframe: string;
}

export interface TradeRules {
  longEntry: string[];
  shortEntry: string[];
  exitLong: string[];
  exitShort: string[];
}

export interface StrategyPerformance {
  totalReturn: number;
  winRate: number;
  profitFactor: number;
  maxDrawdown: number;
  sharpeRatio: number;
  totalTrades: number;
  averageTrade: number;
  lastBacktestDate?: Date;
}

// 回测相关类型
export interface BacktestConfig {
  startDate: Date;
  endDate: Date;
  initialCapital: number;
  tradingFee: number; // 百分比
  slippage: number; // 百分比
  symbol: string;
  timeframe: string;
}

export interface BacktestResult {
  id: string;
  strategyId: string;
  config: BacktestConfig;
  trades: Trade[];
  equityCurve: EquityPoint[];
  metrics: BacktestMetrics;
  completedAt: Date;
}

export interface Trade {
  id: string;
  entryTime: Date;
  exitTime?: Date;
  entryPrice: number;
  exitPrice?: number;
  side: 'long' | 'short';
  size: number;
  pnl: number;
  pnlPercent: number;
  status: 'open' | 'closed';
  exitReason?: string;
}

export interface EquityPoint {
  timestamp: Date;
  equity: number;
  drawdown: number;
}

export interface BacktestMetrics {
  totalReturn: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  averageWin: number;
  averageLoss: number;
  profitFactor: number;
  maxDrawdown: number;
  maxDrawdownPercent: number;
  sharpeRatio: number;
  sortinoRatio: number;
  calmarRatio: number;
  averageTrade: number;
  averageTradePercent: number;
}

// AI分析相关类型
export interface AIStrategyAnalysis {
  strategyId: string;
  overallScore: number; // 0-100
  strengths: StrategyStrength[];
  weaknesses: StrategyWeakness[];
  recommendations: string[];
  extractedPatterns: Pattern[];
  comparableStrategies: string[];
  integrationPotential: IntegrationPotential;
}

export interface StrategyStrength {
  aspect: string;
  score: number;
  description: string;
  evidence: string[];
}

export interface StrategyWeakness {
  aspect: string;
  severity: 'low' | 'medium' | 'high';
  description: string;
  suggestion: string;
}

export interface Pattern {
  type: 'entry' | 'exit' | 'risk' | 'indicator';
  description: string;
  effectiveness: number;
  conditions: string[];
}

export interface IntegrationPotential {
  score: number; // 0-100
  compatibleStrategies: string[];
  suggestedIntegrations: SuggestedIntegration[];
}

export interface SuggestedIntegration {
  targetStrategyId: string;
  targetStrategyName: string;
  integrationType: 'entry' | 'exit' | 'risk' | 'indicator';
  description: string;
  expectedImprovement: number;
  confidence: number;
}

// AI代理状态
export interface AIAgentState {
  isRunning: boolean;
  currentTask?: string;
  progress: number;
  lastAnalysis?: AIStrategyAnalysis;
  pendingStrategies: string[];
  analyzedStrategies: string[];
  integratedStrategies: string[];
}

// 策略集成结果
export interface StrategyIntegration {
  id: string;
  sourceStrategyId: string;
  targetStrategyId: string;
  integratedAt: Date;
  integrationType: 'entry' | 'exit' | 'risk' | 'indicator' | 'full';
  changes: IntegrationChange[];
  beforePerformance: StrategyPerformance;
  afterPerformance: StrategyPerformance;
  improvement: number;
  aiNotes: string;
}

export interface IntegrationChange {
  field: string;
  oldValue: any;
  newValue: any;
  reason: string;
}

// 市场数据
export interface MarketData {
  symbol: string;
  timestamp: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

// 用户设置
export interface UserSettings {
  apiKeys: Record<string, string>;
  notifications: NotificationSettings;
  riskDefaults: RiskParameters;
  aiPreferences: AIPreferences;
}

export interface NotificationSettings {
  email: boolean;
  push: boolean;
  tradeAlerts: boolean;
  backtestComplete: boolean;
  aiRecommendations: boolean;
}

export interface AIPreferences {
  autoBacktest: boolean;
  autoIntegrate: boolean;
  minScoreForIntegration: number;
  maxIntegrationPerDay: number;
  preferredIntegrationType: 'conservative' | 'balanced' | 'aggressive';
}

// 系统状态
export interface SystemState {
  isLoading: boolean;
  error: string | null;
  connected: boolean;
  lastSync: Date;
}
