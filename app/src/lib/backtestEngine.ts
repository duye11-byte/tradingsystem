import type { 
  Strategy, 
  BacktestConfig, 
  BacktestResult, 
  BacktestMetrics, 
  Trade, 
  EquityPoint,
  MarketData
} from '@/types';

// 模拟市场数据生成器
export class MarketDataGenerator {
  private symbol: string;
  private timeframe: string;
  private volatility: number;
  private trend: number;

  constructor(symbol: string, timeframe: string, volatility: number = 0.02, trend: number = 0) {
    this.symbol = symbol;
    this.timeframe = timeframe;
    this.volatility = volatility;
    this.trend = trend;
  }

  generateData(startDate: Date, endDate: Date): MarketData[] {
    const data: MarketData[] = [];
    const timeframeMs = this.getTimeframeMs();
    let currentPrice = this.getInitialPrice();
    let currentTime = new Date(startDate);

    while (currentTime <= endDate) {
      const change = (Math.random() - 0.5) * this.volatility + this.trend;
      const open = currentPrice;
      const close = currentPrice * (1 + change);
      const high = Math.max(open, close) * (1 + Math.random() * this.volatility * 0.5);
      const low = Math.min(open, close) * (1 - Math.random() * this.volatility * 0.5);
      const volume = Math.random() * 1000000 + 500000;

      data.push({
        symbol: this.symbol,
        timestamp: new Date(currentTime),
        open,
        high,
        low,
        close,
        volume,
      });

      currentPrice = close;
      currentTime = new Date(currentTime.getTime() + timeframeMs);
    }

    return data;
  }

  private getTimeframeMs(): number {
    const timeframes: Record<string, number> = {
      '1m': 60 * 1000,
      '5m': 5 * 60 * 1000,
      '15m': 15 * 60 * 1000,
      '1h': 60 * 60 * 1000,
      '4h': 4 * 60 * 60 * 1000,
      '1d': 24 * 60 * 60 * 1000,
    };
    return timeframes[this.timeframe] || 60 * 60 * 1000;
  }

  private getInitialPrice(): number {
    const prices: Record<string, number> = {
      'BTC/USDT': 45000,
      'ETH/USDT': 3000,
      'SOL/USDT': 100,
      'BNB/USDT': 300,
      'XRP/USDT': 0.5,
    };
    return prices[this.symbol] || 100;
  }
}

// 技术指标计算器
export class IndicatorCalculator {
  static SMA(data: number[], period: number): number[] {
    const result: number[] = [];
    for (let i = 0; i < data.length; i++) {
      if (i < period - 1) {
        result.push(NaN);
        continue;
      }
      let sum = 0;
      for (let j = 0; j < period; j++) {
        sum += data[i - j];
      }
      result.push(sum / period);
    }
    return result;
  }

  static EMA(data: number[], period: number): number[] {
    const result: number[] = [];
    const multiplier = 2 / (period + 1);
    
    for (let i = 0; i < data.length; i++) {
      if (i === 0) {
        result.push(data[i]);
      } else if (i < period - 1) {
        result.push(NaN);
      } else {
        const ema = (data[i] - result[i - 1]) * multiplier + result[i - 1];
        result.push(ema);
      }
    }
    return result;
  }

  static RSI(data: number[], period: number = 14): number[] {
    const result: number[] = [];
    const gains: number[] = [];
    const losses: number[] = [];

    for (let i = 1; i < data.length; i++) {
      const change = data[i] - data[i - 1];
      gains.push(change > 0 ? change : 0);
      losses.push(change < 0 ? -change : 0);
    }

    for (let i = 0; i < data.length; i++) {
      if (i < period) {
        result.push(NaN);
        continue;
      }

      let avgGain = 0;
      let avgLoss = 0;
      for (let j = i - period; j < i; j++) {
        avgGain += gains[j];
        avgLoss += losses[j];
      }
      avgGain /= period;
      avgLoss /= period;

      if (avgLoss === 0) {
        result.push(100);
      } else {
        const rs = avgGain / avgLoss;
        result.push(100 - (100 / (1 + rs)));
      }
    }

    return result;
  }

  static MACD(
    data: number[], 
    fastPeriod: number = 12, 
    slowPeriod: number = 26, 
    signalPeriod: number = 9
  ): { macd: number[]; signal: number[]; histogram: number[] } {
    const fastEMA = this.EMA(data, fastPeriod);
    const slowEMA = this.EMA(data, slowPeriod);
    const macd: number[] = [];

    for (let i = 0; i < data.length; i++) {
      if (isNaN(fastEMA[i]) || isNaN(slowEMA[i])) {
        macd.push(NaN);
      } else {
        macd.push(fastEMA[i] - slowEMA[i]);
      }
    }

    const signal = this.EMA(macd.filter(v => !isNaN(v)), signalPeriod);
    const histogram: number[] = [];

    for (let i = 0; i < macd.length; i++) {
      if (isNaN(macd[i]) || i >= signal.length || isNaN(signal[i])) {
        histogram.push(NaN);
      } else {
        histogram.push(macd[i] - signal[i]);
      }
    }

    return { macd, signal: signal.concat(Array(macd.length - signal.length).fill(NaN)), histogram };
  }

  static BollingerBands(
    data: number[], 
    period: number = 20, 
    stdDev: number = 2
  ): { upper: number[]; middle: number[]; lower: number[] } {
    const middle = this.SMA(data, period);
    const upper: number[] = [];
    const lower: number[] = [];

    for (let i = 0; i < data.length; i++) {
      if (i < period - 1) {
        upper.push(NaN);
        lower.push(NaN);
        continue;
      }

      let sum = 0;
      for (let j = 0; j < period; j++) {
        sum += Math.pow(data[i - j] - middle[i], 2);
      }
      const std = Math.sqrt(sum / period);

      upper.push(middle[i] + stdDev * std);
      lower.push(middle[i] - stdDev * std);
    }

    return { upper, middle, lower };
  }

  static ATR(data: { high: number; low: number; close: number }[], period: number = 14): number[] {
    const tr: number[] = [];
    
    for (let i = 0; i < data.length; i++) {
      if (i === 0) {
        tr.push(data[i].high - data[i].low);
      } else {
        const tr1 = data[i].high - data[i].low;
        const tr2 = Math.abs(data[i].high - data[i - 1].close);
        const tr3 = Math.abs(data[i].low - data[i - 1].close);
        tr.push(Math.max(tr1, tr2, tr3));
      }
    }

    return this.SMA(tr, period);
  }
}

// 回测引擎
export class BacktestEngine {
  private strategy: Strategy;
  private config: BacktestConfig;
  private marketData: MarketData[];
  private trades: Trade[] = [];
  private equityCurve: EquityPoint[] = [];
  private currentEquity: number;
  private position: Trade | null = null;
  private indicators: Map<string, number[]> = new Map();

  constructor(strategy: Strategy, config: BacktestConfig, marketData: MarketData[]) {
    this.strategy = strategy;
    this.config = config;
    this.marketData = marketData;
    this.currentEquity = config.initialCapital;
  }

  run(): BacktestResult {
    // 计算指标
    this.calculateIndicators();

    // 遍历市场数据
    for (let i = 50; i < this.marketData.length; i++) {
      const candle = this.marketData[i];
      const context = this.getContext(i);

      // 检查是否有持仓
      if (this.position) {
        // 检查退出条件
        if (this.shouldExit(i, context)) {
          this.closePosition(candle, i, 'signal');
        } else {
          // 检查止损止盈
          const exitResult = this.checkStopLossTakeProfit(candle);
          if (exitResult) {
            this.closePosition(candle, i, exitResult);
          }
        }
      } else {
        // 检查入场条件
        const entrySignal = this.checkEntry(i, context);
        if (entrySignal) {
          this.openPosition(candle, entrySignal, i);
        }
      }

      // 更新权益曲线
      this.updateEquity(candle);
    }

    // 关闭所有未平仓头寸
    if (this.position) {
      this.closePosition(this.marketData[this.marketData.length - 1], this.marketData.length - 1, 'end_of_test');
    }

    // 计算指标
    const metrics = this.calculateMetrics();

    return {
      id: `backtest_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      strategyId: this.strategy.id,
      config: this.config,
      trades: this.trades,
      equityCurve: this.equityCurve,
      metrics,
      completedAt: new Date(),
    };
  }

  private calculateIndicators() {
    const closes = this.marketData.map(d => d.close);

    this.strategy.parameters.indicators.forEach(indicator => {
      switch (indicator.name) {
        case 'SMA':
          this.indicators.set(`SMA_${indicator.parameters.period}`, 
            IndicatorCalculator.SMA(closes, indicator.parameters.period));
          break;
        case 'EMA':
          this.indicators.set(`EMA_${indicator.parameters.period}`, 
            IndicatorCalculator.EMA(closes, indicator.parameters.period));
          break;
        case 'RSI':
          this.indicators.set('RSI', 
            IndicatorCalculator.RSI(closes, indicator.parameters.period));
          break;
        case 'MACD':
          const macd = IndicatorCalculator.MACD(closes, 
            indicator.parameters.fastPeriod,
            indicator.parameters.slowPeriod,
            indicator.parameters.signalPeriod);
          this.indicators.set('MACD', macd.macd);
          this.indicators.set('MACD_SIGNAL', macd.signal);
          this.indicators.set('MACD_HISTOGRAM', macd.histogram);
          break;
        case 'BB':
          const bb = IndicatorCalculator.BollingerBands(closes,
            indicator.parameters.period,
            indicator.parameters.stdDev);
          this.indicators.set('BB_UPPER', bb.upper);
          this.indicators.set('BB_MIDDLE', bb.middle);
          this.indicators.set('BB_LOWER', bb.lower);
          break;
        case 'ATR':
          this.indicators.set('ATR', 
            IndicatorCalculator.ATR(this.marketData.map(d => ({ high: d.high, low: d.low, close: d.close })),
              indicator.parameters.period));
          break;
      }
    });
  }

  private getContext(index: number): Record<string, number> {
    const context: Record<string, number> = {};
    const candle = this.marketData[index];
    
    context.price = candle.close;
    context.high = candle.high;
    context.low = candle.low;
    context.open = candle.open;
    context.volume = candle.volume;

    this.indicators.forEach((values, name) => {
      if (index < values.length && !isNaN(values[index])) {
        context[name] = values[index];
      }
    });

    return context;
  }

  private checkEntry(_index: number, context: Record<string, number>): 'long' | 'short' | null {
    const longConditions = this.strategy.parameters.entryConditions.filter(c => 
      this.strategy.rules.longEntry.includes(c.id));
    const shortConditions = this.strategy.parameters.entryConditions.filter(c => 
      this.strategy.rules.shortEntry.includes(c.id));

    let longMet = 0;
    let longTotal = longConditions.length;
    let shortMet = 0;
    let shortTotal = shortConditions.length;

    for (const condition of longConditions) {
      if (this.evaluateCondition(condition, context)) {
        longMet++;
      }
    }

    for (const condition of shortConditions) {
      if (this.evaluateCondition(condition, context)) {
        shortMet++;
      }
    }

    // 需要至少70%的条件满足
    if (longTotal > 0 && longMet / longTotal >= 0.7) {
      return 'long';
    }
    if (shortTotal > 0 && shortMet / shortTotal >= 0.7) {
      return 'short';
    }

    return null;
  }

  private shouldExit(_index: number, context: Record<string, number>): boolean {
    const exitConditions = this.position?.side === 'long' 
      ? this.strategy.parameters.exitConditions.filter(c => 
          this.strategy.rules.exitLong.includes(c.id))
      : this.strategy.parameters.exitConditions.filter(c => 
          this.strategy.rules.exitShort.includes(c.id));

    let met = 0;
    for (const condition of exitConditions) {
      if (this.evaluateCondition(condition, context)) {
        met++;
      }
    }

    return exitConditions.length > 0 && met / exitConditions.length >= 0.7;
  }

  private evaluateCondition(condition: { indicator: string; operator: string; value: number | string }, 
    context: Record<string, number>): boolean {
    const indicatorValue = context[condition.indicator];
    if (indicatorValue === undefined) return false;

    const value = typeof condition.value === 'string' ? context[condition.value] : condition.value;
    if (value === undefined) return false;

    switch (condition.operator) {
      case '>': return indicatorValue > value;
      case '<': return indicatorValue < value;
      case '==': return indicatorValue === value;
      case '>=': return indicatorValue >= value;
      case '<=': return indicatorValue <= value;
      default: return false;
    }
  }

  private checkStopLossTakeProfit(candle: MarketData): string | null {
    if (!this.position) return null;

    const { stopLossPercent, takeProfitPercent } = 
      this.strategy.parameters.riskManagement;

    if (this.position.side === 'long') {
      const stopPrice = this.position.entryPrice * (1 - stopLossPercent / 100);
      const tpPrice = this.position.entryPrice * (1 + takeProfitPercent / 100);

      if (candle.low <= stopPrice) return 'stop_loss';
      if (candle.high >= tpPrice) return 'take_profit';
    } else {
      const stopPrice = this.position.entryPrice * (1 + stopLossPercent / 100);
      const tpPrice = this.position.entryPrice * (1 - takeProfitPercent / 100);

      if (candle.high >= stopPrice) return 'stop_loss';
      if (candle.low <= tpPrice) return 'take_profit';
    }

    return null;
  }

  private openPosition(candle: MarketData, side: 'long' | 'short', _index: number) {
    const positionSize = this.currentEquity * 
      (this.strategy.parameters.riskManagement.maxPositionSize / 100);
    const size = positionSize / candle.close;

    this.position = {
      id: `trade_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      entryTime: candle.timestamp,
      entryPrice: candle.close,
      side,
      size,
      pnl: 0,
      pnlPercent: 0,
      status: 'open',
    };
  }

  private closePosition(candle: MarketData, _index: number, reason: string) {
    if (!this.position) return;

    const exitPrice = candle.close;
    const pnl = this.position.side === 'long'
      ? (exitPrice - this.position.entryPrice) * this.position.size
      : (this.position.entryPrice - exitPrice) * this.position.size;

    const pnlPercent = (pnl / (this.position.entryPrice * this.position.size)) * 100;

    const closedTrade: Trade = {
      ...this.position,
      exitTime: candle.timestamp,
      exitPrice,
      pnl,
      pnlPercent,
      status: 'closed',
      exitReason: reason,
    };

    this.trades.push(closedTrade);
    this.currentEquity += pnl;
    this.position = null;
  }

  private updateEquity(candle: MarketData) {
    let unrealizedPnl = 0;
    if (this.position) {
      unrealizedPnl = this.position.side === 'long'
        ? (candle.close - this.position.entryPrice) * this.position.size
        : (this.position.entryPrice - candle.close) * this.position.size;
    }

    const totalEquity = this.currentEquity + unrealizedPnl;
    const peak = Math.max(...this.equityCurve.map(e => e.equity), this.config.initialCapital);
    const drawdown = peak - totalEquity;
    const drawdownPercent = (drawdown / peak) * 100;

    this.equityCurve.push({
      timestamp: candle.timestamp,
      equity: totalEquity,
      drawdown: drawdownPercent,
    });
  }

  private calculateMetrics(): BacktestMetrics {
    const closedTrades = this.trades.filter(t => t.status === 'closed');
    const winningTrades = closedTrades.filter(t => t.pnl > 0);
    const losingTrades = closedTrades.filter(t => t.pnl <= 0);

    const totalProfit = winningTrades.reduce((sum, t) => sum + t.pnl, 0);
    const totalLoss = Math.abs(losingTrades.reduce((sum, t) => sum + t.pnl, 0));

    const averageWin = winningTrades.length > 0 ? totalProfit / winningTrades.length : 0;
    const averageLoss = losingTrades.length > 0 ? totalLoss / losingTrades.length : 0;

    const returns = this.equityCurve.map(e => 
      (e.equity - this.config.initialCapital) / this.config.initialCapital);
    
    const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length;
    const stdDev = Math.sqrt(variance);

    const maxDrawdown = Math.max(...this.equityCurve.map(e => e.drawdown));

    // 年化夏普比率 (简化计算)
    const riskFreeRate = 0.02;
    const sharpeRatio = stdDev > 0 
      ? ((avgReturn * 252) - riskFreeRate) / (stdDev * Math.sqrt(252)) 
      : 0;

    // 索提诺比率
    const downsideReturns = returns.filter(r => r < 0);
    const downsideDev = downsideReturns.length > 0
      ? Math.sqrt(downsideReturns.reduce((sum, r) => sum + Math.pow(r, 2), 0) / downsideReturns.length)
      : 0;
    const sortinoRatio = downsideDev > 0
      ? ((avgReturn * 252) - riskFreeRate) / (downsideDev * Math.sqrt(252))
      : 0;

    // 卡尔玛比率
    const calmarRatio = maxDrawdown > 0
      ? ((this.currentEquity - this.config.initialCapital) / this.config.initialCapital * 100) / maxDrawdown
      : 0;

    return {
      totalReturn: ((this.currentEquity - this.config.initialCapital) / this.config.initialCapital) * 100,
      totalTrades: closedTrades.length,
      winningTrades: winningTrades.length,
      losingTrades: losingTrades.length,
      winRate: closedTrades.length > 0 ? (winningTrades.length / closedTrades.length) * 100 : 0,
      averageWin,
      averageLoss,
      profitFactor: totalLoss > 0 ? totalProfit / totalLoss : totalProfit > 0 ? Infinity : 0,
      maxDrawdown,
      maxDrawdownPercent: maxDrawdown,
      sharpeRatio,
      sortinoRatio,
      calmarRatio,
      averageTrade: closedTrades.length > 0 
        ? closedTrades.reduce((sum, t) => sum + t.pnl, 0) / closedTrades.length 
        : 0,
      averageTradePercent: closedTrades.length > 0
        ? closedTrades.reduce((sum, t) => sum + t.pnlPercent, 0) / closedTrades.length
        : 0,
    };
  }
}

// 回测服务
export class BacktestService {
  static async runBacktest(
    strategy: Strategy, 
    config: BacktestConfig
  ): Promise<BacktestResult> {
    // 生成市场数据
    const dataGenerator = new MarketDataGenerator(
      config.symbol,
      config.timeframe,
      0.025, // 波动率
      0.0001 // 趋势
    );
    const marketData = dataGenerator.generateData(config.startDate, config.endDate);

    // 运行回测
    const engine = new BacktestEngine(strategy, config, marketData);
    return engine.run();
  }
}
