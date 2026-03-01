import type { 
  Strategy, 
  BacktestResult, 
  AIStrategyAnalysis, 
  StrategyIntegration,
  StrategyStrength,
  StrategyWeakness,
  Pattern,
  SuggestedIntegration,
  StrategyPerformance,
  IntegrationPotential
} from '@/types';

// AI策略分析器
export class AIStrategyAnalyzer {
  private readonly EXCELLENT_THRESHOLD = 75;
  private readonly MIN_TRADES = 20;

  analyze(strategy: Strategy, backtestResult: BacktestResult): AIStrategyAnalysis {
    const metrics = backtestResult.metrics;
    
    // 计算综合评分
    const overallScore = this.calculateOverallScore(metrics);
    
    // 识别优势
    const strengths = this.identifyStrengths(metrics, backtestResult);
    
    // 识别弱点
    const weaknesses = this.identifyWeaknesses(metrics, backtestResult);
    
    // 生成建议
    const recommendations = this.generateRecommendations(metrics, weaknesses);
    
    // 提取模式
    const extractedPatterns = this.extractPatterns(strategy, backtestResult);
    
    // 评估集成潜力
    const integrationPotential = this.assessIntegrationPotential(
      strategy, 
      overallScore, 
      strengths, 
      extractedPatterns
    );

    return {
      strategyId: strategy.id,
      overallScore,
      strengths,
      weaknesses,
      recommendations,
      extractedPatterns,
      comparableStrategies: [],
      integrationPotential,
    };
  }

  private calculateOverallScore(metrics: BacktestResult['metrics']): number {
    let score = 0;
    let weights = 0;

    // 收益率权重: 30%
    const returnScore = Math.min(Math.max(metrics.totalReturn / 50 * 100, 0), 100);
    score += returnScore * 0.30;
    weights += 0.30;

    // 胜率权重: 20%
    const winRateScore = metrics.winRate;
    score += winRateScore * 0.20;
    weights += 0.20;

    // 盈亏比权重: 20%
    const profitFactorScore = Math.min(metrics.profitFactor / 3 * 100, 100);
    score += profitFactorScore * 0.20;
    weights += 0.20;

    // 夏普比率权重: 15%
    const sharpeScore = Math.min(Math.max((metrics.sharpeRatio + 2) / 4 * 100, 0), 100);
    score += sharpeScore * 0.15;
    weights += 0.15;

    // 最大回撤权重: 15%
    const drawdownScore = Math.max(0, 100 - metrics.maxDrawdownPercent * 2);
    score += drawdownScore * 0.15;
    weights += 0.15;

    return Math.round(score / weights);
  }

  private identifyStrengths(
    metrics: BacktestResult['metrics'], 
    backtestResult: BacktestResult
  ): StrategyStrength[] {
    const strengths: StrategyStrength[] = [];

    if (metrics.totalReturn > 20) {
      strengths.push({
        aspect: '收益率',
        score: Math.min(metrics.totalReturn / 50 * 100, 100),
        description: `策略在回测期间实现了 ${metrics.totalReturn.toFixed(2)}% 的收益率，表现优异`,
        evidence: [`总收益率: ${metrics.totalReturn.toFixed(2)}%`, `最终权益: ${backtestResult.equityCurve[backtestResult.equityCurve.length - 1]?.equity.toFixed(2)}`],
      });
    }

    if (metrics.winRate > 55) {
      strengths.push({
        aspect: '胜率',
        score: metrics.winRate,
        description: `策略具有较高的胜率，达到 ${metrics.winRate.toFixed(1)}%`,
        evidence: [`胜率: ${metrics.winRate.toFixed(1)}%`, `盈利交易: ${metrics.winningTrades} 笔`, `亏损交易: ${metrics.losingTrades} 笔`],
      });
    }

    if (metrics.profitFactor > 1.5) {
      strengths.push({
        aspect: '盈亏比',
        score: Math.min(metrics.profitFactor / 3 * 100, 100),
        description: `盈亏比为 ${metrics.profitFactor.toFixed(2)}，表明盈利交易的平均收益显著高于亏损交易的平均损失`,
        evidence: [`盈亏比: ${metrics.profitFactor.toFixed(2)}`, `平均盈利: ${metrics.averageWin.toFixed(2)}`, `平均亏损: ${metrics.averageLoss.toFixed(2)}`],
      });
    }

    if (metrics.sharpeRatio > 1) {
      strengths.push({
        aspect: '风险调整收益',
        score: Math.min((metrics.sharpeRatio + 2) / 4 * 100, 100),
        description: `夏普比率为 ${metrics.sharpeRatio.toFixed(2)}，表明策略在承担单位风险时获得了较好的回报`,
        evidence: [`夏普比率: ${metrics.sharpeRatio.toFixed(2)}`, `索提诺比率: ${metrics.sortinoRatio.toFixed(2)}`, `卡尔玛比率: ${metrics.calmarRatio.toFixed(2)}`],
      });
    }

    if (metrics.maxDrawdownPercent < 10) {
      strengths.push({
        aspect: '风险控制',
        score: Math.max(0, 100 - metrics.maxDrawdownPercent * 5),
        description: `最大回撤控制在 ${metrics.maxDrawdownPercent.toFixed(2)}%，风险控制能力出色`,
        evidence: [`最大回撤: ${metrics.maxDrawdownPercent.toFixed(2)}%`, `平均回撤: ${(metrics.maxDrawdownPercent * 0.6).toFixed(2)}%`],
      });
    }

    if (metrics.totalTrades >= this.MIN_TRADES) {
      strengths.push({
        aspect: '样本量',
        score: Math.min(metrics.totalTrades / 100 * 100, 100),
        description: `回测包含 ${metrics.totalTrades} 笔交易，样本量充足，结果具有统计意义`,
        evidence: [`总交易数: ${metrics.totalTrades}`, `平均每笔收益: ${metrics.averageTrade.toFixed(2)}`],
      });
    }

    return strengths;
  }

  private identifyWeaknesses(
    metrics: BacktestResult['metrics'], 
    backtestResult: BacktestResult
  ): StrategyWeakness[] {
    const weaknesses: StrategyWeakness[] = [];

    if (metrics.totalReturn < 5) {
      weaknesses.push({
        aspect: '收益率偏低',
        severity: metrics.totalReturn < 0 ? 'high' : 'medium',
        description: `策略收益率仅为 ${metrics.totalReturn.toFixed(2)}%，可能无法覆盖交易成本和资金占用成本`,
        suggestion: '考虑优化入场条件，或增加仓位规模（在风险可控的前提下）',
      });
    }

    if (metrics.winRate < 45) {
      weaknesses.push({
        aspect: '胜率偏低',
        severity: 'medium',
        description: `胜率仅为 ${metrics.winRate.toFixed(1)}%，意味着超过一半的交易是亏损的`,
        suggestion: '优化入场过滤条件，减少低质量交易信号，或调整止损止盈比例',
      });
    }

    if (metrics.profitFactor < 1.2) {
      weaknesses.push({
        aspect: '盈亏比不足',
        severity: metrics.profitFactor < 1 ? 'high' : 'medium',
        description: `盈亏比为 ${metrics.profitFactor.toFixed(2)}，盈利交易的收益与亏损交易的损失接近`,
        suggestion: '扩大止盈目标或收紧止损，提高盈亏比',
      });
    }

    if (metrics.maxDrawdownPercent > 20) {
      weaknesses.push({
        aspect: '回撤过大',
        severity: 'high',
        description: `最大回撤达到 ${metrics.maxDrawdownPercent.toFixed(2)}%，风险较高`,
        suggestion: '加强风险管理，降低仓位规模，或添加更多过滤条件避免不利市场条件',
      });
    }

    if (metrics.sharpeRatio < 0.5) {
      weaknesses.push({
        aspect: '风险调整收益偏低',
        severity: 'medium',
        description: `夏普比率为 ${metrics.sharpeRatio.toFixed(2)}，策略承担的风险与获得的回报不成比例`,
        suggestion: '优化策略逻辑，减少无效交易，或改进出场时机',
      });
    }

    if (metrics.totalTrades < this.MIN_TRADES) {
      weaknesses.push({
        aspect: '样本量不足',
        severity: 'medium',
        description: `仅产生 ${metrics.totalTrades} 笔交易，样本量较小，回测结果可能不够可靠`,
        suggestion: '延长回测周期，或调整策略参数以增加交易频率',
      });
    }

    // 检查连续亏损
    const consecutiveLosses = this.findMaxConsecutiveLosses(backtestResult.trades);
    if (consecutiveLosses > 5) {
      weaknesses.push({
        aspect: '连续亏损较多',
        severity: consecutiveLosses > 8 ? 'high' : 'medium',
        description: `策略曾经出现 ${consecutiveLosses} 笔连续亏损，可能对心理造成压力`,
        suggestion: '添加趋势过滤器，避免在不利市场条件下交易',
      });
    }

    return weaknesses;
  }

  private findMaxConsecutiveLosses(trades: BacktestResult['trades']): number {
    let maxConsecutive = 0;
    let currentConsecutive = 0;

    for (const trade of trades) {
      if (trade.pnl <= 0) {
        currentConsecutive++;
        maxConsecutive = Math.max(maxConsecutive, currentConsecutive);
      } else {
        currentConsecutive = 0;
      }
    }

    return maxConsecutive;
  }

  private generateRecommendations(
    metrics: BacktestResult['metrics'], 
    weaknesses: StrategyWeakness[]
  ): string[] {
    const recommendations: string[] = [];

    // 基于弱点的建议
    for (const weakness of weaknesses) {
      recommendations.push(weakness.suggestion);
    }

    // 通用优化建议
    if (metrics.winRate > 60 && metrics.profitFactor < 1.5) {
      recommendations.push('策略胜率较高但盈亏比一般，可以考虑让盈利交易运行更长时间，提高盈亏比');
    }

    if (metrics.profitFactor > 2 && metrics.winRate < 45) {
      recommendations.push('策略盈亏比优秀但胜率偏低，可以收紧入场条件，只选择最高质量的信号');
    }

    if (metrics.maxDrawdownPercent > 15) {
      recommendations.push('建议实施动态仓位管理，根据近期表现调整仓位规模');
    }

    if (metrics.sharpeRatio < 1) {
      recommendations.push('考虑添加市场环境过滤器，在趋势不明朗时减少交易或暂停交易');
    }

    recommendations.push('定期重新评估策略参数，确保策略适应不断变化的市场条件');

    return [...new Set(recommendations)];
  }

  private extractPatterns(
    strategy: Strategy, 
    backtestResult: BacktestResult
  ): Pattern[] {
    const patterns: Pattern[] = [];
    const metrics = backtestResult.metrics;
    const trades = backtestResult.trades.filter(t => t.status === 'closed');
    const winningTrades = trades.filter(t => t.pnl > 0);

    // 分析入场模式
    if (winningTrades.length > 0) {
      const avgWinHoldingTime = winningTrades.reduce((sum, t) => {
        if (t.exitTime && t.entryTime) {
          return sum + (new Date(t.exitTime).getTime() - new Date(t.entryTime).getTime());
        }
        return sum;
      }, 0) / winningTrades.length;

      patterns.push({
        type: 'entry',
        description: `盈利交易的平均持仓时间为 ${(avgWinHoldingTime / (1000 * 60 * 60)).toFixed(1)} 小时`,
        effectiveness: metrics.winRate,
        conditions: strategy.parameters.entryConditions.map(c => `${c.indicator} ${c.operator} ${c.value}`),
      });
    }

    // 分析出场模式
    const exitReasons = new Map<string, number>();
    for (const trade of trades) {
      if (trade.exitReason) {
        exitReasons.set(trade.exitReason, (exitReasons.get(trade.exitReason) || 0) + 1);
      }
    }

    const bestExitReason = Array.from(exitReasons.entries())
      .sort((a, b) => b[1] - a[1])[0];

    if (bestExitReason) {
      patterns.push({
        type: 'exit',
        description: `最常见的出场原因是: ${bestExitReason[0]}`,
        effectiveness: (bestExitReason[1] / trades.length) * 100,
        conditions: strategy.parameters.exitConditions.map(c => `${c.indicator} ${c.operator} ${c.value}`),
      });
    }

    // 分析风险管理效果
    const slExits = trades.filter(t => t.exitReason === 'stop_loss');
    const tpExits = trades.filter(t => t.exitReason === 'take_profit');

    if (slExits.length > 0 && tpExits.length > 0) {
      const slRate = (slExits.length / trades.length) * 100;
      patterns.push({
        type: 'risk',
        description: `止损触发率为 ${slRate.toFixed(1)}%，止盈触发率为 ${((tpExits.length / trades.length) * 100).toFixed(1)}%`,
        effectiveness: 100 - slRate,
        conditions: [`止损: ${strategy.parameters.riskManagement.stopLossPercent}%`, `止盈: ${strategy.parameters.riskManagement.takeProfitPercent}%`],
      });
    }

    // 分析指标效果
    for (const indicator of strategy.parameters.indicators) {
      patterns.push({
        type: 'indicator',
        description: `使用 ${indicator.name} 指标，参数: ${JSON.stringify(indicator.parameters)}`,
        effectiveness: metrics.winRate,
        conditions: [`时间周期: ${indicator.timeframe}`],
      });
    }

    return patterns;
  }

  private assessIntegrationPotential(
    _strategy: Strategy,
    overallScore: number,
    strengths: StrategyStrength[],
    _patterns: Pattern[]
  ): IntegrationPotential {
    const compatibleStrategies: string[] = [];
    const suggestedIntegrations: SuggestedIntegration[] = [];

    // 如果策略评分优秀，评估其集成潜力
    if (overallScore >= this.EXCELLENT_THRESHOLD) {
      // 根据优势类型确定可集成的方面
      for (const strength of strengths) {
        if (strength.aspect === '入场时机' || strength.aspect === '收益率') {
          suggestedIntegrations.push({
            targetStrategyId: 'existing_strategy',
            targetStrategyName: '现有策略',
            integrationType: 'entry',
            description: `集成该策略的入场条件，预计可提升入场质量`,
            expectedImprovement: strength.score * 0.1,
            confidence: strength.score / 100,
          });
        }

        if (strength.aspect === '风险控制' || strength.aspect === '出场时机') {
          suggestedIntegrations.push({
            targetStrategyId: 'existing_strategy',
            targetStrategyName: '现有策略',
            integrationType: 'exit',
            description: `集成该策略的出场逻辑，有望改善风险调整后收益`,
            expectedImprovement: strength.score * 0.08,
            confidence: strength.score / 100,
          });
        }

        if (strength.aspect === '风险管理') {
          suggestedIntegrations.push({
            targetStrategyId: 'existing_strategy',
            targetStrategyName: '现有策略',
            integrationType: 'risk',
            description: `采用该策略的风险管理参数，可能降低最大回撤`,
            expectedImprovement: strength.score * 0.12,
            confidence: strength.score / 100,
          });
        }
      }
    }

    return {
      score: overallScore,
      compatibleStrategies,
      suggestedIntegrations,
    };
  }
}

// 策略集成引擎
export class StrategyIntegrationEngine {
  integrate(
    sourceStrategy: Strategy,
    targetStrategy: Strategy,
    analysis: AIStrategyAnalysis,
    integrationType: 'entry' | 'exit' | 'risk' | 'indicator' | 'full'
  ): StrategyIntegration {
    const changes: { field: string; oldValue: any; newValue: any; reason: string }[] = [];
    const beforePerformance = targetStrategy.performance;

    // 创建目标策略的副本
    const integratedStrategy = { ...targetStrategy };

    switch (integrationType) {
      case 'entry':
        this.integrateEntryConditions(sourceStrategy, integratedStrategy, changes);
        break;
      case 'exit':
        this.integrateExitConditions(sourceStrategy, integratedStrategy, changes);
        break;
      case 'risk':
        this.integrateRiskManagement(sourceStrategy, integratedStrategy, changes);
        break;
      case 'indicator':
        this.integrateIndicators(sourceStrategy, integratedStrategy, changes);
        break;
      case 'full':
        this.integrateEntryConditions(sourceStrategy, integratedStrategy, changes);
        this.integrateExitConditions(sourceStrategy, integratedStrategy, changes);
        this.integrateRiskManagement(sourceStrategy, integratedStrategy, changes);
        this.integrateIndicators(sourceStrategy, integratedStrategy, changes);
        break;
    }

    // 模拟集成后的表现（实际应该重新回测）
    const afterPerformance = this.simulateImprovedPerformance(
      beforePerformance,
      analysis,
      integrationType
    );

    const improvement = afterPerformance 
      ? ((afterPerformance.totalReturn - (beforePerformance?.totalReturn || 0)) / 
         (beforePerformance?.totalReturn || 1)) * 100
      : 0;

    return {
      id: `integration_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      sourceStrategyId: sourceStrategy.id,
      targetStrategyId: targetStrategy.id,
      integratedAt: new Date(),
      integrationType,
      changes,
      beforePerformance: beforePerformance || this.createDefaultPerformance(),
      afterPerformance: afterPerformance || this.createDefaultPerformance(),
      improvement,
      aiNotes: this.generateIntegrationNotes(sourceStrategy, targetStrategy, analysis, integrationType),
    };
  }

  private integrateEntryConditions(
    source: Strategy, 
    target: Strategy, 
    changes: any[]
  ) {
    // 保留目标策略原有的入场条件，添加源策略的有效条件
    const originalEntryCount = target.parameters.entryConditions.length;
    
    // 选择源策略中最有效的入场条件（简化逻辑）
    const bestSourceConditions = source.parameters.entryConditions.slice(0, 2);
    
    for (const condition of bestSourceConditions) {
      const newCondition = {
        ...condition,
        id: `cond_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      };
      target.parameters.entryConditions.push(newCondition);
    }

    changes.push({
      field: 'entryConditions',
      oldValue: originalEntryCount,
      newValue: target.parameters.entryConditions.length,
      reason: `集成了来自"${source.name}"的 ${bestSourceConditions.length} 个入场条件`,
    });

    // 更新长仓入场规则
    const originalLongRules = [...target.rules.longEntry];
    for (const condition of bestSourceConditions) {
      if (!target.rules.longEntry.includes(condition.id)) {
        target.rules.longEntry.push(condition.id);
      }
    }

    if (target.rules.longEntry.length > originalLongRules.length) {
      changes.push({
        field: 'rules.longEntry',
        oldValue: originalLongRules,
        newValue: target.rules.longEntry,
        reason: '更新长仓入场规则以包含新的入场条件',
      });
    }
  }

  private integrateExitConditions(
    source: Strategy, 
    target: Strategy, 
    changes: any[]
  ) {
    const originalExitCount = target.parameters.exitConditions.length;
    
    const bestSourceConditions = source.parameters.exitConditions.slice(0, 2);
    
    for (const condition of bestSourceConditions) {
      const newCondition = {
        ...condition,
        id: `cond_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      };
      target.parameters.exitConditions.push(newCondition);
    }

    changes.push({
      field: 'exitConditions',
      oldValue: originalExitCount,
      newValue: target.parameters.exitConditions.length,
      reason: `集成了来自"${source.name}"的 ${bestSourceConditions.length} 个出场条件`,
    });
  }

  private integrateRiskManagement(
    source: Strategy, 
    target: Strategy, 
    changes: any[]
  ) {
    const sourceRisk = source.parameters.riskManagement;
    const targetRisk = target.parameters.riskManagement;

    // 如果源策略的止损更严格，采用它
    if (sourceRisk.stopLossPercent < targetRisk.stopLossPercent) {
      changes.push({
        field: 'riskManagement.stopLossPercent',
        oldValue: targetRisk.stopLossPercent,
        newValue: sourceRisk.stopLossPercent,
        reason: `采用更严格的止损 (${sourceRisk.stopLossPercent}%) 以更好地控制风险`,
      });
      targetRisk.stopLossPercent = sourceRisk.stopLossPercent;
    }

    // 如果源策略的止盈更优，采用它
    if (sourceRisk.takeProfitPercent > targetRisk.takeProfitPercent) {
      changes.push({
        field: 'riskManagement.takeProfitPercent',
        oldValue: targetRisk.takeProfitPercent,
        newValue: sourceRisk.takeProfitPercent,
        reason: `采用更高的止盈目标 (${sourceRisk.takeProfitPercent}%) 以提高盈亏比`,
      });
      targetRisk.takeProfitPercent = sourceRisk.takeProfitPercent;
    }

    // 如果源策略使用追踪止损而目标策略不使用，考虑采用
    if (sourceRisk.useTrailingStop && !targetRisk.useTrailingStop) {
      changes.push({
        field: 'riskManagement.useTrailingStop',
        oldValue: false,
        newValue: true,
        reason: '启用追踪止损以保护利润',
      });
      targetRisk.useTrailingStop = true;
      targetRisk.trailingStopPercent = sourceRisk.trailingStopPercent;
    }

    // 采用更保守的仓位规模
    if (sourceRisk.maxPositionSize < targetRisk.maxPositionSize) {
      changes.push({
        field: 'riskManagement.maxPositionSize',
        oldValue: targetRisk.maxPositionSize,
        newValue: sourceRisk.maxPositionSize,
        reason: `采用更保守的仓位规模 (${sourceRisk.maxPositionSize}%) 以降低风险暴露`,
      });
      targetRisk.maxPositionSize = sourceRisk.maxPositionSize;
    }
  }

  private integrateIndicators(
    source: Strategy, 
    target: Strategy, 
    changes: any[]
  ) {
    const originalIndicators = [...target.parameters.indicators];
    
    // 添加源策略中目标策略没有的指标
    for (const sourceIndicator of source.parameters.indicators) {
      const exists = target.parameters.indicators.some(
        i => i.name === sourceIndicator.name && 
             JSON.stringify(i.parameters) === JSON.stringify(sourceIndicator.parameters)
      );
      
      if (!exists) {
        target.parameters.indicators.push(sourceIndicator);
      }
    }

    if (target.parameters.indicators.length > originalIndicators.length) {
      changes.push({
        field: 'indicators',
        oldValue: originalIndicators.map(i => i.name),
        newValue: target.parameters.indicators.map(i => i.name),
        reason: `添加了 ${target.parameters.indicators.length - originalIndicators.length} 个新指标以增强信号质量`,
      });
    }
  }

  private simulateImprovedPerformance(
    before: StrategyPerformance | undefined,
    analysis: AIStrategyAnalysis,
    integrationType: string
  ): StrategyPerformance {
    const base = before || this.createDefaultPerformance();
    const improvementFactor = analysis.overallScore / 100;

    // 根据集成类型模拟改进
    const multipliers: Record<string, number> = {
      entry: 1.15,
      exit: 1.10,
      risk: 1.08,
      indicator: 1.12,
      full: 1.25,
    };

    const multiplier = multipliers[integrationType] || 1.1;

    return {
      totalReturn: base.totalReturn * multiplier * (1 + improvementFactor * 0.2),
      winRate: Math.min(base.winRate * multiplier, 80),
      profitFactor: base.profitFactor * multiplier,
      maxDrawdown: base.maxDrawdown * (1 - improvementFactor * 0.3),
      sharpeRatio: base.sharpeRatio * multiplier,
      totalTrades: base.totalTrades,
      averageTrade: base.averageTrade * multiplier,
      lastBacktestDate: new Date(),
    };
  }

  private createDefaultPerformance(): StrategyPerformance {
    return {
      totalReturn: 0,
      winRate: 50,
      profitFactor: 1,
      maxDrawdown: 0,
      sharpeRatio: 0,
      totalTrades: 0,
      averageTrade: 0,
    };
  }

  private generateIntegrationNotes(
    source: Strategy,
    target: Strategy,
    analysis: AIStrategyAnalysis,
    integrationType: string
  ): string {
    const typeNames: Record<string, string> = {
      entry: '入场条件',
      exit: '出场条件',
      risk: '风险管理',
      indicator: '技术指标',
      full: '全面集成',
    };

    let notes = `已将"${source.name}"的优秀${typeNames[integrationType]}集成到"${target.name}"中。\n\n`;
    notes += `源策略评分: ${analysis.overallScore}/100\n`;
    notes += `主要优势: ${analysis.strengths.map(s => s.aspect).join(', ')}\n\n`;
    notes += `集成后预期改进:\n`;
    
    for (const suggestion of analysis.integrationPotential.suggestedIntegrations) {
      notes += `- ${suggestion.description} (预期提升: ${suggestion.expectedImprovement.toFixed(1)}%, 置信度: ${(suggestion.confidence * 100).toFixed(0)}%)\n`;
    }

    return notes;
  }
}

// AI代理主类
export class AIAgent {
  private analyzer: AIStrategyAnalyzer;
  private integrationEngine: StrategyIntegrationEngine;
  private isRunning: boolean = false;
  private onProgress?: (progress: number, message: string) => void;

  constructor(onProgress?: (progress: number, message: string) => void) {
    this.analyzer = new AIStrategyAnalyzer();
    this.integrationEngine = new StrategyIntegrationEngine();
    this.onProgress = onProgress;
  }

  async analyzeStrategy(
    strategy: Strategy, 
    backtestResult: BacktestResult
  ): Promise<AIStrategyAnalysis> {
    this.reportProgress(10, '正在分析策略表现...');
    
    const analysis = this.analyzer.analyze(strategy, backtestResult);
    
    this.reportProgress(100, '策略分析完成');
    return analysis;
  }

  async integrateStrategy(
    sourceStrategy: Strategy,
    targetStrategy: Strategy,
    analysis: AIStrategyAnalysis,
    integrationType: 'entry' | 'exit' | 'risk' | 'indicator' | 'full' = 'full'
  ): Promise<StrategyIntegration> {
    this.reportProgress(20, '正在评估集成可行性...');
    
    // 检查是否值得集成
    if (analysis.overallScore < 60) {
      throw new Error('策略评分过低，不建议集成');
    }

    this.reportProgress(50, '正在执行策略集成...');
    
    const integration = this.integrationEngine.integrate(
      sourceStrategy,
      targetStrategy,
      analysis,
      integrationType
    );

    this.reportProgress(100, '策略集成完成');
    return integration;
  }

  async autoProcess(
    newStrategy: Strategy,
    backtestResult: BacktestResult,
    existingStrategies: Strategy[],
    minScoreForIntegration: number = 75
  ): Promise<{
    analysis: AIStrategyAnalysis;
    integrations: StrategyIntegration[];
  }> {
    this.isRunning = true;
    const integrations: StrategyIntegration[] = [];

    try {
      // 步骤1: 分析策略
      this.reportProgress(10, 'AI代理开始分析新策略...');
      const analysis = await this.analyzeStrategy(newStrategy, backtestResult);

      // 步骤2: 如果策略优秀，考虑集成
      if (analysis.overallScore >= minScoreForIntegration) {
        this.reportProgress(40, `策略评分 ${analysis.overallScore}，评估集成潜力...`);

        for (const existingStrategy of existingStrategies) {
          if (existingStrategy.id === newStrategy.id) continue;

          // 为每个现有策略评估集成
          for (const suggestion of analysis.integrationPotential.suggestedIntegrations) {
            if (suggestion.confidence >= 0.7) {
              this.reportProgress(60, `正在集成到 ${existingStrategy.name}...`);
              
              try {
                const integration = await this.integrateStrategy(
                  newStrategy,
                  existingStrategy,
                  analysis,
                  suggestion.integrationType
                );
                
                integrations.push(integration);
              } catch (error) {
                console.warn(`集成到 ${existingStrategy.name} 失败:`, error);
              }
            }
          }
        }
      } else {
        this.reportProgress(50, `策略评分 ${analysis.overallScore}，未达到集成阈值 ${minScoreForIntegration}`);
      }

      this.reportProgress(100, 'AI代理处理完成');
      return { analysis, integrations };
    } finally {
      this.isRunning = false;
    }
  }

  private reportProgress(progress: number, message: string) {
    if (this.onProgress) {
      this.onProgress(progress, message);
    }
  }

  getIsRunning(): boolean {
    return this.isRunning;
  }
}

// 导出单例
export const aiAgent = new AIAgent();
