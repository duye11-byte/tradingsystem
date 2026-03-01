import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { 
  TrendingUp, 
  Target, 
  Shield, 
  BarChart3,
  DollarSign,
  Download
} from 'lucide-react';
import type { BacktestResult as BacktestResultType } from '@/types';

interface BacktestResultProps {
  result: BacktestResultType;
  strategyName: string;
}

export function BacktestResultView({ result, strategyName }: BacktestResultProps) {
  const metrics = result.metrics;

  const getScoreColor = (value: number, threshold: number, higherIsBetter = true) => {
    if (higherIsBetter) {
      if (value >= threshold * 1.5) return 'text-green-400';
      if (value >= threshold) return 'text-yellow-400';
      return 'text-red-400';
    } else {
      if (value <= threshold * 0.5) return 'text-green-400';
      if (value <= threshold) return 'text-yellow-400';
      return 'text-red-400';
    }
  };

  const formatNumber = (num: number, decimals = 2) => {
    if (isNaN(num) || !isFinite(num)) return 'N/A';
    return num.toFixed(decimals);
  };

  const exportResults = () => {
    const data = JSON.stringify(result, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `backtest_${strategyName}_${new Date().toISOString().split('T')[0]}.json`;
    a.click();
  };

  return (
    <Card className="bg-gradient-to-br from-[#12121C] to-[#0a0a12] border-[#2a2a3c]">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lg text-white flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-cyan-400" />
              回测结果
            </CardTitle>
            <p className="text-sm text-gray-500 mt-1">
              {strategyName} • {new Date(result.completedAt).toLocaleString()}
            </p>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={exportResults}
            className="border-[#2a2a3c] text-gray-300 hover:text-white"
          >
            <Download className="w-4 h-4 mr-1" />
            导出
          </Button>
        </div>
      </CardHeader>

      <CardContent>
        <Tabs defaultValue="overview">
          <TabsList className="grid w-full grid-cols-4 bg-[#1a1a28]">
            <TabsTrigger value="overview">概览</TabsTrigger>
            <TabsTrigger value="trades">交易记录</TabsTrigger>
            <TabsTrigger value="equity">权益曲线</TabsTrigger>
            <TabsTrigger value="metrics">详细指标</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-4">
            {/* 核心指标卡片 */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-[#1a1a28] rounded-lg p-4 border border-[#2a2a3c]">
                <div className="flex items-center gap-2 mb-2">
                  <DollarSign className="w-4 h-4 text-cyan-400" />
                  <span className="text-xs text-gray-500">总收益率</span>
                </div>
                <div className={`text-2xl font-bold ${metrics.totalReturn >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {metrics.totalReturn >= 0 ? '+' : ''}{formatNumber(metrics.totalReturn)}%
                </div>
                <Progress 
                  value={Math.min(Math.abs(metrics.totalReturn) / 100 * 50, 100)} 
                  className="h-1 mt-2 bg-[#2a2a3c]"
                />
              </div>

              <div className="bg-[#1a1a28] rounded-lg p-4 border border-[#2a2a3c]">
                <div className="flex items-center gap-2 mb-2">
                  <Target className="w-4 h-4 text-purple-400" />
                  <span className="text-xs text-gray-500">胜率</span>
                </div>
                <div className={`text-2xl font-bold ${getScoreColor(metrics.winRate, 50)}`}>
                  {formatNumber(metrics.winRate)}%
                </div>
                <Progress 
                  value={metrics.winRate} 
                  className="h-1 mt-2 bg-[#2a2a3c]"
                />
              </div>

              <div className="bg-[#1a1a28] rounded-lg p-4 border border-[#2a2a3c]">
                <div className="flex items-center gap-2 mb-2">
                  <TrendingUp className="w-4 h-4 text-green-400" />
                  <span className="text-xs text-gray-500">盈亏比</span>
                </div>
                <div className={`text-2xl font-bold ${getScoreColor(metrics.profitFactor, 1.5)}`}>
                  {formatNumber(metrics.profitFactor)}
                </div>
                <Progress 
                  value={Math.min(metrics.profitFactor / 3 * 100, 100)} 
                  className="h-1 mt-2 bg-[#2a2a3c]"
                />
              </div>

              <div className="bg-[#1a1a28] rounded-lg p-4 border border-[#2a2a3c]">
                <div className="flex items-center gap-2 mb-2">
                  <Shield className="w-4 h-4 text-red-400" />
                  <span className="text-xs text-gray-500">最大回撤</span>
                </div>
                <div className={`text-2xl font-bold ${getScoreColor(metrics.maxDrawdownPercent, 15, false)}`}>
                  {formatNumber(metrics.maxDrawdownPercent)}%
                </div>
                <Progress 
                  value={metrics.maxDrawdownPercent} 
                  className="h-1 mt-2 bg-[#2a2a3c]"
                />
              </div>
            </div>

            {/* 统计摘要 */}
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div className="bg-[#1a1a28] rounded-lg p-3">
                <div className="text-xs text-gray-500 mb-1">总交易次数</div>
                <div className="text-lg font-semibold text-white">{metrics.totalTrades}</div>
              </div>
              <div className="bg-[#1a1a28] rounded-lg p-3">
                <div className="text-xs text-gray-500 mb-1">盈利交易</div>
                <div className="text-lg font-semibold text-green-400">{metrics.winningTrades}</div>
              </div>
              <div className="bg-[#1a1a28] rounded-lg p-3">
                <div className="text-xs text-gray-500 mb-1">亏损交易</div>
                <div className="text-lg font-semibold text-red-400">{metrics.losingTrades}</div>
              </div>
              <div className="bg-[#1a1a28] rounded-lg p-3">
                <div className="text-xs text-gray-500 mb-1">平均盈利</div>
                <div className="text-lg font-semibold text-green-400">${formatNumber(metrics.averageWin)}</div>
              </div>
              <div className="bg-[#1a1a28] rounded-lg p-3">
                <div className="text-xs text-gray-500 mb-1">平均亏损</div>
                <div className="text-lg font-semibold text-red-400">${formatNumber(metrics.averageLoss)}</div>
              </div>
              <div className="bg-[#1a1a28] rounded-lg p-3">
                <div className="text-xs text-gray-500 mb-1">平均每笔交易</div>
                <div className={`text-lg font-semibold ${metrics.averageTrade >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  ${formatNumber(metrics.averageTrade)}
                </div>
              </div>
            </div>

            {/* 风险调整指标 */}
            <div className="bg-gradient-to-r from-purple-500/10 to-cyan-500/10 rounded-lg p-4 border border-purple-500/20">
              <h4 className="text-sm font-medium text-white mb-3">风险调整指标</h4>
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <div className="text-xs text-gray-500 mb-1">夏普比率</div>
                  <div className={`text-xl font-bold ${getScoreColor(metrics.sharpeRatio, 1)}`}>
                    {formatNumber(metrics.sharpeRatio)}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-gray-500 mb-1">索提诺比率</div>
                  <div className={`text-xl font-bold ${getScoreColor(metrics.sortinoRatio, 1)}`}>
                    {formatNumber(metrics.sortinoRatio)}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-gray-500 mb-1">卡尔玛比率</div>
                  <div className={`text-xl font-bold ${getScoreColor(metrics.calmarRatio, 1)}`}>
                    {formatNumber(metrics.calmarRatio)}
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="trades">
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {result.trades.map((trade, index) => (
                <div 
                  key={trade.id}
                  className={`p-3 rounded-lg border ${
                    trade.pnl > 0 
                      ? 'bg-green-500/5 border-green-500/20' 
                      : trade.pnl < 0 
                        ? 'bg-red-500/5 border-red-500/20'
                        : 'bg-[#1a1a28] border-[#2a2a3c]'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <Badge 
                        variant="outline" 
                        className={trade.side === 'long' 
                          ? 'text-green-400 border-green-500/30' 
                          : 'text-red-400 border-red-500/30'
                        }
                      >
                        {trade.side === 'long' ? '做多' : '做空'}
                      </Badge>
                      <span className="text-sm text-gray-400">
                        #{index + 1}
                      </span>
                      {trade.exitReason && (
                        <Badge variant="outline" className="text-xs text-gray-400">
                          {trade.exitReason === 'stop_loss' && '止损'}
                          {trade.exitReason === 'take_profit' && '止盈'}
                          {trade.exitReason === 'signal' && '信号'}
                          {trade.exitReason === 'end_of_test' && '测试结束'}
                        </Badge>
                      )}
                    </div>
                    <div className={`text-sm font-semibold ${
                      trade.pnl > 0 ? 'text-green-400' : trade.pnl < 0 ? 'text-red-400' : 'text-gray-400'
                    }`}>
                      {trade.pnl > 0 ? '+' : ''}${formatNumber(trade.pnl)}
                      <span className="text-xs ml-1">({trade.pnlPercent >= 0 ? '+' : ''}{formatNumber(trade.pnlPercent)}%)</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
                    <span>入场: ${formatNumber(trade.entryPrice)}</span>
                    {trade.exitPrice && <span>出场: ${formatNumber(trade.exitPrice)}</span>}
                    <span>数量: {formatNumber(trade.size, 4)}</span>
                    {trade.entryTime && (
                      <span>
                        {new Date(trade.entryTime).toLocaleDateString()}
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="equity">
            <div className="bg-[#1a1a28] rounded-lg p-4 h-96">
              {/* 简化的权益曲线可视化 */}
              <div className="relative h-full">
                <svg className="w-full h-full" viewBox="0 0 100 50" preserveAspectRatio="none">
                  {/* 网格线 */}
                  {[0, 25, 50, 75, 100].map((y) => (
                    <line
                      key={y}
                      x1="0"
                      y1={y}
                      x2="100"
                      y2={y}
                      stroke="#2a2a3c"
                      strokeWidth="0.2"
                    />
                  ))}
                  
                  {/* 权益曲线 */}
                  <polyline
                    fill="none"
                    stroke="#00E0FF"
                    strokeWidth="0.3"
                    points={result.equityCurve.map((point, index) => {
                      const x = (index / (result.equityCurve.length - 1)) * 100;
                      const maxEquity = Math.max(...result.equityCurve.map(e => e.equity));
                      const minEquity = Math.min(...result.equityCurve.map(e => e.equity));
                      const range = maxEquity - minEquity || 1;
                      const y = 50 - ((point.equity - minEquity) / range) * 40 - 5;
                      return `${x},${y}`;
                    }).join(' ')}
                  />
                  
                  {/* 填充区域 */}
                  <polygon
                    fill="url(#equityGradient)"
                    opacity="0.3"
                    points={`
                      0,50
                      ${result.equityCurve.map((point, index) => {
                        const x = (index / (result.equityCurve.length - 1)) * 100;
                        const maxEquity = Math.max(...result.equityCurve.map(e => e.equity));
                        const minEquity = Math.min(...result.equityCurve.map(e => e.equity));
                        const range = maxEquity - minEquity || 1;
                        const y = 50 - ((point.equity - minEquity) / range) * 40 - 5;
                        return `${x},${y}`;
                      }).join(' ')}
                      100,50
                    `}
                  />
                  
                  <defs>
                    <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#00E0FF" />
                      <stop offset="100%" stopColor="#00E0FF" stopOpacity="0" />
                    </linearGradient>
                  </defs>
                </svg>
                
                {/* 标签 */}
                <div className="absolute left-0 top-0 text-xs text-gray-500">
                  ${formatNumber(Math.max(...result.equityCurve.map(e => e.equity)))}
                </div>
                <div className="absolute left-0 bottom-0 text-xs text-gray-500">
                  ${formatNumber(Math.min(...result.equityCurve.map(e => e.equity)))}
                </div>
                <div className="absolute right-0 bottom-0 text-xs text-gray-500">
                  {result.equityCurve.length} 个点
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="metrics">
            <div className="grid grid-cols-2 gap-4">
              {Object.entries(metrics).map(([key, value]) => {
                const labels: Record<string, string> = {
                  totalReturn: '总收益率',
                  totalTrades: '总交易次数',
                  winningTrades: '盈利交易次数',
                  losingTrades: '亏损交易次数',
                  winRate: '胜率',
                  averageWin: '平均盈利',
                  averageLoss: '平均亏损',
                  profitFactor: '盈亏比',
                  maxDrawdown: '最大回撤',
                  maxDrawdownPercent: '最大回撤百分比',
                  sharpeRatio: '夏普比率',
                  sortinoRatio: '索提诺比率',
                  calmarRatio: '卡尔玛比率',
                  averageTrade: '平均每笔交易',
                  averageTradePercent: '平均每笔交易百分比',
                };
                
                return (
                  <div key={key} className="bg-[#1a1a28] rounded-lg p-3 flex justify-between items-center">
                    <span className="text-sm text-gray-400">{labels[key] || key}</span>
                    <span className="text-sm font-mono text-white">
                      {typeof value === 'number' ? formatNumber(value) : value}
                      {key.includes('Percent') || key.includes('Rate') || key === 'totalReturn' ? '%' : ''}
                    </span>
                  </div>
                );
              })}
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
