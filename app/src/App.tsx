import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
import { Toaster, toast } from 'sonner';
import { 
  Plus, 
  Upload, 
  Brain, 
  BarChart3, 
  Settings, 
  TrendingUp,
  Zap,
  Activity,
  ChevronRight,
  Sparkles,
  GitMerge,
  TrendingDown
} from 'lucide-react';
import { StrategyCard } from '@/components/StrategyCard';
import { StrategyForm } from '@/components/StrategyForm';
import { BacktestResultView } from '@/components/BacktestResult';
import { AIAgentPanel } from '@/components/AIAgentPanel';
import { useStrategyStore } from '@/hooks/useStrategyStore';
import { BacktestService } from '@/lib/backtestEngine';
import type { Strategy, BacktestResult, AIStrategyAnalysis, StrategyIntegration } from '@/types';
import './App.css';

function App() {
  const [strategyFormOpen, setStrategyFormOpen] = useState(false);
  const [editingStrategy, setEditingStrategy] = useState<Strategy | null>(null);
  const [backtestResult, setBacktestResult] = useState<BacktestResult | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AIStrategyAnalysis | null>(null);
  const [integrationDialogOpen, setIntegrationDialogOpen] = useState(false);
  const [currentIntegration, setCurrentIntegration] = useState<StrategyIntegration | null>(null);
  const [importDialogOpen, setImportDialogOpen] = useState(false);
  const [importData, setImportData] = useState('');
  const [activeTab, setActiveTab] = useState('strategies');

  const {
    strategies,
    addStrategy,
    updateStrategy,
    deleteStrategy,
    importStrategy,
    exportStrategy,
    addBacktestResult,

    getAnalysisForStrategy,
    settings,
    defaultBacktestConfig,
  } = useStrategyStore();

  // 处理策略创建/编辑
  const handleStrategySubmit = (strategyData: Omit<Strategy, 'id' | 'createdAt' | 'updatedAt'>) => {
    if (editingStrategy) {
      updateStrategy(editingStrategy.id, strategyData);
      toast.success('策略已更新');
    } else {
      addStrategy(strategyData);
      toast.success('策略已创建');
      
      // 如果开启了自动回测，提示用户
      if (settings.aiPreferences.autoBacktest) {
        toast.info('策略已加入AI代理处理队列');
      }
    }
    setEditingStrategy(null);
  };

  // 运行回测
  const handleRunBacktest = async (strategy: Strategy) => {
    toast.promise(
      async () => {
        const result = await BacktestService.runBacktest(strategy, defaultBacktestConfig);
        addBacktestResult(result);
        
        // 更新策略性能
        updateStrategy(strategy.id, {
          performance: {
            totalReturn: result.metrics.totalReturn,
            winRate: result.metrics.winRate,
            profitFactor: result.metrics.profitFactor,
            maxDrawdown: result.metrics.maxDrawdownPercent,
            sharpeRatio: result.metrics.sharpeRatio,
            totalTrades: result.metrics.totalTrades,
            averageTrade: result.metrics.averageTrade,
            lastBacktestDate: new Date(),
          }
        });
        
        setBacktestResult(result);
        return result;
      },
      {
        loading: `正在为 "${strategy.name}" 运行回测...`,
        success: (result) => `回测完成！收益率: ${result.metrics.totalReturn.toFixed(2)}%`,
        error: (err) => `回测失败: ${err.message}`,
      }
    );
  };

  // 导入策略
  const handleImport = () => {
    try {
      importStrategy(importData);
      setImportDialogOpen(false);
      setImportData('');
      toast.success('策略导入成功');
    } catch (error) {
      toast.error('导入失败: 无效的JSON格式');
    }
  };

  // 导出策略
  const handleExport = (strategy: Strategy) => {
    const data = exportStrategy(strategy.id);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${strategy.name.replace(/\s+/g, '_')}_strategy.json`;
    a.click();
    toast.success('策略已导出');
  };

  // 打开编辑表单
  const openEditForm = (strategy: Strategy) => {
    setEditingStrategy(strategy);
    setStrategyFormOpen(true);
  };

  // 打开创建表单
  const openCreateForm = () => {
    setEditingStrategy(null);
    setStrategyFormOpen(true);
  };

  return (
    <div className="min-h-screen bg-[#05050A] text-white">
      <Toaster 
        position="top-right" 
        theme="dark"
        toastOptions={{
          style: {
            background: '#12121C',
            border: '1px solid #2a2a3c',
            color: '#fff',
          },
        }}
      />

      {/* 头部导航 */}
      <header className="border-b border-[#2a2a3c] bg-[#12121C]/80 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="relative">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500 to-purple-500 flex items-center justify-center">
                  <Brain className="w-6 h-6 text-white" />
                </div>
                <span className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full border-2 border-[#12121C]" />
              </div>
              <div>
                <h1 className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                  CryptoAI Trading
                </h1>
                <p className="text-xs text-gray-500">AI驱动的策略优化系统</p>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 text-sm text-gray-400">
                <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                系统运行中
              </div>
              <Button
                variant="outline"
                size="sm"
                className="border-[#2a2a3c] text-gray-300"
              >
                <Settings className="w-4 h-4 mr-2" />
                设置
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* 主内容区 */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-4 bg-[#12121C] border border-[#2a2a3c]">
            <TabsTrigger value="strategies" className="data-[state=active]:bg-cyan-500/20 data-[state=active]:text-cyan-400">
              <Zap className="w-4 h-4 mr-2" />
              策略管理
            </TabsTrigger>
            <TabsTrigger value="ai_agent" className="data-[state=active]:bg-purple-500/20 data-[state=active]:text-purple-400">
              <Brain className="w-4 h-4 mr-2" />
              AI 代理
            </TabsTrigger>
            <TabsTrigger value="backtests" className="data-[state=active]:bg-green-500/20 data-[state=active]:text-green-400">
              <BarChart3 className="w-4 h-4 mr-2" />
              回测结果
            </TabsTrigger>
            <TabsTrigger value="integrations" className="data-[state=active]:bg-orange-500/20 data-[state=active]:text-orange-400">
              <Activity className="w-4 h-4 mr-2" />
              集成历史
            </TabsTrigger>
          </TabsList>

          {/* 策略管理 */}
          <TabsContent value="strategies" className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold text-white">策略管理</h2>
                <p className="text-gray-400">管理您的交易策略，支持导入和手动创建</p>
              </div>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  onClick={() => setImportDialogOpen(true)}
                  className="border-[#2a2a3c] text-gray-300"
                >
                  <Upload className="w-4 h-4 mr-2" />
                  导入策略
                </Button>
                <Button
                  onClick={openCreateForm}
                  className="bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-600 hover:to-purple-600"
                >
                  <Plus className="w-4 h-4 mr-2" />
                  创建策略
                </Button>
              </div>
            </div>

            {strategies.length === 0 ? (
              <div className="text-center py-20">
                <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-[#1a1a28] flex items-center justify-center">
                  <Sparkles className="w-10 h-10 text-gray-500" />
                </div>
                <h3 className="text-xl font-medium text-white mb-2">还没有策略</h3>
                <p className="text-gray-400 mb-6">创建您的第一个交易策略，让AI帮您优化</p>
                <Button
                  onClick={openCreateForm}
                  className="bg-gradient-to-r from-cyan-500 to-purple-500"
                >
                  <Plus className="w-4 h-4 mr-2" />
                  创建策略
                </Button>
              </div>
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {strategies.map((strategy) => (
                  <StrategyCard
                    key={strategy.id}
                    strategy={strategy}
                    analysis={getAnalysisForStrategy(strategy.id)}
                    onEdit={() => openEditForm(strategy)}
                    onDelete={() => {
                      if (confirm(`确定要删除策略 "${strategy.name}" 吗？`)) {
                        deleteStrategy(strategy.id);
                        toast.success('策略已删除');
                      }
                    }}
                    onRunBacktest={() => handleRunBacktest(strategy)}
                    onToggleActive={() => {
                      updateStrategy(strategy.id, { isActive: !strategy.isActive });
                      toast.success(strategy.isActive ? '策略已暂停' : '策略已激活');
                    }}
                    onExport={() => handleExport(strategy)}
                  />
                ))}
              </div>
            )}
          </TabsContent>

          {/* AI代理 */}
          <TabsContent value="ai_agent" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-1">
                <AIAgentPanel
                  onAnalysisComplete={(analysis) => {
                    setAnalysisResult(analysis);
                    toast.success(`AI分析完成: 评分 ${analysis.overallScore}/100`);
                  }}
                  onIntegrationComplete={(integration) => {
                    setCurrentIntegration(integration);
                    setIntegrationDialogOpen(true);
                  }}
                />
              </div>
              <div className="lg:col-span-2 space-y-6">
                {/* AI分析结果展示 */}
                {analysisResult && (
                  <div className="bg-gradient-to-br from-[#12121C] to-[#0a0a12] border border-[#2a2a3c] rounded-lg p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                        <Brain className="w-5 h-5 text-purple-400" />
                        AI 分析详情
                      </h3>
                      <Badge className="bg-gradient-to-r from-purple-500 to-cyan-500 text-white">
                        评分: {analysisResult.overallScore}/100
                      </Badge>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4 mb-4">
                      <div>
                        <h4 className="text-sm font-medium text-gray-300 mb-2">优势</h4>
                        <ul className="space-y-1">
                          {analysisResult.strengths.map((s, i) => (
                            <li key={i} className="text-sm text-green-400 flex items-center gap-2">
                              <TrendingUp className="w-3 h-3" />
                              {s.aspect}: {s.score.toFixed(0)}分
                            </li>
                          ))}
                        </ul>
                      </div>
                      <div>
                        <h4 className="text-sm font-medium text-gray-300 mb-2">待改进</h4>
                        <ul className="space-y-1">
                          {analysisResult.weaknesses.map((w, i) => (
                            <li key={i} className="text-sm text-red-400 flex items-center gap-2">
                              <TrendingDown className="w-3 h-3" />
                              {w.aspect} ({w.severity})
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>

                    <div>
                      <h4 className="text-sm font-medium text-gray-300 mb-2">AI 建议</h4>
                      <ul className="space-y-1">
                        {analysisResult.recommendations.slice(0, 3).map((rec, i) => (
                          <li key={i} className="text-sm text-gray-400 flex items-start gap-2">
                            <ChevronRight className="w-4 h-4 text-cyan-400 mt-0.5 flex-shrink-0" />
                            {rec}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                )}

                {/* 策略对比 */}
                {strategies.length >= 2 && (
                  <div className="bg-gradient-to-br from-[#12121C] to-[#0a0a12] border border-[#2a2a3c] rounded-lg p-6">
                    <h3 className="text-lg font-semibold text-white mb-4">策略对比</h3>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b border-[#2a2a3c]">
                            <th className="text-left py-2 text-gray-400">策略</th>
                            <th className="text-right py-2 text-gray-400">收益率</th>
                            <th className="text-right py-2 text-gray-400">胜率</th>
                            <th className="text-right py-2 text-gray-400">盈亏比</th>
                            <th className="text-right py-2 text-gray-400">回撤</th>
                            <th className="text-right py-2 text-gray-400">AI评分</th>
                          </tr>
                        </thead>
                        <tbody>
                          {strategies.map(strategy => {
                            const analysis = getAnalysisForStrategy(strategy.id);
                            return (
                              <tr key={strategy.id} className="border-b border-[#2a2a3c]/50">
                                <td className="py-2 text-white">{strategy.name}</td>
                                <td className="text-right py-2">
                                  <span className={strategy.performance?.totalReturn && strategy.performance.totalReturn >= 0 ? 'text-green-400' : 'text-red-400'}>
                                    {strategy.performance?.totalReturn?.toFixed(2) || '0.00'}%
                                  </span>
                                </td>
                                <td className="text-right py-2 text-cyan-400">
                                  {strategy.performance?.winRate?.toFixed(1) || '0.0'}%
                                </td>
                                <td className="text-right py-2 text-purple-400">
                                  {strategy.performance?.profitFactor?.toFixed(2) || '0.00'}
                                </td>
                                <td className="text-right py-2 text-red-400">
                                  {strategy.performance?.maxDrawdown?.toFixed(1) || '0.0'}%
                                </td>
                                <td className="text-right py-2">
                                  {analysis ? (
                                    <Badge className={
                                      analysis.overallScore >= 80 ? 'bg-green-500/20 text-green-400' :
                                      analysis.overallScore >= 60 ? 'bg-yellow-500/20 text-yellow-400' :
                                      'bg-red-500/20 text-red-400'
                                    }>
                                      {analysis.overallScore}
                                    </Badge>
                                  ) : (
                                    <span className="text-gray-500">-</span>
                                  )}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </TabsContent>

          {/* 回测结果 */}
          <TabsContent value="backtests" className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold text-white">回测结果</h2>
                <p className="text-gray-400">查看所有策略的历史回测表现</p>
              </div>
            </div>

            {backtestResult ? (
              <BacktestResultView 
                result={backtestResult} 
                strategyName={strategies.find(s => s.id === backtestResult.strategyId)?.name || 'Unknown'}
              />
            ) : (
              <div className="text-center py-20">
                <BarChart3 className="w-16 h-16 mx-auto mb-4 text-gray-600" />
                <h3 className="text-xl font-medium text-white mb-2">暂无回测结果</h3>
                <p className="text-gray-400">在策略管理页面运行回测查看结果</p>
              </div>
            )}
          </TabsContent>

          {/* 集成历史 */}
          <TabsContent value="integrations" className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold text-white">集成历史</h2>
                <p className="text-gray-400">查看AI代理自动集成的策略变更记录</p>
              </div>
            </div>

            {useStrategyStore.getState().integrations.length === 0 ? (
              <div className="text-center py-20">
                <Activity className="w-16 h-16 mx-auto mb-4 text-gray-600" />
                <h3 className="text-xl font-medium text-white mb-2">暂无集成记录</h3>
                <p className="text-gray-400">启用AI自动集成后会在此显示记录</p>
              </div>
            ) : (
              <div className="space-y-4">
                {useStrategyStore.getState().integrations.map(integration => (
                  <div 
                    key={integration.id}
                    className="bg-gradient-to-br from-[#12121C] to-[#0a0a12] border border-[#2a2a3c] rounded-lg p-6"
                  >
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-cyan-500 flex items-center justify-center">
                          <GitMerge className="w-5 h-5 text-white" />
                        </div>
                        <div>
                          <h3 className="text-lg font-semibold text-white">
                            策略集成
                          </h3>
                          <p className="text-sm text-gray-500">
                            {new Date(integration.integratedAt).toLocaleString()}
                          </p>
                        </div>
                      </div>
                      <Badge className={integration.improvement > 0 
                        ? 'bg-green-500/20 text-green-400' 
                        : 'bg-red-500/20 text-red-400'
                      }>
                        {integration.improvement >= 0 ? '+' : ''}
                        {integration.improvement.toFixed(2)}%
                      </Badge>
                    </div>

                    <div className="grid grid-cols-2 gap-4 mb-4">
                      <div className="bg-[#1a1a28] rounded-lg p-3">
                        <div className="text-xs text-gray-500 mb-1">源策略</div>
                        <div className="text-white font-medium">
                          {strategies.find(s => s.id === integration.sourceStrategyId)?.name || 'Unknown'}
                        </div>
                      </div>
                      <div className="bg-[#1a1a28] rounded-lg p-3">
                        <div className="text-xs text-gray-500 mb-1">目标策略</div>
                        <div className="text-white font-medium">
                          {strategies.find(s => s.id === integration.targetStrategyId)?.name || 'Unknown'}
                        </div>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <h4 className="text-sm font-medium text-gray-300">变更详情</h4>
                      <ul className="space-y-1">
                        {integration.changes.map((change, i) => (
                          <li key={i} className="text-sm text-gray-400 flex items-center gap-2">
                            <ChevronRight className="w-4 h-4 text-cyan-400" />
                            {change.reason}
                          </li>
                        ))}
                      </ul>
                    </div>

                    {integration.aiNotes && (
                      <div className="mt-4 p-3 bg-purple-500/10 rounded-lg border border-purple-500/20">
                        <h4 className="text-sm font-medium text-purple-400 mb-1">AI 备注</h4>
                        <p className="text-sm text-gray-400 whitespace-pre-line">
                          {integration.aiNotes}
                        </p>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </TabsContent>
        </Tabs>
      </main>

      {/* 策略表单对话框 */}
      <StrategyForm
        open={strategyFormOpen}
        onOpenChange={setStrategyFormOpen}
        onSubmit={handleStrategySubmit}
        initialData={editingStrategy || undefined}
      />

      {/* 导入对话框 */}
      <Dialog open={importDialogOpen} onOpenChange={setImportDialogOpen}>
        <DialogContent className="bg-[#12121C] border-[#2a2a3c]">
          <DialogHeader>
            <DialogTitle className="text-white">导入策略</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <textarea
              value={importData}
              onChange={(e) => setImportData(e.target.value)}
              placeholder="粘贴策略JSON数据..."
              rows={10}
              className="w-full bg-[#1a1a28] border border-[#2a2a3c] rounded-lg p-3 text-sm text-white font-mono resize-none"
            />
            <div className="flex justify-end gap-2">
              <Button
                variant="outline"
                onClick={() => setImportDialogOpen(false)}
                className="border-[#2a2a3c] text-gray-300"
              >
                取消
              </Button>
              <Button
                onClick={handleImport}
                disabled={!importData}
                className="bg-gradient-to-r from-cyan-500 to-purple-500"
              >
                <Upload className="w-4 h-4 mr-2" />
                导入
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* 集成详情对话框 */}
      <Dialog open={integrationDialogOpen} onOpenChange={setIntegrationDialogOpen}>
        <DialogContent className="bg-[#12121C] border-[#2a2a3c] max-w-2xl">
          <DialogHeader>
            <DialogTitle className="text-white flex items-center gap-2">
              <GitMerge className="w-5 h-5 text-purple-400" />
              策略集成完成
            </DialogTitle>
          </DialogHeader>
          {currentIntegration && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-[#1a1a28] rounded-lg p-4">
                  <div className="text-sm text-gray-500 mb-1">集成前表现</div>
                  <div className="text-lg font-semibold text-white">
                    收益率: {currentIntegration.beforePerformance.totalReturn.toFixed(2)}%
                  </div>
                  <div className="text-sm text-gray-400">
                    胜率: {currentIntegration.beforePerformance.winRate.toFixed(1)}%
                  </div>
                </div>
                <div className="bg-[#1a1a28] rounded-lg p-4">
                  <div className="text-sm text-gray-500 mb-1">集成后表现（预期）</div>
                  <div className="text-lg font-semibold text-green-400">
                    收益率: {currentIntegration.afterPerformance.totalReturn.toFixed(2)}%
                  </div>
                  <div className="text-sm text-gray-400">
                    胜率: {currentIntegration.afterPerformance.winRate.toFixed(1)}%
                  </div>
                </div>
              </div>

              <div className="p-4 bg-green-500/10 rounded-lg border border-green-500/20">
                <div className="flex items-center justify-between">
                  <span className="text-green-400 font-medium">预期改进</span>
                  <span className="text-2xl font-bold text-green-400">
                    +{currentIntegration.improvement.toFixed(2)}%
                  </span>
                </div>
              </div>

              <div>
                <h4 className="text-sm font-medium text-gray-300 mb-2">变更详情</h4>
                <ul className="space-y-2">
                  {currentIntegration.changes.map((change, i) => (
                    <li key={i} className="text-sm text-gray-400 bg-[#1a1a28] p-2 rounded">
                      {change.reason}
                    </li>
                  ))}
                </ul>
              </div>

              <Button
                onClick={() => setIntegrationDialogOpen(false)}
                className="w-full bg-gradient-to-r from-cyan-500 to-purple-500"
              >
                确认
              </Button>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}

export default App;
