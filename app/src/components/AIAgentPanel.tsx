import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { 
  Brain, 
  Play, 
  Pause, 
  Settings
} from 'lucide-react';
import { useStrategyStore } from '@/hooks/useStrategyStore';
import { AIAgent } from '@/lib/aiAgent';
import { BacktestService } from '@/lib/backtestEngine';
import type { Strategy, AIStrategyAnalysis, StrategyIntegration } from '@/types';

interface AIAgentPanelProps {
  onAnalysisComplete?: (analysis: AIStrategyAnalysis) => void;
  onIntegrationComplete?: (integration: StrategyIntegration) => void;
}

export function AIAgentPanel({ onAnalysisComplete, onIntegrationComplete }: AIAgentPanelProps) {
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentTask, setCurrentTask] = useState<string>('');
  const [logs, setLogs] = useState<string[]>([]);
  
  const { 
    strategies, 
    aiAgentState, 
    setAIAgentState, 
    addAIAnalysis, 
    addIntegration,
    addBacktestResult,
    updateStrategy,
    settings,
    defaultBacktestConfig,
  } = useStrategyStore();

  const addLog = (message: string) => {
    setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${message}`]);
  };

  const runAIAgent = async () => {
    if (isRunning) return;
    
    setIsRunning(true);
    setProgress(0);
    setLogs([]);
    
    const agent = new AIAgent((p, task) => {
      setProgress(p);
      setCurrentTask(task);
      addLog(task);
    });

    try {
      // 获取待处理的策略
      const pendingStrategies = strategies.filter(s => 
        aiAgentState.pendingStrategies.includes(s.id)
      );

      if (pendingStrategies.length === 0) {
        addLog('没有待处理的策略');
        setIsRunning(false);
        return;
      }

      for (const strategy of pendingStrategies) {
        addLog(`开始处理策略: ${strategy.name}`);

        // 步骤1: 运行回测
        addLog('运行回测...');
        const backtestResult = await BacktestService.runBacktest(
          strategy,
          defaultBacktestConfig
        );
        addBacktestResult(backtestResult);
        addLog(`回测完成: ${backtestResult.metrics.totalTrades} 笔交易, 收益率 ${backtestResult.metrics.totalReturn.toFixed(2)}%`);

        // 更新策略性能
        updateStrategy(strategy.id, {
          performance: {
            totalReturn: backtestResult.metrics.totalReturn,
            winRate: backtestResult.metrics.winRate,
            profitFactor: backtestResult.metrics.profitFactor,
            maxDrawdown: backtestResult.metrics.maxDrawdownPercent,
            sharpeRatio: backtestResult.metrics.sharpeRatio,
            totalTrades: backtestResult.metrics.totalTrades,
            averageTrade: backtestResult.metrics.averageTrade,
            lastBacktestDate: new Date(),
          }
        });

        // 步骤2: AI分析
        addLog('AI正在分析策略...');
        const analysis = await agent.analyzeStrategy(strategy, backtestResult);
        addAIAnalysis(analysis);
        onAnalysisComplete?.(analysis);
        addLog(`分析完成: 评分 ${analysis.overallScore}/100`);

        // 步骤3: 如果策略优秀且开启了自动集成，执行集成
        if (analysis.overallScore >= settings.aiPreferences.minScoreForIntegration &&
            settings.aiPreferences.autoIntegrate) {
          addLog('策略评分优秀，开始评估集成潜力...');
          
          const existingStrategies = strategies.filter(s => s.id !== strategy.id);
          
          for (const existingStrategy of existingStrategies) {
            try {
              const integration = await agent.integrateStrategy(
                strategy,
                existingStrategy,
                analysis,
                'full'
              );
              
              addIntegration(integration);
              onIntegrationComplete?.(integration);
              addLog(`成功集成到 ${existingStrategy.name}: 预期改进 ${integration.improvement.toFixed(2)}%`);
            } catch (error) {
              addLog(`集成到 ${existingStrategy.name} 失败: ${error}`);
            }
          }
        }

        // 从待处理列表中移除
        setAIAgentState({
          pendingStrategies: aiAgentState.pendingStrategies.filter(id => id !== strategy.id),
          analyzedStrategies: [...aiAgentState.analyzedStrategies, strategy.id],
        });
      }

      addLog('AI代理处理完成');
    } catch (error) {
      addLog(`错误: ${error}`);
    } finally {
      setIsRunning(false);
      setProgress(100);
      setCurrentTask('完成');
    }
  };

  const processSingleStrategy = async (strategy: Strategy) => {
    if (isRunning) return;
    
    setIsRunning(true);
    setProgress(0);
    setLogs([]);
    
    const agent = new AIAgent((p, task) => {
      setProgress(p);
      setCurrentTask(task);
      addLog(task);
    });

    try {
      addLog(`开始处理策略: ${strategy.name}`);

      // 运行回测
      addLog('运行回测...');
      const backtestResult = await BacktestService.runBacktest(
        strategy,
        defaultBacktestConfig
      );
      addBacktestResult(backtestResult);
      addLog(`回测完成: ${backtestResult.metrics.totalTrades} 笔交易`);

      // 更新策略性能
      updateStrategy(strategy.id, {
        performance: {
          totalReturn: backtestResult.metrics.totalReturn,
          winRate: backtestResult.metrics.winRate,
          profitFactor: backtestResult.metrics.profitFactor,
          maxDrawdown: backtestResult.metrics.maxDrawdownPercent,
          sharpeRatio: backtestResult.metrics.sharpeRatio,
          totalTrades: backtestResult.metrics.totalTrades,
          averageTrade: backtestResult.metrics.averageTrade,
          lastBacktestDate: new Date(),
        }
      });

      // AI分析
      addLog('AI正在分析策略...');
      const analysis = await agent.analyzeStrategy(strategy, backtestResult);
      addAIAnalysis(analysis);
      onAnalysisComplete?.(analysis);
      addLog(`分析完成: 评分 ${analysis.overallScore}/100`);

      // 如果优秀，询问是否集成
      if (analysis.overallScore >= settings.aiPreferences.minScoreForIntegration) {
        addLog('策略评分优秀，可以集成到现有策略');
      }

    } catch (error) {
      addLog(`错误: ${error}`);
    } finally {
      setIsRunning(false);
      setProgress(100);
    }
  };

  return (
    <Card className="bg-gradient-to-br from-[#12121C] to-[#0a0a12] border-[#2a2a3c]">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="relative">
              <Brain className="w-6 h-6 text-purple-400" />
              {isRunning && (
                <span className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full animate-pulse" />
              )}
            </div>
            <div>
              <CardTitle className="text-lg text-white">AI 策略代理</CardTitle>
              <p className="text-xs text-gray-500">
                自动回测、分析和优化策略
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge 
              variant="outline" 
              className={isRunning 
                ? 'bg-green-500/10 text-green-400 border-green-500/30' 
                : 'bg-gray-500/10 text-gray-400 border-gray-500/30'
              }
            >
              {isRunning ? '运行中' : '就绪'}
            </Badge>
            <Button
              variant="outline"
              size="icon"
              className="h-8 w-8 border-[#2a2a3c]"
            >
              <Settings className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* 状态概览 */}
        <div className="grid grid-cols-3 gap-3">
          <div className="bg-[#1a1a28] rounded-lg p-3 text-center">
            <div className="text-2xl font-bold text-cyan-400">
              {aiAgentState.pendingStrategies.length}
            </div>
            <div className="text-xs text-gray-500">待处理</div>
          </div>
          <div className="bg-[#1a1a28] rounded-lg p-3 text-center">
            <div className="text-2xl font-bold text-purple-400">
              {aiAgentState.analyzedStrategies.length}
            </div>
            <div className="text-xs text-gray-500">已分析</div>
          </div>
          <div className="bg-[#1a1a28] rounded-lg p-3 text-center">
            <div className="text-2xl font-bold text-green-400">
              {useStrategyStore.getState().integrations.length}
            </div>
            <div className="text-xs text-gray-500">已集成</div>
          </div>
        </div>

        {/* 进度条 */}
        {isRunning && (
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">{currentTask}</span>
              <span className="text-cyan-400">{progress}%</span>
            </div>
            <Progress 
              value={progress} 
              className="h-2 bg-[#2a2a3c]"
            />
          </div>
        )}

        {/* 控制按钮 */}
        <div className="flex gap-2">
          <Button
            className="flex-1 bg-gradient-to-r from-purple-500 to-cyan-500 hover:from-purple-600 hover:to-cyan-600 text-white"
            onClick={runAIAgent}
            disabled={isRunning || aiAgentState.pendingStrategies.length === 0}
          >
            {isRunning ? (
              <>
                <Pause className="w-4 h-4 mr-2" />
                处理中...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 mr-2" />
                运行 AI 代理
              </>
            )}
          </Button>
        </div>

        {/* 待处理策略列表 */}
        {aiAgentState.pendingStrategies.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-sm font-medium text-gray-300">待处理策略</h4>
            <div className="space-y-1">
              {aiAgentState.pendingStrategies.map(id => {
                const strategy = strategies.find(s => s.id === id);
                if (!strategy) return null;
                return (
                  <div 
                    key={id}
                    className="flex items-center justify-between p-2 bg-[#1a1a28] rounded text-sm"
                  >
                    <span className="text-gray-300">{strategy.name}</span>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 text-xs text-cyan-400 hover:text-cyan-300"
                      onClick={() => processSingleStrategy(strategy)}
                      disabled={isRunning}
                    >
                      立即处理
                    </Button>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* 日志输出 */}
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-gray-300">运行日志</h4>
          <ScrollArea className="h-40 bg-[#0a0a12] rounded-lg border border-[#2a2a3c] p-3">
            <div className="space-y-1">
              {logs.length === 0 ? (
                <p className="text-xs text-gray-500 italic">等待启动...</p>
              ) : (
                logs.map((log, index) => (
                  <p key={index} className="text-xs text-gray-400 font-mono">
                    {log}
                  </p>
                ))
              )}
            </div>
          </ScrollArea>
        </div>

        {/* AI 配置 */}
        <div className="p-3 bg-[#1a1a28] rounded-lg space-y-2">
          <h4 className="text-sm font-medium text-gray-300">AI 配置</h4>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="flex justify-between">
              <span className="text-gray-500">自动回测</span>
              <span className={settings.aiPreferences.autoBacktest ? 'text-green-400' : 'text-gray-400'}>
                {settings.aiPreferences.autoBacktest ? '开启' : '关闭'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">自动集成</span>
              <span className={settings.aiPreferences.autoIntegrate ? 'text-green-400' : 'text-gray-400'}>
                {settings.aiPreferences.autoIntegrate ? '开启' : '关闭'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">集成阈值</span>
              <span className="text-cyan-400">{settings.aiPreferences.minScoreForIntegration} 分</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">集成模式</span>
              <span className="text-purple-400">
                {settings.aiPreferences.preferredIntegrationType === 'conservative' && '保守'}
                {settings.aiPreferences.preferredIntegrationType === 'balanced' && '平衡'}
                {settings.aiPreferences.preferredIntegrationType === 'aggressive' && '激进'}
              </span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
