import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { 
  TrendingUp, 
  TrendingDown, 
  Brain, 
  Play, 
  Pause, 
  Edit2, 
  Trash2,
  ChevronDown,
  ChevronUp,
  Activity,
  Target,
  Shield,
  Download
} from 'lucide-react';
import type { Strategy, AIStrategyAnalysis } from '@/types';

interface StrategyCardProps {
  strategy: Strategy;
  analysis?: AIStrategyAnalysis;
  isSelected?: boolean;
  onSelect?: () => void;
  onEdit?: () => void;
  onDelete?: () => void;
  onRunBacktest?: () => void;
  onToggleActive?: () => void;
  onExport?: () => void;
}

export function StrategyCard({
  strategy,
  analysis,
  isSelected = false,
  onSelect,
  onEdit,
  onDelete,
  onRunBacktest,
  onToggleActive,
  onExport,
}: StrategyCardProps) {
  const [expanded, setExpanded] = useState(false);

  const getTypeColor = (type: string) => {
    const colors: Record<string, string> = {
      trend_following: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
      mean_reversion: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
      breakout: 'bg-green-500/20 text-green-400 border-green-500/30',
      scalping: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
      custom: 'bg-gray-500/20 text-gray-400 border-gray-500/30',
    };
    return colors[type] || colors.custom;
  };

  const getTypeName = (type: string) => {
    const names: Record<string, string> = {
      trend_following: '趋势跟踪',
      mean_reversion: '均值回归',
      breakout: '突破策略',
      scalping: '剥头皮',
      custom: '自定义',
    };
    return names[type] || type;
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-400';
    if (score >= 60) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <Card 
      className={`relative overflow-hidden transition-all duration-300 cursor-pointer
        ${isSelected ? 'ring-2 ring-cyan-500 ring-offset-2 ring-offset-[#05050A]' : ''}
        hover:shadow-lg hover:shadow-cyan-500/10
        bg-gradient-to-br from-[#12121C] to-[#0a0a12] border-[#2a2a3c]`}
      onClick={onSelect}
    >
      {/* 霓虹边框效果 */}
      <div className="absolute inset-0 rounded-lg opacity-0 hover:opacity-100 transition-opacity duration-500 pointer-events-none">
        <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-cyan-500/20 via-purple-500/20 to-cyan-500/20" />
      </div>

      <CardHeader className="pb-3 relative">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              <CardTitle className="text-lg font-semibold text-white">
                {strategy.name}
              </CardTitle>
              <Badge 
                variant="outline" 
                className={`text-xs ${getTypeColor(strategy.type)}`}
              >
                {getTypeName(strategy.type)}
              </Badge>
              {strategy.source === 'ai_generated' && (
                <Badge className="bg-gradient-to-r from-purple-500 to-cyan-500 text-white text-xs">
                  <Brain className="w-3 h-3 mr-1" />
                  AI
                </Badge>
              )}
            </div>
            <p className="text-sm text-gray-400 line-clamp-2">{strategy.description}</p>
          </div>
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 text-gray-400 hover:text-white"
              onClick={(e) => {
                e.stopPropagation();
                onToggleActive?.();
              }}
            >
              {strategy.isActive ? (
                <Pause className="h-4 w-4" />
              ) : (
                <Play className="h-4 w-4" />
              )}
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 text-gray-400 hover:text-white"
              onClick={(e) => {
                e.stopPropagation();
                onEdit?.();
              }}
            >
              <Edit2 className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 text-gray-400 hover:text-cyan-400"
              onClick={(e) => {
                e.stopPropagation();
                onExport?.();
              }}
            >
              <Download className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 text-gray-400 hover:text-red-400"
              onClick={(e) => {
                e.stopPropagation();
                onDelete?.();
              }}
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="relative">
        {/* 性能指标 */}
        {strategy.performance && (
          <div className="grid grid-cols-4 gap-3 mb-4">
            <div className="bg-[#1a1a28] rounded-lg p-2">
              <div className="text-xs text-gray-500 mb-1">收益率</div>
              <div className={`text-sm font-bold ${strategy.performance.totalReturn >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {strategy.performance.totalReturn >= 0 ? '+' : ''}
                {strategy.performance.totalReturn.toFixed(2)}%
              </div>
            </div>
            <div className="bg-[#1a1a28] rounded-lg p-2">
              <div className="text-xs text-gray-500 mb-1">胜率</div>
              <div className="text-sm font-bold text-cyan-400">
                {strategy.performance.winRate.toFixed(1)}%
              </div>
            </div>
            <div className="bg-[#1a1a28] rounded-lg p-2">
              <div className="text-xs text-gray-500 mb-1">盈亏比</div>
              <div className="text-sm font-bold text-purple-400">
                {strategy.performance.profitFactor.toFixed(2)}
              </div>
            </div>
            <div className="bg-[#1a1a28] rounded-lg p-2">
              <div className="text-xs text-gray-500 mb-1">回撤</div>
              <div className="text-sm font-bold text-red-400">
                {strategy.performance.maxDrawdown.toFixed(1)}%
              </div>
            </div>
          </div>
        )}

        {/* AI分析结果 */}
        {analysis && (
          <div className="mb-4 p-3 bg-gradient-to-r from-purple-500/10 to-cyan-500/10 rounded-lg border border-purple-500/20">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Brain className="w-4 h-4 text-purple-400" />
                <span className="text-sm font-medium text-white">AI 分析评分</span>
              </div>
              <span className={`text-xl font-bold ${getScoreColor(analysis.overallScore)}`}>
                {analysis.overallScore}
              </span>
            </div>
            <Progress 
              value={analysis.overallScore} 
              className="h-2 bg-[#2a2a3c]"
            />
            <div className="flex gap-2 mt-2">
              {analysis.strengths.slice(0, 3).map((strength, idx) => (
                <Badge 
                  key={idx} 
                  variant="outline" 
                  className="text-xs bg-green-500/10 text-green-400 border-green-500/30"
                >
                  <TrendingUp className="w-3 h-3 mr-1" />
                  {strength.aspect}
                </Badge>
              ))}
              {analysis.weaknesses.length > 0 && (
                <Badge 
                  variant="outline" 
                  className="text-xs bg-red-500/10 text-red-400 border-red-500/30"
                >
                  <TrendingDown className="w-3 h-3 mr-1" />
                  {analysis.weaknesses.length} 项待改进
                </Badge>
              )}
            </div>
          </div>
        )}

        {/* 展开详情 */}
        {expanded && (
          <div className="space-y-3 mt-4 pt-4 border-t border-[#2a2a3c]">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="text-xs text-gray-500 mb-1 flex items-center gap-1">
                  <Activity className="w-3 h-3" />
                  入场条件
                </div>
                <div className="text-sm text-gray-300">
                  {strategy.parameters.entryConditions.length} 个条件
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-500 mb-1 flex items-center gap-1">
                  <Target className="w-3 h-3" />
                  出场条件
                </div>
                <div className="text-sm text-gray-300">
                  {strategy.parameters.exitConditions.length} 个条件
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-500 mb-1 flex items-center gap-1">
                  <Shield className="w-3 h-3" />
                  止损 / 止盈
                </div>
                <div className="text-sm text-gray-300">
                  {strategy.parameters.riskManagement.stopLossPercent}% / {strategy.parameters.riskManagement.takeProfitPercent}%
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-500 mb-1 flex items-center gap-1">
                  <TrendingUp className="w-3 h-3" />
                  仓位规模
                </div>
                <div className="text-sm text-gray-300">
                  {strategy.parameters.riskManagement.maxPositionSize}%
                </div>
              </div>
            </div>

            {analysis && (
              <div className="mt-4">
                <h4 className="text-sm font-medium text-white mb-2">AI 建议</h4>
                <ul className="space-y-1">
                  {analysis.recommendations.slice(0, 3).map((rec, idx) => (
                    <li key={idx} className="text-xs text-gray-400 flex items-start gap-2">
                      <span className="text-cyan-400 mt-0.5">•</span>
                      {rec}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {/* 操作栏 */}
        <div className="flex items-center justify-between mt-4 pt-4 border-t border-[#2a2a3c]">
          <Button
            variant="ghost"
            size="sm"
            className="text-gray-400 hover:text-white"
            onClick={(e) => {
              e.stopPropagation();
              setExpanded(!expanded);
            }}
          >
            {expanded ? (
              <>
                <ChevronUp className="w-4 h-4 mr-1" />
                收起
              </>
            ) : (
              <>
                <ChevronDown className="w-4 h-4 mr-1" />
                详情
              </>
            )}
          </Button>
          <Button
            size="sm"
            className="bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-600 hover:to-purple-600 text-white"
            onClick={(e) => {
              e.stopPropagation();
              onRunBacktest?.();
            }}
          >
            <Play className="w-4 h-4 mr-1" />
            运行回测
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
