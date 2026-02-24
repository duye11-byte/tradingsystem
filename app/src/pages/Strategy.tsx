import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  Target, 
  Shield, 
  BarChart3,
  Settings,
  Play,
  Pause,
  RotateCcw,
  CheckCircle,
  AlertTriangle,
  Zap
} from 'lucide-react';
import { XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';



// 策略状态
interface StrategyStatus {
  isRunning: boolean;
  totalSignals: number;
  activePositions: number;
  winRate: number;
  profitFactor: number;
  sharpeRatio: number;
  maxDrawdown: number;
  todayPnl: number;
  totalPnl: number;
}

// 信号记录
interface Signal {
  id: string;
  symbol: string;
  direction: 'LONG' | 'SHORT';
  entryPrice: number;
  stopLoss: number;
  takeProfit: number;
  confidence: number;
  score: number;
  timestamp: string;
  status: 'ACTIVE' | 'EXECUTED' | 'CLOSED' | 'EXPIRED';
  reasons: string[];
}

// 模拟策略数据
const mockStrategyStatus: StrategyStatus = {
  isRunning: true,
  totalSignals: 156,
  activePositions: 3,
  winRate: 68.5,
  profitFactor: 2.1,
  sharpeRatio: 1.85,
  maxDrawdown: 8.2,
  todayPnl: 342.80,
  totalPnl: 12580.50
};

const mockSignals: Signal[] = [
  {
    id: 'SIG001',
    symbol: 'BTCUSDT',
    direction: 'LONG',
    entryPrice: 67200,
    stopLoss: 65800,
    takeProfit: 70200,
    confidence: 85,
    score: 8.2,
    timestamp: '2026-02-24 00:01:30',
    status: 'ACTIVE',
    reasons: ['强上升趋势 (EMA多头排列)', '强趋势 (ADX=32.5)', '成交量放大 (1.45x)']
  },
  {
    id: 'SIG002',
    symbol: 'ETHUSDT',
    direction: 'LONG',
    entryPrice: 3500,
    stopLoss: 3400,
    takeProfit: 3750,
    confidence: 72,
    score: 7.1,
    timestamp: '2026-02-23 23:01:30',
    status: 'EXECUTED',
    reasons: ['中等上升趋势', '多头动量 (MACD)', 'RSI中性区域 (52.3)']
  },
  {
    id: 'SIG003',
    symbol: 'SOLUSDT',
    direction: 'SHORT',
    entryPrice: 180,
    stopLoss: 188,
    takeProfit: 165,
    confidence: 68,
    score: 6.8,
    timestamp: '2026-02-23 22:01:30',
    status: 'CLOSED',
    reasons: ['强下降趋势 (EMA空头排列)', '空头动量', '价格接近阻力位']
  },
  {
    id: 'SIG004',
    symbol: 'BTCUSDT',
    direction: 'LONG',
    entryPrice: 67400,
    stopLoss: 66600,
    takeProfit: 69000,
    confidence: 55,
    score: 5.5,
    timestamp: '2026-02-23 21:01:30',
    status: 'EXPIRED',
    reasons: ['弱趋势 (ADX=18.5)', '成交量正常']
  },
  {
    id: 'SIG005',
    symbol: 'LINKUSDT',
    direction: 'LONG',
    entryPrice: 18.2,
    stopLoss: 17.5,
    takeProfit: 20.0,
    confidence: 78,
    score: 7.8,
    timestamp: '2026-02-23 20:01:30',
    status: 'ACTIVE',
    reasons: ['强上升趋势', '强多头动量 (MACD)', '成交量放大 (1.62x)']
  }
];

// 权益曲线数据
const equityData = [
  { time: '00:00', equity: 10000 },
  { time: '04:00', equity: 10120 },
  { time: '08:00', equity: 10080 },
  { time: '12:00', equity: 10250 },
  { time: '16:00', equity: 10340 },
  { time: '20:00', equity: 10580 },
  { time: '00:00', equity: 10650 },
  { time: '04:00', equity: 10820 },
  { time: '08:00', equity: 10750 },
  { time: '12:00', equity: 10980 },
  { time: '16:00', equity: 11100 },
  { time: '20:00', equity: 11250 },
  { time: '00:00', equity: 11340 },
  { time: '04:00', equity: 11580 },
  { time: '08:00', equity: 11450 },
  { time: '12:00', equity: 11680 },
  { time: '16:00', equity: 11800 },
  { time: '20:00', equity: 11950 },
  { time: '00:00', equity: 12080 },
  { time: '04:00', equity: 12250 },
  { time: '08:00', equity: 12150 },
  { time: '12:00', equity: 12340 },
  { time: '16:00', equity: 12480 },
  { time: '20:00', equity: 12580 }
];

// 策略参数
const strategyParams = {
  trend: {
    emaFast: 9,
    emaSlow: 21,
    emaTrend: 50,
    adxPeriod: 14,
    adxThreshold: 25
  },
  momentum: {
    rsiPeriod: 14,
    rsiOverbought: 70,
    rsiOversold: 30,
    macdFast: 12,
    macdSlow: 26,
    macdSignal: 9
  },
  risk: {
    atrMultiplierSL: 2.0,
    atrMultiplierTP1: 2.5,
    atrMultiplierTP2: 4.0,
    atrMultiplierTP3: 6.0,
    maxPositionSize: 10,
    maxLeverage: 5,
    riskPerTrade: 1
  },
  filters: {
    minConfidence: 65,
    minScore: 6.0,
    useSentiment: true,
    fearGreedThreshold: 20
  }
};

export default function Strategy() {
  const [status, setStatus] = useState<StrategyStatus>(mockStrategyStatus);
  const [signals] = useState<Signal[]>(mockSignals);
  const [isRunning, setIsRunning] = useState(true);
  const [_selectedSignal, setSelectedSignal] = useState<Signal | null>(null);

  const handleToggleStrategy = () => {
    setIsRunning(!isRunning);
    setStatus(prev => ({ ...prev, isRunning: !isRunning }));
  };

  const getDirectionColor = (direction: string) => {
    return direction === 'LONG' ? 'text-emerald-400' : 'text-rose-400';
  };

  const getDirectionBg = (direction: string) => {
    return direction === 'LONG' ? 'bg-emerald-500/20' : 'bg-rose-500/20';
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'ACTIVE': return 'bg-emerald-500';
      case 'EXECUTED': return 'bg-blue-500';
      case 'CLOSED': return 'bg-slate-500';
      case 'EXPIRED': return 'bg-amber-500';
      default: return 'bg-slate-500';
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* 页面标题 */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-100">策略管理</h1>
          <p className="text-slate-400 mt-1">配置和监控 MTF-Momentum 策略</p>
        </div>
        <div className="flex gap-3">
          <Button 
            variant="outline" 
            className="border-slate-700 text-slate-300 hover:bg-slate-800"
          >
            <Settings className="w-4 h-4 mr-2" />
            参数设置
          </Button>
          <Button 
            onClick={handleToggleStrategy}
            className={isRunning ? 'bg-amber-600 hover:bg-amber-700' : 'bg-emerald-600 hover:bg-emerald-700'}
          >
            {isRunning ? <Pause className="w-4 h-4 mr-2" /> : <Play className="w-4 h-4 mr-2" />}
            {isRunning ? '暂停策略' : '启动策略'}
          </Button>
        </div>
      </div>

      {/* 策略状态卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="bg-slate-900 border-slate-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">策略状态</p>
                <div className="flex items-center gap-2 mt-1">
                  <div className={`w-3 h-3 rounded-full ${isRunning ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'}`} />
                  <p className={`text-xl font-bold ${isRunning ? 'text-emerald-400' : 'text-amber-400'}`}>
                    {isRunning ? '运行中' : '已暂停'}
                  </p>
                </div>
              </div>
              <Activity className="w-8 h-8 text-slate-600" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900 border-slate-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">胜率</p>
                <p className="text-2xl font-bold text-emerald-400">{status.winRate}%</p>
                <p className="text-xs text-slate-500">目标: &gt;65%</p>
              </div>
              <Target className="w-8 h-8 text-emerald-600" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900 border-slate-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">盈亏比</p>
                <p className="text-2xl font-bold text-blue-400">{status.profitFactor}</p>
                <p className="text-xs text-slate-500">目标: &gt;1.5</p>
              </div>
              <BarChart3 className="w-8 h-8 text-blue-600" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900 border-slate-800">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">夏普比率</p>
                <p className="text-2xl font-bold text-purple-400">{status.sharpeRatio}</p>
                <p className="text-xs text-slate-500">目标: &gt;1.5</p>
              </div>
              <TrendingUp className="w-8 h-8 text-purple-600" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* 主要内容区域 */}
      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList className="bg-slate-900 border border-slate-800">
          <TabsTrigger value="overview" className="data-[state=active]:bg-slate-800">概览</TabsTrigger>
          <TabsTrigger value="signals" className="data-[state=active]:bg-slate-800">信号历史</TabsTrigger>
          <TabsTrigger value="performance" className="data-[state=active]:bg-slate-800">绩效分析</TabsTrigger>
          <TabsTrigger value="params" className="data-[state=active]:bg-slate-800">参数配置</TabsTrigger>
        </TabsList>

        {/* 概览标签 */}
        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* 权益曲线 */}
            <Card className="lg:col-span-2 bg-slate-900 border-slate-800">
              <CardHeader>
                <CardTitle className="text-slate-100 flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-emerald-500" />
                  权益曲线
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={equityData}>
                      <defs>
                        <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                          <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="time" stroke="#64748b" />
                      <YAxis stroke="#64748b" domain={['dataMin - 500', 'dataMax + 500']} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                        labelStyle={{ color: '#94a3b8' }}
                      />
                      <Area 
                        type="monotone" 
                        dataKey="equity" 
                        stroke="#10b981" 
                        fillOpacity={1} 
                        fill="url(#equityGradient)" 
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* 关键指标 */}
            <Card className="bg-slate-900 border-slate-800">
              <CardHeader>
                <CardTitle className="text-slate-100">关键指标</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-between items-center p-3 bg-slate-800 rounded-lg">
                  <span className="text-slate-400">总信号数</span>
                  <span className="text-xl font-bold text-slate-100">{status.totalSignals}</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-slate-800 rounded-lg">
                  <span className="text-slate-400">活跃持仓</span>
                  <span className="text-xl font-bold text-blue-400">{status.activePositions}</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-slate-800 rounded-lg">
                  <span className="text-slate-400">最大回撤</span>
                  <span className="text-xl font-bold text-amber-400">{status.maxDrawdown}%</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-slate-800 rounded-lg">
                  <span className="text-slate-400">今日盈亏</span>
                  <span className={`text-xl font-bold ${status.todayPnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {status.todayPnl >= 0 ? '+' : ''}{status.todayPnl.toFixed(2)}
                  </span>
                </div>
                <div className="flex justify-between items-center p-3 bg-slate-800 rounded-lg">
                  <span className="text-slate-400">总盈亏</span>
                  <span className={`text-xl font-bold ${status.totalPnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {status.totalPnl >= 0 ? '+' : ''}{status.totalPnl.toFixed(2)}
                  </span>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* 最新信号 */}
          <Card className="bg-slate-900 border-slate-800">
            <CardHeader>
              <CardTitle className="text-slate-100">最新信号</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-slate-800">
                      <th className="text-left py-3 px-4 text-slate-400 font-medium">时间</th>
                      <th className="text-left py-3 px-4 text-slate-400 font-medium">交易对</th>
                      <th className="text-left py-3 px-4 text-slate-400 font-medium">方向</th>
                      <th className="text-left py-3 px-4 text-slate-400 font-medium">入场价</th>
                      <th className="text-left py-3 px-4 text-slate-400 font-medium">止损</th>
                      <th className="text-left py-3 px-4 text-slate-400 font-medium">止盈</th>
                      <th className="text-left py-3 px-4 text-slate-400 font-medium">置信度</th>
                      <th className="text-left py-3 px-4 text-slate-400 font-medium">评分</th>
                      <th className="text-left py-3 px-4 text-slate-400 font-medium">状态</th>
                    </tr>
                  </thead>
                  <tbody>
                    {signals.slice(0, 5).map((signal) => (
                      <tr key={signal.id} className="border-b border-slate-800/50 hover:bg-slate-800/30">
                        <td className="py-3 px-4 text-slate-300">{signal.timestamp}</td>
                        <td className="py-3 px-4 text-slate-100 font-medium">{signal.symbol}</td>
                        <td className="py-3 px-4">
                          <Badge className={`${getDirectionBg(signal.direction)} ${getDirectionColor(signal.direction)} border-0`}>
                            {signal.direction === 'LONG' ? (
                              <TrendingUp className="w-3 h-3 mr-1" />
                            ) : (
                              <TrendingDown className="w-3 h-3 mr-1" />
                            )}
                            {signal.direction}
                          </Badge>
                        </td>
                        <td className="py-3 px-4 text-slate-300">${signal.entryPrice.toLocaleString()}</td>
                        <td className="py-3 px-4 text-rose-400">${signal.stopLoss.toLocaleString()}</td>
                        <td className="py-3 px-4 text-emerald-400">${signal.takeProfit.toLocaleString()}</td>
                        <td className="py-3 px-4">
                          <div className="flex items-center gap-2">
                            <Progress value={signal.confidence} className="w-16 h-2" />
                            <span className="text-slate-300">{signal.confidence}%</span>
                          </div>
                        </td>
                        <td className="py-3 px-4">
                          <span className={`font-bold ${signal.score >= 7 ? 'text-emerald-400' : signal.score >= 6 ? 'text-blue-400' : 'text-amber-400'}`}>
                            {signal.score.toFixed(1)}
                          </span>
                        </td>
                        <td className="py-3 px-4">
                          <div className="flex items-center gap-2">
                            <div className={`w-2 h-2 rounded-full ${getStatusColor(signal.status)}`} />
                            <span className="text-slate-300">{signal.status}</span>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* 信号历史标签 */}
        <TabsContent value="signals" className="space-y-4">
          <Card className="bg-slate-900 border-slate-800">
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="text-slate-100">全部信号</CardTitle>
              <div className="flex gap-2">
                <Button variant="outline" size="sm" className="border-slate-700 text-slate-300">
                  全部
                </Button>
                <Button variant="outline" size="sm" className="border-slate-700 text-slate-300">
                  活跃
                </Button>
                <Button variant="outline" size="sm" className="border-slate-700 text-slate-300">
                  已执行
                </Button>
                <Button variant="outline" size="sm" className="border-slate-700 text-slate-300">
                  已关闭
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {signals.map((signal) => (
                  <div 
                    key={signal.id} 
                    className="p-4 bg-slate-800 rounded-lg hover:bg-slate-750 cursor-pointer transition-colors"
                    onClick={() => setSelectedSignal(signal)}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-4">
                        <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${getDirectionBg(signal.direction)}`}>
                          {signal.direction === 'LONG' ? (
                            <TrendingUp className={`w-5 h-5 ${getDirectionColor(signal.direction)}`} />
                          ) : (
                            <TrendingDown className={`w-5 h-5 ${getDirectionColor(signal.direction)}`} />
                          )}
                        </div>
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="text-slate-100 font-medium">{signal.symbol}</span>
                            <Badge className={`${getDirectionBg(signal.direction)} ${getDirectionColor(signal.direction)} border-0 text-xs`}>
                              {signal.direction}
                            </Badge>
                            <div className={`w-2 h-2 rounded-full ${getStatusColor(signal.status)}`} />
                          </div>
                          <p className="text-sm text-slate-400">{signal.timestamp}</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-6">
                        <div className="text-right">
                          <p className="text-sm text-slate-400">评分</p>
                          <p className={`font-bold ${signal.score >= 7 ? 'text-emerald-400' : signal.score >= 6 ? 'text-blue-400' : 'text-amber-400'}`}>
                            {signal.score.toFixed(1)}/10
                          </p>
                        </div>
                        <div className="text-right">
                          <p className="text-sm text-slate-400">置信度</p>
                          <p className="text-slate-100 font-medium">{signal.confidence}%</p>
                        </div>
                        <div className="text-right">
                          <p className="text-sm text-slate-400">入场价</p>
                          <p className="text-slate-100 font-medium">${signal.entryPrice.toLocaleString()}</p>
                        </div>
                      </div>
                    </div>
                    {signal.reasons.length > 0 && (
                      <div className="mt-3 flex flex-wrap gap-2">
                        {signal.reasons.map((reason, idx) => (
                          <span key={idx} className="text-xs px-2 py-1 bg-slate-700 rounded text-slate-300">
                            {reason}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* 绩效分析标签 */}
        <TabsContent value="performance" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card className="bg-slate-900 border-slate-800">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-slate-400">做多胜率</p>
                    <p className="text-2xl font-bold text-emerald-400">72.3%</p>
                    <p className="text-xs text-slate-500">98笔交易</p>
                  </div>
                  <TrendingUp className="w-8 h-8 text-emerald-600" />
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-900 border-slate-800">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-slate-400">做空胜率</p>
                    <p className="text-2xl font-bold text-rose-400">61.2%</p>
                    <p className="text-xs text-slate-500">58笔交易</p>
                  </div>
                  <TrendingDown className="w-8 h-8 text-rose-600" />
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-900 border-slate-800">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-slate-400">平均盈利</p>
                    <p className="text-2xl font-bold text-emerald-400">$245.80</p>
                    <p className="text-xs text-slate-500">107笔</p>
                  </div>
                  <CheckCircle className="w-8 h-8 text-emerald-600" />
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-900 border-slate-800">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-slate-400">平均亏损</p>
                    <p className="text-2xl font-bold text-rose-400">-$118.50</p>
                    <p className="text-xs text-slate-500">49笔</p>
                  </div>
                  <AlertTriangle className="w-8 h-8 text-rose-600" />
                </div>
              </CardContent>
            </Card>
          </div>

          <Card className="bg-slate-900 border-slate-800">
            <CardHeader>
              <CardTitle className="text-slate-100">出场原因分析</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="p-4 bg-slate-800 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <Shield className="w-5 h-5 text-rose-500" />
                    <span className="text-slate-100 font-medium">止损出场</span>
                  </div>
                  <p className="text-2xl font-bold text-rose-400">42</p>
                  <p className="text-sm text-slate-400">胜率: 0% (预期)</p>
                  <p className="text-sm text-slate-500">平均: -$125.30</p>
                </div>

                <div className="p-4 bg-slate-800 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <Target className="w-5 h-5 text-emerald-500" />
                    <span className="text-slate-100 font-medium">止盈出场</span>
                  </div>
                  <p className="text-2xl font-bold text-emerald-400">89</p>
                  <p className="text-sm text-slate-400">胜率: 100% (预期)</p>
                  <p className="text-sm text-slate-500">平均: $312.50</p>
                </div>

                <div className="p-4 bg-slate-800 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <Zap className="w-5 h-5 text-amber-500" />
                    <span className="text-slate-100 font-medium">追踪止损</span>
                  </div>
                  <p className="text-2xl font-bold text-amber-400">18</p>
                  <p className="text-sm text-slate-400">胜率: 100%</p>
                  <p className="text-sm text-slate-500">平均: $185.20</p>
                </div>

                <div className="p-4 bg-slate-800 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <RotateCcw className="w-5 h-5 text-blue-500" />
                    <span className="text-slate-100 font-medium">信号反转</span>
                  </div>
                  <p className="text-2xl font-bold text-blue-400">7</p>
                  <p className="text-sm text-slate-400">胜率: 57%</p>
                  <p className="text-sm text-slate-500">平均: $45.80</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* 参数配置标签 */}
        <TabsContent value="params" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card className="bg-slate-900 border-slate-800">
              <CardHeader>
                <CardTitle className="text-slate-100 flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-blue-500" />
                  趋势参数
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-between items-center p-3 bg-slate-800 rounded-lg">
                  <span className="text-slate-300">EMA 快线</span>
                  <span className="text-slate-100 font-medium">{strategyParams.trend.emaFast}</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-slate-800 rounded-lg">
                  <span className="text-slate-300">EMA 慢线</span>
                  <span className="text-slate-100 font-medium">{strategyParams.trend.emaSlow}</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-slate-800 rounded-lg">
                  <span className="text-slate-300">EMA 趋势</span>
                  <span className="text-slate-100 font-medium">{strategyParams.trend.emaTrend}</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-slate-800 rounded-lg">
                  <span className="text-slate-300">ADX 周期</span>
                  <span className="text-slate-100 font-medium">{strategyParams.trend.adxPeriod}</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-slate-800 rounded-lg">
                  <span className="text-slate-300">ADX 阈值</span>
                  <span className="text-slate-100 font-medium">{strategyParams.trend.adxThreshold}</span>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-900 border-slate-800">
              <CardHeader>
                <CardTitle className="text-slate-100 flex items-center gap-2">
                  <Activity className="w-5 h-5 text-purple-500" />
                  动量参数
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-between items-center p-3 bg-slate-800 rounded-lg">
                  <span className="text-slate-300">RSI 周期</span>
                  <span className="text-slate-100 font-medium">{strategyParams.momentum.rsiPeriod}</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-slate-800 rounded-lg">
                  <span className="text-slate-300">RSI 超买</span>
                  <span className="text-slate-100 font-medium">{strategyParams.momentum.rsiOverbought}</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-slate-800 rounded-lg">
                  <span className="text-slate-300">RSI 超卖</span>
                  <span className="text-slate-100 font-medium">{strategyParams.momentum.rsiOversold}</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-slate-800 rounded-lg">
                  <span className="text-slate-300">MACD 快线</span>
                  <span className="text-slate-100 font-medium">{strategyParams.momentum.macdFast}</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-slate-800 rounded-lg">
                  <span className="text-slate-300">MACD 慢线</span>
                  <span className="text-slate-100 font-medium">{strategyParams.momentum.macdSlow}</span>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-900 border-slate-800">
              <CardHeader>
                <CardTitle className="text-slate-100 flex items-center gap-2">
                  <Shield className="w-5 h-5 text-rose-500" />
                  风险管理
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-between items-center p-3 bg-slate-800 rounded-lg">
                  <span className="text-slate-300">ATR 止损倍数</span>
                  <span className="text-slate-100 font-medium">{strategyParams.risk.atrMultiplierSL}x</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-slate-800 rounded-lg">
                  <span className="text-slate-300">ATR 止盈1倍数</span>
                  <span className="text-slate-100 font-medium">{strategyParams.risk.atrMultiplierTP1}x</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-slate-800 rounded-lg">
                  <span className="text-slate-300">ATR 止盈2倍数</span>
                  <span className="text-slate-100 font-medium">{strategyParams.risk.atrMultiplierTP2}x</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-slate-800 rounded-lg">
                  <span className="text-slate-300">最大仓位</span>
                  <span className="text-slate-100 font-medium">{strategyParams.risk.maxPositionSize}%</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-slate-800 rounded-lg">
                  <span className="text-slate-300">单笔风险</span>
                  <span className="text-slate-100 font-medium">{strategyParams.risk.riskPerTrade}%</span>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-900 border-slate-800">
              <CardHeader>
                <CardTitle className="text-slate-100 flex items-center gap-2">
                  <Target className="w-5 h-5 text-emerald-500" />
                  信号过滤
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-between items-center p-3 bg-slate-800 rounded-lg">
                  <span className="text-slate-300">最小置信度</span>
                  <span className="text-slate-100 font-medium">{strategyParams.filters.minConfidence}%</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-slate-800 rounded-lg">
                  <span className="text-slate-300">最小评分</span>
                  <span className="text-slate-100 font-medium">{strategyParams.filters.minScore}</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-slate-800 rounded-lg">
                  <span className="text-slate-300">情绪过滤</span>
                  <span className="text-slate-100 font-medium">{strategyParams.filters.useSentiment ? '启用' : '禁用'}</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-slate-800 rounded-lg">
                  <span className="text-slate-300">恐惧贪婪阈值</span>
                  <span className="text-slate-100 font-medium">{strategyParams.filters.fearGreedThreshold}</span>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
