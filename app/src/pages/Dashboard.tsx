import { useEffect, useState } from 'react';
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Activity, 
  BarChart3, 
  Target,
  ArrowUpRight,
  ArrowDownRight,
  Zap
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell
} from 'recharts';
import { mockMarketData, mockTradingSignals, mockPositions, mockPerformanceMetrics, generatePnlHistory, generatePriceHistory } from '@/lib/mockData';
import type { MarketData, TradingSignal, Position } from '@/types';

function StatCard({ 
  title, 
  value, 
  change, 
  changeType, 
  icon: Icon,
  subtitle
}: { 
  title: string; 
  value: string; 
  change?: string; 
  changeType?: 'positive' | 'negative' | 'neutral';
  icon: React.ElementType;
  subtitle?: string;
}) {
  return (
    <Card className="bg-slate-900/50 border-slate-800">
      <CardContent className="p-6">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-sm text-slate-400">{title}</p>
            <h3 className="text-2xl font-bold text-white mt-1">{value}</h3>
            {subtitle && <p className="text-xs text-slate-500 mt-1">{subtitle}</p>}
            {change && (
              <div className={`flex items-center gap-1 mt-2 text-sm ${
                changeType === 'positive' ? 'text-emerald-400' : 
                changeType === 'negative' ? 'text-red-400' : 'text-slate-400'
              }`}>
                {changeType === 'positive' ? <ArrowUpRight className="w-4 h-4" /> : 
                 changeType === 'negative' ? <ArrowDownRight className="w-4 h-4" /> : null}
                {change}
              </div>
            )}
          </div>
          <div className="w-10 h-10 bg-slate-800 rounded-lg flex items-center justify-center">
            <Icon className="w-5 h-5 text-slate-400" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function MarketCard({ data }: { data: MarketData }) {
  const isPositive = data.priceChangePercent24h >= 0;
  
  return (
    <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
      <div className="flex items-center gap-3">
        <div className={`w-2 h-2 rounded-full ${isPositive ? 'bg-emerald-500' : 'bg-red-500'}`} />
        <div>
          <p className="text-sm font-medium text-white">{data.symbol}</p>
          <p className="text-xs text-slate-400">Vol: ${(data.volume24h / 1e9).toFixed(2)}B</p>
        </div>
      </div>
      <div className="text-right">
        <p className="text-sm font-medium text-white">${data.price.toLocaleString()}</p>
        <p className={`text-xs ${isPositive ? 'text-emerald-400' : 'text-red-400'}`}>
          {isPositive ? '+' : ''}{data.priceChangePercent24h.toFixed(2)}%
        </p>
      </div>
    </div>
  );
}

function SignalCard({ signal }: { signal: TradingSignal }) {
  const typeColors = {
    BUY: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
    SELL: 'bg-red-500/10 text-red-400 border-red-500/20',
    HOLD: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
  };

  return (
    <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
      <div className="flex items-center gap-3">
        <Badge variant="outline" className={typeColors[signal.type]}>
          {signal.type}
        </Badge>
        <div>
          <p className="text-sm font-medium text-white">{signal.symbol}</p>
          <p className="text-xs text-slate-400">置信度: {(signal.confidence * 100).toFixed(0)}%</p>
        </div>
      </div>
      <div className="text-right">
        <p className="text-sm text-white">${signal.entryPrice.toLocaleString()}</p>
        <p className="text-xs text-slate-500">{new Date(signal.timestamp).toLocaleTimeString()}</p>
      </div>
    </div>
  );
}

function PositionCard({ position }: { position: Position }) {
  const isProfit = position.pnl >= 0;
  
  return (
    <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
      <div className="flex items-center gap-3">
        <div className={`w-2 h-2 rounded-full ${isProfit ? 'bg-emerald-500' : 'bg-red-500'}`} />
        <div>
          <p className="text-sm font-medium text-white">{position.symbol}</p>
          <p className="text-xs text-slate-400">{position.leverage}x 杠杆</p>
        </div>
      </div>
      <div className="text-right">
        <p className={`text-sm font-medium ${isProfit ? 'text-emerald-400' : 'text-red-400'}`}>
          {isProfit ? '+' : ''}${position.pnl.toFixed(2)}
        </p>
        <p className="text-xs text-slate-500">{(position.pnlPercent).toFixed(2)}%</p>
      </div>
    </div>
  );
}

export function Dashboard() {
  const [marketData] = useState<MarketData[]>(mockMarketData);
  const [signals] = useState<TradingSignal[]>(mockTradingSignals.slice(0, 3));
  const [positions] = useState<Position[]>(mockPositions);
  const [metrics] = useState(mockPerformanceMetrics);
  const [pnlHistory, setPnlHistory] = useState<{ date: string; pnl: number; cumulative: number }[]>([]);
  const [btcHistory, setBtcHistory] = useState<{ time: string; price: number }[]>([]);

  useEffect(() => {
    setPnlHistory(generatePnlHistory(30));
    setBtcHistory(generatePriceHistory(67432, 50));
  }, []);

  // Auto-update BTC price
  useEffect(() => {
    const interval = setInterval(() => {
      setBtcHistory(prev => {
        const lastPrice = prev[prev.length - 1]?.price || 67432;
        const change = (Math.random() - 0.48) * 100;
        const newPrice = Math.max(lastPrice + change, 60000);
        return [...prev.slice(1), { time: new Date().toISOString(), price: newPrice }];
      });
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="p-6 space-y-6">
      {/* Page Title */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">仪表盘</h1>
          <p className="text-slate-400 mt-1">实时监控您的交易活动和系统状态</p>
        </div>
        <Button className="bg-emerald-500 hover:bg-emerald-600 text-white">
          <Zap className="w-4 h-4 mr-2" />
          立即交易
        </Button>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="今日盈亏"
          value={`+$${metrics.dailyPnl.toFixed(2)}`}
          change="+5.2%"
          changeType="positive"
          icon={DollarSign}
          subtitle="较昨日"
        />
        <StatCard
          title="总盈亏"
          value={`+$${metrics.totalPnl.toFixed(2)}`}
          change="+12.8%"
          changeType="positive"
          icon={TrendingUp}
          subtitle="所有时间"
        />
        <StatCard
          title="胜率"
          value={`${metrics.winRate.toFixed(1)}%`}
          change="+2.3%"
          changeType="positive"
          icon={Target}
          subtitle={`${metrics.winningTrades}/${metrics.totalTrades} 笔交易`}
        />
        <StatCard
          title="夏普比率"
          value={metrics.sharpeRatio.toFixed(2)}
          change="+0.15"
          changeType="positive"
          icon={Activity}
          subtitle="风险调整后收益"
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* PnL Chart */}
        <Card className="lg:col-span-2 bg-slate-900/50 border-slate-800">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-white flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-emerald-400" />
                盈亏趋势 (30天)
              </CardTitle>
              <Badge variant="outline" className="bg-emerald-500/10 text-emerald-400">
                +${metrics.monthlyPnl.toFixed(2)}
              </Badge>
            </div>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={pnlHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis 
                    dataKey="date" 
                    stroke="#64748b" 
                    fontSize={10}
                    tickFormatter={(value) => value.slice(5)}
                  />
                  <YAxis stroke="#64748b" fontSize={10} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }}
                    labelStyle={{ color: '#94a3b8' }}
                  />
                  <Bar dataKey="pnl" radius={[2, 2, 0, 0]}>
                    {pnlHistory.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.pnl >= 0 ? '#10b981' : '#ef4444'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* BTC Price Chart */}
        <Card className="bg-slate-900/50 border-slate-800">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-white flex items-center gap-2">
                <Activity className="w-5 h-5 text-cyan-400" />
                BTC/USDT
              </CardTitle>
              <Badge variant="outline" className="bg-emerald-500/10 text-emerald-400">
                +1.86%
              </Badge>
            </div>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={btcHistory}>
                  <defs>
                    <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis 
                    dataKey="time" 
                    stroke="#64748b" 
                    fontSize={10}
                    tickFormatter={(value) => new Date(value).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  />
                  <YAxis 
                    stroke="#64748b" 
                    fontSize={10}
                    domain={['dataMin - 500', 'dataMax + 500']}
                    tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
                  />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }}
                    labelStyle={{ color: '#94a3b8' }}
                    formatter={(value: number) => [`$${value.toFixed(2)}`, '价格']}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="price" 
                    stroke="#10b981" 
                    strokeWidth={2}
                    fillOpacity={1} 
                    fill="url(#colorPrice)" 
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Three Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Market Overview */}
        <Card className="bg-slate-900/50 border-slate-800">
          <CardHeader className="pb-2">
            <CardTitle className="text-white flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-emerald-400" />
              市场行情
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {marketData.slice(0, 5).map((data) => (
                <MarketCard key={data.symbol} data={data} />
              ))}
            </div>
            <Button variant="ghost" className="w-full mt-4 text-slate-400 hover:text-white">
              查看全部
            </Button>
          </CardContent>
        </Card>

        {/* Trading Signals */}
        <Card className="bg-slate-900/50 border-slate-800">
          <CardHeader className="pb-2">
            <CardTitle className="text-white flex items-center gap-2">
              <Zap className="w-5 h-5 text-amber-400" />
              最新信号
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {signals.map((signal) => (
                <SignalCard key={signal.id} signal={signal} />
              ))}
            </div>
            <Button variant="ghost" className="w-full mt-4 text-slate-400 hover:text-white">
              查看全部
            </Button>
          </CardContent>
        </Card>

        {/* Positions */}
        <Card className="bg-slate-900/50 border-slate-800">
          <CardHeader className="pb-2">
            <CardTitle className="text-white flex items-center gap-2">
              <TrendingDown className="w-5 h-5 text-cyan-400" />
              当前持仓
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {positions.map((position) => (
                <PositionCard key={position.id} position={position} />
              ))}
            </div>
            <div className="mt-4 pt-4 border-t border-slate-800">
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">总保证金</span>
                <span className="text-white font-medium">
                  ${positions.reduce((sum, p) => sum + p.margin, 0).toFixed(2)}
                </span>
              </div>
              <div className="flex justify-between text-sm mt-2">
                <span className="text-slate-400">总盈亏</span>
                <span className={`font-medium ${positions.reduce((sum, p) => sum + p.pnl, 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {positions.reduce((sum, p) => sum + p.pnl, 0) >= 0 ? '+' : ''}
                  ${positions.reduce((sum, p) => sum + p.pnl, 0).toFixed(2)}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
