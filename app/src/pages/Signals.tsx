import { useState } from 'react';
import { 
  Signal, 
  Filter, 
  CheckCircle, 
  XCircle, 
  TrendingUp, 
  TrendingDown,
  Minus,
  Brain,
  Zap
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from '@/components/ui/table';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog';
import { Progress } from '@/components/ui/progress';
import { mockTradingSignals } from '@/lib/mockData';
import type { TradingSignal } from '@/types';

function SignalIcon({ type }: { type: string }) {
  if (type === 'BUY') return <TrendingUp className="w-5 h-5 text-emerald-400" />;
  if (type === 'SELL') return <TrendingDown className="w-5 h-5 text-red-400" />;
  return <Minus className="w-5 h-5 text-amber-400" />;
}

function SignalBadge({ type }: { type: string }) {
  const styles = {
    BUY: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
    SELL: 'bg-red-500/10 text-red-400 border-red-500/20',
    HOLD: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
  };
  
  return (
    <Badge variant="outline" className={styles[type as keyof typeof styles]}>
      {type}
    </Badge>
  );
}

function StatusBadge({ status }: { status: string }) {
  const styles = {
    active: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
    executed: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
    expired: 'bg-slate-500/10 text-slate-400 border-slate-500/20',
    cancelled: 'bg-red-500/10 text-red-400 border-red-500/20',
  };
  
  const labels = {
    active: '活跃',
    executed: '已执行',
    expired: '已过期',
    cancelled: '已取消',
  };
  
  return (
    <Badge variant="outline" className={styles[status as keyof typeof styles]}>
      {labels[status as keyof typeof labels]}
    </Badge>
  );
}

function SignalDetailDialog({ signal, open, onClose }: { 
  signal: TradingSignal | null; 
  open: boolean; 
  onClose: () => void;
}) {
  if (!signal) return null;

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="bg-slate-900 border-slate-800 text-white max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3">
            <SignalIcon type={signal.type} />
            <span>{signal.symbol} 交易信号</span>
            <SignalBadge type={signal.type} />
          </DialogTitle>
          <DialogDescription className="text-slate-400">
            生成时间: {new Date(signal.timestamp).toLocaleString()}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 mt-4">
          {/* Confidence */}
          <div>
            <div className="flex justify-between text-sm mb-2">
              <span className="text-slate-400">置信度</span>
              <span className="text-white font-medium">{(signal.confidence * 100).toFixed(0)}%</span>
            </div>
            <Progress value={signal.confidence * 100} className="h-2" />
          </div>

          {/* Price Levels */}
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-slate-800/50 p-3 rounded-lg text-center">
              <p className="text-xs text-slate-400">入场价格</p>
              <p className="text-lg font-bold text-white">${signal.entryPrice.toLocaleString()}</p>
            </div>
            <div className="bg-red-500/10 p-3 rounded-lg text-center">
              <p className="text-xs text-red-400">止损价格</p>
              <p className="text-lg font-bold text-red-400">${signal.stopLoss.toLocaleString()}</p>
            </div>
            <div className="bg-emerald-500/10 p-3 rounded-lg text-center">
              <p className="text-xs text-emerald-400">止盈价格</p>
              <p className="text-lg font-bold text-emerald-400">${signal.takeProfit.toLocaleString()}</p>
            </div>
          </div>

          {/* Risk/Reward */}
          <div className="flex justify-between items-center py-3 border-y border-slate-800">
            <span className="text-slate-400">风险收益比</span>
            <span className="text-white font-medium">
              {((signal.takeProfit - signal.entryPrice) / (signal.entryPrice - signal.stopLoss)).toFixed(2)}
            </span>
          </div>

          {/* Reasoning */}
          <div>
            <h4 className="text-sm font-medium text-white mb-2 flex items-center gap-2">
              <Brain className="w-4 h-4 text-cyan-400" />
              AI 推理依据
            </h4>
            <p className="text-sm text-slate-400 bg-slate-800/50 p-3 rounded-lg">
              {signal.reasoning}
            </p>
          </div>

          {/* Actions */}
          <div className="flex gap-3">
            <Button className="flex-1 bg-emerald-500 hover:bg-emerald-600 text-white">
              <CheckCircle className="w-4 h-4 mr-2" />
              执行信号
            </Button>
            <Button variant="outline" className="flex-1 border-slate-700 text-slate-300 hover:bg-slate-800">
              <XCircle className="w-4 h-4 mr-2" />
              忽略
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

export function Signals() {
  const [signals] = useState<TradingSignal[]>(mockTradingSignals);
  const [selectedSignal, setSelectedSignal] = useState<TradingSignal | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [filter, setFilter] = useState<'all' | 'active' | 'executed' | 'expired'>('all');

  const filteredSignals = signals.filter(s => {
    if (filter === 'all') return true;
    return s.status === filter;
  });

  const stats = {
    total: signals.length,
    active: signals.filter(s => s.status === 'active').length,
    executed: signals.filter(s => s.status === 'executed').length,
    expired: signals.filter(s => s.status === 'expired').length,
    avgConfidence: signals.reduce((sum, s) => sum + s.confidence, 0) / signals.length,
  };

  const handleSignalClick = (signal: TradingSignal) => {
    setSelectedSignal(signal);
    setDialogOpen(true);
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">交易信号</h1>
          <p className="text-slate-400 mt-1">AI 生成的交易信号和推理分析</p>
        </div>
        <Button className="bg-emerald-500 hover:bg-emerald-600 text-white">
          <Zap className="w-4 h-4 mr-2" />
          生成新信号
        </Button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-slate-900/50 border-slate-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">总信号数</p>
                <p className="text-2xl font-bold text-white">{stats.total}</p>
              </div>
              <Signal className="w-8 h-8 text-cyan-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900/50 border-slate-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">活跃信号</p>
                <p className="text-2xl font-bold text-emerald-400">{stats.active}</p>
              </div>
              <div className="w-8 h-8 bg-emerald-500/20 rounded-lg flex items-center justify-center">
                <TrendingUp className="w-5 h-5 text-emerald-400" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900/50 border-slate-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">已执行</p>
                <p className="text-2xl font-bold text-blue-400">{stats.executed}</p>
              </div>
              <CheckCircle className="w-8 h-8 text-blue-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900/50 border-slate-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">平均置信度</p>
                <p className="text-2xl font-bold text-white">{(stats.avgConfidence * 100).toFixed(0)}%</p>
              </div>
              <Brain className="w-8 h-8 text-amber-400" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Signals Table */}
      <Card className="bg-slate-900/50 border-slate-800">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-white flex items-center gap-2">
              <Signal className="w-5 h-5 text-cyan-400" />
              信号列表
            </CardTitle>
            <div className="flex items-center gap-2">
              <Filter className="w-4 h-4 text-slate-400" />
              <select 
                value={filter}
                onChange={(e) => setFilter(e.target.value as any)}
                className="bg-slate-800 border border-slate-700 text-white text-sm rounded-lg px-3 py-1"
              >
                <option value="all">全部</option>
                <option value="active">活跃</option>
                <option value="executed">已执行</option>
                <option value="expired">已过期</option>
              </select>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader className="bg-slate-800/50">
              <TableRow>
                <TableHead className="text-slate-400">信号</TableHead>
                <TableHead className="text-slate-400">交易对</TableHead>
                <TableHead className="text-slate-400">置信度</TableHead>
                <TableHead className="text-slate-400">入场价</TableHead>
                <TableHead className="text-slate-400">止损/止盈</TableHead>
                <TableHead className="text-slate-400">状态</TableHead>
                <TableHead className="text-slate-400">时间</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredSignals.map((signal) => (
                <TableRow 
                  key={signal.id}
                  className="cursor-pointer hover:bg-slate-800/50"
                  onClick={() => handleSignalClick(signal)}
                >
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <SignalIcon type={signal.type} />
                      <SignalBadge type={signal.type} />
                    </div>
                  </TableCell>
                  <TableCell className="font-medium text-white">{signal.symbol}</TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <Progress value={signal.confidence * 100} className="w-16 h-2" />
                      <span className="text-sm text-white">{(signal.confidence * 100).toFixed(0)}%</span>
                    </div>
                  </TableCell>
                  <TableCell className="text-white">${signal.entryPrice.toLocaleString()}</TableCell>
                  <TableCell>
                    <div className="text-xs">
                      <span className="text-red-400">${signal.stopLoss.toLocaleString()}</span>
                      <span className="text-slate-500 mx-1">/</span>
                      <span className="text-emerald-400">${signal.takeProfit.toLocaleString()}</span>
                    </div>
                  </TableCell>
                  <TableCell>
                    <StatusBadge status={signal.status} />
                  </TableCell>
                  <TableCell className="text-slate-400 text-sm">
                    {new Date(signal.timestamp).toLocaleString()}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Signal Detail Dialog */}
      <SignalDetailDialog 
        signal={selectedSignal} 
        open={dialogOpen} 
        onClose={() => setDialogOpen(false)} 
      />
    </div>
  );
}
