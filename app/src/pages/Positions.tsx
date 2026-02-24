import { useState } from 'react';
import { 
  Wallet, 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle,
  Target,
  Shield,
  DollarSign,
  Percent,
  BarChart3
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog';
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from '@/components/ui/table';
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { mockPositions, generatePnlHistory } from '@/lib/mockData';
import type { Position } from '@/types';

function PositionDetailDialog({ position, open, onClose }: { 
  position: Position | null; 
  open: boolean; 
  onClose: () => void;
}) {
  if (!position) return null;

  const isProfit = position.pnl >= 0;
  const liquidationDistance = ((position.currentPrice - position.liquidationPrice) / position.currentPrice) * 100;

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="bg-slate-900 border-slate-800 text-white max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3">
            <Badge className={position.side === 'long' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'}>
              {position.side === 'long' ? '多头' : '空头'}
            </Badge>
            <span>{position.symbol} 仓位详情</span>
          </DialogTitle>
          <DialogDescription className="text-slate-400">
            仓位ID: {position.id}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 mt-4">
          {/* PnL */}
          <div className={`p-4 rounded-lg ${isProfit ? 'bg-emerald-500/10' : 'bg-red-500/10'}`}>
            <p className="text-sm text-slate-400">当前盈亏</p>
            <p className={`text-3xl font-bold ${isProfit ? 'text-emerald-400' : 'text-red-400'}`}>
              {isProfit ? '+' : ''}${position.pnl.toFixed(2)}
            </p>
            <p className={`text-sm ${isProfit ? 'text-emerald-400' : 'text-red-400'}`}>
              {position.pnlPercent.toFixed(2)}%
            </p>
          </div>

          {/* Price Levels */}
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-slate-800/50 p-3 rounded-lg">
              <p className="text-xs text-slate-400">入场价格</p>
              <p className="text-lg font-bold text-white">${position.entryPrice.toLocaleString()}</p>
            </div>
            <div className="bg-slate-800/50 p-3 rounded-lg">
              <p className="text-xs text-slate-400">当前价格</p>
              <p className="text-lg font-bold text-white">${position.currentPrice.toLocaleString()}</p>
            </div>
          </div>

          {/* Position Info */}
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-slate-800/50 p-3 rounded-lg text-center">
              <p className="text-xs text-slate-400">数量</p>
              <p className="text-lg font-bold text-white">{position.quantity}</p>
            </div>
            <div className="bg-slate-800/50 p-3 rounded-lg text-center">
              <p className="text-xs text-slate-400">杠杆</p>
              <p className="text-lg font-bold text-white">{position.leverage}x</p>
            </div>
            <div className="bg-slate-800/50 p-3 rounded-lg text-center">
              <p className="text-xs text-slate-400">保证金</p>
              <p className="text-lg font-bold text-white">${position.margin.toFixed(2)}</p>
            </div>
          </div>

          {/* Liquidation Warning */}
          <div className="bg-red-500/10 border border-red-500/20 p-3 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <AlertTriangle className="w-4 h-4 text-red-400" />
              <span className="text-sm text-red-400">强平价格</span>
            </div>
            <p className="text-xl font-bold text-red-400">${position.liquidationPrice.toLocaleString()}</p>
            <p className="text-xs text-slate-400 mt-1">
              距离强平: {liquidationDistance.toFixed(2)}%
            </p>
          </div>

          {/* Actions */}
          <div className="flex gap-3">
            <Button className="flex-1 bg-emerald-500 hover:bg-emerald-600 text-white">
              <Target className="w-4 h-4 mr-2" />
              平仓
            </Button>
            <Button variant="outline" className="flex-1 border-slate-700 text-slate-300 hover:bg-slate-800">
              <Shield className="w-4 h-4 mr-2" />
              调整止损
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

export function Positions() {
  const [positions] = useState<Position[]>(mockPositions);
  const [selectedPosition, setSelectedPosition] = useState<Position | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);

  const totalPnl = positions.reduce((sum, p) => sum + p.pnl, 0);
  const totalMargin = positions.reduce((sum, p) => sum + p.margin, 0);
  const profitCount = positions.filter(p => p.pnl > 0).length;
  const lossCount = positions.filter(p => p.pnl < 0).length;

  const pnlHistory = generatePnlHistory(30);

  const allocationData = positions.map(p => ({
    name: p.symbol,
    value: p.margin,
    pnl: p.pnl
  }));

  const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6'];

  const handlePositionClick = (position: Position) => {
    setSelectedPosition(position);
    setDialogOpen(true);
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">仓位管理</h1>
          <p className="text-slate-400 mt-1">查看和管理您的当前持仓</p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline" className="border-slate-700 text-slate-300">
            <Target className="w-4 h-4 mr-2" />
            一键平仓
          </Button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-slate-900/50 border-slate-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">总盈亏</p>
                <p className={`text-2xl font-bold ${totalPnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(2)}
                </p>
              </div>
              <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${totalPnl >= 0 ? 'bg-emerald-500/20' : 'bg-red-500/20'}`}>
                {totalPnl >= 0 ? <TrendingUp className="w-5 h-5 text-emerald-400" /> : <TrendingDown className="w-5 h-5 text-red-400" />}
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900/50 border-slate-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">总保证金</p>
                <p className="text-2xl font-bold text-white">${totalMargin.toFixed(2)}</p>
              </div>
              <DollarSign className="w-8 h-8 text-cyan-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900/50 border-slate-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">盈利/亏损</p>
                <p className="text-2xl font-bold text-white">
                  <span className="text-emerald-400">{profitCount}</span>
                  <span className="text-slate-500 mx-1">/</span>
                  <span className="text-red-400">{lossCount}</span>
                </p>
              </div>
              <Percent className="w-8 h-8 text-amber-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900/50 border-slate-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">持仓数量</p>
                <p className="text-2xl font-bold text-white">{positions.length}</p>
              </div>
              <Wallet className="w-8 h-8 text-purple-400" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* PnL Chart */}
        <Card className="lg:col-span-2 bg-slate-900/50 border-slate-800">
          <CardHeader className="pb-2">
            <CardTitle className="text-white flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-emerald-400" />
              盈亏趋势
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={pnlHistory}>
                  <defs>
                    <linearGradient id="colorPnl" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis 
                    dataKey="date" 
                    stroke="#64748b" 
                    fontSize={10}
                  />
                  <YAxis stroke="#64748b" fontSize={10} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }}
                    labelStyle={{ color: '#94a3b8' }}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="cumulative" 
                    stroke="#10b981" 
                    strokeWidth={2}
                    fillOpacity={1} 
                    fill="url(#colorPnl)" 
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Allocation Chart */}
        <Card className="bg-slate-900/50 border-slate-800">
          <CardHeader className="pb-2">
            <CardTitle className="text-white flex items-center gap-2">
              <Wallet className="w-5 h-5 text-cyan-400" />
              资金分配
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={allocationData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={80}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {allocationData.map((_entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }}
                    formatter={(value: number) => `$${value.toFixed(2)}`}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="flex justify-center gap-4 mt-2">
              {allocationData.map((item, index) => (
                <div key={item.name} className="flex items-center gap-1">
                  <div 
                    className="w-3 h-3 rounded-full" 
                    style={{ backgroundColor: COLORS[index % COLORS.length] }}
                  />
                  <span className="text-xs text-slate-400">{item.name}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Positions Table */}
      <Card className="bg-slate-900/50 border-slate-800">
        <CardHeader className="pb-2">
          <CardTitle className="text-white flex items-center gap-2">
            <Wallet className="w-5 h-5 text-cyan-400" />
            持仓列表
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader className="bg-slate-800/50">
              <TableRow>
                <TableHead className="text-slate-400">交易对</TableHead>
                <TableHead className="text-slate-400">方向</TableHead>
                <TableHead className="text-slate-400">入场价</TableHead>
                <TableHead className="text-slate-400">当前价</TableHead>
                <TableHead className="text-slate-400">数量</TableHead>
                <TableHead className="text-slate-400">杠杆</TableHead>
                <TableHead className="text-slate-400">盈亏</TableHead>
                <TableHead className="text-slate-400">强平价</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {positions.map((position) => {
                const isProfit = position.pnl >= 0;
                return (
                  <TableRow 
                    key={position.id}
                    className="cursor-pointer hover:bg-slate-800/50"
                    onClick={() => handlePositionClick(position)}
                  >
                    <TableCell className="font-medium text-white">{position.symbol}</TableCell>
                    <TableCell>
                      <Badge className={position.side === 'long' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'}>
                        {position.side === 'long' ? '多头' : '空头'}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-white">${position.entryPrice.toLocaleString()}</TableCell>
                    <TableCell className="text-white">${position.currentPrice.toLocaleString()}</TableCell>
                    <TableCell className="text-white">{position.quantity}</TableCell>
                    <TableCell className="text-white">{position.leverage}x</TableCell>
                    <TableCell>
                      <div className={`flex flex-col ${isProfit ? 'text-emerald-400' : 'text-red-400'}`}>
                        <span className="font-medium">
                          {isProfit ? '+' : ''}${position.pnl.toFixed(2)}
                        </span>
                        <span className="text-xs">
                          {position.pnlPercent.toFixed(2)}%
                        </span>
                      </div>
                    </TableCell>
                    <TableCell className="text-red-400">${position.liquidationPrice.toLocaleString()}</TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Position Detail Dialog */}
      <PositionDetailDialog 
        position={selectedPosition} 
        open={dialogOpen} 
        onClose={() => setDialogOpen(false)} 
      />
    </div>
  );
}
