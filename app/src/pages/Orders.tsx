import { useState } from 'react';
import { 
  ListOrdered, 
  Search, 
  X, 
  CheckCircle, 
  Clock, 
  TrendingUp,
  RefreshCw
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog';
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from '@/components/ui/table';
import { mockOrders } from '@/lib/mockData';
import type { Order } from '@/types';

function OrderTypeBadge({ type }: { type: string }) {
  const styles = {
    market: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
    limit: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
    stop_loss: 'bg-red-500/10 text-red-400 border-red-500/20',
    take_profit: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
  };
  
  const labels = {
    market: '市价',
    limit: '限价',
    stop_loss: '止损',
    take_profit: '止盈',
  };
  
  return (
    <Badge variant="outline" className={styles[type as keyof typeof styles]}>
      {labels[type as keyof typeof labels]}
    </Badge>
  );
}

function OrderStatusBadge({ status }: { status: string }) {
  const styles = {
    pending: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
    filled: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
    partially_filled: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
    cancelled: 'bg-slate-500/10 text-slate-400 border-slate-500/20',
    rejected: 'bg-red-500/10 text-red-400 border-red-500/20',
  };
  
  const labels = {
    pending: '待执行',
    filled: '已成交',
    partially_filled: '部分成交',
    cancelled: '已取消',
    rejected: '已拒绝',
  };
  
  return (
    <Badge variant="outline" className={styles[status as keyof typeof styles]}>
      {labels[status as keyof typeof labels]}
    </Badge>
  );
}

function SideBadge({ side }: { side: string }) {
  return (
    <Badge className={side === 'buy' ? 'bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30' : 'bg-red-500/20 text-red-400 hover:bg-red-500/30'}>
      {side === 'buy' ? '买入' : '卖出'}
    </Badge>
  );
}

function OrderDetailDialog({ order, open, onClose }: { 
  order: Order | null; 
  open: boolean; 
  onClose: () => void;
}) {
  if (!order) return null;

  const fillPercentage = (order.filledQuantity / order.quantity) * 100;

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="bg-slate-900 border-slate-800 text-white max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3">
            <SideBadge side={order.side} />
            <span>{order.symbol} 订单详情</span>
          </DialogTitle>
          <DialogDescription className="text-slate-400">
            订单ID: {order.id}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 mt-4">
          {/* Order Info */}
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-slate-800/50 p-3 rounded-lg">
              <p className="text-xs text-slate-400">订单类型</p>
              <p className="text-sm font-medium text-white mt-1">
                <OrderTypeBadge type={order.type} />
              </p>
            </div>
            <div className="bg-slate-800/50 p-3 rounded-lg">
              <p className="text-xs text-slate-400">订单状态</p>
              <p className="text-sm font-medium text-white mt-1">
                <OrderStatusBadge status={order.status} />
              </p>
            </div>
          </div>

          {/* Price & Quantity */}
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-slate-800/50 p-3 rounded-lg text-center">
              <p className="text-xs text-slate-400">价格</p>
              <p className="text-lg font-bold text-white">${order.price.toLocaleString()}</p>
            </div>
            <div className="bg-slate-800/50 p-3 rounded-lg text-center">
              <p className="text-xs text-slate-400">数量</p>
              <p className="text-lg font-bold text-white">{order.quantity}</p>
            </div>
            <div className="bg-slate-800/50 p-3 rounded-lg text-center">
              <p className="text-xs text-slate-400">总金额</p>
              <p className="text-lg font-bold text-white">${order.total.toLocaleString()}</p>
            </div>
          </div>

          {/* Fill Progress */}
          <div>
            <div className="flex justify-between text-sm mb-2">
              <span className="text-slate-400">成交进度</span>
              <span className="text-white">{order.filledQuantity} / {order.quantity} ({fillPercentage.toFixed(1)}%)</span>
            </div>
            <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
              <div 
                className="h-full bg-emerald-500 transition-all duration-500"
                style={{ width: `${fillPercentage}%` }}
              />
            </div>
          </div>

          {/* Timestamps */}
          <div className="space-y-2 py-3 border-y border-slate-800 text-sm">
            <div className="flex justify-between">
              <span className="text-slate-400">创建时间</span>
              <span className="text-white">{new Date(order.timestamp).toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">更新时间</span>
              <span className="text-white">{new Date(order.updatedAt).toLocaleString()}</span>
            </div>
          </div>

          {/* Actions */}
          {order.status === 'pending' || order.status === 'partially_filled' ? (
            <div className="flex gap-3">
              <Button variant="outline" className="flex-1 border-red-500/50 text-red-400 hover:bg-red-500/10">
                <X className="w-4 h-4 mr-2" />
                取消订单
              </Button>
            </div>
          ) : null}
        </div>
      </DialogContent>
    </Dialog>
  );
}

export function Orders() {
  const [orders] = useState<Order[]>(mockOrders);
  const [selectedOrder, setSelectedOrder] = useState<Order | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [activeTab, setActiveTab] = useState('all');

  const filteredOrders = orders.filter(order => {
    const matchesSearch = order.symbol.toLowerCase().includes(searchTerm.toLowerCase());
    
    if (activeTab === 'all') return matchesSearch;
    if (activeTab === 'open') return matchesSearch && (order.status === 'pending' || order.status === 'partially_filled');
    if (activeTab === 'filled') return matchesSearch && order.status === 'filled';
    if (activeTab === 'cancelled') return matchesSearch && (order.status === 'cancelled' || order.status === 'rejected');
    
    return matchesSearch;
  });

  const stats = {
    total: orders.length,
    open: orders.filter(o => o.status === 'pending' || o.status === 'partially_filled').length,
    filled: orders.filter(o => o.status === 'filled').length,
    cancelled: orders.filter(o => o.status === 'cancelled' || o.status === 'rejected').length,
    totalVolume: orders.reduce((sum, o) => sum + o.total, 0),
  };

  const handleOrderClick = (order: Order) => {
    setSelectedOrder(order);
    setDialogOpen(true);
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">订单管理</h1>
          <p className="text-slate-400 mt-1">查看和管理您的所有订单</p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline" size="sm" className="border-slate-700 text-slate-300">
            <RefreshCw className="w-4 h-4 mr-2" />
            刷新
          </Button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-slate-900/50 border-slate-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">总订单数</p>
                <p className="text-2xl font-bold text-white">{stats.total}</p>
              </div>
              <ListOrdered className="w-8 h-8 text-cyan-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900/50 border-slate-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">待执行</p>
                <p className="text-2xl font-bold text-amber-400">{stats.open}</p>
              </div>
              <Clock className="w-8 h-8 text-amber-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900/50 border-slate-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">已成交</p>
                <p className="text-2xl font-bold text-emerald-400">{stats.filled}</p>
              </div>
              <CheckCircle className="w-8 h-8 text-emerald-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900/50 border-slate-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">总交易额</p>
                <p className="text-2xl font-bold text-white">${stats.totalVolume.toLocaleString()}</p>
              </div>
              <TrendingUp className="w-8 h-8 text-purple-400" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Orders Table */}
      <Card className="bg-slate-900/50 border-slate-800">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-white flex items-center gap-2">
              <ListOrdered className="w-5 h-5 text-cyan-400" />
              订单列表
            </CardTitle>
            <div className="flex items-center gap-3">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                <Input 
                  placeholder="搜索交易对..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10 bg-slate-800 border-slate-700 text-white placeholder:text-slate-500 w-48"
                />
              </div>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="bg-slate-800 mb-4">
              <TabsTrigger value="all" className="data-[state=active]:bg-slate-700">全部</TabsTrigger>
              <TabsTrigger value="open" className="data-[state=active]:bg-slate-700">待执行</TabsTrigger>
              <TabsTrigger value="filled" className="data-[state=active]:bg-slate-700">已成交</TabsTrigger>
              <TabsTrigger value="cancelled" className="data-[state=active]:bg-slate-700">已取消</TabsTrigger>
            </TabsList>

            <Table>
              <TableHeader className="bg-slate-800/50">
                <TableRow>
                  <TableHead className="text-slate-400">方向</TableHead>
                  <TableHead className="text-slate-400">交易对</TableHead>
                  <TableHead className="text-slate-400">类型</TableHead>
                  <TableHead className="text-slate-400">价格</TableHead>
                  <TableHead className="text-slate-400">数量</TableHead>
                  <TableHead className="text-slate-400">已成交</TableHead>
                  <TableHead className="text-slate-400">状态</TableHead>
                  <TableHead className="text-slate-400">时间</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredOrders.map((order) => (
                  <TableRow 
                    key={order.id}
                    className="cursor-pointer hover:bg-slate-800/50"
                    onClick={() => handleOrderClick(order)}
                  >
                    <TableCell>
                      <SideBadge side={order.side} />
                    </TableCell>
                    <TableCell className="font-medium text-white">{order.symbol}</TableCell>
                    <TableCell>
                      <OrderTypeBadge type={order.type} />
                    </TableCell>
                    <TableCell className="text-white">${order.price.toLocaleString()}</TableCell>
                    <TableCell className="text-white">{order.quantity}</TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <span className="text-white">{order.filledQuantity}</span>
                        <span className="text-slate-500">/</span>
                        <span className="text-slate-400">{order.quantity}</span>
                      </div>
                    </TableCell>
                    <TableCell>
                      <OrderStatusBadge status={order.status} />
                    </TableCell>
                    <TableCell className="text-slate-400 text-sm">
                      {new Date(order.timestamp).toLocaleString()}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Tabs>
        </CardContent>
      </Card>

      {/* Order Detail Dialog */}
      <OrderDetailDialog 
        order={selectedOrder} 
        open={dialogOpen} 
        onClose={() => setDialogOpen(false)} 
      />
    </div>
  );
}
