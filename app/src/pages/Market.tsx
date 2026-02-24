import { useState, useEffect } from 'react';
import { 
  Search, 
  BarChart3, 
  Activity,
  ArrowUpRight,
  ArrowDownRight,
  RefreshCw
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
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
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer
} from 'recharts';
import { mockMarketData, mockOrderBook, mockSentimentData, generatePriceHistory } from '@/lib/mockData';
import type { MarketData, OrderBook, SentimentData } from '@/types';

function OrderBookPanel({ orderBook }: { orderBook: OrderBook }) {
  const maxBidQty = Math.max(...orderBook.bids.map(b => b[1]));
  const maxAskQty = Math.max(...orderBook.asks.map(a => a[1]));

  return (
    <div className="grid grid-cols-2 gap-4">
      {/* Bids */}
      <div>
        <h4 className="text-sm font-medium text-emerald-400 mb-2">买单 (Bids)</h4>
        <div className="space-y-1">
          {orderBook.bids.map(([price, qty], i) => (
            <div key={i} className="relative">
              <div 
                className="absolute right-0 top-0 bottom-0 bg-emerald-500/10 rounded"
                style={{ width: `${(qty / maxBidQty) * 100}%` }}
              />
              <div className="relative flex justify-between text-xs py-1 px-2">
                <span className="text-emerald-400">{price.toLocaleString()}</span>
                <span className="text-slate-400">{qty.toFixed(4)}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Asks */}
      <div>
        <h4 className="text-sm font-medium text-red-400 mb-2">卖单 (Asks)</h4>
        <div className="space-y-1">
          {orderBook.asks.map(([price, qty], i) => (
            <div key={i} className="relative">
              <div 
                className="absolute right-0 top-0 bottom-0 bg-red-500/10 rounded"
                style={{ width: `${(qty / maxAskQty) * 100}%` }}
              />
              <div className="relative flex justify-between text-xs py-1 px-2">
                <span className="text-red-400">{price.toLocaleString()}</span>
                <span className="text-slate-400">{qty.toFixed(4)}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function SentimentPanel({ data }: { data: SentimentData }) {
  const getFearGreedColor = (value: number) => {
    if (value <= 25) return 'text-red-500';
    if (value <= 45) return 'text-orange-500';
    if (value <= 55) return 'text-yellow-500';
    if (value <= 75) return 'text-emerald-500';
    return 'text-green-500';
  };

  const getFearGreedBg = (value: number) => {
    if (value <= 25) return 'bg-red-500';
    if (value <= 45) return 'bg-orange-500';
    if (value <= 55) return 'bg-yellow-500';
    if (value <= 75) return 'bg-emerald-500';
    return 'bg-green-500';
  };

  return (
    <div className="space-y-4">
      {/* Fear & Greed */}
      <div>
        <div className="flex justify-between text-sm mb-1">
          <span className="text-slate-400">恐惧贪婪指数</span>
          <span className={getFearGreedColor(data.fearGreedIndex)}>{data.fearGreedIndex} - {data.fearGreedClassification}</span>
        </div>
        <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
          <div 
            className={`h-full ${getFearGreedBg(data.fearGreedIndex)} transition-all duration-500`}
            style={{ width: `${data.fearGreedIndex}%` }}
          />
        </div>
      </div>

      {/* Funding Rate */}
      <div className="flex justify-between items-center py-2 border-b border-slate-800">
        <span className="text-slate-400">资金费率</span>
        <span className={data.fundingRate >= 0 ? 'text-red-400' : 'text-emerald-400'}>
          {data.fundingRate >= 0 ? '+' : ''}{(data.fundingRate * 100).toFixed(4)}%
        </span>
      </div>

      {/* Long/Short Ratio */}
      <div className="flex justify-between items-center py-2 border-b border-slate-800">
        <span className="text-slate-400">多空比</span>
        <span className="text-white">{data.longShortRatio.toFixed(2)}</span>
      </div>

      {/* Liquidation */}
      <div className="flex justify-between items-center py-2">
        <span className="text-slate-400">24h 清算量</span>
        <span className="text-white">${(data.liquidationVolume / 1e6).toFixed(2)}M</span>
      </div>
    </div>
  );
}

export function Market() {
  const [marketData, setMarketData] = useState<MarketData[]>(mockMarketData);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedSymbol, setSelectedSymbol] = useState('BTCUSDT');
  const [priceHistory, setPriceHistory] = useState<{ time: string; price: number }[]>([]);
  const [orderBook] = useState<OrderBook>(mockOrderBook);
  const [sentimentData] = useState<SentimentData[]>(mockSentimentData);

  const selectedMarket = marketData.find(m => m.symbol === selectedSymbol) || marketData[0];
  const selectedSentiment = sentimentData.find(s => s.symbol === selectedSymbol) || sentimentData[0];

  useEffect(() => {
    setPriceHistory(generatePriceHistory(selectedMarket.price, 100));
  }, [selectedSymbol]);

  // Auto-update prices
  useEffect(() => {
    const interval = setInterval(() => {
      setMarketData(prev => prev.map(m => {
        const change = (Math.random() - 0.5) * m.price * 0.001;
        return { ...m, price: m.price + change };
      }));
      
      setPriceHistory(prev => {
        const lastPrice = prev[prev.length - 1]?.price || selectedMarket.price;
        const change = (Math.random() - 0.5) * 50;
        return [...prev.slice(1), { time: new Date().toISOString(), price: lastPrice + change }];
      });
    }, 2000);
    return () => clearInterval(interval);
  }, [selectedMarket.price]);

  const filteredData = marketData.filter(m => 
    m.symbol.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">市场行情</h1>
          <p className="text-slate-400 mt-1">实时监控加密货币市场数据</p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline" size="sm" className="border-slate-700 text-slate-300">
            <RefreshCw className="w-4 h-4 mr-2" />
            刷新
          </Button>
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Market List */}
        <Card className="lg:col-span-1 bg-slate-900/50 border-slate-800">
          <CardHeader className="pb-2">
            <div className="flex items-center gap-2">
              <Search className="w-4 h-4 text-slate-400" />
              <Input 
                placeholder="搜索交易对..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="bg-slate-800 border-slate-700 text-white placeholder:text-slate-500"
              />
            </div>
          </CardHeader>
          <CardContent className="p-0">
            <div className="max-h-[600px] overflow-auto">
              <Table>
                <TableHeader className="bg-slate-800/50">
                  <TableRow>
                    <TableHead className="text-slate-400">交易对</TableHead>
                    <TableHead className="text-slate-400 text-right">价格</TableHead>
                    <TableHead className="text-slate-400 text-right">24h</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredData.map((data) => {
                    const isPositive = data.priceChangePercent24h >= 0;
                    const isSelected = data.symbol === selectedSymbol;
                    
                    return (
                      <TableRow 
                        key={data.symbol}
                        className={`cursor-pointer hover:bg-slate-800/50 ${isSelected ? 'bg-emerald-500/5' : ''}`}
                        onClick={() => setSelectedSymbol(data.symbol)}
                      >
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <div className={`w-2 h-2 rounded-full ${isPositive ? 'bg-emerald-500' : 'bg-red-500'}`} />
                            <span className="font-medium text-white">{data.symbol}</span>
                          </div>
                        </TableCell>
                        <TableCell className="text-right text-white">
                          ${data.price.toLocaleString()}
                        </TableCell>
                        <TableCell className={`text-right ${isPositive ? 'text-emerald-400' : 'text-red-400'}`}>
                          <div className="flex items-center justify-end gap-1">
                            {isPositive ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
                            {Math.abs(data.priceChangePercent24h).toFixed(2)}%
                          </div>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>

        {/* Chart & Details */}
        <div className="lg:col-span-2 space-y-6">
          {/* Price Chart */}
          <Card className="bg-slate-900/50 border-slate-800">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-white text-2xl">{selectedSymbol}</CardTitle>
                  <div className="flex items-center gap-4 mt-1">
                    <span className="text-3xl font-bold text-white">
                      ${selectedMarket.price.toLocaleString()}
                    </span>
                    <Badge className={selectedMarket.priceChangePercent24h >= 0 ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'}>
                      {selectedMarket.priceChangePercent24h >= 0 ? '+' : ''}
                      {selectedMarket.priceChangePercent24h.toFixed(2)}%
                    </Badge>
                  </div>
                </div>
                <div className="text-right text-sm text-slate-400">
                  <p>24h 高: <span className="text-white">${selectedMarket.high24h.toLocaleString()}</span></p>
                  <p>24h 低: <span className="text-white">${selectedMarket.low24h.toLocaleString()}</span></p>
                  <p>24h 量: <span className="text-white">${(selectedMarket.volume24h / 1e9).toFixed(2)}B</span></p>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={priceHistory}>
                    <defs>
                      <linearGradient id="colorPrice2" x1="0" y1="0" x2="0" y2="1">
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
                      domain={['dataMin - 200', 'dataMax + 200']}
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
                      fill="url(#colorPrice2)" 
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* Order Book & Sentiment */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card className="bg-slate-900/50 border-slate-800">
              <CardHeader className="pb-2">
                <CardTitle className="text-white flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-cyan-400" />
                  订单簿
                </CardTitle>
              </CardHeader>
              <CardContent>
                <OrderBookPanel orderBook={orderBook} />
              </CardContent>
            </Card>

            <Card className="bg-slate-900/50 border-slate-800">
              <CardHeader className="pb-2">
                <CardTitle className="text-white flex items-center gap-2">
                  <Activity className="w-5 h-5 text-amber-400" />
                  市场情绪
                </CardTitle>
              </CardHeader>
              <CardContent>
                <SentimentPanel data={selectedSentiment} />
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
