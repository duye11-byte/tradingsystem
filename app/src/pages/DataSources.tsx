import { useState } from 'react';
import { 
  Database, 
  CheckCircle, 
  XCircle, 
  AlertCircle, 
  RefreshCw,
  Settings,
  Activity,
  Clock,
  TrendingUp,
  Zap,
  Globe,
  MessageSquare
} from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { mockDataSources } from '@/lib/mockData';
import type { DataSource } from '@/types';

const typeIcons = {
  price: TrendingUp,
  onchain: Globe,
  sentiment: Activity,
  news: MessageSquare,
};

const typeColors = {
  price: 'text-emerald-400',
  onchain: 'text-blue-400',
  sentiment: 'text-amber-400',
  news: 'text-purple-400',
};

function StatusBadge({ status }: { status: string }) {
  const styles = {
    active: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
    inactive: 'bg-slate-500/10 text-slate-400 border-slate-500/20',
    error: 'bg-red-500/10 text-red-400 border-red-500/20',
  };
  
  const icons = {
    active: CheckCircle,
    inactive: XCircle,
    error: AlertCircle,
  };
  
  const Icon = icons[status as keyof typeof icons];
  
  return (
    <Badge variant="outline" className={`${styles[status as keyof typeof styles]} flex items-center gap-1`}>
      <Icon className="w-3 h-3" />
      {status === 'active' ? '正常' : status === 'inactive' ? '停用' : '错误'}
    </Badge>
  );
}

function DataSourceDialog({ source, open, onClose }: { 
  source: DataSource | null; 
  open: boolean; 
  onClose: () => void;
}) {
  if (!source) return null;

  const Icon = typeIcons[source.type];

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="bg-slate-900 border-slate-800 text-white max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3">
            <div className={`w-10 h-10 bg-slate-800 rounded-lg flex items-center justify-center ${typeColors[source.type]}`}>
              <Icon className="w-5 h-5" />
            </div>
            <div>
              <span>{source.name}</span>
              <p className="text-sm text-slate-400 font-normal">{source.type.toUpperCase()} 数据源</p>
            </div>
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-4 mt-4">
          {/* Status */}
          <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
            <span className="text-slate-400">状态</span>
            <StatusBadge status={source.status} />
          </div>

          {/* Metrics */}
          <div className="grid grid-cols-2 gap-4">
            <div className="p-3 bg-slate-800/50 rounded-lg">
              <p className="text-xs text-slate-400">延迟</p>
              <p className="text-lg font-bold text-white">{source.latency}ms</p>
            </div>
            <div className="p-3 bg-slate-800/50 rounded-lg">
              <p className="text-xs text-slate-400">错误率</p>
              <p className="text-lg font-bold text-white">{(source.errorRate * 100).toFixed(2)}%</p>
            </div>
          </div>

          {/* Last Update */}
          <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
            <span className="text-slate-400">最后更新</span>
            <span className="text-white">{new Date(source.lastUpdate).toLocaleString()}</span>
          </div>

          {/* Config */}
          <div>
            <p className="text-sm text-slate-400 mb-2">配置参数</p>
            <div className="bg-slate-800/50 p-3 rounded-lg">
              <pre className="text-xs text-slate-300 overflow-auto">
                {JSON.stringify(source.config, null, 2)}
              </pre>
            </div>
          </div>

          {/* Actions */}
          <div className="flex gap-3">
            <Button className="flex-1 bg-emerald-500 hover:bg-emerald-600 text-white">
              <RefreshCw className="w-4 h-4 mr-2" />
              测试连接
            </Button>
            <Button variant="outline" className="flex-1 border-slate-700 text-slate-300 hover:bg-slate-800">
              <Settings className="w-4 h-4 mr-2" />
              编辑配置
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

export function DataSources() {
  const [dataSources, setDataSources] = useState<DataSource[]>(mockDataSources);
  const [selectedSource, setSelectedSource] = useState<DataSource | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);

  const toggleSource = (id: string) => {
    setDataSources(prev => prev.map(ds => 
      ds.id === id ? { ...ds, status: ds.status === 'active' ? 'inactive' : 'active' as any } : ds
    ));
  };

  const handleSourceClick = (source: DataSource) => {
    setSelectedSource(source);
    setDialogOpen(true);
  };

  const stats = {
    total: dataSources.length,
    active: dataSources.filter(ds => ds.status === 'active').length,
    inactive: dataSources.filter(ds => ds.status === 'inactive').length,
    error: dataSources.filter(ds => ds.status === 'error').length,
    avgLatency: Math.round(dataSources.filter(ds => ds.status === 'active').reduce((sum, ds) => sum + ds.latency, 0) / dataSources.filter(ds => ds.status === 'active').length),
  };

  const groupedSources = {
    price: dataSources.filter(ds => ds.type === 'price'),
    onchain: dataSources.filter(ds => ds.type === 'onchain'),
    sentiment: dataSources.filter(ds => ds.type === 'sentiment'),
    news: dataSources.filter(ds => ds.type === 'news'),
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">数据源管理</h1>
          <p className="text-slate-400 mt-1">配置和管理系统数据源</p>
        </div>
        <Button className="bg-emerald-500 hover:bg-emerald-600 text-white">
          <Zap className="w-4 h-4 mr-2" />
          添加数据源
        </Button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-slate-900/50 border-slate-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">总数据源</p>
                <p className="text-2xl font-bold text-white">{stats.total}</p>
              </div>
              <Database className="w-8 h-8 text-cyan-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900/50 border-slate-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">正常运行</p>
                <p className="text-2xl font-bold text-emerald-400">{stats.active}</p>
              </div>
              <CheckCircle className="w-8 h-8 text-emerald-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900/50 border-slate-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">停用/错误</p>
                <p className="text-2xl font-bold text-red-400">{stats.inactive + stats.error}</p>
              </div>
              <AlertCircle className="w-8 h-8 text-red-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900/50 border-slate-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">平均延迟</p>
                <p className="text-2xl font-bold text-white">{stats.avgLatency}ms</p>
              </div>
              <Clock className="w-8 h-8 text-amber-400" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Data Sources by Type */}
      <Tabs defaultValue="price" className="space-y-4">
        <TabsList className="bg-slate-800">
          <TabsTrigger value="price" className="data-[state=active]:bg-slate-700 flex items-center gap-2">
            <TrendingUp className="w-4 h-4" />
            价格数据
          </TabsTrigger>
          <TabsTrigger value="onchain" className="data-[state=active]:bg-slate-700 flex items-center gap-2">
            <Globe className="w-4 h-4" />
            链上数据
          </TabsTrigger>
          <TabsTrigger value="sentiment" className="data-[state=active]:bg-slate-700 flex items-center gap-2">
            <Activity className="w-4 h-4" />
            情绪数据
          </TabsTrigger>
          <TabsTrigger value="news" className="data-[state=active]:bg-slate-700 flex items-center gap-2">
            <MessageSquare className="w-4 h-4" />
            新闻数据
          </TabsTrigger>
        </TabsList>

        {Object.entries(groupedSources).map(([type, sources]) => (
          <TabsContent key={type} value={type} className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {sources.map((source) => {
                const Icon = typeIcons[source.type];
                return (
                  <Card 
                    key={source.id} 
                    className="bg-slate-900/50 border-slate-800 cursor-pointer hover:border-slate-700 transition-colors"
                    onClick={() => handleSourceClick(source)}
                  >
                    <CardContent className="p-4">
                      <div className="flex items-start justify-between">
                        <div className="flex items-start gap-3">
                          <div className={`w-10 h-10 bg-slate-800 rounded-lg flex items-center justify-center ${typeColors[source.type]}`}>
                            <Icon className="w-5 h-5" />
                          </div>
                          <div>
                            <h3 className="font-medium text-white">{source.name}</h3>
                            <p className="text-xs text-slate-400 mt-1">
                              延迟: {source.latency}ms | 错误率: {(source.errorRate * 100).toFixed(2)}%
                            </p>
                            <div className="flex items-center gap-2 mt-2">
                              <Clock className="w-3 h-3 text-slate-500" />
                              <span className="text-xs text-slate-500">
                                {new Date(source.lastUpdate).toLocaleTimeString()}
                              </span>
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <StatusBadge status={source.status} />
                          <Switch 
                            checked={source.status === 'active'}
                            onCheckedChange={() => toggleSource(source.id)}
                            onClick={(e) => e.stopPropagation()}
                          />
                        </div>
                      </div>

                      {/* Health Bar */}
                      <div className="mt-4">
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-slate-400">健康度</span>
                          <span className={source.errorRate < 0.01 ? 'text-emerald-400' : source.errorRate < 0.05 ? 'text-amber-400' : 'text-red-400'}>
                            {((1 - source.errorRate) * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                          <div 
                            className={`h-full transition-all duration-500 ${
                              source.errorRate < 0.01 ? 'bg-emerald-500' : 
                              source.errorRate < 0.05 ? 'bg-amber-500' : 'bg-red-500'
                            }`}
                            style={{ width: `${(1 - source.errorRate) * 100}%` }}
                          />
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </TabsContent>
        ))}
      </Tabs>

      {/* Data Source Detail Dialog */}
      <DataSourceDialog 
        source={selectedSource} 
        open={dialogOpen} 
        onClose={() => setDialogOpen(false)} 
      />
    </div>
  );
}
