import { useState } from 'react';
import { 
  FileText, 
  Search, 
  Download, 
  Trash2,
  Info,
  AlertTriangle,
  AlertCircle,
  Bug,
  Clock,
  RefreshCw
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from '@/components/ui/table';
import { mockSystemLogs } from '@/lib/mockData';
import type { SystemLog } from '@/types';

const levelIcons = {
  info: Info,
  warning: AlertTriangle,
  error: AlertCircle,
  debug: Bug,
};

const levelColors = {
  info: 'text-blue-400 bg-blue-500/10 border-blue-500/20',
  warning: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
  error: 'text-red-400 bg-red-500/10 border-red-500/20',
  debug: 'text-slate-400 bg-slate-500/10 border-slate-500/20',
};

const levelLabels = {
  info: '信息',
  warning: '警告',
  error: '错误',
  debug: '调试',
};

function LevelBadge({ level }: { level: string }) {
  const Icon = levelIcons[level as keyof typeof levelIcons];
  
  return (
    <Badge variant="outline" className={`${levelColors[level as keyof typeof levelColors]} flex items-center gap-1`}>
      <Icon className="w-3 h-3" />
      {levelLabels[level as keyof typeof levelLabels]}
    </Badge>
  );
}

export function Logs() {
  const [logs] = useState<SystemLog[]>(mockSystemLogs);
  const [searchTerm, setSearchTerm] = useState('');
  const [activeTab, setActiveTab] = useState('all');

  const filteredLogs = logs.filter(log => {
    const matchesSearch = 
      log.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
      log.source.toLowerCase().includes(searchTerm.toLowerCase());
    
    if (activeTab === 'all') return matchesSearch;
    return matchesSearch && log.level === activeTab;
  });

  const stats = {
    total: logs.length,
    info: logs.filter(l => l.level === 'info').length,
    warning: logs.filter(l => l.level === 'warning').length,
    error: logs.filter(l => l.level === 'error').length,
    debug: logs.filter(l => l.level === 'debug').length,
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">系统日志</h1>
          <p className="text-slate-400 mt-1">查看系统运行日志和事件记录</p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline" size="sm" className="border-slate-700 text-slate-300">
            <Download className="w-4 h-4 mr-2" />
            导出
          </Button>
          <Button variant="outline" size="sm" className="border-slate-700 text-slate-300">
            <Trash2 className="w-4 h-4 mr-2" />
            清空
          </Button>
          <Button variant="outline" size="sm" className="border-slate-700 text-slate-300">
            <RefreshCw className="w-4 h-4 mr-2" />
            刷新
          </Button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        <Card className="bg-slate-900/50 border-slate-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">总日志数</p>
                <p className="text-2xl font-bold text-white">{stats.total}</p>
              </div>
              <FileText className="w-8 h-8 text-cyan-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900/50 border-slate-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">信息</p>
                <p className="text-2xl font-bold text-blue-400">{stats.info}</p>
              </div>
              <Info className="w-8 h-8 text-blue-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900/50 border-slate-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">警告</p>
                <p className="text-2xl font-bold text-amber-400">{stats.warning}</p>
              </div>
              <AlertTriangle className="w-8 h-8 text-amber-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900/50 border-slate-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">错误</p>
                <p className="text-2xl font-bold text-red-400">{stats.error}</p>
              </div>
              <AlertCircle className="w-8 h-8 text-red-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900/50 border-slate-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">调试</p>
                <p className="text-2xl font-bold text-slate-400">{stats.debug}</p>
              </div>
              <Bug className="w-8 h-8 text-slate-400" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Logs Table */}
      <Card className="bg-slate-900/50 border-slate-800">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-white flex items-center gap-2">
              <FileText className="w-5 h-5 text-cyan-400" />
              日志列表
            </CardTitle>
            <div className="flex items-center gap-3">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                <Input 
                  placeholder="搜索日志..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10 bg-slate-800 border-slate-700 text-white placeholder:text-slate-500 w-64"
                />
              </div>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="bg-slate-800 mb-4">
              <TabsTrigger value="all" className="data-[state=active]:bg-slate-700">全部</TabsTrigger>
              <TabsTrigger value="info" className="data-[state=active]:bg-slate-700">信息</TabsTrigger>
              <TabsTrigger value="warning" className="data-[state=active]:bg-slate-700">警告</TabsTrigger>
              <TabsTrigger value="error" className="data-[state=active]:bg-slate-700">错误</TabsTrigger>
              <TabsTrigger value="debug" className="data-[state=active]:bg-slate-700">调试</TabsTrigger>
            </TabsList>

            <div className="max-h-[500px] overflow-auto">
              <Table>
                <TableHeader className="bg-slate-800/50 sticky top-0">
                  <TableRow>
                    <TableHead className="text-slate-400 w-24">级别</TableHead>
                    <TableHead className="text-slate-400">时间</TableHead>
                    <TableHead className="text-slate-400">来源</TableHead>
                    <TableHead className="text-slate-400">消息</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredLogs.map((log) => (
                    <TableRow key={log.id} className="hover:bg-slate-800/50">
                      <TableCell>
                        <LevelBadge level={log.level} />
                      </TableCell>
                      <TableCell className="text-slate-400 text-sm whitespace-nowrap">
                        <div className="flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          {new Date(log.timestamp).toLocaleString()}
                        </div>
                      </TableCell>
                      <TableCell>
                        <Badge variant="outline" className="bg-slate-800 text-slate-300 border-slate-700">
                          {log.source}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-white">{log.message}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}
