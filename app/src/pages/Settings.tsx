import { useState } from 'react';
import { 
  Settings as SettingsIcon, 
  Key, 
  Bell, 
  Moon, 
  Sun, 
  Globe,
  Shield,
  Save,
  CheckCircle,
  Eye,
  EyeOff
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';

export function Settings() {
  const [saved, setSaved] = useState(false);
  const [showApiSecret, setShowApiSecret] = useState(false);
  const [settings, setSettings] = useState({
    apiKey: '',
    apiSecret: '',
    testnet: true,
    defaultLeverage: 5,
    notifications: true,
    emailAlerts: true,
    theme: 'dark' as 'light' | 'dark',
    language: 'zh',
    autoRefresh: true,
    refreshInterval: 5,
  });

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  };

  const updateSetting = <K extends keyof typeof settings>(key: K, value: typeof settings[K]) => {
    setSettings(prev => ({ ...prev, [key]: value }));
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">系统设置</h1>
          <p className="text-slate-400 mt-1">配置系统参数和偏好设置</p>
        </div>
        <Button 
          onClick={handleSave}
          className={saved ? 'bg-emerald-500 hover:bg-emerald-600' : 'bg-blue-500 hover:bg-blue-600'}
        >
          {saved ? <CheckCircle className="w-4 h-4 mr-2" /> : <Save className="w-4 h-4 mr-2" />}
          {saved ? '已保存' : '保存设置'}
        </Button>
      </div>

      {/* Settings Tabs */}
      <Tabs defaultValue="api" className="space-y-4">
        <TabsList className="bg-slate-800">
          <TabsTrigger value="api" className="data-[state=active]:bg-slate-700 flex items-center gap-2">
            <Key className="w-4 h-4" />
            API 设置
          </TabsTrigger>
          <TabsTrigger value="trading" className="data-[state=active]:bg-slate-700 flex items-center gap-2">
            <Globe className="w-4 h-4" />
            交易设置
          </TabsTrigger>
          <TabsTrigger value="notifications" className="data-[state=active]:bg-slate-700 flex items-center gap-2">
            <Bell className="w-4 h-4" />
            通知设置
          </TabsTrigger>
          <TabsTrigger value="appearance" className="data-[state=active]:bg-slate-700 flex items-center gap-2">
            <SettingsIcon className="w-4 h-4" />
            外观设置
          </TabsTrigger>
        </TabsList>

        {/* API Settings */}
        <TabsContent value="api" className="space-y-4">
          <Card className="bg-slate-900/50 border-slate-800">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Key className="w-5 h-5 text-cyan-400" />
                Binance API 配置
              </CardTitle>
              <CardDescription className="text-slate-400">
                配置您的 Binance API 密钥以进行交易
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* API Key */}
              <div className="space-y-2">
                <Label className="text-white">API Key</Label>
                <Input 
                  type="text"
                  placeholder="输入您的 API Key"
                  value={settings.apiKey}
                  onChange={(e) => updateSetting('apiKey', e.target.value)}
                  className="bg-slate-800 border-slate-700 text-white"
                />
              </div>

              {/* API Secret */}
              <div className="space-y-2">
                <Label className="text-white">API Secret</Label>
                <div className="relative">
                  <Input 
                    type={showApiSecret ? 'text' : 'password'}
                    placeholder="输入您的 API Secret"
                    value={settings.apiSecret}
                    onChange={(e) => updateSetting('apiSecret', e.target.value)}
                    className="bg-slate-800 border-slate-700 text-white pr-10"
                  />
                  <button
                    onClick={() => setShowApiSecret(!showApiSecret)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-white"
                  >
                    {showApiSecret ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>

              <Separator className="bg-slate-800" />

              {/* Testnet */}
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-white font-medium">使用测试网</p>
                  <p className="text-sm text-slate-400">启用后将在 Binance 测试网进行交易</p>
                </div>
                <Switch 
                  checked={settings.testnet}
                  onCheckedChange={(v) => updateSetting('testnet', v)}
                />
              </div>

              {/* Test Connection */}
              <Button variant="outline" className="w-full border-slate-700 text-slate-300 hover:bg-slate-800">
                <Shield className="w-4 h-4 mr-2" />
                测试连接
              </Button>
            </CardContent>
          </Card>

          <Card className="bg-slate-900/50 border-slate-800">
            <CardHeader>
              <CardTitle className="text-white">API 权限</CardTitle>
              <CardDescription className="text-slate-400">
                确保您的 API Key 具有以下权限
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
                  <div className="flex items-center gap-3">
                    <CheckCircle className="w-5 h-5 text-emerald-400" />
                    <span className="text-white">读取账户信息</span>
                  </div>
                  <Badge className="bg-emerald-500/10 text-emerald-400">必需</Badge>
                </div>
                <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
                  <div className="flex items-center gap-3">
                    <CheckCircle className="w-5 h-5 text-emerald-400" />
                    <span className="text-white">现货交易</span>
                  </div>
                  <Badge className="bg-emerald-500/10 text-emerald-400">必需</Badge>
                </div>
                <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
                  <div className="flex items-center gap-3">
                    <CheckCircle className="w-5 h-5 text-emerald-400" />
                    <span className="text-white">合约交易</span>
                  </div>
                  <Badge className="bg-blue-500/10 text-blue-400">可选</Badge>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Trading Settings */}
        <TabsContent value="trading" className="space-y-4">
          <Card className="bg-slate-900/50 border-slate-800">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Globe className="w-5 h-5 text-emerald-400" />
                默认交易设置
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Default Leverage */}
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label className="text-white">默认杠杆倍数</Label>
                  <span className="text-emerald-400">{settings.defaultLeverage}x</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="125"
                  value={settings.defaultLeverage}
                  onChange={(e) => updateSetting('defaultLeverage', Number(e.target.value))}
                  className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                />
              </div>

              <Separator className="bg-slate-800" />

              {/* Auto Refresh */}
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-white font-medium">自动刷新数据</p>
                  <p className="text-sm text-slate-400">自动更新市场行情和仓位数据</p>
                </div>
                <Switch 
                  checked={settings.autoRefresh}
                  onCheckedChange={(v) => updateSetting('autoRefresh', v)}
                />
              </div>

              {/* Refresh Interval */}
              {settings.autoRefresh && (
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label className="text-white">刷新间隔</Label>
                    <span className="text-emerald-400">{settings.refreshInterval} 秒</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="60"
                    value={settings.refreshInterval}
                    onChange={(e) => updateSetting('refreshInterval', Number(e.target.value))}
                    className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                  />
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Notifications */}
        <TabsContent value="notifications" className="space-y-4">
          <Card className="bg-slate-900/50 border-slate-800">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Bell className="w-5 h-5 text-amber-400" />
                通知设置
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between p-4 bg-slate-800/50 rounded-lg">
                <div>
                  <p className="text-white font-medium">启用通知</p>
                  <p className="text-sm text-slate-400">接收交易信号和系统通知</p>
                </div>
                <Switch 
                  checked={settings.notifications}
                  onCheckedChange={(v) => updateSetting('notifications', v)}
                />
              </div>

              <div className="flex items-center justify-between p-4 bg-slate-800/50 rounded-lg">
                <div>
                  <p className="text-white font-medium">邮件警报</p>
                  <p className="text-sm text-slate-400">通过邮件接收重要警报</p>
                </div>
                <Switch 
                  checked={settings.emailAlerts}
                  onCheckedChange={(v) => updateSetting('emailAlerts', v)}
                />
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Appearance */}
        <TabsContent value="appearance" className="space-y-4">
          <Card className="bg-slate-900/50 border-slate-800">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <SettingsIcon className="w-5 h-5 text-purple-400" />
                外观设置
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Theme */}
              <div className="flex items-center justify-between p-4 bg-slate-800/50 rounded-lg">
                <div className="flex items-center gap-3">
                  {settings.theme === 'dark' ? <Moon className="w-5 h-5 text-slate-400" /> : <Sun className="w-5 h-5 text-amber-400" />}
                  <div>
                    <p className="text-white font-medium">深色模式</p>
                    <p className="text-sm text-slate-400">切换深色/浅色主题</p>
                  </div>
                </div>
                <Switch 
                  checked={settings.theme === 'dark'}
                  onCheckedChange={(v) => updateSetting('theme', v ? 'dark' : 'light')}
                />
              </div>

              {/* Language */}
              <div className="flex items-center justify-between p-4 bg-slate-800/50 rounded-lg">
                <div>
                  <p className="text-white font-medium">语言</p>
                  <p className="text-sm text-slate-400">选择界面语言</p>
                </div>
                <select 
                  value={settings.language}
                  onChange={(e) => updateSetting('language', e.target.value)}
                  className="bg-slate-800 border border-slate-700 text-white rounded-lg px-3 py-2"
                >
                  <option value="zh">简体中文</option>
                  <option value="en">English</option>
                </select>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
