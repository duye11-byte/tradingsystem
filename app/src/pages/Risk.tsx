import { useState } from 'react';
import { 
  Shield, 
  AlertTriangle, 
  Target, 
  Percent, 
  DollarSign,
  TrendingDown,
  Activity,
  CheckCircle,
  Info
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

import { Slider } from '@/components/ui/slider';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { mockRiskSettings } from '@/lib/mockData';
import type { RiskSettings } from '@/types';

export function Risk() {
  const [settings, setSettings] = useState<RiskSettings>(mockRiskSettings);
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  };

  const updateSetting = <K extends keyof RiskSettings>(key: K, value: RiskSettings[K]) => {
    setSettings(prev => ({ ...prev, [key]: value }));
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">风险管理</h1>
          <p className="text-slate-400 mt-1">配置交易风险参数和风控规则</p>
        </div>
        <Button 
          onClick={handleSave}
          className={saved ? 'bg-emerald-500 hover:bg-emerald-600' : 'bg-blue-500 hover:bg-blue-600'}
        >
          {saved ? <CheckCircle className="w-4 h-4 mr-2" /> : null}
          {saved ? '已保存' : '保存设置'}
        </Button>
      </div>

      {/* Risk Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-slate-900/50 border-slate-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">最大回撤</p>
                <p className="text-2xl font-bold text-white">{settings.maxDrawdown}%</p>
              </div>
              <TrendingDown className="w-8 h-8 text-red-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900/50 border-slate-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">单笔风险</p>
                <p className="text-2xl font-bold text-white">{settings.riskPerTrade}%</p>
              </div>
              <Percent className="w-8 h-8 text-amber-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900/50 border-slate-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">最大杠杆</p>
                <p className="text-2xl font-bold text-white">{settings.maxLeverage}x</p>
              </div>
              <Activity className="w-8 h-8 text-cyan-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900/50 border-slate-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-400">最大仓位</p>
                <p className="text-2xl font-bold text-white">{settings.maxPositionSize * 100}%</p>
              </div>
              <Target className="w-8 h-8 text-purple-400" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Settings Tabs */}
      <Tabs defaultValue="general" className="space-y-4">
        <TabsList className="bg-slate-800">
          <TabsTrigger value="general" className="data-[state=active]:bg-slate-700">通用设置</TabsTrigger>
          <TabsTrigger value="position" className="data-[state=active]:bg-slate-700">仓位管理</TabsTrigger>
          <TabsTrigger value="alerts" className="data-[state=active]:bg-slate-700">风险警报</TabsTrigger>
        </TabsList>

        {/* General Settings */}
        <TabsContent value="general" className="space-y-4">
          <Card className="bg-slate-900/50 border-slate-800">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Shield className="w-5 h-5 text-emerald-400" />
                基础风险设置
              </CardTitle>
              <CardDescription className="text-slate-400">
                配置系统的基本风险控制参数
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Max Drawdown */}
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label className="text-white">最大回撤限制</Label>
                  <span className="text-emerald-400">{settings.maxDrawdown}%</span>
                </div>
                <Slider 
                  value={[settings.maxDrawdown]} 
                  onValueChange={([v]) => updateSetting('maxDrawdown', v)}
                  min={5} 
                  max={50} 
                  step={1}
                  className="w-full"
                />
                <p className="text-xs text-slate-500">当账户回撤超过此值时，系统将停止开新仓</p>
              </div>

              {/* Risk Per Trade */}
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label className="text-white">单笔交易风险</Label>
                  <span className="text-emerald-400">{settings.riskPerTrade}%</span>
                </div>
                <Slider 
                  value={[settings.riskPerTrade]} 
                  onValueChange={([v]) => updateSetting('riskPerTrade', v)}
                  min={0.5} 
                  max={10} 
                  step={0.5}
                  className="w-full"
                />
                <p className="text-xs text-slate-500">每笔交易最多承担账户资金的百分比</p>
              </div>

              {/* Max Daily Loss */}
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label className="text-white">日最大亏损</Label>
                  <span className="text-emerald-400">${settings.maxDailyLoss}</span>
                </div>
                <Input 
                  type="number"
                  value={settings.maxDailyLoss}
                  onChange={(e) => updateSetting('maxDailyLoss', Number(e.target.value))}
                  className="bg-slate-800 border-slate-700 text-white"
                />
                <p className="text-xs text-slate-500">单日亏损达到此值时，系统停止交易</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Position Settings */}
        <TabsContent value="position" className="space-y-4">
          <Card className="bg-slate-900/50 border-slate-800">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Target className="w-5 h-5 text-cyan-400" />
                仓位管理设置
              </CardTitle>
              <CardDescription className="text-slate-400">
                配置仓位大小和杠杆限制
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Max Position Size */}
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label className="text-white">最大仓位比例</Label>
                  <span className="text-cyan-400">{settings.maxPositionSize * 100}%</span>
                </div>
                <Slider 
                  value={[settings.maxPositionSize * 100]} 
                  onValueChange={([v]) => updateSetting('maxPositionSize', v / 100)}
                  min={5} 
                  max={50} 
                  step={5}
                  className="w-full"
                />
                <p className="text-xs text-slate-500">单个仓位最大占用账户资金比例</p>
              </div>

              {/* Max Leverage */}
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label className="text-white">最大杠杆倍数</Label>
                  <span className="text-cyan-400">{settings.maxLeverage}x</span>
                </div>
                <Slider 
                  value={[settings.maxLeverage]} 
                  onValueChange={([v]) => updateSetting('maxLeverage', v)}
                  min={1} 
                  max={125} 
                  step={1}
                  className="w-full"
                />
                <p className="text-xs text-slate-500">系统允许的最大杠杆倍数</p>
              </div>

              {/* Stop Loss */}
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label className="text-white">默认止损比例</Label>
                  <span className="text-red-400">{settings.stopLossPercent}%</span>
                </div>
                <Slider 
                  value={[settings.stopLossPercent]} 
                  onValueChange={([v]) => updateSetting('stopLossPercent', v)}
                  min={1} 
                  max={10} 
                  step={0.5}
                  className="w-full"
                />
                <p className="text-xs text-slate-500">相对于入场价的止损距离</p>
              </div>

              {/* Take Profit */}
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label className="text-white">默认止盈比例</Label>
                  <span className="text-emerald-400">{settings.takeProfitPercent}%</span>
                </div>
                <Slider 
                  value={[settings.takeProfitPercent]} 
                  onValueChange={([v]) => updateSetting('takeProfitPercent', v)}
                  min={1} 
                  max={20} 
                  step={0.5}
                  className="w-full"
                />
                <p className="text-xs text-slate-500">相对于入场价的止盈距离</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Alerts Settings */}
        <TabsContent value="alerts" className="space-y-4">
          <Card className="bg-slate-900/50 border-slate-800">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-amber-400" />
                风险警报设置
              </CardTitle>
              <CardDescription className="text-slate-400">
                配置风险警报触发条件
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Alert className="bg-amber-500/10 border-amber-500/20">
                <AlertTriangle className="w-4 h-4 text-amber-400" />
                <AlertDescription className="text-amber-400">
                  当以下风险条件触发时，系统将发送警报通知
                </AlertDescription>
              </Alert>

              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 bg-slate-800/50 rounded-lg">
                  <div className="flex items-center gap-3">
                    <TrendingDown className="w-5 h-5 text-red-400" />
                    <div>
                      <p className="text-white font-medium">仓位亏损警报</p>
                      <p className="text-sm text-slate-400">当单个仓位亏损超过 5% 时触发</p>
                    </div>
                  </div>
                  <Switch defaultChecked />
                </div>

                <div className="flex items-center justify-between p-4 bg-slate-800/50 rounded-lg">
                  <div className="flex items-center gap-3">
                    <Activity className="w-5 h-5 text-cyan-400" />
                    <div>
                      <p className="text-white font-medium">强平风险警报</p>
                      <p className="text-sm text-slate-400">当仓位接近强平价 10% 时触发</p>
                    </div>
                  </div>
                  <Switch defaultChecked />
                </div>

                <div className="flex items-center justify-between p-4 bg-slate-800/50 rounded-lg">
                  <div className="flex items-center gap-3">
                    <DollarSign className="w-5 h-5 text-emerald-400" />
                    <div>
                      <p className="text-white font-medium">日亏损限额警报</p>
                      <p className="text-sm text-slate-400">当日亏损达到设定限额的 80% 时触发</p>
                    </div>
                  </div>
                  <Switch defaultChecked />
                </div>

                <div className="flex items-center justify-between p-4 bg-slate-800/50 rounded-lg">
                  <div className="flex items-center gap-3">
                    <Shield className="w-5 h-5 text-purple-400" />
                    <div>
                      <p className="text-white font-medium">数据源异常警报</p>
                      <p className="text-sm text-slate-400">当数据源连接异常时触发</p>
                    </div>
                  </div>
                  <Switch defaultChecked />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Risk Summary */}
      <Card className="bg-slate-900/50 border-slate-800">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Info className="w-5 h-5 text-blue-400" />
            风险概览
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 bg-slate-800/50 rounded-lg">
              <p className="text-sm text-slate-400 mb-1">风险收益比</p>
              <p className="text-xl font-bold text-white">
                1:{(settings.takeProfitPercent / settings.stopLossPercent).toFixed(1)}
              </p>
            </div>
            <div className="p-4 bg-slate-800/50 rounded-lg">
              <p className="text-sm text-slate-400 mb-1">预期胜率需求</p>
              <p className="text-xl font-bold text-white">
                {((settings.stopLossPercent / (settings.stopLossPercent + settings.takeProfitPercent)) * 100).toFixed(1)}%
              </p>
            </div>
            <div className="p-4 bg-slate-800/50 rounded-lg">
              <p className="text-sm text-slate-400 mb-1">最大连续亏损次数</p>
              <p className="text-xl font-bold text-white">
                {Math.floor(settings.maxDrawdown / settings.riskPerTrade)} 次
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
