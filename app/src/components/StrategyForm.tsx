import { useState } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Plus, Trash2, TrendingUp, TrendingDown, Shield, Activity } from 'lucide-react';
import type { Strategy, Condition, IndicatorConfig } from '@/types';

interface StrategyFormProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSubmit: (strategy: Omit<Strategy, 'id' | 'createdAt' | 'updatedAt'>) => void;
  initialData?: Partial<Strategy>;
}

const STRATEGY_TYPES = [
  { value: 'trend_following', label: '趋势跟踪', description: '跟随市场趋势进行交易' },
  { value: 'mean_reversion', label: '均值回归', description: '价格偏离均值时反向交易' },
  { value: 'breakout', label: '突破策略', description: '价格突破关键位时交易' },
  { value: 'scalping', label: '剥头皮', description: '频繁小额盈利交易' },
  { value: 'custom', label: '自定义', description: '自定义策略逻辑' },
];

const INDICATORS = [
  { value: 'SMA', label: '简单移动平均线 (SMA)', params: { period: 20 } },
  { value: 'EMA', label: '指数移动平均线 (EMA)', params: { period: 20 } },
  { value: 'RSI', label: '相对强弱指标 (RSI)', params: { period: 14 } },
  { value: 'MACD', label: 'MACD', params: { fastPeriod: 12, slowPeriod: 26, signalPeriod: 9 } },
  { value: 'BB', label: '布林带 (Bollinger Bands)', params: { period: 20, stdDev: 2 } },
  { value: 'ATR', label: '平均真实波幅 (ATR)', params: { period: 14 } },
];

const OPERATORS = [
  { value: '>', label: '大于' },
  { value: '<', label: '小于' },
  { value: '==', label: '等于' },
  { value: '>=', label: '大于等于' },
  { value: '<=', label: '小于等于' },
  { value: 'crosses_above', label: '上穿' },
  { value: 'crosses_below', label: '下穿' },
];

const TIMEFRAMES = [
  { value: '1m', label: '1分钟' },
  { value: '5m', label: '5分钟' },
  { value: '15m', label: '15分钟' },
  { value: '1h', label: '1小时' },
  { value: '4h', label: '4小时' },
  { value: '1d', label: '1天' },
];

export function StrategyForm({ open, onOpenChange, onSubmit, initialData }: StrategyFormProps) {
  const [formData, setFormData] = useState({
    name: initialData?.name || '',
    description: initialData?.description || '',
    type: initialData?.type || 'custom',
    isActive: initialData?.isActive ?? true,
    source: 'manual' as const,
    parameters: {
      entryConditions: initialData?.parameters?.entryConditions || [],
      exitConditions: initialData?.parameters?.exitConditions || [],
      riskManagement: {
        maxPositionSize: initialData?.parameters?.riskManagement?.maxPositionSize || 10,
        stopLossPercent: initialData?.parameters?.riskManagement?.stopLossPercent || 2,
        takeProfitPercent: initialData?.parameters?.riskManagement?.takeProfitPercent || 4,
        maxDailyLoss: initialData?.parameters?.riskManagement?.maxDailyLoss || 5,
        maxDrawdownPercent: initialData?.parameters?.riskManagement?.maxDrawdownPercent || 15,
        useTrailingStop: initialData?.parameters?.riskManagement?.useTrailingStop || false,
        trailingStopPercent: initialData?.parameters?.riskManagement?.trailingStopPercent || 1,
      },
      indicators: initialData?.parameters?.indicators || [],
    },
    rules: {
      longEntry: initialData?.rules?.longEntry || [],
      shortEntry: initialData?.rules?.shortEntry || [],
      exitLong: initialData?.rules?.exitLong || [],
      exitShort: initialData?.rules?.exitShort || [],
    },
  });

  const [newCondition, setNewCondition] = useState<Partial<Condition>>({
    indicator: 'SMA',
    operator: '>',
    value: '',
    timeframe: '1h',
  });

  const [newIndicator, setNewIndicator] = useState<Partial<IndicatorConfig>>({
    name: 'SMA',
    parameters: { period: 20 },
    timeframe: '1h',
  });

  const addCondition = (type: 'entry' | 'exit') => {
    if (!newCondition.indicator || !newCondition.value) return;

    const condition: Condition = {
      id: `cond_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      indicator: newCondition.indicator,
      operator: newCondition.operator as any,
      value: parseFloat(newCondition.value as string) || newCondition.value,
      timeframe: newCondition.timeframe || '1h',
    };

    setFormData(prev => ({
      ...prev,
      parameters: {
        ...prev.parameters,
        [type === 'entry' ? 'entryConditions' : 'exitConditions']: [
          ...prev.parameters[type === 'entry' ? 'entryConditions' : 'exitConditions'],
          condition,
        ],
      },
      rules: {
        ...prev.rules,
        [type === 'entry' ? 'longEntry' : 'exitLong']: [
          ...prev.rules[type === 'entry' ? 'longEntry' : 'exitLong'],
          condition.id,
        ],
      },
    }));

    setNewCondition({
      indicator: 'SMA',
      operator: '>',
      value: '',
      timeframe: '1h',
    });
  };

  const removeCondition = (type: 'entry' | 'exit', index: number) => {
    const key = type === 'entry' ? 'entryConditions' : 'exitConditions';
    const condition = formData.parameters[key][index];
    
    setFormData(prev => ({
      ...prev,
      parameters: {
        ...prev.parameters,
        [key]: prev.parameters[key].filter((_, i) => i !== index),
      },
      rules: {
        ...prev.rules,
        longEntry: prev.rules.longEntry.filter(id => id !== condition.id),
        exitLong: prev.rules.exitLong.filter(id => id !== condition.id),
      },
    }));
  };

  const addIndicator = () => {
    if (!newIndicator.name) return;

    const indicatorConfig = INDICATORS.find(i => i.value === newIndicator.name);
    if (!indicatorConfig) return;

    const indicator: IndicatorConfig = {
      name: newIndicator.name,
      parameters: { ...indicatorConfig.params, ...(newIndicator.parameters || {}) } as unknown as Record<string, number>,
      timeframe: newIndicator.timeframe || '1h',
    };

    setFormData(prev => ({
      ...prev,
      parameters: {
        ...prev.parameters,
        indicators: [...prev.parameters.indicators, indicator],
      },
    }));
  };

  const removeIndicator = (index: number) => {
    setFormData(prev => ({
      ...prev,
      parameters: {
        ...prev.parameters,
        indicators: prev.parameters.indicators.filter((_, i) => i !== index),
      },
    }));
  };

  const handleSubmit = () => {
    onSubmit(formData);
    onOpenChange(false);
  };

  const renderConditionForm = (type: 'entry' | 'exit') => (
    <div className="space-y-3 p-4 bg-[#1a1a28] rounded-lg">
      <div className="grid grid-cols-4 gap-2">
        <Select
          value={newCondition.indicator}
          onValueChange={(value) => setNewCondition(prev => ({ ...prev, indicator: value }))}
        >
          <SelectTrigger className="bg-[#12121C] border-[#2a2a3c]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent className="bg-[#12121C] border-[#2a2a3c]">
            {INDICATORS.map(ind => (
              <SelectItem key={ind.value} value={ind.value}>{ind.label}</SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Select
          value={newCondition.operator}
          onValueChange={(value) => setNewCondition(prev => ({ ...prev, operator: value as Condition['operator'] }))}
        >
          <SelectTrigger className="bg-[#12121C] border-[#2a2a3c]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent className="bg-[#12121C] border-[#2a2a3c]">
            {OPERATORS.map(op => (
              <SelectItem key={op.value} value={op.value}>{op.label}</SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Input
          placeholder="数值"
          value={newCondition.value}
          onChange={(e) => setNewCondition(prev => ({ ...prev, value: e.target.value }))}
          className="bg-[#12121C] border-[#2a2a3c]"
        />

        <Select
          value={newCondition.timeframe}
          onValueChange={(value) => setNewCondition(prev => ({ ...prev, timeframe: value }))}
        >
          <SelectTrigger className="bg-[#12121C] border-[#2a2a3c]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent className="bg-[#12121C] border-[#2a2a3c]">
            {TIMEFRAMES.map(tf => (
              <SelectItem key={tf.value} value={tf.value}>{tf.label}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <Button
        type="button"
        variant="outline"
        size="sm"
        onClick={() => addCondition(type)}
        className="w-full border-dashed border-[#2a2a3c] hover:border-cyan-500 hover:text-cyan-400"
      >
        <Plus className="w-4 h-4 mr-1" />
        添加{type === 'entry' ? '入场' : '出场'}条件
      </Button>
    </div>
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto bg-[#12121C] border-[#2a2a3c]">
        <DialogHeader>
          <DialogTitle className="text-xl font-bold text-white">
            {initialData ? '编辑策略' : '创建新策略'}
          </DialogTitle>
        </DialogHeader>

        <Tabs defaultValue="basic" className="mt-4">
          <TabsList className="grid w-full grid-cols-4 bg-[#1a1a28]">
            <TabsTrigger value="basic">基本信息</TabsTrigger>
            <TabsTrigger value="conditions">交易条件</TabsTrigger>
            <TabsTrigger value="indicators">技术指标</TabsTrigger>
            <TabsTrigger value="risk">风险管理</TabsTrigger>
          </TabsList>

          <TabsContent value="basic" className="space-y-4">
            <div className="space-y-2">
              <Label className="text-gray-300">策略名称</Label>
              <Input
                value={formData.name}
                onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
                placeholder="输入策略名称"
                className="bg-[#1a1a28] border-[#2a2a3c] text-white"
              />
            </div>

            <div className="space-y-2">
              <Label className="text-gray-300">策略类型</Label>
              <Select
                value={formData.type}
                onValueChange={(value) => setFormData(prev => ({ ...prev, type: value as any }))}
              >
                <SelectTrigger className="bg-[#1a1a28] border-[#2a2a3c]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-[#12121C] border-[#2a2a3c]">
                  {STRATEGY_TYPES.map(type => (
                    <SelectItem key={type.value} value={type.value}>
                      <div>
                        <div className="font-medium">{type.label}</div>
                        <div className="text-xs text-gray-500">{type.description}</div>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label className="text-gray-300">策略描述</Label>
              <Textarea
                value={formData.description}
                onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
                placeholder="描述策略的逻辑和特点"
                rows={4}
                className="bg-[#1a1a28] border-[#2a2a3c] text-white resize-none"
              />
            </div>

            <div className="flex items-center gap-2">
              <Switch
                checked={formData.isActive}
                onCheckedChange={(checked) => setFormData(prev => ({ ...prev, isActive: checked }))}
              />
              <Label className="text-gray-300">激活策略</Label>
            </div>
          </TabsContent>

          <TabsContent value="conditions" className="space-y-6">
            {/* 入场条件 */}
            <Card className="bg-[#1a1a28] border-[#2a2a3c]">
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2 text-white">
                  <TrendingUp className="w-4 h-4 text-green-400" />
                  入场条件
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {formData.parameters.entryConditions.map((condition, index) => (
                  <div key={condition.id} className="flex items-center gap-2 p-2 bg-[#12121C] rounded">
                    <Badge variant="outline" className="text-cyan-400 border-cyan-500/30">
                      {condition.indicator}
                    </Badge>
                    <span className="text-gray-400">{OPERATORS.find(o => o.value === condition.operator)?.label}</span>
                    <span className="text-white font-medium">{condition.value}</span>
                    <span className="text-gray-500 text-sm">({condition.timeframe})</span>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="ml-auto h-6 w-6 text-gray-400 hover:text-red-400"
                      onClick={() => removeCondition('entry', index)}
                    >
                      <Trash2 className="w-3 h-3" />
                    </Button>
                  </div>
                ))}
                {renderConditionForm('entry')}
              </CardContent>
            </Card>

            {/* 出场条件 */}
            <Card className="bg-[#1a1a28] border-[#2a2a3c]">
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2 text-white">
                  <TrendingDown className="w-4 h-4 text-red-400" />
                  出场条件
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {formData.parameters.exitConditions.map((condition, index) => (
                  <div key={condition.id} className="flex items-center gap-2 p-2 bg-[#12121C] rounded">
                    <Badge variant="outline" className="text-purple-400 border-purple-500/30">
                      {condition.indicator}
                    </Badge>
                    <span className="text-gray-400">{OPERATORS.find(o => o.value === condition.operator)?.label}</span>
                    <span className="text-white font-medium">{condition.value}</span>
                    <span className="text-gray-500 text-sm">({condition.timeframe})</span>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="ml-auto h-6 w-6 text-gray-400 hover:text-red-400"
                      onClick={() => removeCondition('exit', index)}
                    >
                      <Trash2 className="w-3 h-3" />
                    </Button>
                  </div>
                ))}
                {renderConditionForm('exit')}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="indicators" className="space-y-4">
            <Card className="bg-[#1a1a28] border-[#2a2a3c]">
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2 text-white">
                  <Activity className="w-4 h-4 text-cyan-400" />
                  技术指标
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {formData.parameters.indicators.map((indicator, index) => (
                  <div key={index} className="flex items-center gap-2 p-3 bg-[#12121C] rounded">
                    <Badge className="bg-cyan-500/20 text-cyan-400">
                      {INDICATORS.find(i => i.value === indicator.name)?.label || indicator.name}
                    </Badge>
                    <span className="text-gray-400 text-sm">
                      参数: {Object.entries(indicator.parameters).map(([k, v]) => `${k}=${v}`).join(', ')}
                    </span>
                    <span className="text-gray-500 text-sm">({indicator.timeframe})</span>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="ml-auto h-6 w-6 text-gray-400 hover:text-red-400"
                      onClick={() => removeIndicator(index)}
                    >
                      <Trash2 className="w-3 h-3" />
                    </Button>
                  </div>
                ))}

                <div className="p-4 bg-[#12121C] rounded-lg space-y-3">
                  <div className="grid grid-cols-2 gap-2">
                    <Select
                      value={newIndicator.name}
                      onValueChange={(value) => {
                        const ind = INDICATORS.find(i => i.value === value);
                        setNewIndicator(prev => ({ 
                          ...prev, 
                          name: value,
                          parameters: ind?.params || {}
                        }));
                      }}
                    >
                      <SelectTrigger className="bg-[#1a1a28] border-[#2a2a3c]">
                        <SelectValue placeholder="选择指标" />
                      </SelectTrigger>
                      <SelectContent className="bg-[#12121C] border-[#2a2a3c]">
                        {INDICATORS.map(ind => (
                          <SelectItem key={ind.value} value={ind.value}>{ind.label}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>

                    <Select
                      value={newIndicator.timeframe}
                      onValueChange={(value) => setNewIndicator(prev => ({ ...prev, timeframe: value }))}
                    >
                      <SelectTrigger className="bg-[#1a1a28] border-[#2a2a3c]">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-[#12121C] border-[#2a2a3c]">
                        {TIMEFRAMES.map(tf => (
                          <SelectItem key={tf.value} value={tf.value}>{tf.label}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={addIndicator}
                    className="w-full border-dashed border-[#2a2a3c] hover:border-cyan-500 hover:text-cyan-400"
                  >
                    <Plus className="w-4 h-4 mr-1" />
                    添加指标
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="risk" className="space-y-4">
            <Card className="bg-[#1a1a28] border-[#2a2a3c]">
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2 text-white">
                  <Shield className="w-4 h-4 text-purple-400" />
                  风险管理参数
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label className="text-gray-300">最大仓位规模 (%)</Label>
                    <span className="text-cyan-400 font-mono">
                      {formData.parameters.riskManagement.maxPositionSize}%
                    </span>
                  </div>
                  <Slider
                    value={[formData.parameters.riskManagement.maxPositionSize]}
                    onValueChange={([value]) => setFormData(prev => ({
                      ...prev,
                      parameters: {
                        ...prev.parameters,
                        riskManagement: { ...prev.parameters.riskManagement, maxPositionSize: value }
                      }
                    }))}
                    min={1}
                    max={100}
                    step={1}
                    className="w-full"
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label className="text-gray-300">止损比例 (%)</Label>
                    <span className="text-red-400 font-mono">
                      {formData.parameters.riskManagement.stopLossPercent}%
                    </span>
                  </div>
                  <Slider
                    value={[formData.parameters.riskManagement.stopLossPercent]}
                    onValueChange={([value]) => setFormData(prev => ({
                      ...prev,
                      parameters: {
                        ...prev.parameters,
                        riskManagement: { ...prev.parameters.riskManagement, stopLossPercent: value }
                      }
                    }))}
                    min={0.5}
                    max={10}
                    step={0.5}
                    className="w-full"
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label className="text-gray-300">止盈比例 (%)</Label>
                    <span className="text-green-400 font-mono">
                      {formData.parameters.riskManagement.takeProfitPercent}%
                    </span>
                  </div>
                  <Slider
                    value={[formData.parameters.riskManagement.takeProfitPercent]}
                    onValueChange={([value]) => setFormData(prev => ({
                      ...prev,
                      parameters: {
                        ...prev.parameters,
                        riskManagement: { ...prev.parameters.riskManagement, takeProfitPercent: value }
                      }
                    }))}
                    min={1}
                    max={20}
                    step={0.5}
                    className="w-full"
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label className="text-gray-300">每日最大亏损 (%)</Label>
                    <span className="text-orange-400 font-mono">
                      {formData.parameters.riskManagement.maxDailyLoss}%
                    </span>
                  </div>
                  <Slider
                    value={[formData.parameters.riskManagement.maxDailyLoss]}
                    onValueChange={([value]) => setFormData(prev => ({
                      ...prev,
                      parameters: {
                        ...prev.parameters,
                        riskManagement: { ...prev.parameters.riskManagement, maxDailyLoss: value }
                      }
                    }))}
                    min={1}
                    max={20}
                    step={1}
                    className="w-full"
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label className="text-gray-300">最大回撤限制 (%)</Label>
                    <span className="text-red-400 font-mono">
                      {formData.parameters.riskManagement.maxDrawdownPercent}%
                    </span>
                  </div>
                  <Slider
                    value={[formData.parameters.riskManagement.maxDrawdownPercent]}
                    onValueChange={([value]) => setFormData(prev => ({
                      ...prev,
                      parameters: {
                        ...prev.parameters,
                        riskManagement: { ...prev.parameters.riskManagement, maxDrawdownPercent: value }
                      }
                    }))}
                    min={5}
                    max={50}
                    step={5}
                    className="w-full"
                  />
                </div>

                <div className="flex items-center justify-between p-3 bg-[#12121C] rounded-lg">
                  <div>
                    <Label className="text-gray-300">启用追踪止损</Label>
                    <p className="text-xs text-gray-500">价格向有利方向移动时自动调整止损</p>
                  </div>
                  <Switch
                    checked={formData.parameters.riskManagement.useTrailingStop}
                    onCheckedChange={(checked) => setFormData(prev => ({
                      ...prev,
                      parameters: {
                        ...prev.parameters,
                        riskManagement: { ...prev.parameters.riskManagement, useTrailingStop: checked }
                      }
                    }))}
                  />
                </div>

                {formData.parameters.riskManagement.useTrailingStop && (
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <Label className="text-gray-300">追踪止损比例 (%)</Label>
                      <span className="text-purple-400 font-mono">
                        {formData.parameters.riskManagement.trailingStopPercent}%
                      </span>
                    </div>
                    <Slider
                      value={[formData.parameters.riskManagement.trailingStopPercent || 1]}
                      onValueChange={([value]) => setFormData(prev => ({
                        ...prev,
                        parameters: {
                          ...prev.parameters,
                          riskManagement: { ...prev.parameters.riskManagement, trailingStopPercent: value }
                        }
                      }))}
                      min={0.5}
                      max={5}
                      step={0.5}
                      className="w-full"
                    />
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        <div className="flex justify-end gap-3 mt-6">
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            className="border-[#2a2a3c] text-gray-300 hover:text-white"
          >
            取消
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={!formData.name}
            className="bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-600 hover:to-purple-600 text-white"
          >
            {initialData ? '保存修改' : '创建策略'}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
