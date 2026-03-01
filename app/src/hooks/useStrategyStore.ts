import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { 
  Strategy, 
  BacktestResult, 
  AIStrategyAnalysis, 
  StrategyIntegration,
  AIAgentState,
  BacktestConfig,
  UserSettings
} from '@/types';

interface StrategyState {
  // 策略列表
  strategies: Strategy[];
  addStrategy: (strategy: Omit<Strategy, 'id' | 'createdAt' | 'updatedAt'>) => void;
  updateStrategy: (id: string, updates: Partial<Strategy>) => void;
  deleteStrategy: (id: string) => void;
  importStrategy: (strategyData: string) => void;
  exportStrategy: (id: string) => string;
  
  // 回测结果
  backtestResults: BacktestResult[];
  addBacktestResult: (result: BacktestResult) => void;
  getBacktestResultsForStrategy: (strategyId: string) => BacktestResult[];
  
  // AI分析
  aiAnalyses: AIStrategyAnalysis[];
  addAIAnalysis: (analysis: AIStrategyAnalysis) => void;
  getAnalysisForStrategy: (strategyId: string) => AIStrategyAnalysis | undefined;
  
  // 策略集成
  integrations: StrategyIntegration[];
  addIntegration: (integration: StrategyIntegration) => void;
  getIntegrationsForStrategy: (strategyId: string) => StrategyIntegration[];
  
  // AI代理状态
  aiAgentState: AIAgentState;
  setAIAgentState: (state: Partial<AIAgentState>) => void;
  
  // 用户设置
  settings: UserSettings;
  updateSettings: (settings: Partial<UserSettings>) => void;
  
  // 当前选中
  selectedStrategyId: string | null;
  setSelectedStrategy: (id: string | null) => void;
  
  // 默认回测配置
  defaultBacktestConfig: BacktestConfig;
  updateDefaultBacktestConfig: (config: Partial<BacktestConfig>) => void;
}

const defaultSettings: UserSettings = {
  apiKeys: {},
  notifications: {
    email: true,
    push: true,
    tradeAlerts: true,
    backtestComplete: true,
    aiRecommendations: true,
  },
  riskDefaults: {
    maxPositionSize: 10,
    stopLossPercent: 2,
    takeProfitPercent: 4,
    maxDailyLoss: 5,
    maxDrawdownPercent: 15,
    useTrailingStop: false,
    trailingStopPercent: 1,
  },
  aiPreferences: {
    autoBacktest: true,
    autoIntegrate: false,
    minScoreForIntegration: 75,
    maxIntegrationPerDay: 3,
    preferredIntegrationType: 'balanced',
  },
};

const defaultBacktestConfig: BacktestConfig = {
  startDate: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000),
  endDate: new Date(),
  initialCapital: 10000,
  tradingFee: 0.1,
  slippage: 0.05,
  symbol: 'BTC/USDT',
  timeframe: '1h',
};

export const useStrategyStore = create<StrategyState>()(
  persist(
    (set, get) => ({
      // 初始状态
      strategies: [],
      backtestResults: [],
      aiAnalyses: [],
      integrations: [],
      aiAgentState: {
        isRunning: false,
        progress: 0,
        pendingStrategies: [],
        analyzedStrategies: [],
        integratedStrategies: [],
      },
      settings: defaultSettings,
      selectedStrategyId: null,
      defaultBacktestConfig,

      // 策略操作
      addStrategy: (strategy) => {
        const newStrategy: Strategy = {
          ...strategy,
          id: `strategy_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          createdAt: new Date(),
          updatedAt: new Date(),
        };
        set((state) => ({
          strategies: [...state.strategies, newStrategy],
        }));
        
        // 如果开启了自动回测，添加到AI代理队列
        if (get().settings.aiPreferences.autoBacktest) {
          set((state) => ({
            aiAgentState: {
              ...state.aiAgentState,
              pendingStrategies: [...state.aiAgentState.pendingStrategies, newStrategy.id],
            },
          }));
        }
      },

      updateStrategy: (id, updates) => {
        set((state) => ({
          strategies: state.strategies.map((s) =>
            s.id === id ? { ...s, ...updates, updatedAt: new Date() } : s
          ),
        }));
      },

      deleteStrategy: (id) => {
        set((state) => ({
          strategies: state.strategies.filter((s) => s.id !== id),
          backtestResults: state.backtestResults.filter((r) => r.strategyId !== id),
          aiAnalyses: state.aiAnalyses.filter((a) => a.strategyId !== id),
        }));
      },

      importStrategy: (strategyData) => {
        try {
          const parsed = JSON.parse(strategyData);
          const importedStrategy: Strategy = {
            ...parsed,
            id: `strategy_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            createdAt: new Date(),
            updatedAt: new Date(),
            source: 'imported',
            name: `${parsed.name} (Imported)`,
          };
          set((state) => ({
            strategies: [...state.strategies, importedStrategy],
          }));
          
          if (get().settings.aiPreferences.autoBacktest) {
            set((state) => ({
              aiAgentState: {
                ...state.aiAgentState,
                pendingStrategies: [...state.aiAgentState.pendingStrategies, importedStrategy.id],
              },
            }));
          }
        } catch (error) {
          console.error('Failed to import strategy:', error);
        }
      },

      exportStrategy: (id) => {
        const strategy = get().strategies.find((s) => s.id === id);
        if (!strategy) return '';
        return JSON.stringify(strategy, null, 2);
      },

      // 回测结果操作
      addBacktestResult: (result) => {
        set((state) => ({
          backtestResults: [...state.backtestResults, result],
        }));
      },

      getBacktestResultsForStrategy: (strategyId) => {
        return get().backtestResults.filter((r) => r.strategyId === strategyId);
      },

      // AI分析操作
      addAIAnalysis: (analysis) => {
        set((state) => ({
          aiAnalyses: [...state.aiAnalyses.filter(a => a.strategyId !== analysis.strategyId), analysis],
        }));
      },

      getAnalysisForStrategy: (strategyId) => {
        return get().aiAnalyses.find((a) => a.strategyId === strategyId);
      },

      // 集成操作
      addIntegration: (integration) => {
        set((state) => ({
          integrations: [...state.integrations, integration],
        }));
      },

      getIntegrationsForStrategy: (strategyId) => {
        return get().integrations.filter(
          (i) => i.sourceStrategyId === strategyId || i.targetStrategyId === strategyId
        );
      },

      // AI代理状态操作
      setAIAgentState: (state) => {
        set((prev) => ({
          aiAgentState: { ...prev.aiAgentState, ...state },
        }));
      },

      // 设置操作
      updateSettings: (settings) => {
        set((state) => ({
          settings: { ...state.settings, ...settings },
        }));
      },

      // 选中策略
      setSelectedStrategy: (id) => {
        set({ selectedStrategyId: id });
      },

      // 默认回测配置
      updateDefaultBacktestConfig: (config) => {
        set((state) => ({
          defaultBacktestConfig: { ...state.defaultBacktestConfig, ...config },
        }));
      },
    }),
    {
      name: 'crypto-ai-trading-storage',
      partialize: (state) => ({
        strategies: state.strategies,
        backtestResults: state.backtestResults,
        aiAnalyses: state.aiAnalyses,
        integrations: state.integrations,
        settings: state.settings,
        defaultBacktestConfig: state.defaultBacktestConfig,
      }),
    }
  )
);
