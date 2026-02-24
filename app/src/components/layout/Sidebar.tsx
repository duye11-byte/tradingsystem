import { NavLink } from 'react-router-dom';
import { 
  LayoutDashboard, 
  LineChart, 
  Signal, 
  ListOrdered, 
  Wallet, 
  Shield, 
  Database, 
  FileText, 
  Settings,
  TrendingUp,
  Brain
} from 'lucide-react';
import { cn } from '@/lib/utils';

const navItems = [
  { path: '/', label: '仪表盘', icon: LayoutDashboard },
  { path: '/market', label: '市场行情', icon: LineChart },
  { path: '/signals', label: '交易信号', icon: Signal },
  { path: '/orders', label: '订单管理', icon: ListOrdered },
  { path: '/positions', label: '仓位管理', icon: Wallet },
  { path: '/risk', label: '风险管理', icon: Shield },
  { path: '/strategy', label: '策略管理', icon: Brain },
  { path: '/datasources', label: '数据源', icon: Database },
  { path: '/logs', label: '系统日志', icon: FileText },
  { path: '/settings', label: '设置', icon: Settings },
];

export function Sidebar() {
  return (
    <aside className="w-64 bg-slate-900 border-r border-slate-800 min-h-screen flex flex-col">
      {/* Logo */}
      <div className="p-6 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-emerald-500 to-cyan-500 rounded-xl flex items-center justify-center">
            <TrendingUp className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white">OpenClaw</h1>
            <p className="text-xs text-slate-400">Trading System</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-1">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }: { isActive: boolean }) =>
                cn(
                  'flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200',
                  isActive
                    ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                    : 'text-slate-400 hover:text-white hover:bg-slate-800'
                )
              }
            >
              <Icon className="w-5 h-5" />
              {item.label}
            </NavLink>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-slate-800">
        <div className="flex items-center gap-3 px-4 py-2">
          <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
          <span className="text-xs text-slate-400">系统运行正常</span>
        </div>
        <p className="text-xs text-slate-500 px-4 mt-2">v1.0.0</p>
      </div>
    </aside>
  );
}
