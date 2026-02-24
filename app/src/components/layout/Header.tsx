import { Bell, User, Moon, Sun } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useState } from 'react';

export function Header() {
  const [isDark, setIsDark] = useState(true);
  const [notifications] = useState(3);

  return (
    <header className="h-16 bg-slate-900/50 backdrop-blur-xl border-b border-slate-800 flex items-center justify-between px-6 sticky top-0 z-50">
      {/* Left side - could add breadcrumbs here */}
      <div className="flex items-center gap-4">
        <h2 className="text-lg font-semibold text-white">交易控制台</h2>
        <Badge variant="outline" className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20">
          实时
        </Badge>
      </div>

      {/* Right side */}
      <div className="flex items-center gap-3">
        {/* Theme toggle */}
        <Button
          variant="ghost"
          size="icon"
          className="text-slate-400 hover:text-white"
          onClick={() => setIsDark(!isDark)}
        >
          {isDark ? <Moon className="w-5 h-5" /> : <Sun className="w-5 h-5" />}
        </Button>

        {/* Notifications */}
        <Button variant="ghost" size="icon" className="text-slate-400 hover:text-white relative">
          <Bell className="w-5 h-5" />
          {notifications > 0 && (
            <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 rounded-full text-xs flex items-center justify-center text-white font-medium">
              {notifications}
            </span>
          )}
        </Button>

        {/* User */}
        <div className="flex items-center gap-3 ml-4 pl-4 border-l border-slate-700">
          <div className="text-right hidden sm:block">
            <p className="text-sm font-medium text-white">管理员</p>
            <p className="text-xs text-slate-400">admin@openclaw.io</p>
          </div>
          <Button variant="ghost" size="icon" className="rounded-full bg-slate-800">
            <User className="w-5 h-5 text-slate-400" />
          </Button>
        </div>
      </div>
    </header>
  );
}
