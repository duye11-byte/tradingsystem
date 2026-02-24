import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';
import { Dashboard } from '@/pages/Dashboard';
import { Market } from '@/pages/Market';
import { Signals } from '@/pages/Signals';
import { Orders } from '@/pages/Orders';
import { Positions } from '@/pages/Positions';
import { Risk } from '@/pages/Risk';
import { DataSources } from '@/pages/DataSources';
import { Logs } from '@/pages/Logs';
import { Settings } from '@/pages/Settings';
import Strategy from '@/pages/Strategy';

function App() {
  return (
    <BrowserRouter>
      <div className="flex min-h-screen bg-slate-950">
        {/* Sidebar */}
        <Sidebar />
        
        {/* Main Content */}
        <div className="flex-1 flex flex-col min-w-0">
          <Header />
          <main className="flex-1 overflow-auto">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/market" element={<Market />} />
              <Route path="/signals" element={<Signals />} />
              <Route path="/orders" element={<Orders />} />
              <Route path="/positions" element={<Positions />} />
              <Route path="/risk" element={<Risk />} />
              <Route path="/strategy" element={<Strategy />} />
              <Route path="/datasources" element={<DataSources />} />
              <Route path="/logs" element={<Logs />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </main>
        </div>
      </div>
    </BrowserRouter>
  );
}

export default App;
