import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  Home, 
  Calculator, 
  FileText, 
  Brain, 
  BarChart3, 
  Calendar,
  Menu,
  X,
  Sparkles
} from 'lucide-react';

const Layout = ({ children }) => {
  const location = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const navigation = [
    { name: '仪表盘', path: '/dashboard', icon: Home },
    { name: '智能解题', path: '/solve', icon: Calculator },
    { name: '题目生成', path: '/generate', icon: FileText },
    { name: '学习记忆', path: '/memory', icon: Brain },
    { name: '统计分析', path: '/statistics', icon: BarChart3 },
    { name: '每日一题', path: '/daily', icon: Calendar },
  ];

  const isActive = (path) => location.pathname === path;

  return (
    <div className="min-h-screen bg-slate-50">
      {/* 侧边栏 - 桌面端 */}
      <aside className="hidden md:flex md:flex-shrink-0 fixed inset-y-0 left-0 z-50">
        <div className="flex flex-col w-72 bg-white/95 backdrop-blur-xl border-r border-slate-200/60 shadow-large">
          {/* Logo */}
          <div className="flex items-center h-20 px-6 border-b border-slate-200/60 bg-gradient-to-br from-primary-600 via-primary-700 to-primary-800">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 rounded-xl bg-white/20 backdrop-blur-sm flex items-center justify-center">
                <Sparkles className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-white tracking-tight">WiseStar</h1>
                <p className="text-xs text-primary-100">数学智能体</p>
              </div>
            </div>
          </div>

          {/* 导航菜单 */}
          <nav className="flex-1 px-3 py-6 space-y-1 overflow-y-auto">
            {navigation.map((item) => {
              const Icon = item.icon;
              const active = isActive(item.path);
              
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`flex items-center px-4 py-3.5 rounded-xl transition-all duration-200 group relative ${
                    active
                      ? 'bg-primary-50 text-primary-700 shadow-soft'
                      : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
                  }`}
                >
                  {active && (
                    <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-8 bg-primary-600 rounded-r-full" />
                  )}
                  <Icon className={`w-5 h-5 mr-3 ${active ? 'text-primary-600' : 'text-slate-400 group-hover:text-slate-600'}`} />
                  <span className="font-medium text-sm">{item.name}</span>
                </Link>
              );
            })}
          </nav>

          {/* 底部信息 */}
          <div className="p-4 border-t border-slate-200/60 bg-slate-50/50">
            <div className="text-xs text-slate-500 text-center space-y-1">
              <p className="font-semibold text-slate-700">智多星数学智能体</p>
              <p className="text-slate-400">Version 1.2.0</p>
            </div>
          </div>
        </div>
      </aside>

      {/* 移动端侧边栏 */}
      {sidebarOpen && (
        <div className="fixed inset-0 z-50 md:hidden">
          <div className="fixed inset-0 bg-gray-600 bg-opacity-75" onClick={() => setSidebarOpen(false)} />
          <div className="fixed inset-y-0 left-0 flex flex-col w-64 bg-white shadow-xl">
            <div className="flex items-center justify-between h-16 px-6 border-b border-gray-200 bg-gradient-to-r from-primary-600 to-secondary-600">
              <div className="flex items-center">
                <Sparkles className="w-6 h-6 text-white mr-2" />
                <h1 className="text-lg font-bold text-white">WiseStar</h1>
              </div>
              <button onClick={() => setSidebarOpen(false)} className="text-white">
                <X className="w-6 h-6" />
              </button>
            </div>
            
            <nav className="flex-1 px-4 py-6 space-y-2 overflow-y-auto">
              {navigation.map((item) => {
                const Icon = item.icon;
                const active = isActive(item.path);
                
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    onClick={() => setSidebarOpen(false)}
                    className={`flex items-center px-4 py-3 rounded-lg transition-all duration-200 ${
                      active
                        ? 'bg-primary-50 text-primary-700'
                        : 'text-gray-600 hover:bg-gray-50'
                    }`}
                  >
                    <Icon className={`w-5 h-5 mr-3 ${active ? 'text-primary-600' : 'text-gray-400'}`} />
                    <span className="font-medium">{item.name}</span>
                  </Link>
                );
              })}
            </nav>
          </div>
        </div>
      )}

      {/* 主内容区 */}
      <div className="md:pl-72 flex flex-col min-h-screen">
        {/* 顶部导航栏 - 移动端 */}
        <header className="md:hidden sticky top-0 z-40 bg-white/95 backdrop-blur-xl border-b border-slate-200/60 shadow-soft">
          <div className="flex items-center justify-between h-16 px-4">
            <button onClick={() => setSidebarOpen(true)} className="text-slate-600 hover:text-slate-900">
              <Menu className="w-6 h-6" />
            </button>
            <div className="flex items-center">
              <Sparkles className="w-6 h-6 text-primary-600 mr-2" />
              <h1 className="text-lg font-bold text-slate-900">WiseStar</h1>
            </div>
            <div className="w-6" />
          </div>
        </header>

        {/* 页面内容 */}
        <main className="flex-1 p-6 md:p-10">
          <div className="max-w-7xl mx-auto">
            {children}
          </div>
        </main>

        {/* 页脚 */}
        <footer className="bg-white/80 backdrop-blur-sm border-t border-slate-200/60 py-8 mt-auto">
          <div className="max-w-7xl mx-auto px-6 md:px-10">
            <p className="text-center text-sm text-slate-500">
              © 2025 WiseStar-MathAgent. 淘天集团 - 用户&内容技术团队
            </p>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default Layout;
