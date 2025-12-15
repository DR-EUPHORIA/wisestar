import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  MessageSquare,
  Grid3x3,
  User,
  Settings,
  Plus,
  Trash2,
  Sparkles,
  ChevronLeft,
  ChevronRight,
  ClipboardList,
  Pencil
} from 'lucide-react';
import { useChatContext } from '../contexts/ChatContext';

const Sidebar = () => {
  const location = useLocation();
  const [collapsed, setCollapsed] = useState(false);
  const { 
    conversations, 
    currentConversationId, 
    createNewConversation, 
    switchConversation,
    deleteConversation 
  } = useChatContext();

  const mainNavigation = [
    { name: '对话', path: '/', icon: MessageSquare },
    { name: '功能', path: '/features', icon: Grid3x3 },
    { name: '智能组卷', path: '/exam-paper', icon: ClipboardList },
    { name: '几何画图', path: '/plot', icon: Pencil },
    { name: '个人', path: '/profile', icon: User },
  ];

  const isActive = (path) => {
    if (path === '/') {
      return location.pathname === '/';
    }
    return location.pathname.startsWith(path);
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffDays = Math.floor((now - date) / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) return '今天';
    if (diffDays === 1) return '昨天';
    if (diffDays < 7) return `${diffDays}天前`;
    return date.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' });
  };

  const groupedConversations = conversations.reduce((groups, conv) => {
    const date = formatDate(conv.createdAt);
    if (!groups[date]) groups[date] = [];
    groups[date].push(conv);
    return groups;
  }, {});

  return (
    <aside className={`fixed inset-y-0 left-0 z-50 flex flex-col bg-gradient-to-b from-primary-700 via-primary-600 to-primary-700 text-white transition-all duration-300 ${collapsed ? 'w-16' : 'w-16 md:w-72'}`}>
      <div className="flex items-center justify-between h-16 px-4 border-b border-white/10">
        {!collapsed && (
          <div className="hidden md:flex items-center space-x-2">
            <div className="w-8 h-8 rounded-lg bg-white/10 backdrop-blur-sm flex items-center justify-center">
              <Sparkles className="w-5 h-5" />
            </div>
            <span className="text-lg font-bold">WiseStar</span>
          </div>
        )}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="p-1.5 hover:bg-white/10 rounded-lg transition-colors hidden md:block"
        >
          {collapsed ? <ChevronRight className="w-5 h-5" /> : <ChevronLeft className="w-5 h-5" />}
        </button>
        <div className="md:hidden w-8 h-8 rounded-lg bg-white/10 backdrop-blur-sm flex items-center justify-center mx-auto">
          <Sparkles className="w-5 h-5" />
        </div>
      </div>

      {!collapsed && (
        <div className="px-3 py-4 hidden md:block">
          <button
            onClick={createNewConversation}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-white/10 hover:bg-white/20 rounded-xl transition-all duration-200 font-medium"
          >
            <Plus className="w-5 h-5" />
            新建对话
          </button>
        </div>
      )}
      <div className="px-3 py-4 md:hidden">
        <button
          onClick={createNewConversation}
          className="w-full flex items-center justify-center p-2 bg-white/10 hover:bg-white/20 rounded-xl transition-all duration-200"
        >
          <Plus className="w-5 h-5" />
        </button>
      </div>

      <nav className="flex-1 px-3 overflow-y-auto scrollbar-hide">
        {!collapsed ? (
          <div className="space-y-6 hidden md:block">
            {Object.entries(groupedConversations).map(([date, convs]) => (
              <div key={date}>
                <div className="text-xs font-semibold text-primary-200 mb-2 px-2">{date}</div>
                <div className="space-y-1">
                  {convs.map((conv) => (
                    <div
                      key={conv.id}
                      className={`group relative flex items-center gap-2 px-3 py-2.5 rounded-lg transition-all duration-200 cursor-pointer ${
                        conv.id === currentConversationId
                          ? 'bg-white/20'
                          : 'hover:bg-white/10'
                      }`}
                      onClick={() => switchConversation(conv.id)}
                    >
                      <MessageSquare className="w-4 h-4 flex-shrink-0 text-primary-200" />
                      <span className="flex-1 text-sm truncate">{conv.title}</span>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteConversation(conv.id);
                        }}
                        className="opacity-0 group-hover:opacity-100 p-1 hover:bg-white/10 rounded transition-all"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="space-y-2">
            {conversations.slice(0, 5).map((conv) => (
              <button
                key={conv.id}
                onClick={() => switchConversation(conv.id)}
                className={`w-full p-2 rounded-lg transition-colors ${
                  conv.id === currentConversationId ? 'bg-white/20' : 'hover:bg-white/10'
                }`}
              >
                <MessageSquare className="w-5 h-5 mx-auto" />
              </button>
            ))}
          </div>
        )}
      </nav>

      <div className="border-t border-white/10">
        {mainNavigation.map((item) => {
          const Icon = item.icon;
          const active = isActive(item.path);
          
          return (
            <Link
              key={item.path}
              to={item.path}
              className={`flex items-center gap-3 px-4 py-3 transition-colors ${
                active ? 'bg-white/20' : 'hover:bg-white/10'
              } ${collapsed ? 'justify-center' : ''}`}
            >
              <Icon className="w-5 h-5" />
              {!collapsed && <span className="font-medium">{item.name}</span>}
            </Link>
          );
        })}
      </div>
    </aside>
  );
};

export default Sidebar;
