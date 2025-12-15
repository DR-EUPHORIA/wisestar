import React from 'react';
import { Lightbulb, RefreshCw, BookOpen, Zap, TrendingUp } from 'lucide-react';

const QuickActions = ({ onAction, disabled = false }) => {
  const actions = [
    {
      id: 'hint',
      label: '给我提示',
      icon: Lightbulb,
      color: 'from-yellow-500 to-orange-500',
      prompt: '请给我一些解题提示，不要直接给出答案'
    },
    {
      id: 'alternative',
      label: '换个解法',
      icon: RefreshCw,
      color: 'from-blue-500 to-cyan-500',
      prompt: '请用另一种方法解这道题'
    },
    {
      id: 'detailed',
      label: '详细步骤',
      icon: BookOpen,
      color: 'from-purple-500 to-pink-500',
      prompt: '请给出更详细的解题步骤'
    },
    {
      id: 'similar',
      label: '相似题目',
      icon: TrendingUp,
      color: 'from-green-500 to-emerald-500',
      prompt: '请生成一道类似的题目'
    },
    {
      id: 'simplify',
      label: '简化说明',
      icon: Zap,
      color: 'from-red-500 to-orange-500',
      prompt: '请用更简单的方式解释这道题'
    }
  ];

  return (
    <div className="flex flex-wrap gap-2">
      {actions.map((action) => {
        const Icon = action.icon;
        return (
          <button
            key={action.id}
            onClick={() => onAction(action.prompt)}
            disabled={disabled}
            className={`group relative px-3 py-2 rounded-lg text-sm font-medium text-white bg-gradient-to-r ${action.color} hover:shadow-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2`}
          >
            <Icon className="w-4 h-4" />
            <span>{action.label}</span>
          </button>
        );
      })}
    </div>
  );
};

export default QuickActions;
