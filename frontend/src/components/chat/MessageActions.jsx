import React, { useState } from 'react';
import {
  RotateCcw, MessageCircle, Copy, CheckCircle, Calculator, FileText,
  Brain, BarChart3, Lightbulb, Play, Wand2, BookOpen, Target, Zap
} from 'lucide-react';

const MessageActions = ({ message, onRegenerate, onFollowUp, onQuickAction }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const getSmartActions = () => {
    const actions = [];
    const metadata = message.metadata;
    const content = message.content.toLowerCase();

    // 基于消息类型的智能按钮
    if (metadata?.type === 'solve_result' && metadata.data?.success) {
      actions.push({
        icon: FileText,
        label: '生成类似题目',
        prompt: '生成一道类似的题目',
        color: 'purple'
      });
      actions.push({
        icon: Brain,
        label: '解题思路',
        prompt: '请详细解释这道题的解题思路和方法',
        color: 'blue'
      });
    }

    if (metadata?.type === 'generate_result' && metadata.data?.problem) {
      actions.push({
        icon: Calculator,
        label: '解这道题',
        prompt: `解这道题：${metadata.data.problem}`,
        color: 'green'
      });
      actions.push({
        icon: Wand2,
        label: '生成变式',
        prompt: '基于这道题生成一道变式题目',
        color: 'orange'
      });
    }

    // 基于内容的智能推荐
    if (content.includes('函数') || content.includes('导数')) {
      actions.push({
        icon: Target,
        label: '函数练习',
        prompt: '生成一道函数相关的练习题',
        color: 'indigo'
      });
    }

    if (content.includes('几何') || content.includes('三角形') || content.includes('圆')) {
      actions.push({
        icon: BookOpen,
        label: '几何练习',
        prompt: '生成一道几何题目',
        color: 'teal'
      });
    }

    // 通用智能推荐
    if (message.role === 'assistant' && !actions.length) {
      actions.push({
        icon: Lightbulb,
        label: '相关练习',
        prompt: '基于刚才的内容，给我推荐一些相关的练习题',
        color: 'yellow'
      });
      actions.push({
        icon: BarChart3,
        label: '查看统计',
        prompt: '查看我的学习统计数据',
        color: 'blue'
      });
    }

    return actions.slice(0, 3); // 最多显示3个智能按钮
  };

  const smartActions = getSmartActions();

  if (message.role !== 'assistant') return null;

  return (
    <div className="mt-3 pt-3 border-t border-slate-100 space-y-2">
      {/* 智能辅助按钮 */}
      {smartActions.length > 0 && (
        <div className="flex flex-wrap gap-2">
          <div className="flex items-center gap-1 text-xs text-slate-500 font-medium">
            <Zap className="w-3 h-3" />
            智能推荐:
          </div>
          {smartActions.map((action, index) => {
            const Icon = action.icon;
            const colorClasses = {
              purple: 'text-purple-600 bg-purple-50 hover:bg-purple-100 border-purple-200',
              blue: 'text-blue-600 bg-blue-50 hover:bg-blue-100 border-blue-200',
              green: 'text-green-600 bg-green-50 hover:bg-green-100 border-green-200',
              orange: 'text-orange-600 bg-orange-50 hover:bg-orange-100 border-orange-200',
              indigo: 'text-indigo-600 bg-indigo-50 hover:bg-indigo-100 border-indigo-200',
              teal: 'text-teal-600 bg-teal-50 hover:bg-teal-100 border-teal-200',
              yellow: 'text-yellow-700 bg-yellow-50 hover:bg-yellow-100 border-yellow-200'
            };

            return (
              <button
                key={index}
                onClick={() => onQuickAction?.(action.prompt)}
                className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border transition-all hover:scale-105 hover:shadow-sm ${colorClasses[action.color] || colorClasses.blue}`}
              >
                <Icon className="w-3.5 h-3.5" />
                {action.label}
              </button>
            );
          })}
        </div>
      )}

      {/* 基础操作按钮 */}
      <div className="flex items-center gap-2">
        <button
          onClick={onRegenerate}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-600 hover:text-primary-600 hover:bg-primary-50 rounded-lg transition-colors"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          重新生成
        </button>

        <button
          onClick={onFollowUp}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-600 hover:text-primary-600 hover:bg-primary-50 rounded-lg transition-colors"
        >
          <MessageCircle className="w-3.5 h-3.5" />
          追问
        </button>

        <button
          onClick={handleCopy}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-600 hover:text-primary-600 hover:bg-primary-50 rounded-lg transition-colors"
        >
          {copied ? (
            <>
              <CheckCircle className="w-3.5 h-3.5" />
              已复制
            </>
          ) : (
            <>
              <Copy className="w-3.5 h-3.5" />
              复制
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default MessageActions;
