import React, { useState, useEffect } from 'react';
import { User, Bot, CheckCircle, XCircle, Loader, Brain, Code, TrendingUp, FileText, Wrench, Play, Sparkles } from 'lucide-react';
import MarkdownRenderer from '../MarkdownRenderer';
import MessageActions from './MessageActions';

const DEFAULT_AVATAR = 'https://avatars.githubusercontent.com/u/9919?s=200&v=4';

const Message = ({ message }) => {
  const isUser = message.role === 'user';
  const isSystem = message.role === 'system';
  const [userAvatar, setUserAvatar] = useState(DEFAULT_AVATAR);

  useEffect(() => {
    const saved = localStorage.getItem('userProfile');
    if (saved) {
      const parsed = JSON.parse(saved);
      setUserAvatar(parsed.avatar || DEFAULT_AVATAR);
    }
  }, []);

  if (isSystem) {
    return (
      <div className="flex justify-center my-4">
        <div className="bg-slate-100 text-slate-600 px-4 py-2 rounded-full text-sm">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className={`flex gap-4 mb-6 ${isUser ? 'justify-end' : 'justify-start'}`}>
      {!isUser && (
        <div className="flex-shrink-0 w-10 h-10 rounded-xl bg-gradient-to-br from-primary-600 to-primary-700 flex items-center justify-center shadow-soft">
          <Sparkles className="w-6 h-6 text-white" />
        </div>
      )}
      
      <div className={`flex-1 max-w-3xl ${isUser ? 'flex justify-end' : ''}`}>
        <div className={`${isUser ? 'bg-primary-600 text-white' : 'bg-white border border-slate-200'} rounded-2xl px-5 py-4 shadow-soft`}>
          {message.isStreaming && (
            <div className="flex items-center gap-2 mb-2 text-sm opacity-70">
              <Loader className="w-4 h-4 animate-spin" />
              <span>思考中...</span>
            </div>
          )}
          
          <div className={`${isUser ? 'text-white' : 'text-slate-700'}`}>
            <MarkdownRenderer content={message.content} />
          </div>

          {message.toolCalls && message.toolCalls.length > 0 && (
            <ToolCallsDisplay toolCalls={message.toolCalls} isUser={isUser} />
          )}

          {message.metadata && (
            <MessageMetadata metadata={message.metadata} isUser={isUser} />
          )}

          {message.timestamp && (
            <div className={`text-xs mt-3 ${isUser ? 'text-primary-100' : 'text-slate-400'}`}>
              {new Date(message.timestamp).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}
            </div>
          )}

          {!isUser && (
            <MessageActions 
              message={message}
              onRegenerate={() => console.log('Regenerate:', message.id)}
              onFollowUp={() => console.log('Follow up:', message.id)}
            />
          )}
        </div>
      </div>

      {isUser && (
        <div className="flex-shrink-0 w-10 h-10 rounded-xl overflow-hidden shadow-soft ring-2 ring-white">
          <img 
            src={userAvatar} 
            alt="User Avatar" 
            className="w-full h-full object-cover"
            onError={(e) => {
              e.target.onerror = null;
              e.target.src = DEFAULT_AVATAR;
            }}
          />
        </div>
      )}
    </div>
  );
};

const ToolCallsDisplay = ({ toolCalls, isUser }) => {
  return (
    <div className="mt-3 space-y-2">
      <div className={`text-xs font-semibold ${isUser ? 'text-primary-100' : 'text-slate-500'} flex items-center gap-1`}>
        <Wrench className="w-3 h-3" />
        工具调用
      </div>
      {toolCalls.map((call, index) => (
        <div key={index} className={`p-3 rounded-lg text-sm ${isUser ? 'bg-white/10' : 'bg-slate-50 border border-slate-200'}`}>
          <div className="flex items-center gap-2 mb-2">
            <Play className={`w-3.5 h-3.5 ${isUser ? 'text-primary-200' : 'text-primary-600'}`} />
            <span className={`font-semibold ${isUser ? 'text-white' : 'text-slate-900'}`}>
              {call.tool}
            </span>
            {call.status && (
              <span className={`text-xs px-2 py-0.5 rounded-full ${
                call.status === 'success' 
                  ? 'bg-green-100 text-green-700' 
                  : call.status === 'running'
                  ? 'bg-blue-100 text-blue-700'
                  : 'bg-red-100 text-red-700'
              }`}>
                {call.status === 'success' ? '成功' : call.status === 'running' ? '执行中' : '失败'}
              </span>
            )}
          </div>
          {call.params && (
            <div className={`text-xs ${isUser ? 'text-primary-100' : 'text-slate-600'} mt-1`}>
              <span className="opacity-70">参数: </span>
              <code className={`${isUser ? 'bg-white/10' : 'bg-slate-100'} px-1.5 py-0.5 rounded`}>
                {typeof call.params === 'string' ? call.params : JSON.stringify(call.params)}
              </code>
            </div>
          )}
          {call.result && (
            <div className={`text-xs ${isUser ? 'text-primary-100' : 'text-slate-600'} mt-2`}>
              <span className="opacity-70">结果: </span>
              <div className={`${isUser ? 'bg-white/10' : 'bg-slate-100'} px-2 py-1.5 rounded mt-1 font-mono`}>
                {call.result}
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

const MessageMetadata = ({ metadata, isUser }) => {
  const { type, data } = metadata;

  if (type === 'solve_result') {
    return (
      <div className="mt-4 space-y-3">
        <div className={`flex items-center gap-2 ${isUser ? 'text-white' : 'text-slate-700'}`}>
          {data.success ? (
            <CheckCircle className="w-5 h-5 text-green-500" />
          ) : (
            <XCircle className="w-5 h-5 text-red-500" />
          )}
          <span className="font-semibold">
            {data.success ? '解题成功' : '解题失败'}
          </span>
        </div>

        {data.answer && (
          <div className={`p-4 rounded-xl ${isUser ? 'bg-white/10' : 'bg-gradient-to-r from-green-50 to-emerald-50 border border-green-100'}`}>
            <div className={`text-sm mb-1 ${isUser ? 'text-primary-100' : 'text-slate-600'}`}>最终答案：</div>
            <div className={`text-lg font-bold ${isUser ? 'text-white' : 'text-slate-900'}`}>{data.answer}</div>
          </div>
        )}

        {data.statistics && (
          <div className="grid grid-cols-3 gap-3">
            <StatCard label="总步数" value={data.statistics.total_steps} isUser={isUser} />
            <StatCard label="推理" value={data.statistics.reasoning_steps} isUser={isUser} />
            <StatCard label="计算" value={data.statistics.calculation_steps} isUser={isUser} />
          </div>
        )}

        {data.steps && data.steps.length > 0 && (
          <details className={`mt-3 ${isUser ? 'text-white' : 'text-slate-700'}`}>
            <summary className="cursor-pointer font-semibold mb-2">查看解题步骤</summary>
            <div className="space-y-3 mt-3">
              {data.steps.map((step, index) => (
                <StepCard key={index} step={step} index={index} isUser={isUser} />
              ))}
            </div>
          </details>
        )}
      </div>
    );
  }

  if (type === 'generate_result') {
    return (
      <div className="mt-4 space-y-3">
        <div className={`flex items-center gap-2 ${isUser ? 'text-white' : 'text-slate-700'}`}>
          <FileText className="w-5 h-5 text-purple-500" />
          <span className="font-semibold">题目生成成功</span>
        </div>

        {data.problem && (
          <div className={`p-4 rounded-xl ${isUser ? 'bg-white/10' : 'bg-gradient-to-r from-purple-50 to-pink-50 border border-purple-100'}`}>
            <div className={`text-sm mb-2 ${isUser ? 'text-primary-100' : 'text-slate-600'}`}>生成的题目：</div>
            <div className={`${isUser ? 'text-white' : 'text-slate-900'}`}>{data.problem}</div>
          </div>
        )}

        {data.quality_score && (
          <div className={`p-3 rounded-xl ${isUser ? 'bg-white/10' : 'bg-slate-50'}`}>
            <div className={`text-sm mb-2 ${isUser ? 'text-primary-100' : 'text-slate-600'}`}>质量评分：</div>
            <div className="flex items-center gap-2">
              <div className="flex-1 bg-slate-200 rounded-full h-2">
                <div 
                  className="bg-gradient-to-r from-green-500 to-emerald-500 h-2 rounded-full transition-all"
                  style={{ width: `${data.quality_score * 10}%` }}
                />
              </div>
              <span className={`font-bold ${isUser ? 'text-white' : 'text-slate-900'}`}>
                {data.quality_score.toFixed(1)}
              </span>
            </div>
          </div>
        )}
      </div>
    );
  }

  if (type === 'statistics') {
    return (
      <div className="mt-4 space-y-3">
        <div className={`flex items-center gap-2 ${isUser ? 'text-white' : 'text-slate-700'}`}>
          <TrendingUp className="w-5 h-5 text-blue-500" />
          <span className="font-semibold">学习统计</span>
        </div>

        <div className="grid grid-cols-2 gap-3">
          {data.total_questions !== undefined && (
            <StatCard label="总题数" value={data.total_questions} isUser={isUser} />
          )}
          {data.success_rate !== undefined && (
            <StatCard label="成功率" value={`${(data.success_rate * 100).toFixed(1)}%`} isUser={isUser} />
          )}
        </div>
      </div>
    );
  }

  return null;
};

const StatCard = ({ label, value, isUser }) => (
  <div className={`p-3 rounded-xl text-center ${isUser ? 'bg-white/10' : 'bg-slate-50'}`}>
    <div className={`text-2xl font-bold ${isUser ? 'text-white' : 'text-primary-600'}`}>{value}</div>
    <div className={`text-xs ${isUser ? 'text-primary-100' : 'text-slate-600'}`}>{label}</div>
  </div>
);

const StepCard = ({ step, index, isUser }) => (
  <div className={`flex gap-3 p-3 rounded-xl ${isUser ? 'bg-white/5' : 'bg-slate-50'}`}>
    <div className={`flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center font-bold text-sm ${isUser ? 'bg-white/20 text-white' : 'bg-primary-100 text-primary-700'}`}>
      {index + 1}
    </div>
    <div className="flex-1">
      <div className="flex items-center gap-2 mb-1">
        {step.type === 'reasoning' ? (
          <Brain className="w-4 h-4 text-purple-500" />
        ) : (
          <Code className="w-4 h-4 text-blue-500" />
        )}
        <span className={`text-xs font-semibold uppercase ${isUser ? 'text-primary-100' : 'text-slate-500'}`}>
          {step.type === 'reasoning' ? '推理' : '计算'}
        </span>
      </div>
      <p className={`text-sm ${isUser ? 'text-white' : 'text-slate-700'}`}>{step.content}</p>
      {step.code && (
        <pre className="mt-2 bg-slate-900 text-slate-100 p-2 rounded-lg text-xs overflow-x-auto">
          <code>{step.code}</code>
        </pre>
      )}
    </div>
  </div>
);

export default Message;
