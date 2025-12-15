import React, { useState, useEffect } from 'react';
import { Calendar, RefreshCw, CheckCircle, Clock, Tag, TrendingUp, Loader } from 'lucide-react';
import { getDailyQuestion, submitDailyAnswer } from '../utils/api';

const DailyQuestion = () => {
  const [loading, setLoading] = useState(true);
  const [completed, setCompleted] = useState(false);
  const [showAnswer, setShowAnswer] = useState(false);
  const [todayQuestion, setTodayQuestion] = useState(null);
  const [stats, setStats] = useState({
    streak: 0,
    total_completed: 0,
    success_rate: 0
  });
  const [history, setHistory] = useState([]);

  useEffect(() => {
    loadDailyQuestion();
  }, []);

  const loadDailyQuestion = async () => {
    setLoading(true);
    try {
      const data = await getDailyQuestion();
      setTodayQuestion({
        date: data.date || new Date().toISOString().split('T')[0],
        question: data.question || '暂无每日一题',
        tags: data.tags || [],
        difficulty: data.difficulty || '中等',
        strategy: data.strategy || 'balanced',
        answer: data.answer || '',
        hint: data.hint || ''
      });
      setStats({
        streak: data.streak || 0,
        total_completed: data.total_completed || 0,
        success_rate: data.success_rate || 0
      });
      setHistory(data.history || []);
    } catch (error) {
      console.error('加载每日一题失败:', error);
      setTodayQuestion({
        date: new Date().toISOString().split('T')[0],
        question: '加载失败，请稍后重试',
        tags: [],
        difficulty: '中等',
        strategy: 'balanced',
        answer: '',
        hint: ''
      });
    } finally {
      setLoading(false);
    }
  };

  const handleComplete = async () => {
    if (!todayQuestion) return;
    try {
      await submitDailyAnswer(1, '已完成');
      setCompleted(true);
    } catch (error) {
      console.error('提交答案失败:', error);
    }
  };

  const handleRefresh = () => {
    loadDailyQuestion();
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <Loader className="w-12 h-12 text-primary-600 animate-spin mx-auto mb-4" />
          <p className="text-slate-600">加载每日一题中...</p>
        </div>
      </div>
    );
  }

  if (!todayQuestion) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <p className="text-slate-600">暂无每日一题</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-fade-in">
      {/* 页面标题 */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-900 flex items-center tracking-tight">
            <Calendar className="w-8 h-8 mr-3 text-primary-600" />
            每日一题
          </h1>
          <p className="text-slate-600 mt-2">每天推荐一道题目，巩固学习效果</p>
        </div>
        <button onClick={handleRefresh} className="btn-outline flex items-center">
          <RefreshCw className="w-4 h-4 mr-2" />
          换一题
        </button>
      </div>

      {/* 统计卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="card bg-gradient-to-br from-orange-50 to-red-50">
          <div className="flex items-center justify-between mb-2">
            <TrendingUp className="w-6 h-6 text-orange-600" />
            <span className="text-3xl font-bold text-orange-600">{stats.streak}</span>
          </div>
          <div className="text-sm text-slate-600">连续完成天数</div>
        </div>

        <div className="card bg-gradient-to-br from-blue-50 to-cyan-50">
          <div className="flex items-center justify-between mb-2">
            <CheckCircle className="w-6 h-6 text-blue-600" />
            <span className="text-3xl font-bold text-blue-600">{stats.total_completed}</span>
          </div>
          <div className="text-sm text-slate-600">累计完成题数</div>
        </div>

        <div className="card bg-gradient-to-br from-green-50 to-emerald-50">
          <div className="flex items-center justify-between mb-2">
            <Clock className="w-6 h-6 text-green-600" />
            <span className="text-3xl font-bold text-green-600">{(stats.success_rate * 100).toFixed(0)}%</span>
          </div>
          <div className="text-sm text-slate-600">正确率</div>
        </div>
      </div>

      {/* 今日题目 */}
      <div className="card-elevated">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center">
            <Calendar className="w-5 h-5 text-primary-600 mr-2" />
            <h2 className="text-xl font-bold text-slate-900 tracking-tight">今日题目</h2>
          </div>
          <span className="text-sm text-slate-500">{todayQuestion.date}</span>
        </div>

        <div className="bg-gradient-to-br from-blue-50 to-purple-50 rounded-xl p-6 mb-4 border border-blue-100">
          <div className="whitespace-pre-wrap text-slate-800 leading-relaxed">
            {todayQuestion.question}
          </div>
        </div>

        {/* 标签和信息 */}
        <div className="flex flex-wrap items-center gap-2 mb-4">
          {todayQuestion.tags.map((tag, i) => (
            <span key={i} className="badge badge-primary">
              <Tag className="w-3 h-3 mr-1" />
              {tag}
            </span>
          ))}
          <span className={`badge ${
            todayQuestion.difficulty === '简单' ? 'badge-success' :
            todayQuestion.difficulty === '中等' ? 'badge-warning' :
            'badge-error'
          }`}>
            {todayQuestion.difficulty}
          </span>
        </div>

        {/* 提示 */}
        {!showAnswer && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-4">
            <p className="text-sm text-yellow-800">
              <span className="font-semibold">💡 提示：</span>
              {todayQuestion.hint}
            </p>
          </div>
        )}

        {/* 答案区域 */}
        {showAnswer && (
          <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-4 animate-slide-up">
            <h3 className="font-semibold text-green-900 mb-2">参考答案：</h3>
            <div className="whitespace-pre-wrap text-green-800">
              {todayQuestion.answer}
            </div>
          </div>
        )}

        {/* 操作按钮 */}
        <div className="flex gap-3">
          {!completed ? (
            <>
              <button onClick={handleComplete} className="btn-primary flex-1">
                <CheckCircle className="w-4 h-4 mr-2" />
                标记为已完成
              </button>
              <button onClick={() => setShowAnswer(!showAnswer)} className="btn-outline flex-1">
                {showAnswer ? '隐藏答案' : '查看答案'}
              </button>
            </>
          ) : (
            <div className="w-full text-center py-3 bg-green-100 text-green-700 rounded-lg font-medium">
              <CheckCircle className="w-5 h-5 inline mr-2" />
              今日题目已完成
            </div>
          )}
        </div>
      </div>

      {/* 完成历史 */}
      <div className="card-elevated">
        <h2 className="text-lg font-bold text-slate-900 mb-4 tracking-tight">完成历史</h2>
        <div className="grid grid-cols-7 gap-2">
          {history.map((day, index) => (
            <div
              key={index}
              className={`aspect-square rounded-lg flex flex-col items-center justify-center text-xs ${
                day.completed
                  ? day.success
                    ? 'bg-green-100 text-green-700'
                    : 'bg-red-100 text-red-700'
                  : 'bg-slate-100 text-slate-400'
              }`}
            >
              <div className="font-semibold">{day.date.split('-')[2]}</div>
              {day.completed && (
                <div className="mt-1">
                  {day.success ? '✓' : '✗'}
                </div>
              )}
            </div>
          ))}
        </div>
        <p className="text-xs text-slate-500 mt-3">
          绿色表示完成且正确，红色表示完成但错误，灰色表示未完成
        </p>
      </div>

      {/* 推荐策略说明 */}
      <div className="card-elevated bg-gradient-to-br from-purple-50 to-pink-50 border-2 border-purple-200">
        <h2 className="text-lg font-bold text-slate-900 mb-3 tracking-tight">📚 推荐策略</h2>
        <p className="text-sm text-slate-700 mb-2">
          当前使用：<span className="font-semibold text-purple-700">平衡模式</span>
        </p>
        <p className="text-sm text-slate-600 leading-relaxed">
          系统会综合考虑你的薄弱知识点和复习需求，智能推荐最适合的题目。
          70%概率推荐薄弱知识点题目，30%概率推荐复习题目。
        </p>
      </div>
    </div>
  );
};

export default DailyQuestion;
