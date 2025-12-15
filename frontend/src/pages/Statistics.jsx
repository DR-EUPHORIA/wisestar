import React, { useState, useEffect } from 'react';
import { BarChart3, TrendingUp, Award, Target, Calendar, Activity, Loader } from 'lucide-react';
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { getStatistics } from '../utils/api';

const Statistics = () => {
  const [loading, setLoading] = useState(true);
  const [statsData, setStatsData] = useState({
    weeklyData: [],
    knowledgeData: [],
    difficultyData: [],
    progressData: [],
    summary: {
      successRate: 0,
      weeklyCount: 0,
      masteredTopics: 0,
      streakDays: 0
    }
  });

  useEffect(() => {
    loadStatistics();
  }, []);

  const loadStatistics = async () => {
    setLoading(true);
    try {
      const data = await getStatistics();
      setStatsData({
        weeklyData: data.weekly_data || [
          { day: '周一', solved: 3, generated: 2 },
          { day: '周二', solved: 5, generated: 1 },
          { day: '周三', solved: 4, generated: 3 },
          { day: '周四', solved: 6, generated: 2 },
          { day: '周五', solved: 4, generated: 4 },
          { day: '周六', solved: 7, generated: 3 },
          { day: '周日', solved: 5, generated: 2 }
        ],
        knowledgeData: data.knowledge_data || [
          { name: '函数', value: 15, color: '#3b82f6' },
          { name: '导数', value: 12, color: '#8b5cf6' },
          { name: '不等式', value: 10, color: '#10b981' },
          { name: '立体几何', value: 8, color: '#f59e0b' },
          { name: '数列', value: 6, color: '#ef4444' },
          { name: '其他', value: 9, color: '#6b7280' }
        ],
        difficultyData: data.difficulty_data || [
          { name: '简单', value: 20, color: '#10b981' },
          { name: '中等', value: 25, color: '#f59e0b' },
          { name: '困难', value: 15, color: '#ef4444' }
        ],
        progressData: data.progress_data || [
          { month: '1月', rate: 65 },
          { month: '2月', rate: 70 },
          { month: '3月', rate: 68 },
          { month: '4月', rate: 75 },
          { month: '5月', rate: 76 }
        ],
        summary: {
          successRate: data.success_rate || 0.76,
          weeklyCount: data.weekly_count || 34,
          masteredTopics: data.mastered_topics || 12,
          streakDays: data.streak_days || 15
        }
      });
    } catch (error) {
      console.error('加载统计数据失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const { weeklyData, knowledgeData, difficultyData, progressData, summary } = statsData;

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <Loader className="w-12 h-12 text-primary-600 animate-spin mx-auto mb-4" />
          <p className="text-slate-600">加载统计数据中...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-fade-in">
      {/* 页面标题 */}
      <div>
        <h1 className="text-3xl font-bold text-slate-900 flex items-center tracking-tight">
          <BarChart3 className="w-8 h-8 mr-3 text-primary-600" />
          统计分析
        </h1>
        <p className="text-slate-600 mt-2">可视化展示学习数据和进度趋势</p>
      </div>

      {/* 关键指标 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="card-elevated bg-gradient-to-br from-blue-500 to-cyan-500 text-white shadow-medium">
          <div className="flex items-center justify-between mb-2">
            <Target className="w-8 h-8 opacity-80" />
            <span className="text-2xl font-bold">{Math.round(summary.successRate * 100)}%</span>
          </div>
          <div className="text-sm opacity-90">总体成功率</div>
          <div className="mt-2 text-xs opacity-75">↑ 比上月提升 6%</div>
        </div>

        <div className="card-elevated bg-gradient-to-br from-purple-500 to-pink-500 text-white shadow-medium">
          <div className="flex items-center justify-between mb-2">
            <Activity className="w-8 h-8 opacity-80" />
            <span className="text-2xl font-bold">{summary.weeklyCount}</span>
          </div>
          <div className="text-sm opacity-90">本周解题数</div>
          <div className="mt-2 text-xs opacity-75">↑ 比上周增加 8 道</div>
        </div>

        <div className="card-elevated bg-gradient-to-br from-green-500 to-emerald-500 text-white shadow-medium">
          <div className="flex items-center justify-between mb-2">
            <Award className="w-8 h-8 opacity-80" />
            <span className="text-2xl font-bold">{summary.masteredTopics}</span>
          </div>
          <div className="text-sm opacity-90">已掌握知识点</div>
          <div className="mt-2 text-xs opacity-75">新增 2 个知识点</div>
        </div>

        <div className="card-elevated bg-gradient-to-br from-orange-500 to-red-500 text-white shadow-medium">
          <div className="flex items-center justify-between mb-2">
            <Calendar className="w-8 h-8 opacity-80" />
            <span className="text-2xl font-bold">{summary.streakDays}</span>
          </div>
          <div className="text-sm opacity-90">连续学习天数</div>
          <div className="mt-2 text-xs opacity-75">保持良好习惯</div>
        </div>
      </div>

      {/* 图表区域 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 每周活动趋势 */}
        <div className="card-elevated">
          <h2 className="text-lg font-bold text-slate-900 mb-4 flex items-center tracking-tight">
            <TrendingUp className="w-5 h-5 text-primary-600 mr-2" />
            每周活动趋势
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={weeklyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="day" stroke="#64748b" />
              <YAxis stroke="#64748b" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#fff', 
                  border: '1px solid #e2e8f0',
                  borderRadius: '8px',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                }}
              />
              <Legend />
              <Bar dataKey="solved" fill="#3b82f6" name="解题数" radius={[8, 8, 0, 0]} />
              <Bar dataKey="generated" fill="#8b5cf6" name="生成数" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* 知识点分布 */}
        <div className="card-elevated">
          <h2 className="text-lg font-bold text-gray-900 mb-4">知识点分布</h2>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={knowledgeData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {knowledgeData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* 成功率趋势 */}
        <div className="card-elevated">
          <h2 className="text-lg font-bold text-gray-900 mb-4">成功率趋势</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={progressData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="month" stroke="#6b7280" />
              <YAxis stroke="#6b7280" domain={[0, 100]} />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#fff', 
                  border: '1px solid #e2e8f0',
                  borderRadius: '8px',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                }}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="rate" 
                stroke="#10b981" 
                strokeWidth={3}
                name="成功率 (%)"
                dot={{ fill: '#10b981', r: 6 }}
                activeDot={{ r: 8 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* 难度分布 */}
        <div className="card-elevated">
          <h2 className="text-lg font-bold text-gray-900 mb-4">难度分布</h2>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={difficultyData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {difficultyData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 学习建议 */}
      <div className="card-elevated bg-gradient-to-br from-blue-50 to-purple-50 border-2 border-primary-200">
        <h2 className="text-lg font-bold text-slate-900 mb-4 tracking-tight">💡 学习建议</h2>
        <div className="space-y-3">
          <div className="flex items-start">
            <div className="flex-shrink-0 w-2 h-2 rounded-full bg-primary-600 mt-2 mr-3" />
            <p className="text-slate-700">
              <span className="font-semibold">立体几何</span>是当前的薄弱知识点，建议多做相关练习题
            </p>
          </div>
          <div className="flex items-start">
            <div className="flex-shrink-0 w-2 h-2 rounded-full bg-primary-600 mt-2 mr-3" />
            <p className="text-slate-700">
              本周学习活跃度较高，保持这个节奏可以更快提升
            </p>
          </div>
          <div className="flex items-start">
            <div className="flex-shrink-0 w-2 h-2 rounded-full bg-primary-600 mt-2 mr-3" />
            <p className="text-slate-700">
              <span className="font-semibold">函数与导数</span>掌握较好，可以尝试更高难度的题目
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Statistics;
