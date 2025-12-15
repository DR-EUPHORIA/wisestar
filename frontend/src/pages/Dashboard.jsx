import React from 'react';
import { Link } from 'react-router-dom';
import { Calculator, FileText, Brain, TrendingUp, Calendar, Sparkles } from 'lucide-react';

const Dashboard = () => {
  const features = [
    {
      title: '智能解题',
      description: '使用RePI系统自动解答数学题目，支持推理和计算',
      icon: Calculator,
      path: '/solve',
      color: 'from-blue-500 to-cyan-500',
      stats: '已解题 0 道'
    },
    {
      title: '题目生成',
      description: 'AI自动生成高质量数学题目，支持多种难度和类型',
      icon: FileText,
      path: '/generate',
      color: 'from-purple-500 to-pink-500',
      stats: '已生成 0 道'
    },
    {
      title: '学习记忆',
      description: '智能记录学习历史，分析薄弱知识点',
      icon: Brain,
      path: '/memory',
      color: 'from-green-500 to-emerald-500',
      stats: '记录 0 条'
    },
    {
      title: '统计分析',
      description: '可视化展示学习数据和进度趋势',
      icon: TrendingUp,
      path: '/statistics',
      color: 'from-orange-500 to-red-500',
      stats: '成功率 0%'
    },
    {
      title: '每日一题',
      description: '每天推荐一道题目，巩固学习效果',
      icon: Calendar,
      path: '/daily',
      color: 'from-indigo-500 to-purple-500',
      stats: '今日未完成'
    }
  ];

  return (
    <div className="space-y-8 animate-fade-in">
      {/* 欢迎区域 */}
      <div className="relative overflow-hidden bg-gradient-to-br from-primary-600 via-primary-700 to-primary-800 rounded-3xl shadow-large p-10 text-white">
        <div className="absolute top-0 right-0 w-96 h-96 bg-white/5 rounded-full -translate-y-1/2 translate-x-1/2" />
        <div className="absolute bottom-0 left-0 w-64 h-64 bg-white/5 rounded-full translate-y-1/2 -translate-x-1/2" />
        <div className="relative z-10">
          <div className="flex items-center mb-6">
            <div className="w-14 h-14 rounded-2xl bg-white/20 backdrop-blur-sm flex items-center justify-center mr-4">
              <Sparkles className="w-8 h-8" />
            </div>
            <div>
              <h1 className="text-4xl font-bold tracking-tight">WiseStar-MathAgent</h1>
              <p className="text-primary-100 text-sm mt-1">企业级数学智能体系统</p>
            </div>
          </div>
          <p className="text-lg text-primary-50 mb-8 max-w-2xl leading-relaxed">
            集智能解题、题目生成、学习记忆于一体的AI数学助手，为您提供专业的数学学习解决方案
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20">
              <div className="text-3xl font-bold mb-2">v1.2.0</div>
              <div className="text-sm text-primary-100">当前版本</div>
            </div>
            <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20">
              <div className="text-3xl font-bold mb-2">6+</div>
              <div className="text-sm text-primary-100">核心功能</div>
            </div>
            <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20">
              <div className="text-3xl font-bold mb-2">AI驱动</div>
              <div className="text-sm text-primary-100">智能推荐</div>
            </div>
          </div>
        </div>
      </div>

      {/* 功能卡片 */}
      <div>
        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="text-3xl font-bold text-slate-900 tracking-tight">核心功能</h2>
            <p className="text-slate-500 mt-2">探索强大的AI数学工具集</p>
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <Link
                key={index}
                to={feature.path}
                className="group card-elevated hover:scale-[1.02] transition-all duration-300"
              >
                <div className={`w-14 h-14 rounded-2xl bg-gradient-to-br ${feature.color} flex items-center justify-center mb-5 group-hover:scale-110 group-hover:rotate-3 transition-all duration-300 shadow-medium`}>
                  <Icon className="w-7 h-7 text-white" />
                </div>
                <h3 className="text-xl font-bold text-slate-900 mb-3 group-hover:text-primary-600 transition-colors tracking-tight">
                  {feature.title}
                </h3>
                <p className="text-slate-600 mb-5 text-sm leading-relaxed">
                  {feature.description}
                </p>
                <div className="flex items-center justify-between pt-5 border-t border-slate-100">
                  <span className="text-sm text-slate-500 font-medium">{feature.stats}</span>
                  <span className="text-primary-600 text-sm font-semibold group-hover:translate-x-2 transition-transform flex items-center">
                    进入
                    <svg className="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </span>
                </div>
              </Link>
            );
          })}
        </div>
      </div>

      {/* 快速开始 */}
      <div className="card-elevated bg-gradient-to-br from-slate-50 via-blue-50 to-primary-50 border-2 border-primary-200/50">
        <div className="flex items-center mb-6">
          <div className="w-10 h-10 rounded-xl bg-primary-600 flex items-center justify-center mr-3">
            <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <h2 className="text-2xl font-bold text-slate-900 tracking-tight">快速开始</h2>
        </div>
        <div className="space-y-4">
          <div className="flex items-start p-4 bg-white/60 backdrop-blur-sm rounded-xl border border-slate-200/50 hover:shadow-soft transition-all duration-200">
            <div className="flex-shrink-0 w-10 h-10 rounded-xl bg-gradient-to-br from-primary-600 to-primary-700 text-white flex items-center justify-center font-bold mr-4 shadow-soft">
              1
            </div>
            <div>
              <h3 className="font-bold text-slate-900 mb-1">智能解题</h3>
              <p className="text-sm text-slate-600 leading-relaxed">输入数学题目，让AI帮你自动求解，支持复杂推理和计算</p>
            </div>
          </div>
          <div className="flex items-start p-4 bg-white/60 backdrop-blur-sm rounded-xl border border-slate-200/50 hover:shadow-soft transition-all duration-200">
            <div className="flex-shrink-0 w-10 h-10 rounded-xl bg-gradient-to-br from-primary-600 to-primary-700 text-white flex items-center justify-center font-bold mr-4 shadow-soft">
              2
            </div>
            <div>
              <h3 className="font-bold text-slate-900 mb-1">生成题目</h3>
              <p className="text-sm text-slate-600 leading-relaxed">根据知识点和难度自动生成高质量练习题，支持多种题型</p>
            </div>
          </div>
          <div className="flex items-start p-4 bg-white/60 backdrop-blur-sm rounded-xl border border-slate-200/50 hover:shadow-soft transition-all duration-200">
            <div className="flex-shrink-0 w-10 h-10 rounded-xl bg-gradient-to-br from-primary-600 to-primary-700 text-white flex items-center justify-center font-bold mr-4 shadow-soft">
              3
            </div>
            <div>
              <h3 className="font-bold text-slate-900 mb-1">学习记忆</h3>
              <p className="text-sm text-slate-600 leading-relaxed">智能分析学习历史，精准定位薄弱知识点，个性化推荐</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
