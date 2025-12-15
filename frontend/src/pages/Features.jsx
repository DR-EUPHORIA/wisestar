import React from 'react';
import { Link } from 'react-router-dom';
import { Calculator, FileText, Brain, BarChart3, Calendar, Sparkles, ArrowRight, MessageSquare, ClipboardList } from 'lucide-react';

const Features = () => {
  const features = [
    {
      title: '智能解题',
      description: '使用RePI系统自动解答数学题目，支持推理和计算步骤展示',
      icon: Calculator,
      path: '/solve',
      color: 'from-blue-500 to-cyan-500',
      features: ['步骤详解', '代码执行', '多种题型']
    },
    {
      title: '题目生成',
      description: 'AI自动生成高质量数学题目，支持批量生成和自定义难度',
      icon: FileText,
      path: '/generate',
      color: 'from-purple-500 to-pink-500',
      features: ['批量出题', '质量评估', '自定义类型']
    },
    {
      title: '智能组卷',
      description: '根据知识点智能生成试卷，支持自定义章节和题型配置',
      icon: ClipboardList,
      path: '/exam-paper',
      color: 'from-violet-500 to-purple-500',
      features: ['知识树选择', '题型配置', '导出试卷']
    },
    {
      title: '学习记忆',
      description: '智能记录学习历史，分析薄弱知识点，提供个性化建议',
      icon: Brain,
      path: '/memory',
      color: 'from-green-500 to-emerald-500',
      features: ['错题分析', '知识点追踪', '学习建议']
    },
    {
      title: '统计分析',
      description: '可视化展示学习数据和进度趋势，多维度分析学习效果',
      icon: BarChart3,
      path: '/statistics',
      color: 'from-orange-500 to-red-500',
      features: ['数据可视化', '趋势分析', '成绩报告']
    },
    {
      title: '每日一题',
      description: '每天推荐一道题目，基于学习情况智能推荐，巩固学习效果',
      icon: Calendar,
      path: '/daily',
      color: 'from-indigo-500 to-purple-500',
      features: ['智能推荐', '连续打卡', '难度适配']
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 via-white to-purple-50 py-12 px-6">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-16">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-primary-600 to-primary-700 mb-6 shadow-lg">
            <Sparkles className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-4xl font-bold text-slate-900 mb-4">功能中心</h1>
          <p className="text-lg text-slate-600 max-w-2xl mx-auto">
            探索WiseStar的强大功能，提升你的数学学习效率
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12 overflow-y-auto max-h-[calc(100vh-400px)]">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <Link
                key={index}
                to={feature.path}
                className="group bg-white rounded-2xl p-8 shadow-soft hover:shadow-large transition-all duration-300 border border-slate-200/60 hover:border-primary-200 hover:-translate-y-1"
              >
                <div className={`w-14 h-14 rounded-xl bg-gradient-to-br ${feature.color} flex items-center justify-center mb-6 group-hover:scale-110 transition-transform shadow-medium`}>
                  <Icon className="w-7 h-7 text-white" />
                </div>
                
                <h3 className="text-xl font-bold text-slate-900 mb-3 group-hover:text-primary-600 transition-colors">
                  {feature.title}
                </h3>
                
                <p className="text-slate-600 mb-6 text-sm leading-relaxed">
                  {feature.description}
                </p>

                <div className="space-y-2 mb-6">
                  {feature.features.map((item, i) => (
                    <div key={i} className="flex items-center text-sm text-slate-500">
                      <div className="w-1.5 h-1.5 rounded-full bg-primary-400 mr-2" />
                      {item}
                    </div>
                  ))}
                </div>

                <div className="flex items-center text-primary-600 font-semibold text-sm group-hover:translate-x-2 transition-transform">
                  进入功能
                  <ArrowRight className="w-4 h-4 ml-2" />
                </div>
              </Link>
            );
          })}
        </div>

        <div className="bg-gradient-to-br from-primary-600 to-primary-700 rounded-2xl p-10 text-white shadow-large">
          <div className="max-w-3xl mx-auto text-center">
            <h2 className="text-2xl font-bold mb-4">还有更多功能等你探索</h2>
            <p className="text-primary-100 mb-6">
              通过对话助手，你可以更自然地使用这些功能，只需告诉我你想做什么
            </p>
            <Link
              to="/"
              className="inline-flex items-center gap-2 px-6 py-3 bg-white text-primary-600 rounded-xl font-semibold hover:shadow-xl transition-all hover:scale-105"
            >
              <MessageSquare className="w-5 h-5" />
              开始对话
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Features;
