import React, { useState, useEffect } from 'react';
import { Brain, Search, Tag, TrendingDown, TrendingUp, Clock, CheckCircle, XCircle, Loader, Star, BookmarkX } from 'lucide-react';
import { getMemoryRecords } from '../utils/api';
import { useFavoriteContext } from '../contexts/FavoriteContext';
import QuestionCard from '../components/QuestionCard';

const LearningMemory = () => {
  const [activeTab, setActiveTab] = useState('history');
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedTag, setSelectedTag] = useState('all');
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    total: 0,
    success_rate: 0,
    weak_points: [],
    mastered_points: []
  });
  const [questions, setQuestions] = useState([]);
  const { favorites, wrongQuestions, removeFromFavorites, removeFromWrongQuestions } = useFavoriteContext();

  const tags = ['all', '函数', '导数', '不等式', '立体几何', '数列', '三角函数'];

  useEffect(() => {
    loadMemoryData();
  }, []);

  const loadMemoryData = async () => {
    setLoading(true);
    try {
      const data = await getMemoryRecords();
      if (data.total !== undefined) {
        setStats({
          total: data.total || 0,
          success_rate: data.success_rate || 0,
          weak_points: data.weak_points || [],
          mastered_points: data.mastered_points || []
        });
      }
      setQuestions(data.records || []);
    } catch (error) {
      console.error('加载记忆数据失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const filteredQuestions = questions.filter(q => {
    const matchSearch = q.question.toLowerCase().includes(searchTerm.toLowerCase());
    const matchTag = selectedTag === 'all' || q.tags.includes(selectedTag);
    return matchSearch && matchTag;
  });

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <Loader className="w-12 h-12 text-primary-600 animate-spin mx-auto mb-4" />
          <p className="text-slate-600">加载学习记忆数据中...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-fade-in">
      {/* 页面标题 */}
      <div>
        <h1 className="text-3xl font-bold text-slate-900 flex items-center tracking-tight">
          <Brain className="w-8 h-8 mr-3 text-primary-600" />
          学习记忆
        </h1>
        <p className="text-slate-600 mt-2">智能记录学习历史，分析薄弱知识点</p>
      </div>

      {/* 统计卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card bg-gradient-to-br from-blue-50 to-cyan-50">
          <div className="text-3xl font-bold text-blue-600 mb-1">{stats.total}</div>
          <div className="text-sm text-slate-600">总题目数</div>
        </div>
        <div className="card bg-gradient-to-br from-green-50 to-emerald-50">
          <div className="text-3xl font-bold text-green-600 mb-1">{(stats.success_rate * 100).toFixed(0)}%</div>
          <div className="text-sm text-slate-600">成功率</div>
        </div>
        <div className="card bg-gradient-to-br from-red-50 to-orange-50">
          <div className="text-3xl font-bold text-red-600 mb-1">{stats.weak_points.length}</div>
          <div className="text-sm text-slate-600">薄弱知识点</div>
        </div>
        <div className="card bg-gradient-to-br from-purple-50 to-pink-50">
          <div className="text-3xl font-bold text-purple-600 mb-1">{stats.mastered_points.length}</div>
          <div className="text-sm text-slate-600">已掌握知识点</div>
        </div>
      </div>

      {/* 知识点分析 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="card-elevated">
          <h2 className="text-lg font-bold text-slate-900 mb-4 flex items-center tracking-tight">
            <TrendingDown className="w-5 h-5 text-red-500 mr-2" />
            薄弱知识点
          </h2>
          <div className="space-y-3">
            {stats.weak_points.map((point, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-red-50 rounded-lg border border-red-100">
                <span className="font-medium text-slate-900">{point}</span>
                <span className="badge badge-error">需加强</span>
              </div>
            ))}
          </div>
        </div>

        <div className="card-elevated">
          <h2 className="text-lg font-bold text-slate-900 mb-4 flex items-center tracking-tight">
            <TrendingUp className="w-5 h-5 text-green-500 mr-2" />
            已掌握知识点
          </h2>
          <div className="space-y-3">
            {stats.mastered_points.map((point, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-green-50 rounded-lg border border-green-100">
                <span className="font-medium text-slate-900">{point}</span>
                <span className="badge badge-success">已掌握</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* 标签页切换 */}
      <div className="card-elevated">
        <div className="flex gap-2 mb-6 border-b border-slate-200">
          <button
            onClick={() => setActiveTab('history')}
            className={`px-4 py-2 font-semibold transition-colors relative ${
              activeTab === 'history'
                ? 'text-primary-600'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            学习历史
            {activeTab === 'history' && (
              <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary-600" />
            )}
          </button>
          <button
            onClick={() => setActiveTab('favorites')}
            className={`px-4 py-2 font-semibold transition-colors relative flex items-center gap-2 ${
              activeTab === 'favorites'
                ? 'text-primary-600'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Star className="w-4 h-4" />
            收藏夹 ({favorites.length})
            {activeTab === 'favorites' && (
              <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary-600" />
            )}
          </button>
          <button
            onClick={() => setActiveTab('wrong')}
            className={`px-4 py-2 font-semibold transition-colors relative flex items-center gap-2 ${
              activeTab === 'wrong'
                ? 'text-primary-600'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <BookmarkX className="w-4 h-4" />
            错题本 ({wrongQuestions.length})
            {activeTab === 'wrong' && (
              <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary-600" />
            )}
          </button>
        </div>

        {activeTab === 'history' && (
        <div className="flex flex-col md:flex-row gap-4 mb-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-slate-400" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="搜索题目..."
              className="input-field pl-10"
            />
          </div>
        </div>

        <div className="flex flex-wrap gap-2 mb-4">
          {tags.map(tag => (
            <button
              key={tag}
              onClick={() => setSelectedTag(tag)}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                selectedTag === tag
                  ? 'bg-primary-600 text-white shadow-soft'
                  : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
              }`}
            >
              {tag === 'all' ? '全部' : tag}
            </button>
          ))}
        </div>

        {/* 题目列表 */}
        <div className="space-y-3">
          {filteredQuestions.map(q => (
            <div key={q.id} className="border border-slate-200 rounded-xl p-4 hover:shadow-medium transition-all duration-200 bg-white">
              <div className="flex items-start justify-between mb-3">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    {q.success ? (
                      <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0" />
                    ) : (
                      <XCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
                    )}
                    <h3 className="font-medium text-slate-900">{q.question}</h3>
                  </div>
                  <p className="text-sm text-slate-600 ml-7">{q.answer}</p>
                </div>
              </div>

              <div className="flex items-center justify-between ml-7">
                <div className="flex items-center gap-2 flex-wrap">
                  {q.tags.map((tag, i) => (
                    <span key={i} className="badge badge-primary text-xs">{tag}</span>
                  ))}
                  <span className={`badge text-xs ${
                    q.difficulty === '简单' ? 'badge-success' :
                    q.difficulty === '中等' ? 'badge-warning' :
                    'badge-error'
                  }`}>
                    {q.difficulty}
                  </span>
                </div>
                <div className="flex items-center gap-4 text-xs text-slate-500">
                  <span className="flex items-center">
                    <Clock className="w-3 h-3 mr-1" />
                    {q.timestamp}
                  </span>
                  <span>{q.steps} 步</span>
                </div>
              </div>
            </div>
          ))}
        </div>

        {filteredQuestions.length === 0 && (
          <div className="text-center py-12 text-slate-500">
            <Brain className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p>暂无匹配的记录</p>
          </div>
        )}
        )}

        {activeTab === 'favorites' && (
          <div>
            <div className="mb-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-slate-400" />
                <input
                  type="text"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  placeholder="搜索收藏的题目..."
                  className="input-field pl-10"
                />
              </div>
            </div>
            <div className="space-y-4">
              {favorites
                .filter(item => item.question.toLowerCase().includes(searchTerm.toLowerCase()))
                .map(item => (
                  <QuestionCard
                    key={item.id}
                    item={item}
                    type="favorite"
                    onRemove={removeFromFavorites}
                  />
                ))}
              {favorites.length === 0 && (
                <div className="text-center py-12 text-slate-500">
                  <Star className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p>还没有收藏任何题目</p>
                  <p className="text-sm mt-2">在解题页面点击星标按钮收藏题目</p>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'wrong' && (
          <div>
            <div className="mb-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-slate-400" />
                <input
                  type="text"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  placeholder="搜索错题..."
                  className="input-field pl-10"
                />
              </div>
            </div>
            <div className="space-y-4">
              {wrongQuestions
                .filter(item => item.question.toLowerCase().includes(searchTerm.toLowerCase()))
                .map(item => (
                  <QuestionCard
                    key={item.id}
                    item={item}
                    type="wrong"
                    onRemove={removeFromWrongQuestions}
                  />
                ))}
              {wrongQuestions.length === 0 && (
                <div className="text-center py-12 text-slate-500">
                  <BookmarkX className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p>还没有错题记录</p>
                  <p className="text-sm mt-2">在解题页面标记错题以便后续复习</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default LearningMemory;
