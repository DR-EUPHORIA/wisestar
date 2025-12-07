import React, { useMemo, useState } from 'react';
import { FileText, ChevronDown, ChevronRight, Plus, Minus, Download, Wand2, Loader, AlertCircle } from 'lucide-react';
import { knowledgeTreePresets } from '../data/knowledgeTree';
import { generateQuestion } from '../utils/api';

const ExamPaper = () => {
  const [selectedVersion, setSelectedVersion] = useState('人教A版');
  const [expandedNodes, setExpandedNodes] = useState(new Set(['required-1']));
  const [selectedChapters, setSelectedChapters] = useState(new Set());
  
  const [examConfig, setExamConfig] = useState({
    scenario: 'practice',
    difficulty: 'medium',
    region: 'all'
  });

  const [questionTypes, setQuestionTypes] = useState({
    single: 0,
    multiple: 0,
    fillBlank: 0,
    answer: 0,
    judge: 0
  });

  const [generating, setGenerating] = useState(false);
  const [generatedPaper, setGeneratedPaper] = useState(null);
  const [error, setError] = useState('');

  const currentTree = knowledgeTreePresets[selectedVersion];

  const selectedChapterNames = useMemo(() => {
    const names = [];
    const walk = (nodes = []) => {
      nodes.forEach((node) => {
        if (selectedChapters.has(node.id)) {
          names.push(node.name);
        }
        if (node.children) walk(node.children);
      });
    };
    walk(currentTree?.children || []);
    return names;
  }, [currentTree, selectedChapters]);

  const toggleNode = (nodeId) => {
    setExpandedNodes(prev => {
      const newSet = new Set(prev);
      if (newSet.has(nodeId)) {
        newSet.delete(nodeId);
      } else {
        newSet.add(nodeId);
      }
      return newSet;
    });
  };

  const toggleChapter = (chapterId) => {
    setSelectedChapters(prev => {
      const newSet = new Set(prev);
      if (newSet.has(chapterId)) {
        newSet.delete(chapterId);
      } else {
        newSet.add(chapterId);
      }
      return newSet;
    });
  };

  const updateQuestionType = (type, delta) => {
    setQuestionTypes(prev => ({
      ...prev,
      [type]: Math.max(0, prev[type] + delta)
    }));
  };

  const getTotalQuestions = () => {
    return Object.values(questionTypes).reduce((sum, count) => sum + count, 0);
  };

  const handleGenerate = async () => {
    if (selectedChapters.size === 0) {
      alert('请至少选择一个章节');
      return;
    }
    if (getTotalQuestions() === 0) {
      alert('请至少设置一种题型的数量');
      return;
    }

    setGenerating(true);
    setError('');

    const basePayload = {
      difficulty_level: examConfig.difficulty,
      problem_type: '综合',
      topic_keywords: selectedChapterNames,
      requirements: `${selectedVersion} ${examConfig.scenario} ${examConfig.region}`
    };

    const generationTasks = [];
    Object.entries(questionTypes).forEach(([type, count]) => {
      for (let i = 0; i < count; i++) {
        generationTasks.push({ type, call: generateQuestion(basePayload) });
      }
    });

    try {
      const responses = await Promise.all(generationTasks.map(task => task.call));
      let questionNumber = 1;
      const questions = responses.map((res, idx) => ({
        type: generationTasks[idx].type,
        number: questionNumber++,
        content: res.problem || res.question || '生成结果为空',
        answer: res.validation?.answer || res.answer || '',
        score: 5,
        options: undefined
      }));

      setGeneratedPaper({
        title: '智能组卷',
        subtitle: `${selectedVersion} - ${examConfig.scenario === 'practice' ? '课时练习' : examConfig.scenario === 'test' ? '阶段测试' : '高考备考'}`,
        totalScore: questions.length * 5,
        duration: 120,
        questions
      });
    } catch (err) {
      setError(err.message || '组卷失败，请稍后重试');
    } finally {
      setGenerating(false);
    }
  };

  const exportPaper = (withAnswers = false) => {
    if (!generatedPaper) return;

    let markdown = `# ${generatedPaper.title}\n\n`;
    markdown += `**${generatedPaper.subtitle}**\n\n`;
    markdown += `**总分：${generatedPaper.totalScore}分　　考试时间：${generatedPaper.duration}分钟**\n\n`;
    markdown += `---\n\n`;

    const typeNames = {
      single: '一、单项选择题',
      multiple: '二、多项选择题',
      fillBlank: '三、填空题',
      answer: '四、解答题',
      judge: '五、判断题'
    };

    let currentType = null;
    generatedPaper.questions.forEach(q => {
      if (q.type !== currentType) {
        currentType = q.type;
        markdown += `## ${typeNames[q.type]}\n\n`;
      }

      markdown += `**${q.number}. ** ${q.content}\n\n`;
      
      if (q.options) {
        q.options.forEach(opt => {
          markdown += `${opt}\n\n`;
        });
      }

      if (withAnswers) {
        markdown += `**答案：** ${q.answer}\n\n`;
      }

      markdown += `---\n\n`;
    });

    const blob = new Blob([markdown], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${generatedPaper.title}_${withAnswers ? '含答案' : '学生版'}_${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const renderTreeNode = (node, level = 0) => {
    const isExpanded = expandedNodes.has(node.id);
    const isSelected = selectedChapters.has(node.id);
    const hasChildren = node.children && node.children.length > 0;

    return (
      <div key={node.id}>
        <div
          className={`flex items-center gap-2 py-2 px-3 rounded-lg cursor-pointer transition-colors ${
            isSelected ? 'bg-primary-50 text-primary-700' : 'hover:bg-slate-50'
          }`}
          style={{ paddingLeft: `${level * 20 + 12}px` }}
        >
          {hasChildren && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                toggleNode(node.id);
              }}
              className="p-0.5 hover:bg-slate-200 rounded"
            >
              {isExpanded ? (
                <ChevronDown className="w-4 h-4 text-slate-500" />
              ) : (
                <ChevronRight className="w-4 h-4 text-slate-500" />
              )}
            </button>
          )}
          
          {!hasChildren && (
            <input
              type="checkbox"
              checked={isSelected}
              onChange={() => toggleChapter(node.id)}
              className="w-4 h-4 text-primary-600 rounded border-slate-300 focus:ring-primary-500"
            />
          )}

          <span
            className={`text-sm ${hasChildren ? 'font-semibold text-slate-900' : 'text-slate-700'}`}
            onClick={() => !hasChildren && toggleChapter(node.id)}
          >
            {node.name}
          </span>
        </div>

        {hasChildren && isExpanded && (
          <div>
            {node.children.map(child => renderTreeNode(child, level + 1))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 via-white to-purple-50 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-slate-900 flex items-center">
            <FileText className="w-8 h-8 mr-3 text-primary-600" />
            智能组卷
          </h1>
          <p className="text-slate-600 mt-2">根据知识点智能生成高质量试卷</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1">
            <div className="card-elevated sticky top-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-bold text-slate-900">章节</h2>
                <select
                  value={selectedVersion}
                  onChange={(e) => setSelectedVersion(e.target.value)}
                  className="text-sm border border-slate-300 rounded-lg px-3 py-1.5 focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none"
                >
                  {Object.keys(knowledgeTreePresets).map(version => (
                    <option key={version} value={version}>{version}</option>
                  ))}
                </select>
              </div>

              <div className="max-h-[600px] overflow-y-auto scrollbar-thin">
                {currentTree.children.map(node => renderTreeNode(node))}
              </div>

              <div className="mt-4 pt-4 border-t border-slate-200">
                <p className="text-sm text-slate-600">
                  已选择 <span className="font-semibold text-primary-600">{selectedChapters.size}</span> 个章节
                </p>
              </div>
            </div>
          </div>

          <div className="lg:col-span-2 space-y-6">
            <div className="card-elevated">
              <div className="flex items-center gap-2 mb-6">
                <div className="w-8 h-8 rounded-lg bg-primary-600 text-white flex items-center justify-center font-bold">
                  01
                </div>
                <h2 className="text-lg font-bold text-slate-900">选择章节</h2>
                <span className="text-sm text-slate-500">（从左侧勾选章节）</span>
              </div>
            </div>

            <div className="card-elevated">
              <div className="flex items-center gap-2 mb-6">
                <div className="w-8 h-8 rounded-lg bg-primary-600 text-white flex items-center justify-center font-bold">
                  02
                </div>
                <h2 className="text-lg font-bold text-slate-900">组卷设置</h2>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-semibold text-slate-700 mb-2">出题场景</label>
                  <div className="flex gap-4">
                    {[
                      { value: 'practice', label: '课时练习' },
                      { value: 'test', label: '阶段测试' },
                      { value: 'gaokao', label: '高考备考' }
                    ].map(option => (
                      <label key={option.value} className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="radio"
                          name="scenario"
                          value={option.value}
                          checked={examConfig.scenario === option.value}
                          onChange={(e) => setExamConfig({...examConfig, scenario: e.target.value})}
                          className="w-4 h-4 text-primary-600 focus:ring-primary-500"
                        />
                        <span className="text-sm text-slate-700">{option.label}</span>
                      </label>
                    ))}
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-semibold text-slate-700 mb-2">题目难度</label>
                    <select
                      value={examConfig.difficulty}
                      onChange={(e) => setExamConfig({...examConfig, difficulty: e.target.value})}
                      className="input-field"
                    >
                      <option value="easy">简单</option>
                      <option value="medium">适中</option>
                      <option value="hard">困难</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-semibold text-slate-700 mb-2">优先地区</label>
                    <select
                      value={examConfig.region}
                      onChange={(e) => setExamConfig({...examConfig, region: e.target.value})}
                      className="input-field"
                    >
                      <option value="all">全部</option>
                      <option value="beijing">北京</option>
                      <option value="shanghai">上海</option>
                      <option value="guangdong">广东</option>
                    </select>
                  </div>
                </div>
              </div>
            </div>

            <div className="card-elevated">
              <div className="flex items-center gap-2 mb-6">
                <div className="w-8 h-8 rounded-lg bg-primary-600 text-white flex items-center justify-center font-bold">
                  03
                </div>
                <h2 className="text-lg font-bold text-slate-900">试题设置</h2>
                <span className="text-sm text-slate-500">（模板选题）</span>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {[
                  { key: 'single', label: '单选题' },
                  { key: 'multiple', label: '多选题' },
                  { key: 'fillBlank', label: '填空题' },
                  { key: 'answer', label: '解答题' },
                  { key: 'judge', label: '判断题' }
                ].map(type => (
                  <div key={type.key} className="flex items-center justify-between p-4 bg-slate-50 rounded-xl border border-slate-200">
                    <span className="text-sm font-medium text-slate-700">{type.label}</span>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => updateQuestionType(type.key, -1)}
                        className="w-7 h-7 rounded-lg bg-white border border-slate-300 hover:bg-slate-100 flex items-center justify-center transition-colors"
                      >
                        <Minus className="w-4 h-4 text-slate-600" />
                      </button>
                      <span className="w-8 text-center font-semibold text-slate-900">
                        {questionTypes[type.key]}
                      </span>
                      <button
                        onClick={() => updateQuestionType(type.key, 1)}
                        className="w-7 h-7 rounded-lg bg-white border border-slate-300 hover:bg-slate-100 flex items-center justify-center transition-colors"
                      >
                        <Plus className="w-4 h-4 text-slate-600" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-6 p-4 bg-primary-50 rounded-xl border border-primary-200">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-semibold text-slate-700">总题数</span>
                  <span className="text-2xl font-bold text-primary-600">{getTotalQuestions()}</span>
                </div>
              </div>
            </div>

            <button
              onClick={handleGenerate}
              disabled={generating || selectedChapters.size === 0 || getTotalQuestions() === 0}
              className="btn-primary w-full py-4 text-lg disabled:opacity-50 disabled:cursor-not-allowed"
            >
            {generating ? (
              <>
                <Loader className="w-6 h-6 mr-2 animate-spin inline" />
                生成中...
              </>
              ) : (
                <>
                  <Wand2 className="w-6 h-6 mr-2 inline" />
                  生成试卷
              </>
            )}
          </button>

          {error && (
            <div className="mt-3 flex items-center text-sm text-red-600 bg-red-50 border border-red-100 rounded-lg px-3 py-2">
              <AlertCircle className="w-4 h-4 mr-2" />
              {error}
            </div>
          )}

          {generatedPaper && (
              <div className="card-elevated animate-slide-up">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-xl font-bold text-slate-900">生成结果</h2>
                  <div className="flex gap-2">
                    <button
                      onClick={() => exportPaper(false)}
                      className="btn-secondary flex items-center text-sm"
                    >
                      <Download className="w-4 h-4 mr-1" />
                      学生版
                    </button>
                    <button
                      onClick={() => exportPaper(true)}
                      className="btn-secondary flex items-center text-sm"
                    >
                      <Download className="w-4 h-4 mr-1" />
                      含答案版
                    </button>
                  </div>
                </div>

                <div className="bg-gradient-to-br from-primary-50 to-purple-50 rounded-xl p-6 border border-primary-200">
                  <h3 className="text-2xl font-bold text-slate-900 mb-2">{generatedPaper.title}</h3>
                  <p className="text-slate-600 mb-4">{generatedPaper.subtitle}</p>
                  <div className="flex gap-6 text-sm text-slate-700">
                    <span>总分：<span className="font-semibold">{generatedPaper.totalScore}分</span></span>
                    <span>考试时间：<span className="font-semibold">{generatedPaper.duration}分钟</span></span>
                    <span>题目数：<span className="font-semibold">{generatedPaper.questions.length}道</span></span>
                  </div>
                </div>

                <div className="mt-6 space-y-4 max-h-96 overflow-y-auto scrollbar-thin">
                  {generatedPaper.questions.slice(0, 3).map(q => (
                    <div key={q.number} className="p-4 bg-slate-50 rounded-lg border border-slate-200">
                      <p className="font-medium text-slate-900 mb-2">
                        {q.number}. {q.content}
                      </p>
                      {q.options && (
                        <div className="space-y-1 text-sm text-slate-700">
                          {q.options.map((opt, i) => (
                            <p key={i}>{opt}</p>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                  {generatedPaper.questions.length > 3 && (
                    <p className="text-center text-sm text-slate-500">
                      还有 {generatedPaper.questions.length - 3} 道题目，请导出查看完整试卷
                    </p>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ExamPaper;
