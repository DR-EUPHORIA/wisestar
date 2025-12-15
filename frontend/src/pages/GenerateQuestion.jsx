import React, { useState } from 'react';
import { FileText, Wand2, Loader, Download, Copy, CheckCircle, Plus, Trash2, Layers, AlertCircle } from 'lucide-react';
import { generateQuestion } from '../utils/api';

const GenerateQuestion = () => {
  const [mode, setMode] = useState('single');
  const [generationMode, setGenerationMode] = useState('simple');
  const [config, setConfig] = useState({
    task_scenario: '',
    problem_type: '',
    difficulty_level: '',
    topic_keywords: '',
    requirements: '',
    batch_count: 5
  });
  const [generating, setGenerating] = useState(false);
  const [results, setResults] = useState([]);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState('');

  const problemTypesSuggestions = [
    '函数与导数', '数列', '不等式', '三角函数', '立体几何', 
    '解析几何', '概率统计', '向量', '复数', '数论'
  ];

  const difficultyLevelsSuggestions = [
    '简单', '中等', '困难', '高考压轴题', '竞赛级'
  ];

  const handleGenerate = async () => {
    if (!config.task_scenario.trim()) return;

    setGenerating(true);
    setResults([]);
    setError('');

    const count = mode === 'batch' ? config.batch_count : 1;

    const payload = {
      difficulty_level: config.difficulty_level || '中等',
      problem_type: config.problem_type || '综合',
      topic_keywords: config.topic_keywords
        ? config.topic_keywords.split(',').map(k => k.trim()).filter(Boolean)
        : [],
      requirements: `${config.task_scenario}${config.requirements ? `\n${config.requirements}` : ''}`
    };

    try {
      const calls = Array.from({ length: count }, () => generateQuestion(payload));
      const responses = await Promise.all(calls);
      const newResults = responses.map((res, i) => {
        return {
          id: Date.now() + i,
          success: res.success !== false,
          problem: res.problem || res.question || '生成结果为空',
          latex: res.latex || '',
          evaluation: res.evaluation || {
            overall_score: 0,
            originality_score: 0,
            solvability_score: 0,
            complexity_score: 0,
            knowledge_coverage_score: 0,
            educational_value_score: 0
          },
          validation: res.validation || { success: true, answer: res.answer || '' },
          iterations: res.iterations || 1
        };
      });
      setResults(newResults);
    } catch (err) {
      setError(err.message || '生成失败，请稍后重试');
    } finally {
      setGenerating(false);
    }
  };

  const handleCopy = (text) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadAll = () => {
    const allContent = results.map((r, i) => {
      return `题目 ${i + 1}:\n\n${r.problem}\n\n${'='.repeat(50)}\n\n`;
    }).join('');

    const blob = new Blob([allContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = mode === 'batch' ? 'questions_batch.txt' : 'question.txt';
    a.click();
  };

  const handleDownloadLatex = (latex, index) => {
    const blob = new Blob([latex], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `question_${index + 1}.tex`;
    a.click();
  };

  const removeResult = (id) => {
    setResults(results.filter(r => r.id !== id));
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="text-3xl font-bold text-slate-900 flex items-center tracking-tight">
          <FileText className="w-8 h-8 mr-3 text-primary-600" />
          题目生成
        </h1>
        <p className="text-slate-600 mt-2">AI自动生成高质量数学题目，支持批量生成</p>
      </div>

      <div className="card-elevated">
        <div className="flex gap-3 mb-6">
          <button
            onClick={() => setMode('single')}
            className={`flex-1 py-3 px-4 rounded-xl font-semibold transition-all ${
              mode === 'single'
                ? 'bg-primary-600 text-white shadow-medium'
                : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
            }`}
          >
            <FileText className="w-5 h-5 inline mr-2" />
            单题生成
          </button>
          <button
            onClick={() => setMode('batch')}
            className={`flex-1 py-3 px-4 rounded-xl font-semibold transition-all ${
              mode === 'batch'
                ? 'bg-primary-600 text-white shadow-medium'
                : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
            }`}
          >
            <Layers className="w-5 h-5 inline mr-2" />
            批量出题
          </button>
        </div>

        <div className="mb-6 p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl border border-blue-200">
          <h3 className="text-sm font-semibold text-slate-900 mb-3">生成模式</h3>
          <div className="flex gap-3">
            <button
              onClick={() => setGenerationMode('simple')}
              className={`flex-1 py-2 px-3 rounded-lg text-sm font-medium transition-all ${
                generationMode === 'simple'
                  ? 'bg-green-600 text-white shadow-soft'
                  : 'bg-white text-slate-700 hover:bg-slate-50 border border-slate-200'
              }`}
            >
              <Wand2 className="w-4 h-4 inline mr-1" />
              简易模式
            </button>
            <button
              onClick={() => setGenerationMode('complex')}
              className={`flex-1 py-2 px-3 rounded-lg text-sm font-medium transition-all ${
                generationMode === 'complex'
                  ? 'bg-purple-600 text-white shadow-soft'
                  : 'bg-white text-slate-700 hover:bg-slate-50 border border-slate-200'
              }`}
            >
              <Layers className="w-4 h-4 inline mr-1" />
              复杂模式
            </button>
          </div>
          <p className="text-xs text-slate-600 mt-2">
            {generationMode === 'simple' 
              ? '✨ 简易模式：快速生成，一次性输出结果，适合快速出题' 
              : '🔄 复杂模式：多轮反馈优化，质量评估，适合高质量题目生成'}
          </p>
        </div>

        <h2 className="text-lg font-bold text-slate-900 mb-4 tracking-tight">生成配置</h2>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-semibold text-slate-700 mb-2">
              任务描述 *
            </label>
            <textarea
              value={config.task_scenario}
              onChange={(e) => setConfig({...config, task_scenario: e.target.value})}
              placeholder="例如：为准备高考的学生设计一道函数与导数的压轴题..."
              className="textarea-field h-24"
              disabled={generating}
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-semibold text-slate-700 mb-2">
                题目类型（可自定义）
              </label>
              <input
                type="text"
                list="problem-types"
                value={config.problem_type}
                onChange={(e) => setConfig({...config, problem_type: e.target.value})}
                placeholder="选择或输入题目类型"
                className="input-field"
                disabled={generating}
              />
              <datalist id="problem-types">
                {problemTypesSuggestions.map(type => (
                  <option key={type} value={type} />
                ))}
              </datalist>
              <p className="text-xs text-slate-500 mt-1">参考：{problemTypesSuggestions.slice(0, 3).join('、')}等</p>
            </div>

            <div>
              <label className="block text-sm font-semibold text-slate-700 mb-2">
                难度级别（可自定义）
              </label>
              <input
                type="text"
                list="difficulty-levels"
                value={config.difficulty_level}
                onChange={(e) => setConfig({...config, difficulty_level: e.target.value})}
                placeholder="选择或输入难度级别"
                className="input-field"
                disabled={generating}
              />
              <datalist id="difficulty-levels">
                {difficultyLevelsSuggestions.map(level => (
                  <option key={level} value={level} />
                ))}
              </datalist>
              <p className="text-xs text-slate-500 mt-1">参考：{difficultyLevelsSuggestions.slice(0, 3).join('、')}等</p>
            </div>
          </div>

          {mode === 'batch' && (
            <div>
              <label className="block text-sm font-semibold text-slate-700 mb-2">
                生成数量
              </label>
              <input
                type="number"
                min="1"
                max="20"
                value={config.batch_count}
                onChange={(e) => setConfig({...config, batch_count: parseInt(e.target.value) || 1})}
                className="input-field"
                disabled={generating}
              />
              <p className="text-xs text-slate-500 mt-1">建议：5-10道题为一组</p>
            </div>
          )}

          <div>
            <label className="block text-sm font-semibold text-slate-700 mb-2">
              关键词（用逗号分隔）
            </label>
            <input
              type="text"
              value={config.topic_keywords}
              onChange={(e) => setConfig({...config, topic_keywords: e.target.value})}
              placeholder="例如：导数, 单调性, 极值"
              className="input-field"
              disabled={generating}
            />
          </div>

          <div>
            <label className="block text-sm font-semibold text-slate-700 mb-2">
              具体要求
            </label>
            <textarea
              value={config.requirements}
              onChange={(e) => setConfig({...config, requirements: e.target.value})}
              placeholder="例如：需要包含参数分类讨论..."
              className="textarea-field h-20"
              disabled={generating}
            />
          </div>
        </div>

        <div className="flex gap-3 mt-6">
          <button
            onClick={handleGenerate}
            disabled={!config.task_scenario.trim() || generating}
            className="btn-primary flex-1 md:flex-none flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {generating ? (
              <>
                <Loader className="w-5 h-5 mr-2 animate-spin" />
                生成中...
              </>
            ) : (
              <>
                <Wand2 className="w-5 h-5 mr-2" />
                {mode === 'batch' ? `生成 ${config.batch_count} 道题` : '开始生成'}
              </>
            )}
          </button>
          
          {results.length > 0 && (
            <button
              onClick={handleDownloadAll}
              className="btn-secondary flex items-center"
            >
              <Download className="w-4 h-4 mr-2" />
              导出全部
            </button>
          )}
        </div>

        {error && (
          <div className="mt-3 flex items-center text-sm text-red-600 bg-red-50 border border-red-100 rounded-lg px-3 py-2">
            <AlertCircle className="w-4 h-4 mr-2" />
            {error}
          </div>
        )}
      </div>

      {results.length > 0 && (
        <div className="space-y-4 animate-slide-up">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-bold text-slate-900">
              生成结果 ({results.length} 道题)
            </h2>
          </div>

          {results.map((result, index) => (
            <div key={result.id} className="card-elevated">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold text-slate-900 flex items-center">
                  <CheckCircle className="w-5 h-5 text-green-500 mr-2" />
                  题目 {index + 1}
                </h3>
                <div className="flex gap-2">
                  <button 
                    onClick={() => handleCopy(result.problem)} 
                    className="btn-secondary flex items-center text-sm"
                  >
                    {copied ? <CheckCircle className="w-4 h-4 mr-1" /> : <Copy className="w-4 h-4 mr-1" />}
                    {copied ? '已复制' : '复制'}
                  </button>
                  <button 
                    onClick={() => handleDownloadLatex(result.latex, index)} 
                    className="btn-secondary flex items-center text-sm"
                  >
                    <Download className="w-4 h-4 mr-1" />
                    LaTeX
                  </button>
                  {mode === 'batch' && (
                    <button 
                      onClick={() => removeResult(result.id)} 
                      className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  )}
                </div>
              </div>

              <div className="bg-gradient-to-br from-primary-50 to-purple-50 rounded-xl p-6 mb-4 border border-primary-100">
                <div className="whitespace-pre-wrap text-slate-800 leading-relaxed">
                  {result.problem}
                </div>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-3">
                <div className="text-center p-3 bg-slate-50 rounded-lg">
                  <div className="text-2xl font-bold text-primary-600">
                    {result.evaluation.overall_score.toFixed(1)}
                  </div>
                  <div className="text-xs text-slate-600">综合评分</div>
                </div>
                <div className="text-center p-3 bg-slate-50 rounded-lg">
                  <div className="text-2xl font-bold text-purple-600">
                    {result.evaluation.originality_score.toFixed(1)}
                  </div>
                  <div className="text-xs text-slate-600">原创性</div>
                </div>
                <div className="text-center p-3 bg-slate-50 rounded-lg">
                  <div className="text-2xl font-bold text-green-600">
                    {result.evaluation.educational_value_score.toFixed(1)}
                  </div>
                  <div className="text-xs text-slate-600">教学价值</div>
                </div>
              </div>

              <div className="flex items-center gap-2 text-sm">
                {generationMode === 'complex' && (
                  <>
                    <span className="badge badge-primary">迭代 {result.iterations} 次</span>
                    <span className="badge badge-success">REPI验证通过</span>
                  </>
                )}
                {generationMode === 'simple' && (
                  <span className="badge badge-info">简易模式生成</span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default GenerateQuestion;
