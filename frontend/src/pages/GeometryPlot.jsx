import React, { useState } from 'react';
import { Pencil, Play, Download, Trash2, Code, Image as ImageIcon, Loader, Sparkles, Wand2 } from 'lucide-react';
import { executePythonPlot, generatePlotCode } from '../utils/api';

const GeometryPlot = () => {
  const [naturalLanguage, setNaturalLanguage] = useState('');
  const [pythonCode, setPythonCode] = useState(`import matplotlib.pyplot as plt
import numpy as np

# Draw circle
theta = np.linspace(0, 2*np.pi, 100)
x = 2 * np.cos(theta)
y = 2 * np.sin(theta)
plt.plot(x, y, 'b-', label='Circle: x² + y² = 4')

# Draw line
x_line = np.linspace(-3, 3, 100)
y_line = 0.5 * x_line + 1
plt.plot(x_line, y_line, 'r-', label='Line: y = 0.5x + 1')

plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Analytic Geometry')
plt.show()`);
  const [plotImage, setPlotImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [activeTab, setActiveTab] = useState('natural');

  const templates = [
    {
      name: '圆与直线',
      code: `import matplotlib.pyplot as plt
import numpy as np

theta = np.linspace(0, 2*np.pi, 100)
x = 2 * np.cos(theta)
y = 2 * np.sin(theta)
plt.plot(x, y, 'b-', label='Circle: x² + y² = 4')

x_line = np.linspace(-3, 3, 100)
y_line = 0.5 * x_line + 1
plt.plot(x_line, y_line, 'r-', label='Line: y = 0.5x + 1')

plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Circle and Line')
plt.show()`
    },
    {
      name: '椭圆',
      code: `import matplotlib.pyplot as plt
import numpy as np

theta = np.linspace(0, 2*np.pi, 100)
a, b = 4, 2
x = a * np.cos(theta)
y = b * np.sin(theta)
plt.plot(x, y, 'g-', linewidth=2, label=f'Ellipse: x²/{a}² + y²/{b}² = 1')

plt.plot([-np.sqrt(a**2 - b**2), np.sqrt(a**2 - b**2)], [0, 0], 'ro', markersize=8, label='Foci')

plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Ellipse and Foci')
plt.show()`
    },
    {
      name: '双曲线',
      code: `import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(-3, 3, 200)
a, b = 2, 1.5

x1 = a * np.cosh(t)
y1 = b * np.sinh(t)
x2 = -a * np.cosh(t)
y2 = b * np.sinh(t)

plt.plot(x1, y1, 'b-', linewidth=2, label=f'Hyperbola: x²/{a}² - y²/{b}² = 1')
plt.plot(x2, y2, 'b-', linewidth=2)

x_asymptote = np.linspace(-5, 5, 100)
plt.plot(x_asymptote, (b/a)*x_asymptote, 'r--', alpha=0.5, label='Asymptotes')
plt.plot(x_asymptote, -(b/a)*x_asymptote, 'r--', alpha=0.5)

plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Hyperbola and Asymptotes')
plt.show()`
    },
    {
      name: '抛物线',
      code: `import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 100)
y = x**2

plt.plot(x, y, 'purple', linewidth=2, label='Parabola: y = x²')
plt.plot(0, 0, 'ro', markersize=8, label='Vertex')

plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Parabola')
plt.show()`
    }
  ];

  const handleGenerateCode = async () => {
    if (!naturalLanguage.trim()) {
      setError('请输入自然语言描述');
      return;
    }

    setGenerating(true);
    setError(null);

    try {
      const result = await generatePlotCode(naturalLanguage);

      if (result.success) {
        setPythonCode(result.code);
        setActiveTab('code');
        setError(null);
      } else {
        setError(result.error || 'AI代码生成失败');
      }
    } catch (err) {
      setError(err.message || '网络请求失败');
    } finally {
      setGenerating(false);
    }
  };

  const handleExecute = async () => {
    setLoading(true);
    setError(null);

    try {
      const result = await executePythonPlot(pythonCode);

      if (result.success) {
        setPlotImage(result.image);
        setHistory(prev => [{
          code: pythonCode,
          image: result.image,
          description: naturalLanguage || '手动编写',
          timestamp: new Date().toLocaleString()
        }, ...prev.slice(0, 9)]);
      } else {
        setError(result.error || '执行失败');
      }
    } catch (err) {
      setError(err.message || '网络请求失败');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (!plotImage) return;

    const link = document.createElement('a');
    link.href = plotImage;
    link.download = `geometry_plot_${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleClear = () => {
    setPythonCode('');
    setPlotImage(null);
    setError(null);
  };

  const loadTemplate = (template) => {
    setPythonCode(template.code);
    setActiveTab('code');
    setError(null);
  };

  const loadFromHistory = (item) => {
    setPythonCode(item.code);
    setPlotImage(item.image);
    setNaturalLanguage(item.description === '手动编写' ? '' : item.description);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <div className="flex items-center space-x-3 mb-2">
            <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg">
              <Pencil className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900">解析几何画图</h1>
              <p className="text-gray-600 mt-1">使用自然语言或Python代码绘制解析几何图形</p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-white rounded-2xl shadow-lg border border-gray-200 overflow-hidden">
              <div className="bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 px-6 py-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <button
                      onClick={() => setActiveTab('natural')}
                      className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
                        activeTab === 'natural'
                          ? 'bg-white text-purple-600 shadow-md'
                          : 'text-white hover:bg-white/20'
                      }`}
                    >
                      <Sparkles className="w-4 h-4" />
                      <span className="font-medium">自然语言</span>
                    </button>
                    <button
                      onClick={() => setActiveTab('code')}
                      className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
                        activeTab === 'code'
                          ? 'bg-white text-purple-600 shadow-md'
                          : 'text-white hover:bg-white/20'
                      }`}
                    >
                      <Code className="w-4 h-4" />
                      <span className="font-medium">代码编辑</span>
                    </button>
                  </div>
                  <div className="flex items-center space-x-2">
                    {activeTab === 'code' && (
                      <button
                        onClick={handleClear}
                        className="px-4 py-2 bg-white/20 hover:bg-white/30 text-white rounded-lg transition-colors flex items-center space-x-2"
                      >
                        <Trash2 className="w-4 h-4" />
                        <span>清空</span>
                      </button>
                    )}
                    <button
                      onClick={activeTab === 'natural' ? handleGenerateCode : handleExecute}
                      disabled={(activeTab === 'natural' ? generating : loading) || (activeTab === 'natural' ? !naturalLanguage.trim() : !pythonCode.trim())}
                      className="px-4 py-2 bg-white text-purple-600 hover:bg-purple-50 rounded-lg transition-colors flex items-center space-x-2 font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {activeTab === 'natural' ? (
                        generating ? (
                          <>
                            <Loader className="w-4 h-4 animate-spin" />
                            <span>生成中...</span>
                          </>
                        ) : (
                          <>
                            <Wand2 className="w-4 h-4" />
                            <span>生成代码</span>
                          </>
                        )
                      ) : (
                        loading ? (
                          <>
                            <Loader className="w-4 h-4 animate-spin" />
                            <span>执行中...</span>
                          </>
                        ) : (
                          <>
                            <Play className="w-4 h-4" />
                            <span>执行代码</span>
                          </>
                        )
                      )}
                    </button>
                  </div>
                </div>
              </div>

              <div className="p-6">
                {activeTab === 'natural' ? (
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        描述你想绘制的几何图形
                      </label>
                      <textarea
                        value={naturalLanguage}
                        onChange={(e) => setNaturalLanguage(e.target.value)}
                        className="w-full h-64 p-4 text-sm bg-gradient-to-br from-purple-50 to-pink-50 border-2 border-purple-200 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none"
                        placeholder="例如：绘制一个圆心在原点、半径为3的圆，以及一条经过点(1,2)且斜率为2的直线"
                        spellCheck={false}
                      />
                    </div>
                    <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-4 border border-blue-200">
                      <h4 className="font-semibold text-gray-900 mb-2 flex items-center">
                        <Sparkles className="w-4 h-4 mr-2 text-purple-600" />
                        示例描述
                      </h4>
                      <ul className="space-y-2 text-sm text-gray-700">
                        <li className="flex items-start">
                          <span className="text-purple-600 mr-2">•</span>
                          <span>绘制椭圆 x²/16 + y²/9 = 1 及其焦点</span>
                        </li>
                        <li className="flex items-start">
                          <span className="text-purple-600 mr-2">•</span>
                          <span>画出双曲线 x²/4 - y²/9 = 1 和它的渐近线</span>
                        </li>
                        <li className="flex items-start">
                          <span className="text-purple-600 mr-2">•</span>
                          <span>绘制抛物线 y = x² - 2x + 1 并标注顶点</span>
                        </li>
                      </ul>
                    </div>
                  </div>
                ) : (
                  <textarea
                    value={pythonCode}
                    onChange={(e) => setPythonCode(e.target.value)}
                    className="w-full h-96 p-4 font-mono text-sm bg-gray-50 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none"
                    placeholder="在此输入Python代码..."
                    spellCheck={false}
                  />
                )}

                {error && (
                  <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
                    <p className="text-red-700 text-sm font-medium">错误：</p>
                    <pre className="text-red-600 text-xs mt-2 whitespace-pre-wrap font-mono">{error}</pre>
                  </div>
                )}
              </div>
            </div>

            <div className="bg-white rounded-2xl shadow-lg border border-gray-200 overflow-hidden">
              <div className="bg-gradient-to-r from-green-500 to-teal-600 px-6 py-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2 text-white">
                    <ImageIcon className="w-5 h-5" />
                    <h2 className="text-lg font-semibold">图形输出</h2>
                  </div>
                  {plotImage && (
                    <button
                      onClick={handleDownload}
                      className="px-4 py-2 bg-white text-green-600 hover:bg-green-50 rounded-lg transition-colors flex items-center space-x-2 font-medium"
                    >
                      <Download className="w-4 h-4" />
                      <span>下载图片</span>
                    </button>
                  )}
                </div>
              </div>

              <div className="p-6">
                {plotImage ? (
                  <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                    <img
                      src={plotImage}
                      alt="Geometry Plot"
                      className="w-full h-auto rounded-lg shadow-md"
                    />
                  </div>
                ) : (
                  <div className="h-96 flex items-center justify-center bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
                    <div className="text-center">
                      <ImageIcon className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                      <p className="text-gray-500 font-medium">暂无图形输出</p>
                      <p className="text-gray-400 text-sm mt-2">执行代码后将在此显示图形</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-white rounded-2xl shadow-lg border border-gray-200 overflow-hidden">
              <div className="bg-gradient-to-r from-purple-500 to-pink-600 px-6 py-4">
                <h2 className="text-lg font-semibold text-white">代码模板</h2>
              </div>
              <div className="p-4 space-y-2 max-h-96 overflow-y-auto">
                {templates.map((template, index) => (
                  <button
                    key={index}
                    onClick={() => loadTemplate(template)}
                    className="w-full text-left px-4 py-3 bg-gradient-to-r from-purple-50 to-pink-50 hover:from-purple-100 hover:to-pink-100 rounded-lg transition-all border border-purple-200 hover:border-purple-300"
                  >
                    <p className="font-medium text-purple-900">{template.name}</p>
                    <p className="text-xs text-purple-600 mt-1">点击加载模板代码</p>
                  </button>
                ))}
              </div>
            </div>

            {history.length > 0 && (
              <div className="bg-white rounded-2xl shadow-lg border border-gray-200 overflow-hidden">
                <div className="bg-gradient-to-r from-orange-500 to-red-600 px-6 py-4">
                  <h2 className="text-lg font-semibold text-white">历史记录</h2>
                </div>
                <div className="p-4 space-y-3 max-h-96 overflow-y-auto">
                  {history.map((item, index) => (
                    <button
                      key={index}
                      onClick={() => loadFromHistory(item)}
                      className="w-full text-left p-3 bg-gradient-to-r from-orange-50 to-red-50 hover:from-orange-100 hover:to-red-100 rounded-lg transition-all border border-orange-200 hover:border-orange-300"
                    >
                      <div className="flex items-start space-x-3">
                        <img
                          src={item.image}
                          alt="History"
                          className="w-16 h-16 rounded object-cover border border-gray-200"
                        />
                        <div className="flex-1 min-w-0">
                          <p className="text-xs text-orange-600 font-medium">{item.timestamp}</p>
                          <p className="text-xs text-gray-600 mt-1 truncate">{item.description}</p>
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            <div className="bg-gradient-to-br from-blue-50 to-purple-50 rounded-2xl p-6 border border-blue-200">
              <h3 className="font-semibold text-gray-900 mb-3">使用说明</h3>
              <ul className="space-y-2 text-sm text-gray-700">
                <li className="flex items-start">
                  <span className="text-blue-600 mr-2">•</span>
                  <span>使用自然语言描述，AI自动生成代码</span>
                </li>
                <li className="flex items-start">
                  <span className="text-blue-600 mr-2">•</span>
                  <span>支持圆、椭圆、双曲线、抛物线等</span>
                </li>
                <li className="flex items-start">
                  <span className="text-blue-600 mr-2">•</span>
                  <span>可以选择模板快速开始</span>
                </li>
                <li className="flex items-start">
                  <span className="text-blue-600 mr-2">•</span>
                  <span>执行后可下载生成的图片</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default GeometryPlot;