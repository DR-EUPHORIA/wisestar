import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader, Sparkles, Lightbulb, Calculator, FileText, Brain, BarChart3, Bot, Download, MoreVertical, Image as ImageIcon, X, Zap } from 'lucide-react';
import { useChatContext } from '../../contexts/ChatContext';
import Message from './Message';
import { processUserMessage } from '../../utils/intentRouter';
import { exportChatToMarkdown, exportChatToJSON } from '../../utils/exportChat';
import { performOCR, extractMathProblem } from '../../utils/ocr';
import QuickActions from '../QuickActions';

const ChatInterface = () => {
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [showExportMenu, setShowExportMenu] = useState(false);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [isOCRProcessing, setIsOCRProcessing] = useState(false);
  const [showQuickActions, setShowQuickActions] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const imageInputRef = useRef(null);
  const { messages, addMessage, isProcessing, setIsProcessing, conversations, currentConversationId } = useChatContext();
  
  const currentConversation = conversations.find(c => c.id === currentConversationId);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleImageUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      alert('请上传图片文件');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      alert('图片大小不能超过10MB');
      return;
    }

    const reader = new FileReader();
    reader.onloadend = () => {
      setUploadedImage({
        file,
        preview: reader.result
      });
    };
    reader.readAsDataURL(file);

    setIsOCRProcessing(true);
    try {
      const result = await performOCR(file);
      if (result.success) {
        const mathProblem = extractMathProblem(result.text);
        setInput(prev => prev ? `${prev}\n\n识别的题目：${mathProblem}` : `识别的题目：${mathProblem}`);
      } else {
        alert('OCR识别失败：' + result.error);
      }
    } catch (error) {
      console.error('OCR Error:', error);
      alert('图片识别出错，请重试');
    } finally {
      setIsOCRProcessing(false);
    }
  };

  const removeImage = () => {
    setUploadedImage(null);
    if (imageInputRef.current) {
      imageInputRef.current.value = '';
    }
  };

  const handleSend = async () => {
    if (!input.trim() || isProcessing) return;

    const userMessage = input.trim();
    const imageData = uploadedImage;
    const conversationContext = messages.slice(-5).map(m => ({
      role: m.role,
      content: m.content
    }));
    
    setInput('');
    setUploadedImage(null);
    if (imageInputRef.current) {
      imageInputRef.current.value = '';
    }
    setIsProcessing(true);
    setShowQuickActions(false);

    addMessage({
      role: 'user',
      content: userMessage,
      image: imageData?.preview
    });

    try {
      const response = await processUserMessage(userMessage, conversationContext);
      
      addMessage({
        role: 'assistant',
        content: response.content,
        metadata: response.metadata
      });
    } catch (error) {
      addMessage({
        role: 'assistant',
        content: '抱歉，处理您的请求时出现了错误。请稍后再试。',
        metadata: { type: 'error', data: { message: error.message } }
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleQuickAction = (prompt) => {
    setInput(prompt);
    setShowQuickActions(false);
    inputRef.current?.focus();
  };

  // 监听来自Message组件的快捷操作事件
  useEffect(() => {
    const handleQuickActionEvent = (event) => {
      const { prompt } = event.detail;
      setInput(prompt);
      inputRef.current?.focus();
      // 滚动到输入框
      inputRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    };

    window.addEventListener('quickAction', handleQuickActionEvent);
    return () => window.removeEventListener('quickAction', handleQuickActionEvent);
  }, []);

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const quickActions = [
    { icon: Calculator, label: '解题', prompt: '帮我解这道题：' },
    { icon: FileText, label: '出题', prompt: '帮我生成一道关于' },
    { icon: Brain, label: '错题', prompt: '查看我的错题本' },
    { icon: BarChart3, label: '统计', prompt: '查看我的学习统计' }
  ];

  const exampleQuestions = [
    '帮我解这道题：若一个等比数列的前4项和为4，前8项和为68，则该等比数列的公比为？',
    '生成一道高考难度的函数与导数题目',
    '查看我最近的学习统计',
    '推荐一些我薄弱知识点的题目'
  ];

  const handleExport = (format) => {
    const title = currentConversation?.title || '对话记录';
    if (format === 'markdown') {
      const markdown = exportChatToMarkdown(messages, title);
      const blob = new Blob([markdown], { type: 'text/markdown' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${title}_${Date.now()}.md`;
      a.click();
      URL.revokeObjectURL(url);
    } else if (format === 'json') {
      exportChatToJSON(messages, title);
    }
    setShowExportMenu(false);
  };

  return (
    <div className="flex flex-col h-full">
      {messages.length > 0 && (
        <div className="border-b border-slate-200 bg-white px-6 py-3 flex items-center justify-between">
          <h2 className="font-semibold text-slate-900 truncate">
            {currentConversation?.title || '新对话'}
          </h2>
          <div className="relative">
            <button
              onClick={() => setShowExportMenu(!showExportMenu)}
              className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
            >
              <MoreVertical className="w-5 h-5 text-slate-600" />
            </button>
            {showExportMenu && (
              <div className="absolute right-0 mt-2 w-48 bg-white rounded-xl shadow-large border border-slate-200 py-2 z-50">
                <button
                  onClick={() => handleExport('markdown')}
                  className="w-full px-4 py-2 text-left hover:bg-slate-50 flex items-center gap-2 text-sm"
                >
                  <Download className="w-4 h-4" />
                  导出为 Markdown
                </button>
                <button
                  onClick={() => handleExport('json')}
                  className="w-full px-4 py-2 text-left hover:bg-slate-50 flex items-center gap-2 text-sm"
                >
                  <Download className="w-4 h-4" />
                  导出为 JSON
                </button>
              </div>
            )}
          </div>
        </div>
      )}
      
      {messages.length === 0 ? (
        <div className="flex-1 flex items-center justify-center p-6">
          <div className="max-w-3xl w-full space-y-8 animate-fade-in">
            <div className="text-center">
              <div className="inline-flex items-center justify-center w-20 h-20 rounded-3xl bg-gradient-to-br from-primary-600 to-primary-700 mb-6 shadow-large">
                <Sparkles className="w-10 h-10 text-white" />
              </div>
              <h1 className="text-4xl font-bold text-slate-900 mb-3 tracking-tight">
                WiseStar 数学助手
              </h1>
              <p className="text-lg text-slate-600">
                我可以帮你解题、出题、分析学习数据，开始对话吧！
              </p>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {quickActions.map((action, index) => {
                const Icon = action.icon;
                return (
                  <button
                    key={index}
                    onClick={() => setInput(action.prompt)}
                    className="flex flex-col items-center gap-3 p-4 bg-white border-2 border-slate-200 rounded-2xl hover:border-primary-300 hover:shadow-soft transition-all duration-200 group"
                  >
                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary-50 to-primary-100 flex items-center justify-center group-hover:scale-110 transition-transform">
                      <Icon className="w-6 h-6 text-primary-600" />
                    </div>
                    <span className="text-sm font-semibold text-slate-700">{action.label}</span>
                  </button>
                );
              })}
            </div>

            <div className="space-y-3">
              <div className="flex items-center gap-2 text-slate-600">
                <Lightbulb className="w-5 h-5" />
                <span className="text-sm font-semibold">试试这些问题：</span>
              </div>
              <div className="space-y-2">
                {exampleQuestions.map((question, index) => (
                  <button
                    key={index}
                    onClick={() => setInput(question)}
                    className="w-full text-left p-4 bg-slate-50 hover:bg-slate-100 rounded-xl transition-colors text-sm text-slate-700 border border-slate-200 hover:border-slate-300"
                  >
                    {question}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.map((message) => (
            <Message key={message.id} message={message} />
          ))}
          {isProcessing && (
            <div className="flex gap-4 mb-6">
              <div className="flex-shrink-0 w-10 h-10 rounded-xl bg-gradient-to-br from-primary-600 to-primary-700 flex items-center justify-center shadow-soft">
                <Sparkles className="w-6 h-6 text-white" />
              </div>
              <div className="flex-1 max-w-3xl">
                <div className="bg-white border border-slate-200 rounded-2xl px-5 py-4 shadow-soft">
                  <div className="flex items-center gap-2 text-slate-600">
                    <Loader className="w-4 h-4 animate-spin" />
                    <span className="text-sm">正在思考...</span>
                  </div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      )}

      <div className="border-t border-slate-200 bg-white p-4">
        <div className="max-w-4xl mx-auto">
          {uploadedImage && (
            <div className="mb-3 relative inline-block">
              <div className="relative w-32 h-32 rounded-lg overflow-hidden border-2 border-primary-300">
                <img 
                  src={uploadedImage.preview} 
                  alt="Uploaded" 
                  className="w-full h-full object-cover"
                />
                {isOCRProcessing && (
                  <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                    <Loader className="w-6 h-6 text-white animate-spin" />
                  </div>
                )}
              </div>
              <button
                onClick={removeImage}
                className="absolute -top-2 -right-2 w-6 h-6 bg-red-500 text-white rounded-full flex items-center justify-center hover:bg-red-600 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          )}
          
          <div className="flex gap-3 items-end">
            <input
              ref={imageInputRef}
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="hidden"
            />
            <button
              onClick={() => imageInputRef.current?.click()}
              disabled={isProcessing || isOCRProcessing}
              className="flex-shrink-0 w-12 h-12 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-600 flex items-center justify-center transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              title="上传图片并识别"
            >
              <ImageIcon className="w-5 h-5" />
            </button>
            
            <div className="flex-1 relative">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="输入你的问题或上传图片... (Shift+Enter 换行)"
                className="w-full px-4 py-3 pr-12 border-2 border-slate-200 rounded-2xl focus:border-primary-500 focus:ring-4 focus:ring-primary-100 outline-none resize-none transition-all"
                rows={1}
                style={{ minHeight: '52px', maxHeight: '200px' }}
                disabled={isProcessing}
              />
            </div>
            <button
              onClick={handleSend}
              disabled={!input.trim() || isProcessing}
              className="flex-shrink-0 w-12 h-12 rounded-xl bg-gradient-to-br from-primary-600 to-primary-700 text-white flex items-center justify-center hover:shadow-large transition-all disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-none"
            >
              {isProcessing ? (
                <Loader className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </button>
          </div>
          {messages.length > 0 && (
            <div className="mt-3">
              <button
                onClick={() => setShowQuickActions(!showQuickActions)}
                className="flex items-center gap-2 text-sm text-slate-600 hover:text-primary-600 transition-colors mx-auto"
              >
                <Zap className="w-4 h-4" />
                {showQuickActions ? '隐藏快捷操作' : '显示快捷操作'}
              </button>
              {showQuickActions && (
                <div className="mt-3 animate-slide-up">
                  <QuickActions onAction={handleQuickAction} disabled={isProcessing} />
                </div>
              )}
            </div>
          )}
          <div className="mt-2 text-xs text-slate-500 text-center">
            WiseStar 可能会出错，请核对重要信息 · 支持图片上传识别 · 支持LaTeX公式（用$包裹）
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
