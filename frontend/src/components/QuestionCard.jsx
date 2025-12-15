import React, { useState } from 'react';
import { Star, BookmarkX, Trash2, Tag, StickyNote, ChevronDown, ChevronUp } from 'lucide-react';
import { useFavoriteContext } from '../contexts/FavoriteContext';
import MarkdownRenderer from './MarkdownRenderer';

const QuestionCard = ({ item, type = 'favorite', onRemove }) => {
  const [expanded, setExpanded] = useState(false);
  const [showNoteEditor, setShowNoteEditor] = useState(false);
  const [note, setNote] = useState(item.note || '');
  const [showTagEditor, setShowTagEditor] = useState(false);
  const [selectedTags, setSelectedTags] = useState(item.tags || []);
  
  const { 
    tags, 
    updateFavoriteNote, 
    updateWrongQuestionNote,
    updateFavoriteTags,
    updateWrongQuestionTags,
    markAsMastered,
    retryWrongQuestion
  } = useFavoriteContext();

  const handleSaveNote = () => {
    if (type === 'favorite') {
      updateFavoriteNote(item.id, note);
    } else {
      updateWrongQuestionNote(item.id, note);
    }
    setShowNoteEditor(false);
  };

  const handleSaveTags = () => {
    if (type === 'favorite') {
      updateFavoriteTags(item.id, selectedTags);
    } else {
      updateWrongQuestionTags(item.id, selectedTags);
    }
    setShowTagEditor(false);
  };

  const toggleTag = (tag) => {
    if (selectedTags.includes(tag)) {
      setSelectedTags(selectedTags.filter(t => t !== tag));
    } else {
      setSelectedTags([...selectedTags, tag]);
    }
  };

  return (
    <div className="border border-slate-200 rounded-xl p-5 hover:shadow-medium transition-all duration-200 bg-white">
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            {type === 'favorite' ? (
              <Star className="w-5 h-5 text-yellow-400 fill-yellow-400 flex-shrink-0" />
            ) : (
              <BookmarkX className="w-5 h-5 text-red-500 flex-shrink-0" />
            )}
            <h3 className="font-semibold text-slate-900 line-clamp-2">{item.question}</h3>
          </div>
          
          <div className="flex items-center gap-2 flex-wrap mb-2">
            {item.tags && item.tags.map((tag, i) => (
              <span key={i} className="badge badge-primary text-xs">{tag}</span>
            ))}
            <span className={`badge text-xs ${
              item.difficulty === '简单' ? 'badge-success' :
              item.difficulty === '中等' ? 'badge-warning' :
              'badge-error'
            }`}>
              {item.difficulty}
            </span>
            <span className="badge badge-secondary text-xs">{item.topic}</span>
          </div>

          {type === 'wrong' && (
            <div className="flex items-center gap-3 text-xs text-slate-500 mb-2">
              <span>重做次数: {item.retryCount}</span>
              {item.mastered && (
                <span className="text-green-600 font-semibold">✓ 已掌握</span>
              )}
            </div>
          )}
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowTagEditor(!showTagEditor)}
            className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
            title="管理标签"
          >
            <Tag className="w-4 h-4 text-slate-600" />
          </button>
          <button
            onClick={() => setShowNoteEditor(!showNoteEditor)}
            className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
            title="添加笔记"
          >
            <StickyNote className="w-4 h-4 text-slate-600" />
          </button>
          <button
            onClick={() => setExpanded(!expanded)}
            className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
          >
            {expanded ? (
              <ChevronUp className="w-4 h-4 text-slate-600" />
            ) : (
              <ChevronDown className="w-4 h-4 text-slate-600" />
            )}
          </button>
          <button
            onClick={() => onRemove(item.id)}
            className="p-2 hover:bg-red-100 rounded-lg transition-colors"
            title="删除"
          >
            <Trash2 className="w-4 h-4 text-red-600" />
          </button>
        </div>
      </div>

      {showTagEditor && (
        <div className="mb-3 p-3 bg-slate-50 rounded-lg border border-slate-200 animate-slide-up">
          <p className="text-sm font-semibold text-slate-700 mb-2">选择标签：</p>
          <div className="flex flex-wrap gap-2 mb-3">
            {tags.map((tag, i) => (
              <button
                key={i}
                onClick={() => toggleTag(tag)}
                className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors ${
                  selectedTags.includes(tag)
                    ? 'bg-primary-600 text-white'
                    : 'bg-white border border-slate-300 text-slate-700 hover:bg-slate-100'
                }`}
              >
                {tag}
              </button>
            ))}
          </div>
          <button onClick={handleSaveTags} className="btn-primary text-sm py-1.5">
            保存标签
          </button>
        </div>
      )}

      {showNoteEditor && (
        <div className="mb-3 p-3 bg-slate-50 rounded-lg border border-slate-200 animate-slide-up">
          <p className="text-sm font-semibold text-slate-700 mb-2">添加笔记（支持Markdown）：</p>
          <textarea
            value={note}
            onChange={(e) => setNote(e.target.value)}
            className="w-full p-2 border border-slate-300 rounded-lg text-sm mb-2 resize-none"
            rows={4}
            placeholder="在这里记录你的想法、易错点、解题技巧等..."
          />
          <button onClick={handleSaveNote} className="btn-primary text-sm py-1.5">
            保存笔记
          </button>
        </div>
      )}

      {expanded && (
        <div className="mt-4 pt-4 border-t border-slate-200 space-y-3 animate-slide-up">
          <div>
            <p className="text-sm font-semibold text-slate-700 mb-1">答案：</p>
            <p className="text-slate-800 bg-green-50 p-3 rounded-lg">{item.answer}</p>
          </div>

          {item.note && (
            <div>
              <p className="text-sm font-semibold text-slate-700 mb-1">我的笔记：</p>
              <div className="bg-yellow-50 p-3 rounded-lg border border-yellow-200">
                <MarkdownRenderer content={item.note} />
              </div>
            </div>
          )}

          {type === 'wrong' && !item.mastered && (
            <div className="flex gap-2">
              <button
                onClick={() => retryWrongQuestion(item.id)}
                className="btn-outline text-sm flex-1"
              >
                重做此题
              </button>
              <button
                onClick={() => markAsMastered(item.id)}
                className="btn-primary text-sm flex-1"
              >
                标记为已掌握
              </button>
            </div>
          )}
        </div>
      )}

      <div className="mt-3 text-xs text-slate-500">
        添加时间: {new Date(item.createdAt).toLocaleString('zh-CN')}
      </div>
    </div>
  );
};

export default QuestionCard;
