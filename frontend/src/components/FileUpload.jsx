import React, { useRef, useState } from 'react';
import { Upload, File, X, Image, FileText, Loader, Camera } from 'lucide-react';
import { performOCR, extractMathProblem } from '../utils/ocr';

const FileUpload = ({ onFileSelect, accept = '.md,.png,.jpg,.jpeg,.pdf', maxSize = 10 }) => {
  const fileInputRef = useRef(null);
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);

  const handleFileChange = async (e) => {
    const selectedFiles = Array.from(e.target.files);
    
    const validFiles = selectedFiles.filter(file => {
      const sizeMB = file.size / 1024 / 1024;
      if (sizeMB > maxSize) {
        alert(`文件 ${file.name} 超过 ${maxSize}MB 限制`);
        return false;
      }
      return true;
    });

    setFiles(prev => [...prev, ...validFiles]);
    
    if (onFileSelect) {
      setUploading(true);
      try {
        for (const file of validFiles) {
          await onFileSelect(file);
        }
      } finally {
        setUploading(false);
      }
    }
  };

  const removeFile = (index) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const getFileIcon = (file) => {
    if (file.type.startsWith('image/')) return Image;
    if (file.type === 'application/pdf') return FileText;
    return File;
  };

  return (
    <div className="space-y-3">
      <div
        onClick={() => fileInputRef.current?.click()}
        className="border-2 border-dashed border-slate-300 rounded-xl p-6 text-center cursor-pointer hover:border-primary-400 hover:bg-primary-50/50 transition-all"
      >
        <Upload className="w-8 h-8 mx-auto mb-2 text-slate-400" />
        <p className="text-sm text-slate-600 mb-1">
          点击上传或拖拽文件到此处
        </p>
        <p className="text-xs text-slate-500">
          支持 Markdown、PNG、JPG、PDF 格式，最大 {maxSize}MB
        </p>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept={accept}
          onChange={handleFileChange}
          className="hidden"
        />
      </div>

      {files.length > 0 && (
        <div className="space-y-2">
          {files.map((file, index) => {
            const Icon = getFileIcon(file);
            return (
              <div
                key={index}
                className="flex items-center gap-3 p-3 bg-slate-50 rounded-lg border border-slate-200"
              >
                <Icon className="w-5 h-5 text-slate-500 flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-slate-900 truncate">
                    {file.name}
                  </p>
                  <p className="text-xs text-slate-500">
                    {(file.size / 1024).toFixed(1)} KB
                  </p>
                </div>
                {uploading ? (
                  <Loader className="w-4 h-4 text-primary-600 animate-spin" />
                ) : (
                  <button
                    onClick={() => removeFile(index)}
                    className="p-1 hover:bg-slate-200 rounded transition-colors"
                  >
                    <X className="w-4 h-4 text-slate-500" />
                  </button>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default FileUpload;
