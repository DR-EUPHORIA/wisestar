import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

const MarkdownRenderer = ({ content, className = '' }) => {
  return (
    <div className={`markdown-content ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          code({ node, inline, className, children, ...props }) {
            return !inline ? (
              <pre className="bg-slate-900 text-slate-100 p-4 rounded-xl overflow-x-auto my-4 shadow-inner-soft">
                <code className={className} {...props}>
                  {children}
                </code>
              </pre>
            ) : (
              <code className="bg-slate-100 text-slate-800 px-1.5 py-0.5 rounded text-sm font-mono" {...props}>
                {children}
              </code>
            );
          },
          table({ children }) {
            return (
              <div className="overflow-x-auto my-4">
                <table className="min-w-full divide-y divide-slate-200 border border-slate-200 rounded-lg">
                  {children}
                </table>
              </div>
            );
          },
          thead({ children }) {
            return <thead className="bg-slate-50">{children}</thead>;
          },
          th({ children }) {
            return (
              <th className="px-4 py-3 text-left text-xs font-semibold text-slate-700 uppercase tracking-wider border-b border-slate-200">
                {children}
              </th>
            );
          },
          td({ children }) {
            return (
              <td className="px-4 py-3 text-sm text-slate-700 border-b border-slate-100">
                {children}
              </td>
            );
          },
          blockquote({ children }) {
            return (
              <blockquote className="border-l-4 border-primary-400 bg-primary-50 pl-4 py-2 my-4 italic text-slate-700">
                {children}
              </blockquote>
            );
          },
          a({ href, children }) {
            return (
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary-600 hover:text-primary-700 underline"
              >
                {children}
              </a>
            );
          },
          h1({ children }) {
            return <h1 className="text-2xl md:text-3xl font-bold text-slate-900 mt-8 mb-4 border-b-2 border-slate-200 pb-2">{children}</h1>;
          },
          h2({ children }) {
            return <h2 className="text-xl md:text-2xl font-bold text-slate-900 mt-6 mb-3">{children}</h2>;
          },
          h3({ children }) {
            return <h3 className="text-lg md:text-xl font-semibold text-slate-900 mt-5 mb-2">{children}</h3>;
          },
          ul({ children }) {
            return <ul className="list-disc list-inside my-4 space-y-2 text-slate-700">{children}</ul>;
          },
          ol({ children }) {
            return <ol className="list-decimal list-inside my-4 space-y-2 text-slate-700">{children}</ol>;
          },
          p({ children }) {
            return <p className="my-3 text-slate-700 leading-relaxed">{children}</p>;
          }
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};

export default MarkdownRenderer;
