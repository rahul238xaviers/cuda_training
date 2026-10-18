'use client';

import React, { useEffect, useState, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { BookOpen, Bookmark, Highlighter } from 'lucide-react';

import { HighlightPalette } from './HighlightPalette';
import { AnnotationsDrawer } from './AnnotationsDrawer';
import { CalloutBlock } from './reader/CalloutBlock';
import { AdaptiveAsset } from './reader/AdaptiveAsset';
import { CodeBlock } from './reader/CodeBlock';
import {
  getBookmarks,
  getHighlights,
  isChapterBookmarked,
  toggleChapterBookmark,
  ChapterHighlight,
} from '../lib/annotations';

interface ReaderViewProps {
  title: string;
  content: string;
  chapterId?: string;
  type: 'theory' | 'cheat_sheet';
  onNavigateChapter?: (chapterId: string) => void;
}

export function ReaderView({ title, content, chapterId = '1.1', type, onNavigateChapter }: ReaderViewProps) {
  const [isBookmarked, setIsBookmarked] = useState(false);
  const [highlights, setHighlights] = useState<ChapterHighlight[]>([]);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Sync bookmarks and highlights
  const refreshAnnotations = () => {
    setIsBookmarked(isChapterBookmarked(chapterId));
    setHighlights(getHighlights(chapterId));
  };

  useEffect(() => {
    refreshAnnotations();
    const handleUpdate = () => refreshAnnotations();
    window.addEventListener('cuda-annotations-updated', handleUpdate);
    return () => window.removeEventListener('cuda-annotations-updated', handleUpdate);
  }, [chapterId]);

  const handleToggleBookmark = () => {
    const newState = toggleChapterBookmark(chapterId, title);
    setIsBookmarked(newState);
  };

  return (
    <div
      ref={containerRef}
      className="flex-1 overflow-y-auto bg-[var(--bg-card)] text-[var(--text-primary)] p-8 md:p-12 lg:px-16 w-full select-text relative reader-view-container"
    >
      {/* Floating Text Highlighter Palette */}
      <HighlightPalette
        containerRef={containerRef}
        chapterId={chapterId}
        chapterTitle={title}
        onHighlightCreated={refreshAnnotations}
      />

      {/* Annotations & Notes Drawer */}
      <AnnotationsDrawer
        isOpen={isDrawerOpen}
        onClose={() => setIsDrawerOpen(false)}
        onNavigateChapter={(id) => onNavigateChapter?.(id)}
        currentChapterId={chapterId}
      />

      <div className="max-w-4xl mx-auto">
        {/* Top Header & Study Tools */}
        <div className="mb-8 pb-6 border-b border-[var(--border-subtle)] flex flex-col md:flex-row md:items-center md:justify-between gap-4 reader-header">
          <div>
            <div className="flex items-center gap-2 text-xs font-mono mb-2.5">
              {type === 'theory' ? (
                <span className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-md bg-sky-500/10 text-sky-400 border border-sky-500/20 font-medium">
                  <BookOpen className="w-3.5 h-3.5" /> CHAPTER {chapterId} THEORY
                </span>
              ) : (
                <span className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-md bg-amber-500/10 text-amber-300 border border-amber-500/20 font-medium">
                  REVISION CHEAT SHEET • CHAPTER {chapterId}
                </span>
              )}
            </div>
            <h1 className="text-2xl md:text-3xl font-bold text-[var(--text-primary)] tracking-tight leading-tight reader-title">
              {title}
            </h1>
            <p className="mt-2 text-xs text-[var(--text-secondary)] font-sans reader-subtitle">
              Hardware Mental Models • Physical Architecture & Memory Mechanics
            </p>
          </div>

          {/* Action Toolbar: Bookmark & Notes */}
          <div className="flex items-center gap-2 shrink-0">
            <button
              onClick={handleToggleBookmark}
              className={`reader-action-btn flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-medium transition-all ${
                isBookmarked
                  ? 'bg-amber-500/10 text-amber-300 border-amber-500/30 hover:bg-amber-500/20'
                  : 'bg-[#161b22] text-slate-300 border-[#30363d] hover:bg-[#21262d] hover:text-white'
              }`}
            >
              <Bookmark className={`w-3.5 h-3.5 ${isBookmarked ? 'fill-amber-400 text-amber-400' : ''}`} />
              <span>{isBookmarked ? 'Bookmarked' : 'Bookmark'}</span>
            </button>

            <button
              onClick={() => setIsDrawerOpen(true)}
              className="reader-action-btn flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#161b22] hover:bg-[#21262d] border border-[#30363d] text-xs font-medium text-slate-300 hover:text-white transition-all"
            >
              <Highlighter className="w-3.5 h-3.5 text-amber-400" />
              <span>Notes</span>
              {highlights.length > 0 && (
                <span className="px-1.5 py-0.2 rounded-full bg-amber-500/20 text-amber-300 text-[10px] font-mono border border-amber-500/30">
                  {highlights.length}
                </span>
              )}
            </button>
          </div>
        </div>

        {/* Dynamic Markdown AST Pipeline */}
        <div className="prose-dark leading-relaxed select-text space-y-4">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              blockquote: ({ children }) => <CalloutBlock>{children}</CalloutBlock>,
              img: ({ src, alt, title: imgTitle }: any) => (
                <AdaptiveAsset
                  src={typeof src === 'string' ? src : ''}
                  alt={alt}
                  title={imgTitle}
                  chapterId={chapterId}
                />
              ),
              code: ({ className, children, ...props }: any) => (
                <CodeBlock inline={props.inline || !className} className={className}>
                  {children}
                </CodeBlock>
              ),
              table: ({ children }) => (
                <div className="my-6 overflow-x-auto rounded-xl border border-[var(--border-subtle)] bg-[#0c1017] shadow-lg">
                  <table className="w-full text-left text-xs md:text-sm border-collapse font-sans">
                    {children}
                  </table>
                </div>
              ),
              thead: ({ children }) => (
                <thead className="bg-[#141b28] border-b border-[#212c3f] text-slate-200 font-mono text-[11px] uppercase tracking-wider">
                  {children}
                </thead>
              ),
              th: ({ children }) => <th className="px-4 py-3 font-semibold">{children}</th>,
              td: ({ children }) => <td className="px-4 py-3 border-b border-[#1c2433] text-slate-300">{children}</td>,
              h1: ({ children }) => (
                <h1 className="text-2xl md:text-3xl font-extrabold text-[var(--text-primary)] mt-8 mb-4 tracking-tight">
                  {children}
                </h1>
              ),
              h2: ({ children }) => (
                <h2 className="text-xl md:text-2xl font-bold text-[var(--text-primary)] mt-8 mb-3.5 tracking-tight border-b border-[var(--border-subtle)] pb-2">
                  {children}
                </h2>
              ),
              h3: ({ children }) => (
                <h3 className="text-lg md:text-xl font-bold text-[var(--text-primary)] mt-6 mb-2.5 tracking-tight">
                  {children}
                </h3>
              ),
              p: ({ children }) => <div className="my-3.5 leading-relaxed text-slate-300">{children}</div>,
              ul: ({ children }) => <ul className="my-3.5 pl-6 list-disc space-y-1.5 text-slate-300">{children}</ul>,
              ol: ({ children }) => <ol className="my-3.5 pl-6 list-decimal space-y-1.5 text-slate-300">{children}</ol>,
              li: ({ children }) => <li className="leading-relaxed">{children}</li>,
              hr: () => <hr className="my-8 border-[var(--border-subtle)]" />,
            }}
          >
            {content}
          </ReactMarkdown>
        </div>
      </div>
    </div>
  );
}
