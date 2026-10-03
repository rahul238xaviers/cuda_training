'use client';

import React, { useMemo, useEffect, useState, useRef } from 'react';
import { marked } from 'marked';
import hljs from 'highlight.js';
import { BookOpen, Bookmark, Highlighter, Sparkles, Check, Share2, Layers } from 'lucide-react';
import { GenericDiagramRenderer } from './TextbookDiagrams';
import { HighlightPalette } from './HighlightPalette';
import { AnnotationsDrawer } from './AnnotationsDrawer';
import {
  getBookmarks,
  getHighlights,
  isChapterBookmarked,
  toggleChapterBookmark,
  ChapterHighlight,
  MarkerColor,
} from '../lib/annotations';

interface ReaderViewProps {
  title: string;
  content: string;
  chapterId?: string;
  type: 'theory' | 'cheat_sheet';
  onNavigateChapter?: (chapterId: string) => void;
}

interface ContentChunk {
  type: 'markdown' | 'diagram';
  content?: string;
  diagramType?: string;
  payload?: string;
}

const MARKER_COLOR_CLASSES: Record<MarkerColor, string> = {
  yellow: 'bg-amber-400/25 border-b-2 border-amber-400 text-amber-100',
  emerald: 'bg-emerald-400/25 border-b-2 border-emerald-400 text-emerald-100',
  purple: 'bg-purple-400/25 border-b-2 border-purple-400 text-purple-100',
  cyan: 'bg-sky-400/25 border-b-2 border-sky-400 text-sky-100',
};

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

  // Configure marked renderer for IDE-grade code blocks
  const markedRenderer = useMemo(() => {
    const renderer = new marked.Renderer();

    renderer.code = function ({ text, lang }: { text: string; lang?: string }) {
      const cleanLang = (lang || '').toLowerCase().trim();

      // Case 1: Mathematical Formula or Memory Addressing Card (No line numbers)
      if (cleanLang === 'formula' || cleanLang === 'math') {
        const safeText = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        const formulaLines = safeText
          .split('\n')
          .filter((l) => l.trim().length > 0)
          .map((line) => `<div class="py-1 tracking-wide">${line}</div>`)
          .join('');
        return `<div class="my-6 rounded-xl border border-sky-500/30 bg-[#0e1420] shadow-lg overflow-hidden select-text"><div class="flex items-center justify-between px-4 py-2 bg-[#141b2a] border-b border-[#212b3d] text-xs select-none"><span class="text-[11px] font-mono text-sky-400 font-semibold tracking-wider uppercase flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-sky-400"></span> Mathematical Definition & Address Scaling</span><span class="text-[10px] text-slate-400 font-mono">Systems Arithmetic</span></div><div class="p-4 text-center font-mono text-[13px] md:text-sm text-sky-100 font-medium overflow-x-auto">${formulaLines}</div></div>`;
      }

      // Case 2: Plain Monospace Text or Memory Timeline (No line numbers)
      if (!cleanLang || cleanLang === 'text') {
        const safeText = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        return `<div class="my-5 rounded-xl border border-[#262d3d] bg-[#0c1017] p-4 overflow-x-auto select-text shadow-md"><div class="font-mono text-xs leading-relaxed text-slate-300 whitespace-pre">${safeText}</div></div>`;
      }

      // Case 3: Executable Programming Code (IDE-grade with line-number gutter)
      const language = hljs.getLanguage(cleanLang) ? cleanLang : undefined;
      const highlighted = language
        ? hljs.highlight(text, { language }).value
        : hljs.highlightAuto(text).value;

      const lines = highlighted.split('\n');
      if (lines.length > 1 && lines[lines.length - 1].trim() === '') {
        lines.pop();
      }

      const codeLinesHtml = lines
        .map((line, idx) => {
          const lineNumber = idx + 1;
          const content = line || '&nbsp;';
          return `<div class="code-line flex items-baseline hover:bg-white/[0.03] px-3.5 py-[1.5px] transition-colors"><span class="line-gutter select-none text-[11px] font-mono text-[#484f58] pr-3 mr-3 border-r border-[#262d3d] text-right w-8 shrink-0">${lineNumber}</span><span class="line-code font-mono text-xs leading-relaxed flex-1 whitespace-pre">${content}</span></div>`;
        })
        .join('');

      const displayLang = cleanLang.toUpperCase();

      return `<div class="code-block-wrapper my-6 rounded-xl overflow-hidden border border-[#2a3241] bg-[#090d14] shadow-xl"><div class="code-block-header flex items-center justify-between px-3.5 py-1.5 bg-[#121620] border-b border-[#21262d] text-xs select-none"><div class="flex items-center gap-2"><span class="w-2 h-2 rounded-full bg-sky-400"></span><span class="font-mono text-[11px] text-slate-300 font-semibold tracking-wider">${displayLang}</span></div><button class="copy-btn px-2.5 py-1 rounded text-[11px] font-medium text-slate-400 hover:text-slate-100 bg-[#1a212e] hover:bg-[#232c3d] transition-all cursor-pointer flex items-center gap-1 border border-[#30363d]" data-code="${encodeURIComponent(text)}">Copy</button></div><div class="py-2.5 overflow-x-auto"><div class="code-lines font-mono text-xs">${codeLinesHtml}</div></div></div>`;
    };

    marked.setOptions({
      gfm: true,
      breaks: true,
      renderer,
    });

    return renderer;
  }, []);

  // Split content into Markdown chunks and Special Diagram tag blocks
  const chunks = useMemo(() => {
    if (!content) return [];

    const DIAGRAM_REGEX = /```diagram:([a-zA-Z0-9_-]+)\s*\n([\s\S]*?)```/g;
    const result: ContentChunk[] = [];
    let lastIndex = 0;
    let match: RegExpExecArray | null;

    while ((match = DIAGRAM_REGEX.exec(content)) !== null) {
      if (match.index > lastIndex) {
        result.push({
          type: 'markdown',
          content: content.slice(lastIndex, match.index),
        });
      }
      result.push({
        type: 'diagram',
        diagramType: match[1],
        payload: match[2],
      });
      lastIndex = match.index + match[0].length;
    }

    if (lastIndex < content.length) {
      result.push({
        type: 'markdown',
        content: content.slice(lastIndex),
      });
    }

    return result;
  }, [content]);

  // Handle Copy Button clicks inside rendered HTML
  useEffect(() => {
    const handleCopy = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      const btn = target.closest('.copy-btn') as HTMLButtonElement | null;
      if (!btn) return;

      const rawCode = btn.getAttribute('data-code');
      if (rawCode) {
        const decoded = decodeURIComponent(rawCode);
        navigator.clipboard.writeText(decoded).then(() => {
          const originalText = btn.innerText;
          btn.innerText = 'Copied! ✓';
          btn.classList.add('text-emerald-400', 'bg-emerald-950/40');
          setTimeout(() => {
            btn.innerText = originalText;
            btn.classList.remove('text-emerald-400', 'bg-emerald-950/40');
          }, 2000);
        });
      }
    };

    document.addEventListener('click', handleCopy);
    return () => document.removeEventListener('click', handleCopy);
  }, []);

  return (
    <div
      ref={containerRef}
      className="flex-1 overflow-y-auto bg-[#0d1117] p-8 md:p-12 lg:px-16 w-full select-text relative"
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
        <div className="mb-8 pb-6 border-b border-[#262d3d] flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-mono mb-2.5">
              {type === 'theory' ? (
                <span className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-md bg-sky-500/10 text-sky-300 border border-sky-500/20 font-medium">
                  <BookOpen className="w-3.5 h-3.5" /> CHAPTER {chapterId} THEORY
                </span>
              ) : (
                <span className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-md bg-amber-500/10 text-amber-300 border border-amber-500/20 font-medium">
                  REVISION CHEAT SHEET • CHAPTER {chapterId}
                </span>
              )}
            </div>
            <h1 className="text-2xl md:text-3xl font-bold text-[#f0f6fc] tracking-tight leading-tight">
              {title}
            </h1>
            <p className="mt-2 text-xs text-[#8b949e] font-sans">
              Hardware Mental Models • Physical Architecture & Memory Mechanics
            </p>
          </div>

          {/* Action Toolbar: Bookmark & Notes */}
          <div className="flex items-center gap-2 shrink-0">
            <button
              onClick={handleToggleBookmark}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-medium transition-all ${
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
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#161b22] hover:bg-[#21262d] border border-[#30363d] text-xs font-medium text-slate-300 hover:text-white transition-all"
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

        {/* Content Chunks: Alternating between Markdown prose and Embedded SVG Diagrams */}
        <div className="space-y-6">
          {chunks.map((chunk, idx) => {
            if (chunk.type === 'diagram' && chunk.diagramType && chunk.payload) {
              return (
                <div key={idx} className="my-6">
                  <GenericDiagramRenderer type={chunk.diagramType} payload={chunk.payload} />
                </div>
              );
            }

            const html = marked.parse(chunk.content || '', { renderer: markedRenderer }) as string;
            return (
              <div
                key={idx}
                className="prose-dark leading-relaxed select-text"
                dangerouslySetInnerHTML={{ __html: html }}
              />
            );
          })}
        </div>
      </div>
    </div>
  );
}
