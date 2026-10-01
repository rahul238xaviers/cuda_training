'use client';

import React, { useMemo } from 'react';
import { marked } from 'marked';
import { BookOpen, Copy, Check, Info } from 'lucide-react';

interface ReaderViewProps {
  title: string;
  content: string;
  chapterId?: string;
  type: 'theory' | 'cheat_sheet';
}

export function ReaderView({ title, content, chapterId, type }: ReaderViewProps) {
  const parsedHtml = useMemo(() => {
    marked.setOptions({
      gfm: true,
      breaks: true,
    });
    return marked.parse(content || '*No content available.*') as string;
  }, [content]);

  return (
    <div className="flex-1 overflow-y-auto bg-[#080b11] p-8 md:p-12 lg:px-20 max-w-5xl mx-auto w-full">
      {/* Top Banner */}
      <div className="mb-8 pb-6 border-b border-[#1e293b]">
        <div className="flex items-center gap-2 text-xs text-sky-400 font-mono mb-2">
          {type === 'theory' ? (
            <span className="flex items-center gap-1.5 px-2.5 py-0.5 rounded bg-sky-500/10 border border-sky-500/20 font-semibold">
              <BookOpen className="w-3 h-3" /> CHAPTER {chapterId} THEORY
            </span>
          ) : (
            <span className="flex items-center gap-1.5 px-2.5 py-0.5 rounded bg-amber-500/10 text-amber-300 border border-amber-500/20 font-semibold">
              REVISION CHEAT SHEET • CHAPTER {chapterId}
            </span>
          )}
        </div>
        <h1 className="text-3xl font-extrabold text-slate-100 tracking-tight leading-tight">
          {title}
        </h1>
        <p className="mt-2 text-sm text-slate-400">
          Strictly Zero LaTeX • Pure Hardware Mental Models • Plain Text & ASCII Diagrams
        </p>
      </div>

      {/* Rendered Prose Content */}
      <div
        className="prose-dark leading-relaxed select-text"
        dangerouslySetInnerHTML={{ __html: parsedHtml }}
      />
    </div>
  );
}
