'use client';

import React, { useMemo, useEffect } from 'react';
import { marked } from 'marked';
import hljs from 'highlight.js';
import { BookOpen } from 'lucide-react';

interface ReaderViewProps {
  title: string;
  content: string;
  chapterId?: string;
  type: 'theory' | 'cheat_sheet';
}

export function ReaderView({ title, content, chapterId, type }: ReaderViewProps) {
  const parsedHtml = useMemo(() => {
    const renderer = new marked.Renderer();

    renderer.code = function ({ text, lang }: { text: string; lang?: string }) {
      const language = lang && hljs.getLanguage(lang) ? lang : undefined;
      const highlighted = language
        ? hljs.highlight(text, { language }).value
        : hljs.highlightAuto(text).value;
      const displayLang = (lang || 'code').toUpperCase();

      return `
        <div class="code-block-wrapper my-5 rounded-lg overflow-hidden border border-[#1e293b] bg-[#090d16] shadow-lg">
          <div class="code-block-header flex items-center justify-between px-3.5 py-1.5 bg-[#0e1422] border-b border-[#1e293b] text-xs select-none">
            <div class="flex items-center gap-2">
              <span class="w-2.5 h-2.5 rounded-full bg-sky-500/80"></span>
              <span class="font-mono text-[11px] text-sky-400 font-semibold tracking-wider">${displayLang}</span>
            </div>
            <button
              class="copy-btn px-2.5 py-1 rounded text-[11px] font-medium text-slate-300 bg-[#162032] hover:bg-[#1f2d47] hover:text-white transition-all cursor-pointer flex items-center gap-1"
              data-code="${encodeURIComponent(text)}"
            >
              Copy
            </button>
          </div>
          <pre class="!bg-transparent !p-4 !m-0 overflow-x-auto"><code class="hljs ${language || ''} font-mono text-xs leading-5">${highlighted}</code></pre>
        </div>
      `;
    };

    marked.setOptions({
      gfm: true,
      breaks: true,
      renderer,
    });

    return marked.parse(content || '*No content available.*') as string;
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
    <div className="flex-1 overflow-y-auto bg-[#080b11] p-8 md:p-12 lg:px-20 max-w-5xl mx-auto w-full select-text">
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
