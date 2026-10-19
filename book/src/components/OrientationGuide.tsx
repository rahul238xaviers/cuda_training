'use client';

import React, { useEffect, useState } from 'react';
import { ArrowRight, Terminal } from 'lucide-react';
import { ActiveNode } from './SidebarTree';
import { ReaderView } from './ReaderView';

interface OrientationGuideProps {
  onNavigate: (node: ActiveNode) => void;
  content?: string;
}

export function OrientationGuide({ onNavigate, content: initialContent }: OrientationGuideProps) {
  const [content, setContent] = useState<string>(initialContent || '');

  useEffect(() => {
    if (initialContent) {
      setContent(initialContent);
      return;
    }
    fetch('/api/node?type=guide')
      .then((res) => res.json())
      .then((data) => {
        setContent(data.content || '');
      })
      .catch(() => {});
  }, [initialContent]);

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden">
      {/* Quick Navigation Action Bar */}
      <div className="px-8 py-3 bg-[var(--bg-header)] border-b border-[var(--border-subtle)] flex items-center justify-between shrink-0">
        <span className="text-xs font-mono text-[var(--text-secondary)] font-medium">Curriculum Navigation:</span>
        <div className="flex items-center gap-3">
          <button
            onClick={() => onNavigate({ type: 'theory', volumeId: 'vol1', chapterId: '1.1' })}
            className="px-3 py-1.5 rounded-lg bg-[#1f6feb] hover:bg-[#388bfd] text-white text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer shadow-sm"
          >
            Start Chapter 1.1 <ArrowRight className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => onNavigate({ type: 'playground', volumeId: 'vol1', playgroundType: 'cpp' })}
            className="px-3 py-1.5 rounded-lg bg-[var(--bg-card)] hover:bg-[var(--bg-card-hover)] text-[var(--text-primary)] text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer border border-[var(--border-subtle)] shadow-xs"
          >
            <Terminal className="w-3.5 h-3.5 text-sky-500 dark:text-sky-400" /> C++ Playground
          </button>
        </div>
      </div>

      {/* Dynamic Content via ReaderView AST Pipeline */}
      <ReaderView
        title="Curriculum Orientation & Roadmap"
        content={content}
        type="theory"
      />
    </div>
  );
}
