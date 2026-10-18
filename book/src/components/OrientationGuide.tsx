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
      <div className="px-8 py-3 bg-[#161b22] border-b border-[#30363d] flex items-center justify-between shrink-0">
        <span className="text-xs font-mono text-[#8b949e]">Curriculum Navigation:</span>
        <div className="flex items-center gap-3">
          <button
            onClick={() => onNavigate({ type: 'theory', volumeId: 'vol1', chapterId: '1.1' })}
            className="px-3 py-1.5 rounded-lg bg-[#1f6feb] hover:bg-[#388bfd] text-white text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer shadow-sm"
          >
            Start Chapter 1.1 <ArrowRight className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => onNavigate({ type: 'playground', volumeId: 'vol1', playgroundType: 'cpp' })}
            className="px-3 py-1.5 rounded-lg bg-[#21262d] hover:bg-[#30363d] text-[#c9d1d9] text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer border border-[#30363d]"
          >
            <Terminal className="w-3.5 h-3.5 text-sky-400" /> C++ Playground
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
