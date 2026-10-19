'use client';

import React, { useState } from 'react';
import hljs from 'highlight.js';
import { Copy, Check, Terminal, Code2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { DynamicDiagramEngine } from '../diagrams/DynamicDiagramEngine';
import { FormulaCard } from './FormulaCard';

interface CodeBlockProps {
  inline?: boolean;
  className?: string;
  children?: React.ReactNode;
}

export function CodeBlock({ inline, className, children }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  const codeString = String(children || '').replace(/\n$/, '');

  // Handle inline code: `x_ptr`
  if (inline) {
    return (
      <code className="px-1.5 py-0.5 rounded-md font-mono text-[11px] md:text-xs bg-[#161c27] text-sky-300 border border-[#273244] select-text">
        {codeString}
      </code>
    );
  }

  const match = /language-(\S+)/.exec(className || '');
  const langToken = match ? match[1].toLowerCase().trim() : '';

  // Case 1: Custom Diagram Code Fence: ```diagram:<type>
  if (langToken.startsWith('diagram:')) {
    const diagramType = langToken.replace(/^diagram:/, '');
    return <DynamicDiagramEngine type={diagramType} payload={codeString} />;
  }

  // Case 2: Formula Card: ```formula or ```math
  if (langToken === 'formula' || langToken === 'math') {
    return <FormulaCard code={codeString} />;
  }

  // Case 2.5: Table Block: ```table or ```table:<title>
  if (langToken.startsWith('table')) {
    return (
      <div className="my-6 overflow-x-auto rounded-xl border border-[var(--border-subtle)] bg-[var(--bg-card)] shadow-md">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            table: ({ children: tableChildren }) => (
              <table className="w-full text-left text-xs md:text-sm border-collapse font-sans">
                {tableChildren}
              </table>
            ),
            thead: ({ children: theadChildren }) => (
              <thead className="bg-[var(--bg-card-hover)] border-b border-[var(--border-subtle)] text-[var(--text-primary)] font-mono text-[11px] uppercase tracking-wider">
                {theadChildren}
              </thead>
            ),
            th: ({ children: thChildren }) => (
              <th className="px-4 py-3 font-semibold text-[var(--text-primary)] border border-[var(--border-subtle)]">
                {thChildren}
              </th>
            ),
            td: ({ children: tdChildren }) => (
              <td className="px-4 py-3 border border-[var(--border-subtle)] text-[var(--text-primary)]">
                {tdChildren}
              </td>
            ),
          }}
        >
          {codeString}
        </ReactMarkdown>
      </div>
    );
  }

  // Case 3: Executable Programming Code (C++, CUDA, Python, Shell)
  const language = hljs.getLanguage(langToken) ? langToken : undefined;
  const highlighted = language
    ? hljs.highlight(codeString, { language }).value
    : hljs.highlightAuto(codeString).value;

  const lines = highlighted.split('\n');
  if (lines.length > 1 && lines[lines.length - 1].trim() === '') {
    lines.pop();
  }

  const handleCopy = () => {
    navigator.clipboard.writeText(codeString).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const displayLang = langToken ? langToken.toUpperCase() : 'CODE';

  return (
    <div className="code-block-wrapper my-6 rounded-xl overflow-hidden border border-[#2a3241] bg-[#090d14] shadow-xl select-text">
      {/* Code Header */}
      <div className="flex items-center justify-between px-3.5 py-1.5 bg-[#121620] border-b border-[#21262d] text-xs select-none">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-sky-400"></span>
          <span className="font-mono text-[11px] text-slate-300 font-semibold tracking-wider">
            {displayLang}
          </span>
        </div>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 px-2.5 py-1 rounded text-[11px] font-medium text-slate-400 hover:text-slate-100 bg-[#1a212e] hover:bg-[#232c3d] border border-[#30363d] transition-all cursor-pointer"
        >
          {copied ? (
            <>
              <Check className="w-3.5 h-3.5 text-emerald-400" />
              <span className="text-emerald-300">Copied!</span>
            </>
          ) : (
            <>
              <Copy className="w-3.5 h-3.5" />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>

      {/* Code Body with Line Numbers */}
      <div className="py-2.5 overflow-x-auto">
        <div className="code-lines font-mono text-xs">
          {lines.map((lineContent, idx) => {
            const lineNumber = idx + 1;
            return (
              <div
                key={idx}
                className="code-line flex items-baseline hover:bg-white/[0.03] px-3.5 py-[1.5px] transition-colors"
              >
                <span className="line-gutter select-none text-[11px] font-mono text-[#484f58] pr-3 mr-3 border-r border-[#262d3d] text-right w-8 shrink-0">
                  {lineNumber}
                </span>
                <span
                  className="line-code font-mono text-xs leading-relaxed flex-1 whitespace-pre"
                  dangerouslySetInnerHTML={{ __html: lineContent || '&nbsp;' }}
                />
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
