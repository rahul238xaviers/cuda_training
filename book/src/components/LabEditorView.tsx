'use client';

import React, { useRef, useEffect } from 'react';
import { Terminal, CheckCircle2, XCircle, AlertTriangle, Clock, Code2 } from 'lucide-react';

interface LabEditorViewProps {
  filePath: string;
  code: string;
  onChange: (newCode: string) => void;
  output: string;
  runStatus: 'idle' | 'running' | 'success' | 'error';
  durationMs?: number;
  exitCode?: number | null;
}

export function LabEditorView({
  filePath,
  code,
  onChange,
  output,
  runStatus,
  durationMs,
  exitCode,
}: LabEditorViewProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const consoleRef = useRef<HTMLDivElement>(null);

  // Auto scroll console on new output
  useEffect(() => {
    if (consoleRef.current) {
      consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
    }
  }, [output, runStatus]);

  // Tab key handler for 4-space indentation
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Tab') {
      e.preventDefault();
      const textarea = textareaRef.current;
      if (!textarea) return;

      const start = textarea.selectionStart;
      const end = textarea.selectionEnd;
      const spaces = '    ';
      const newCode = code.substring(0, start) + spaces + code.substring(end);

      onChange(newCode);

      setTimeout(() => {
        textarea.selectionStart = textarea.selectionEnd = start + spaces.length;
      }, 0);
    }
  };

  const lineCount = (code || '').split('\n').length;

  return (
    <div className="flex-1 flex flex-col h-full bg-[#080b11] overflow-hidden">
      {/* Upper Editor Pane */}
      <div className="flex-1 flex flex-col border-b border-[#1e293b] min-h-[300px]">
        {/* Editor Sub-header */}
        <div className="h-9 px-4 bg-[#0e1422] border-b border-[#1e293b] flex items-center justify-between text-xs text-slate-400 select-none">
          <div className="flex items-center gap-2">
            <Code2 className="w-3.5 h-3.5 text-sky-400" />
            <span className="font-mono text-slate-200">{filePath}</span>
            <span className="text-[10px] text-slate-500 font-mono">({lineCount} lines)</span>
          </div>
          <div className="flex items-center gap-2 text-[11px] text-slate-400">
            <span>C++20</span>
            <span className="text-slate-600">•</span>
            <span>Clang++ -O3</span>
          </div>
        </div>

        {/* Code Input */}
        <div className="flex-1 relative flex overflow-hidden">
          {/* Line Numbers */}
          <div className="w-12 bg-[#090d16] text-[#475569] font-mono text-xs select-none py-3 text-right pr-3 border-r border-[#1a2333] shrink-0 overflow-hidden">
            {Array.from({ length: lineCount }).map((_, i) => (
              <div key={i} className="leading-6">
                {i + 1}
              </div>
            ))}
          </div>

          {/* Textarea */}
          <textarea
            ref={textareaRef}
            value={code}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            spellCheck={false}
            className="flex-1 bg-[#090d16] text-slate-100 font-mono text-xs p-3 leading-6 resize-none focus:outline-none focus:ring-0 selection:bg-sky-500/30 whitespace-pre overflow-auto"
          />
        </div>
      </div>

      {/* Lower Terminal Console Pane */}
      <div className="h-64 min-h-48 bg-[#06080d] flex flex-col">
        {/* Terminal Header */}
        <div className="h-8 px-4 bg-[#0b0f17] border-b border-[#1e293b] flex items-center justify-between select-none">
          <div className="flex items-center gap-2 text-xs font-semibold text-slate-300">
            <Terminal className="w-3.5 h-3.5 text-sky-400" />
            <span>Interactive Terminal & Test Runner</span>
          </div>

          <div className="flex items-center gap-3">
            {durationMs !== undefined && durationMs > 0 && (
              <span className="flex items-center gap-1 text-[11px] text-slate-400 font-mono">
                <Clock className="w-3 h-3 text-slate-500" />
                {durationMs}ms
              </span>
            )}

            {runStatus === 'running' && (
              <span className="text-[11px] text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/20 font-medium animate-pulse">
                Compiling & Executing...
              </span>
            )}
            {runStatus === 'success' && (
              <span className="flex items-center gap-1 text-[11px] text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20 font-medium">
                <CheckCircle2 className="w-3 h-3" />
                Tests Passed (0)
              </span>
            )}
            {runStatus === 'error' && (
              <span className="flex items-center gap-1 text-[11px] text-rose-400 bg-rose-500/10 px-2 py-0.5 rounded border border-rose-500/20 font-medium">
                <XCircle className="w-3 h-3" />
                Error {exitCode !== null ? `(${exitCode})` : ''}
              </span>
            )}
            {runStatus === 'idle' && (
              <span className="text-[11px] text-slate-500 font-mono">Ready</span>
            )}
          </div>
        </div>

        {/* Console Body */}
        <div
          ref={consoleRef}
          className="flex-1 p-3.5 font-mono text-xs overflow-y-auto whitespace-pre-wrap select-text leading-5"
        >
          {output ? (
            <span
              className={
                runStatus === 'error'
                  ? 'text-rose-300'
                  : runStatus === 'success'
                  ? 'text-emerald-300'
                  : 'text-slate-300'
              }
            >
              {output}
            </span>
          ) : (
            <span className="text-slate-600 italic">
              Click ▶ Run Test Suite or ▶ Run Playground above to compile with clang++ -std=c++20 -O3 and view live output.
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
