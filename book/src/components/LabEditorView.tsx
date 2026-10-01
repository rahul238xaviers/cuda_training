'use client';

import React, { useRef, useEffect, useMemo } from 'react';
import Editor, { OnMount } from '@monaco-editor/react';
import { Terminal, CheckCircle2, XCircle, Clock, Code2, Loader2, Save, Check } from 'lucide-react';

interface LabEditorViewProps {
  filePath: string;
  code: string;
  onChange: (newCode: string) => void;
  onSave?: () => void;
  saveStatus?: 'saved' | 'saving' | 'unsaved';
  output: string;
  runStatus: 'idle' | 'running' | 'success' | 'error';
  durationMs?: number;
  exitCode?: number | null;
}

/**
 * Converts terminal ANSI escape codes to styled HTML spans
 */
function renderAnsi(text: string): string {
  if (!text) return '';
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\u001b\[1;32m/g, '<span class="text-emerald-400 font-bold">')
    .replace(/\u001b\[32m/g, '<span class="text-emerald-400">')
    .replace(/\u001b\[1;31m/g, '<span class="text-rose-400 font-bold">')
    .replace(/\u001b\[31m/g, '<span class="text-rose-400">')
    .replace(/\u001b\[1;33m/g, '<span class="text-amber-400 font-bold">')
    .replace(/\u001b\[33m/g, '<span class="text-amber-400">')
    .replace(/\u001b\[1;34m/g, '<span class="text-sky-400 font-bold">')
    .replace(/\u001b\[34m/g, '<span class="text-sky-400">')
    .replace(/\u001b\[0m/g, '</span>')
    .replace(/\u001b\[\d+;?\d*m/g, '');
}

export function LabEditorView({
  filePath,
  code,
  onChange,
  onSave,
  saveStatus = 'saved',
  output,
  runStatus,
  durationMs,
  exitCode,
}: LabEditorViewProps) {
  const consoleRef = useRef<HTMLDivElement>(null);

  // Auto scroll console on new output
  useEffect(() => {
    if (consoleRef.current) {
      consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
    }
  }, [output, runStatus]);

  const handleEditorDidMount: OnMount = (editor, monaco) => {
    monaco.editor.defineTheme('cudaNightTheme', {
      base: 'vs-dark',
      inherit: true,
      rules: [
        { token: 'comment', foreground: '64748b', fontStyle: 'italic' },
        { token: 'keyword', foreground: '38bdf8', fontStyle: 'bold' },
        { token: 'type', foreground: '34d399' },
        { token: 'number', foreground: 'fbbf24' },
        { token: 'string', foreground: 'a7f3d0' },
        { token: 'operator', foreground: 'f472b6' },
      ],
      colors: {
        'editor.background': '#070a10',
        'editor.foreground': '#e2e8f0',
        'editorLineNumber.foreground': '#334155',
        'editorLineNumber.activeForeground': '#38bdf8',
        'editor.selectionBackground': '#1e293b',
        'editor.lineHighlightBackground': '#0d131f',
        'editorCursor.foreground': '#38bdf8',
        'editorIndentGuide.background': '#1e293b',
        'editorIndentGuide.activeBackground': '#38bdf8',
      },
    });
    monaco.editor.setTheme('cudaNightTheme');

    // Intercept Cmd+S / Ctrl+S inside Monaco to save directly to local disk
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => {
      if (onSave) {
        onSave();
      }
    });
  };

  const lineCount = useMemo(() => (code || '').split('\n').length, [code]);
  const formattedOutputHtml = useMemo(() => renderAnsi(output), [output]);

  return (
    <div className="flex-1 flex flex-col h-full bg-[#070a10] overflow-hidden">
      {/* Upper Editor Pane */}
      <div className="flex-1 flex flex-col border-b border-[#1e293b] min-h-[320px]">
        {/* Editor Sub-header with Live Disk Save Status */}
        <div className="h-10 px-4 bg-[#0e1422] border-b border-[#1e293b] flex items-center justify-between text-xs text-slate-400 select-none">
          <div className="flex items-center gap-2.5">
            <Code2 className="w-3.5 h-3.5 text-sky-400" />
            <span className="font-mono text-slate-200 font-medium">{filePath}</span>
            <span className="text-[10px] text-slate-500 font-mono">({lineCount} lines)</span>
          </div>

          <div className="flex items-center gap-3">
            {/* Live Disk Save Status Indicator */}
            {saveStatus === 'saving' && (
              <span className="flex items-center gap-1.5 text-[11px] text-sky-400 font-medium">
                <Loader2 className="w-3 h-3 animate-spin" /> Saving to disk...
              </span>
            )}
            {saveStatus === 'saved' && (
              <span className="flex items-center gap-1.5 text-[11px] text-emerald-400 font-medium">
                <Check className="w-3.5 h-3.5" /> Saved to disk
              </span>
            )}
            {saveStatus === 'unsaved' && (
              <span className="flex items-center gap-1.5 text-[11px] text-amber-400 font-medium">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" /> Unsaved changes
              </span>
            )}

            {/* Direct Save Button */}
            {onSave && (
              <button
                onClick={onSave}
                title="Save directly to local file (Cmd+S)"
                className="px-2.5 py-1 rounded bg-[#162032] hover:bg-[#1f2d47] text-slate-200 hover:text-white text-[11px] font-medium border border-[#1e293b] flex items-center gap-1.5 transition-all shadow-sm cursor-pointer"
              >
                <Save className="w-3 h-3 text-sky-400" />
                <span>Save to Disk</span>
                <kbd className="text-[9px] font-mono text-slate-400 bg-[#0c1018] px-1 rounded border border-[#1e293b]">
                  ⌘S
                </kbd>
              </button>
            )}
          </div>
        </div>

        {/* Monaco Editor Container */}
        <div className="flex-1 relative overflow-hidden bg-[#070a10]">
          <Editor
            height="100%"
            language="cpp"
            value={code}
            onChange={(val) => onChange(val || '')}
            onMount={handleEditorDidMount}
            loading={
              <div className="flex items-center justify-center h-full text-slate-400 gap-2 font-mono text-xs">
                <Loader2 className="w-4 h-4 animate-spin text-sky-400" />
                Loading C++ IDE Editor...
              </div>
            }
            options={{
              fontSize: 13,
              fontFamily: "'Fira Code', 'JetBrains Mono', Menlo, Monaco, Consolas, monospace",
              fontLigatures: true,
              tabSize: 4,
              insertSpaces: true,
              minimap: { enabled: true, scale: 0.8 },
              scrollBeyondLastLine: false,
              automaticLayout: true,
              padding: { top: 12, bottom: 12 },
              lineNumbers: 'on',
              renderLineHighlight: 'all',
              bracketPairColorization: { enabled: true },
              smoothScrolling: true,
              cursorBlinking: 'smooth',
            }}
          />
        </div>
      </div>

      {/* Lower Terminal Console Pane */}
      <div className="h-64 min-h-48 bg-[#05070c] flex flex-col">
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
              <span className="flex items-center gap-1 text-[11px] text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/20 font-medium animate-pulse">
                <Loader2 className="w-3 h-3 animate-spin" />
                Compiling & Executing...
              </span>
            )}
            {runStatus === 'success' && (
              <span className="flex items-center gap-1 text-[11px] text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20 font-medium">
                <CheckCircle2 className="w-3 h-3" />
                Passed (Exit 0)
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
            <div
              dangerouslySetInnerHTML={{ __html: formattedOutputHtml }}
              className="leading-relaxed"
            />
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
