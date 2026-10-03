'use client';

import React, { useRef, useEffect, useState, useMemo, useCallback } from 'react';
import Editor, { OnMount } from '@monaco-editor/react';
import {
  Terminal,
  CheckCircle2,
  XCircle,
  Clock,
  Code2,
  Loader2,
  Save,
  Check,
  Sparkles,
  AlertCircle,
  AlertTriangle,
  CheckCheck,
  Wrench,
  ChevronDown,
  ChevronUp,
  SlidersHorizontal,
} from 'lucide-react';

interface DiagnosticItem {
  line: number;
  col: number;
  severity: 'error' | 'warning' | 'info';
  message: string;
}

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
  const editorRef = useRef<any>(null);
  const monacoRef = useRef<any>(null);
  const lintTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Linting & Formatting State
  const [diagnostics, setDiagnostics] = useState<DiagnosticItem[]>([]);
  const [isLinting, setIsLinting] = useState<boolean>(false);
  const [isFormatting, setIsFormatting] = useState<boolean>(false);
  const [formatSuccess, setFormatSuccess] = useState<boolean>(false);
  const [activeBottomTab, setActiveBottomTab] = useState<'terminal' | 'problems'>('terminal');
  const [isDockOpen, setIsDockOpen] = useState<boolean>(true);

  // Counts
  const errorCount = useMemo(
    () => diagnostics.filter((d) => d.severity === 'error').length,
    [diagnostics]
  );
  const warningCount = useMemo(
    () => diagnostics.filter((d) => d.severity === 'warning').length,
    [diagnostics]
  );

  // Auto scroll console on new output
  useEffect(() => {
    if (consoleRef.current && activeBottomTab === 'terminal') {
      consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
    }
  }, [output, runStatus, activeBottomTab]);

  // If new test run starts, switch to terminal tab and expand panel automatically
  useEffect(() => {
    if (runStatus === 'running' || runStatus === 'success' || runStatus === 'error') {
      setIsDockOpen(true);
      setActiveBottomTab('terminal');
    }
  }, [runStatus]);

  /**
   * Run clang++ syntax linter and populate Monaco model markers (red/yellow squiggles)
   */
  const runLint = useCallback(async (codeToLint: string) => {
    if (!codeToLint) {
      setDiagnostics([]);
      return;
    }
    setIsLinting(true);
    try {
      const res = await fetch('/api/lint', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: codeToLint }),
      });
      const data = await res.json();
      if (data.success) {
        const diags: DiagnosticItem[] = data.diagnostics || [];
        setDiagnostics(diags);

        // Apply squiggly markers in Monaco editor
        if (editorRef.current && monacoRef.current) {
          const model = editorRef.current.getModel();
          if (model) {
            const markers = diags.map((d) => {
              const lineMax = model.getLineMaxColumn(d.line) || (d.col + 6);
              return {
                severity:
                  d.severity === 'error'
                    ? monacoRef.current.MarkerSeverity.Error
                    : d.severity === 'warning'
                    ? monacoRef.current.MarkerSeverity.Warning
                    : monacoRef.current.MarkerSeverity.Info,
                startLineNumber: d.line,
                startColumn: Math.max(1, d.col),
                endLineNumber: d.line,
                endColumn: Math.max(d.col + 3, lineMax),
                message: d.message,
                source: 'clang++ linter',
              };
            });
            monacoRef.current.editor.setModelMarkers(model, 'clang-linter', markers);
          }
        }
      }
    } catch (err) {
      console.error('Lint check failed:', err);
    } finally {
      setIsLinting(false);
    }
  }, []);

  // Debounced auto-linting as user writes code
  useEffect(() => {
    if (lintTimerRef.current) {
      clearTimeout(lintTimerRef.current);
    }
    lintTimerRef.current = setTimeout(() => {
      runLint(code);
    }, 900);

    return () => {
      if (lintTimerRef.current) {
        clearTimeout(lintTimerRef.current);
      }
    };
  }, [code, runLint]);

  /**
   * Format code with clang-format and update Monaco in-place
   */
  const handleFormat = useCallback(async () => {
    const currentCode = editorRef.current ? editorRef.current.getValue() : code;
    if (!currentCode) return;

    setIsFormatting(true);
    try {
      const res = await fetch('/api/format', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: currentCode }),
      });
      const data = await res.json();
      if (data.success && data.formatted) {
        const formatted = data.formatted;
        if (editorRef.current) {
          const pos = editorRef.current.getPosition();
          editorRef.current.setValue(formatted);
          if (pos) editorRef.current.setPosition(pos);
        }
        onChange(formatted);
        setFormatSuccess(true);
        setTimeout(() => setFormatSuccess(false), 2000);
        // Immediate re-lint after format
        runLint(formatted);
      }
    } catch (err) {
      console.error('Formatting failed:', err);
    } finally {
      setIsFormatting(false);
    }
  }, [code, onChange, runLint]);

  /**
   * Jump to problem line and column in editor
   */
  const jumpToProblem = (diag: DiagnosticItem) => {
    if (editorRef.current) {
      editorRef.current.revealLineInCenter(diag.line);
      editorRef.current.setPosition({ lineNumber: diag.line, column: Math.max(1, diag.col) });
      editorRef.current.focus();
    }
  };

  const handleEditorDidMount: OnMount = (editor, monaco) => {
    editorRef.current = editor;
    monacoRef.current = monaco;

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

    // Register Document Formatting Provider for C++ (Shift+Option+F / Shift+Alt+F / Cmd+Shift+I)
    monaco.languages.registerDocumentFormattingEditProvider('cpp', {
      async provideDocumentFormattingEdits(model: any) {
        const text = model.getValue();
        try {
          const res = await fetch('/api/format', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code: text }),
          });
          const data = await res.json();
          if (data.success && data.formatted) {
            onChange(data.formatted);
            setTimeout(() => runLint(data.formatted), 100);
            return [
              {
                range: model.getFullModelRange(),
                text: data.formatted,
              },
            ];
          }
        } catch {}
        return [];
      },
    });

    // Initial lint
    runLint(code);
  };

  const lineCount = useMemo(() => (code || '').split('\n').length, [code]);
  const formattedOutputHtml = useMemo(() => renderAnsi(output), [output]);

  return (
    <div className="flex-1 flex flex-col h-full bg-[#070a10] overflow-hidden">
      {/* Upper Editor Pane */}
      <div className="flex-1 flex flex-col border-b border-[#1e293b] min-h-[320px]">
        {/* Editor Sub-header with Format, Lint, and Save Tools */}
        <div className="h-10 px-4 bg-[#0e1422] border-b border-[#1e293b] flex items-center justify-between text-xs text-slate-400 select-none">
          {/* Left: File metadata */}
          <div className="flex items-center gap-2.5">
            <Code2 className="w-3.5 h-3.5 text-sky-400" />
            <span className="font-mono text-slate-200 font-medium">{filePath}</span>
            <span className="text-[10px] text-slate-500 font-mono">({lineCount} lines)</span>
          </div>

          {/* Right: Lint Status, Format, and Save Actions */}
          <div className="flex items-center gap-2.5">
            {/* Real-time Lint Diagnostic Pill */}
            {isLinting ? (
              <span className="flex items-center gap-1.5 text-[11px] text-sky-400 font-medium">
                <Loader2 className="w-3 h-3 animate-spin" /> Linting...
              </span>
            ) : errorCount > 0 || warningCount > 0 ? (
              <button
                onClick={() => {
                  setActiveBottomTab('problems');
                  if (diagnostics[0]) jumpToProblem(diagnostics[0]);
                }}
                title="Click to view problems & jump to error"
                className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border border-rose-500/30 text-[11px] font-medium transition-all cursor-pointer shadow-sm"
              >
                <AlertCircle className="w-3.5 h-3.5 text-rose-400 animate-pulse" />
                <span>
                  {errorCount} {errorCount === 1 ? 'Error' : 'Errors'}
                </span>
                {warningCount > 0 && (
                  <span className="text-amber-400 text-[10px]">
                    ({warningCount} {warningCount === 1 ? 'warn' : 'warns'})
                  </span>
                )}
              </button>
            ) : (
              <button
                onClick={() => {
                  setIsDockOpen(true);
                  setActiveBottomTab('problems');
                }}
                title="No syntax errors detected - click to open problems inspector"
                className="flex items-center gap-1 text-[11px] text-emerald-400 bg-emerald-500/10 hover:bg-emerald-500/20 px-2 py-0.5 rounded border border-emerald-500/20 font-medium cursor-pointer transition-colors"
              >
                <CheckCheck className="w-3.5 h-3.5" /> Clean (0 errors)
              </button>
            )}

            {/* Quick Lint Check Button */}
            <button
              onClick={() => runLint(editorRef.current ? editorRef.current.getValue() : code)}
              disabled={isLinting}
              title="Run C++20 / CUDA syntax & semantic linter (clang++)"
              className="px-2 py-1 rounded bg-[#141b2b] hover:bg-[#1e283d] text-slate-300 hover:text-white text-[11px] font-medium border border-[#1e293b] flex items-center gap-1 transition-all cursor-pointer"
            >
              <Wrench className="w-3 h-3 text-sky-400" />
              <span>Lint</span>
            </button>

            {/* Clang-Format Formatter Button */}
            <button
              onClick={handleFormat}
              disabled={isFormatting}
              title="Format C++ / CUDA code with clang-format (Shift+Alt+F / ⇧⌥F)"
              className="px-2.5 py-1 rounded bg-[#141b2b] hover:bg-[#1e283d] text-slate-200 hover:text-white text-[11px] font-medium border border-[#1e293b] flex items-center gap-1.5 transition-all shadow-sm cursor-pointer"
            >
              {isFormatting ? (
                <Loader2 className="w-3 h-3 animate-spin text-sky-400" />
              ) : formatSuccess ? (
                <Check className="w-3 h-3 text-emerald-400" />
              ) : (
                <Sparkles className="w-3 h-3 text-amber-400" />
              )}
              <span>{formatSuccess ? 'Formatted' : 'Format'}</span>
              <kbd className="text-[9px] font-mono text-slate-400 bg-[#070a10] px-1 rounded border border-[#1e293b]">
                ⇧⌥F
              </kbd>
            </button>

            {/* Toggle Panel Button (Always visible) */}
            <button
              onClick={() => setIsDockOpen((prev) => !prev)}
              title={isDockOpen ? 'Collapse Terminal & Problems panel' : 'Expand Terminal & Problems panel'}
              className={`px-2.5 py-1 rounded text-[11px] font-medium border flex items-center gap-1.5 transition-all cursor-pointer ${
                isDockOpen
                  ? 'bg-sky-500/10 text-sky-300 border-sky-500/30'
                  : 'bg-[#141b2b] hover:bg-[#1e283d] text-slate-300 hover:text-white border-[#1e293b]'
              }`}
            >
              <SlidersHorizontal className="w-3 h-3 text-sky-400" />
              <span>{isDockOpen ? 'Hide Panel' : 'Show Panel'}</span>
              {isDockOpen ? <ChevronDown className="w-3 h-3" /> : <ChevronUp className="w-3 h-3" />}
            </button>

            {/* Live Disk Save Status Indicator */}
            {saveStatus === 'saving' && (
              <span className="flex items-center gap-1 text-[11px] text-sky-400 font-medium">
                <Loader2 className="w-3 h-3 animate-spin" /> Saving...
              </span>
            )}
            {saveStatus === 'saved' && (
              <span className="flex items-center gap-1 text-[11px] text-emerald-400 font-medium">
                <Check className="w-3.5 h-3.5" /> Saved
              </span>
            )}
            {saveStatus === 'unsaved' && (
              <span className="flex items-center gap-1 text-[11px] text-amber-400 font-medium">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" /> Unsaved
              </span>
            )}

            {/* Direct Save Button */}
            {onSave && (
              <button
                onClick={onSave}
                title="Save directly to local file on disk (Cmd+S)"
                className="px-2.5 py-1 rounded bg-[#162032] hover:bg-[#1f2d47] text-slate-200 hover:text-white text-[11px] font-medium border border-[#1e293b] flex items-center gap-1.5 transition-all shadow-sm cursor-pointer"
              >
                <Save className="w-3 h-3 text-sky-400" />
                <span>Save</span>
                <kbd className="text-[9px] font-mono text-slate-400 bg-[#070a10] px-1 rounded border border-[#1e293b]">
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

      {/* Lower Dock Pane: Collapsible Terminal vs Problems Tabs */}
      {isDockOpen ? (
        <div className="h-64 min-h-48 bg-[#05070c] flex flex-col border-t border-[#1e293b]">
          {/* Tab Header */}
          <div className="h-8 px-4 bg-[#0b0f17] border-b border-[#1e293b] flex items-center justify-between select-none">
            <div className="flex items-center gap-1">
              {/* Terminal Tab */}
              <button
                onClick={() => setActiveBottomTab('terminal')}
                className={`px-3 py-1 rounded text-xs font-semibold flex items-center gap-1.5 transition-all cursor-pointer ${
                  activeBottomTab === 'terminal'
                    ? 'bg-[#162032] text-sky-300 border border-[#1e293b]'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Terminal className="w-3.5 h-3.5 text-sky-400" />
                <span>Terminal & Test Runner</span>
              </button>

              {/* Problems & Lint Diagnostics Tab */}
              <button
                onClick={() => setActiveBottomTab('problems')}
                className={`px-3 py-1 rounded text-xs font-semibold flex items-center gap-1.5 transition-all cursor-pointer ${
                  activeBottomTab === 'problems'
                    ? 'bg-[#162032] text-slate-100 border border-[#1e293b]'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {errorCount > 0 ? (
                  <AlertCircle className="w-3.5 h-3.5 text-rose-400" />
                ) : warningCount > 0 ? (
                  <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
                ) : (
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                )}
                <span>Problems</span>
                <span
                  className={`text-[10px] px-1.5 py-0.2 rounded-full font-mono font-bold ${
                    errorCount > 0
                      ? 'bg-rose-500/20 text-rose-300'
                      : warningCount > 0
                      ? 'bg-amber-500/20 text-amber-300'
                      : 'bg-emerald-500/20 text-emerald-300'
                  }`}
                >
                  {diagnostics.length}
                </span>
              </button>
            </div>

            {/* Right Header Status + Collapse Toggle Button */}
            <div className="flex items-center gap-3">
              {activeBottomTab === 'terminal' && (
                <>
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
                </>
              )}

              {activeBottomTab === 'problems' && (
                <span className="text-[11px] text-slate-400 font-mono">
                  {errorCount} error{errorCount !== 1 ? 's' : ''}, {warningCount} warning
                  {warningCount !== 1 ? 's' : ''}
                </span>
              )}

              {/* Panel Collapse Toggle Button */}
              <button
                onClick={() => setIsDockOpen(false)}
                title="Collapse Panel (Maximize Editor)"
                className="p-1 rounded hover:bg-[#1a2333] text-slate-400 hover:text-slate-200 transition-colors cursor-pointer"
              >
                <ChevronDown className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Tab Body */}
          {activeBottomTab === 'terminal' ? (
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
                  Click ▶ Run Test Suite or ▶ Run Playground above to compile with clang++ -std=c++20
                  -O3 and view live output.
                </span>
              )}
            </div>
          ) : (
            <div className="flex-1 p-3 font-mono text-xs overflow-y-auto">
              {diagnostics.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-slate-500 gap-2">
                  <CheckCircle2 className="w-7 h-7 text-emerald-500/60" />
                  <span className="text-slate-300 font-medium">No problems detected in this file</span>
                  <span className="text-[11px] text-slate-500">
                    clang++ -fsyntax-only validated C++20 compliance without errors or warnings.
                  </span>
                </div>
              ) : (
                <div className="space-y-1">
                  {diagnostics.map((diag, idx) => (
                    <div
                      key={idx}
                      onClick={() => jumpToProblem(diag)}
                      className={`flex items-start gap-2.5 p-2 rounded border cursor-pointer transition-all ${
                        diag.severity === 'error'
                          ? 'bg-rose-950/20 hover:bg-rose-950/40 border-rose-900/40 text-rose-200'
                          : 'bg-amber-950/20 hover:bg-amber-950/40 border-amber-900/40 text-amber-200'
                      }`}
                    >
                      {diag.severity === 'error' ? (
                        <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                      ) : (
                        <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                      )}
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <span className="font-semibold text-slate-100">
                            {diag.severity.toUpperCase()}
                          </span>
                          <span className="text-sky-400 bg-sky-950/50 px-1.5 py-0.2 rounded border border-sky-800/40 text-[10px]">
                            Line {diag.line}:{diag.col}
                          </span>
                        </div>
                        <p className="text-slate-300 mt-0.5">{diag.message}</p>
                      </div>
                      <span className="text-[10px] text-slate-500 self-center">Jump ➔</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      ) : (
        /* Sleek Collapsed Status Bar with One-Click Restore */
        <div className="h-7 px-4 bg-[#0b0f17] border-t border-[#1e293b] flex items-center justify-between text-xs select-none">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setIsDockOpen(true)}
              className="flex items-center gap-1.5 text-sky-400 hover:text-sky-300 font-medium transition-colors cursor-pointer"
              title="Open Terminal & Problems Panel"
            >
              <ChevronUp className="w-3.5 h-3.5" />
              <span>Show Panel</span>
            </button>

            <span className="text-slate-600">|</span>

            {/* Quick Diagnostics in Collapsed Bar */}
            <button
              onClick={() => {
                setIsDockOpen(true);
                setActiveBottomTab('problems');
              }}
              className="flex items-center gap-1.5 text-[11px] text-slate-400 hover:text-slate-200 cursor-pointer"
              title="Click to view syntax diagnostics"
            >
              {errorCount > 0 ? (
                <span className="text-rose-400 flex items-center gap-1 font-medium">
                  <AlertCircle className="w-3 h-3 text-rose-400" /> {errorCount} error{errorCount !== 1 ? 's' : ''}
                </span>
              ) : (
                <span className="text-emerald-400 flex items-center gap-1 font-medium">
                  <CheckCircle2 className="w-3 h-3 text-emerald-400" /> Syntax Clean
                </span>
              )}
            </button>
          </div>

          <div className="flex items-center gap-3 text-[11px] text-slate-400 font-mono">
            <button
              onClick={handleFormat}
              disabled={isFormatting}
              className="flex items-center gap-1 text-slate-300 hover:text-amber-300 transition-colors cursor-pointer"
              title="Format code (Shift+Alt+F)"
            >
              <Sparkles className="w-3 h-3 text-amber-400" />
              <span>{isFormatting ? 'Formatting...' : 'Format'}</span>
            </button>
            <span className="text-slate-600">•</span>
            <span>C++20 / CUDA</span>
          </div>
        </div>
      )}
    </div>
  );
}
