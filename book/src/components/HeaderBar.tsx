'use client';

import React from 'react';
import {
  Play,
  Save,
  RotateCcw,
  Sparkles,
  BookOpen,
  Zap,
  Terminal,
  Cpu,
  Layers,
  CheckCircle2,
  Trash2,
} from 'lucide-react';
import { ActiveNode } from './SidebarTree';

interface HeaderBarProps {
  activeNode: ActiveNode;
  nodeData: any;
  target: 'solution' | 'exercise';
  onTargetChange: (target: 'solution' | 'exercise') => void;
  isRunning: boolean;
  onRun: () => void;
  onSave: () => void;
  onReset?: () => void;
  onStartOver?: () => void;
  onClearConsole?: () => void;
  onFormat?: () => void;
  onSwitchNode: (node: ActiveNode) => void;
  onToggleTeacher: () => void;
  isTeacherOpen: boolean;
}

export function HeaderBar({
  activeNode,
  nodeData,
  target,
  onTargetChange,
  isRunning,
  onRun,
  onSave,
  onReset,
  onStartOver,
  onClearConsole,
  onFormat,
  onSwitchNode,
  onToggleTeacher,
  isTeacherOpen,
}: HeaderBarProps) {
  // Breadcrumb generator
  const getBreadcrumbs = () => {
    if (activeNode.type === 'playground') {
      return (
        <div className="flex items-center gap-1.5 text-xs text-slate-400">
          <Terminal className="w-3.5 h-3.5 text-sky-400" />
          <span>Sandbox</span>
          <span className="text-slate-600">/</span>
          <span className="text-slate-200 font-mono font-medium">playground.cpp</span>
        </div>
      );
    }

    if (activeNode.type === 'kernel') {
      return (
        <div className="flex items-center gap-1.5 text-xs text-slate-400">
          <Sparkles className="w-3.5 h-3.5 text-amber-400" />
          <span>Volume 3</span>
          <span className="text-slate-600">/</span>
          <span className="text-slate-200 font-medium">{nodeData?.title || 'Kernel Blueprint'}</span>
        </div>
      );
    }

    return (
      <div className="flex items-center gap-1.5 text-xs text-slate-400">
        <Layers className="w-3.5 h-3.5 text-sky-400" />
        <span>Volume 1</span>
        <span className="text-slate-600">/</span>
        <span className="text-slate-300 font-mono">Ch {activeNode.chapterId}</span>
        <span className="text-slate-600">/</span>
        <span className="text-slate-100 font-medium">
          {activeNode.type === 'theory' && '📖 Theory & Models'}
          {activeNode.type === 'cheat_sheet' && '⚡ Cheat Sheet'}
          {activeNode.type === 'workbook' && (
            <span className="capitalize">
              {activeNode.tier === 'beginner' && '🟢 Beginner Workbook'}
              {activeNode.tier === 'intermediate' && '🟡 Intermediate Workbook'}
              {activeNode.tier === 'champion' && '🔴 Champion Workbook'}
            </span>
          )}
        </span>
      </div>
    );
  };

  return (
    <header className="h-14 min-h-14 bg-[#0b1019] border-b border-[#1e293b] px-5 flex items-center justify-between gap-4 select-none">
      {/* Left: Breadcrumb Navigation */}
      <div className="flex items-center gap-3">
        {getBreadcrumbs()}
      </div>

      {/* Center: Context Badges or Target Switches */}
      <div className="flex items-center gap-3">
        {activeNode.type === 'theory' && (
          <div className="flex items-center gap-2">
            <span className="text-xs px-2.5 py-1 rounded-full bg-sky-500/10 text-sky-400 border border-sky-500/20 font-medium flex items-center gap-1.5">
              <BookOpen className="w-3 h-3" />
              Reading Mode • Zero LaTeX Pure Models
            </span>
          </div>
        )}

        {activeNode.type === 'cheat_sheet' && (
          <div className="flex items-center gap-2">
            <span className="text-xs px-2.5 py-1 rounded-full bg-amber-500/10 text-amber-300 border border-amber-500/20 font-medium flex items-center gap-1.5">
              <Zap className="w-3 h-3" />
              Quick Revision • Hardware Formulae
            </span>
          </div>
        )}

        {activeNode.type === 'workbook' && (
          <div className="flex items-center bg-[#111726] p-0.5 rounded-lg border border-[#1e293b]">
            <button
              onClick={() => onTargetChange('solution')}
              className={`px-3 py-1 rounded-md text-xs font-medium transition-all ${
                target === 'solution'
                  ? 'bg-[#1e293b] text-emerald-400 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Solution (Passed ✅)
            </button>
            <button
              onClick={() => onTargetChange('exercise')}
              className={`px-3 py-1 rounded-md text-xs font-medium transition-all ${
                target === 'exercise'
                  ? 'bg-[#1e293b] text-sky-400 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Exercise (WIP ✏️)
            </button>
          </div>
        )}

        {activeNode.type === 'playground' && (
          <span className="text-xs px-2.5 py-1 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 font-mono">
            clang++ -std=c++20 -O3
          </span>
        )}

        {activeNode.type === 'kernel' && (
          <span className="text-xs px-2.5 py-1 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20 font-mono">
            Apple Metal ➔ CUDA C++20 Spec
          </span>
        )}
      </div>

      {/* Right: Context-Segregated Action Buttons */}
      <div className="flex items-center gap-2">
        {/* Theory Mode Actions */}
        {activeNode.type === 'theory' && (
          <>
            <button
              onClick={() =>
                onSwitchNode({
                  type: 'cheat_sheet',
                  volumeId: 'vol1',
                  chapterId: activeNode.chapterId,
                })
              }
              className="px-3 py-1.5 rounded-md bg-[#141d2e] hover:bg-[#1a263c] text-xs font-medium text-amber-300 border border-[#1e293b] flex items-center gap-1.5 transition-colors"
            >
              <Zap className="w-3.5 h-3.5 text-amber-400" />
              Cheat Sheet
            </button>
            <button
              onClick={() =>
                onSwitchNode({
                  type: 'workbook',
                  volumeId: 'vol1',
                  chapterId: activeNode.chapterId,
                  tier: 'champion',
                })
              }
              className="px-3 py-1.5 rounded-md bg-emerald-600/20 hover:bg-emerald-600/30 text-xs font-medium text-emerald-300 border border-emerald-500/30 flex items-center gap-1.5 transition-colors"
            >
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
              Open Lab Workbook
            </button>
          </>
        )}

        {/* Cheat Sheet Mode Actions */}
        {activeNode.type === 'cheat_sheet' && (
          <>
            <button
              onClick={() =>
                onSwitchNode({
                  type: 'theory',
                  volumeId: 'vol1',
                  chapterId: activeNode.chapterId,
                })
              }
              className="px-3 py-1.5 rounded-md bg-[#141d2e] hover:bg-[#1a263c] text-xs font-medium text-sky-300 border border-[#1e293b] flex items-center gap-1.5 transition-colors"
            >
              <BookOpen className="w-3.5 h-3.5 text-sky-400" />
              Full Theory
            </button>
            <button
              onClick={() =>
                onSwitchNode({
                  type: 'workbook',
                  volumeId: 'vol1',
                  chapterId: activeNode.chapterId,
                  tier: 'champion',
                })
              }
              className="px-3 py-1.5 rounded-md bg-emerald-600/20 hover:bg-emerald-600/30 text-xs font-medium text-emerald-300 border border-emerald-500/30 flex items-center gap-1.5 transition-colors"
            >
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
              Test Workbook
            </button>
          </>
        )}

        {/* Workbook Mode Actions */}
        {activeNode.type === 'workbook' && (
          <>
            {onStartOver && (
              <button
                onClick={onStartOver}
                title="Reset code to original exercise starter template"
                className="px-2.5 py-1.5 rounded-md bg-[#161d2d] hover:bg-rose-950/40 text-xs font-medium text-slate-300 hover:text-rose-300 border border-[#1e293b] hover:border-rose-500/40 flex items-center gap-1.5 transition-all shadow-sm cursor-pointer"
              >
                <RotateCcw className="w-3.5 h-3.5 text-slate-400 group-hover:text-rose-400" />
                <span>Start Over</span>
              </button>
            )}
            {onReset && (
              <button
                onClick={onReset}
                title="Reload from disk"
                className="p-1.5 rounded-md bg-[#141d2e] hover:bg-[#1a263c] text-slate-400 hover:text-slate-200 border border-[#1e293b] transition-colors cursor-pointer"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
            )}
            {onFormat && (
              <button
                onClick={onFormat}
                title="Format code with clang-format (Shift+Alt+F)"
                className="px-2.5 py-1.5 rounded-md bg-[#141d2e] hover:bg-[#1a263c] text-xs font-medium text-slate-200 border border-[#1e293b] flex items-center gap-1.5 transition-colors cursor-pointer"
              >
                <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                <span>Format</span>
              </button>
            )}
            <button
              onClick={onSave}
              className="px-3 py-1.5 rounded-md bg-[#141d2e] hover:bg-[#1a263c] text-xs font-medium text-slate-200 border border-[#1e293b] flex items-center gap-1.5 transition-colors cursor-pointer"
            >
              <Save className="w-3.5 h-3.5 text-slate-400" />
              Save
            </button>
            <button
              onClick={onRun}
              disabled={isRunning}
              className={`px-3.5 py-1.5 rounded-md text-xs font-semibold flex items-center gap-1.5 transition-all shadow-md cursor-pointer ${
                isRunning
                  ? 'bg-emerald-700/50 text-emerald-200 cursor-not-allowed'
                  : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-900/30'
              }`}
            >
              <Play className={`w-3.5 h-3.5 ${isRunning ? 'animate-spin' : ''}`} />
              {isRunning ? 'Running Tests...' : '▶ Run Test Suite'}
            </button>
          </>
        )}

        {/* Playground Mode Actions */}
        {activeNode.type === 'playground' && (
          <>
            {onClearConsole && (
              <button
                onClick={onClearConsole}
                title="Clear Terminal"
                className="p-1.5 rounded-md bg-[#141d2e] hover:bg-[#1a263c] text-slate-400 hover:text-slate-200 border border-[#1e293b] transition-colors"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            )}
            {onFormat && (
              <button
                onClick={onFormat}
                title="Format code with clang-format (Shift+Alt+F)"
                className="px-2.5 py-1.5 rounded-md bg-[#141d2e] hover:bg-[#1a263c] text-xs font-medium text-slate-200 border border-[#1e293b] flex items-center gap-1.5 transition-colors cursor-pointer"
              >
                <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                <span>Format</span>
              </button>
            )}
            <button
              onClick={onSave}
              className="px-3 py-1.5 rounded-md bg-[#141d2e] hover:bg-[#1a263c] text-xs font-medium text-slate-200 border border-[#1e293b] flex items-center gap-1.5 transition-colors cursor-pointer"
            >
              <Save className="w-3.5 h-3.5 text-slate-400" />
              Save
            </button>
            <button
              onClick={onRun}
              disabled={isRunning}
              className={`px-3.5 py-1.5 rounded-md text-xs font-semibold flex items-center gap-1.5 transition-all shadow-md ${
                isRunning
                  ? 'bg-sky-700/50 text-sky-200 cursor-not-allowed'
                  : 'bg-sky-600 hover:bg-sky-500 text-white shadow-sky-900/30'
              }`}
            >
              <Play className={`w-3.5 h-3.5 ${isRunning ? 'animate-spin' : ''}`} />
              {isRunning ? 'Compiling...' : '▶ Run Playground'}
            </button>
          </>
        )}

        {/* Antigravity Teacher Toggle (Available in all views) */}
        <button
          onClick={onToggleTeacher}
          className={`px-3 py-1.5 rounded-md text-xs font-medium border flex items-center gap-1.5 transition-all ${
            isTeacherOpen
              ? 'bg-indigo-600/30 text-indigo-300 border-indigo-500/50 shadow-sm shadow-indigo-500/20'
              : 'bg-[#141d2e] hover:bg-[#1a263c] text-slate-300 border-[#1e293b]'
          }`}
        >
          <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
          <span>Ask Teacher</span>
        </button>
      </div>
    </header>
  );
}
