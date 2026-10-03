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
  Share2,
  PanelLeftClose,
  PanelLeftOpen,
  Compass,
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
  onForkToSandbox?: () => void;
  onSwitchNode: (node: ActiveNode) => void;
  onToggleTeacher: () => void;
  isTeacherOpen: boolean;
  isSidebarCollapsed?: boolean;
  onToggleSidebar?: () => void;
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
  onForkToSandbox,
  onSwitchNode,
  onToggleTeacher,
  isTeacherOpen,
  isSidebarCollapsed,
  onToggleSidebar,
}: HeaderBarProps) {
  // Breadcrumb generator
  const getBreadcrumbs = () => {
    if (activeNode.type === 'guide') {
      return (
        <div className="flex items-center gap-1.5 text-xs text-slate-400">
          <Compass className="w-3.5 h-3.5 text-sky-400" />
          <span className="text-slate-300">Orientation</span>
          <span className="text-slate-600">/</span>
          <span className="text-slate-100 font-medium">How to Read & Master This Curriculum</span>
        </div>
      );
    }

    if (activeNode.type === 'playground') {
      const pgType = activeNode.playgroundType || 'cpp';
      const pgNames = {
        cpp: { name: 'C++ Systems Sandbox', file: 'playground.cpp', color: 'text-sky-400' },
        cuda: { name: 'CUDA GPU Scratchpad', file: 'playground.cu', color: 'text-indigo-400' },
        kernel: { name: 'Kernel Benchmark Lab', file: 'playground_kernel.cu', color: 'text-amber-400' },
      };
      const info = pgNames[pgType] || pgNames.cpp;

      return (
        <div className="flex items-center gap-1.5 text-xs text-slate-400">
          <Terminal className={`w-3.5 h-3.5 ${info.color}`} />
          <span>Sandbox</span>
          <span className="text-slate-600">/</span>
          <span className="text-slate-300 font-medium">{info.name}</span>
          <span className="text-slate-600">/</span>
          <span className="text-slate-200 font-mono font-medium">{info.file}</span>
        </div>
      );
    }

    const isVol3 = activeNode.volumeId === 'vol3' || activeNode.chapterId?.startsWith('k') || activeNode.type === 'kernel';
    const isVol2 = !isVol3 && (activeNode.volumeId === 'vol2' || /^[2-7]\./.test(activeNode.chapterId || ''));

    if (isVol3) {
      const cleanId = (activeNode.chapterId || 'k1.1').replace(/^k/, '');
      const [modNum, subNum] = cleanId.split('.');
      return (
        <div className="flex items-center gap-1.5 text-xs text-slate-400">
          <Sparkles className="w-3.5 h-3.5 text-amber-400" />
          <span>Volume 3: Kernels</span>
          <span className="text-slate-600">/</span>
          <span className="text-slate-300 font-mono">Module {modNum} • Kernel {modNum}.{subNum}</span>
          <span className="text-slate-600">/</span>
          <span className="text-slate-100 font-medium truncate max-w-xs">
            {activeNode.type === 'theory' && '📖 Theory & Mental Models'}
            {activeNode.type === 'cheat_sheet' && '⚡ Revision Cheat Sheet'}
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
    }

    const rawModNum = activeNode.chapterId ? parseInt(activeNode.chapterId.split('.')[0], 10) : 2;
    const cudaModNum = isVol2 && rawModNum >= 2 ? rawModNum - 1 : rawModNum;
    const subNum = activeNode.chapterId ? activeNode.chapterId.split('.')[1] : '1';
    const displayTopicId = isVol2 && rawModNum >= 2 ? `${cudaModNum}.${subNum}` : activeNode.chapterId;

    return (
      <div className="flex items-center gap-1.5 text-xs text-slate-400">
        {isVol2 ? (
          <Cpu className="w-3.5 h-3.5 text-indigo-400" />
        ) : (
          <Layers className="w-3.5 h-3.5 text-sky-400" />
        )}
        <span>{isVol2 ? 'Volume 2: CUDA' : 'Volume 1: C++'}</span>
        <span className="text-slate-600">/</span>
        <span className="text-slate-300 font-mono">
          {isVol2 ? `Module ${cudaModNum} • Topic ${displayTopicId}` : `Ch ${activeNode.chapterId}`}
        </span>
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

  const targetVol =
    activeNode.volumeId ||
    (activeNode.chapterId?.startsWith('k') ? 'vol3' : /^[2-7]\./.test(activeNode.chapterId || '') ? 'vol2' : 'vol1');

  return (
    <header className="h-14 min-h-14 bg-[#131720] border-b border-[#21262d] px-4 flex items-center justify-between gap-4 select-none">
      {/* Left: Sidebar Toggle & Breadcrumb Navigation */}
      <div className="flex items-center gap-2.5">
        {onToggleSidebar && isSidebarCollapsed && (
          <button
            onClick={onToggleSidebar}
            title="Show Navigation Sidebar (Cmd+B)"
            className="p-1.5 rounded-md hover:bg-[#162032] text-slate-400 hover:text-slate-200 transition-colors cursor-pointer"
          >
            <PanelLeftOpen className="w-4 h-4 text-sky-400" />
          </button>
        )}
        {getBreadcrumbs()}
      </div>

      {/* Center: Context Badges or Target Switches */}
      <div className="flex items-center gap-3">
        {activeNode.type === 'theory' && (
          <div className="flex items-center gap-2">
            <span className="text-xs px-2.5 py-1 rounded-full bg-sky-500/10 text-sky-400 border border-sky-500/20 font-medium flex items-center gap-1.5">
              <BookOpen className="w-3 h-3" />
              Reading Mode • Systems & Architecture
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
              Solution (Reference ✅)
            </button>
            <button
              onClick={() => onTargetChange('exercise')}
              className={`px-3 py-1 rounded-md text-xs font-medium transition-all ${
                target === 'exercise'
                  ? 'bg-[#1e293b] text-sky-400 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Exercise (Clean Starter ✏️)
            </button>
          </div>
        )}

        {activeNode.type === 'playground' && (
          <span className="text-xs px-2.5 py-1 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 font-mono">
            {activeNode.playgroundType === 'cuda'
              ? 'nvcc -O3 -std=c++17 --extended-lambda'
              : activeNode.playgroundType === 'kernel'
              ? 'nvcc -O3 -std=c++17 (Kernel Benchmark Lab)'
              : 'clang++ -std=c++20 -O3 (C++ Memory Sandbox)'}
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
                  volumeId: targetVol,
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
                  volumeId: targetVol,
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
                  volumeId: targetVol,
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
                  volumeId: targetVol,
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
            {onForkToSandbox && (
              <button
                onClick={onForkToSandbox}
                title="Fork code directly into your Root Sandbox Playground"
                className="px-2.5 py-1.5 rounded-md bg-[#121929] hover:bg-[#18233a] text-xs font-medium text-sky-300 border border-[#1e293b] hover:border-sky-500/40 flex items-center gap-1.5 transition-all shadow-sm cursor-pointer"
              >
                <Share2 className="w-3.5 h-3.5 text-sky-400" />
                <span>Fork to Sandbox</span>
              </button>
            )}
            {onStartOver && (
              <button
                onClick={onStartOver}
                title="Reset code to original exercise starter template"
                className="px-2.5 py-1.5 rounded-md bg-[#161d2d] hover:bg-rose-950/40 text-xs font-medium text-slate-300 hover:text-rose-300 border border-[#1e293b] hover:border-rose-500/40 flex items-center gap-1.5 transition-all shadow-sm cursor-pointer"
              >
                <RotateCcw className="w-3.5 h-3.5 text-slate-400" />
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
              {isRunning ? 'Running in Sandbox...' : '▶ Run in Sandbox'}
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
              {isRunning ? 'Running in Sandbox...' : '▶ Run Sandbox'}
            </button>
          </>
        )}

        {/* Orientation Guide Button */}
        <button
          onClick={() => onSwitchNode({ type: 'guide', volumeId: activeNode.volumeId || 'vol1' })}
          title="How to Read This Book & Curriculum Orientation"
          className={`px-2.5 py-1.5 rounded-md text-xs font-medium border flex items-center gap-1.5 transition-all cursor-pointer ${
            activeNode.type === 'guide'
              ? 'bg-[#1f6feb]/20 text-[#58a6ff] border-[#1f6feb]/40 shadow-xs'
              : 'bg-[#161b24] hover:bg-[#21262d] text-slate-300 hover:text-white border-[#30363d]'
          }`}
        >
          <Compass className="w-3.5 h-3.5 text-sky-400" />
          <span className="hidden sm:inline">Guide</span>
        </button>

        {/* Antigravity Teacher Toggle */}
        <button
          onClick={onToggleTeacher}
          className={`px-3 py-1.5 rounded-md text-xs font-medium border flex items-center gap-1.5 transition-all cursor-pointer ${
            isTeacherOpen
              ? 'bg-indigo-600/30 text-indigo-300 border-indigo-500/50 shadow-sm shadow-indigo-500/20'
              : 'bg-[#161b24] hover:bg-[#21262d] text-slate-300 hover:text-white border-[#30363d]'
          }`}
        >
          <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
          <span>Ask Teacher</span>
        </button>
      </div>
    </header>
  );
}
