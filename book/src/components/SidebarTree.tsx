'use client';

import React, { useState, useEffect, useMemo, useRef } from 'react';
import {
  BookOpen,
  Zap,
  ChevronDown,
  ChevronRight,
  Terminal,
  Cpu,
  Layers,
  Sparkles,
  Flame,
  Search,
  Check,
  PanelLeftClose,
  PanelLeftOpen,
  Compass,
  Bookmark,
} from 'lucide-react';
import { getBookmarks } from '@/lib/annotations';
import {
  ChapterMeta,
  CudaModuleMeta,
  CudaTopicMeta,
  CppModuleMeta,
  KernelModuleMeta,
  KernelTopicMeta,
  PlaygroundType,
} from '@/lib/workspace';

export type ActiveNodeType = 'theory' | 'cheat_sheet' | 'workbook' | 'playground' | 'kernel' | 'guide';

export interface ActiveNode {
  type: ActiveNodeType;
  volumeId: string;
  chapterId?: string;
  tier?: 'beginner' | 'intermediate' | 'champion';
  kernelId?: number;
  playgroundType?: PlaygroundType;
}

interface SidebarTreeProps {
  treeData: any;
  activeNode: ActiveNode;
  onSelectNode: (node: ActiveNode) => void;
  onRefreshTree?: () => void;
  width?: number;
  onResize?: (width: number) => void;
  isCollapsed?: boolean;
  onToggleCollapse?: () => void;
}

export function SidebarTree({
  treeData,
  activeNode,
  onSelectNode,
  onRefreshTree,
  width = 290,
  onResize,
  isCollapsed = false,
  onToggleCollapse,
}: SidebarTreeProps) {
  // Volume Tab selector: 'vol1' | 'vol2' | 'vol3'
  const [activeVolume, setActiveVolume] = useState<'vol1' | 'vol2' | 'vol3'>('vol1');
  const [searchQuery, setSearchQuery] = useState('');

  // Volume 1 Accordions: Module & Chapter
  const [openCppModuleId, setOpenCppModuleId] = useState<string>('1');
  const [openChapterId, setOpenChapterId] = useState<string>('1.1');

  // Volume 2 Accordions: Module & Topic
  const [openCudaModuleId, setOpenCudaModuleId] = useState<string>('3');
  const [openCudaTopicId, setOpenCudaTopicId] = useState<string>('3.1');

  // Volume 3 Accordions: Module & Topic
  const [openKernelModuleId, setOpenKernelModuleId] = useState<string>('1');
  const [openKernelTopicId, setOpenKernelTopicId] = useState<string>('k1.1');

  // Multi-Playground Menu Dropdown
  const [isPlaygroundMenuOpen, setIsPlaygroundMenuOpen] = useState(false);
  const playgroundDropdownRef = useRef<HTMLDivElement>(null);

  // Bookmarks Tracking
  const [bookmarkedIds, setBookmarkedIds] = useState<Set<string>>(new Set());

  useEffect(() => {
    const syncBookmarks = () => {
      const list = getBookmarks();
      setBookmarkedIds(new Set(list.map((b) => b.chapterId)));
    };
    syncBookmarks();
    window.addEventListener('cuda-annotations-updated', syncBookmarks);
    return () => window.removeEventListener('cuda-annotations-updated', syncBookmarks);
  }, []);

  // Drag Resizing Logic
  const isResizingRef = useRef(false);

  const startResizing = (e: React.MouseEvent) => {
    if (!onResize) return;
    e.preventDefault();
    isResizingRef.current = true;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';

    const handleMouseMove = (ev: MouseEvent) => {
      if (!isResizingRef.current) return;
      const newWidth = Math.max(220, Math.min(520, ev.clientX));
      onResize(newWidth);
    };

    const handleMouseUp = () => {
      isResizingRef.current = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      try {
        localStorage.setItem('sidebar_width', String(width));
      } catch {}
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  };

  // Sync open module/topic when activeNode changes
  useEffect(() => {
    if (activeNode.chapterId) {
      if (activeNode.volumeId === 'vol3' || activeNode.chapterId.startsWith('k')) {
        setActiveVolume('vol3');
        const m = activeNode.chapterId.replace(/^k/, '').split('.')[0];
        setOpenKernelModuleId(m);
        setOpenKernelTopicId(activeNode.chapterId);
      } else if (activeNode.volumeId === 'vol2' || /^[2-7]\./.test(activeNode.chapterId)) {
        setActiveVolume('vol2');
        const modNum = activeNode.chapterId.split('.')[0];
        setOpenCudaModuleId(modNum);
        setOpenCudaTopicId(activeNode.chapterId);
      } else {
        setActiveVolume('vol1');
        const chNum = parseInt(activeNode.chapterId.split('.')[1] || '1', 10);
        let m = '1';
        if (chNum <= 5) m = '1';
        else if (chNum <= 8) m = '2';
        else if (chNum <= 12) m = '3';
        else if (chNum <= 15) m = '4';
        else m = '5';
        setOpenCppModuleId(m);
        setOpenChapterId(activeNode.chapterId);
      }
    } else if (activeNode.type === 'kernel') {
      setActiveVolume('vol3');
    }
  }, [activeNode.chapterId, activeNode.volumeId, activeNode.type]);

  // Close playground menu on outside click
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (playgroundDropdownRef.current && !playgroundDropdownRef.current.contains(e.target as Node)) {
        setIsPlaygroundMenuOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const cppModules: CppModuleMeta[] = useMemo(() => treeData?.volume1?.modules || [], [treeData]);
  const cudaModules: CudaModuleMeta[] = useMemo(() => treeData?.volume2?.modules || [], [treeData]);
  const kernelModules: KernelModuleMeta[] = useMemo(() => treeData?.volume3?.modules || [], [treeData]);

  // Filter Volume 1 C++ Modules
  const filteredCppModules = useMemo(() => {
    if (!searchQuery.trim()) return cppModules;
    const q = searchQuery.toLowerCase();
    return cppModules
      .map((mod) => {
        const matchesModule =
          mod.title.toLowerCase().includes(q) ||
          mod.displayTitle.toLowerCase().includes(q) ||
          `m${mod.displayNum}`.includes(q) ||
          mod.tagline?.toLowerCase().includes(q);
        const matchedTopics = mod.topics.filter(
          (t) => t.title.toLowerCase().includes(q) || t.id.includes(q)
        );
        if (matchesModule) return mod;
        if (matchedTopics.length > 0) return { ...mod, topics: matchedTopics };
        return null;
      })
      .filter(Boolean) as CppModuleMeta[];
  }, [cppModules, searchQuery]);

  // Filter Volume 2 CUDA Modules
  const filteredCudaModules = useMemo(() => {
    if (!searchQuery.trim()) return cudaModules;
    const q = searchQuery.toLowerCase();
    return cudaModules
      .map((mod) => {
        const matchesModule =
          mod.title.toLowerCase().includes(q) ||
          (mod.displayTitle && mod.displayTitle.toLowerCase().includes(q)) ||
          `m${mod.displayNum ?? (parseInt(mod.id) - 1)}`.includes(q) ||
          mod.tagline?.toLowerCase().includes(q);
        const matchedTopics = mod.topics.filter(
          (t) =>
            t.title.toLowerCase().includes(q) ||
            t.id.includes(q) ||
            (t.displayId && t.displayId.includes(q))
        );
        if (matchesModule) return mod;
        if (matchedTopics.length > 0) return { ...mod, topics: matchedTopics };
        return null;
      })
      .filter(Boolean) as CudaModuleMeta[];
  }, [cudaModules, searchQuery]);

  // Filter Volume 3 Kernel Modules
  const filteredKernelModules = useMemo(() => {
    if (!searchQuery.trim()) return kernelModules;
    const q = searchQuery.toLowerCase();
    return kernelModules
      .map((mod) => {
        const matchesModule =
          mod.title.toLowerCase().includes(q) ||
          mod.displayTitle.toLowerCase().includes(q) ||
          `m${mod.displayNum}`.includes(q) ||
          mod.tagline?.toLowerCase().includes(q);
        const matchedTopics = mod.topics.filter(
          (t) =>
            t.title.toLowerCase().includes(q) ||
            t.id.includes(q) ||
            (t.displayId && t.displayId.includes(q)) ||
            t.category?.toLowerCase().includes(q)
        );
        if (matchesModule) return mod;
        if (matchedTopics.length > 0) return { ...mod, topics: matchedTopics };
        return null;
      })
      .filter(Boolean) as KernelModuleMeta[];
  }, [kernelModules, searchQuery]);

  const toggleCppModule = (modId: string) => {
    setOpenCppModuleId((prev) => (prev === modId ? '' : modId));
  };

  const toggleChapter = (chId: string) => {
    setOpenChapterId((prev) => (prev === chId ? '' : chId));
  };

  const toggleCudaModule = (modId: string) => {
    setOpenCudaModuleId((prev) => (prev === modId ? '' : modId));
  };

  const toggleCudaTopic = (topicId: string) => {
    setOpenCudaTopicId((prev) => (prev === topicId ? '' : topicId));
  };

  const toggleKernelModule = (modId: string) => {
    setOpenKernelModuleId((prev) => (prev === modId ? '' : modId));
  };

  const toggleKernelTopic = (topicId: string) => {
    setOpenKernelTopicId((prev) => (prev === topicId ? '' : topicId));
  };

  const currentPlaygroundType = activeNode.playgroundType || 'cpp';

  const selectPlayground = (type: PlaygroundType) => {
    setIsPlaygroundMenuOpen(false);
    onSelectNode({
      type: 'playground',
      volumeId: 'sandbox',
      playgroundType: type,
    });
  };

  if (isCollapsed) {
    return (
      <div className="w-12 h-full bg-[#0a0d14] border-r border-[#1a2333] flex flex-col items-center py-3 select-none z-20">
        <button
          onClick={onToggleCollapse}
          title="Expand Navigation (Cmd+B)"
          className="p-2 rounded-md hover:bg-[#162032] text-slate-400 hover:text-slate-200 transition-colors"
        >
          <PanelLeftOpen className="w-4 h-4 text-sky-400" />
        </button>
        <div className="mt-4 flex-1 flex flex-col items-center gap-3">
          <button
            onClick={() => {
              onToggleCollapse?.();
              setActiveVolume('vol1');
            }}
            title="Volume 1: C++ Systems"
            className={`p-2 rounded-lg text-xs font-mono transition-all ${
              activeVolume === 'vol1' ? 'bg-[#162032] text-sky-300 font-bold' : 'text-slate-500 hover:text-slate-300'
            }`}
          >
            C++
          </button>
          <button
            onClick={() => {
              onToggleCollapse?.();
              setActiveVolume('vol2');
            }}
            title="Volume 2: CUDA"
            className={`p-2 rounded-lg text-xs font-mono transition-all ${
              activeVolume === 'vol2' ? 'bg-[#162032] text-indigo-300 font-bold' : 'text-slate-500 hover:text-slate-300'
            }`}
          >
            CU
          </button>
          <button
            onClick={() => {
              onToggleCollapse?.();
              setActiveVolume('vol3');
            }}
            title="Volume 3: Kernels"
            className={`p-2 rounded-lg text-xs font-mono transition-all ${
              activeVolume === 'vol3' ? 'bg-[#162032] text-amber-300 font-bold' : 'text-slate-500 hover:text-slate-300'
            }`}
          >
            KN
          </button>
        </div>
      </div>
    );
  }

  return (
    <aside
      style={{ width }}
      className="h-full bg-[#0a0d14] border-r border-[#1a2333] flex flex-col select-none relative shrink-0 transition-[width] duration-75"
    >
      {/* Draggable resize handle */}
      {onResize && (
        <div
          onMouseDown={startResizing}
          onDoubleClick={() => onResize(290)}
          className="absolute top-0 right-0 bottom-0 w-1.5 cursor-col-resize hover:bg-sky-500/50 active:bg-sky-500 transition-colors z-40 group"
          title="Drag to resize sidebar (Double click to reset)"
        >
          <div className="absolute top-1/2 -translate-y-1/2 right-0.5 w-0.5 h-8 rounded-full bg-slate-700/60 group-hover:bg-sky-400/80 transition-colors" />
        </div>
      )}

      {/* Brand & Sandbox Header */}
      <div className="p-3 border-b border-[#21262d] bg-[#131720]/95 backdrop-blur-md">
        <div className="flex items-center justify-between relative">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-md bg-[#1c2230] border border-[#30363d] flex items-center justify-center text-sky-400">
              <Cpu className="w-3.5 h-3.5" />
            </div>
            <div>
              <h1 className="text-xs font-bold text-[#f0f6fc] tracking-tight flex items-center gap-1.5">
                CUDA & Systems
              </h1>
            </div>
          </div>

          <div className="flex items-center gap-1">
            {/* Interactive Multi-Playground Dropdown Selector */}
            <div className="relative" ref={playgroundDropdownRef}>
              <div className="flex items-center rounded border border-[#30363d] bg-[#161b24] overflow-hidden">
                <button
                  onClick={() => selectPlayground(currentPlaygroundType)}
                  title="Open Active Sandbox Playground"
                  className={`px-2 py-0.5 text-xs flex items-center gap-1.5 transition-all cursor-pointer ${
                    activeNode.type === 'playground'
                      ? 'bg-sky-500/20 text-sky-300 font-medium'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-[#1c2230]'
                  }`}
                >
                  <Terminal className="w-3 h-3 text-sky-400" />
                  <span className="font-mono text-[10px] capitalize">
                    {activeNode.type === 'playground' ? currentPlaygroundType : 'Playground'}
                  </span>
                </button>
                <button
                  onClick={() => setIsPlaygroundMenuOpen((prev) => !prev)}
                  title="Choose Playground Type (C++, CUDA, Kernel)"
                  className="px-1 py-0.5 text-slate-400 hover:text-slate-200 hover:bg-[#1c2230] border-l border-[#30363d] transition-colors cursor-pointer"
                >
                  <ChevronDown className="w-2.5 h-2.5" />
                </button>
              </div>

              {/* Dropdown Menu */}
              {isPlaygroundMenuOpen && (
                <div className="absolute right-0 mt-1 w-56 bg-[#161b24] border border-[#30363d] rounded-lg shadow-xl shadow-black/60 z-50 py-1 overflow-hidden animate-in fade-in zoom-in-95 duration-100">
                  <div className="px-2.5 py-1 text-[10px] font-semibold text-slate-400 uppercase tracking-wider border-b border-[#21262d]">
                    Select Playground Environment
                  </div>

                  <button
                    onClick={() => selectPlayground('cpp')}
                    className={`w-full flex items-start gap-2 px-2.5 py-2 text-left hover:bg-[#1c2230] transition-colors cursor-pointer ${
                      activeNode.type === 'playground' && currentPlaygroundType === 'cpp'
                        ? 'bg-sky-500/10 text-sky-300'
                        : 'text-slate-300'
                    }`}
                  >
                    <span className="text-sky-400 font-bold text-xs mt-0.5">⚡</span>
                    <div>
                      <div className="text-xs font-medium">C++ Systems Sandbox</div>
                      <div className="text-[10px] text-slate-500 font-mono">playground.cpp (clang++ -O3)</div>
                    </div>
                  </button>

                  <button
                    onClick={() => selectPlayground('cuda')}
                    className={`w-full flex items-start gap-2 px-2.5 py-2 text-left hover:bg-[#1c2230] transition-colors cursor-pointer ${
                      activeNode.type === 'playground' && currentPlaygroundType === 'cuda'
                        ? 'bg-indigo-500/10 text-indigo-300'
                        : 'text-slate-300'
                    }`}
                  >
                    <span className="text-indigo-400 font-bold text-xs mt-0.5">🚀</span>
                    <div>
                      <div className="text-xs font-medium">CUDA GPU Scratchpad</div>
                      <div className="text-[10px] text-slate-500 font-mono">playground.cu (nvcc -O3)</div>
                    </div>
                  </button>

                  <button
                    onClick={() => selectPlayground('kernel')}
                    className={`w-full flex items-start gap-2 px-2.5 py-2 text-left hover:bg-[#1c2230] transition-colors cursor-pointer ${
                      activeNode.type === 'playground' && currentPlaygroundType === 'kernel'
                        ? 'bg-amber-500/10 text-amber-300'
                        : 'text-slate-300'
                    }`}
                  >
                    <span className="text-amber-400 font-bold text-xs mt-0.5">🔬</span>
                    <div>
                      <div className="text-xs font-medium">Kernel Benchmark Lab</div>
                      <div className="text-[10px] text-slate-500 font-mono">playground_kernel.cu (harness)</div>
                    </div>
                  </button>
                </div>
              )}
            </div>

            {/* Collapse Sidebar Button */}
            {onToggleCollapse && (
              <button
                onClick={onToggleCollapse}
                title="Collapse Sidebar (Cmd+B)"
                className="p-1 rounded text-slate-400 hover:text-slate-200 hover:bg-[#1c2230] transition-colors cursor-pointer"
              >
                <PanelLeftClose className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        </div>

        {/* Sleek Minimal Progress Tracker (Uncrowded, Refined) */}
        <div className="mt-2.5 px-0.5">
          <div className="flex items-center justify-between text-[10px] mb-1">
            <span className="text-slate-400 font-medium flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
              <span>Course Mastery</span>
            </span>
            <span className="font-mono text-slate-300 font-semibold">
              {treeData?.stats?.percent ?? 0}% <span className="text-slate-500 font-normal">({treeData?.stats?.testsPassed ?? 0}/{treeData?.stats?.totalTests ?? 0})</span>
            </span>
          </div>
          <div className="w-full h-1 bg-[#10141d] rounded-full overflow-hidden border border-[#21262d]">
            <div
              className="h-full bg-emerald-500 rounded-full transition-all duration-500"
              style={{ width: `${Math.max(1, treeData?.stats?.percent ?? 0)}%` }}
            />
          </div>
        </div>

        {/* Pinned Course Orientation Guide Button */}
        <button
          onClick={() => onSelectNode({ type: 'guide', volumeId: activeVolume })}
          className={`mt-2 w-full flex items-center justify-between px-2.5 py-1.5 rounded-md text-xs transition-all cursor-pointer ${
            activeNode.type === 'guide'
              ? 'bg-[#1f6feb]/20 text-[#58a6ff] border border-[#1f6feb]/40 font-semibold'
              : 'bg-[#161b24] hover:bg-[#1c2230] text-slate-300 hover:text-white border border-[#21262d]'
          }`}
        >
          <div className="flex items-center gap-2">
            <Compass className="w-3.5 h-3.5 text-sky-400" />
            <span className="font-medium text-[11px]">Orientation Guide</span>
          </div>
          <span className="text-[10px] text-slate-400 font-mono">How to Read</span>
        </button>

        {/* Category Tabs — C++ Systems, CUDA, GPU Kernels with distinct borders and prominent active highlight */}
        {(() => {
          const volDefs = [
            { id: 'vol1', label: 'C++ Systems', count: '20 Ch', accent: 'sky' },
            { id: 'vol2', label: 'CUDA', count: '21 Ch', accent: 'indigo' },
            { id: 'vol3', label: 'GPU Kernels', count: '28 Kernels', accent: 'amber' },
          ];

          const accentStyles: Record<string, { active: string; dot: string; indicator: string }> = {
            sky: {
              active: 'bg-[#10233b] border-sky-400 text-sky-200 shadow-[0_0_14px_rgba(56,189,248,0.3)]',
              dot: 'bg-sky-400 shadow-[0_0_8px_rgba(56,189,248,0.9)]',
              indicator: 'bg-sky-400',
            },
            indigo: {
              active: 'bg-[#1a183b] border-indigo-400 text-indigo-200 shadow-[0_0_14px_rgba(99,102,241,0.3)]',
              dot: 'bg-indigo-400 shadow-[0_0_8px_rgba(99,102,241,0.9)]',
              indicator: 'bg-indigo-400',
            },
            amber: {
              active: 'bg-[#2b1f0c] border-amber-400 text-amber-200 shadow-[0_0_14px_rgba(245,158,11,0.3)]',
              dot: 'bg-amber-400 shadow-[0_0_8px_rgba(245,158,11,0.9)]',
              indicator: 'bg-amber-400',
            },
          };

          return (
            <div className="mt-3 p-1 rounded-xl bg-[#090d16] border border-[#232b3d] grid grid-cols-3 gap-1.5 shadow-inner">
              {volDefs.map((vol) => {
                const isActive = activeVolume === vol.id;
                const styles = accentStyles[vol.accent];
                return (
                  <button
                    key={vol.id}
                    onClick={() => {
                      const nextVol = vol.id as 'vol1' | 'vol2' | 'vol3';
                      setActiveVolume(nextVol);
                      if (nextVol === 'vol1') {
                        onSelectNode({ type: 'theory', volumeId: 'vol1', chapterId: openChapterId || '1.1' });
                      } else if (nextVol === 'vol2') {
                        onSelectNode({ type: 'theory', volumeId: 'vol2', chapterId: openCudaTopicId || '2.1' });
                      } else if (nextVol === 'vol3') {
                        onSelectNode({ type: 'theory', volumeId: 'vol3', chapterId: openKernelTopicId || 'k1.1' });
                      }
                    }}
                    className={`relative flex flex-col items-center justify-center py-2 px-1 rounded-lg border text-center cursor-pointer transition-all duration-150 select-none ${
                      isActive
                        ? `${styles.active} font-bold`
                        : 'border-[#192233] bg-[#0e1422]/70 text-slate-400 hover:text-slate-200 hover:bg-[#141d30] hover:border-[#2a3854]'
                    }`}
                  >
                    <div className="flex items-center gap-1.5 mb-0.5">
                      <span className={`w-2 h-2 rounded-full shrink-0 transition-transform ${isActive ? `${styles.dot} scale-110` : 'bg-slate-600'}`} />
                      <span className="text-[11px] font-mono font-bold leading-none tracking-tight">{vol.label}</span>
                    </div>
                    <span className={`text-[8.5px] font-mono leading-none ${isActive ? 'text-slate-300 font-semibold' : 'text-slate-500'}`}>
                      {vol.count}
                    </span>
                    {isActive && (
                      <div className={`absolute bottom-0 inset-x-2 h-0.5 rounded-full ${styles.indicator}`} />
                    )}
                  </button>
                );
              })}
            </div>
          );
        })()}

        {/* Compact Search */}
        <div className="mt-2 relative">
          <Search className="w-3 h-3 text-slate-500 absolute left-2 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder={
              activeVolume === 'vol3'
                ? 'Search 7 kernel modules...'
                : activeVolume === 'vol2'
                ? 'Search 6 CUDA modules...'
                : 'Search 5 C++ modules...'
            }
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-[#161b24] text-xs text-slate-200 placeholder-slate-500 pl-6 pr-2 py-1 rounded border border-[#21262d] focus:outline-none focus:border-sky-500/40 transition-colors"
          />
        </div>
      </div>

      {/* Reactive Navigation Body */}
      <div className="flex-1 overflow-y-auto px-2 py-2 space-y-1">
        {/* ===================== VOLUME 1: C++ SYSTEMS (5 MODULES) ===================== */}
        {activeVolume === 'vol1' && (
          <div className="space-y-1">
            {filteredCppModules.map((mod: CppModuleMeta) => {
              const isModOpen = openCppModuleId === mod.id;

              return (
                <div key={mod.id} className="rounded-lg overflow-hidden border border-[#141b29] bg-[#0c1018]/60 mb-1">
                  {/* Module Header Row */}
                  <div
                    onClick={() => toggleCppModule(mod.id)}
                    className="flex items-center justify-between px-2.5 py-1.5 cursor-pointer bg-[#0e1422] hover:bg-[#121929] transition-all"
                  >
                    <div className="flex items-center gap-2 min-w-0 pr-1">
                      <span className="font-mono text-[9px] px-1.5 py-0.5 rounded bg-sky-500/10 text-sky-400 font-bold border border-sky-500/20 shrink-0">
                        Part {mod.displayNum}
                      </span>
                      <span className="truncate text-xs font-semibold text-slate-200" title={mod.displayTitle}>
                        {mod.displayTitle}
                      </span>
                    </div>

                    <div className="flex items-center gap-1.5 shrink-0">
                      <span className="text-[10px] text-slate-500 font-mono">
                        {mod.topics.length}
                      </span>
                      {isModOpen ? (
                        <ChevronDown className="w-3 h-3 text-slate-400" />
                      ) : (
                        <ChevronRight className="w-3 h-3 text-slate-400" />
                      )}
                    </div>
                  </div>

                  {/* Module Topics */}
                  {isModOpen && (
                    <div className="p-1 space-y-1 border-t border-[#141b29]">
                      {mod.topics.map((ch: ChapterMeta) => {
                        const isOpen = openChapterId === ch.id;
                        const isCurrentChapter = activeNode.chapterId === ch.id;

                        return (
                          <div key={ch.id} className="rounded-md overflow-hidden transition-all">
                            {/* Chapter Header */}
                            <div
                              onClick={() => {
                                toggleChapter(ch.id);
                                if (!isOpen) {
                                  onSelectNode({ type: 'theory', volumeId: 'vol1', chapterId: ch.id });
                                }
                              }}
                              className={`flex items-center justify-between px-2 py-1.5 rounded-md text-xs cursor-pointer transition-all ${
                                isCurrentChapter
                                  ? 'bg-[#151f30] text-slate-100 font-medium'
                                  : 'hover:bg-[#101622] text-slate-300'
                              }`}
                            >
                              <div className="flex items-center gap-2 min-w-0 pr-1">
                                <span className="font-mono text-[10px] text-sky-400 font-bold shrink-0">
                                  {ch.id}
                                </span>
                                <span className="truncate text-[11.5px]" title={ch.title}>
                                  {ch.title}
                                </span>
                              </div>

                              <div className="flex items-center gap-1.5 shrink-0">
                                {bookmarkedIds.has(ch.id) && (
                                  <Bookmark className="w-3 h-3 text-amber-400 fill-amber-400 shrink-0" />
                                )}
                                {ch.completed ? (
                                  <span title="All tests passed" className="text-emerald-400 flex items-center">
                                    <Check className="w-3 h-3" />
                                  </span>
                                ) : (
                                  <span className="text-[9.5px] text-slate-500 font-mono">
                                    {ch.totalTests}
                                  </span>
                                )}
                                {isOpen ? (
                                  <ChevronDown className="w-3 h-3 text-slate-500" />
                                ) : (
                                  <ChevronRight className="w-3 h-3 text-slate-500" />
                                )}
                              </div>
                            </div>

                            {/* Sub-Items */}
                            {isOpen && (
                              <div className="ml-3 pl-2 py-1 my-0.5 space-y-0.5 border-l border-[#1a2333]">
                                <button
                                  onClick={() =>
                                    onSelectNode({ type: 'theory', volumeId: 'vol1', chapterId: ch.id })
                                  }
                                  className={`w-full flex items-center gap-2 px-2 py-1 rounded text-[11px] text-left transition-all ${
                                    activeNode.type === 'theory' && activeNode.chapterId === ch.id
                                      ? 'bg-sky-500/15 text-sky-300 font-medium'
                                      : 'text-slate-400 hover:text-slate-200 hover:bg-[#111724]'
                                  }`}
                                >
                                  <BookOpen className="w-3 h-3 text-sky-400 shrink-0" />
                                  <span className="truncate">Theory & Models</span>
                                </button>

                                {ch.hasCheatSheet && (
                                  <button
                                    onClick={() =>
                                      onSelectNode({
                                        type: 'cheat_sheet',
                                        volumeId: 'vol1',
                                        chapterId: ch.id,
                                      })
                                    }
                                    className={`w-full flex items-center gap-2 px-2 py-1 rounded text-[11px] text-left transition-all ${
                                      activeNode.type === 'cheat_sheet' && activeNode.chapterId === ch.id
                                        ? 'bg-amber-500/15 text-amber-300 font-medium'
                                        : 'text-slate-400 hover:text-slate-200 hover:bg-[#111724]'
                                    }`}
                                  >
                                    <Zap className="w-3 h-3 text-amber-400 shrink-0" />
                                    <span className="truncate">Revision Cheat Sheet</span>
                                  </button>
                                )}

                                <div className="pt-0.5 space-y-0.5">
                                  {(['beginner', 'intermediate', 'champion'] as const).map((tier) => (
                                    <button
                                      key={tier}
                                      onClick={() =>
                                        onSelectNode({
                                          type: 'workbook',
                                          volumeId: 'vol1',
                                          chapterId: ch.id,
                                          tier,
                                        })
                                      }
                                      className={`w-full flex items-center justify-between px-2 py-1 rounded text-[11px] text-left transition-all ${
                                        activeNode.type === 'workbook' &&
                                        activeNode.chapterId === ch.id &&
                                        activeNode.tier === tier
                                          ? 'bg-sky-500/15 text-sky-300 font-medium'
                                          : 'text-slate-400 hover:text-slate-200 hover:bg-[#111724]'
                                      }`}
                                    >
                                      <div className="flex items-center gap-1.5 truncate">
                                        <span
                                          className={`w-1.5 h-1.5 rounded-full ${
                                            ch.tiers[tier].status === 'passed'
                                              ? 'bg-emerald-400'
                                              : 'bg-slate-600'
                                          }`}
                                        />
                                        <span className="truncate capitalize">{tier}</span>
                                      </div>
                                      <span className="text-[9.5px] font-mono text-slate-500">
                                        {ch.tiers[tier].status === 'passed' ? 'Passed' : `${ch.tiers[tier].tests} tests`}
                                      </span>
                                    </button>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* ===================== VOLUME 2: CUDA SYSTEMS (6 MODULES) ===================== */}
        {activeVolume === 'vol2' && (
          <div className="space-y-1">
            {filteredCudaModules.map((mod: CudaModuleMeta) => {
              const isModOpen = openCudaModuleId === mod.id;

              return (
                <div key={mod.id} className="rounded-lg overflow-hidden border border-[#141b29] bg-[#0c1018]/60 mb-1">
                  {/* Module Header */}
                  <div
                    onClick={() => toggleCudaModule(mod.id)}
                    className="flex items-center justify-between px-2.5 py-1.5 cursor-pointer bg-[#0e1422] hover:bg-[#121929] transition-all"
                  >
                    <div className="flex items-center gap-2 min-w-0 pr-1">
                      <span className="font-mono text-[9px] px-1.5 py-0.5 rounded bg-indigo-500/10 text-indigo-400 font-bold border border-indigo-500/20 shrink-0">
                        M{mod.displayNum}
                      </span>
                      <span className="truncate text-xs font-semibold text-slate-200" title={mod.displayTitle}>
                        {mod.displayTitle}
                      </span>
                    </div>

                    <div className="flex items-center gap-1.5 shrink-0">
                      <span className="text-[10px] text-slate-500 font-mono">
                        {mod.topics.length}
                      </span>
                      {isModOpen ? (
                        <ChevronDown className="w-3 h-3 text-slate-400" />
                      ) : (
                        <ChevronRight className="w-3 h-3 text-slate-400" />
                      )}
                    </div>
                  </div>

                  {/* Module Topics */}
                  {isModOpen && (
                    <div className="p-1 space-y-1 border-t border-[#141b29]">
                      {mod.topics.map((topic: CudaTopicMeta) => {
                        const isOpen = openCudaTopicId === topic.id;
                        const isCurrentTopic = activeNode.chapterId === topic.id;

                        return (
                          <div key={topic.id} className="rounded-md overflow-hidden transition-all">
                            <div
                              onClick={() => {
                                toggleCudaTopic(topic.id);
                                if (!isOpen) {
                                  onSelectNode({ type: 'theory', volumeId: 'vol2', chapterId: topic.id });
                                }
                              }}
                              className={`flex items-center justify-between px-2 py-1.5 rounded-md text-xs cursor-pointer transition-all ${
                                isCurrentTopic
                                  ? 'bg-[#151f30] text-slate-100 font-medium'
                                  : 'hover:bg-[#101622] text-slate-300'
                              }`}
                            >
                              <div className="flex items-center gap-2 min-w-0 pr-1">
                                <span className="font-mono text-[10px] text-indigo-400 font-bold shrink-0">
                                  {topic.displayId}
                                </span>
                                <span className="truncate text-[11.5px]" title={topic.title}>
                                  {topic.title}
                                </span>
                              </div>

                              <div className="flex items-center gap-1.5 shrink-0">
                                {bookmarkedIds.has(topic.id) && (
                                  <Bookmark className="w-3 h-3 text-amber-400 fill-amber-400 shrink-0" />
                                )}
                                {topic.completed ? (
                                  <span title="All tests passed" className="text-emerald-400 flex items-center">
                                    <Check className="w-3 h-3" />
                                  </span>
                                ) : (
                                  <span className="text-[9.5px] text-slate-500 font-mono">
                                    {topic.totalTests}
                                  </span>
                                )}
                                {isOpen ? (
                                  <ChevronDown className="w-3 h-3 text-slate-500" />
                                ) : (
                                  <ChevronRight className="w-3 h-3 text-slate-500" />
                                )}
                              </div>
                            </div>

                            {/* Subitems */}
                            {isOpen && (
                              <div className="ml-3 pl-2 py-1 my-0.5 space-y-0.5 border-l border-[#1a2333]">
                                <button
                                  onClick={() =>
                                    onSelectNode({ type: 'theory', volumeId: 'vol2', chapterId: topic.id })
                                  }
                                  className={`w-full flex items-center gap-2 px-2 py-1 rounded text-[11px] text-left transition-all ${
                                    activeNode.type === 'theory' && activeNode.chapterId === topic.id
                                      ? 'bg-indigo-500/15 text-indigo-300 font-medium'
                                      : 'text-slate-400 hover:text-slate-200 hover:bg-[#111724]'
                                  }`}
                                >
                                  <BookOpen className="w-3 h-3 text-indigo-400 shrink-0" />
                                  <span className="truncate">Theory & Models</span>
                                </button>

                                {topic.hasCheatSheet && (
                                  <button
                                    onClick={() =>
                                      onSelectNode({
                                        type: 'cheat_sheet',
                                        volumeId: 'vol2',
                                        chapterId: topic.id,
                                      })
                                    }
                                    className={`w-full flex items-center gap-2 px-2 py-1 rounded text-[11px] text-left transition-all ${
                                      activeNode.type === 'cheat_sheet' && activeNode.chapterId === topic.id
                                        ? 'bg-amber-500/15 text-amber-300 font-medium'
                                        : 'text-slate-400 hover:text-slate-200 hover:bg-[#111724]'
                                    }`}
                                  >
                                    <Zap className="w-3 h-3 text-amber-400 shrink-0" />
                                    <span className="truncate">Revision Cheat Sheet</span>
                                  </button>
                                )}

                                <div className="pt-0.5 space-y-0.5">
                                  {(['beginner', 'intermediate', 'champion'] as const).map((tier) => (
                                    <button
                                      key={tier}
                                      onClick={() =>
                                        onSelectNode({
                                          type: 'workbook',
                                          volumeId: 'vol2',
                                          chapterId: topic.id,
                                          tier,
                                        })
                                      }
                                      className={`w-full flex items-center justify-between px-2 py-1 rounded text-[11px] text-left transition-all ${
                                        activeNode.type === 'workbook' &&
                                        activeNode.chapterId === topic.id &&
                                        activeNode.tier === tier
                                          ? 'bg-indigo-500/15 text-indigo-300 font-medium'
                                          : 'text-slate-400 hover:text-slate-200 hover:bg-[#111724]'
                                      }`}
                                    >
                                      <div className="flex items-center gap-1.5 truncate">
                                        <span
                                          className={`w-1.5 h-1.5 rounded-full ${
                                            topic.tiers[tier].status === 'passed'
                                              ? 'bg-emerald-400'
                                              : 'bg-slate-600'
                                          }`}
                                        />
                                        <span className="truncate capitalize">{tier}</span>
                                      </div>
                                      <span className="text-[9.5px] font-mono text-slate-500">
                                        {topic.tiers[tier].status === 'passed' ? 'Passed' : `${topic.tiers[tier].tests} tests`}
                                      </span>
                                    </button>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* ===================== VOLUME 3: PRODUCTION KERNELS (7 MODULES) ===================== */}
        {activeVolume === 'vol3' && (
          <div className="space-y-1">
            {filteredKernelModules.map((mod: KernelModuleMeta) => {
              const isModOpen = openKernelModuleId === mod.id;

              return (
                <div key={mod.id} className="rounded-lg overflow-hidden border border-[#141b29] bg-[#0c1018]/60 mb-1">
                  {/* Module Header */}
                  <div
                    onClick={() => toggleKernelModule(mod.id)}
                    className="flex items-center justify-between px-2.5 py-1.5 cursor-pointer bg-[#0e1422] hover:bg-[#121929] transition-all"
                  >
                    <div className="flex items-center gap-2 min-w-0 pr-1">
                      <span className="font-mono text-[9px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400 font-bold border border-amber-500/20 shrink-0">
                        K{mod.displayNum}
                      </span>
                      <span className="truncate text-xs font-semibold text-slate-200" title={mod.displayTitle}>
                        {mod.displayTitle}
                      </span>
                    </div>

                    <div className="flex items-center gap-1.5 shrink-0">
                      <span className="text-[10px] text-slate-500 font-mono">
                        {mod.topics.length}
                      </span>
                      {isModOpen ? (
                        <ChevronDown className="w-3 h-3 text-slate-400" />
                      ) : (
                        <ChevronRight className="w-3 h-3 text-slate-400" />
                      )}
                    </div>
                  </div>

                  {/* Module Topics */}
                  {isModOpen && (
                    <div className="p-1 space-y-1 border-t border-[#141b29]">
                      {mod.topics.map((topic: KernelTopicMeta) => {
                        const isOpen = openKernelTopicId === topic.id;
                        const isCurrentTopic = activeNode.chapterId === topic.id;

                        return (
                          <div key={topic.id} className="rounded-md overflow-hidden transition-all">
                            <div
                              onClick={() => {
                                toggleKernelTopic(topic.id);
                                if (!isOpen) {
                                  onSelectNode({ type: 'theory', volumeId: 'vol3', chapterId: topic.id });
                                }
                              }}
                              className={`flex items-center justify-between px-2 py-1.5 rounded-md text-xs cursor-pointer transition-all ${
                                isCurrentTopic
                                  ? 'bg-[#151f30] text-slate-100 font-medium'
                                  : 'hover:bg-[#101622] text-slate-300'
                              }`}
                            >
                              <div className="flex items-center gap-2 min-w-0 pr-1">
                                <span className="font-mono text-[10px] text-amber-400 font-bold shrink-0">
                                  {topic.displayId}
                                </span>
                                <span className="truncate text-[11.5px]" title={topic.title}>
                                  {topic.title}
                                </span>
                              </div>

                              <div className="flex items-center gap-1.5 shrink-0">
                                {bookmarkedIds.has(topic.id) && (
                                  <Bookmark className="w-3 h-3 text-amber-400 fill-amber-400 shrink-0" />
                                )}
                                {topic.completed ? (
                                  <span title="All tests passed" className="text-emerald-400 flex items-center">
                                    <Check className="w-3 h-3" />
                                  </span>
                                ) : (
                                  <span className="text-[9.5px] text-slate-500 font-mono">
                                    {topic.totalTests}
                                  </span>
                                )}
                                {isOpen ? (
                                  <ChevronDown className="w-3 h-3 text-slate-500" />
                                ) : (
                                  <ChevronRight className="w-3 h-3 text-slate-500" />
                                )}
                              </div>
                            </div>

                            {/* Subitems */}
                            {isOpen && (
                              <div className="ml-3 pl-2 py-1 my-0.5 space-y-0.5 border-l border-[#1a2333]">
                                <button
                                  onClick={() =>
                                    onSelectNode({ type: 'theory', volumeId: 'vol3', chapterId: topic.id })
                                  }
                                  className={`w-full flex items-center gap-2 px-2 py-1 rounded text-[11px] text-left transition-all ${
                                    activeNode.type === 'theory' && activeNode.chapterId === topic.id
                                      ? 'bg-amber-500/15 text-amber-300 font-medium'
                                      : 'text-slate-400 hover:text-slate-200 hover:bg-[#111724]'
                                  }`}
                                >
                                  <BookOpen className="w-3 h-3 text-amber-400 shrink-0" />
                                  <span className="truncate">Theory & Mental Model</span>
                                </button>

                                {topic.hasCheatSheet && (
                                  <button
                                    onClick={() =>
                                      onSelectNode({
                                        type: 'cheat_sheet',
                                        volumeId: 'vol3',
                                        chapterId: topic.id,
                                      })
                                    }
                                    className={`w-full flex items-center gap-2 px-2 py-1 rounded text-[11px] text-left transition-all ${
                                      activeNode.type === 'cheat_sheet' && activeNode.chapterId === topic.id
                                        ? 'bg-amber-500/15 text-amber-300 font-medium'
                                        : 'text-slate-400 hover:text-slate-200 hover:bg-[#111724]'
                                    }`}
                                  >
                                    <Zap className="w-3 h-3 text-amber-400 shrink-0" />
                                    <span className="truncate">Revision Cheat Sheet</span>
                                  </button>
                                )}

                                <div className="pt-0.5 space-y-0.5">
                                  {(['beginner', 'intermediate', 'champion'] as const).map((tier) => (
                                    <button
                                      key={tier}
                                      onClick={() =>
                                        onSelectNode({
                                          type: 'workbook',
                                          volumeId: 'vol3',
                                          chapterId: topic.id,
                                          tier,
                                        })
                                      }
                                      className={`w-full flex items-center justify-between px-2 py-1 rounded text-[11px] text-left transition-all ${
                                        activeNode.type === 'workbook' &&
                                        activeNode.chapterId === topic.id &&
                                        activeNode.tier === tier
                                          ? 'bg-amber-500/15 text-amber-300 font-medium'
                                          : 'text-slate-400 hover:text-slate-200 hover:bg-[#111724]'
                                      }`}
                                    >
                                      <div className="flex items-center gap-1.5 truncate">
                                        <span
                                          className={`w-1.5 h-1.5 rounded-full ${
                                            topic.tiers[tier].status === 'passed'
                                              ? 'bg-emerald-400'
                                              : 'bg-slate-600'
                                          }`}
                                        />
                                        <span className="truncate capitalize">{tier}</span>
                                      </div>
                                      <span className="text-[9.5px] font-mono text-slate-500">
                                        {topic.tiers[tier].status === 'passed' ? 'Passed' : `${topic.tiers[tier].tests} tests`}
                                      </span>
                                    </button>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </aside>
  );
}
