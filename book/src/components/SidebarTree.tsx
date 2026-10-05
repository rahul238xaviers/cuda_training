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
  Code2,
  Plus,
  Trash2,
  X,
} from 'lucide-react';
import { getBookmarks } from '@/lib/annotations';
import { ThemeToggle } from './ThemeToggle';
import {
  ChapterMeta,
  CudaModuleMeta,
  CudaTopicMeta,
  CppModuleMeta,
  KernelModuleMeta,
  KernelTopicMeta,
  PlaygroundType,
  PracticeKernelMeta,
} from '@/lib/workspace';

export type ActiveNodeType = 'theory' | 'cheat_sheet' | 'workbook' | 'playground' | 'kernel' | 'guide' | 'practice';

export interface ActiveNode {
  type: ActiveNodeType;
  volumeId: string;
  chapterId?: string;
  tier?: 'beginner' | 'intermediate' | 'champion';
  kernelId?: number;
  playgroundType?: PlaygroundType;
  practiceKernel?: string;
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

type ModuleStatus = 'completed' | 'in_progress' | 'pending';

interface ModuleStatusInfo {
  status: ModuleStatus;
  completedTopics: number;
  totalTopics: number;
  passedTests: number;
  totalTests: number;
}

function getModuleStatus(
  topics: Array<{ completed: boolean; passedTests?: number; totalTests?: number }>,
  completedFlag?: boolean,
  passedTests?: number
): ModuleStatusInfo {
  const completedTopics = (topics || []).filter((t) => t.completed).length;
  const totalTopics = (topics || []).length;
  const sumPassedTests = passedTests ?? (topics || []).reduce((acc, t) => acc + (t.passedTests || 0), 0);
  const sumTotalTests = (topics || []).reduce((acc, t) => acc + (t.totalTests || 0), 0);

  const isCompleted = completedFlag || (totalTopics > 0 && completedTopics === totalTopics);
  const isInProgress = !isCompleted && (completedTopics > 0 || sumPassedTests > 0);
  const status: ModuleStatus = isCompleted ? 'completed' : isInProgress ? 'in_progress' : 'pending';

  return {
    status,
    completedTopics,
    totalTopics,
    passedTests: sumPassedTests,
    totalTests: sumTotalTests,
  };
}

function getModuleStyle(status: ModuleStatus, accent: 'sky' | 'indigo' | 'amber') {
  if (status === 'completed') {
    return {
      card: 'module-card module-card-completed border-emerald-500/40 bg-gradient-to-b from-[#0e1d17] to-[#08120e] shadow-[0_0_14px_rgba(16,185,129,0.07)]',
      header: 'module-header module-header-completed bg-[#10241d]/90 hover:bg-[#152e25] text-emerald-100',
      badge: 'module-badge module-badge-completed bg-emerald-500/25 text-emerald-300 border border-emerald-500/40 font-bold',
      treeLine: 'module-treeline module-treeline-completed border-emerald-500/40',
      title: 'module-title module-title-completed text-emerald-100 font-semibold',
    };
  }
  if (status === 'in_progress') {
    return {
      card: 'module-card module-card-inprogress border-amber-500/40 bg-gradient-to-b from-[#1c1509] to-[#0f0c05] shadow-[0_0_14px_rgba(245,158,11,0.07)]',
      header: 'module-header module-header-inprogress bg-[#221b0b]/90 hover:bg-[#2b220e] text-amber-100',
      badge: 'module-badge module-badge-inprogress bg-amber-500/25 text-amber-300 border border-amber-500/40 font-bold',
      treeLine: 'module-treeline module-treeline-inprogress border-amber-500/40',
      title: 'module-title module-title-inprogress text-amber-100 font-semibold',
    };
  }
  return {
    card: 'module-card module-card-pending border-[#1b2333] bg-[#0c1018]/60',
    header: 'module-header module-header-pending bg-[#0e1422] hover:bg-[#121929] text-slate-300',
    badge: 'module-badge module-badge-pending bg-slate-800/90 text-slate-400 border border-slate-700/50 font-bold',
    treeLine: 'module-treeline module-treeline-pending border-slate-800/80',
    title: 'module-title module-title-pending text-slate-300 font-medium',
  };
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

  // Top-Level Navigation Mode: 'curriculum' vs 'practice'
  const [navMode, setNavMode] = useState<'curriculum' | 'practice'>(
    activeNode.type === 'practice' ? 'practice' : 'curriculum'
  );

  // Sync navMode when activeNode changes externally
  useEffect(() => {
    if (activeNode.type === 'practice') {
      setNavMode('practice');
    } else if (activeNode.type === 'theory' || activeNode.type === 'workbook' || activeNode.type === 'guide') {
      setNavMode('curriculum');
    }
  }, [activeNode.type]);

  const practiceKernels: PracticeKernelMeta[] = treeData?.practiceKernels || [];

  // Practice Lab state: new kernel creator & search query
  const [isCreatingKernel, setIsCreatingKernel] = useState(false);
  const [newKernelName, setNewKernelName] = useState('');
  const [practiceSearchQuery, setPracticeSearchQuery] = useState('');
  const [isCreatingLoading, setIsCreatingLoading] = useState(false);
  const [createError, setCreateError] = useState('');
  const [deleteConfirmFile, setDeleteConfirmFile] = useState<string | null>(null);

  const handleCreateNewKernel = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const name = newKernelName.trim();
    if (!name) return;

    setIsCreatingLoading(true);
    setCreateError('');
    try {
      const res = await fetch('/api/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: 'create_practice_kernel',
          filename: name,
        }),
      });
      const data = await res.json();
      if (data.success) {
        setNewKernelName('');
        setIsCreatingKernel(false);
        onRefreshTree?.();
        selectPracticeKernel(data.filename);
      } else {
        setCreateError(data.error || 'Failed to create kernel');
      }
    } catch (err: any) {
      setCreateError(err.message || 'Network error');
    } finally {
      setIsCreatingLoading(false);
    }
  };

  const handleDeleteKernel = async (filename: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      const res = await fetch('/api/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: 'delete_practice_kernel',
          filename,
        }),
      });
      const data = await res.json();
      if (data.success) {
        setDeleteConfirmFile(null);
        onRefreshTree?.();
        if (activeNode.type === 'practice' && activeNode.practiceKernel === filename) {
          const remaining = practiceKernels.filter((k) => k.filename !== filename);
          if (remaining.length > 0) {
            selectPracticeKernel(remaining[0].filename);
          } else {
            setNavMode('curriculum');
          }
        }
      }
    } catch (err) {
      console.error('Delete kernel failed:', err);
    }
  };

  const filteredPracticeKernels = useMemo(() => {
    if (!practiceSearchQuery.trim()) return practiceKernels;
    const q = practiceSearchQuery.toLowerCase();
    return practiceKernels.filter(
      (k) =>
        k.filename.toLowerCase().includes(q) ||
        k.title.toLowerCase().includes(q) ||
        k.subtitle.toLowerCase().includes(q)
    );
  }, [practiceKernels, practiceSearchQuery]);

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

  const selectPracticeKernel = (filename: string) => {
    setIsPlaygroundMenuOpen(false);
    onSelectNode({
      type: 'practice',
      volumeId: 'practice',
      practiceKernel: filename,
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
              setNavMode('curriculum');
              setActiveVolume('vol1');
            }}
            title="Volume 1: C++ Systems"
            className={`p-2 rounded-lg text-xs font-mono transition-all ${
              navMode === 'curriculum' && activeVolume === 'vol1'
                ? 'bg-[#162032] text-sky-300 font-bold'
                : 'text-slate-500 hover:text-slate-300'
            }`}
          >
            C++
          </button>
          <button
            onClick={() => {
              onToggleCollapse?.();
              setNavMode('curriculum');
              setActiveVolume('vol2');
            }}
            title="Volume 2: CUDA"
            className={`p-2 rounded-lg text-xs font-mono transition-all ${
              navMode === 'curriculum' && activeVolume === 'vol2'
                ? 'bg-[#162032] text-indigo-300 font-bold'
                : 'text-slate-500 hover:text-slate-300'
            }`}
          >
            CU
          </button>
          <button
            onClick={() => {
              onToggleCollapse?.();
              setNavMode('curriculum');
              setActiveVolume('vol3');
            }}
            title="Volume 3: Kernels"
            className={`p-2 rounded-lg text-xs font-mono transition-all ${
              navMode === 'curriculum' && activeVolume === 'vol3'
                ? 'bg-[#162032] text-amber-300 font-bold'
                : 'text-slate-500 hover:text-slate-300'
            }`}
          >
            KN
          </button>
          <div className="w-5 h-[1px] bg-[#21262d] my-1" />
          <button
            onClick={() => {
              onToggleCollapse?.();
              setNavMode('practice');
              if (practiceKernels.length > 0) {
                selectPracticeKernel(practiceKernels[0].filename);
              }
            }}
            title="Open Practice Lab (kernels/practice)"
            className={`p-2 rounded-lg text-xs font-mono transition-all ${
              navMode === 'practice'
                ? 'bg-teal-500/20 text-teal-300 font-bold border border-teal-500/40'
                : 'text-slate-500 hover:text-teal-400'
            }`}
          >
            <Code2 className="w-4 h-4" />
          </button>
        </div>
      </div>
    );
  }

  return (
    <aside
      style={{ width }}
      className="sidebar-aside h-full bg-[var(--bg-sidebar)] border-r border-[var(--border-subtle)] flex flex-col select-none relative shrink-0 transition-[width] duration-75"
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
      <div className="sidebar-brand-header p-3 border-b border-[var(--border-subtle)] bg-[var(--bg-header)]/95 backdrop-blur-md">
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
                  onClick={() => {
                    if (activeNode.type === 'practice') {
                      selectPracticeKernel(activeNode.practiceKernel || 'gpu_check.cu');
                    } else {
                      selectPlayground(currentPlaygroundType);
                    }
                  }}
                  title="Open Active Sandbox or Practice Kernel"
                  className={`px-2 py-0.5 text-xs flex items-center gap-1.5 transition-all cursor-pointer ${
                    activeNode.type === 'playground'
                      ? 'bg-sky-500/20 text-sky-300 font-medium'
                      : activeNode.type === 'practice'
                      ? 'bg-teal-500/20 text-teal-300 font-medium'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-[#1c2230]'
                  }`}
                >
                  <Terminal className={`w-3 h-3 ${activeNode.type === 'practice' ? 'text-teal-400' : 'text-sky-400'}`} />
                  <span className="font-mono text-[10px] capitalize">
                    {activeNode.type === 'playground'
                      ? currentPlaygroundType
                      : activeNode.type === 'practice'
                      ? 'Practice'
                      : 'Playground'}
                  </span>
                </button>
                <button
                  onClick={() => setIsPlaygroundMenuOpen((prev) => !prev)}
                  title="Choose Playground or Open Practice Kernel"
                  className="px-1 py-0.5 text-slate-400 hover:text-slate-200 hover:bg-[#1c2230] border-l border-[#30363d] transition-colors cursor-pointer"
                >
                  <ChevronDown className="w-2.5 h-2.5" />
                </button>
              </div>

              {/* Dropdown Menu */}
              {isPlaygroundMenuOpen && (
                <div className="absolute right-0 mt-1 w-64 bg-[#161b24] border border-[#30363d] rounded-lg shadow-xl shadow-black/60 z-50 py-1 overflow-hidden animate-in fade-in zoom-in-95 duration-100">
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

                  {practiceKernels.length > 0 && (
                    <>
                      <div className="px-2.5 py-1 text-[10px] font-semibold text-teal-400 uppercase tracking-wider border-t border-b border-[#21262d] bg-[#0c1618] flex items-center justify-between">
                        <span>Open Practice (kernels/practice)</span>
                        <span className="text-[9px] font-mono text-teal-500">{practiceKernels.length}</span>
                      </div>
                      <div className="max-h-52 overflow-y-auto divide-y divide-[#21262d]/40">
                        {practiceKernels.map((pk) => {
                          const isCur = activeNode.type === 'practice' && (activeNode.practiceKernel === pk.filename || activeNode.practiceKernel === pk.id);
                          return (
                            <button
                              key={pk.id}
                              onClick={() => selectPracticeKernel(pk.filename)}
                              className={`w-full flex items-start gap-2 px-2.5 py-1.5 text-left hover:bg-[#1c2230] transition-colors cursor-pointer ${
                                isCur ? 'bg-teal-500/15 text-teal-300' : 'text-slate-300'
                              }`}
                            >
                              <span className="text-teal-400 font-mono text-[10px] mt-0.5 px-1 py-0.2 rounded bg-teal-500/10 border border-teal-500/20 shrink-0">
                                cu
                              </span>
                              <div className="min-w-0 flex-1">
                                <div className="text-xs font-medium truncate">{pk.title}</div>
                                <div className="text-[9.5px] text-slate-500 font-mono truncate">{pk.filename}</div>
                              </div>
                            </button>
                          );
                        })}
                      </div>
                    </>
                  )}
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

        {/* Top-Level Mode Switcher: Curriculum vs Open Practice Lab */}
        <div className="mt-2.5 p-1 rounded-xl bg-[#090d16] border border-[#232b3d] grid grid-cols-2 gap-1 shadow-inner">
          <button
            onClick={() => setNavMode('curriculum')}
            className={`flex items-center justify-center gap-1.5 py-1.5 px-2 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
              navMode === 'curriculum'
                ? 'bg-[#10233b] border border-sky-400/80 text-sky-200 shadow-[0_0_12px_rgba(56,189,248,0.25)]'
                : 'text-slate-400 hover:text-slate-200 hover:bg-[#161b24]'
            }`}
          >
            <BookOpen className="w-3.5 h-3.5 text-sky-400" />
            <span>Course Tracks</span>
          </button>
          <button
            onClick={() => {
              setNavMode('practice');
              if (activeNode.type !== 'practice' && practiceKernels.length > 0) {
                selectPracticeKernel(practiceKernels[0].filename);
              }
            }}
            className={`flex items-center justify-center gap-1.5 py-1.5 px-2 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
              navMode === 'practice'
                ? 'bg-[#082422] border border-teal-400/80 text-teal-200 shadow-[0_0_12px_rgba(45,212,191,0.25)]'
                : 'text-slate-400 hover:text-slate-200 hover:bg-[#161b24]'
            }`}
          >
            <Code2 className="w-3.5 h-3.5 text-teal-400" />
            <span>Practice Lab</span>
            <span className="text-[10px] font-mono px-1.5 py-0.2 rounded-full bg-teal-500/20 text-teal-300 border border-teal-500/30 font-bold ml-0.5">
              {practiceKernels.length}
            </span>
          </button>
        </div>

        {/* ----------------- CURRICULUM MODE CONTROLS ----------------- */}
        {navMode === 'curriculum' && (
          <>
            {/* Sleek Minimal Progress Tracker (Volume-Specific Chapters & Overall Tests) */}
            {(() => {
              const currentVolDone =
                activeVolume === 'vol1'
                  ? treeData?.stats?.chaptersDone ?? 0
                  : activeVolume === 'vol2'
                  ? treeData?.stats?.cudaTopicsDone ?? 0
                  : treeData?.stats?.kernelTopicsDone ?? 0;

              const currentVolTotal =
                activeVolume === 'vol1'
                  ? treeData?.stats?.totalChapters ?? 20
                  : activeVolume === 'vol2'
                  ? treeData?.stats?.totalCudaTopics ?? 22
                  : treeData?.stats?.totalKernelTopics ?? 25;

              const currentVolPercent =
                currentVolTotal > 0 ? Math.round((currentVolDone / currentVolTotal) * 100) : 0;

              return (
                <div className="mt-2.5 px-0.5">
                  <div className="flex items-center justify-between text-[10px] mb-1">
                    <span className="text-slate-300 font-medium flex items-center gap-1.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
                      <span>
                        {activeVolume === 'vol1'
                          ? 'Volume 1 (C++ Systems)'
                          : activeVolume === 'vol2'
                          ? 'Volume 2 (CUDA Systems)'
                          : 'Volume 3 (GPU Kernels)'}
                      </span>
                    </span>
                    <span className="font-mono text-slate-200 font-semibold">
                      {currentVolDone}/{currentVolTotal}{' '}
                      <span className="text-slate-400 font-normal">({currentVolPercent}%)</span>
                    </span>
                  </div>
                  <div className="w-full h-1 bg-[#10141d] rounded-full overflow-hidden border border-[#21262d]">
                    <div
                      className="h-full bg-emerald-500 rounded-full transition-all duration-500"
                      style={{ width: `${Math.max(1, currentVolPercent)}%` }}
                    />
                  </div>
                  <div className="flex items-center justify-between mt-1 text-[9px] text-slate-500 font-mono">
                    <span>
                      {activeVolume === 'vol1'
                        ? 'Chapters Mastered'
                        : activeVolume === 'vol2'
                        ? 'Topics Mastered'
                        : 'Kernels Mastered'}
                    </span>
                    <span title="Total individual verified test cases passed across all workbooks">
                      {treeData?.stats?.testsPassed ?? 0} tests passed total
                    </span>
                  </div>
                </div>
              );
            })()}

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

            {/* Category Tabs — C++ Systems, CUDA, GPU Kernels */}
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
                <div className="sidebar-vol-tabs mt-2.5 p-1 rounded-xl bg-[#090d16] border border-[#232b3d] grid grid-cols-3 gap-1.5 shadow-inner">
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
                        className={`sidebar-vol-btn relative flex flex-col items-center justify-center py-2 px-1 rounded-lg border text-center cursor-pointer transition-all duration-150 select-none ${
                          isActive
                            ? `sidebar-vol-btn-active ${styles.active} font-bold`
                            : 'border-[#192233] bg-[#0e1422]/70 text-slate-400 hover:text-slate-200 hover:bg-[#141d30] hover:border-[#2a3854]'
                        }`}
                      >
                        <div className="flex items-center gap-1.5 mb-0.5">
                          <span
                            className={`w-2 h-2 rounded-full shrink-0 transition-transform ${
                              isActive ? `${styles.dot} scale-110` : 'bg-slate-600'
                            }`}
                          />
                          <span className="text-[11px] font-mono font-bold leading-none tracking-tight">{vol.label}</span>
                        </div>
                        <span
                          className={`text-[8.5px] font-mono leading-none ${
                            isActive ? 'text-slate-300 font-semibold' : 'text-slate-500'
                          }`}
                        >
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

            {/* Compact Curriculum Search */}
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
          </>
        )}

        {/* ----------------- PRACTICE LAB MODE CONTROLS ----------------- */}
        {navMode === 'practice' && (
          <div className="mt-2.5 p-2 rounded-lg border border-teal-500/30 bg-[#091517] space-y-2">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-[11px] font-bold text-teal-300 flex items-center gap-1.5">
                  <Terminal className="w-3 h-3 text-teal-400" />
                  <span>Open Practice Lab</span>
                </div>
                <div className="text-[9.5px] font-mono text-slate-400">kernels/practice/*.cu</div>
              </div>
              <button
                onClick={() => {
                  setIsCreatingKernel((prev) => !prev);
                  setCreateError('');
                }}
                className="px-2.5 py-1 rounded-md text-[11px] font-semibold bg-teal-500 hover:bg-teal-400 text-black flex items-center gap-1 shadow-sm transition-all cursor-pointer"
                title="Write a new custom CUDA practice kernel"
              >
                <Plus className="w-3 h-3 stroke-[3]" />
                <span>New Kernel</span>
              </button>
            </div>

            {/* Inline New Kernel Creation Form */}
            {isCreatingKernel && (
              <form onSubmit={handleCreateNewKernel} className="p-2 rounded-md bg-[#060e10] border border-teal-500/40 space-y-2">
                <div className="text-[10.5px] font-medium text-teal-200">New Kernel Filename:</div>
                <div className="flex items-center gap-1">
                  <input
                    type="text"
                    autoFocus
                    value={newKernelName}
                    onChange={(e) => setNewKernelName(e.target.value)}
                    placeholder="e.g. gemm_tiled or fast_gelu"
                    className="w-full bg-[#101d20] text-teal-100 placeholder-slate-500 px-2 py-1 rounded border border-teal-500/40 focus:outline-none focus:border-teal-300 font-mono text-[11px]"
                  />
                  <span className="text-[10px] text-slate-400 font-mono">.cu</span>
                </div>
                <p className="text-[9px] text-slate-400">
                  Creates a runnable host+device CUDA template ready to edit, save (Cmd+S), and run with nvcc.
                </p>
                {createError && (
                  <div className="text-[10px] text-rose-400 font-medium bg-rose-950/40 border border-rose-800/50 p-1.5 rounded">
                    {createError}
                  </div>
                )}
                <div className="flex items-center justify-end gap-1.5 pt-0.5">
                  <button
                    type="button"
                    onClick={() => {
                      setIsCreatingKernel(false);
                      setCreateError('');
                    }}
                    className="px-2 py-0.5 rounded text-[10.5px] text-slate-400 hover:text-white hover:bg-slate-800 transition-colors cursor-pointer"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={isCreatingLoading || !newKernelName.trim()}
                    className="px-2.5 py-0.5 rounded text-[10.5px] font-semibold bg-teal-400 hover:bg-teal-300 text-black disabled:opacity-50 transition-all cursor-pointer"
                  >
                    {isCreatingLoading ? 'Creating...' : 'Create & Open'}
                  </button>
                </div>
              </form>
            )}

            {/* Search / Filter for Practice Kernels */}
            <div className="relative">
              <Search className="w-3 h-3 text-teal-500/60 absolute left-2 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                placeholder={`Filter ${practiceKernels.length} practice kernels...`}
                value={practiceSearchQuery}
                onChange={(e) => setPracticeSearchQuery(e.target.value)}
                className="w-full bg-[#060e10] text-teal-100 placeholder-slate-500 pl-6 pr-2 py-1 rounded border border-teal-500/25 focus:outline-none focus:border-teal-400/50 transition-colors text-[11px]"
              />
            </div>
          </div>
        )}
      </div>

      {/* Reactive Navigation Body */}
      <div className="flex-1 overflow-y-auto px-2 py-2 space-y-1.5">
        {/* ===================== OPEN PRACTICE LAB (FREE-FORM KERNELS) ===================== */}
        {navMode === 'practice' && (
          <div className="space-y-1.5">
            {filteredPracticeKernels.length === 0 ? (
              <div className="p-4 rounded-lg border border-[#21262d] bg-[#0c1018] text-center space-y-2">
                <Code2 className="w-6 h-6 text-slate-500 mx-auto" />
                <div className="text-xs text-slate-300 font-medium">No practice kernels found</div>
                <div className="text-[10px] text-slate-500">
                  {practiceSearchQuery ? `No match for "${practiceSearchQuery}"` : 'Your practice lab is empty.'}
                </div>
                <button
                  onClick={() => setIsCreatingKernel(true)}
                  className="px-2.5 py-1 rounded text-xs font-semibold bg-teal-500 text-black hover:bg-teal-400 transition-colors cursor-pointer"
                >
                  Create New Kernel (.cu)
                </button>
              </div>
            ) : (
              filteredPracticeKernels.map((pk) => {
                const isSelected =
                  activeNode.type === 'practice' &&
                  (activeNode.practiceKernel === pk.filename || activeNode.practiceKernel === pk.id);
                const isConfirmingDelete = deleteConfirmFile === pk.filename;

                return (
                  <div
                    key={pk.id}
                    onClick={() => selectPracticeKernel(pk.filename)}
                    className={`group relative p-2.5 rounded-lg border transition-all cursor-pointer ${
                      isSelected
                        ? 'bg-teal-500/15 border-teal-500/60 shadow-[0_0_12px_rgba(45,212,191,0.18)]'
                        : 'bg-[#091517]/80 hover:bg-[#0c1e21] border-teal-500/20 text-slate-300 hover:text-white'
                    }`}
                  >
                    <div className="flex items-start justify-between gap-1.5">
                      <div className="flex items-start gap-2 min-w-0 flex-1">
                        <Terminal
                          className={`w-3.5 h-3.5 mt-0.5 shrink-0 ${
                            isSelected ? 'text-teal-300' : 'text-teal-400/80 group-hover:text-teal-300'
                          }`}
                        />
                        <div className="min-w-0 flex-1">
                          <div
                            className={`text-xs font-semibold truncate ${
                              isSelected ? 'text-teal-100 font-bold' : 'text-slate-200 group-hover:text-white'
                            }`}
                          >
                            {pk.title}
                          </div>
                          {pk.subtitle && (
                            <div className="text-[10px] text-slate-400 line-clamp-1 mt-0.5">
                              {pk.subtitle}
                            </div>
                          )}
                          <div className="flex items-center gap-2 mt-1.5">
                            <span className="text-[9.5px] font-mono text-teal-400/90 bg-[#061214] px-1.5 py-0.5 rounded border border-teal-500/25 truncate">
                              {pk.filename}
                            </span>
                            <span className="text-[9px] font-mono px-1 py-0.2 rounded bg-teal-500/10 text-teal-400 border border-teal-500/20">
                              nvcc -O3
                            </span>
                          </div>
                        </div>
                      </div>

                      {/* Delete action */}
                      <div className="shrink-0 flex items-center">
                        {isConfirmingDelete ? (
                          <div
                            className="flex items-center gap-1 bg-[#1a0808] border border-rose-500/40 p-1 rounded"
                            onClick={(e) => e.stopPropagation()}
                          >
                            <span className="text-[9px] text-rose-300 font-medium">Delete?</span>
                            <button
                              onClick={(e) => handleDeleteKernel(pk.filename, e)}
                              className="px-1.5 py-0.5 text-[9px] bg-rose-600 hover:bg-rose-500 text-white rounded font-bold cursor-pointer"
                            >
                              Yes
                            </button>
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                setDeleteConfirmFile(null);
                              }}
                              className="px-1 py-0.5 text-[9px] text-slate-400 hover:text-white cursor-pointer"
                            >
                              <X className="w-2.5 h-2.5" />
                            </button>
                          </div>
                        ) : (
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setDeleteConfirmFile(pk.filename);
                            }}
                            title="Delete this practice kernel"
                            className="opacity-0 group-hover:opacity-100 p-1 text-slate-500 hover:text-rose-400 hover:bg-rose-500/10 rounded transition-all cursor-pointer"
                          >
                            <Trash2 className="w-3 h-3" />
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        )}

        {/* ===================== VOLUME 1: C++ SYSTEMS (5 MODULES) ===================== */}
        {navMode === 'curriculum' && activeVolume === 'vol1' && (
          <div className="space-y-2">
            {filteredCppModules.map((mod: CppModuleMeta) => {
              const isModOpen = openCppModuleId === mod.id;
              const statusInfo = getModuleStatus(mod.topics, mod.completed, mod.passedTests);
              const style = getModuleStyle(statusInfo.status, 'sky');

              return (
                <div key={mod.id} className={`rounded-lg overflow-hidden border ${style.card} transition-all`}>
                  {/* Module Header Row (Parent Level) */}
                  <div
                    onClick={() => toggleCppModule(mod.id)}
                    className={`flex items-center justify-between px-2.5 py-2 cursor-pointer transition-all ${style.header}`}
                  >
                    <div className="flex items-center gap-2 min-w-0 pr-1">
                      <span className={`font-mono text-[9px] px-1.5 py-0.5 rounded shrink-0 ${style.badge}`}>
                        Part {mod.displayNum}
                      </span>
                      <span className={`truncate text-xs ${style.title}`} title={mod.displayTitle}>
                        {mod.displayTitle}
                      </span>
                    </div>

                    <div className="flex items-center gap-1.5 shrink-0">
                      {statusInfo.status === 'completed' ? (
                        <span
                          title="All chapters in this module completed"
                          className="flex items-center gap-1 text-[9px] font-mono px-1.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-bold"
                        >
                          <Check className="w-2.5 h-2.5 stroke-[3]" /> Mastered
                        </span>
                      ) : statusInfo.status === 'in_progress' ? (
                        <span
                          title={`${statusInfo.completedTopics} of ${statusInfo.totalTopics} chapters mastered (${statusInfo.passedTests} tests passed)`}
                          className="flex items-center gap-1 text-[9px] font-mono px-1.5 py-0.5 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/40 font-bold"
                        >
                          <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
                          {statusInfo.completedTopics}/{statusInfo.totalTopics}
                        </span>
                      ) : (
                        <span
                          title="Pending module"
                          className="text-[9px] font-mono px-1.5 py-0.5 rounded-full bg-slate-800/80 text-slate-500 border border-slate-700/40"
                        >
                          0/{statusInfo.totalTopics}
                        </span>
                      )}

                      {isModOpen ? (
                        <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
                      ) : (
                        <ChevronRight className="w-3.5 h-3.5 text-slate-400" />
                      )}
                    </div>
                  </div>

                  {/* Module Topics (Child Level — visibly indented with tree guide line) */}
                  {isModOpen && (
                    <div className="module-topics-container pt-1.5 pb-2 pl-3.5 pr-1.5 bg-[#06080e]/70 border-t border-[#141b29]/80">
                      <div className={`space-y-1 pl-2.5 border-l-2 ${style.treeLine}`}>
                        {mod.topics.map((ch: ChapterMeta) => {
                          const isOpen = openChapterId === ch.id;
                          const isCurrentChapter = activeNode.chapterId === ch.id;

                          return (
                            <div key={ch.id} className="rounded-md overflow-hidden transition-all">
                              {/* Chapter Header (Child Item) */}
                              <div
                                onClick={() => {
                                  toggleChapter(ch.id);
                                  if (!isOpen) {
                                    onSelectNode({ type: 'theory', volumeId: 'vol1', chapterId: ch.id });
                                  }
                                }}
                                className={`topic-item flex items-center justify-between px-2 py-1.5 rounded-md text-xs cursor-pointer transition-all ${
                                  isCurrentChapter
                                    ? 'topic-item-active bg-[#15233c] text-white font-medium border border-sky-500/40 shadow-sm'
                                    : 'hover:bg-[#101726] text-slate-300 hover:text-white border border-transparent'
                                }`}
                              >
                                <div className="flex items-center gap-2 min-w-0 pr-1">
                                  <span className="topic-badge font-mono text-[10px] px-1.5 py-0.5 rounded bg-sky-500/10 text-sky-400 font-bold border border-sky-500/25 shrink-0">
                                    {ch.id}
                                  </span>
                                  <span className="topic-title truncate text-[11.5px]" title={ch.title}>
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
                                  ) : (ch.passedTests ?? 0) > 0 ? (
                                    <span className="text-[9.5px] text-amber-400 font-mono font-medium">
                                      {ch.passedTests}/{ch.totalTests}
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

                              {/* Grandchildren (Theory, Cheat Sheet, Workbooks — indented deeper) */}
                              {isOpen && (
                                <div className="topic-children-container ml-3 pl-2.5 py-1 my-1 space-y-0.5 border-l-2 border-sky-500/25 bg-[#0e1422]/50 rounded-r-md">
                                  <button
                                    onClick={() =>
                                      onSelectNode({ type: 'theory', volumeId: 'vol1', chapterId: ch.id })
                                    }
                                    className={`topic-sublink w-full flex items-center gap-2 px-2 py-1 rounded text-[11px] text-left transition-all ${
                                      activeNode.type === 'theory' && activeNode.chapterId === ch.id
                                        ? 'topic-sublink-active bg-sky-500/15 text-sky-300 font-medium'
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
                                      className={`topic-sublink w-full flex items-center gap-2 px-2 py-1 rounded text-[11px] text-left transition-all ${
                                        activeNode.type === 'cheat_sheet' && activeNode.chapterId === ch.id
                                          ? 'topic-sublink-active bg-amber-500/15 text-amber-300 font-medium'
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
                                        className={`topic-sublink w-full flex items-center justify-between px-2 py-1 rounded text-[11px] text-left transition-all ${
                                          activeNode.type === 'workbook' &&
                                          activeNode.chapterId === ch.id &&
                                          activeNode.tier === tier
                                            ? 'topic-sublink-active bg-sky-500/15 text-sky-300 font-medium'
                                            : 'text-slate-400 hover:text-slate-200 hover:bg-[#111724]'
                                        }`}
                                      >
                                        <div className="flex items-center gap-1.5 truncate">
                                          <span
                                            className={`w-1.5 h-1.5 rounded-full ${
                                              ch.tiers[tier].status === 'passed'
                                                ? 'bg-emerald-400'
                                                : (ch.tiers[tier].testsPassed ?? 0) > 0
                                                ? 'bg-amber-400'
                                                : 'bg-slate-600'
                                            }`}
                                          />
                                          <span className="truncate capitalize">{tier}</span>
                                          {(ch.tiers[tier].newProblemsCount ?? 0) > 0 && (ch.tiers[tier].testsPassed ?? 0) > 0 && (
                                            <span className="text-[9px] px-1 py-0.2 rounded bg-amber-500/20 text-amber-300 font-semibold border border-amber-500/30">
                                              +{ch.tiers[tier].newProblemsCount} New
                                            </span>
                                          )}
                                        </div>
                                        <span className="text-[9.5px] font-mono text-slate-500">
                                          {ch.tiers[tier].status === 'passed'
                                            ? 'Passed'
                                            : (ch.tiers[tier].testsPassed ?? 0) > 0
                                            ? `${ch.tiers[tier].testsPassed}/${ch.tiers[tier].tests}`
                                            : `${ch.tiers[tier].tests} tests`}
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
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* ===================== VOLUME 2: CUDA SYSTEMS (6 MODULES) ===================== */}
        {navMode === 'curriculum' && activeVolume === 'vol2' && (
          <div className="space-y-2">
            {filteredCudaModules.map((mod: CudaModuleMeta) => {
              const isModOpen = openCudaModuleId === mod.id;
              const statusInfo = getModuleStatus(mod.topics, mod.completed, mod.passedTests);
              const style = getModuleStyle(statusInfo.status, 'indigo');

              return (
                <div key={mod.id} className={`rounded-lg overflow-hidden border ${style.card} transition-all`}>
                  {/* Module Header Row (Parent Level) */}
                  <div
                    onClick={() => toggleCudaModule(mod.id)}
                    className={`flex items-center justify-between px-2.5 py-2 cursor-pointer transition-all ${style.header}`}
                  >
                    <div className="flex items-center gap-2 min-w-0 pr-1">
                      <span className={`font-mono text-[9px] px-1.5 py-0.5 rounded shrink-0 ${style.badge}`}>
                        M{mod.displayNum}
                      </span>
                      <span className={`truncate text-xs ${style.title}`} title={mod.displayTitle}>
                        {mod.displayTitle}
                      </span>
                    </div>

                    <div className="flex items-center gap-1.5 shrink-0">
                      {statusInfo.status === 'completed' ? (
                        <span
                          title="All topics in this module completed"
                          className="flex items-center gap-1 text-[9px] font-mono px-1.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-bold"
                        >
                          <Check className="w-2.5 h-2.5 stroke-[3]" /> Mastered
                        </span>
                      ) : statusInfo.status === 'in_progress' ? (
                        <span
                          title={`${statusInfo.completedTopics} of ${statusInfo.totalTopics} topics mastered (${statusInfo.passedTests} tests passed)`}
                          className="flex items-center gap-1 text-[9px] font-mono px-1.5 py-0.5 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/40 font-bold"
                        >
                          <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
                          {statusInfo.completedTopics}/{statusInfo.totalTopics}
                        </span>
                      ) : (
                        <span
                          title="Pending module"
                          className="text-[9px] font-mono px-1.5 py-0.5 rounded-full bg-slate-800/80 text-slate-500 border border-slate-700/40"
                        >
                          0/{statusInfo.totalTopics}
                        </span>
                      )}

                      {isModOpen ? (
                        <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
                      ) : (
                        <ChevronRight className="w-3.5 h-3.5 text-slate-400" />
                      )}
                    </div>
                  </div>

                  {/* Module Topics (Child Level — visibly indented with tree guide line) */}
                  {isModOpen && (
                    <div className="module-topics-container pt-1.5 pb-2 pl-3.5 pr-1.5 bg-[#06080e]/70 border-t border-[#141b29]/80">
                      <div className={`space-y-1 pl-2.5 border-l-2 ${style.treeLine}`}>
                        {mod.topics.map((topic: CudaTopicMeta) => {
                          const isOpen = openCudaTopicId === topic.id;
                          const isCurrentTopic = activeNode.chapterId === topic.id;

                          return (
                            <div key={topic.id} className="rounded-md overflow-hidden transition-all">
                              {/* Topic Header (Child Item) */}
                              <div
                                onClick={() => {
                                  toggleCudaTopic(topic.id);
                                  if (!isOpen) {
                                    onSelectNode({ type: 'theory', volumeId: 'vol2', chapterId: topic.id });
                                  }
                                }}
                                className={`topic-item flex items-center justify-between px-2 py-1.5 rounded-md text-xs cursor-pointer transition-all ${
                                  isCurrentTopic
                                    ? 'topic-item-active bg-[#1b1c38] text-white font-medium border border-indigo-500/40 shadow-sm'
                                    : 'hover:bg-[#121326] text-slate-300 hover:text-white border border-transparent'
                                }`}
                              >
                                <div className="flex items-center gap-2 min-w-0 pr-1">
                                  <span className="topic-badge font-mono text-[10px] px-1.5 py-0.5 rounded bg-indigo-500/10 text-indigo-400 font-bold border border-indigo-500/25 shrink-0">
                                    {topic.displayId}
                                  </span>
                                  <span className="topic-title truncate text-[11.5px]" title={topic.title}>
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
                                  ) : (topic.passedTests ?? 0) > 0 ? (
                                    <span className="text-[9.5px] text-amber-400 font-mono font-medium">
                                      {topic.passedTests}/{topic.totalTests}
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

                              {/* Grandchildren (Theory, Cheat Sheet, Workbooks — indented deeper) */}
                              {isOpen && (
                                <div className="topic-children-container ml-3 pl-2.5 py-1 my-1 space-y-0.5 border-l-2 border-indigo-500/25 bg-[#0e1122]/50 rounded-r-md">
                                  <button
                                    onClick={() =>
                                      onSelectNode({ type: 'theory', volumeId: 'vol2', chapterId: topic.id })
                                    }
                                    className={`topic-sublink w-full flex items-center gap-2 px-2 py-1 rounded text-[11px] text-left transition-all ${
                                      activeNode.type === 'theory' && activeNode.chapterId === topic.id
                                        ? 'topic-sublink-active bg-indigo-500/15 text-indigo-300 font-medium'
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
                                      className={`topic-sublink w-full flex items-center gap-2 px-2 py-1 rounded text-[11px] text-left transition-all ${
                                        activeNode.type === 'cheat_sheet' && activeNode.chapterId === topic.id
                                          ? 'topic-sublink-active bg-amber-500/15 text-amber-300 font-medium'
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
                                        className={`topic-sublink w-full flex items-center justify-between px-2 py-1 rounded text-[11px] text-left transition-all ${
                                          activeNode.type === 'workbook' &&
                                          activeNode.chapterId === topic.id &&
                                          activeNode.tier === tier
                                            ? 'topic-sublink-active bg-indigo-500/15 text-indigo-300 font-medium'
                                            : 'text-slate-400 hover:text-slate-200 hover:bg-[#111724]'
                                        }`}
                                      >
                                        <div className="flex items-center gap-1.5 truncate">
                                          <span
                                            className={`w-1.5 h-1.5 rounded-full ${
                                              topic.tiers[tier].status === 'passed'
                                                ? 'bg-emerald-400'
                                                : (topic.tiers[tier].testsPassed ?? 0) > 0
                                                ? 'bg-amber-400'
                                                : 'bg-slate-600'
                                            }`}
                                          />
                                          <span className="truncate capitalize">{tier}</span>
                                          {(topic.tiers[tier].newProblemsCount ?? 0) > 0 && (topic.tiers[tier].testsPassed ?? 0) > 0 && (
                                            <span className="text-[9px] px-1 py-0.2 rounded bg-amber-500/20 text-amber-300 font-semibold border border-amber-500/30">
                                              +{topic.tiers[tier].newProblemsCount} New
                                            </span>
                                          )}
                                        </div>
                                        <span className="text-[9.5px] font-mono text-slate-500">
                                          {topic.tiers[tier].status === 'passed'
                                            ? 'Passed'
                                            : (topic.tiers[tier].testsPassed ?? 0) > 0
                                            ? `${topic.tiers[tier].testsPassed}/${topic.tiers[tier].tests}`
                                            : `${topic.tiers[tier].tests} tests`}
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
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* ===================== VOLUME 3: PRODUCTION KERNELS (7 MODULES) ===================== */}
        {navMode === 'curriculum' && activeVolume === 'vol3' && (
          <div className="space-y-2">
            {filteredKernelModules.map((mod: KernelModuleMeta) => {
              const isModOpen = openKernelModuleId === mod.id;
              const statusInfo = getModuleStatus(mod.topics, mod.completed, mod.passedTests);
              const style = getModuleStyle(statusInfo.status, 'amber');

              return (
                <div key={mod.id} className={`rounded-lg overflow-hidden border ${style.card} transition-all`}>
                  {/* Module Header Row (Parent Level) */}
                  <div
                    onClick={() => toggleKernelModule(mod.id)}
                    className={`flex items-center justify-between px-2.5 py-2 cursor-pointer transition-all ${style.header}`}
                  >
                    <div className="flex items-center gap-2 min-w-0 pr-1">
                      <span className={`font-mono text-[9px] px-1.5 py-0.5 rounded shrink-0 ${style.badge}`}>
                        K{mod.displayNum}
                      </span>
                      <span className={`truncate text-xs ${style.title}`} title={mod.displayTitle}>
                        {mod.displayTitle}
                      </span>
                    </div>

                    <div className="flex items-center gap-1.5 shrink-0">
                      {statusInfo.status === 'completed' ? (
                        <span
                          title="All topics in this module completed"
                          className="flex items-center gap-1 text-[9px] font-mono px-1.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-bold"
                        >
                          <Check className="w-2.5 h-2.5 stroke-[3]" /> Mastered
                        </span>
                      ) : statusInfo.status === 'in_progress' ? (
                        <span
                          title={`${statusInfo.completedTopics} of ${statusInfo.totalTopics} topics mastered (${statusInfo.passedTests} tests passed)`}
                          className="flex items-center gap-1 text-[9px] font-mono px-1.5 py-0.5 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/40 font-bold"
                        >
                          <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
                          {statusInfo.completedTopics}/{statusInfo.totalTopics}
                        </span>
                      ) : (
                        <span
                          title="Pending module"
                          className="text-[9px] font-mono px-1.5 py-0.5 rounded-full bg-slate-800/80 text-slate-500 border border-slate-700/40"
                        >
                          0/{statusInfo.totalTopics}
                        </span>
                      )}

                      {isModOpen ? (
                        <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
                      ) : (
                        <ChevronRight className="w-3.5 h-3.5 text-slate-400" />
                      )}
                    </div>
                  </div>

                  {/* Module Topics (Child Level — visibly indented with tree guide line) */}
                  {isModOpen && (
                    <div className="module-topics-container pt-1.5 pb-2 pl-3.5 pr-1.5 bg-[#06080e]/70 border-t border-[#141b29]/80">
                      <div className={`space-y-1 pl-2.5 border-l-2 ${style.treeLine}`}>
                        {mod.topics.map((topic: KernelTopicMeta) => {
                          const isOpen = openKernelTopicId === topic.id;
                          const isCurrentTopic = activeNode.chapterId === topic.id;

                          return (
                            <div key={topic.id} className="rounded-md overflow-hidden transition-all">
                              {/* Topic Header (Child Item) */}
                              <div
                                onClick={() => {
                                  toggleKernelTopic(topic.id);
                                  if (!isOpen) {
                                    onSelectNode({ type: 'theory', volumeId: 'vol3', chapterId: topic.id });
                                  }
                                }}
                                className={`topic-item flex items-center justify-between px-2 py-1.5 rounded-md text-xs cursor-pointer transition-all ${
                                  isCurrentTopic
                                    ? 'topic-item-active bg-[#291e0e] text-white font-medium border border-amber-500/40 shadow-sm'
                                    : 'hover:bg-[#1a1409] text-slate-300 hover:text-white border border-transparent'
                                }`}
                              >
                                <div className="flex items-center gap-2 min-w-0 pr-1">
                                  <span className="topic-badge font-mono text-[10px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400 font-bold border border-amber-500/25 shrink-0">
                                    {topic.displayId}
                                  </span>
                                  <span className="topic-title truncate text-[11.5px]" title={topic.title}>
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
                                  ) : (topic.passedTests ?? 0) > 0 ? (
                                    <span className="text-[9.5px] text-amber-400 font-mono font-medium">
                                      {topic.passedTests}/{topic.totalTests}
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

                              {/* Grandchildren (Theory, Cheat Sheet, Workbooks — indented deeper) */}
                              {isOpen && (
                                <div className="topic-children-container ml-3 pl-2.5 py-1 my-1 space-y-0.5 border-l-2 border-amber-500/25 bg-[#17120a]/50 rounded-r-md">
                                  <button
                                    onClick={() =>
                                      onSelectNode({ type: 'theory', volumeId: 'vol3', chapterId: topic.id })
                                    }
                                    className={`topic-sublink w-full flex items-center gap-2 px-2 py-1 rounded text-[11px] text-left transition-all ${
                                      activeNode.type === 'theory' && activeNode.chapterId === topic.id
                                        ? 'topic-sublink-active bg-amber-500/15 text-amber-300 font-medium'
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
                                      className={`topic-sublink w-full flex items-center gap-2 px-2 py-1 rounded text-[11px] text-left transition-all ${
                                        activeNode.type === 'cheat_sheet' && activeNode.chapterId === topic.id
                                          ? 'topic-sublink-active bg-amber-500/15 text-amber-300 font-medium'
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
                                        className={`topic-sublink w-full flex items-center justify-between px-2 py-1 rounded text-[11px] text-left transition-all ${
                                          activeNode.type === 'workbook' &&
                                          activeNode.chapterId === topic.id &&
                                          activeNode.tier === tier
                                            ? 'topic-sublink-active bg-amber-500/15 text-amber-300 font-medium'
                                            : 'text-slate-400 hover:text-slate-200 hover:bg-[#111724]'
                                        }`}
                                      >
                                        <div className="flex items-center gap-1.5 truncate">
                                          <span
                                            className={`w-1.5 h-1.5 rounded-full ${
                                              topic.tiers[tier].status === 'passed'
                                                ? 'bg-emerald-400'
                                                : (topic.tiers[tier].testsPassed ?? 0) > 0
                                                ? 'bg-amber-400'
                                                : 'bg-slate-600'
                                            }`}
                                          />
                                          <span className="truncate capitalize">{tier}</span>
                                          {(topic.tiers[tier].newProblemsCount ?? 0) > 0 && (topic.tiers[tier].testsPassed ?? 0) > 0 && (
                                            <span className="text-[9px] px-1 py-0.2 rounded bg-amber-500/20 text-amber-300 font-semibold border border-amber-500/30">
                                              +{topic.tiers[tier].newProblemsCount} New
                                            </span>
                                          )}
                                        </div>
                                        <span className="text-[9.5px] font-mono text-slate-500">
                                          {topic.tiers[tier].status === 'passed'
                                            ? 'Passed'
                                            : (topic.tiers[tier].testsPassed ?? 0) > 0
                                            ? `${topic.tiers[tier].testsPassed}/${topic.tiers[tier].tests}`
                                            : `${topic.tiers[tier].tests} tests`}
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
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Sidebar Footer with Theme Selector */}
      <div className="p-2 border-t border-[#1a202c] bg-[#0b0e14] flex items-center justify-between shrink-0">
        <ThemeToggle variant="sidebar" />
        <span className="text-[10px] font-mono text-slate-500">⌘S format+save</span>
      </div>
    </aside>
  );
}
