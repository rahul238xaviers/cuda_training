'use client';

import React, { useState, useEffect, useMemo } from 'react';
import {
  BookOpen,
  Zap,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Terminal,
  Cpu,
  Layers,
  Sparkles,
  Flame,
  Search,
  Check,
} from 'lucide-react';
import { ChapterMeta, MetalKernelMeta } from '@/lib/workspace';

export type ActiveNodeType = 'theory' | 'cheat_sheet' | 'workbook' | 'playground' | 'kernel';

export interface ActiveNode {
  type: ActiveNodeType;
  volumeId: string;
  chapterId?: string;
  tier?: 'beginner' | 'intermediate' | 'champion';
  kernelId?: number;
}

interface SidebarTreeProps {
  treeData: any;
  activeNode: ActiveNode;
  onSelectNode: (node: ActiveNode) => void;
  onRefreshTree?: () => void;
}

export function SidebarTree({ treeData, activeNode, onSelectNode, onRefreshTree }: SidebarTreeProps) {
  // Volume Tab selector: 'vol1' | 'vol2' | 'vol3'
  const [activeVolume, setActiveVolume] = useState<'vol1' | 'vol2' | 'vol3'>('vol1');
  const [searchQuery, setSearchQuery] = useState('');

  // Reactive Accordion: automatically track the active chapter
  const [openChapterId, setOpenChapterId] = useState<string>('1.1');

  // Sync open chapter when activeNode changes
  useEffect(() => {
    if (activeNode.chapterId) {
      setOpenChapterId(activeNode.chapterId);
      setActiveVolume('vol1');
    } else if (activeNode.type === 'kernel') {
      setActiveVolume('vol3');
    }
  }, [activeNode.chapterId, activeNode.type]);

  const chapters: ChapterMeta[] = useMemo(() => treeData?.volume1?.chapters || [], [treeData]);
  const kernels: MetalKernelMeta[] = useMemo(() => treeData?.volume3?.kernels || [], [treeData]);

  // Filtered lists for search
  const filteredChapters = useMemo(() => {
    if (!searchQuery.trim()) return chapters;
    const q = searchQuery.toLowerCase();
    return chapters.filter(
      (ch) => ch.title.toLowerCase().includes(q) || ch.id.includes(q)
    );
  }, [chapters, searchQuery]);

  // Group Volume 3 kernels into categories
  const kernelCategories = useMemo(() => {
    const cats: Record<string, MetalKernelMeta[]> = {};
    for (const k of kernels) {
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        if (!k.name.toLowerCase().includes(q) && !k.category.toLowerCase().includes(q)) {
          continue;
        }
      }
      if (!cats[k.category]) cats[k.category] = [];
      cats[k.category].push(k);
    }
    return cats;
  }, [kernels, searchQuery]);

  const toggleChapter = (chId: string) => {
    setOpenChapterId((prev) => (prev === chId ? '' : chId));
  };

  const bookmarkChapter = chapters.find((c) => c.id === treeData?.currentBookmark);

  return (
    <aside className="w-80 min-w-80 h-full bg-[#0a0d14] border-r border-[#1a2333] flex flex-col select-none">
      {/* Brand & Sandbox Header */}
      <div className="p-3.5 border-b border-[#1a2333] bg-[#0c1018]/80 backdrop-blur-md">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-tr from-sky-600 to-indigo-600 flex items-center justify-center text-white shadow-md shadow-sky-500/20">
              <Cpu className="w-3.5 h-3.5" />
            </div>
            <div>
              <h1 className="text-xs font-bold text-slate-100 tracking-tight flex items-center gap-1.5">
                CUDA & Systems
                <span className="text-[9px] px-1 py-0.2 rounded bg-sky-500/10 text-sky-400 font-mono font-medium border border-sky-500/20">
                  BFF
                </span>
              </h1>
            </div>
          </div>
          <button
            onClick={() => onSelectNode({ type: 'playground', volumeId: 'sandbox' })}
            title="Open Interactive Playground Sandbox"
            className={`px-2 py-1 rounded border text-xs flex items-center gap-1.5 transition-all ${
              activeNode.type === 'playground'
                ? 'bg-sky-500/20 text-sky-300 border-sky-500/40 shadow-sm shadow-sky-500/20 font-medium'
                : 'bg-[#111724] text-slate-400 border-[#1e293b] hover:text-slate-200 hover:bg-[#161f30]'
            }`}
          >
            <Terminal className="w-3 h-3 text-sky-400" />
            <span className="font-mono text-[10px]">Playground</span>
          </button>
        </div>

        {/* Clean, Non-Cluttered Progress Banner */}
        <div className="mt-3 px-3 py-2 rounded-lg bg-[#101622] border border-[#1a2333] flex items-center justify-between">
          <div className="flex items-center gap-2 text-[11px]">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-slate-300 font-medium">
              {treeData?.stats?.percent ?? 0}% Complete
            </span>
            <span className="text-slate-500">•</span>
            <span className="text-emerald-400 font-mono font-semibold">
              {treeData?.stats?.testsPassed ?? 0}/{treeData?.stats?.totalTests ?? 0}
            </span>
          </div>
          <span className="text-[10px] text-slate-400 font-mono">
            {bookmarkChapter ? `Next: Ch ${bookmarkChapter.id}` : 'All Passed'}
          </span>
        </div>

        {/* Volume Segmented Control Tabs */}
        <div className="mt-3 grid grid-cols-3 gap-1 bg-[#070a10] p-1 rounded-lg border border-[#1a2333]">
          <button
            onClick={() => setActiveVolume('vol1')}
            className={`py-1.5 text-[10.5px] font-medium rounded-md transition-all text-center truncate ${
              activeVolume === 'vol1'
                ? 'bg-[#162032] text-sky-300 font-semibold shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            C++ (20)
          </button>
          <button
            onClick={() => setActiveVolume('vol2')}
            className={`py-1.5 text-[10.5px] font-medium rounded-md transition-all text-center truncate ${
              activeVolume === 'vol2'
                ? 'bg-[#162032] text-indigo-300 font-semibold shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            CUDA (3)
          </button>
          <button
            onClick={() => setActiveVolume('vol3')}
            className={`py-1.5 text-[10.5px] font-medium rounded-md transition-all text-center truncate ${
              activeVolume === 'vol3'
                ? 'bg-[#162032] text-amber-300 font-semibold shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            22 Kernels
          </button>
        </div>

        {/* Search */}
        <div className="mt-2.5 relative">
          <Search className="w-3 h-3 text-slate-500 absolute left-2.5 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder={
              activeVolume === 'vol3' ? 'Search 22 kernels...' : 'Search chapters & topics...'
            }
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-[#0d121c] text-xs text-slate-200 placeholder-slate-500 pl-7 pr-3 py-1.2 rounded-md border border-[#1a2333] focus:outline-none focus:border-sky-500/40 transition-colors"
          />
        </div>
      </div>

      {/* Reactive Navigation Body */}
      <div className="flex-1 overflow-y-auto px-2 py-2 space-y-1">
        {/* ===================== VOLUME 1: C++ SYSTEMS ===================== */}
        {activeVolume === 'vol1' && (
          <div className="space-y-1">
            {filteredChapters.map((ch: ChapterMeta) => {
              const isOpen = openChapterId === ch.id;
              const isCurrentChapter = activeNode.chapterId === ch.id;

              return (
                <div key={ch.id} className="rounded-md overflow-hidden transition-all">
                  {/* Chapter Header Row */}
                  <div
                    onClick={() => {
                      toggleChapter(ch.id);
                      if (!isOpen) {
                        onSelectNode({ type: 'theory', volumeId: 'vol1', chapterId: ch.id });
                      }
                    }}
                    className={`flex items-center justify-between px-2.5 py-1.5 rounded-md text-xs cursor-pointer transition-all ${
                      isCurrentChapter
                        ? 'bg-[#151f30] text-slate-100 font-medium'
                        : 'hover:bg-[#101622] text-slate-300'
                    }`}
                  >
                    <div className="flex items-center gap-2 min-w-0 pr-1">
                      <span className="font-mono text-[10.5px] text-sky-400 font-bold shrink-0">
                        {ch.id}
                      </span>
                      <span className="truncate text-[11.5px]" title={ch.title}>
                        {ch.title}
                      </span>
                    </div>

                    <div className="flex items-center gap-1.5 shrink-0">
                      {ch.completed ? (
                        <span title="All tests passed" className="text-emerald-400 flex items-center">
                          <Check className="w-3.5 h-3.5" />
                        </span>
                      ) : ch.id === treeData?.currentBookmark ? (
                        <span className="w-1.5 h-1.5 rounded-full bg-sky-400 shadow-sm shadow-sky-400/80" />
                      ) : (
                        <span className="text-[10px] text-slate-500 font-mono">
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

                  {/* Reactive Sub-Items Accordion */}
                  {isOpen && (
                    <div className="ml-3 pl-2 py-1 my-0.5 space-y-0.5 border-l border-[#1a2333]">
                      {/* Theory */}
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

                      {/* Cheat Sheet */}
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

                      {/* 3 Workbooks */}
                      <div className="pt-0.5 space-y-0.5">
                        {/* Beginner */}
                        <button
                          onClick={() =>
                            onSelectNode({
                              type: 'workbook',
                              volumeId: 'vol1',
                              chapterId: ch.id,
                              tier: 'beginner',
                            })
                          }
                          className={`w-full flex items-center justify-between px-2 py-1 rounded text-[11px] text-left transition-all ${
                            activeNode.type === 'workbook' &&
                            activeNode.chapterId === ch.id &&
                            activeNode.tier === 'beginner'
                              ? 'bg-emerald-500/15 text-emerald-300 font-medium'
                              : 'text-slate-400 hover:text-slate-200 hover:bg-[#111724]'
                          }`}
                        >
                          <div className="flex items-center gap-1.5 truncate">
                            <span
                              className={`w-1.5 h-1.5 rounded-full ${
                                ch.tiers.beginner.status === 'passed'
                                  ? 'bg-emerald-400'
                                  : 'bg-slate-600'
                              }`}
                            />
                            <span className="truncate">Beginner</span>
                          </div>
                          <span
                            className={`text-[9.5px] font-mono ${
                              ch.tiers.beginner.status === 'passed'
                                ? 'text-emerald-400 font-semibold'
                                : 'text-slate-500'
                            }`}
                          >
                            {ch.tiers.beginner.status === 'passed'
                              ? 'Passed'
                              : `${ch.tiers.beginner.tests} tests`}
                          </span>
                        </button>

                        {/* Intermediate */}
                        <button
                          onClick={() =>
                            onSelectNode({
                              type: 'workbook',
                              volumeId: 'vol1',
                              chapterId: ch.id,
                              tier: 'intermediate',
                            })
                          }
                          className={`w-full flex items-center justify-between px-2 py-1 rounded text-[11px] text-left transition-all ${
                            activeNode.type === 'workbook' &&
                            activeNode.chapterId === ch.id &&
                            activeNode.tier === 'intermediate'
                              ? 'bg-amber-500/15 text-amber-300 font-medium'
                              : 'text-slate-400 hover:text-slate-200 hover:bg-[#111724]'
                          }`}
                        >
                          <div className="flex items-center gap-1.5 truncate">
                            <span
                              className={`w-1.5 h-1.5 rounded-full ${
                                ch.tiers.intermediate.status === 'passed'
                                  ? 'bg-amber-400'
                                  : 'bg-slate-600'
                              }`}
                            />
                            <span className="truncate">Intermediate</span>
                          </div>
                          <span
                            className={`text-[9.5px] font-mono ${
                              ch.tiers.intermediate.status === 'passed'
                                ? 'text-amber-400 font-semibold'
                                : 'text-slate-500'
                            }`}
                          >
                            {ch.tiers.intermediate.status === 'passed'
                              ? 'Passed'
                              : `${ch.tiers.intermediate.tests} tests`}
                          </span>
                        </button>

                        {/* Champion */}
                        <button
                          onClick={() =>
                            onSelectNode({
                              type: 'workbook',
                              volumeId: 'vol1',
                              chapterId: ch.id,
                              tier: 'champion',
                            })
                          }
                          className={`w-full flex items-center justify-between px-2 py-1 rounded text-[11px] text-left transition-all ${
                            activeNode.type === 'workbook' &&
                            activeNode.chapterId === ch.id &&
                            activeNode.tier === 'champion'
                              ? 'bg-rose-500/15 text-rose-300 font-medium'
                              : 'text-slate-400 hover:text-slate-200 hover:bg-[#111724]'
                          }`}
                        >
                          <div className="flex items-center gap-1.5 truncate">
                            <Flame
                              className={`w-2.5 h-2.5 ${
                                ch.tiers.champion.status === 'passed'
                                  ? 'text-rose-400'
                                  : 'text-slate-600'
                              }`}
                            />
                            <span className="truncate">Champion</span>
                          </div>
                          <span
                            className={`text-[9.5px] font-mono ${
                              ch.tiers.champion.status === 'passed'
                                ? 'text-rose-400 font-semibold'
                                : 'text-slate-500'
                            }`}
                          >
                            {ch.tiers.champion.status === 'passed'
                              ? 'Passed'
                              : `${ch.tiers.champion.tests} tests`}
                          </span>
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* ===================== VOLUME 2: CUDA ARCHITECTURE ===================== */}
        {activeVolume === 'vol2' && (
          <div className="space-y-2 p-1">
            <div className="p-3 rounded-lg bg-[#0e1422] border border-[#1a2333]">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-semibold text-slate-200">Module 2</span>
                <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">Queued</span>
              </div>
              <div className="text-xs text-sky-400 font-medium">CUDA Execution Model</div>
              <p className="text-[10.5px] text-slate-400 mt-1">Warps, Blocks, Grids & Hardware Occupancy</p>
            </div>

            <div className="p-3 rounded-lg bg-[#0e1422] border border-[#1a2333]">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-semibold text-slate-200">Module 3</span>
                <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">Queued</span>
              </div>
              <div className="text-xs text-indigo-400 font-medium">CUDA Memory Hierarchy</div>
              <p className="text-[10.5px] text-slate-400 mt-1">Global DRAM, Shared SRAM Bank Conflicts, Registers</p>
            </div>

            <div className="p-3 rounded-lg bg-[#0e1422] border border-[#1a2333]">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-semibold text-slate-200">Module 4</span>
                <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">Queued</span>
              </div>
              <div className="text-xs text-emerald-400 font-medium">Parallel Primitives</div>
              <p className="text-[10.5px] text-slate-400 mt-1">Tree Reductions, Prefix Scan, Tiled GEMM</p>
            </div>
          </div>
        )}

        {/* ===================== VOLUME 3: 22 LLM KERNELS ===================== */}
        {activeVolume === 'vol3' && (
          <div className="space-y-3 p-1">
            {Object.entries(kernelCategories).map(([category, items]) => (
              <div key={category} className="space-y-1">
                <div className="px-1 text-[10px] font-bold text-slate-500 uppercase tracking-wider">
                  {category} ({items.length})
                </div>
                <div className="space-y-0.5">
                  {items.map((k) => {
                    const isSelected = activeNode.type === 'kernel' && activeNode.kernelId === k.id;
                    return (
                      <button
                        key={k.id}
                        onClick={() =>
                          onSelectNode({
                            type: 'kernel',
                            volumeId: 'vol3',
                            kernelId: k.id,
                          })
                        }
                        className={`w-full flex items-center justify-between px-2.5 py-1.5 rounded text-xs text-left transition-all ${
                          isSelected
                            ? 'bg-amber-500/15 text-amber-200 font-medium border-l-2 border-amber-400'
                            : 'text-slate-400 hover:text-slate-200 hover:bg-[#111724]'
                        }`}
                      >
                        <div className="flex items-center gap-2 min-w-0 pr-1">
                          <span className="font-mono text-[10px] text-slate-500 shrink-0">#{k.id}</span>
                          <span className="truncate">{k.name}</span>
                        </div>
                        <span className="text-[9px] font-mono text-slate-500 shrink-0">.cu</span>
                      </button>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </aside>
  );
}
