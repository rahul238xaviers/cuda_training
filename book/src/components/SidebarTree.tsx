'use client';

import React, { useState } from 'react';
import {
  BookOpen,
  Zap,
  CheckCircle2,
  Clock,
  ChevronDown,
  ChevronRight,
  Terminal,
  Cpu,
  Layers,
  Sparkles,
  Flame,
  Search,
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
}

export function SidebarTree({ treeData, activeNode, onSelectNode }: SidebarTreeProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedChapters, setExpandedChapters] = useState<Record<string, boolean>>({
    '1.1': true,
    '1.2': true,
  });
  const [expandedVolumes, setExpandedVolumes] = useState<Record<string, boolean>>({
    vol1: true,
    vol2: false,
    vol3: true,
  });

  const toggleVolume = (volId: string) => {
    setExpandedVolumes((prev) => ({ ...prev, [volId]: !prev[volId] }));
  };

  const toggleChapter = (chId: string) => {
    setExpandedChapters((prev) => ({ ...prev, [chId]: !prev[chId] }));
  };

  const filteredChapters = (treeData?.volume1?.chapters || []).filter((ch: ChapterMeta) =>
    ch.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    ch.id.includes(searchQuery)
  );

  const filteredKernels = (treeData?.volume3?.kernels || []).filter((k: MetalKernelMeta) =>
    k.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    k.category.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <aside className="w-80 min-w-80 h-full bg-[#0b1019] border-r border-[#1e293b] flex flex-col select-none">
      {/* Brand Header */}
      <div className="p-4 border-b border-[#1e293b] bg-[#0d1422]/50 backdrop-blur-sm">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-sky-600 to-indigo-500 flex items-center justify-center text-white shadow-lg shadow-sky-500/20">
              <Cpu className="w-4 h-4" />
            </div>
            <div>
              <h1 className="text-sm font-bold text-slate-100 tracking-tight flex items-center gap-1.5">
                CUDA & Systems
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-sky-500/10 text-sky-400 font-mono font-medium border border-sky-500/20">
                  BFF
                </span>
              </h1>
              <p className="text-[11px] text-slate-400">Digital Book & Compiler Lab</p>
            </div>
          </div>
          <button
            onClick={() => onSelectNode({ type: 'playground', volumeId: 'sandbox' })}
            title="Open Interactive Playground Sandbox"
            className={`p-1.5 rounded-md border text-xs flex items-center gap-1 transition-all ${
              activeNode.type === 'playground'
                ? 'bg-sky-500/20 text-sky-300 border-sky-500/40 shadow-sm shadow-sky-500/20'
                : 'bg-[#141d2e] text-slate-400 border-[#1e293b] hover:text-slate-200 hover:bg-[#1a263c]'
            }`}
          >
            <Terminal className="w-3.5 h-3.5 text-sky-400" />
            <span className="font-mono font-medium text-[11px]">Lab</span>
          </button>
        </div>

        {/* Master Progress Card */}
        <div className="mt-3.5 p-3 rounded-lg bg-[#111726] border border-[#1e293b]">
          <div className="flex items-center justify-between text-[11px] font-medium mb-1.5">
            <span className="text-slate-400 flex items-center gap-1">
              <Sparkles className="w-3 h-3 text-amber-400" />
              Curriculum Mastery
            </span>
            <span className="font-mono text-emerald-400 font-semibold">
              {treeData?.stats?.percent ?? 5}% (10 Tests)
            </span>
          </div>
          <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-emerald-500 via-sky-500 to-indigo-500 rounded-full transition-all duration-500"
              style={{ width: `${treeData?.stats?.percent ?? 5}%` }}
            />
          </div>
          <div className="mt-2 flex items-center justify-between text-[10px] text-slate-400">
            <span>Bookmark: <strong className="text-sky-300 font-mono">1.2 Strides</strong></span>
            <span className="text-emerald-400 font-medium">1 / 20 Ch Complete</span>
          </div>
        </div>

        {/* Quick Search */}
        <div className="mt-2.5 relative">
          <Search className="w-3.5 h-3.5 text-slate-500 absolute left-2.5 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search chapters, kernels..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-[#111726] text-xs text-slate-200 placeholder-slate-500 pl-8 pr-3 py-1.5 rounded-md border border-[#1e293b] focus:outline-none focus:border-sky-500/50 transition-colors"
          />
        </div>
      </div>

      {/* Tree Content Area */}
      <div className="flex-1 overflow-y-auto px-2 py-3 space-y-4">
        {/* VOLUME 1: C++ Low-Level Systems */}
        <div className="space-y-1">
          <div
            onClick={() => toggleVolume('vol1')}
            className="flex items-center justify-between px-2.5 py-1.5 rounded-md text-xs font-semibold text-slate-300 hover:bg-[#111726] cursor-pointer transition-colors"
          >
            <div className="flex items-center gap-2">
              <Layers className="w-3.5 h-3.5 text-sky-400" />
              <span>Volume 1: C++ Memory Systems</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="text-[10px] px-1.5 py-0.2 bg-emerald-500/10 text-emerald-400 rounded font-mono">
                1/20
              </span>
              {expandedVolumes.vol1 ? <ChevronDown className="w-3.5 h-3.5 text-slate-500" /> : <ChevronRight className="w-3.5 h-3.5 text-slate-500" />}
            </div>
          </div>

          {expandedVolumes.vol1 && (
            <div className="pl-1 space-y-1 mt-1">
              {filteredChapters.map((ch: ChapterMeta) => {
                const isExpanded = !!expandedChapters[ch.id];
                const isCurrentChapter = activeNode.chapterId === ch.id;

                return (
                  <div key={ch.id} className="rounded-lg bg-[#0e1422]/60 border border-[#1a2333] overflow-hidden">
                    {/* Chapter Header Card */}
                    <div
                      onClick={() => toggleChapter(ch.id)}
                      className={`flex items-center justify-between px-2.5 py-2 text-xs cursor-pointer transition-all ${
                        isCurrentChapter
                          ? 'bg-[#151f33] text-slate-100 font-medium'
                          : 'hover:bg-[#121a2c] text-slate-300'
                      }`}
                    >
                      <div className="flex items-center gap-2 min-w-0 pr-1">
                        <span className="font-mono text-[11px] text-sky-400 font-bold shrink-0">
                          {ch.id}
                        </span>
                        <span className="truncate text-[11.5px]" title={ch.title}>
                          {ch.title}
                        </span>
                      </div>
                      <div className="flex items-center gap-1.5 shrink-0">
                        {ch.completed ? (
                          <span className="flex items-center gap-1 text-[10px] text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/20 font-medium">
                            <CheckCircle2 className="w-2.5 h-2.5" /> Passed
                          </span>
                        ) : ch.id === treeData?.currentBookmark ? (
                          <span className="text-[10px] text-sky-400 bg-sky-500/10 px-1.5 py-0.5 rounded border border-sky-500/20 font-medium">
                            Current
                          </span>
                        ) : (
                          <span className="text-[10px] text-slate-500 font-mono">Queued</span>
                        )}
                        {isExpanded ? (
                          <ChevronDown className="w-3 h-3 text-slate-500" />
                        ) : (
                          <ChevronRight className="w-3 h-3 text-slate-500" />
                        )}
                      </div>
                    </div>

                    {/* Sub-Nodes: Theory, Cheat Sheet, Live Workbooks */}
                    {isExpanded && (
                      <div className="bg-[#090d16] px-2 py-1.5 space-y-0.5 border-t border-[#1a2333]/60">
                        {/* 1. Theory Node */}
                        <div
                          onClick={() => onSelectNode({ type: 'theory', volumeId: 'vol1', chapterId: ch.id })}
                          className={`flex items-center gap-2 px-2.5 py-1.5 rounded-md text-[11px] cursor-pointer transition-all ${
                            activeNode.type === 'theory' && activeNode.chapterId === ch.id
                              ? 'bg-sky-500/15 text-sky-300 font-medium border-l-2 border-sky-400'
                              : 'text-slate-400 hover:text-slate-200 hover:bg-[#111726]'
                          }`}
                        >
                          <BookOpen className="w-3 h-3 text-sky-400 shrink-0" />
                          <span className="truncate">Theory & Mental Models</span>
                        </div>

                        {/* 2. Cheat Sheet Node */}
                        {ch.hasCheatSheet && (
                          <div
                            onClick={() => onSelectNode({ type: 'cheat_sheet', volumeId: 'vol1', chapterId: ch.id })}
                            className={`flex items-center gap-2 px-2.5 py-1.5 rounded-md text-[11px] cursor-pointer transition-all ${
                              activeNode.type === 'cheat_sheet' && activeNode.chapterId === ch.id
                                ? 'bg-amber-500/15 text-amber-300 font-medium border-l-2 border-amber-400'
                                : 'text-slate-400 hover:text-slate-200 hover:bg-[#111726]'
                            }`}
                          >
                            <Zap className="w-3 h-3 text-amber-400 shrink-0" />
                            <span className="truncate">Revision Cheat Sheet</span>
                          </div>
                        )}

                        {/* 3. Live Workbook Nodes */}
                        <div className="pt-1 mt-1 border-t border-[#162032] space-y-0.5">
                          {/* Beginner Workbook */}
                          <div
                            onClick={() =>
                              onSelectNode({
                                type: 'workbook',
                                volumeId: 'vol1',
                                chapterId: ch.id,
                                tier: 'beginner',
                              })
                            }
                            className={`flex items-center justify-between px-2.5 py-1.5 rounded-md text-[11px] cursor-pointer transition-all ${
                              activeNode.type === 'workbook' &&
                              activeNode.chapterId === ch.id &&
                              activeNode.tier === 'beginner'
                                ? 'bg-emerald-500/15 text-emerald-300 font-medium border-l-2 border-emerald-400'
                                : 'text-slate-400 hover:text-slate-200 hover:bg-[#111726]'
                            }`}
                          >
                            <div className="flex items-center gap-2 truncate">
                              <span className="w-2 h-2 rounded-full bg-emerald-400 shadow-sm shadow-emerald-400/50" />
                              <span className="truncate">Beginner Workbook</span>
                            </div>
                            <span className="text-[10px] font-mono text-emerald-400 font-semibold">
                              {ch.tiers.beginner.tests > 0 ? `${ch.tiers.beginner.tests}/4` : 'WIP'}
                            </span>
                          </div>

                          {/* Intermediate Workbook */}
                          <div
                            onClick={() =>
                              onSelectNode({
                                type: 'workbook',
                                volumeId: 'vol1',
                                chapterId: ch.id,
                                tier: 'intermediate',
                              })
                            }
                            className={`flex items-center justify-between px-2.5 py-1.5 rounded-md text-[11px] cursor-pointer transition-all ${
                              activeNode.type === 'workbook' &&
                              activeNode.chapterId === ch.id &&
                              activeNode.tier === 'intermediate'
                                ? 'bg-amber-500/15 text-amber-300 font-medium border-l-2 border-amber-400'
                                : 'text-slate-400 hover:text-slate-200 hover:bg-[#111726]'
                            }`}
                          >
                            <div className="flex items-center gap-2 truncate">
                              <span className="w-2 h-2 rounded-full bg-amber-400 shadow-sm shadow-amber-400/50" />
                              <span className="truncate">Intermediate Workbook</span>
                            </div>
                            <span className="text-[10px] font-mono text-amber-400 font-semibold">
                              {ch.tiers.intermediate.tests > 0 ? `${ch.tiers.intermediate.tests}/3` : 'WIP'}
                            </span>
                          </div>

                          {/* Champion Workbook */}
                          <div
                            onClick={() =>
                              onSelectNode({
                                type: 'workbook',
                                volumeId: 'vol1',
                                chapterId: ch.id,
                                tier: 'champion',
                              })
                            }
                            className={`flex items-center justify-between px-2.5 py-1.5 rounded-md text-[11px] cursor-pointer transition-all ${
                              activeNode.type === 'workbook' &&
                              activeNode.chapterId === ch.id &&
                              activeNode.tier === 'champion'
                                ? 'bg-rose-500/15 text-rose-300 font-medium border-l-2 border-rose-400'
                                : 'text-slate-400 hover:text-slate-200 hover:bg-[#111726]'
                            }`}
                          >
                            <div className="flex items-center gap-2 truncate">
                              <Flame className="w-2.5 h-2.5 text-rose-400 shrink-0" />
                              <span className="truncate">Champion Workbook</span>
                            </div>
                            <span className="text-[10px] font-mono text-rose-400 font-semibold">
                              {ch.tiers.champion.tests > 0 ? `${ch.tiers.champion.tests}/3` : 'WIP'}
                            </span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* VOLUME 2: CUDA Architecture & Execution Foundations */}
        <div className="space-y-1">
          <div
            onClick={() => toggleVolume('vol2')}
            className="flex items-center justify-between px-2.5 py-1.5 rounded-md text-xs font-semibold text-slate-300 hover:bg-[#111726] cursor-pointer transition-colors"
          >
            <div className="flex items-center gap-2">
              <Cpu className="w-3.5 h-3.5 text-indigo-400" />
              <span>Volume 2: CUDA Architecture</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="text-[10px] text-slate-500 font-mono">Modules 2-4</span>
              {expandedVolumes.vol2 ? <ChevronDown className="w-3.5 h-3.5 text-slate-500" /> : <ChevronRight className="w-3.5 h-3.5 text-slate-500" />}
            </div>
          </div>

          {expandedVolumes.vol2 && (
            <div className="pl-3 space-y-1 mt-1 text-[11.5px] text-slate-400">
              <div className="p-2 rounded-md bg-[#0e1422] border border-[#1a2333]">
                <div className="text-slate-300 font-medium">Module 2: CUDA Execution Model</div>
                <div className="text-[10px] text-slate-500 mt-0.5">Warps, Blocks, Grids & Hardware Occupancy</div>
              </div>
              <div className="p-2 rounded-md bg-[#0e1422] border border-[#1a2333]">
                <div className="text-slate-300 font-medium">Module 3: CUDA Memory Hierarchy</div>
                <div className="text-[10px] text-slate-500 mt-0.5">Global DRAM, Shared SRAM Bank Conflicts, Registers</div>
              </div>
              <div className="p-2 rounded-md bg-[#0e1422] border border-[#1a2333]">
                <div className="text-slate-300 font-medium">Module 4: Parallel Primitives</div>
                <div className="text-[10px] text-slate-500 mt-0.5">Tree Reductions, Prefix Scan, Tiled GEMM</div>
              </div>
            </div>
          )}
        </div>

        {/* VOLUME 3: The 22 Production LLM CUDA Kernels */}
        <div className="space-y-1">
          <div
            onClick={() => toggleVolume('vol3')}
            className="flex items-center justify-between px-2.5 py-1.5 rounded-md text-xs font-semibold text-slate-300 hover:bg-[#111726] cursor-pointer transition-colors"
          >
            <div className="flex items-center gap-2">
              <Sparkles className="w-3.5 h-3.5 text-amber-400" />
              <span>Volume 3: 22 LLM CUDA Kernels</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="text-[10px] px-1.5 py-0.2 bg-amber-500/10 text-amber-400 rounded font-mono">
                22 Kernels
              </span>
              {expandedVolumes.vol3 ? <ChevronDown className="w-3.5 h-3.5 text-slate-500" /> : <ChevronRight className="w-3.5 h-3.5 text-slate-500" />}
            </div>
          </div>

          {expandedVolumes.vol3 && (
            <div className="pl-1 space-y-1 mt-1">
              {filteredKernels.map((k: MetalKernelMeta) => {
                const isSelected = activeNode.type === 'kernel' && activeNode.kernelId === k.id;

                return (
                  <div
                    key={k.id}
                    onClick={() =>
                      onSelectNode({
                        type: 'kernel',
                        volumeId: 'vol3',
                        kernelId: k.id,
                      })
                    }
                    className={`flex items-center justify-between px-2.5 py-1.5 rounded-md text-[11px] cursor-pointer transition-all ${
                      isSelected
                        ? 'bg-amber-500/15 text-amber-200 font-medium border-l-2 border-amber-400'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-[#111726]'
                    }`}
                  >
                    <div className="flex items-center gap-2 min-w-0 pr-1">
                      <span className="font-mono text-[10px] text-slate-500 shrink-0">#{k.id}</span>
                      <span className="truncate">{k.name}</span>
                    </div>
                    <span className="text-[9px] px-1.5 py-0.5 rounded bg-[#162032] text-slate-400 font-mono shrink-0">
                      {k.category}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </aside>
  );
}
