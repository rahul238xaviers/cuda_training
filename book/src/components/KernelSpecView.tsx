'use client';

import React from 'react';
import { Sparkles, Cpu, Layers, HardDrive, CheckCircle2, ArrowRight } from 'lucide-react';

interface KernelSpecViewProps {
  kernelData: any;
}

export function KernelSpecView({ kernelData }: KernelSpecViewProps) {
  const { kernel, spec } = kernelData || {};

  if (!kernel) {
    return <div className="p-8 text-slate-400">Loading kernel blueprint...</div>;
  }

  return (
    <div className="flex-1 overflow-y-auto bg-[#080b11] p-8 md:p-12 max-w-5xl mx-auto w-full select-text">
      {/* Header */}
      <div className="mb-8 pb-6 border-b border-[#1e293b]">
        <div className="flex items-center gap-2 text-xs font-mono text-amber-400 mb-2">
          <span className="px-2.5 py-0.5 rounded bg-amber-500/10 border border-amber-500/20 font-semibold flex items-center gap-1.5">
            <Sparkles className="w-3 h-3" /> VOLUME 3: PRODUCTION LLM CUDA KERNEL #{kernel.id}
          </span>
          <span className="px-2 py-0.5 rounded bg-[#162032] text-slate-400">
            {kernel.category}
          </span>
        </div>
        <h1 className="text-3xl font-extrabold text-slate-100 tracking-tight">
          {kernel.name}
        </h1>
        <p className="mt-2 text-sm text-slate-400 leading-relaxed">
          {spec?.overview}
        </p>
      </div>

      {/* Comparison Cards: Apple Silicon Metal vs NVIDIA CUDA */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
        {/* Metal Reference Card */}
        <div className="p-5 rounded-xl bg-[#0e1422] border border-[#1e293b]">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-semibold text-sky-400 uppercase tracking-wider flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5" /> Metal Implementation (Current)
            </span>
            <span className="text-[11px] font-mono text-slate-400 bg-[#162032] px-2 py-0.5 rounded">
              Apple Silicon
            </span>
          </div>
          <div className="text-xs font-mono text-slate-200 bg-[#090d16] p-3 rounded-lg border border-[#1a2333] mb-3">
            dev/large-language-model/cpp/src/gpu_kernel/{kernel.metalFile}
          </div>
          <ul className="text-xs text-slate-400 space-y-2 leading-relaxed">
            <li>• Unified memory architecture with direct host-device pointer sharing.</li>
            <li>• Threadgroup dispatch with SIMD-width 32 execution.</li>
            <li>• <strong className="text-slate-300">Porting note:</strong> {spec?.metalVsCuda}</li>
          </ul>
        </div>

        {/* CUDA Target Card */}
        <div className="p-5 rounded-xl bg-[#0e1422] border border-[#1e293b]">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-semibold text-emerald-400 uppercase tracking-wider flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5" /> CUDA C++20 Target (Planned)
            </span>
            <span className="text-[11px] font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
              NVIDIA GPU
            </span>
          </div>
          <div className="text-xs font-mono text-emerald-300 bg-[#090d16] p-3 rounded-lg border border-[#1a2333] mb-3">
            src/cuda_kernels/{kernel.cudaFile}
          </div>
          <ul className="text-xs text-slate-400 space-y-2 leading-relaxed">
            <li>• Discrete VRAM with PCIe / NVLink transfers and pinned host memory.</li>
            <li>• Warp execution (32 threads) with warp-shuffle intrinsics.</li>
            <li>• Shared Memory (L1 SRAM) tile buffers with bank conflict avoidance.</li>
          </ul>
        </div>
      </div>

      {/* Architectural Specifications */}
      <div className="space-y-6">
        <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2 border-b border-[#1e293b] pb-2">
          <Layers className="w-4 h-4 text-sky-400" />
          Hardware & Memory Specifications
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 rounded-lg bg-[#0e1422] border border-[#1e293b]">
            <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2 flex items-center gap-1.5">
              <HardDrive className="w-3.5 h-3.5 text-sky-400" />
              Memory Access Pattern
            </h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              {spec?.memoryPattern}
            </p>
          </div>

          <div className="p-4 rounded-lg bg-[#0e1422] border border-[#1e293b]">
            <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2 flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-indigo-400" />
              Thread Layout & Warps
            </h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              {spec?.threadLayout}
            </p>
          </div>

          <div className="p-4 rounded-lg bg-[#0e1422] border border-[#1e293b]">
            <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2 flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-amber-400" />
              Key Optimizations
            </h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              {spec?.optimizations}
            </p>
          </div>
        </div>

        {/* Porting Roadmap Step */}
        <div className="p-5 rounded-xl bg-gradient-to-r from-sky-950/30 to-indigo-950/30 border border-sky-500/20">
          <h3 className="text-sm font-bold text-sky-200 mb-2 flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-sky-400" />
            Curriculum Roadmap Alignment
          </h3>
          <p className="text-xs text-slate-300 leading-relaxed">
            Writing this kernel is the ultimate milestone of this course. The pointer arithmetic and alignment concepts learned in <strong>Volume 1</strong> ensure coalesced 128-byte transactions. The warp and SRAM primitives from <strong>Volume 2</strong> prevent memory stall bubbles.
          </p>
        </div>
      </div>
    </div>
  );
}
