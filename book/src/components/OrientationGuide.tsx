'use client';

import React from 'react';
import { BookOpen, Cpu, Terminal, Compass, Layers, CheckCircle2, ShieldCheck, Zap, ArrowRight, Database, Server } from 'lucide-react';
import { ActiveNode } from './SidebarTree';

interface OrientationGuideProps {
  onNavigate: (node: ActiveNode) => void;
}

export function OrientationGuide({ onNavigate }: OrientationGuideProps) {
  return (
    <div className="flex-1 overflow-y-auto bg-[#0d1117] p-8 md:p-12 lg:px-16 w-full select-text">
      <div className="max-w-4xl mx-auto space-y-10">
        {/* Hero Header */}
        <div className="border-b border-[#21262d] pb-8">
          <div className="flex items-center gap-2 text-xs font-mono text-sky-400 mb-3">
            <span className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-md bg-sky-500/10 border border-sky-500/20 font-medium">
              <Compass className="w-3.5 h-3.5" /> CURRICULUM ORIENTATION & ROADMAP
            </span>
          </div>
          <h1 className="text-3xl md:text-4xl font-extrabold text-[#f0f6fc] tracking-tight leading-tight">
            How to Read & Master This Curriculum
          </h1>
          <p className="mt-3 text-base text-[#8b949e] leading-relaxed">
            From physical byte arithmetic and zero-cost C++ abstractions to saturating GPU memory bandwidth, warp scheduling, and production LLM kernels.
          </p>

          <div className="mt-6 flex flex-wrap gap-3">
            <button
              onClick={() => onNavigate({ type: 'theory', volumeId: 'vol1', chapterId: '1.1' })}
              className="px-4 py-2 rounded-lg bg-[#1f6feb] hover:bg-[#388bfd] text-white text-xs font-semibold flex items-center gap-2 transition-colors cursor-pointer shadow-sm"
            >
              Start Chapter 1.1 Theory <ArrowRight className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => onNavigate({ type: 'playground', volumeId: 'vol1', playgroundType: 'cpp' })}
              className="px-4 py-2 rounded-lg bg-[#21262d] hover:bg-[#30363d] text-[#c9d1d9] text-xs font-semibold flex items-center gap-2 transition-colors cursor-pointer border border-[#30363d]"
            >
              <Terminal className="w-3.5 h-3.5 text-sky-400" /> Open C++ Playground
            </button>
          </div>
        </div>

        {/* Section 1: The Three Volumes Architecture */}
        <section className="space-y-4">
          <div className="flex items-center gap-2 text-xs font-bold text-[#f0f6fc] uppercase tracking-wider">
            <span className="w-2 h-2 rounded-full bg-sky-400"></span>
            <span>1. The Three-Volume Architectural Progression</span>
          </div>
          <p className="text-sm text-[#8b949e] leading-relaxed">
            GPU kernels do not exist in isolation. Writing optimal CUDA requires deep intuition for pointers, memory alignments, cache line structures, and value semantics. The course is partitioned into 3 progressive volumes:
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 rounded-xl bg-[#161b22] border border-[#30363d] hover:border-sky-500/40 transition-colors">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-mono font-bold text-sky-400">VOLUME 1</span>
                <span className="text-[10px] px-2 py-0.5 rounded bg-sky-500/10 text-sky-300 font-mono">5 Modules</span>
              </div>
              <h3 className="text-sm font-bold text-[#f0f6fc] mb-1">C++ Low-Level Systems</h3>
              <p className="text-xs text-[#8b949e] leading-relaxed">
                Pointer arithmetic, strides, memory alignment, cache lines, custom arenas, zero-copy buffers, SIMD, and acquire-release concurrency.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-[#161b22] border border-[#30363d] hover:border-indigo-500/40 transition-colors">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-mono font-bold text-indigo-400">VOLUME 2</span>
                <span className="text-[10px] px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-300 font-mono">6 Modules</span>
              </div>
              <h3 className="text-sm font-bold text-[#f0f6fc] mb-1">CUDA Hardware Execution</h3>
              <p className="text-xs text-[#8b949e] leading-relaxed">
                SMs, warps, divergent branches, coalesced DRAM transactions, bank-conflict-free shared SRAM, tree reductions, and WMMA Tensor Cores.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-[#161b22] border border-[#30363d] hover:border-amber-500/40 transition-colors">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-mono font-bold text-amber-400">VOLUME 3</span>
                <span className="text-[10px] px-2 py-0.5 rounded bg-amber-500/10 text-amber-300 font-mono">7 Modules</span>
              </div>
              <h3 className="text-sm font-bold text-[#f0f6fc] mb-1">Production GPU Kernels</h3>
              <p className="text-xs text-[#8b949e] leading-relaxed">
                Dense GEMM, Sparse/MoE, FlashAttention-2, Fused RMSNorm, Multimodal Audio/Video, INT4/FP8 Quantization, and AdamW Optimizers.
              </p>
            </div>
          </div>
        </section>

        {/* Section 2: The Three-Tier Practice Model */}
        <section className="space-y-4">
          <div className="flex items-center gap-2 text-xs font-bold text-[#f0f6fc] uppercase tracking-wider">
            <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
            <span>2. The Three-Tier Mastery Methodology</span>
          </div>
          <p className="text-sm text-[#8b949e] leading-relaxed">
            Every chapter provides three tiers of hands-on practice. You only master a technique when you progress through all three levels:
          </p>

          <div className="space-y-3 font-mono text-xs">
            <div className="flex items-start gap-3 p-3.5 rounded-lg bg-[#161b22] border border-[#30363d]">
              <span className="px-2 py-0.5 rounded bg-sky-500/10 text-sky-300 border border-sky-500/20 text-[10px] font-bold">
                TIER 1: BEGINNER
              </span>
              <div className="flex-1 font-sans text-xs text-[#c9d1d9]">
                <strong className="text-[#f0f6fc] font-semibold">Correctness & Memory Invariants:</strong> Write working, safe implementations. Verify indexing boundaries, pointer casts, and structural memory layout.
              </div>
            </div>

            <div className="flex items-start gap-3 p-3.5 rounded-lg bg-[#161b22] border border-[#30363d]">
              <span className="px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-300 border border-indigo-500/20 text-[10px] font-bold">
                TIER 2: INTERMEDIATE
              </span>
              <div className="flex-1 font-sans text-xs text-[#c9d1d9]">
                <strong className="text-[#f0f6fc] font-semibold">Hardware Alignment & Optimization:</strong> Align buffers to 64-byte/128-byte cache lines, eliminate bank conflicts in shared SRAM, and leverage SIMD or warp shuffle primitives.
              </div>
            </div>

            <div className="flex items-start gap-3 p-3.5 rounded-lg bg-[#161b22] border border-[#30363d]">
              <span className="px-2 py-0.5 rounded bg-amber-500/10 text-amber-300 border border-amber-500/20 text-[10px] font-bold">
                TIER 3: CHAMPION
              </span>
              <div className="flex-1 font-sans text-xs text-[#c9d1d9]">
                <strong className="text-[#f0f6fc] font-semibold">Roofline Saturation & Zero-Copy:</strong> Push throughput to physical memory bandwidth limits or Tensor Core theoretical compute peaks (TFLOPS).
              </div>
            </div>
          </div>
        </section>

        {/* Section 3: Interactive Sandbox Playgrounds */}
        <section className="space-y-4">
          <div className="flex items-center gap-2 text-xs font-bold text-[#f0f6fc] uppercase tracking-wider">
            <span className="w-2 h-2 rounded-full bg-purple-400"></span>
            <span>3. Three Dedicated Sandbox Playgrounds</span>
          </div>
          <p className="text-sm text-[#8b949e] leading-relaxed">
            Use the sandbox playgrounds to experiment, test hypotheses, and prototype ideas without touching workbook exercises:
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div
              onClick={() => onNavigate({ type: 'playground', volumeId: 'vol1', playgroundType: 'cpp' })}
              className="p-3.5 rounded-xl bg-[#161b22] border border-[#30363d] cursor-pointer hover:border-sky-500 transition-colors"
            >
              <div className="flex items-center gap-1.5 text-xs font-bold text-sky-400 mb-1">
                <Terminal className="w-3.5 h-3.5" /> C++ Systems Sandbox
              </div>
              <p className="text-[11px] text-[#8b949e]">
                playground.cpp — compiled with Clang -O3 -std=c++20. Ideal for testing pointer math, SIMD, and custom allocators.
              </p>
            </div>

            <div
              onClick={() => onNavigate({ type: 'playground', volumeId: 'vol2', playgroundType: 'cuda' })}
              className="p-3.5 rounded-xl bg-[#161b22] border border-[#30363d] cursor-pointer hover:border-indigo-500 transition-colors"
            >
              <div className="flex items-center gap-1.5 text-xs font-bold text-indigo-400 mb-1">
                <Cpu className="w-3.5 h-3.5" /> CUDA GPU Scratchpad
              </div>
              <p className="text-[11px] text-[#8b949e]">
                playground.cu — compiled with NVCC -O3 --extended-lambda. Ideal for testing warp intrinsics and device kernels.
              </p>
            </div>

            <div
              onClick={() => onNavigate({ type: 'playground', volumeId: 'vol3', playgroundType: 'kernel' })}
              className="p-3.5 rounded-xl bg-[#161b22] border border-[#30363d] cursor-pointer hover:border-amber-500 transition-colors"
            >
              <div className="flex items-center gap-1.5 text-xs font-bold text-amber-400 mb-1">
                <Zap className="w-3.5 h-3.5" /> Kernel Benchmark Lab
              </div>
              <p className="text-[11px] text-[#8b949e]">
                playground_kernel.cu — microbenchmarking harness to evaluate TFLOPS, memory bandwidth, and roofline efficiency.
              </p>
            </div>
          </div>
        </section>

        {/* Section 4: Strict Pedagogy Rules */}
        <section className="p-5 rounded-xl bg-[#161b22] border border-[#30363d]">
          <h3 className="text-xs font-bold text-[#f0f6fc] uppercase tracking-wider mb-2 flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-emerald-400" /> Strict Hardware-First Pedagogy
          </h3>
          <ul className="text-xs text-[#8b949e] space-y-2 list-disc list-inside leading-relaxed font-sans">
            <li><strong className="text-[#c9d1d9]">Intuitive Systems Math:</strong> Math is expressed through direct, practical arithmetic and physical hardware concepts (bytes, strides, cache lines, clocks).</li>
            <li><strong className="text-[#c9d1d9]">One Step at a Time:</strong> Each chapter introduces a single core architectural concept. Master it before proceeding.</li>
            <li><strong className="text-[#c9d1d9]">Practice in Playground First:</strong> Always prototype unfamiliar APIs or constructs in the playground before attempting chapter exercises.</li>
            <li><strong className="text-[#c9d1d9]">Hardware Intent First:</strong> Understand why the GPU needs a structure (e.g. why DRAM coalescing avoids bus stalls) before typing code.</li>
          </ul>
        </section>
      </div>
    </div>
  );
}
