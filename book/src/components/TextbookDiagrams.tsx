'use client';

import React, { useState } from 'react';
import { Layers, Database, Cpu, Info, CheckCircle2, AlertTriangle } from 'lucide-react';

export interface PointerAddressingSpec {
  title?: string;
  subtitle?: string;
  pointer?: {
    name?: string;
    type?: string;
    location?: string;
    address?: string;
    value?: string;
    size?: string;
  };
  target?: {
    location?: string;
    baseAddress?: string;
    typeName?: string;
    totalBytes?: number;
    cells?: Array<{
      name: string;
      offset: string;
      address: string;
      bytes: string;
      hex?: string;
      val?: string | number;
    }>;
  };
}

export function PointerAddressingDiagram({ data }: { data: PointerAddressingSpec }) {
  const [activeCellIndex, setActiveCellIndex] = useState<number | null>(null);

  const ptrName = data.pointer?.name || 'ptr';
  const ptrType = data.pointer?.type || 'uint32_t*';
  const ptrAddress = data.pointer?.address || '0x7ffee2bc81a0';
  const ptrValue = data.pointer?.value || '0x1000';
  const ptrSize = data.pointer?.size || '8 bytes (64-bit)';
  const ptrStorage = data.pointer?.location || 'Stack Frame / CPU Register';

  const targetLocation = data.target?.location || 'RAM / VRAM (Global Memory Heap)';
  const baseAddress = data.target?.baseAddress || '0x1000';
  const targetTypeName = data.target?.typeName || 'uint32_t[4] (16 Bytes Total)';

  const defaultCells = [
    { name: `${ptrName}[0]`, offset: '+0', address: '0x1000', bytes: '4 bytes', hex: '0x0000002A', val: '42' },
    { name: `${ptrName}[1]`, offset: '+4', address: '0x1004', bytes: '4 bytes', hex: '0x00000054', val: '84' },
    { name: `${ptrName}[2]`, offset: '+8', address: '0x1008', bytes: '4 bytes', hex: '0x0000007E', val: '126' },
    { name: `${ptrName}[3]`, offset: '+12', address: '0x100C', bytes: '4 bytes', hex: '0x000000A8', val: '168' },
  ];

  const cells = data.target?.cells && data.target.cells.length > 0 ? data.target.cells : defaultCells;

  return (
    <div className="my-8 rounded-2xl border border-[#30363d] bg-[#0c1017] shadow-2xl overflow-hidden select-none">
      {/* Header bar */}
      <div className="flex items-center justify-between px-5 py-3.5 bg-[#141923] border-b border-[#21262d]">
        <div className="flex items-center gap-2.5">
          <span className="p-1.5 rounded-lg bg-amber-500/10 text-amber-400 border border-amber-500/20">
            <Database className="w-4 h-4" />
          </span>
          <div>
            <h4 className="text-xs font-bold text-slate-100 tracking-wide uppercase font-mono">
              {data.title || 'Textbook Schematic: 64-Bit Pointer Resolution & Byte Ladder'}
            </h4>
            <p className="text-[11px] text-slate-400 font-sans mt-0.5">
              {data.subtitle || 'Modeled after CS:APP memory schematics. Direct 64-bit address translation.'}
            </p>
          </div>
        </div>
        <div className="hidden sm:flex items-center gap-2">
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-sky-500/10 text-sky-400 border border-sky-500/20 font-semibold">
            64-BIT ARCHITECTURE
          </span>
        </div>
      </div>

      {/* SVG Canvas Area */}
      <div className="p-6 md:p-8 bg-[#090d14] relative">
        <svg
          viewBox="0 0 800 370"
          className="w-full h-auto max-w-[800px] mx-auto filter drop-shadow-md"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <defs>
            {/* Arrow Marker */}
            <marker
              id="pointer-arrow"
              viewBox="0 0 10 10"
              refX="6"
              refY="5"
              markerWidth="7"
              markerHeight="7"
              orient="auto"
            >
              <path d="M 0 1 L 10 5 L 0 9 z" fill="#38bdf8" />
            </marker>

            {/* Gradients */}
            <linearGradient id="stackGrad" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#1e1e38" />
              <stop offset="100%" stopColor="#141428" />
            </linearGradient>

            <linearGradient id="heapGrad" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#0f221b" />
              <stop offset="100%" stopColor="#0b1713" />
            </linearGradient>

            <linearGradient id="cellGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#14202c" />
              <stop offset="100%" stopColor="#0d151e" />
            </linearGradient>

            <linearGradient id="cellGradHover" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#1e344a" />
              <stop offset="100%" stopColor="#122230" />
            </linearGradient>

            <linearGradient id="arrowGrad" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#f59e0b" />
              <stop offset="50%" stopColor="#0ea5e9" />
              <stop offset="100%" stopColor="#38bdf8" />
            </linearGradient>
          </defs>

          {/* ================= SECTION 1: STACK / REGISTER ================= */}
          <g>
            {/* Region container */}
            <rect
              x="40"
              y="20"
              width="310"
              height="100"
              rx="10"
              fill="url(#stackGrad)"
              stroke="#4338ca"
              strokeWidth="1.5"
            />

            {/* Region label */}
            <text x="56" y="44" fill="#a5b4fc" fontSize="11" fontWeight="700" fontFamily="ui-monospace, monospace">
              REGION: {ptrStorage.toUpperCase()}
            </text>
            <text x="56" y="58" fill="#6366f1" fontSize="10" fontFamily="sans-serif">
              Stack Address: {ptrAddress}
            </text>

            {/* Pointer Variable Slot */}
            <rect
              x="56"
              y="68"
              width="278"
              height="40"
              rx="6"
              fill="#18182e"
              stroke="#6366f1"
              strokeWidth="1"
            />

            {/* Slot text */}
            <text x="70" y="86" fill="#e0e7ff" fontSize="12" fontWeight="600" fontFamily="ui-monospace, monospace">
              {ptrType} <tspan fill="#f59e0b">{ptrName}</tspan> ({ptrSize})
            </text>
            <text x="70" y="101" fill="#94a3b8" fontSize="11" fontFamily="ui-monospace, monospace">
              Stores Value: <tspan fill="#38bdf8" fontWeight="bold">{ptrValue}</tspan>
            </text>

            {/* Circle origin port directly under Stores Value */}
            <circle cx="180" cy="108" r="4.5" fill="#f59e0b" stroke="#ffffff" strokeWidth="1.5" />
          </g>

          {/* ================= SECTION 2: RAM / VRAM HEAP TARGET ================= */}
          <g transform="translate(40, 205)">
            {/* Memory Region Container */}
            <rect
              x="0"
              y="0"
              width="720"
              height="140"
              rx="12"
              fill="url(#heapGrad)"
              stroke="#059669"
              strokeWidth="1.5"
            />

            {/* Region Label (positioned to the right of ptr[0] entry path) */}
            <text x="230" y="24" fill="#6ee7b7" fontSize="11" fontWeight="700" fontFamily="ui-monospace, monospace">
              TARGET BUFFER: {targetLocation.toUpperCase()}
            </text>
            <text x="230" y="38" fill="#34d399" fontSize="10" fontFamily="sans-serif">
              Base Virtual Address: <tspan fill="#38bdf8" fontWeight="bold">{baseAddress}</tspan> • Type: {targetTypeName}
            </text>

            {/* Memory Byte Ladder Cells */}
            {cells.map((cell, idx) => {
              const cellWidth = 160;
              const cellX = 20 + idx * 172;
              const isSelected = activeCellIndex === idx;

              return (
                <g
                  key={idx}
                  transform={`translate(${cellX}, 50)`}
                  className="cursor-pointer transition-all"
                  onMouseEnter={() => setActiveCellIndex(idx)}
                  onMouseLeave={() => setActiveCellIndex(null)}
                >
                  {/* Cell Box */}
                  <rect
                    x="0"
                    y="0"
                    width={cellWidth}
                    height="72"
                    rx="8"
                    fill={isSelected ? 'url(#cellGradHover)' : 'url(#cellGrad)'}
                    stroke={isSelected ? '#38bdf8' : '#334155'}
                    strokeWidth={isSelected ? '2' : '1'}
                  />

                  {/* Cell Element Name */}
                  <text x="12" y="20" fill={isSelected ? '#38bdf8' : '#f1f5f9'} fontSize="12" fontWeight="bold" fontFamily="ui-monospace, monospace">
                    {cell.name}
                  </text>

                  {/* Offset & Byte Size */}
                  <rect x="100" y="8" width="48" height="16" rx="4" fill="#1e293b" />
                  <text x="124" y="20" textAnchor="middle" fill="#94a3b8" fontSize="9" fontFamily="ui-monospace, monospace">
                    {cell.bytes}
                  </text>

                  {/* Address */}
                  <text x="12" y="38" fill="#64748b" fontSize="10" fontFamily="ui-monospace, monospace">
                    Addr: <tspan fill="#38bdf8">{cell.address}</tspan> <tspan fill="#94a3b8">({cell.offset})</tspan>
                  </text>

                  {/* Stored Value */}
                  <text x="12" y="58" fill="#10b981" fontSize="12" fontWeight="bold" fontFamily="ui-monospace, monospace">
                    Val: {cell.val} <tspan fill="#64748b" fontSize="10">({cell.hex})</tspan>
                  </text>
                </g>
              );
            })}
          </g>

          {/* ================= TOP-LAYER POINTER ARROW & DEREFERENCE PILL ================= */}
          {/* Origin at (180, 108), flows through (160, 165) and plunges vertically into ptr[0] at (140, 245) */}
          <path
            d="M 180 108 C 180 165 140 185 140 245"
            fill="none"
            stroke="url(#arrowGrad)"
            strokeWidth="2.5"
            strokeDasharray="4 2"
          />

          {/* Concrete SVG Arrowhead on ptr[0] */}
          <polygon
            points="140,255 133,241 147,241"
            fill="#38bdf8"
            stroke="#090d14"
            strokeWidth="1.5"
          />
          {/* Landing Target Ring on ptr[0] */}
          <circle cx="140" cy="255" r="3.5" fill="#38bdf8" stroke="#ffffff" strokeWidth="1" />

          {/* Dereferences (*ptr) Pill dynamically sized and centered on curve at (160, 165) */}
          {(() => {
            const derefLabel = `Dereferences (*${ptrName})`;
            const pillWidth = Math.max(165, derefLabel.length * 8.2 + 36);
            const pillX = Math.max(20, 160 - pillWidth / 2);
            return (
              <g transform={`translate(${pillX}, 152)`}>
                <rect
                  x="0"
                  y="0"
                  width={pillWidth}
                  height="26"
                  rx="13"
                  fill="#0b1329"
                  stroke="#38bdf8"
                  strokeWidth="1.5"
                />
                <text
                  x={pillWidth / 2}
                  y="17"
                  textAnchor="middle"
                  fill="#38bdf8"
                  fontSize="11"
                  fontWeight="700"
                  fontFamily="ui-monospace, monospace"
                >
                  {derefLabel}
                </text>
              </g>
            );
          })()}
        </svg>

        {/* Dynamic Detail Card on Hover */}
        <div className="mt-4 p-3 rounded-xl bg-[#121824] border border-[#212836] flex items-center justify-between text-xs">
          <div className="flex items-center gap-2">
            <Info className="w-4 h-4 text-sky-400 shrink-0" />
            <span className="text-slate-300 font-sans">
              {activeCellIndex !== null ? (
                <>
                  Inspecting <strong className="text-sky-300 font-mono">{cells[activeCellIndex].name}</strong>:
                  Physical address <span className="font-mono text-amber-300">{cells[activeCellIndex].address}</span>, offset
                  by <span className="font-mono text-emerald-300">{cells[activeCellIndex].offset}</span> bytes from pointer base.
                </>
              ) : (
                'Hover over any memory cell above to inspect byte boundaries, scaling calculations, and physical addresses.'
              )}
            </span>
          </div>
          <span className="text-[11px] font-mono text-slate-500 hidden md:block">
            Addr = Base + (Index * sizeof(T))
          </span>
        </div>
      </div>
    </div>
  );
}

export function CacheLineDiagram({
  title = 'Hardware Cache Line Alignment: 64-Byte Burst vs Unaligned Straddling',
  subtitle = 'DRAM memory controller transfers memory in discrete 64-byte aligned sectors.',
}: {
  title?: string;
  subtitle?: string;
}) {
  return (
    <div className="my-8 rounded-2xl border border-[#30363d] bg-[#0c1017] shadow-2xl overflow-hidden select-none">
      <div className="flex items-center justify-between px-5 py-3.5 bg-[#141923] border-b border-[#21262d]">
        <div className="flex items-center gap-2.5">
          <span className="p-1.5 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <Cpu className="w-4 h-4" />
          </span>
          <div>
            <h4 className="text-xs font-bold text-slate-100 tracking-wide uppercase font-mono">{title}</h4>
            <p className="text-[11px] text-slate-400 font-sans mt-0.5">{subtitle}</p>
          </div>
        </div>
        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-semibold">
          BUS PROTOCOL
        </span>
      </div>

      <div className="p-6 md:p-8 bg-[#090d14]">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Case 1: Aligned */}
          <div className="rounded-xl bg-[#0e141e] border border-emerald-500/30 p-4">
            <div className="flex items-center justify-between mb-3 pb-2 border-b border-emerald-500/20">
              <span className="text-xs font-bold text-emerald-400 flex items-center gap-1.5 font-mono">
                <CheckCircle2 className="w-4 h-4" /> CASE 1: ALIGNED ACCESS
              </span>
              <span className="text-[11px] font-mono text-emerald-300 bg-emerald-950/40 px-2 py-0.5 rounded">
                1 Memory Transaction
              </span>
            </div>

            <p className="text-xs text-slate-300 mb-3">
              Base address <code className="text-emerald-300 font-mono">0x1000</code> is divisible by 64 (0x1000 mod 64 == 0). A 16-byte vector load fits entirely within Cache Line 0.
            </p>

            <div className="space-y-2 font-mono text-xs">
              <div className="flex justify-between text-[11px] text-slate-400">
                <span>0x1000 (Line 0 Base)</span>
                <span>0x103F (Line 0 End)</span>
              </div>
              <div className="h-10 w-full bg-[#16202e] rounded-lg border border-slate-700 p-1 flex gap-1">
                <div className="w-1/4 h-full bg-emerald-500/30 rounded border border-emerald-400 flex items-center justify-center text-[10px] font-bold text-emerald-200">
                  16B Float4 [0..15]
                </div>
                <div className="flex-1 h-full bg-[#0d1520] rounded flex items-center justify-center text-[10px] text-slate-500">
                  48 Bytes Remaining
                </div>
              </div>
            </div>

            <div className="mt-4 p-2.5 rounded-lg bg-emerald-950/20 border border-emerald-500/20 text-[11px] text-emerald-300 flex items-center gap-2">
              <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
              <span>Full 100% bus utilization. Zero pipeline stall cycles.</span>
            </div>
          </div>

          {/* Case 2: Unaligned Straddle */}
          <div className="rounded-xl bg-[#0e141e] border border-rose-500/30 p-4">
            <div className="flex items-center justify-between mb-3 pb-2 border-b border-rose-500/20">
              <span className="text-xs font-bold text-rose-400 flex items-center gap-1.5 font-mono">
                <AlertTriangle className="w-4 h-4" /> CASE 2: UNALIGNED STRADDLE
              </span>
              <span className="text-[11px] font-mono text-rose-300 bg-rose-950/40 px-2 py-0.5 rounded">
                2 Memory Transactions
              </span>
            </div>

            <p className="text-xs text-slate-300 mb-3">
              Base address <code className="text-rose-300 font-mono">0x103A</code> is offset by 58 bytes. A 16-byte load spans across the 64-byte boundary at <code className="text-rose-300 font-mono">0x1040</code>.
            </p>

            <div className="space-y-2 font-mono text-xs">
              <div className="flex justify-between text-[11px] text-slate-400">
                <span>Line 0 (0x1000)</span>
                <span className="text-rose-400 font-bold">Boundary (0x1040)</span>
                <span>Line 1 (0x107F)</span>
              </div>
              <div className="h-10 w-full bg-[#16202e] rounded-lg border border-slate-700 p-1 flex gap-1">
                <div className="w-[45%] h-full bg-[#0d1520] rounded flex items-center justify-center text-[10px] text-slate-500">
                  Line 0 Bytes
                </div>
                <div className="w-[20%] h-full bg-rose-500/30 rounded border border-rose-400 flex items-center justify-center text-[10px] font-bold text-rose-200">
                  6B
                </div>
                <div className="w-[20%] h-full bg-rose-500/50 rounded border border-rose-300 flex items-center justify-center text-[10px] font-bold text-rose-100">
                  10B
                </div>
                <div className="w-[15%] h-full bg-[#0d1520] rounded flex items-center justify-center text-[10px] text-slate-500">
                  ...
                </div>
              </div>
            </div>

            <div className="mt-4 p-2.5 rounded-lg bg-rose-950/20 border border-rose-500/20 text-[11px] text-rose-300 flex items-center gap-2">
              <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
              <span>DRAM hardware issues 2 fetch cycles for 1 vector. 2x latency penalty!</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export function TiledGemmDiagram({
  title = '2D Shared Memory Matrix Tiling Schematic (BM x BN Tile)',
  subtitle = 'High-bandwidth on-chip SRAM cache buffering eliminating DRAM round-trips.',
}: {
  title?: string;
  subtitle?: string;
}) {
  return (
    <div className="my-8 rounded-2xl border border-[#30363d] bg-[#0c1017] shadow-2xl overflow-hidden select-none">
      <div className="flex items-center justify-between px-5 py-3.5 bg-[#141923] border-b border-[#21262d]">
        <div className="flex items-center gap-2.5">
          <span className="p-1.5 rounded-lg bg-purple-500/10 text-purple-400 border border-purple-500/20">
            <Layers className="w-4 h-4" />
          </span>
          <div>
            <h4 className="text-xs font-bold text-slate-100 tracking-wide uppercase font-mono">{title}</h4>
            <p className="text-[11px] text-slate-400 font-sans mt-0.5">{subtitle}</p>
          </div>
        </div>
        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-purple-500/10 text-purple-400 border border-purple-500/20 font-semibold">
          CUDA TILING ALGORITHM
        </span>
      </div>

      <div className="p-6 md:p-8 bg-[#090d14]">
        <svg
          viewBox="0 0 800 280"
          className="w-full h-auto max-w-[800px] mx-auto filter drop-shadow-md"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          {/* Matrix A Tile */}
          <g transform="translate(40, 30)">
            <rect x="0" y="0" width="160" height="140" rx="8" fill="#141a28" stroke="#38bdf8" strokeWidth="1.5" />
            <text x="80" y="24" textAnchor="middle" fill="#7dd3fc" fontSize="11" fontWeight="bold" fontFamily="ui-monospace, monospace">
              MATRIX A [M x K]
            </text>
            <rect x="20" y="40" width="120" height="50" rx="4" fill="#0284c7" fillOpacity="0.3" stroke="#38bdf8" strokeDasharray="3 3" />
            <text x="80" y="70" textAnchor="middle" fill="#e0f2fe" fontSize="11" fontWeight="bold" fontFamily="ui-monospace, monospace">
              sh_A [BM x BK]
            </text>
            <text x="80" y="115" textAnchor="middle" fill="#94a3b8" fontSize="10" fontFamily="sans-serif">
              Global DRAM Buffer
            </text>
          </g>

          {/* Multiplication Operator */}
          <text x="240" y="110" fill="#f8fafc" fontSize="24" fontWeight="bold" textAnchor="middle" fontFamily="sans-serif">
            ×
          </text>

          {/* Matrix B Tile */}
          <g transform="translate(280, 30)">
            <rect x="0" y="0" width="160" height="140" rx="8" fill="#141a28" stroke="#a855f7" strokeWidth="1.5" />
            <text x="80" y="24" textAnchor="middle" fill="#d8b4fe" fontSize="11" fontWeight="bold" fontFamily="ui-monospace, monospace">
              MATRIX B [K x N]
            </text>
            <rect x="40" y="35" width="80" height="70" rx="4" fill="#9333ea" fillOpacity="0.3" stroke="#c084fc" strokeDasharray="3 3" />
            <text x="80" y="75" textAnchor="middle" fill="#faf5ff" fontSize="11" fontWeight="bold" fontFamily="ui-monospace, monospace">
              sh_B [BK x BN]
            </text>
            <text x="80" y="125" textAnchor="middle" fill="#94a3b8" fontSize="10" fontFamily="sans-serif">
              Global DRAM Buffer
            </text>
          </g>

          {/* Arrow pointing to C */}
          <path d="M 455 100 L 510 100" stroke="#f59e0b" strokeWidth="2.5" />
          <polygon points="522,100 510,94 510,106" fill="#f59e0b" />

          {/* Matrix C Output Tile */}
          <g transform="translate(530, 20)">
            <rect x="0" y="0" width="220" height="160" rx="10" fill="#0f1f1a" stroke="#10b981" strokeWidth="2" />
            <text x="110" y="26" textAnchor="middle" fill="#6ee7b7" fontSize="12" fontWeight="bold" fontFamily="ui-monospace, monospace">
              OUTPUT MATRIX C [M x N]
            </text>

            <rect x="40" y="45" width="140" height="70" rx="6" fill="#059669" fillOpacity="0.35" stroke="#34d399" />
            <text x="110" y="75" textAnchor="middle" fill="#ffffff" fontSize="12" fontWeight="bold" fontFamily="ui-monospace, monospace">
              BLOCK TILE [BM x BN]
            </text>
            <text x="110" y="95" textAnchor="middle" fill="#a7f3d0" fontSize="10" fontFamily="sans-serif">
              Accumulated in Registers
            </text>

            <text x="110" y="140" textAnchor="middle" fill="#34d399" fontSize="10" fontFamily="ui-monospace, monospace">
              Written back to DRAM once
            </text>
          </g>
        </svg>

        <div className="mt-4 p-3 rounded-xl bg-[#121824] border border-[#212836] text-xs text-slate-300 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Info className="w-4 h-4 text-purple-400 shrink-0" />
            <span>
              Instead of reading elements from high-latency Global DRAM for every multiply, tiles are copied once to on-chip <strong className="text-purple-300">Shared Memory</strong> and reused across the entire warp block.
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

export function PitchedMemoryDiagram({
  title = "Hardware Schematic: 2D Pitched Memory in DRAM (cudaMallocPitch)",
  subtitle = "Hardware row padding aligns each row start to a 256-byte coalescing boundary."
}: { title?: string; subtitle?: string }) {
  return (
    <div className="my-8 rounded-2xl border border-[#30363d] bg-[#0c1017] shadow-2xl overflow-hidden select-none">
      <div className="flex items-center justify-between px-5 py-3.5 bg-[#141923] border-b border-[#21262d]">
        <div className="flex items-center gap-2.5">
          <span className="p-1.5 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <Database className="w-4 h-4" />
          </span>
          <div>
            <h4 className="text-xs font-bold text-slate-100 tracking-wide uppercase font-mono">
              {title}
            </h4>
            <p className="text-[11px] text-slate-400 font-sans mt-0.5">
              {subtitle}
            </p>
          </div>
        </div>
        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-semibold">
          PITCH = 256 BYTES
        </span>
      </div>

      <div className="p-6 md:p-8 bg-[#090d14] relative">
        <svg viewBox="0 0 760 260" className="w-full h-auto max-w-[760px] mx-auto filter drop-shadow-md">
          <defs>
            <pattern id="diagonalHatch" width="8" height="8" patternTransform="rotate(45 0 0)" patternUnits="userSpaceOnUse">
              <line x1="0" y1="0" x2="0" y2="8" stroke="#f59e0b" strokeWidth="1.5" strokeOpacity="0.4" />
            </pattern>
          </defs>

          {/* Row 0 */}
          <g transform="translate(40, 25)">
            <text x="0" y="27" fill="#94a3b8" fontSize="11" fontWeight="bold" fontFamily="ui-monospace, monospace">
              Row 0 [Base + 0]
            </text>
            
            {/* Useful Data: 128 Bytes */}
            <rect x="135" y="5" width="280" height="42" rx="6" fill="#064e3b" stroke="#10b981" strokeWidth="1.5" />
            <text x="275" y="30" textAnchor="middle" fill="#a7f3d0" fontSize="11" fontWeight="bold" fontFamily="ui-monospace, monospace">
              Useful Data (32 Floats = 128 Bytes)
            </text>

            {/* Padding: 128 Bytes */}
            <rect x="425" y="5" width="235" height="42" rx="6" fill="url(#diagonalHatch)" stroke="#d97706" strokeWidth="1.5" strokeDasharray="3 3" />
            <text x="542" y="30" textAnchor="middle" fill="#fcd34d" fontSize="11" fontWeight="bold" fontFamily="ui-monospace, monospace">
              Padding Slack (128 Bytes)
            </text>

            {/* Total Row 0 indicator */}
            <text x="670" y="30" fill="#64748b" fontSize="11" fontFamily="ui-monospace, monospace">
              -&gt; 256 B
            </text>
          </g>

          {/* Pitch Stride Arrow connecting Row 0 to Row 1 */}
          <path d="M 125 50 C 75 85 75 115 125 145" fill="none" stroke="#38bdf8" strokeWidth="2" strokeDasharray="4 2" />
          <polygon points="132,145 122,139 122,151" fill="#38bdf8" />
          <g transform="translate(18, 86)">
            <rect x="0" y="0" width="90" height="22" rx="11" fill="#0b1329" stroke="#38bdf8" strokeWidth="1" />
            <text x="45" y="15" textAnchor="middle" fill="#38bdf8" fontSize="9" fontWeight="bold" fontFamily="ui-monospace, monospace">
              + Pitch (256B)
            </text>
          </g>

          {/* Row 1 */}
          <g transform="translate(40, 125)">
            <text x="0" y="27" fill="#94a3b8" fontSize="11" fontWeight="bold" fontFamily="ui-monospace, monospace">
              Row 1 [Base + 256]
            </text>
            
            {/* Useful Data: 128 Bytes */}
            <rect x="135" y="5" width="280" height="42" rx="6" fill="#064e3b" stroke="#10b981" strokeWidth="1.5" />
            <text x="275" y="30" textAnchor="middle" fill="#a7f3d0" fontSize="11" fontWeight="bold" fontFamily="ui-monospace, monospace">
              Useful Data (32 Floats = 128 Bytes)
            </text>

            {/* Padding: 128 Bytes */}
            <rect x="425" y="5" width="235" height="42" rx="6" fill="url(#diagonalHatch)" stroke="#d97706" strokeWidth="1.5" strokeDasharray="3 3" />
            <text x="542" y="30" textAnchor="middle" fill="#fcd34d" fontSize="11" fontWeight="bold" fontFamily="ui-monospace, monospace">
              Padding Slack (128 Bytes)
            </text>

            {/* Total Row 1 indicator */}
            <text x="670" y="30" fill="#64748b" fontSize="11" fontFamily="ui-monospace, monospace">
              -&gt; 256 B
            </text>
          </g>

          {/* Dimension Guidelines below Row 1 */}
          <g transform="translate(175, 185)">
            {/* Width Dimension */}
            <line x1="0" y1="10" x2="280" y2="10" stroke="#10b981" strokeWidth="1.5" />
            <polygon points="0,10 6,7 6,13" fill="#10b981" />
            <polygon points="280,10 274,7 274,13" fill="#10b981" />
            <text x="140" y="28" textAnchor="middle" fill="#6ee7b7" fontSize="10" fontFamily="ui-monospace, monospace" fontWeight="bold">
              Width = 128 Bytes (Payload)
            </text>

            {/* Pitch Dimension */}
            <line x1="0" y1="45" x2="525" y2="45" stroke="#38bdf8" strokeWidth="1.5" />
            <polygon points="0,45 6,42 6,48" fill="#38bdf8" />
            <polygon points="525,45 519,42 519,48" fill="#38bdf8" />
            <text x="262" y="60" textAnchor="middle" fill="#7dd3fc" fontSize="10" fontFamily="ui-monospace, monospace" fontWeight="bold">
              Total Pitch = 256 Bytes (Hardware Cache-Line Stride)
            </text>
          </g>
        </svg>

        <div className="mt-4 p-3 rounded-xl bg-[#121824] border border-[#212836] text-xs text-slate-300 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Info className="w-4 h-4 text-emerald-400 shrink-0" />
            <span>
              By advancing each row by <strong className="text-emerald-300 font-mono">PITCH (256 bytes)</strong> rather than raw width, every row starts at a 256-byte aligned hardware boundary, guaranteeing coalesced single-cycle memory transactions.
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

export function DoubleBufferingDiagram({
  title = "Timeline Schematic: Asynchronous Double-Buffering (Pointer Ping-Pong)",
  subtitle = "Hiding PCIe latency by overlapping Host-to-Device transfer with GPU kernel computation."
}: { title?: string; subtitle?: string }) {
  return (
    <div className="my-8 rounded-2xl border border-[#30363d] bg-[#0c1017] shadow-2xl overflow-hidden select-none">
      <div className="flex items-center justify-between px-5 py-3.5 bg-[#141923] border-b border-[#21262d]">
        <div className="flex items-center gap-2.5">
          <span className="p-1.5 rounded-lg bg-sky-500/10 text-sky-400 border border-sky-500/20">
            <Cpu className="w-4 h-4" />
          </span>
          <div>
            <h4 className="text-xs font-bold text-slate-100 tracking-wide uppercase font-mono">
              {title}
            </h4>
            <p className="text-[11px] text-slate-400 font-sans mt-0.5">
              {subtitle}
            </p>
          </div>
        </div>
        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-sky-500/10 text-sky-400 border border-sky-500/20 font-semibold">
          ZERO-COPY POINTER SWAP
        </span>
      </div>

      <div className="p-6 md:p-8 bg-[#090d14] relative">
        <svg viewBox="0 0 760 270" className="w-full h-auto max-w-[760px] mx-auto filter drop-shadow-md">
          {/* Stream 1 Track: PCIe DMA */}
          <g transform="translate(40, 20)">
            <text x="0" y="24" fill="#38bdf8" fontSize="11" fontWeight="bold" fontFamily="ui-monospace, monospace">
              STREAM 1 (PCIe DMA Engine)
            </text>
            <rect x="235" y="5" width="445" height="36" rx="8" fill="#0c2340" stroke="#0284c7" strokeWidth="1.5" />
            <text x="457" y="27" textAnchor="middle" fill="#7dd3fc" fontSize="11" fontWeight="bold" fontFamily="ui-monospace, monospace">
              PCIe Fills transfer_buf (Next Batch N+1)
            </text>
          </g>

          {/* Parallel Concurrency Bracket */}
          <g transform="translate(40, 70)">
            <path d="M 235 0 L 215 0 L 215 45 L 235 45" fill="none" stroke="#f59e0b" strokeWidth="2" />
            <path d="M 680 0 L 700 0 L 700 45 L 680 45" fill="none" stroke="#f59e0b" strokeWidth="2" />
            <text x="457" y="26" textAnchor="middle" fill="#fcd34d" fontSize="10" fontWeight="bold" fontFamily="ui-monospace, monospace" letterSpacing="1.5">
              &lt;--- RUNNING SIMULTANEOUSLY IN HARDWARE ---&gt;
            </text>
          </g>

          {/* Stream 0 Track: GPU SM Cores */}
          <g transform="translate(40, 115)">
            <text x="0" y="24" fill="#a855f7" fontSize="11" fontWeight="bold" fontFamily="ui-monospace, monospace">
              STREAM 0 (GPU Tensor Cores)
            </text>
            <rect x="235" y="5" width="445" height="36" rx="8" fill="#2e1065" stroke="#9333ea" strokeWidth="1.5" />
            <text x="457" y="27" textAnchor="middle" fill="#d8b4fe" fontSize="11" fontWeight="bold" fontFamily="ui-monospace, monospace">
              GPU Computes on compute_buf (Active Batch N)
            </text>
          </g>

          {/* Sync Barrier */}
          <g transform="translate(40, 165)">
            <line x1="235" y1="12" x2="680" y2="12" stroke="#64748b" strokeWidth="2" strokeDasharray="6 4" />
            <rect x="345" y="0" width="225" height="24" rx="12" fill="#1e293b" stroke="#94a3b8" strokeWidth="1.5" />
            <text x="457" y="16" textAnchor="middle" fill="#f1f5f9" fontSize="10" fontWeight="bold" fontFamily="ui-monospace, monospace">
              Sync Barrier: Both Streams Complete
            </text>
          </g>

          {/* Instantaneous Pointer Swap Operation */}
          <g transform="translate(40, 205)">
            <rect x="235" y="5" width="445" height="38" rx="8" fill="#064e3b" stroke="#10b981" strokeWidth="1.5" />
            <text x="457" y="28" textAnchor="middle" fill="#a7f3d0" fontSize="11" fontWeight="bold" fontFamily="ui-monospace, monospace">
              std::swap(compute_buf, transfer_buf)  [ 0 Physical Bytes Moved! ]
            </text>
          </g>
        </svg>

        <div className="mt-4 p-3 rounded-xl bg-[#121824] border border-[#212836] text-xs text-slate-300 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
            <span>
              Transfer and computation run fully overlapped in parallel. When the barrier hits, swapping two 8-byte pointer variables takes a single CPU clock cycle (0 nanoseconds of data copying).
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

export function GenericDiagramRenderer({ type, payload }: { type: string; payload: string }) {
  let parsed: any = {};
  try {
    const trimmed = payload ? payload.trim() : '';
    if (trimmed.startsWith('{')) {
      parsed = JSON.parse(trimmed);
    }
  } catch (err) {
    console.error('Failed to parse diagram JSON payload', err);
  }

  const cleanType = type.toLowerCase().replace(/_/g, '-');

  if (cleanType.includes('pitched') || cleanType.includes('pitch') || cleanType.includes('2d-pitch')) {
    return <PitchedMemoryDiagram title={parsed.title} subtitle={parsed.subtitle} />;
  }

  if (cleanType.includes('double-buffering') || cleanType.includes('double-buf') || cleanType.includes('ping-pong') || cleanType.includes('timeline')) {
    return <DoubleBufferingDiagram title={parsed.title} subtitle={parsed.subtitle} />;
  }

  if (cleanType.includes('pointer') || cleanType.includes('offset') || cleanType.includes('ladder') || cleanType.includes('memory')) {
    return <PointerAddressingDiagram data={parsed} />;
  }

  if (cleanType.includes('cache') || cleanType.includes('align') || cleanType.includes('straddle')) {
    return <CacheLineDiagram title={parsed.title} subtitle={parsed.subtitle} />;
  }

  if (cleanType.includes('gemm') || cleanType.includes('tile') || cleanType.includes('matrix')) {
    return <TiledGemmDiagram title={parsed.title} subtitle={parsed.subtitle} />;
  }

  return <PointerAddressingDiagram data={parsed} />;
}

