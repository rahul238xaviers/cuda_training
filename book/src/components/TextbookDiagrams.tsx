'use client';

import React, { useState } from 'react';
import { Layers, Database, Cpu, Info, CheckCircle2, AlertTriangle } from 'lucide-react';

// ============================================================================
// Stride Layout Diagram — Chapter 1.2: 2D logical grid → flat 1D memory strip
// ============================================================================
export function StrideLayoutDiagram({ data }: { data: any }) {
  const title = data?.title || 'Row-Major Stride: 2D Logical Grid → Flat Physical Memory';
  const subtitle = data?.subtitle || 'float[Rows][Cols] stored in row-major order. Stride_Row = Cols elements. Stride_Col = 1.';
  const rows = data?.rows || 2;
  const cols = data?.cols || 4;
  const base = data?.base || '0x2000';
  const typeStr = data?.typeName || `float[${rows}][${cols}]`;

  // Generate cell labels: (r,c) and flat address
  const baseInt = 0x2000;
  const cells = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const offset = (r * cols + c) * 4;
      const addr = `0x${(baseInt + offset).toString(16).toUpperCase()}`;
      cells.push({ r, c, addr, offset, flat: r * cols + c });
    }
  }

  const cellW = 70;
  const cellH = 38;
  const gutter = 4;
  const gridStartX = 24;
  const gridStartY = 48;
  const flatStartY = 170;
  const rowColors = ['#0c2a4a', '#1a1a3a', '#0a2a1a', '#2a1a04'];
  const rowStrokes = ['#38bdf8', '#818cf8', '#34d399', '#f59e0b'];

  const svgW = gridStartX + cols * (cellW + gutter) + 40;
  const svgH = 260;

  return (
    <div className="my-8 rounded-2xl border border-[#30363d] bg-[#0c1017] shadow-2xl overflow-hidden select-none">
      <div className="flex items-center justify-between px-5 py-3.5 bg-[#141923] border-b border-[#21262d]">
        <div className="flex items-center gap-2.5">
          <span className="p-1.5 rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
            <Layers className="w-4 h-4" />
          </span>
          <div>
            <h4 className="text-xs font-bold text-slate-100 tracking-wide uppercase font-mono">{title}</h4>
            <p className="text-[11px] text-slate-400 font-sans mt-0.5">{subtitle}</p>
          </div>
        </div>
        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 font-semibold">ROW-MAJOR LAYOUT</span>
      </div>

      <div className="p-4 bg-[#090d14] overflow-x-auto">
        <svg viewBox={`0 0 ${svgW} ${svgH}`} className="w-full min-w-[480px]" aria-label="Stride layout diagram">

          {/* === LOGICAL 2D GRID === */}
          <text x={gridStartX} y="18" fill="#94a3b8" fontSize="10" fontWeight="bold" fontFamily="ui-monospace, monospace" letterSpacing="1">
            LOGICAL VIEW: {typeStr} (Row-Major)
          </text>

          {cells.map(({ r, c, flat }) => {
            const x = gridStartX + c * (cellW + gutter);
            const y = gridStartY + r * (cellH + gutter);
            const color = rowColors[r % rowColors.length];
            const stroke = rowStrokes[r % rowStrokes.length];
            return (
              <g key={`grid-${flat}`}>
                <rect x={x} y={y} width={cellW} height={cellH} rx="5" fill={color} stroke={stroke} strokeWidth="1.5" />
                <text x={x + cellW / 2} y={y + 14} textAnchor="middle" fill={stroke} fontSize="9" fontWeight="bold" fontFamily="ui-monospace, monospace">
                  [{r}][{c}]
                </text>
                <text x={x + cellW / 2} y={y + 26} textAnchor="middle" fill="#cbd5e1" fontSize="9" fontFamily="ui-monospace, monospace">
                  row*{cols}+{c}={flat}
                </text>
              </g>
            );
          })}

          {/* Row stride label */}
          {[...Array(rows)].map((_, r) => (
            <text key={`stride-r${r}`}
              x={gridStartX + cols * (cellW + gutter) + 6}
              y={gridStartY + r * (cellH + gutter) + 22}
              fill={rowStrokes[r % rowStrokes.length]} fontSize="9" fontFamily="ui-monospace, monospace">
              Row {r}
            </text>
          ))}

          {/* Stride_Row annotation */}
          <line
            x1={gridStartX + 0 * (cellW + gutter) + cellW / 2}
            y1={gridStartY + 0 * (cellH + gutter) + cellH + 4}
            x2={gridStartX + 0 * (cellW + gutter) + cellW / 2}
            y2={gridStartY + 1 * (cellH + gutter) - 4}
            stroke="#64748b" strokeWidth="1" strokeDasharray="3 2"
          />
          <text x={gridStartX + cellW + 10} y={gridStartY + cellH + (cellH + gutter) / 2 - 6} fill="#64748b" fontSize="9" fontFamily="ui-monospace, monospace">
            Stride_Row = {cols} elements ({cols * 4}B)
          </text>

          {/* Down arrow from grid to flat */}
          <line x1={svgW / 2} y1={gridStartY + rows * (cellH + gutter) + 4} x2={svgW / 2} y2={flatStartY - 10} stroke="#475569" strokeWidth="1.5" strokeDasharray="4 3" markerEnd="url(#arrowGrey)" />
          <text x={svgW / 2} y={gridStartY + rows * (cellH + gutter) + 16} textAnchor="middle" fill="#64748b" fontSize="9" fontFamily="ui-monospace, monospace">
            flatten → physical memory
          </text>

          {/* === FLAT PHYSICAL MEMORY STRIP === */}
          <text x={gridStartX} y={flatStartY + 12} fill="#94a3b8" fontSize="10" fontWeight="bold" fontFamily="ui-monospace, monospace" letterSpacing="1">
            PHYSICAL MEMORY: Base = {base}
          </text>

          {cells.map(({ r, c, addr, offset, flat }) => {
            const x = gridStartX + flat * (cellW + gutter);
            const color = rowColors[r % rowColors.length];
            const stroke = rowStrokes[r % rowStrokes.length];
            return (
              <g key={`flat-${flat}`}>
                <rect x={x} y={flatStartY + 18} width={cellW} height={cellH} rx="5" fill={color} stroke={stroke} strokeWidth="1.5" />
                <text x={x + cellW / 2} y={flatStartY + 30} textAnchor="middle" fill={stroke} fontSize="8" fontWeight="bold" fontFamily="ui-monospace, monospace">
                  [{r}][{c}]
                </text>
                <text x={x + cellW / 2} y={flatStartY + 41} textAnchor="middle" fill="#94a3b8" fontSize="8" fontFamily="ui-monospace, monospace">
                  +{offset}B
                </text>
                <text x={x + cellW / 2} y={flatStartY + 72} textAnchor="middle" fill="#475569" fontSize="7.5" fontFamily="ui-monospace, monospace">
                  {addr}
                </text>
              </g>
            );
          })}

          <defs>
            <marker id="arrowGrey" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto">
              <path d="M0,0 L0,6 L7,3 z" fill="#475569" />
            </marker>
          </defs>
        </svg>

        <div className="mt-3 grid grid-cols-3 gap-3 text-xs">
          <div className="p-2.5 rounded-lg bg-sky-950/30 border border-sky-500/20 text-sky-300 font-mono text-center">
            <div className="font-bold mb-0.5">Row Stride</div>
            <div className="text-[10px] opacity-80">{cols} elements = {cols * 4}B</div>
          </div>
          <div className="p-2.5 rounded-lg bg-indigo-950/30 border border-indigo-500/20 text-indigo-300 font-mono text-center">
            <div className="font-bold mb-0.5">Col Stride</div>
            <div className="text-[10px] opacity-80">1 element = 4B</div>
          </div>
          <div className="p-2.5 rounded-lg bg-emerald-950/30 border border-emerald-500/20 text-emerald-300 font-mono text-center">
            <div className="font-bold mb-0.5">Address Formula</div>
            <div className="text-[10px] opacity-80">Base + (r*{cols}+c)*4</div>
          </div>
        </div>
      </div>
    </div>
  );
}

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

// ============================================================================
// Pointer Swap Diagram — Chapter 1.3: O(1) zero-copy register-level swap
// ============================================================================
export function PointerSwapDiagram({ data }: { data: any }) {
  const title = data?.title || 'O(1) Zero-Copy Pointer Swap';
  const subtitle = data?.subtitle || 'Swapping two 64-bit register variables redirects entire data buffers without touching physical memory.';
  const bufALabel = data?.bufA || 'Buffer A (4 KB)';
  const bufBLabel = data?.bufB || 'Buffer B (4 KB)';
  const addrA = data?.addrA || '0x1000';
  const addrB = data?.addrB || '0x5000';

  return (
    <div className="my-8 rounded-2xl border border-[#30363d] bg-[#0c1017] shadow-2xl overflow-hidden select-none">
      <div className="flex items-center justify-between px-5 py-3.5 bg-[#141923] border-b border-[#21262d]">
        <div className="flex items-center gap-2.5">
          <span className="p-1.5 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <Layers className="w-4 h-4" />
          </span>
          <div>
            <h4 className="text-xs font-bold text-slate-100 tracking-wide uppercase font-mono">{title}</h4>
            <p className="text-[11px] text-slate-400 font-sans mt-0.5">{subtitle}</p>
          </div>
        </div>
        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-semibold">REGISTER SWAP</span>
      </div>

      <div className="p-6 bg-[#090d14]">
        <svg viewBox="0 0 760 220" className="w-full" aria-label="Pointer swap diagram">
          {/* === BEFORE SWAP === */}
          <text x="30" y="20" fill="#94a3b8" fontSize="11" fontWeight="bold" fontFamily="ui-monospace, monospace" letterSpacing="1">BEFORE std::swap()</text>

          {/* read_ptr register */}
          <rect x="30" y="30" width="150" height="44" rx="8" fill="#1c2233" stroke="#3b82f6" strokeWidth="1.5" />
          <text x="105" y="48" textAnchor="middle" fill="#93c5fd" fontSize="10" fontWeight="bold" fontFamily="ui-monospace, monospace">read_ptr</text>
          <text x="105" y="62" textAnchor="middle" fill="#60a5fa" fontSize="11" fontFamily="ui-monospace, monospace">{addrA}</text>

          {/* write_ptr register */}
          <rect x="30" y="90" width="150" height="44" rx="8" fill="#1c2233" stroke="#8b5cf6" strokeWidth="1.5" />
          <text x="105" y="108" textAnchor="middle" fill="#c4b5fd" fontSize="10" fontWeight="bold" fontFamily="ui-monospace, monospace">write_ptr</text>
          <text x="105" y="122" textAnchor="middle" fill="#a78bfa" fontSize="11" fontFamily="ui-monospace, monospace">{addrB}</text>

          {/* Buffer A block */}
          <rect x="250" y="30" width="160" height="44" rx="8" fill="#14291f" stroke="#10b981" strokeWidth="1.5" />
          <text x="330" y="48" textAnchor="middle" fill="#6ee7b7" fontSize="10" fontWeight="bold" fontFamily="ui-monospace, monospace">BUFFER A</text>
          <text x="330" y="62" textAnchor="middle" fill="#34d399" fontSize="10" fontFamily="ui-monospace, monospace">{addrA} — {bufALabel}</text>

          {/* Buffer B block */}
          <rect x="250" y="90" width="160" height="44" rx="8" fill="#1e1b2e" stroke="#a855f7" strokeWidth="1.5" />
          <text x="330" y="108" textAnchor="middle" fill="#e9d5ff" fontSize="10" fontWeight="bold" fontFamily="ui-monospace, monospace">BUFFER B</text>
          <text x="330" y="122" textAnchor="middle" fill="#c084fc" fontSize="10" fontFamily="ui-monospace, monospace">{addrB} — {bufBLabel}</text>

          {/* Arrow: read_ptr → A */}
          <line x1="180" y1="52" x2="248" y2="52" stroke="#3b82f6" strokeWidth="2" markerEnd="url(#arrowBlue)" />
          {/* Arrow: write_ptr → B */}
          <line x1="180" y1="112" x2="248" y2="112" stroke="#8b5cf6" strokeWidth="2" markerEnd="url(#arrowPurple)" />

          {/* === SWAP OPERATION === */}
          <rect x="475" y="55" width="180" height="46" rx="10" fill="#062318" stroke="#10b981" strokeWidth="2" />
          <text x="565" y="74" textAnchor="middle" fill="#86efac" fontSize="10" fontWeight="bold" fontFamily="ui-monospace, monospace">std::swap(read_ptr,</text>
          <text x="565" y="88" textAnchor="middle" fill="#86efac" fontSize="10" fontWeight="bold" fontFamily="ui-monospace, monospace">           write_ptr)</text>

          {/* === AFTER SWAP === */}
          <text x="475" y="145" fill="#94a3b8" fontSize="11" fontWeight="bold" fontFamily="ui-monospace, monospace" letterSpacing="1">AFTER: read_ptr → B, write_ptr → A</text>
          <text x="475" y="163" fill="#10b981" fontSize="10" fontFamily="ui-monospace, monospace">Only 3 register moves (temp = a; a = b; b = temp)</text>
          <text x="475" y="177" fill="#10b981" fontSize="10" fontFamily="ui-monospace, monospace">0 bytes of physical data copied</text>
          <text x="475" y="191" fill="#10b981" fontSize="10" fontFamily="ui-monospace, monospace">O(1) — independent of buffer size!</text>

          {/* Arrows markers */}
          <defs>
            <marker id="arrowBlue" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
              <path d="M0,0 L0,6 L8,3 z" fill="#3b82f6" />
            </marker>
            <marker id="arrowPurple" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
              <path d="M0,0 L0,6 L8,3 z" fill="#8b5cf6" />
            </marker>
          </defs>
        </svg>

        <div className="mt-3 p-3 rounded-xl bg-[#121824] border border-[#212836] text-xs text-slate-300 flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
          <span>The addresses stored in <code className="text-sky-300 font-mono">read_ptr</code> and <code className="text-purple-300 font-mono">write_ptr</code> are exchanged in 3 register instructions. The data in DRAM/VRAM is untouched — only the 8-byte pointer variables change.</span>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Strided Access Diagram — Chapter 1.4: Dual-stream SAXPY traversal
// ============================================================================
export function StridedAccessDiagram({ data }: { data: any }) {
  const title = data?.title || 'Dual-Stream Vector Traversal: SAXPY';
  const subtitle = data?.subtitle || 'Two independent pointer streams traverse separate arrays simultaneously, limited by memory bus bandwidth.';

  const numCells = 6;
  const cellW = 64;
  const gutter = 8;
  const startX = 30;

  return (
    <div className="my-8 rounded-2xl border border-[#30363d] bg-[#0c1017] shadow-2xl overflow-hidden select-none">
      <div className="flex items-center justify-between px-5 py-3.5 bg-[#141923] border-b border-[#21262d]">
        <div className="flex items-center gap-2.5">
          <span className="p-1.5 rounded-lg bg-sky-500/10 text-sky-400 border border-sky-500/20">
            <Database className="w-4 h-4" />
          </span>
          <div>
            <h4 className="text-xs font-bold text-slate-100 tracking-wide uppercase font-mono">{title}</h4>
            <p className="text-[11px] text-slate-400 font-sans mt-0.5">{subtitle}</p>
          </div>
        </div>
        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-sky-500/10 text-sky-400 border border-sky-500/20 font-semibold">MEMORY BANDWIDTH BOUND</span>
      </div>

      <div className="p-6 bg-[#090d14]">
        <svg viewBox="0 0 760 230" className="w-full" aria-label="Dual stream memory diagram">
          {/* === Array X === */}
          <text x={startX} y="24" fill="#38bdf8" fontSize="11" fontWeight="bold" fontFamily="ui-monospace, monospace">READ: x[ ] — Base 0x1000</text>
          {[...Array(numCells)].map((_, i) => (
            <g key={`x${i}`} transform={`translate(${startX + i * (cellW + gutter)}, 30)`}>
              <rect width={cellW} height="40" rx="6" fill={i === 0 ? '#0c2a4a' : '#0d1520'} stroke={i === 0 ? '#38bdf8' : '#1e3046'} strokeWidth={i === 0 ? 2 : 1} />
              <text x={cellW / 2} y="17" textAnchor="middle" fill="#94a3b8" fontSize="9" fontFamily="ui-monospace, monospace">x[{i}]</text>
              <text x={cellW / 2} y="30" textAnchor="middle" fill={i === 0 ? '#7dd3fc' : '#475569'} fontSize="10" fontWeight={i === 0 ? 'bold' : 'normal'} fontFamily="ui-monospace, monospace">
                {i === 0 ? 'LOAD' : '...'}
              </text>
            </g>
          ))}
          {/* Pointer x_ptr arrow */}
          <rect x="30" y="80" width="80" height="28" rx="6" fill="#1c2233" stroke="#38bdf8" strokeWidth="1.5" />
          <text x="70" y="98" textAnchor="middle" fill="#7dd3fc" fontSize="10" fontFamily="ui-monospace, monospace">x_ptr →</text>
          <line x1="110" y1="94" x2="52" y2="71" stroke="#38bdf8" strokeWidth="1.5" strokeDasharray="4 2" markerEnd="url(#arrowSky)" />

          {/* === Array Y === */}
          <text x={startX} y="140" fill="#f59e0b" fontSize="11" fontWeight="bold" fontFamily="ui-monospace, monospace">READ+WRITE: y[ ] — Base 0x4000</text>
          {[...Array(numCells)].map((_, i) => (
            <g key={`y${i}`} transform={`translate(${startX + i * (cellW + gutter)}, 148)`}>
              <rect width={cellW} height="40" rx="6" fill={i === 0 ? '#2a1a04' : '#0d1520'} stroke={i === 0 ? '#f59e0b' : '#362a10'} strokeWidth={i === 0 ? 2 : 1} />
              <text x={cellW / 2} y="17" textAnchor="middle" fill="#94a3b8" fontSize="9" fontFamily="ui-monospace, monospace">y[{i}]</text>
              <text x={cellW / 2} y="30" textAnchor="middle" fill={i === 0 ? '#fcd34d' : '#475569'} fontSize="10" fontWeight={i === 0 ? 'bold' : 'normal'} fontFamily="ui-monospace, monospace">
                {i === 0 ? 'LD+ST' : '...'}
              </text>
            </g>
          ))}
          {/* Pointer y_ptr arrow */}
          <rect x="30" y="198" width="80" height="28" rx="6" fill="#1c2233" stroke="#f59e0b" strokeWidth="1.5" />
          <text x="70" y="216" textAnchor="middle" fill="#fcd34d" fontSize="10" fontFamily="ui-monospace, monospace">y_ptr →</text>
          <line x1="110" y1="212" x2="52" y2="189" stroke="#f59e0b" strokeWidth="1.5" strokeDasharray="4 2" markerEnd="url(#arrowAmber)" />

          {/* FMA operation box */}
          <rect x="490" y="85" width="240" height="58" rx="10" fill="#0d1a28" stroke="#1d4ed8" strokeWidth="1.5" />
          <text x="610" y="107" textAnchor="middle" fill="#93c5fd" fontSize="11" fontWeight="bold" fontFamily="ui-monospace, monospace">y[i] = α·x[i] + y[i]</text>
          <text x="610" y="124" textAnchor="middle" fill="#60a5fa" fontSize="10" fontFamily="ui-monospace, monospace">2 FLOPs / 12 bytes = 0.17 FLOPs/B</text>
          <text x="610" y="137" textAnchor="middle" fill="#3b82f6" fontSize="9" fontFamily="ui-monospace, monospace">→ Memory Bandwidth Bound</text>

          <defs>
            <marker id="arrowSky" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto">
              <path d="M0,0 L0,6 L7,3 z" fill="#38bdf8" />
            </marker>
            <marker id="arrowAmber" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto">
              <path d="M0,0 L0,6 L7,3 z" fill="#f59e0b" />
            </marker>
          </defs>
        </svg>

        <div className="mt-3 grid grid-cols-3 gap-3 text-xs">
          <div className="p-2.5 rounded-lg bg-sky-950/30 border border-sky-500/20 text-sky-300 font-mono text-center">
            <div className="font-bold mb-0.5">Load x[i]</div>
            <div className="text-[10px] text-sky-400 opacity-80">4 bytes / element</div>
          </div>
          <div className="p-2.5 rounded-lg bg-amber-950/30 border border-amber-500/20 text-amber-300 font-mono text-center">
            <div className="font-bold mb-0.5">Load + Store y[i]</div>
            <div className="text-[10px] text-amber-400 opacity-80">8 bytes / element</div>
          </div>
          <div className="p-2.5 rounded-lg bg-rose-950/30 border border-rose-500/20 text-rose-300 font-mono text-center">
            <div className="font-bold mb-0.5">Total Bus Traffic</div>
            <div className="text-[10px] text-rose-400 opacity-80">12 bytes for 2 FLOPs</div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Tree Reduction Diagram — Chapter 1.3: Hierarchical halving tree in memory
// ============================================================================
export function TreeReductionDiagram({ data }: { data: any }) {
  const [activeTab, setActiveTab] = useState<'pass1' | 'pass2' | 'pass3' | 'odd'>('pass1');
  const title = data?.title || 'CUDA In-Place Tree Reduction';
  const subtitle = data?.subtitle || 'Halving stride s = N/2 down to 1. Upper half folds into lower half with zero extra memory.';

  // Cell positions for N = 8
  const cellW = 72;
  const cellH = 48;
  const gutter = 16;
  const startX = 36;
  const startY = 110;

  // Render 8 cells for power-of-2 passes
  const renderCells = (step: 'pass1' | 'pass2' | 'pass3') => {
    const stride = step === 'pass1' ? 4 : step === 'pass2' ? 2 : 1;
    const destCount = stride;

    return (
      <g>
        {/* Memory Cells 0..7 */}
        {[...Array(8)].map((_, i) => {
          const x = startX + i * (cellW + gutter);
          const isDest = i < destCount;
          const isSrc = i >= stride && i < stride + destCount;
          const isInactive = !isDest && !isSrc;

          const fillColor = isDest ? 'var(--diag-dest-fill, #07271a)' : isSrc ? 'var(--diag-src-fill, #24143a)' : 'var(--diag-inactive-fill, #0c1017)';
          const strokeColor = isDest ? 'var(--diag-dest-stroke, #10b981)' : isSrc ? 'var(--diag-src-stroke, #a855f7)' : 'var(--diag-inactive-stroke, #273142)';
          const textColor = isDest ? 'var(--diag-dest-text, #34d399)' : isSrc ? 'var(--diag-src-text, #c084fc)' : 'var(--diag-inactive-text, #4b5563)';

          // Values inside cells depending on pass
          let valLabel = `x[${i}]`;
          if (step === 'pass1') {
            if (i === 0) valLabel = 'x0+x4';
            else if (i === 1) valLabel = 'x1+x5';
            else if (i === 2) valLabel = 'x2+x6';
            else if (i === 3) valLabel = 'x3+x7';
          } else if (step === 'pass2') {
            if (i === 0) valLabel = 'x0..x6';
            else if (i === 1) valLabel = 'x1..x7';
          } else if (step === 'pass3') {
            if (i === 0) valLabel = 'SUM TOTAL';
          }

          return (
            <g key={`cell-${i}`}>
              <rect
                x={x}
                y={startY}
                width={cellW}
                height={cellH}
                rx="8"
                fill={fillColor}
                stroke={strokeColor}
                strokeWidth={isDest || isSrc ? '2' : '1'}
              />
              <text
                x={x + cellW / 2}
                y={startY + 18}
                textAnchor="middle"
                fill={isInactive ? '#4b5563' : '#94a3b8'}
                fontSize="10"
                fontFamily="ui-monospace, monospace"
                fontWeight="bold"
              >
                data[{i}]
              </text>
              <text
                x={x + cellW / 2}
                y={startY + 36}
                textAnchor="middle"
                fill={textColor}
                fontSize={step === 'pass3' && i === 0 ? '10' : '11'}
                fontFamily="ui-monospace, monospace"
                fontWeight="bold"
              >
                {valLabel}
              </text>
            </g>
          );
        })}

        {/* Curved folding arrows from source to destination */}
        {[...Array(destCount)].map((_, i) => {
          const srcIdx = i + stride;
          const xDest = startX + i * (cellW + gutter) + cellW / 2;
          const xSrc = startX + srcIdx * (cellW + gutter) + cellW / 2;
          const arcTop = startY - 28 - (stride === 4 ? 24 : stride === 2 ? 14 : 6);

          return (
            <g key={`arrow-${i}`}>
              <path
                d={`M ${xSrc} ${startY} Q ${(xSrc + xDest) / 2} ${arcTop} ${xDest} ${startY - 4}`}
                fill="none"
                stroke="#a855f7"
                strokeWidth="2"
                strokeDasharray="4 3"
                markerEnd="url(#arrowPurpleHead)"
              />
              <rect
                x={(xSrc + xDest) / 2 - 24}
                y={arcTop - 12}
                width="48"
                height="16"
                rx="4"
                fill="#160e29"
                stroke="#a855f7"
                strokeWidth="1"
              />
              <text
                x={(xSrc + xDest) / 2}
                y={arcTop}
                textAnchor="middle"
                fill="#d8b4fe"
                fontSize="9"
                fontFamily="ui-monospace, monospace"
                fontWeight="bold"
              >
                + s={stride}
              </text>
            </g>
          );
        })}
      </g>
    );
  };

  // Render N = 5 odd length fold
  const renderOddFold = () => {
    return (
      <g>
        {/* Memory Cells 0..4 */}
        {[...Array(5)].map((_, i) => {
          const x = startX + i * (cellW + gutter + 20);
          const isDest = i < 2;
          const isSrc = i >= 2 && i < 4;
          const isOddTail = i === 4;

          const fillColor = isDest ? 'var(--diag-dest-fill, #07271a)' : isSrc ? 'var(--diag-src-fill, #24143a)' : 'var(--diag-tail-fill, #2b1b04)';
          const strokeColor = isDest ? 'var(--diag-dest-stroke, #10b981)' : isSrc ? 'var(--diag-src-stroke, #a855f7)' : 'var(--diag-tail-stroke, #f59e0b)';
          const textColor = isDest ? 'var(--diag-dest-text, #34d399)' : isSrc ? 'var(--diag-src-text, #c084fc)' : 'var(--diag-tail-text, #fbbf24)';

          let valLabel = i === 0 ? 'x0+x2+x4' : i === 1 ? 'x1+x3' : `x[${i}]`;

          return (
            <g key={`odd-cell-${i}`}>
              <rect
                x={x}
                y={startY}
                width={cellW + 10}
                height={cellH}
                rx="8"
                fill={fillColor}
                stroke={strokeColor}
                strokeWidth="2"
              />
              <text
                x={x + (cellW + 10) / 2}
                y={startY + 18}
                textAnchor="middle"
                fill="#94a3b8"
                fontSize="10"
                fontFamily="ui-monospace, monospace"
                fontWeight="bold"
              >
                data[{i}] {isOddTail ? '(ODD TAIL)' : ''}
              </text>
              <text
                x={x + (cellW + 10) / 2}
                y={startY + 36}
                textAnchor="middle"
                fill={textColor}
                fontSize="10"
                fontFamily="ui-monospace, monospace"
                fontWeight="bold"
              >
                {valLabel}
              </text>
            </g>
          );
        })}

        {/* Regular pairs: 2->0 and 3->1 */}
        {[0, 1].map((i) => {
          const srcIdx = i + 2;
          const xDest = startX + i * (cellW + gutter + 20) + (cellW + 10) / 2;
          const xSrc = startX + srcIdx * (cellW + gutter + 20) + (cellW + 10) / 2;
          const arcTop = startY - 34;

          return (
            <path
              key={`odd-pair-${i}`}
              d={`M ${xSrc} ${startY} Q ${(xSrc + xDest) / 2} ${arcTop} ${xDest} ${startY - 4}`}
              fill="none"
              stroke="#a855f7"
              strokeWidth="2"
              strokeDasharray="4 3"
              markerEnd="url(#arrowPurpleHead)"
            />
          );
        })}

        {/* Big Amber Arc for Odd Tail [4] -> [0] */}
        {(() => {
          const xDest = startX + (cellW + 10) / 2;
          const xTail = startX + 4 * (cellW + gutter + 20) + (cellW + 10) / 2;
          const arcTop = startY - 60;

          return (
            <g>
              <path
                d={`M ${xTail} ${startY} Q ${(xTail + xDest) / 2} ${arcTop} ${xDest} ${startY - 4}`}
                fill="none"
                stroke="#f59e0b"
                strokeWidth="2.5"
                markerEnd="url(#arrowAmberHead)"
              />
              <rect
                x={(xTail + xDest) / 2 - 60}
                y={arcTop - 14}
                width="120"
                height="18"
                rx="6"
                fill="#241403"
                stroke="#f59e0b"
                strokeWidth="1"
              />
              <text
                x={(xTail + xDest) / 2}
                y={arcTop}
                textAnchor="middle"
                fill="#fcd34d"
                fontSize="9"
                fontFamily="ui-monospace, monospace"
                fontWeight="bold"
              >
                Odd tail folded into [0]
              </text>
            </g>
          );
        })()}
      </g>
    );
  };

  return (
    <div className="my-8 rounded-2xl border border-[#30363d] bg-[#0c1017] shadow-2xl overflow-hidden select-none">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between px-5 py-3.5 bg-[#141923] border-b border-[#21262d] gap-2">
        <div className="flex items-center gap-2.5">
          <span className="p-1.5 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <Layers className="w-4 h-4" />
          </span>
          <div>
            <h4 className="text-xs font-bold text-slate-100 tracking-wide uppercase font-mono">{title}</h4>
            <p className="text-[11px] text-slate-400 font-sans mt-0.5">{subtitle}</p>
          </div>
        </div>

        {/* Interactive Step Switcher */}
        <div className="flex items-center gap-1 bg-[#090d14] p-1 rounded-lg border border-[#21262d]">
          <button
            onClick={() => setActiveTab('pass1')}
            className={`px-2.5 py-1 rounded text-[11px] font-mono transition-all cursor-pointer ${
              activeTab === 'pass1'
                ? 'bg-purple-500/20 text-purple-300 font-bold border border-purple-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Pass 1 (s=4)
          </button>
          <button
            onClick={() => setActiveTab('pass2')}
            className={`px-2.5 py-1 rounded text-[11px] font-mono transition-all cursor-pointer ${
              activeTab === 'pass2'
                ? 'bg-purple-500/20 text-purple-300 font-bold border border-purple-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Pass 2 (s=2)
          </button>
          <button
            onClick={() => setActiveTab('pass3')}
            className={`px-2.5 py-1 rounded text-[11px] font-mono transition-all cursor-pointer ${
              activeTab === 'pass3'
                ? 'bg-emerald-500/20 text-emerald-300 font-bold border border-emerald-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Pass 3 (s=1)
          </button>
          <button
            onClick={() => setActiveTab('odd')}
            className={`px-2.5 py-1 rounded text-[11px] font-mono transition-all cursor-pointer ${
              activeTab === 'odd'
                ? 'bg-amber-500/20 text-amber-300 font-bold border border-amber-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Odd Length (N=5)
          </button>
        </div>
      </div>

      <div className="p-6 bg-[#090d14] overflow-x-auto">
        <svg viewBox="0 0 760 210" className="w-full min-w-[680px]" aria-label="Tree reduction diagram">
          <defs>
            <marker id="arrowPurpleHead" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
              <path d="M0,0 L0,6 L8,3 z" fill="#a855f7" />
            </marker>
            <marker id="arrowAmberHead" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
              <path d="M0,0 L0,6 L8,3 z" fill="#f59e0b" />
            </marker>
          </defs>

          {/* Stride Status Headline */}
          <text x="36" y="28" fill="var(--diag-headline, #cbd5e1)" fontSize="11" fontWeight="bold" fontFamily="ui-monospace, monospace" letterSpacing="1">
            {activeTab === 'pass1' && 'PASS 1 (stride = 4): 4 Threads add data[i + 4] into data[i]'}
            {activeTab === 'pass2' && 'PASS 2 (stride = 2): 2 Threads add data[i + 2] into data[i]'}
            {activeTab === 'pass3' && 'PASS 3 (stride = 1): Thread 0 adds data[1] into data[0] -> FINAL SUM'}
            {activeTab === 'odd' && 'ODD LENGTH (N = 5): Half = 2 pairs 0<->2, 1<->3; Unpaired tail [4] folds into [0]'}
          </text>

          {/* Active Content */}
          {activeTab === 'odd' ? renderOddFold() : renderCells(activeTab)}

          {/* Legend Footer */}
          <g transform="translate(36, 185)">
            <rect x="0" y="0" width="12" height="12" rx="3" fill="var(--diag-dest-fill, #07271a)" stroke="var(--diag-dest-stroke, #10b981)" />
            <text x="18" y="10" fill="var(--diag-dest-legend, #34d399)" fontSize="10" fontFamily="ui-monospace, monospace" fontWeight="bold">Destination (Accumulator)</text>

            <rect x="180" y="0" width="12" height="12" rx="3" fill="var(--diag-src-fill, #24143a)" stroke="var(--diag-src-stroke, #a855f7)" />
            <text x="198" y="10" fill="var(--diag-src-legend, #c084fc)" fontSize="10" fontFamily="ui-monospace, monospace" fontWeight="bold">Source (Folded In)</text>

            <rect x="330" y="0" width="12" height="12" rx="3" fill="var(--diag-inactive-fill, #0c1017)" stroke="var(--diag-inactive-stroke, #273142)" />
            <text x="348" y="10" fill="var(--diag-inactive-legend, #64748b)" fontSize="10" fontFamily="ui-monospace, monospace">Inactive / Sleeping</text>

            <rect x="490" y="0" width="12" height="12" rx="3" fill="var(--diag-tail-fill, #2b1b04)" stroke="var(--diag-tail-stroke, #f59e0b)" />
            <text x="508" y="10" fill="var(--diag-tail-legend, #fbbf24)" fontSize="10" fontFamily="ui-monospace, monospace" fontWeight="bold">Odd Tail Element</text>
          </g>
        </svg>
      </div>
    </div>
  );
}

// ============================================================================
// Two-Pointer Partition Diagram — Chapter 1.3: Top-P Threshold Partitioning
// ============================================================================
export function TwoPointerPartitionDiagram({ data }: { data: any }) {
  const [step, setStep] = useState<0 | 1 | 2>(0);
  const title = data?.title || 'In-Place Two-Pointer Threshold Partitioning';
  const subtitle = data?.subtitle || 'Zero memory allocation: left & right pointers scan from ends and swap out-of-order logits.';

  // Sample data states
  const states = [
    {
      label: 'Step 0: Initial Scan (Left stops at -1.5, Right stops at 5.0)',
      arr: [3.2, -1.5, 4.1, 0.8, 2.5, -0.2, 5.0, 1.1],
      left: 1,
      right: 6,
      swapping: false,
    },
    {
      label: 'Step 1: First Swap (data[1] <-> data[6])',
      arr: [3.2, 5.0, 4.1, 0.8, 2.5, -0.2, -1.5, 1.1],
      left: 3,
      right: 4,
      swapping: true,
      swapLeft: 1,
      swapRight: 6,
    },
    {
      label: 'Step 2: Second Swap (data[3] <-> data[4]) -> Partition Complete!',
      arr: [3.2, 5.0, 4.1, 2.5, 0.8, -0.2, -1.5, 1.1],
      left: 4,
      right: 3,
      swapping: false,
      completed: true,
    },
  ];

  const current = states[step];
  const cellW = 74;
  const cellH = 46;
  const gutter = 14;
  const startX = 36;
  const startY = 85;

  return (
    <div className="my-8 rounded-2xl border border-[#30363d] bg-[#0c1017] shadow-2xl overflow-hidden select-none">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between px-5 py-3.5 bg-[#141923] border-b border-[#21262d] gap-2">
        <div className="flex items-center gap-2.5">
          <span className="p-1.5 rounded-lg bg-sky-500/10 text-sky-400 border border-sky-500/20">
            <Layers className="w-4 h-4" />
          </span>
          <div>
            <h4 className="text-xs font-bold text-slate-100 tracking-wide uppercase font-mono">{title}</h4>
            <p className="text-[11px] text-slate-400 font-sans mt-0.5">{subtitle}</p>
          </div>
        </div>

        {/* Step Buttons */}
        <div className="flex items-center gap-1 bg-[#090d14] p-1 rounded-lg border border-[#21262d]">
          <button
            onClick={() => setStep(0)}
            className={`px-2.5 py-1 rounded text-[11px] font-mono transition-all cursor-pointer ${
              step === 0 ? 'bg-sky-500/20 text-sky-300 font-bold border border-sky-500/40' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Step 0 (Scan)
          </button>
          <button
            onClick={() => setStep(1)}
            className={`px-2.5 py-1 rounded text-[11px] font-mono transition-all cursor-pointer ${
              step === 1 ? 'bg-indigo-500/20 text-indigo-300 font-bold border border-indigo-500/40' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Step 1 (Swap)
          </button>
          <button
            onClick={() => setStep(2)}
            className={`px-2.5 py-1 rounded text-[11px] font-mono transition-all cursor-pointer ${
              step === 2 ? 'bg-emerald-500/20 text-emerald-300 font-bold border border-emerald-500/40' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Step 2 (Final)
          </button>
        </div>
      </div>

      <div className="p-6 bg-[#090d14] overflow-x-auto">
        <svg viewBox="0 0 760 210" className="w-full min-w-[680px]" aria-label="Two pointer partition diagram">
          <text x="36" y="28" fill="#94a3b8" fontSize="11" fontWeight="bold" fontFamily="ui-monospace, monospace" letterSpacing="1">
            {current.label} | Pivot = 2.0
          </text>

          {/* Partition Dividing Line when completed */}
          {step === 2 && (
            <g>
              <line
                x1={startX + 4 * (cellW + gutter) - gutter / 2}
                y1="60"
                x2={startX + 4 * (cellW + gutter) - gutter / 2}
                y2="175"
                stroke="#10b981"
                strokeWidth="2.5"
                strokeDasharray="4 2"
              />
              <rect
                x={startX + 4 * (cellW + gutter) - 55}
                y="55"
                width="110"
                height="18"
                rx="5"
                fill="#07271a"
                stroke="#10b981"
                strokeWidth="1"
              />
              <text
                x={startX + 4 * (cellW + gutter)}
                y="68"
                textAnchor="middle"
                fill="#34d399"
                fontSize="9"
                fontFamily="ui-monospace, monospace"
                fontWeight="bold"
              >
                DIVIDER: num_above=4
              </text>
            </g>
          )}

          {/* Array Elements */}
          {current.arr.map((val, i) => {
            const x = startX + i * (cellW + gutter);
            const isAbove = val >= 2.0;
            const isLeft = i === current.left;
            const isRight = i === current.right;

            const fillColor = isAbove ? '#0c231a' : '#1e141a';
            const strokeColor = isAbove ? '#10b981' : '#f43f5e';
            const textColor = isAbove ? '#6ee7b7' : '#fda4af';

            return (
              <g key={`part-cell-${i}`}>
                <rect
                  x={x}
                  y={startY}
                  width={cellW}
                  height={cellH}
                  rx="8"
                  fill={fillColor}
                  stroke={strokeColor}
                  strokeWidth="1.5"
                />
                <text
                  x={x + cellW / 2}
                  y={startY + 18}
                  textAnchor="middle"
                  fill="#94a3b8"
                  fontSize="10"
                  fontFamily="ui-monospace, monospace"
                  fontWeight="bold"
                >
                  [{i}]
                </text>
                <text
                  x={x + cellW / 2}
                  y={startY + 35}
                  textAnchor="middle"
                  fill={textColor}
                  fontSize="12"
                  fontFamily="ui-monospace, monospace"
                  fontWeight="bold"
                >
                  {val.toFixed(1)}
                </text>

                {/* Left pointer indicator */}
                {isLeft && step < 2 && (
                  <g>
                    <path d={`M ${x + cellW / 2} ${startY + cellH + 6} L ${x + cellW / 2 - 5} ${startY + cellH + 16} L ${x + cellW / 2 + 5} ${startY + cellH + 16} Z`} fill="#38bdf8" />
                    <text x={x + cellW / 2} y={startY + cellH + 28} textAnchor="middle" fill="#38bdf8" fontSize="10" fontFamily="ui-monospace, monospace" fontWeight="bold">
                      left
                    </text>
                  </g>
                )}

                {/* Right pointer indicator */}
                {isRight && step < 2 && (
                  <g>
                    <path d={`M ${x + cellW / 2} ${startY + cellH + 6} L ${x + cellW / 2 - 5} ${startY + cellH + 16} L ${x + cellW / 2 + 5} ${startY + cellH + 16} Z`} fill="#f59e0b" />
                    <text x={x + cellW / 2} y={startY + cellH + 28} textAnchor="middle" fill="#f59e0b" fontSize="10" fontFamily="ui-monospace, monospace" fontWeight="bold">
                      right
                    </text>
                  </g>
                )}
              </g>
            );
          })}

          {/* Sub-Array labels at step 2 */}
          {step === 2 && (
            <g transform="translate(36, 185)">
              <text x="140" y="0" textAnchor="middle" fill="#34d399" fontSize="11" fontFamily="ui-monospace, monospace" fontWeight="bold">
                Front Region: All logits &gt;= 2.0 (Indices 0..3)
              </text>
              <text x="500" y="0" textAnchor="middle" fill="#fb7185" fontSize="11" fontFamily="ui-monospace, monospace" fontWeight="bold">
                Back Region: All logits &lt; 2.0 (Indices 4..7)
              </text>
            </g>
          )}
        </svg>
      </div>
    </div>
  );
}

// ============================================================================
// Ring AllReduce Diagram — Chapter 1.3: Distributed Ring-AllReduce Shift
// ============================================================================
export function RingAllReduceDiagram({ data }: { data: any }) {
  const [step, setStep] = useState<0 | 1 | 2>(0);
  const title = data?.title || 'Distributed Ring-AllReduce (NCCL / Megatron-LM)';
  const subtitle = data?.subtitle || '4 GPUs arranged in a ring exchanging chunks across (P - 1) Reduce-Scatter steps.';

  // Node coordinates for 4 GPUs in a diamond/ring
  const gpus = [
    { id: 0, x: 380, y: 55, name: 'GPU 0' },
    { id: 1, x: 570, y: 140, name: 'GPU 1' },
    { id: 2, x: 380, y: 225, name: 'GPU 2' },
    { id: 3, x: 190, y: 140, name: 'GPU 3' },
  ];

  return (
    <div className="my-8 rounded-2xl border border-[#30363d] bg-[#0c1017] shadow-2xl overflow-hidden select-none">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between px-5 py-3.5 bg-[#141923] border-b border-[#21262d] gap-2">
        <div className="flex items-center gap-2.5">
          <span className="p-1.5 rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
            <Layers className="w-4 h-4" />
          </span>
          <div>
            <h4 className="text-xs font-bold text-slate-100 tracking-wide uppercase font-mono">{title}</h4>
            <p className="text-[11px] text-slate-400 font-sans mt-0.5">{subtitle}</p>
          </div>
        </div>

        {/* Step Buttons */}
        <div className="flex items-center gap-1 bg-[#090d14] p-1 rounded-lg border border-[#21262d]">
          <button
            onClick={() => setStep(0)}
            className={`px-2.5 py-1 rounded text-[11px] font-mono transition-all cursor-pointer ${
              step === 0 ? 'bg-indigo-500/20 text-indigo-300 font-bold border border-indigo-500/40' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Step 0: Send c=r
          </button>
          <button
            onClick={() => setStep(1)}
            className={`px-2.5 py-1 rounded text-[11px] font-mono transition-all cursor-pointer ${
              step === 1 ? 'bg-indigo-500/20 text-indigo-300 font-bold border border-indigo-500/40' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Step 1: Send c=r-1
          </button>
          <button
            onClick={() => setStep(2)}
            className={`px-2.5 py-1 rounded text-[11px] font-mono transition-all cursor-pointer ${
              step === 2 ? 'bg-emerald-500/20 text-emerald-300 font-bold border border-emerald-500/40' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Step 2: Fully Reduced
          </button>
        </div>
      </div>

      <div className="p-6 bg-[#090d14] overflow-x-auto">
        <svg viewBox="0 0 760 300" className="w-full min-w-[680px]" aria-label="Ring AllReduce diagram">
          <defs>
            <marker id="ringArrowHead" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
              <path d="M0,0 L0,6 L8,3 z" fill="#6366f1" />
            </marker>
          </defs>

          {/* Ring Transfer Arcs */}
          <path d="M 440 60 Q 550 70 560 120" fill="none" stroke="#6366f1" strokeWidth="2.5" markerEnd="url(#ringArrowHead)" />
          <path d="M 570 180 Q 540 230 440 235" fill="none" stroke="#6366f1" strokeWidth="2.5" markerEnd="url(#ringArrowHead)" />
          <path d="M 320 235 Q 220 230 200 180" fill="none" stroke="#6366f1" strokeWidth="2.5" markerEnd="url(#ringArrowHead)" />
          <path d="M 200 120 Q 230 70 320 60" fill="none" stroke="#6366f1" strokeWidth="2.5" markerEnd="url(#ringArrowHead)" />

          {/* Center Info Badge */}
          <rect x="290" y="125" width="180" height="42" rx="8" fill="#141926" stroke="#312e81" strokeWidth="1" />
          <text x="380" y="142" textAnchor="middle" fill="#a5b4fc" fontSize="10" fontFamily="ui-monospace, monospace" fontWeight="bold">
            Clockwise Ring Network
          </text>
          <text x="380" y="157" textAnchor="middle" fill="#818cf8" fontSize="9" fontFamily="ui-monospace, monospace">
            Send Chunk -&gt; Rank (r + 1) % 4
          </text>

          {/* 4 GPU Nodes */}
          {gpus.map((gpu) => (
            <g key={`gpu-${gpu.id}`}>
              <rect
                x={gpu.x - 70}
                y={gpu.y - 28}
                width="140"
                height="56"
                rx="10"
                fill="#0d131f"
                stroke={step === 2 ? '#10b981' : '#4f46e5'}
                strokeWidth="2"
              />
              <text
                x={gpu.x}
                y={gpu.y - 10}
                textAnchor="middle"
                fill="#f8fafc"
                fontSize="11"
                fontFamily="ui-monospace, monospace"
                fontWeight="bold"
              >
                {gpu.name}
              </text>

              {/* 4 Chunks inside GPU */}
              {[0, 1, 2, 3].map((c) => {
                const cx = gpu.x - 52 + c * 26;
                const cy = gpu.y + 4;
                const isFullyReduced = step === 2 && c === (gpu.id + 1) % 4;

                return (
                  <rect
                    key={`gpu-${gpu.id}-c-${c}`}
                    x={cx}
                    y={cy}
                    width="22"
                    height="16"
                    rx="3"
                    fill={isFullyReduced ? '#064e3b' : '#1e1b4b'}
                    stroke={isFullyReduced ? '#10b981' : '#6366f1'}
                    strokeWidth="1"
                  />
                );
              })}
            </g>
          ))}
        </svg>
      </div>
    </div>
  );
}

// ============================================================================
// Two-Pass Grid-Strided Vector Reduction Diagram — Chapter 1.3
// ============================================================================
export function TwoPassGridReductionDiagram({ data }: { data: any }) {
  const title = data?.title || 'Two-Pass Grid-Strided Vector Reduction (The Mark Harris Pattern)';
  const subtitle = data?.subtitle || 'Saturating 900+ GB/s Memory Bandwidth: Pass 1 reduces blocks into partial sums, Pass 2 yields the final scalar.';

  return (
    <div className="my-8 rounded-2xl border border-[#30363d] bg-[#0c1017] shadow-2xl overflow-hidden select-none">
      <div className="flex items-center justify-between px-5 py-3.5 bg-[#141923] border-b border-[#21262d]">
        <div className="flex items-center gap-2.5">
          <span className="p-1.5 rounded-lg bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
            <Layers className="w-4 h-4" />
          </span>
          <div>
            <h4 className="text-xs font-bold text-slate-100 tracking-wide uppercase font-mono">{title}</h4>
            <p className="text-[11px] text-slate-400 font-sans mt-0.5">{subtitle}</p>
          </div>
        </div>
        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-semibold">
          TWO-PASS PIPELINE
        </span>
      </div>

      <div className="p-6 bg-[#090d14] overflow-x-auto">
        <svg viewBox="0 0 760 250" className="w-full min-w-[680px]" aria-label="Two-pass grid reduction diagram">
          <defs>
            <marker id="arrowCyanHead" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
              <path d="M0,0 L0,6 L8,3 z" fill="#06b6d4" />
            </marker>
            <marker id="arrowEmeraldHead" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
              <path d="M0,0 L0,6 L8,3 z" fill="#10b981" />
            </marker>
          </defs>

          {/* Level 1: Global Input Array */}
          <text x="36" y="20" fill="#94a3b8" fontSize="10" fontWeight="bold" fontFamily="ui-monospace, monospace" letterSpacing="1">
            LEVEL 1: GLOBAL INPUT TENSOR IN HIGH-BANDWIDTH MEMORY (N = 262,144 FLOATS = 1 MB)
          </text>

          {[
            { label: 'Block 0 [256 floats]', x: 36, w: 125 },
            { label: 'Block 1 [256 floats]', x: 171, w: 125 },
            { label: 'Block 2 [256 floats]', x: 306, w: 125 },
            { label: '...', x: 441, w: 60, isEllipsis: true },
            { label: 'Block 1023 [256 floats]', x: 511, w: 145 },
          ].map((b, i) => (
            <g key={`l1-blk-${i}`}>
              <rect
                x={b.x}
                y="30"
                width={b.w}
                height="36"
                rx="6"
                fill={b.isEllipsis ? '#0c1017' : '#0e1c2a'}
                stroke={b.isEllipsis ? '#263043' : '#0ea5e9'}
                strokeWidth="1.5"
              />
              <text
                x={b.x + b.w / 2}
                y="52"
                textAnchor="middle"
                fill={b.isEllipsis ? '#64748b' : '#7dd3fc'}
                fontSize="10"
                fontFamily="ui-monospace, monospace"
                fontWeight="bold"
              >
                {b.label}
              </text>
            </g>
          ))}

          {/* Pass 1 Reduction Arrows */}
          {[98, 233, 368, 583].map((cx, i) => (
            <path
              key={`arrow-p1-${i}`}
              d={`M ${cx} 66 L ${cx} 96`}
              stroke="#06b6d4"
              strokeWidth="1.8"
              strokeDasharray="3 2"
              markerEnd="url(#arrowCyanHead)"
            />
          ))}

          {/* Pass 1 Phase Annotation Badge */}
          <rect x="664" y="66" width="86" height="20" rx="4" fill="#08202d" stroke="#06b6d4" strokeWidth="1" />
          <text x="707" y="80" textAnchor="middle" fill="#67e8f9" fontSize="8" fontFamily="ui-monospace, monospace" fontWeight="bold">
            PASS 1: BLK TREE
          </text>

          {/* Level 2: Partial Sums Buffer */}
          <text x="36" y="112" fill="#94a3b8" fontSize="10" fontWeight="bold" fontFamily="ui-monospace, monospace" letterSpacing="1">
            LEVEL 2: PARTIAL SUMS BUFFER (1,024 PARTIAL SUMS = 4 KB)
          </text>

          {[
            { label: 'partial[0]', x: 36, w: 125 },
            { label: 'partial[1]', x: 171, w: 125 },
            { label: 'partial[2]', x: 306, w: 125 },
            { label: '...', x: 441, w: 60, isEllipsis: true },
            { label: 'partial[1023]', x: 511, w: 145 },
          ].map((b, i) => (
            <g key={`l2-part-${i}`}>
              <rect
                x={b.x}
                y="122"
                width={b.w}
                height="34"
                rx="6"
                fill={b.isEllipsis ? '#0c1017' : '#072418'}
                stroke={b.isEllipsis ? '#263043' : '#10b981'}
                strokeWidth="1.5"
              />
              <text
                x={b.x + b.w / 2}
                y="143"
                textAnchor="middle"
                fill={b.isEllipsis ? '#64748b' : '#6ee7b7'}
                fontSize="10"
                fontFamily="ui-monospace, monospace"
                fontWeight="bold"
              >
                {b.label}
              </text>
            </g>
          ))}

          {/* Pass 2 Converging Arrows */}
          <path d="M 98 156 Q 250 190 310 192" fill="none" stroke="#10b981" strokeWidth="1.8" markerEnd="url(#arrowEmeraldHead)" />
          <path d="M 233 156 Q 280 185 310 192" fill="none" stroke="#10b981" strokeWidth="1.8" markerEnd="url(#arrowEmeraldHead)" />
          <path d="M 368 156 Q 370 180 370 188" fill="none" stroke="#10b981" strokeWidth="1.8" markerEnd="url(#arrowEmeraldHead)" />
          <path d="M 583 156 Q 490 190 430 192" fill="none" stroke="#10b981" strokeWidth="1.8" markerEnd="url(#arrowEmeraldHead)" />

          {/* Level 3: Final Output Scalar */}
          <rect x="290" y="194" width="200" height="40" rx="8" fill="#042014" stroke="#10b981" strokeWidth="2.5" />
          <text x="390" y="212" textAnchor="middle" fill="#86efac" fontSize="10" fontFamily="ui-monospace, monospace" fontWeight="bold">
            FINAL GLOBAL SUM
          </text>
          <text x="390" y="226" textAnchor="middle" fill="#4ade80" fontSize="10" fontFamily="ui-monospace, monospace">
            total = sum(partial[0..1023])
          </text>

          {/* Hardware Badges */}
          <g transform="translate(530, 202)">
            <rect x="0" y="0" width="170" height="24" rx="4" fill="#0d1829" stroke="#38bdf8" strokeWidth="1" />
            <text x="85" y="15" textAnchor="middle" fill="#38bdf8" fontSize="9" fontFamily="ui-monospace, monospace" fontWeight="bold">
              Peak Coalesced Bandwidth
            </text>
          </g>
        </svg>
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

  // Two-pass reduction / grid-stride reduction (Chapter 1.3)
  if (cleanType.includes('two-pass') || cleanType.includes('grid-stride') || cleanType.includes('two-pass-reduction')) {
    return <TwoPassGridReductionDiagram data={parsed} />;
  }

  // Tree reduction (Chapter 1.3)
  if (cleanType.includes('tree-reduction') || cleanType.includes('reduction')) {
    return <TreeReductionDiagram data={parsed} />;
  }

  // Two-pointer threshold partitioning (Chapter 1.3)
  if (cleanType.includes('two-pointer') || cleanType.includes('partition')) {
    return <TwoPointerPartitionDiagram data={parsed} />;
  }

  // Ring AllReduce (Chapter 1.3)
  if (cleanType.includes('ring') || cleanType.includes('allreduce')) {
    return <RingAllReduceDiagram data={parsed} />;
  }

  if (cleanType.includes('pitched') || cleanType.includes('pitch') || cleanType.includes('2d-pitch')) {
    return <PitchedMemoryDiagram title={parsed.title} subtitle={parsed.subtitle} />;
  }

  if (cleanType.includes('double-buffering') || cleanType.includes('double-buf') || cleanType.includes('ping-pong') || cleanType.includes('timeline')) {
    return <DoubleBufferingDiagram title={parsed.title} subtitle={parsed.subtitle} />;
  }

  // Pointer swap / register swap (Chapter 1.3) — distinct from stack-frame pointer diagram
  if (cleanType === 'pointer-swap' || cleanType.includes('swap') || cleanType.includes('ping-pong-swap')) {
    return <PointerSwapDiagram data={parsed} />;
  }

  // Dual-stream / SAXPY / strided access (Chapter 1.4)
  if (cleanType.includes('strided') || cleanType.includes('saxpy') || cleanType.includes('dual-stream') || cleanType.includes('multistream')) {
    return <StridedAccessDiagram data={parsed} />;
  }

  // Stride / 2D-to-flat layout (Chapter 1.2) — shows 2D grid folding to flat strip
  if (cleanType.includes('stride-layout') || cleanType.includes('stride-matrix') || cleanType.includes('stride-layout')) {
    return <StrideLayoutDiagram data={parsed} />;
  }

  // Stack-frame pointer addressing — Chapter 1.1 only
  if (cleanType === 'pointer-addressing' || cleanType.includes('ladder') || cleanType.includes('byte-ladder')) {
    return <PointerAddressingDiagram data={parsed} />;
  }

  if (cleanType.includes('cache') || cleanType.includes('align') || cleanType.includes('straddle')) {
    return <CacheLineDiagram title={parsed.title} subtitle={parsed.subtitle} />;
  }

  if (cleanType.includes('gemm') || cleanType.includes('tile') || cleanType.includes('matrix')) {
    return <TiledGemmDiagram title={parsed.title} subtitle={parsed.subtitle} />;
  }

  // Generic pointer/memory fallback
  if (cleanType.includes('pointer') || cleanType.includes('offset') || cleanType.includes('memory')) {
    return <PointerAddressingDiagram data={parsed} />;
  }

  return <PointerAddressingDiagram data={parsed} />;
}

