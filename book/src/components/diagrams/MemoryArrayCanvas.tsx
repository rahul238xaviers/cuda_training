'use client';

import React from 'react';
import { Database } from 'lucide-react';
import { StridedAccessPayload } from './types';

interface MemoryArrayCanvasProps {
  payload: StridedAccessPayload;
}

export function MemoryArrayCanvas({ payload }: MemoryArrayCanvasProps) {
  const numCells = payload.numCells || 6;
  const cellW = payload.cellW || 64;
  const gutter = 8;
  const startX = 30;

  const streamX = payload.streamX || {
    name: 'x[ ]',
    baseAddr: '0x1000',
    mode: 'READ',
    ptrLabel: 'x_ptr',
    cellLabel: 'LOAD',
  };

  const streamY = payload.streamY || {
    name: 'y[ ]',
    baseAddr: '0x4000',
    mode: 'READ+WRITE',
    ptrLabel: 'y_ptr',
    cellLabel: 'LD+ST',
  };

  const fma = payload.fmaBox || {
    equation: 'y[i] = α · x[i] + y[i]',
    metrics: '2 FLOPs / 12 bytes = 0.17 FLOPs/B',
    regime: 'Memory Bandwidth Bound',
  };

  const defaultMetrics = [
    { title: 'Load x[i]', value: '4 bytes / element', subtext: 'Read Stream', color: 'sky' as const },
    { title: 'Load + Store y[i]', value: '8 bytes / element', subtext: 'Read + Write Stream', color: 'amber' as const },
    { title: 'Total Bus Traffic', value: '12 bytes for 2 FLOPs', subtext: 'Bus Traffic Ratio', color: 'rose' as const },
  ];

  const metrics = payload.metrics && payload.metrics.length > 0 ? payload.metrics : defaultMetrics;

  return (
    <div className="my-8 rounded-2xl border border-[#1e293b] bg-[#090d14] shadow-2xl overflow-hidden select-none hardware-diagram-viewport">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-3.5 bg-[#0e1422] border-b border-[#1e293b] gap-3">
        <div className="flex items-center gap-2.5 min-w-0">
          <span className="p-1.5 rounded-lg bg-sky-500/10 text-sky-400 border border-sky-500/20 shrink-0">
            <Database className="w-4 h-4" />
          </span>
          <div className="min-w-0">
            <h4 className="text-xs font-bold text-slate-100 tracking-wide uppercase font-mono truncate">{payload.title}</h4>
            <p className="text-[11px] text-slate-400 font-sans mt-0.5 truncate">{payload.subtitle}</p>
          </div>
        </div>
        <span className="text-[10px] font-mono px-2.5 py-0.5 rounded bg-sky-500/15 text-sky-300 border border-sky-500/30 font-semibold uppercase shrink-0 whitespace-nowrap">
          {fma.regime}
        </span>
      </div>

      <div className="p-6 bg-[#090d14]">
        <svg viewBox="0 0 760 238" className="w-full" aria-label="Dual stream memory diagram">
          <defs>
            <marker id="memArrowSky" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
              <path d="M0,0.5 L0,5.5 L6,3 z" fill="#38bdf8" />
            </marker>
            <marker id="memArrowAmber" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
              <path d="M0,0.5 L0,5.5 L6,3 z" fill="#f59e0b" />
            </marker>
          </defs>

          {/* === Array X === */}
          <text x={startX} y="24" fill="#38bdf8" fontSize="11" fontWeight="bold" fontFamily="ui-monospace, monospace">
            {streamX.mode}: {streamX.name} — Base {streamX.baseAddr}
          </text>
          {[...Array(numCells)].map((_, i) => (
            <g key={`x${i}`} transform={`translate(${startX + i * (cellW + gutter)}, 30)`}>
              <rect width={cellW} height="40" rx="6" fill={i === 0 ? '#0c2a4a' : '#0d1520'} stroke={i === 0 ? '#38bdf8' : '#1e3046'} strokeWidth={i === 0 ? 2 : 1} />
              <text x={cellW / 2} y="17" textAnchor="middle" fill="#94a3b8" fontSize="9" fontFamily="ui-monospace, monospace">x[{i}]</text>
              <text x={cellW / 2} y="30" textAnchor="middle" fill={i === 0 ? '#7dd3fc' : '#475569'} fontSize="10" fontWeight={i === 0 ? 'bold' : 'normal'} fontFamily="ui-monospace, monospace">
                {i === 0 ? streamX.cellLabel : '...'}
              </text>
            </g>
          ))}

          {/* Pointer x_ptr arrow — perfectly aligned vertical pointer into x[0] */}
          <rect x="30" y="82" width={cellW} height="26" rx="6" fill="#0c2a4a" stroke="#38bdf8" strokeWidth="1.5" />
          <text x={startX + cellW / 2} y="99" textAnchor="middle" fill="#38bdf8" fontSize="11" fontWeight="bold" fontFamily="ui-monospace, monospace">
            {streamX.ptrLabel}
          </text>
          <line x1={startX + cellW / 2} y1="82" x2={startX + cellW / 2} y2="72" stroke="#38bdf8" strokeWidth="2" markerEnd="url(#memArrowSky)" />

          {/* === Array Y === */}
          <text x={startX} y="138" fill="#f59e0b" fontSize="11" fontWeight="bold" fontFamily="ui-monospace, monospace">
            {streamY.mode}: {streamY.name} — Base {streamY.baseAddr}
          </text>
          {[...Array(numCells)].map((_, i) => (
            <g key={`y${i}`} transform={`translate(${startX + i * (cellW + gutter)}, 146)`}>
              <rect width={cellW} height="40" rx="6" fill={i === 0 ? '#2a1a04' : '#0d1520'} stroke={i === 0 ? '#f59e0b' : '#362a10'} strokeWidth={i === 0 ? 2 : 1} />
              <text x={cellW / 2} y="17" textAnchor="middle" fill="#94a3b8" fontSize="9" fontFamily="ui-monospace, monospace">y[{i}]</text>
              <text x={cellW / 2} y="30" textAnchor="middle" fill={i === 0 ? '#fcd34d' : '#475569'} fontSize="10" fontWeight={i === 0 ? 'bold' : 'normal'} fontFamily="ui-monospace, monospace">
                {i === 0 ? streamY.cellLabel : '...'}
              </text>
            </g>
          ))}

          {/* Pointer y_ptr arrow — perfectly aligned vertical pointer into y[0] */}
          <rect x="30" y="198" width={cellW} height="26" rx="6" fill="#2a1a04" stroke="#f59e0b" strokeWidth="1.5" />
          <text x={startX + cellW / 2} y="215" textAnchor="middle" fill="#fcd34d" fontSize="11" fontWeight="bold" fontFamily="ui-monospace, monospace">
            {streamY.ptrLabel}
          </text>
          <line x1={startX + cellW / 2} y1="198" x2={startX + cellW / 2} y2="188" stroke="#f59e0b" strokeWidth="2" markerEnd="url(#memArrowAmber)" />

          {/* FMA ALU Operation Box */}
          <rect x="490" y="82" width="240" height="64" rx="10" fill="#0d1a28" stroke="#1d4ed8" strokeWidth="1.5" />
          <text x="610" y="104" textAnchor="middle" fill="#93c5fd" fontSize="11" fontWeight="bold" fontFamily="ui-monospace, monospace">
            {fma.equation}
          </text>
          <text x="610" y="122" textAnchor="middle" fill="#60a5fa" fontSize="10" fontFamily="ui-monospace, monospace">
            {fma.metrics}
          </text>
          <text x="610" y="137" textAnchor="middle" fill="#3b82f6" fontSize="9" fontFamily="ui-monospace, monospace">
            → {fma.regime}
          </text>
        </svg>

        {/* Dynamic Metric Cards */}
        <div className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
          {metrics.map((m, idx) => {
            const cardBg = m.color === 'amber'
              ? 'bg-amber-500/10 border-amber-500/30 text-amber-300'
              : m.color === 'rose'
              ? 'bg-rose-500/10 border-rose-500/30 text-rose-300'
              : m.color === 'emerald'
              ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
              : 'bg-sky-500/10 border-sky-500/30 text-sky-300';
            const titleColor = m.color === 'amber' ? 'text-amber-300' : m.color === 'rose' ? 'text-rose-300' : m.color === 'emerald' ? 'text-emerald-300' : 'text-sky-300';
            const valColor = m.color === 'amber' ? 'text-amber-400' : m.color === 'rose' ? 'text-rose-400' : m.color === 'emerald' ? 'text-emerald-400' : 'text-sky-400';

            return (
              <div key={idx} className={`p-3 rounded-xl border font-mono text-center ${cardBg}`}>
                <div className={`font-bold text-[11px] mb-0.5 ${titleColor}`}>{m.title}</div>
                <div className={`text-sm font-bold mb-0.5 ${valColor}`}>{m.value}</div>
                {m.subtext && <div className="text-[10px] text-slate-400">{m.subtext}</div>}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
