'use client';

import React, { useState } from 'react';
import { Activity, Cpu, Gauge } from 'lucide-react';
import { RooflineDiagramPayload, RooflineArch, RooflineWorkload } from './types';

interface RooflineCanvasProps {
  payload: RooflineDiagramPayload;
}

const DEFAULT_ARCHS: Record<string, RooflineArch> = {
  rtx5090: {
    id: 'rtx5090',
    name: 'NVIDIA RTX 5090',
    chip: 'Blackwell GB202',
    tech: '512-bit GDDR7 @ 28 Gbps',
    peakComputeTflops: 105.0,
    bandwidthGBs: 1792,
    ridgePoint: 58.6,
    peakComputeStr: '105.0 TFLOPS FP32',
    bandwidthStr: '1,792 GB/s',
    color: '#38bdf8',
  },
  ryzen9950x: {
    id: 'ryzen9950x',
    name: 'AMD Ryzen 9 9950X',
    chip: 'Zen 5 (16 Cores / 32 Threads)',
    tech: 'Dual-Channel DDR5-6000 CL30',
    peakComputeTflops: 3.5,
    bandwidthGBs: 96,
    ridgePoint: 36.5,
    peakComputeStr: '3.5 TFLOPS (3,500 GFLOPS)',
    bandwidthStr: '96 GB/s',
    color: '#f59e0b',
  },
};

const DEFAULT_WORKLOADS: RooflineWorkload[] = [
  {
    id: 'saxpy',
    name: 'SAXPY Vector Traversal',
    tag: 'BLAS Level 1 · Element-wise',
    expr: 'y[i] = α · x[i] + y[i]',
    intensity: 0.167,
    busTraffic: '12 Bytes per 2 FLOPs',
    badgeColor: 'sky',
    isMemoryBound: true,
    diagnosis: 'Severe Memory Wall Choke',
    explanation: 'Streams 4B x[i] and 8B y[i] over the memory bus for every 2 FLOPs. Over 99% of compute cores sit completely idle waiting for DRAM.',
  },
  {
    id: 'fused',
    name: 'Fused GeLU + Residual',
    tag: 'Deep Learning Activation Block',
    expr: 'y[i] = GeLU(x[i] + res[i])',
    intensity: 0.80,
    busTraffic: '10 Bytes per 8 FLOPs',
    badgeColor: 'amber',
    isMemoryBound: true,
    diagnosis: 'Bandwidth Bound (4.8x Intensity Boost)',
    explanation: 'Operator fusion keeps intermediate results in registers, eliminating redundant round-trips to DRAM and speeding up execution by ~4x.',
  },
  {
    id: 'gemm',
    name: 'Tiled Matrix Multiply (GEMM)',
    tag: 'BLAS Level 3 · Dense Linear Algebra',
    expr: 'C = α · A · B + β · C',
    intensity: 120.0,
    busTraffic: 'Shared Memory Tile Reuse O(N)',
    badgeColor: 'emerald',
    isMemoryBound: false,
    diagnosis: 'Compute Bound (Full Saturation)',
    explanation: 'High tile reuse inside SRAM/registers enables every loaded byte to be reused dozens of times, reaching 100% of theoretical compute peak.',
  },
];

export function RooflineCanvas({ payload }: RooflineCanvasProps) {
  const archs = payload.archs || DEFAULT_ARCHS;
  const workloads = payload.workloads || DEFAULT_WORKLOADS;
  const archKeys = Object.keys(archs);

  const [activeArchKey, setActiveArchKey] = useState<string>(payload.defaultArch || archKeys[0] || 'rtx5090');
  const [selectedWorkloadId, setSelectedWorkloadId] = useState<string>(workloads[0]?.id || 'saxpy');

  const currentArch: RooflineArch = ((archs as any)[activeArchKey] || (archs as any)[archKeys[0]] || DEFAULT_ARCHS.rtx5090) as RooflineArch;
  const activeWl = workloads.find((w) => w.id === selectedWorkloadId) || workloads[0] || DEFAULT_WORKLOADS[0];

  // Attainable throughput calculation
  const calcThroughputTflops = (intensity: number) => {
    const memThroughput = (intensity * currentArch.bandwidthGBs) / 1000;
    return Math.min(currentArch.peakComputeTflops, memThroughput);
  };

  const activeThroughput = calcThroughputTflops(activeWl.intensity);
  const activeThroughputDisplay =
    currentArch.peakComputeTflops < 10
      ? `${(activeThroughput * 1000).toFixed(1)} GFLOPS`
      : activeThroughput < 1.0
      ? `${(activeThroughput * 1000).toFixed(0)} GFLOPS`
      : `${activeThroughput.toFixed(1)} TFLOPS`;

  const activeSaturationPct = ((activeThroughput / currentArch.peakComputeTflops) * 100).toFixed(2);
  const activeStalledPct = (100 - parseFloat(activeSaturationPct)).toFixed(2);

  // SVG Coordinates (Log scale X: 0.0625 to 256 -> 12 octaves)
  const plotLeft = 85;
  const plotRight = 710;
  const plotWidth = plotRight - plotLeft;
  const minLog = -4; // 2^-4 = 0.0625
  const logSpan = 12;

  const getX = (intensity: number) => {
    const logVal = Math.log2(intensity);
    const ratio = Math.max(0, Math.min(1, (logVal - minLog) / logSpan));
    return plotLeft + ratio * plotWidth;
  };

  const ceilingY = 52;
  const axisY = 240;
  const slopeLeftY = axisY - 15;
  const ridgeX = getX(currentArch.ridgePoint);

  const getY = (intensity: number) => {
    if (intensity >= currentArch.ridgePoint) return ceilingY;
    const x = getX(intensity);
    const slope = (slopeLeftY - ceilingY) / (ridgeX - plotLeft);
    return ceilingY + (ridgeX - x) * slope;
  };

  const xTicks = [
    { val: 0.1, label: '0.1' },
    { val: 0.5, label: '0.5' },
    { val: 1.0, label: '1.0' },
    { val: 4.0, label: '4.0' },
    { val: 16.0, label: '16' },
    { val: 64.0, label: '64' },
    { val: 256.0, label: '256' },
  ];

  return (
    <div className="my-8 rounded-2xl border border-[var(--border-subtle)] bg-[var(--bg-canvas)] shadow-2xl overflow-hidden select-none">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-3.5 bg-[var(--bg-card)] border-b border-[var(--border-subtle)]">
        <div className="flex items-center gap-2.5">
          <span className="p-1.5 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <Activity className="w-4 h-4" />
          </span>
          <div>
            <h4 className="text-xs font-bold text-slate-100 tracking-wide uppercase font-mono">{payload.title}</h4>
            <p className="text-[11px] text-slate-400 font-sans mt-0.5">{payload.subtitle}</p>
          </div>
        </div>

        {/* Hardware Selector Tabs */}
        <div className="flex items-center gap-1.5 p-1 rounded-xl bg-[#090d14] border border-[#21262d]">
          {archKeys.map((key) => {
            const arch = archs[key];
            const isSelected = activeArchKey === key;
            return (
              <button
                key={key}
                onClick={() => setActiveArchKey(key)}
                className={`px-3 py-1 rounded-lg text-xs font-mono transition-all flex items-center gap-1.5 ${
                  isSelected
                    ? 'bg-sky-500/20 text-sky-300 border border-sky-500/40 font-semibold shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Cpu className="w-3.5 h-3.5" />
                <span>{arch.name}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Workload Filter Bar */}
      <div className="px-5 py-2.5 bg-[#0e131d] border-b border-[#1b2230] flex flex-wrap items-center justify-between gap-2 text-xs">
        <div className="flex items-center gap-2">
          <span className="text-[11px] text-slate-400 font-mono uppercase font-semibold">Inspect Workload:</span>
          <div className="flex items-center gap-1.5">
            {workloads.map((w) => {
              const isSelected = selectedWorkloadId === w.id;
              const color = w.badgeColor || 'sky';
              return (
                <button
                  key={w.id}
                  onClick={() => setSelectedWorkloadId(w.id)}
                  className={`px-2.5 py-1 rounded-lg font-mono text-[11px] transition-all flex items-center gap-1.5 ${
                    isSelected
                      ? color === 'sky'
                        ? 'bg-sky-500/20 text-sky-300 border border-sky-500/50 font-bold'
                        : color === 'amber'
                        ? 'bg-amber-500/20 text-amber-300 border border-amber-500/50 font-bold'
                        : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/50 font-bold'
                      : 'bg-[#151c28] text-slate-400 border border-[#212b3d] hover:text-slate-200'
                  }`}
                >
                  <span
                    className={`w-1.5 h-1.5 rounded-full ${
                      color === 'sky' ? 'bg-sky-400' : color === 'amber' ? 'bg-amber-400' : 'bg-emerald-400'
                    }`}
                  />
                  <span>{w.name.split(' ')[0]}</span>
                  <span className="text-[10px] opacity-75">({w.intensity})</span>
                </button>
              );
            })}
          </div>
        </div>

        <div className="flex items-center gap-2 font-mono text-[11px] text-slate-300">
          <span className="text-slate-400">Ridge Point:</span>
          <span className="px-2 py-0.5 rounded bg-purple-500/15 text-purple-300 border border-purple-500/30 font-bold">
            {currentArch.ridgePoint} FLOPs / Byte
          </span>
        </div>
      </div>

      {/* SVG Canvas */}
      <div className="p-6 bg-[#090d14]">
        <svg viewBox="0 0 760 300" className="w-full" aria-label="Hardware Roofline Model Diagram">
          <defs>
            <linearGradient id="roofMemGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.14" />
              <stop offset="100%" stopColor="#38bdf8" stopOpacity="0.01" />
            </linearGradient>
            <linearGradient id="roofComputeGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#10b981" stopOpacity="0.14" />
              <stop offset="100%" stopColor="#10b981" stopOpacity="0.01" />
            </linearGradient>
            <filter id="roofGlow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Shaded Regions */}
          <polygon
            points={`${plotLeft},${axisY} ${plotLeft},${slopeLeftY} ${ridgeX},${ceilingY} ${ridgeX},${axisY}`}
            fill="url(#roofMemGrad)"
          />
          <polygon
            points={`${ridgeX},${axisY} ${ridgeX},${ceilingY} ${plotRight},${ceilingY} ${plotRight},${axisY}`}
            fill="url(#roofComputeGrad)"
          />

          <text
            x={(plotLeft + ridgeX) / 2}
            y="120"
            textAnchor="middle"
            fill="#38bdf8"
            fontSize="10"
            fontWeight="bold"
            fontFamily="ui-monospace, monospace"
            opacity="0.6"
            letterSpacing="1"
          >
            MEMORY BANDWIDTH BOUND REGIME
          </text>
          <text
            x={(plotLeft + ridgeX) / 2}
            y="134"
            textAnchor="middle"
            fill="#94a3b8"
            fontSize="8.5"
            fontFamily="ui-monospace, monospace"
            opacity="0.6"
          >
            P = Arithmetic_Intensity × Memory_Bandwidth
          </text>

          <text
            x={(ridgeX + plotRight) / 2}
            y="120"
            textAnchor="middle"
            fill="#10b981"
            fontSize="10"
            fontWeight="bold"
            fontFamily="ui-monospace, monospace"
            opacity="0.6"
            letterSpacing="1"
          >
            COMPUTE BOUND REGIME
          </text>
          <text
            x={(ridgeX + plotRight) / 2}
            y="134"
            textAnchor="middle"
            fill="#94a3b8"
            fontSize="8.5"
            fontFamily="ui-monospace, monospace"
            opacity="0.6"
          >
            P = Peak_Compute (ALUs 100% Saturated)
          </text>

          {/* Grid Ticks */}
          {xTicks.map((t) => {
            const tx = getX(t.val);
            return (
              <g key={`xtick-${t.val}`}>
                <line x1={tx} y1={ceilingY} x2={tx} y2={axisY} stroke="#1b2230" strokeWidth="1" strokeDasharray="3 3" />
                <line x1={tx} y1={axisY} x2={tx} y2={axisY + 4} stroke="#475569" strokeWidth="1" />
                <text x={tx} y={axisY + 16} textAnchor="middle" fill="#64748b" fontSize="9" fontFamily="ui-monospace, monospace">
                  {t.label}
                </text>
              </g>
            );
          })}

          <line x1={plotLeft} y1={ceilingY - 15} x2={plotLeft} y2={axisY} stroke="#334155" strokeWidth="1.5" />
          <line x1={plotLeft} y1={axisY} x2={plotRight + 15} y2={axisY} stroke="#334155" strokeWidth="1.5" />

          <text
            x={plotLeft - 10}
            y={ceilingY - 20}
            textAnchor="start"
            fill="#94a3b8"
            fontSize="9"
            fontWeight="bold"
            fontFamily="ui-monospace, monospace"
          >
            Attainable Throughput (GFLOPS / TFLOPS)
          </text>

          <text
            x={(plotLeft + plotRight) / 2}
            y={axisY + 36}
            textAnchor="middle"
            fill="#94a3b8"
            fontSize="9.5"
            fontWeight="bold"
            fontFamily="ui-monospace, monospace"
          >
            Arithmetic Intensity (FLOPs / Byte — Log Scale)
          </text>

          {/* Dashed Peak Compute Line */}
          <line
            x1={plotLeft}
            y1={ceilingY}
            x2={ridgeX}
            y2={ceilingY}
            stroke="#10b981"
            strokeWidth="1"
            strokeDasharray="3 3"
            opacity="0.4"
          />

          {/* Roofline Slope & Flat Ceiling */}
          <line x1={plotLeft} y1={slopeLeftY} x2={ridgeX} y2={ceilingY} stroke="#38bdf8" strokeWidth="3" strokeLinecap="round" />
          <line x1={ridgeX} y1={ceilingY} x2={plotRight} y2={ceilingY} stroke="#10b981" strokeWidth="3" strokeLinecap="round" />

          <text
            x={plotLeft - 8}
            y={ceilingY + 3.5}
            textAnchor="end"
            fill="#34d399"
            fontSize="9.5"
            fontWeight="bold"
            fontFamily="ui-monospace, monospace"
          >
            {currentArch.peakComputeStr}
          </text>
          <line x1={plotLeft - 4} y1={ceilingY} x2={plotLeft} y2={ceilingY} stroke="#10b981" strokeWidth="1.5" />

          <g transform={`translate(${plotLeft + 16}, 72)`}>
            <rect width="136" height="22" rx="5" fill="#082f49" stroke="#38bdf8" strokeWidth="1" />
            <text x="68" y="15" textAnchor="middle" fill="#bae6fd" fontSize="9.5" fontWeight="bold" fontFamily="ui-monospace, monospace">
              Slope: {currentArch.bandwidthStr}
            </text>
          </g>

          {/* Ridge Guide & Marker */}
          <line x1={ridgeX} y1={ceilingY} x2={ridgeX} y2={axisY} stroke="#c084fc" strokeWidth="1.5" strokeDasharray="4 3" />
          <circle cx={ridgeX} cy={ceilingY} r="7" fill="#7c3aed" opacity="0.3" />
          <circle cx={ridgeX} cy={ceilingY} r="4.5" fill="#a855f7" stroke="#ffffff" strokeWidth="1.5" />

          <g transform={`translate(${ridgeX - 52}, ${axisY + 4})`}>
            <rect width="104" height="18" rx="4" fill="#3b0764" stroke="#a855f7" strokeWidth="1" />
            <text x="52" y="13" textAnchor="middle" fill="#e9d5ff" fontSize="8.5" fontWeight="bold" fontFamily="ui-monospace, monospace">
              Ridge: {currentArch.ridgePoint} FLOPs/B
            </text>
          </g>

          {/* Render Workload Points */}
          {workloads.map((w) => {
            const wx = getX(w.intensity);
            const wy = getY(w.intensity);
            const isSel = selectedWorkloadId === w.id;
            const color = w.badgeColor === 'emerald' ? '#10b981' : w.badgeColor === 'amber' ? '#f59e0b' : '#38bdf8';
            const fillColor = w.badgeColor === 'emerald' ? '#059669' : w.badgeColor === 'amber' ? '#d97706' : '#0284c7';
            const badgeBg = isSel
              ? w.badgeColor === 'emerald' ? '#064e3b' : w.badgeColor === 'amber' ? '#2a1a04' : '#0c2a4a'
              : '#090d14';

            return (
              <g
                key={`point-${w.id}`}
                className="cursor-pointer transition-transform"
                onClick={() => setSelectedWorkloadId(w.id)}
              >
                <line x1={wx} y1={wy} x2={wx} y2={axisY} stroke={color} strokeWidth="1" strokeDasharray="2 2" opacity="0.6" />
                {isSel && <circle cx={wx} cy={wy} r="10" fill={color} opacity="0.3" />}
                <circle cx={wx} cy={wy} r="5.5" fill={fillColor} stroke={color} strokeWidth="2" filter={isSel ? 'url(#roofGlow)' : undefined} />
                <circle cx={wx} cy={wy} r="2" fill="#ffffff" />
                <g transform={`translate(${wx - 38}, ${wy - 26})`}>
                  <rect
                    width="76"
                    height="20"
                    rx="5"
                    fill={badgeBg}
                    stroke={isSel ? color : '#1e3046'}
                    strokeWidth={isSel ? '1.5' : '1'}
                  />
                  <text
                    x="38"
                    y="13.5"
                    textAnchor="middle"
                    fill={isSel ? (w.badgeColor === 'emerald' ? '#6ee7b7' : w.badgeColor === 'amber' ? '#fcd34d' : '#7dd3fc') : '#94a3b8'}
                    fontSize="9"
                    fontWeight="bold"
                    fontFamily="ui-monospace, monospace"
                  >
                    {w.name.split(' ')[0]}
                  </text>
                </g>
              </g>
            );
          })}
        </svg>

        {/* Dynamic Architectural Inspector Panel */}
        <div className="mt-4 p-4 rounded-xl bg-[#101622] border border-[#212c3f]">
          <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-[#1c2637]">
            <div className="flex items-center gap-2.5">
              <span
                className={`p-1.5 rounded-lg border ${
                  activeWl.badgeColor === 'emerald'
                    ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                    : activeWl.badgeColor === 'amber'
                    ? 'bg-amber-500/10 text-amber-400 border-amber-500/30'
                    : 'bg-sky-500/10 text-sky-400 border-sky-500/30'
                }`}
              >
                <Gauge className="w-4 h-4" />
              </span>
              <div>
                <div className="flex items-center gap-2">
                  <h5 className="text-xs font-bold text-slate-100 font-mono">{activeWl.name}</h5>
                  {activeWl.diagnosis && (
                    <span
                      className={`text-[10px] font-mono px-2 py-0.5 rounded font-semibold ${
                        activeWl.isMemoryBound
                          ? 'bg-rose-500/15 text-rose-300 border border-rose-500/30'
                          : 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/30'
                      }`}
                    >
                      {activeWl.diagnosis}
                    </span>
                  )}
                </div>
                {activeWl.expr && (
                  <p className="text-[11px] text-slate-400 font-mono mt-0.5">
                    Formula: <span className="text-slate-200">{activeWl.expr}</span> {activeWl.busTraffic && `· ${activeWl.busTraffic}`}
                  </p>
                )}
              </div>
            </div>

            <div className="flex items-center gap-4 text-xs font-mono">
              <div className="text-right">
                <div className="text-[10px] text-slate-400 uppercase">Attainable Speed</div>
                <div className="text-sm font-bold text-sky-300">{activeThroughputDisplay}</div>
              </div>
              <div className="text-right border-l border-[#243044] pl-4">
                <div className="text-[10px] text-slate-400 uppercase">Core Saturation</div>
                <div
                  className={`text-sm font-bold ${
                    parseFloat(activeSaturationPct) >= 90
                      ? 'text-emerald-400'
                      : parseFloat(activeSaturationPct) >= 10
                      ? 'text-amber-400'
                      : 'text-rose-400'
                  }`}
                >
                  {activeSaturationPct}%
                </div>
              </div>
            </div>
          </div>

          <div className="mt-3">
            <div className="flex items-center justify-between text-[10px] font-mono mb-1 text-slate-400">
              <span>ALU Utilization: {activeSaturationPct}% Active</span>
              <span className="text-rose-400">{activeStalledPct}% Stalled on Memory Bus</span>
            </div>
            <div className="w-full h-2 rounded-full bg-[#1c2433] overflow-hidden flex">
              <div
                className={`h-full transition-all duration-300 ${
                  activeWl.badgeColor === 'emerald' ? 'bg-emerald-400' : activeWl.badgeColor === 'amber' ? 'bg-amber-400' : 'bg-sky-400'
                }`}
                style={{ width: `${Math.max(1, parseFloat(activeSaturationPct))}%` }}
              />
              <div
                className="h-full bg-rose-500/40"
                style={{ width: `${parseFloat(activeStalledPct)}%` }}
              />
            </div>
          </div>

          {activeWl.explanation && (
            <p className="text-xs text-slate-300 mt-2.5 font-sans leading-relaxed">
              <strong className="text-slate-100 font-semibold">Architectural Takeaway:</strong> {activeWl.explanation}
            </p>
          )}
        </div>

        {/* 3 Metric Cards */}
        <div className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
          <div className="p-3 rounded-xl bg-sky-500/10 border border-sky-500/25 text-sky-300 font-mono text-center">
            <div className="font-bold text-[11px] text-sky-200 mb-0.5">{activeWl.name.split(' ')[0]} Intensity</div>
            <div className="text-sm font-bold text-sky-400 mb-0.5">{activeWl.intensity} FLOPs / Byte</div>
            <div className="text-[10px] text-slate-400">{activeWl.busTraffic || 'Memory bus traffic ratio'}</div>
          </div>

          <div className="p-3 rounded-xl bg-purple-500/10 border border-purple-500/25 text-purple-300 font-mono text-center">
            <div className="font-bold text-[11px] text-purple-200 mb-0.5">{currentArch.name} Ridge Point</div>
            <div className="text-sm font-bold text-purple-400 mb-0.5">{currentArch.ridgePoint} FLOPs / Byte</div>
            <div className="text-[10px] text-slate-400">Ratio = Peak Compute / Memory Bandwidth</div>
          </div>

          <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/25 text-rose-300 font-mono text-center">
            <div className="font-bold text-[11px] text-rose-200 mb-0.5">Hardware Efficiency</div>
            <div className="text-sm font-bold text-rose-400 mb-0.5">{activeSaturationPct}% Core Utilization</div>
            <div className="text-[10px] text-slate-400">{activeStalledPct}% ALUs waiting for memory</div>
          </div>
        </div>
      </div>
    </div>
  );
}
