'use client';

import React, { useState } from 'react';
import { ArrowRight, Cpu, Layers, Activity, Zap, CheckCircle2, RefreshCw } from 'lucide-react';
import { PipelineFlowPayload, PipelineStage } from './types';

interface PipelineFlowCanvasProps {
  payload: PipelineFlowPayload;
}

export function PipelineFlowCanvas({ payload }: PipelineFlowCanvasProps) {
  const [activeStageId, setActiveStageId] = useState<string | null>(
    payload.stages.length > 0 ? payload.stages[0].id : null
  );

  const stages = payload.stages || [];
  const metrics = payload.metrics || [];
  const comparison = payload.comparison;

  const colorStyles: Record<string, { bg: string; border: string; text: string; badge: string; glow: string }> = {
    sky: {
      bg: 'bg-sky-50 dark:bg-sky-950/20',
      border: 'border-sky-200 dark:border-sky-500/30',
      text: 'text-sky-700 dark:text-sky-300',
      badge: 'bg-sky-100 text-sky-800 dark:bg-sky-500/20 dark:text-sky-300',
      glow: 'shadow-[0_0_15px_rgba(56,189,248,0.2)]',
    },
    amber: {
      bg: 'bg-amber-50 dark:bg-amber-950/20',
      border: 'border-amber-200 dark:border-amber-500/30',
      text: 'text-amber-700 dark:text-amber-300',
      badge: 'bg-amber-100 text-amber-800 dark:bg-amber-500/20 dark:text-amber-300',
      glow: 'shadow-[0_0_15px_rgba(245,158,11,0.2)]',
    },
    purple: {
      bg: 'bg-purple-50 dark:bg-purple-950/20',
      border: 'border-purple-200 dark:border-purple-500/30',
      text: 'text-purple-700 dark:text-purple-300',
      badge: 'bg-purple-100 text-purple-800 dark:bg-purple-500/20 dark:text-purple-300',
      glow: 'shadow-[0_0_15px_rgba(168,85,247,0.2)]',
    },
    emerald: {
      bg: 'bg-emerald-50 dark:bg-emerald-950/20',
      border: 'border-emerald-200 dark:border-emerald-500/30',
      text: 'text-emerald-700 dark:text-emerald-300',
      badge: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-500/20 dark:text-emerald-300',
      glow: 'shadow-[0_0_15px_rgba(16,185,129,0.2)]',
    },
    rose: {
      bg: 'bg-rose-50 dark:bg-rose-950/20',
      border: 'border-rose-200 dark:border-rose-500/30',
      text: 'text-rose-700 dark:text-rose-300',
      badge: 'bg-rose-100 text-rose-800 dark:bg-rose-500/20 dark:text-rose-300',
      glow: 'shadow-[0_0_15px_rgba(244,63,94,0.2)]',
    },
    cyan: {
      bg: 'bg-cyan-50 dark:bg-cyan-950/20',
      border: 'border-cyan-200 dark:border-cyan-500/30',
      text: 'text-cyan-700 dark:text-cyan-300',
      badge: 'bg-cyan-100 text-cyan-800 dark:bg-cyan-500/20 dark:text-cyan-300',
      glow: 'shadow-[0_0_15px_rgba(6,182,212,0.2)]',
    },
  };

  const activeStage = stages.find((s) => s.id === activeStageId) || stages[0];

  return (
    <div className="my-8 rounded-2xl border border-slate-200 dark:border-[#30363d] bg-white dark:bg-[#0d1117] p-6 shadow-sm">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-5 border-b border-slate-100 dark:border-[#21262d] gap-3">
        <div>
          <div className="flex items-center gap-2 text-xs font-mono font-semibold text-sky-600 dark:text-sky-400 uppercase tracking-wider">
            <Layers className="w-3.5 h-3.5" />
            <span>Hardware Execution Pipeline</span>
          </div>
          <h3 className="text-lg font-bold text-slate-900 dark:text-[#f0f6fc] mt-1">
            {payload.title}
          </h3>
          {payload.subtitle && (
            <p className="text-xs text-slate-500 dark:text-[#8b949e] mt-0.5">
              {payload.subtitle}
            </p>
          )}
        </div>
      </div>

      {/* Stage Flow Nodes */}
      {stages.length > 0 && (
        <div className="my-6 overflow-x-auto pb-4">
          <div className="flex items-center justify-between min-w-[650px] gap-2">
            {stages.map((stage, idx) => {
              const isActive = stage.id === activeStageId;
              const styling = colorStyles[stage.color] || colorStyles.sky;

              return (
                <React.Fragment key={stage.id}>
                  <button
                    onClick={() => setActiveStageId(stage.id)}
                    className={`flex-1 p-4 rounded-xl border text-left transition-all cursor-pointer relative ${
                      isActive
                        ? `${styling.bg} ${styling.border} ${styling.glow} ring-2 ring-sky-500/40`
                        : 'bg-slate-50 dark:bg-[#161b22] border-slate-200 dark:border-[#30363d] hover:border-slate-300 dark:hover:border-[#484f58]'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded font-semibold bg-slate-200/80 text-slate-700 dark:bg-[#21262d] dark:text-[#8b949e]">
                        STAGE {idx + 1}
                      </span>
                      {stage.traffic && (
                        <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded font-bold ${styling.badge}`}>
                          {stage.traffic}
                        </span>
                      )}
                    </div>

                    <div className="text-sm font-bold text-slate-900 dark:text-[#f0f6fc]">
                      {stage.name}
                    </div>

                    {stage.subtitle && (
                      <div className="text-xs text-slate-500 dark:text-[#8b949e] mt-0.5 truncate">
                        {stage.subtitle}
                      </div>
                    )}

                    {stage.latency && (
                      <div className="mt-2 text-[11px] font-mono text-slate-600 dark:text-slate-400 flex items-center gap-1">
                        <Activity className="w-3 h-3 text-slate-400" />
                        <span>Latency: {stage.latency}</span>
                      </div>
                    )}
                  </button>

                  {idx < stages.length - 1 && (
                    <div className="shrink-0 flex items-center px-1 text-slate-400 dark:text-slate-600">
                      <ArrowRight className="w-4 h-4 animate-pulse" />
                    </div>
                  )}
                </React.Fragment>
              );
            })}
          </div>
        </div>
      )}

      {/* Active Stage Detail Panel */}
      {activeStage && (
        <div className="p-4 rounded-xl border border-slate-200 dark:border-[#30363d] bg-slate-50/50 dark:bg-[#161b22]/60 mb-6">
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-xs font-mono font-bold text-slate-900 dark:text-[#f0f6fc] flex items-center gap-2">
              <Cpu className="w-3.5 h-3.5 text-sky-500" />
              <span>Microarchitectural Role: {activeStage.name}</span>
            </h4>
            {activeStage.unit && (
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-200 dark:bg-[#21262d] text-slate-700 dark:text-[#8b949e]">
                Hardware Unit: {activeStage.unit}
              </span>
            )}
          </div>

          {activeStage.details && activeStage.details.length > 0 ? (
            <ul className="space-y-1.5 mt-2">
              {activeStage.details.map((detail, dIdx) => (
                <li key={dIdx} className="text-xs text-slate-600 dark:text-[#8b949e] flex items-start gap-2">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 mt-0.5 shrink-0" />
                  <span>{detail}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-xs text-slate-500 dark:text-[#8b949e]">
              {activeStage.subtitle || 'Active pipeline execution stage.'}
            </p>
          )}
        </div>
      )}

      {/* Comparison View (e.g. Unfused vs Fused) */}
      {comparison && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <div className="p-4 rounded-xl border border-amber-200 dark:border-amber-500/20 bg-amber-50/30 dark:bg-amber-950/10">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-bold text-amber-800 dark:text-amber-300">
                {comparison.unfused.title}
              </span>
              <span className="text-xs font-mono font-extrabold text-amber-700 dark:text-amber-400">
                {comparison.unfused.traffic}
              </span>
            </div>
            {comparison.unfused.description && (
              <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
                {comparison.unfused.description}
              </p>
            )}
          </div>

          <div className="p-4 rounded-xl border border-emerald-200 dark:border-emerald-500/20 bg-emerald-50/30 dark:bg-emerald-950/10">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-bold text-emerald-800 dark:text-emerald-300">
                {comparison.fused.title}
              </span>
              <span className="text-xs font-mono font-extrabold text-emerald-700 dark:text-emerald-400">
                {comparison.fused.traffic}
              </span>
            </div>
            {comparison.fused.description && (
              <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
                {comparison.fused.description}
              </p>
            )}
          </div>
        </div>
      )}

      {/* Metrics Row */}
      {metrics.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-4 border-t border-slate-100 dark:border-[#21262d]">
          {metrics.map((m, mIdx) => {
            const styling = colorStyles[m.color] || colorStyles.sky;
            return (
              <div
                key={mIdx}
                className="p-3 rounded-xl border border-slate-100 dark:border-[#21262d] bg-slate-50/50 dark:bg-[#161b22]"
              >
                <div className="text-[10px] font-mono text-slate-500 dark:text-[#8b949e] uppercase">
                  {m.label}
                </div>
                <div className={`text-base font-extrabold font-mono mt-0.5 ${styling.text}`}>
                  {m.value}
                </div>
                {m.subtext && (
                  <div className="text-[10px] text-slate-400 dark:text-slate-500 mt-0.5">
                    {m.subtext}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
