'use client';

import React, { useState } from 'react';
import { Copy, Check, Calculator } from 'lucide-react';

interface FormulaCardProps {
  code: string;
}

export function FormulaCard({ code }: FormulaCardProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const lines = code
    .split('\n')
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  return (
    <div className="my-6 rounded-xl border border-sky-500/30 dark:border-sky-500/30 bg-[var(--bg-card)] shadow-md overflow-hidden select-text">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 bg-[var(--bg-card-hover)] border-b border-[var(--border-subtle)] text-xs select-none">
        <span className="text-[11px] font-mono text-sky-600 dark:text-sky-400 font-semibold tracking-wider uppercase flex items-center gap-1.5">
          <Calculator className="w-3.5 h-3.5 text-sky-500 dark:text-sky-400" />
          <span>Mathematical Definition & Address Scaling</span>
        </span>
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-[var(--text-secondary)] font-mono">Systems Arithmetic</span>
          <button
            onClick={handleCopy}
            className="px-2 py-0.5 rounded text-[10px] font-mono font-medium text-[var(--text-secondary)] hover:text-[var(--text-primary)] bg-[var(--bg-card)] hover:bg-[var(--border-subtle)] border border-[var(--border-subtle)] transition-colors flex items-center gap-1 cursor-pointer"
          >
            {copied ? (
              <>
                <Check className="w-3 h-3 text-emerald-500" />
                <span className="text-emerald-600 dark:text-emerald-300">Copied</span>
              </>
            ) : (
              <>
                <Copy className="w-3 h-3" />
                <span>Copy</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Body */}
      <div className="p-4 text-center font-mono text-[13px] md:text-sm text-[var(--text-primary)] font-medium overflow-x-auto space-y-1.5 bg-[var(--bg-app)]">
        {lines.map((line, idx) => (
          <div key={idx} className="py-0.5 tracking-wide">
            {line}
          </div>
        ))}
      </div>
    </div>
  );
}
