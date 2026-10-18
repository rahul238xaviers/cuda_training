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
    <div className="my-6 rounded-xl border border-sky-500/30 bg-[#0e1420] shadow-lg overflow-hidden select-text">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 bg-[#141b2a] border-b border-[#212b3d] text-xs select-none">
        <span className="text-[11px] font-mono text-sky-400 font-semibold tracking-wider uppercase flex items-center gap-1.5">
          <Calculator className="w-3.5 h-3.5 text-sky-400" />
          <span>Mathematical Definition & Address Scaling</span>
        </span>
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-slate-400 font-mono">Systems Arithmetic</span>
          <button
            onClick={handleCopy}
            className="px-2 py-0.5 rounded text-[10px] font-mono font-medium text-slate-300 hover:text-white bg-[#1e2738] hover:bg-[#2b374d] border border-[#2e3b52] transition-colors flex items-center gap-1 cursor-pointer"
          >
            {copied ? (
              <>
                <Check className="w-3 h-3 text-emerald-400" />
                <span className="text-emerald-300">Copied</span>
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
      <div className="p-4 text-center font-mono text-[13px] md:text-sm text-sky-100 font-medium overflow-x-auto space-y-1.5">
        {lines.map((line, idx) => (
          <div key={idx} className="py-0.5 tracking-wide">
            {line}
          </div>
        ))}
      </div>
    </div>
  );
}
