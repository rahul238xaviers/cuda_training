'use client';

import React, { useEffect, useState, useRef } from 'react';
import { ChevronDown, Check, Sparkles } from 'lucide-react';
import { AppTheme, THEME_OPTIONS, getStoredTheme, applyTheme } from '@/lib/theme';

export function ThemeToggle({ className = '', variant = 'header' }: { className?: string; variant?: 'header' | 'sidebar' }) {
  const [currentTheme, setCurrentTheme] = useState<AppTheme>('default');
  const [isOpen, setIsOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const saved = getStoredTheme();
    setCurrentTheme(saved);
    applyTheme(saved);
    setMounted(true);

    const handleThemeChange = (e: Event) => {
      const customEvent = e as CustomEvent<{ theme: AppTheme }>;
      if (customEvent.detail?.theme) {
        setCurrentTheme(customEvent.detail.theme);
      }
    };

    const handleOutsideClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };

    window.addEventListener('app-theme-changed', handleThemeChange);
    document.addEventListener('mousedown', handleOutsideClick);
    return () => {
      window.removeEventListener('app-theme-changed', handleThemeChange);
      document.removeEventListener('mousedown', handleOutsideClick);
    };
  }, []);

  const selectTheme = (theme: AppTheme) => {
    setCurrentTheme(theme);
    applyTheme(theme);
    setIsOpen(false);
  };

  const activeOption = THEME_OPTIONS.find((t) => t.id === currentTheme) || THEME_OPTIONS[0];

  if (!mounted) {
    return (
      <div className={`h-8 w-24 rounded-md bg-[#161b24] animate-pulse ${className}`} />
    );
  }

  return (
    <div className={`relative inline-block ${className}`} ref={menuRef}>
      <button
        onClick={() => setIsOpen((prev) => !prev)}
        type="button"
        title="Change Platform Theme"
        className={`theme-toggle-btn h-8 flex items-center gap-1.5 px-3 rounded-lg text-xs font-medium transition-all cursor-pointer border ${
          currentTheme === 'next-light'
            ? 'bg-white hover:bg-slate-50 text-slate-800 border-slate-300 shadow-xs'
            : currentTheme === 'cuda-night'
            ? 'bg-[#05110a] hover:bg-[#07190f] text-[#00ff88] border-[#00ff88]/40 shadow-[0_0_12px_rgba(0,255,136,0.15)]'
            : 'bg-[#151c2d] hover:bg-[#1c253b] text-slate-300 border-[#2a3854]'
        }`}
      >
        <span className="text-xs">{activeOption.icon}</span>
        <span className="font-medium text-[11px]">{activeOption.name}</span>
        <ChevronDown className="w-3 h-3 opacity-60 ml-0.5" />
      </button>

      {isOpen && (
        <div
          className={`absolute ${
            variant === 'sidebar' ? 'left-0 bottom-full mb-2' : 'right-0 top-full mt-1.5'
          } w-60 rounded-xl border shadow-2xl z-50 py-1.5 overflow-hidden animate-in fade-in zoom-in-95 duration-100 ${
            currentTheme === 'next-light'
              ? 'bg-white border-slate-200 shadow-slate-300/50'
              : currentTheme === 'cuda-night'
              ? 'bg-[#040906] border-[#00ff88]/30 shadow-black/80'
              : 'bg-[#0d121c] border-[#222c3e] shadow-black/80'
          }`}
        >
          <div className="px-3 py-1.5 text-[10px] font-bold uppercase tracking-wider text-slate-400 border-b border-inherit/40 flex items-center justify-between">
            <span>Select Workspace Theme</span>
            <Sparkles className="w-3 h-3 text-sky-400" />
          </div>

          <div className="p-1 space-y-1">
            {THEME_OPTIONS.map((opt) => {
              const isSelected = opt.id === currentTheme;
              return (
                <button
                  key={opt.id}
                  onClick={() => selectTheme(opt.id)}
                  type="button"
                  className={`w-full flex items-center justify-between px-2.5 py-2 rounded-lg text-left transition-all cursor-pointer ${
                    isSelected
                      ? opt.id === 'next-light'
                        ? 'bg-sky-50 text-sky-900 font-semibold'
                        : opt.id === 'cuda-night'
                        ? 'bg-[#00ff88]/15 text-[#00ff88] font-semibold border border-[#00ff88]/30'
                        : 'bg-sky-500/15 text-sky-300 font-semibold'
                      : currentTheme === 'next-light'
                      ? 'text-slate-700 hover:bg-slate-100'
                      : 'text-slate-300 hover:bg-[#141b29] hover:text-white'
                  }`}
                >
                  <div className="flex items-center gap-2.5">
                    <span className="text-base">{opt.icon}</span>
                    <div>
                      <div className="text-xs font-medium leading-none">{opt.name}</div>
                      <div className="text-[10px] text-slate-400 mt-1">{opt.desc}</div>
                    </div>
                  </div>
                  {isSelected && <Check className="w-3.5 h-3.5 shrink-0 text-current" />}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
