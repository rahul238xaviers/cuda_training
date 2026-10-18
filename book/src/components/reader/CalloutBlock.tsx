'use client';

import React from 'react';
import { Info, AlertTriangle, ShieldCheck, Flame, Sparkles } from 'lucide-react';

interface CalloutBlockProps {
  children?: React.ReactNode;
}

type CalloutType = 'note' | 'warning' | 'important' | 'caution' | 'tip' | 'standard';

export function CalloutBlock({ children }: CalloutBlockProps) {
  // Extract text content to check for GitHub-style [!TYPE] tokens
  const extractText = (node: any): string => {
    if (!node) return '';
    if (typeof node === 'string') return node;
    if (Array.isArray(node)) return node.map(extractText).join(' ');
    if (node.props?.children) return extractText(node.props.children);
    return '';
  };

  const rawText = extractText(children).trim();
  const alertMatch = rawText.match(/^\[!(NOTE|WARNING|IMPORTANT|CAUTION|TIP)\]/i);

  if (!alertMatch) {
    return (
      <blockquote className="my-6 pl-4 py-2 border-l-4 border-sky-500/40 bg-sky-500/5 text-slate-300 italic rounded-r-lg">
        {children}
      </blockquote>
    );
  }

  const alertType = alertMatch[1].toLowerCase() as CalloutType;

  // Filter out the [!TYPE] string from children
  const cleanChildren = (node: any): any => {
    if (typeof node === 'string') {
      return node.replace(/^\[!(NOTE|WARNING|IMPORTANT|CAUTION|TIP)\]\s*/i, '');
    }
    if (Array.isArray(node)) {
      return node.map((child, idx) => (idx === 0 ? cleanChildren(child) : child));
    }
    if (node?.props?.children) {
      return React.cloneElement(node, {
        children: cleanChildren(node.props.children),
      });
    }
    return node;
  };

  const filteredChildren = cleanChildren(children);

  const configs: Record<
    CalloutType,
    { title: string; bg: string; border: string; text: string; titleColor: string; icon: React.ReactNode }
  > = {
    note: {
      title: 'Contextual Note',
      bg: 'bg-sky-500/10',
      border: 'border-sky-500/30',
      text: 'text-sky-100',
      titleColor: 'text-sky-300',
      icon: <Info className="w-4 h-4 text-sky-400 shrink-0" />,
    },
    warning: {
      title: 'Hardware Bottleneck Warning',
      bg: 'bg-amber-500/10',
      border: 'border-amber-500/30',
      text: 'text-amber-100',
      titleColor: 'text-amber-300',
      icon: <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />,
    },
    important: {
      title: 'Essential Architectural Rule',
      bg: 'bg-emerald-500/10',
      border: 'border-emerald-500/30',
      text: 'text-emerald-100',
      titleColor: 'text-emerald-300',
      icon: <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" />,
    },
    caution: {
      title: 'Critical Hardware Trap',
      bg: 'bg-rose-500/10',
      border: 'border-rose-500/30',
      text: 'text-rose-100',
      titleColor: 'text-rose-300',
      icon: <Flame className="w-4 h-4 text-rose-400 shrink-0" />,
    },
    tip: {
      title: 'Performance Optimization Tip',
      bg: 'bg-purple-500/10',
      border: 'border-purple-500/30',
      text: 'text-purple-100',
      titleColor: 'text-purple-300',
      icon: <Sparkles className="w-4 h-4 text-purple-400 shrink-0" />,
    },
    standard: {
      title: 'Note',
      bg: 'bg-slate-800/20',
      border: 'border-slate-700/40',
      text: 'text-slate-200',
      titleColor: 'text-slate-300',
      icon: <Info className="w-4 h-4 text-slate-400 shrink-0" />,
    },
  };

  const config = configs[alertType] || configs.note;

  return (
    <div className={`my-6 rounded-xl border ${config.border} ${config.bg} p-4.5 shadow-md`}>
      <div className="flex items-center gap-2 mb-2 select-none">
        {config.icon}
        <span className={`text-xs font-mono font-bold tracking-wider uppercase ${config.titleColor}`}>
          {config.title}
        </span>
      </div>
      <div className={`text-xs md:text-sm leading-relaxed ${config.text} prose-p:my-1`}>
        {filteredChildren}
      </div>
    </div>
  );
}
