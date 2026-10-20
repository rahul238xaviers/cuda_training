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
      <blockquote className="my-6 pl-4 py-2 border-l-4 border-sky-500 bg-sky-500/10 text-[var(--text-primary)] italic rounded-r-lg">
        {children}
      </blockquote>
    );
  }

  const alertType = alertMatch[1].toLowerCase() as CalloutType;

  // Filter out the [!TYPE] string from children
  const cleanChildren = (node: any): any => {
    if (typeof node === 'string') {
      return node.replace(/^\s*\[!(NOTE|WARNING|IMPORTANT|CAUTION|TIP)\]\s*/i, '');
    }
    if (Array.isArray(node)) {
      return node.map((child, idx) => (idx === 0 ? cleanChildren(child) : child));
    }
    if (React.isValidElement(node)) {
      const childProps = node.props as any;
      if (childProps && childProps.children) {
        return React.cloneElement(node, {
          ...childProps,
          children: cleanChildren(childProps.children),
        });
      }
    }
    return node;
  };

  const filteredChildren = cleanChildren(children);

  const configs: Record<
    CalloutType,
    { title: string; cardClass: string; titleColor: string; icon: React.ReactNode }
  > = {
    note: {
      title: 'Contextual Note',
      cardClass: 'callout-card-note',
      titleColor: 'text-sky-600 dark:text-sky-300',
      icon: <Info className="w-4 h-4 text-sky-500 dark:text-sky-400 shrink-0" />,
    },
    warning: {
      title: 'Hardware Bottleneck Warning',
      cardClass: 'callout-card-warning',
      titleColor: 'text-amber-600 dark:text-amber-300',
      icon: <AlertTriangle className="w-4 h-4 text-amber-500 dark:text-amber-400 shrink-0" />,
    },
    important: {
      title: 'Essential Architectural Rule',
      cardClass: 'callout-card-important',
      titleColor: 'text-emerald-600 dark:text-emerald-300',
      icon: <ShieldCheck className="w-4 h-4 text-emerald-500 dark:text-emerald-400 shrink-0" />,
    },
    caution: {
      title: 'Critical Hardware Trap',
      cardClass: 'callout-card-caution',
      titleColor: 'text-rose-600 dark:text-rose-300',
      icon: <Flame className="w-4 h-4 text-rose-500 dark:text-rose-400 shrink-0" />,
    },
    tip: {
      title: 'Performance Optimization Tip',
      cardClass: 'callout-card-tip',
      titleColor: 'text-purple-600 dark:text-purple-300',
      icon: <Sparkles className="w-4 h-4 text-purple-500 dark:text-purple-400 shrink-0" />,
    },
    standard: {
      title: 'Note',
      cardClass: 'callout-card-note',
      titleColor: 'text-sky-600 dark:text-sky-300',
      icon: <Info className="w-4 h-4 text-sky-500 dark:text-sky-400 shrink-0" />,
    },
  };

  const config = configs[alertType] || configs.note;

  return (
    <div className={`my-6 rounded-xl p-4.5 shadow-sm ${config.cardClass}`}>
      <div className="flex items-center gap-2 mb-2 select-none">
        {config.icon}
        <span className={`text-xs font-mono font-bold tracking-wider uppercase ${config.titleColor}`}>
          {config.title}
        </span>
      </div>
      <div className="text-xs md:text-sm leading-relaxed prose-p:my-1">
        {filteredChildren}
      </div>
    </div>
  );
}
