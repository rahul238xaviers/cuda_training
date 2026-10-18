'use client';

import React, { useState } from 'react';
import { Image as ImageIcon, ZoomIn, X } from 'lucide-react';

interface AdaptiveAssetProps {
  src?: string;
  alt?: string;
  title?: string;
  chapterId?: string;
  volumeId?: string;
}

export function AdaptiveAsset({ src = '', alt = '', title, chapterId = '1.1', volumeId }: AdaptiveAssetProps) {
  const [isZoomed, setIsZoomed] = useState(false);

  // Determine actual asset URL
  let resolvedUrl = src;
  if (src && !src.startsWith('http') && !src.startsWith('/') && !src.startsWith('data:')) {
    // Relative path like 'assets/buffer_aliasing.svg'
    const volParam = volumeId ? `&volumeId=${encodeURIComponent(volumeId)}` : '';
    resolvedUrl = `/api/asset?chapter=${encodeURIComponent(chapterId)}&file=${encodeURIComponent(src)}${volParam}`;
  }

  const caption = title || alt;

  return (
    <div role="figure" aria-label={alt || 'System Architectural Schematic'} className="my-8 rounded-2xl border border-[var(--border-subtle)] bg-[var(--bg-canvas)] overflow-hidden shadow-xl select-none">
      {/* Header bar */}
      <div className="flex items-center justify-between px-4 py-2.5 bg-[var(--bg-card)] border-b border-[var(--border-subtle)] text-xs">
        <div className="flex items-center gap-2">
          <span className="p-1 rounded-md bg-sky-500/10 text-sky-400 border border-sky-500/20">
            <ImageIcon className="w-3.5 h-3.5" />
          </span>
          <span className="font-mono text-[11px] font-semibold text-slate-200 tracking-wide uppercase">
            {alt || 'System Architectural Schematic'}
          </span>
        </div>
        <button
          onClick={() => setIsZoomed(true)}
          className="flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono text-slate-400 hover:text-white bg-[#151b26] hover:bg-[#202736] border border-[#263143] transition-colors"
          title="Zoom View"
        >
          <ZoomIn className="w-3 h-3" />
          <span>Zoom</span>
        </button>
      </div>

      {/* Asset Render Area */}
      <div className="p-6 flex items-center justify-center bg-[#090d14] overflow-x-auto">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={resolvedUrl}
          alt={alt}
          className="max-w-full h-auto max-h-[480px] object-contain rounded-lg transition-transform hover:scale-[1.01]"
          loading="lazy"
        />
      </div>

      {/* Caption footer if alt or title provided */}
      {caption && (
        <div role="caption" className="px-5 py-2.5 bg-[var(--bg-card)] border-t border-[var(--border-subtle)] text-center text-xs text-slate-400 font-sans">
          {caption}
        </div>
      )}

      {/* Lightbox Modal */}
      {isZoomed && (
        <div
          className="fixed inset-0 z-50 bg-black/90 backdrop-blur-md flex flex-col items-center justify-center p-6"
          onClick={() => setIsZoomed(false)}
        >
          <button
            onClick={() => setIsZoomed(false)}
            className="absolute top-6 right-6 p-2 rounded-xl bg-slate-800/80 hover:bg-slate-700 text-white transition-colors border border-slate-600"
          >
            <X className="w-5 h-5" />
          </button>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={resolvedUrl}
            alt={alt}
            className="max-w-[95vw] max-h-[90vh] object-contain rounded-xl shadow-2xl"
          />
          {caption && <p className="mt-4 text-sm text-slate-300 font-mono text-center">{caption}</p>}
        </div>
      )}
    </div>
  );
}
