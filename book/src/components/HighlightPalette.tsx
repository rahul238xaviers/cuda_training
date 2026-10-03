'use client';

import React, { useState, useEffect, useRef } from 'react';
import { MarkerColor, saveHighlight } from '../lib/annotations';
import { MessageSquarePlus, Check, X } from 'lucide-react';

interface HighlightPaletteProps {
  containerRef: React.RefObject<HTMLDivElement | null>;
  chapterId: string;
  chapterTitle: string;
  onHighlightCreated?: () => void;
}

const COLOR_STYLES: Record<MarkerColor, { bg: string; border: string; label: string }> = {
  yellow: { bg: '#eab308', border: '#fef08a', label: 'Yellow (Core Concept)' },
  emerald: { bg: '#10b981', border: '#a7f3d0', label: 'Green (Hardware/Cache)' },
  purple: { bg: '#a855f7', border: '#e9d5ff', label: 'Purple (Race/Pitfall)' },
  cyan: { bg: '#06b6d4', border: '#cffafe', label: 'Cyan (Pointer/Math)' },
};

export function HighlightPalette({
  containerRef,
  chapterId,
  chapterTitle,
  onHighlightCreated,
}: HighlightPaletteProps) {
  const [position, setPosition] = useState<{ x: number; y: number } | null>(null);
  const [selectedText, setSelectedText] = useState<string>('');
  const [showNoteInput, setShowNoteInput] = useState<boolean>(false);
  const [pendingColor, setPendingColor] = useState<MarkerColor>('yellow');
  const [noteText, setNoteText] = useState<string>('');
  const paletteRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleMouseUp = () => {
      // Small timeout to allow selection to finalize
      setTimeout(() => {
        const selection = window.getSelection();
        if (!selection || selection.isCollapsed || !containerRef.current) {
          if (!showNoteInput) {
            setPosition(null);
            setSelectedText('');
          }
          return;
        }

        const text = selection.toString().trim();
        if (text.length < 2) {
          if (!showNoteInput) setPosition(null);
          return;
        }

        // Verify that the selection is inside our container
        const range = selection.getRangeAt(0);
        if (!containerRef.current.contains(range.commonAncestorContainer)) {
          setPosition(null);
          return;
        }

        const rect = range.getBoundingClientRect();
        setSelectedText(text);

        // Position slightly above the selection
        setPosition({
          x: Math.max(10, rect.left + rect.width / 2),
          y: Math.max(10, rect.top - 12 + window.scrollY),
        });
      }, 30);
    };

    const handleMouseDown = (e: MouseEvent) => {
      if (paletteRef.current && paletteRef.current.contains(e.target as Node)) {
        return;
      }
      if (!showNoteInput) {
        setPosition(null);
      }
    };

    document.addEventListener('mouseup', handleMouseUp);
    document.addEventListener('mousedown', handleMouseDown);

    return () => {
      document.removeEventListener('mouseup', handleMouseUp);
      document.removeEventListener('mousedown', handleMouseDown);
    };
  }, [containerRef, showNoteInput]);

  const handleApplyHighlight = (color: MarkerColor, withNote: boolean = false) => {
    if (!selectedText) return;

    if (withNote) {
      setPendingColor(color);
      setShowNoteInput(true);
      return;
    }

    saveHighlight({
      chapterId,
      chapterTitle,
      text: selectedText,
      color,
    });

    // Clear selection
    window.getSelection()?.removeAllRanges();
    setPosition(null);
    setSelectedText('');
    setShowNoteInput(false);
    setNoteText('');
    onHighlightCreated?.();
  };

  const handleSaveWithNote = () => {
    if (!selectedText) return;

    saveHighlight({
      chapterId,
      chapterTitle,
      text: selectedText,
      color: pendingColor,
      note: noteText.trim() || undefined,
    });

    window.getSelection()?.removeAllRanges();
    setPosition(null);
    setSelectedText('');
    setShowNoteInput(false);
    setNoteText('');
    onHighlightCreated?.();
  };

  if (!position) return null;

  return (
    <div
      ref={paletteRef}
      className="fixed z-50 transform -translate-x-1/2 -translate-y-full"
      style={{ left: `${position.x}px`, top: `${position.y}px` }}
    >
      <div className="bg-[#161b22] border border-[#30363d] rounded-xl shadow-2xl p-1.5 flex flex-col gap-2 backdrop-blur-md">
        {!showNoteInput ? (
          <div className="flex items-center gap-1.5">
            {/* Color buttons */}
            {(['yellow', 'emerald', 'purple', 'cyan'] as MarkerColor[]).map((c) => (
              <button
                key={c}
                onClick={() => handleApplyHighlight(c, false)}
                title={COLOR_STYLES[c].label}
                className="w-6 h-6 rounded-full transition-transform hover:scale-125 focus:scale-125 border border-white/20 shadow"
                style={{ backgroundColor: COLOR_STYLES[c].bg }}
              />
            ))}

            <div className="w-[1px] h-4 bg-[#30363d] mx-1" />

            {/* Note prompt button */}
            <button
              onClick={() => handleApplyHighlight(pendingColor, true)}
              className="flex items-center gap-1 px-2 py-1 rounded-md text-[11px] font-medium text-slate-300 hover:text-white bg-[#21262d] hover:bg-[#30363d] transition-colors"
              title="Highlight with a note"
            >
              <MessageSquarePlus className="w-3.5 h-3.5 text-amber-400" />
              <span>Note</span>
            </button>
          </div>
        ) : (
          <div className="flex flex-col gap-2 p-2 w-64">
            <div className="flex items-center justify-between text-xs text-slate-400 font-sans">
              <span className="flex items-center gap-1.5">
                <span
                  className="w-2.5 h-2.5 rounded-full"
                  style={{ backgroundColor: COLOR_STYLES[pendingColor].bg }}
                />
                Attach Study Note
              </span>
              <button
                onClick={() => {
                  setShowNoteInput(false);
                  setPosition(null);
                }}
                className="text-slate-400 hover:text-white"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
            <textarea
              autoFocus
              value={noteText}
              onChange={(e) => setNoteText(e.target.value)}
              placeholder="Add your insight, memory trick, or question..."
              rows={3}
              className="w-full text-xs bg-[#0d1117] border border-[#30363d] rounded-lg p-2 text-slate-200 placeholder-slate-500 focus:outline-none focus:border-amber-400 resize-none font-sans"
            />
            <div className="flex items-center justify-end gap-1.5">
              <button
                onClick={() => setShowNoteInput(false)}
                className="px-2 py-1 text-[11px] rounded text-slate-400 hover:text-slate-200"
              >
                Cancel
              </button>
              <button
                onClick={handleSaveWithNote}
                className="flex items-center gap-1 px-3 py-1 bg-amber-500 hover:bg-amber-400 text-slate-950 font-semibold rounded text-[11px] transition-all"
              >
                <Check className="w-3 h-3" /> Save Note
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
