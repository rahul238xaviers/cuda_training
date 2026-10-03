'use client';

import React, { useState, useEffect } from 'react';
import {
  ChapterBookmark,
  ChapterHighlight,
  getBookmarks,
  getHighlights,
  removeHighlight,
  toggleChapterBookmark,
  updateHighlightNote,
  MarkerColor,
} from '../lib/annotations';
import {
  Bookmark,
  Highlighter,
  Trash2,
  ExternalLink,
  Search,
  X,
  FileDown,
  BookOpen,
  Calendar,
  Edit2,
  Check,
} from 'lucide-react';

interface AnnotationsDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  onNavigateChapter: (chapterId: string) => void;
  currentChapterId?: string;
}

const COLOR_BADGES: Record<MarkerColor, { bg: string; text: string; label: string }> = {
  yellow: { bg: 'bg-amber-500/20 border-amber-500/40 text-amber-300', text: 'text-amber-400', label: 'Concept' },
  emerald: { bg: 'bg-emerald-500/20 border-emerald-500/40 text-emerald-300', text: 'text-emerald-400', label: 'Hardware' },
  purple: { bg: 'bg-purple-500/20 border-purple-500/40 text-purple-300', text: 'text-purple-400', label: 'Pitfall' },
  cyan: { bg: 'bg-sky-500/20 border-sky-500/40 text-sky-300', text: 'text-sky-400', label: 'Formula' },
};

export function AnnotationsDrawer({
  isOpen,
  onClose,
  onNavigateChapter,
  currentChapterId,
}: AnnotationsDrawerProps) {
  const [activeTab, setActiveTab] = useState<'highlights' | 'bookmarks'>('highlights');
  const [highlights, setHighlights] = useState<ChapterHighlight[]>([]);
  const [bookmarks, setBookmarks] = useState<ChapterBookmark[]>([]);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [selectedColor, setSelectedColor] = useState<MarkerColor | 'all'>('all');
  const [editingNoteId, setEditingNoteId] = useState<string | null>(null);
  const [editText, setEditText] = useState<string>('');

  const refreshData = () => {
    setHighlights(getHighlights());
    setBookmarks(getBookmarks());
  };

  useEffect(() => {
    refreshData();
    const handleUpdate = () => refreshData();
    window.addEventListener('cuda-annotations-updated', handleUpdate);
    return () => window.removeEventListener('cuda-annotations-updated', handleUpdate);
  }, []);

  if (!isOpen) return null;

  const filteredHighlights = highlights.filter((h) => {
    const matchesColor = selectedColor === 'all' || h.color === selectedColor;
    const matchesQuery =
      !searchQuery ||
      h.text.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (h.note && h.note.toLowerCase().includes(searchQuery.toLowerCase())) ||
      (h.chapterTitle && h.chapterTitle.toLowerCase().includes(searchQuery.toLowerCase()));
    return matchesColor && matchesQuery;
  });

  const filteredBookmarks = bookmarks.filter((b) => {
    return (
      !searchQuery ||
      b.chapterTitle.toLowerCase().includes(searchQuery.toLowerCase()) ||
      b.chapterId.toLowerCase().includes(searchQuery.toLowerCase())
    );
  });

  const handleExportMarkdown = () => {
    let md = '# Study Notes & Highlights — C++ & CUDA Systems Mastery\n\n';
    md += `*Exported on ${new Date().toLocaleDateString()}*\n\n---\n\n`;

    if (bookmarks.length > 0) {
      md += '## Bookmarked Chapters\n\n';
      bookmarks.forEach((b) => {
        md += `- **Chapter ${b.chapterId}**: ${b.chapterTitle}\n`;
      });
      md += '\n---\n\n';
    }

    if (highlights.length > 0) {
      md += '## Highlighted Passages & Notes\n\n';
      highlights.forEach((h) => {
        md += `### Chapter ${h.chapterId}: ${h.chapterTitle || 'Chapter Notes'}\n`;
        md += `> "${h.text}"\n\n`;
        if (h.note) {
          md += `**Note**: ${h.note}\n\n`;
        }
        md += `*Tag: [${h.color.toUpperCase()}] • Saved on ${new Date(h.createdAt).toLocaleDateString()}*\n\n`;
      });
    }

    const blob = new Blob([md], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `cuda_training_study_notes_${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="fixed inset-0 z-50 overflow-hidden bg-black/60 backdrop-blur-sm flex justify-end">
      <div className="w-full max-w-md bg-[#0e1219] border-l border-[#30363d] h-full flex flex-col shadow-2xl animate-in slide-in-from-right duration-200">
        {/* Drawer Header */}
        <div className="flex items-center justify-between p-4 border-b border-[#21262d] bg-[#141923]">
          <div className="flex items-center gap-2">
            <Highlighter className="w-4 h-4 text-amber-400" />
            <h3 className="font-semibold text-slate-100 text-sm">Study Notes & Highlights</h3>
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={handleExportMarkdown}
              title="Export all notes to Markdown"
              className="p-1.5 rounded hover:bg-[#21262d] text-slate-400 hover:text-slate-200 text-xs flex items-center gap-1"
            >
              <FileDown className="w-4 h-4" />
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded hover:bg-[#21262d] text-slate-400 hover:text-slate-200"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Tab Selection */}
        <div className="flex items-center border-b border-[#21262d] px-4 pt-2 bg-[#121620]">
          <button
            onClick={() => setActiveTab('highlights')}
            className={`pb-2.5 px-3 text-xs font-medium border-b-2 flex items-center gap-1.5 transition-colors ${
              activeTab === 'highlights'
                ? 'border-amber-400 text-amber-300'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <Highlighter className="w-3.5 h-3.5" />
            Highlights & Notes ({highlights.length})
          </button>
          <button
            onClick={() => setActiveTab('bookmarks')}
            className={`pb-2.5 px-3 text-xs font-medium border-b-2 flex items-center gap-1.5 transition-colors ${
              activeTab === 'bookmarks'
                ? 'border-amber-400 text-amber-300'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <Bookmark className="w-3.5 h-3.5" />
            Bookmarks ({bookmarks.length})
          </button>
        </div>

        {/* Search & Filter Bar */}
        <div className="p-3 border-b border-[#21262d] flex flex-col gap-2 bg-[#0c1017]">
          <div className="relative">
            <Search className="w-3.5 h-3.5 text-slate-500 absolute left-2.5 top-2.5" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search quotes, notes, or chapters..."
              className="w-full text-xs pl-8 pr-3 py-1.5 bg-[#141923] border border-[#30363d] rounded-lg text-slate-200 placeholder-slate-500 focus:outline-none focus:border-amber-400"
            />
          </div>

          {activeTab === 'highlights' && (
            <div className="flex items-center gap-1 text-[11px] overflow-x-auto pb-1">
              <button
                onClick={() => setSelectedColor('all')}
                className={`px-2 py-0.5 rounded-full border transition-colors ${
                  selectedColor === 'all'
                    ? 'bg-slate-700 text-white border-slate-500'
                    : 'text-slate-400 border-transparent hover:text-slate-200'
                }`}
              >
                All Colors
              </button>
              {(['yellow', 'emerald', 'purple', 'cyan'] as MarkerColor[]).map((c) => (
                <button
                  key={c}
                  onClick={() => setSelectedColor(c)}
                  className={`px-2 py-0.5 rounded-full border transition-colors ${
                    selectedColor === c
                      ? COLOR_BADGES[c].bg
                      : 'text-slate-400 border-transparent hover:text-slate-200'
                  }`}
                >
                  {COLOR_BADGES[c].label}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* List Content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {activeTab === 'highlights' ? (
            filteredHighlights.length === 0 ? (
              <div className="text-center py-12 text-slate-500 text-xs">
                <Highlighter className="w-8 h-8 mx-auto mb-2 opacity-30 text-amber-400" />
                <p>No highlights found.</p>
                <p className="mt-1 text-[11px] text-slate-600">
                  Select text anywhere in the chapter to highlight and take notes.
                </p>
              </div>
            ) : (
              filteredHighlights.map((h) => {
                const isCurrent = h.chapterId === currentChapterId;
                const isEditing = editingNoteId === h.id;

                return (
                  <div
                    key={h.id}
                    className="p-3 rounded-xl bg-[#121622] border border-[#21262d] hover:border-[#30363d] transition-all flex flex-col gap-2 text-xs"
                  >
                    <div className="flex items-center justify-between">
                      <span className={`text-[10px] font-mono px-2 py-0.5 rounded border ${COLOR_BADGES[h.color].bg}`}>
                        {COLOR_BADGES[h.color].label}
                      </span>
                      <div className="flex items-center gap-1.5 text-slate-500 text-[10px]">
                        <span>{new Date(h.createdAt).toLocaleDateString()}</span>
                        <button
                          onClick={() => removeHighlight(h.id)}
                          className="hover:text-rose-400 p-0.5 rounded"
                          title="Delete highlight"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>

                    {/* Quote */}
                    <div className="italic text-slate-200 border-l-2 pl-2.5 border-slate-700 py-0.5 text-[11.5px] leading-relaxed">
                      "{h.text}"
                    </div>

                    {/* Personal Note */}
                    {isEditing ? (
                      <div className="mt-1 space-y-1.5">
                        <textarea
                          value={editText}
                          onChange={(e) => setEditText(e.target.value)}
                          rows={2}
                          className="w-full text-xs bg-[#090d14] border border-amber-500/50 rounded p-1.5 text-slate-200 focus:outline-none"
                        />
                        <div className="flex justify-end gap-1">
                          <button
                            onClick={() => setEditingNoteId(null)}
                            className="px-2 py-0.5 text-[10px] text-slate-400 hover:text-white"
                          >
                            Cancel
                          </button>
                          <button
                            onClick={() => {
                              updateHighlightNote(h.id, editText.trim());
                              setEditingNoteId(null);
                            }}
                            className="px-2 py-0.5 text-[10px] bg-amber-500 text-black font-semibold rounded flex items-center gap-1"
                          >
                            <Check className="w-3 h-3" /> Save
                          </button>
                        </div>
                      </div>
                    ) : (
                      h.note && (
                        <div className="mt-1 p-2 rounded bg-[#0b0e14] border border-[#1e2330] text-[11px] text-amber-200/90 flex items-start justify-between gap-2">
                          <div className="leading-relaxed">
                            <span className="font-semibold text-amber-400 mr-1">Note:</span>
                            {h.note}
                          </div>
                          <button
                            onClick={() => {
                              setEditingNoteId(h.id);
                              setEditText(h.note || '');
                            }}
                            className="text-slate-500 hover:text-slate-300 shrink-0"
                            title="Edit note"
                          >
                            <Edit2 className="w-3 h-3" />
                          </button>
                        </div>
                      )
                    )}

                    {!h.note && !isEditing && (
                      <button
                        onClick={() => {
                          setEditingNoteId(h.id);
                          setEditText('');
                        }}
                        className="text-[10px] text-slate-500 hover:text-amber-400 self-start mt-0.5 flex items-center gap-1"
                      >
                        + Add note to this highlight
                      </button>
                    )}

                    {/* Chapter Link */}
                    <div className="mt-1 pt-1.5 border-t border-[#1a2130] flex items-center justify-between text-[10px] text-slate-400">
                      <span>Chapter {h.chapterId}</span>
                      <button
                        onClick={() => {
                          onNavigateChapter(h.chapterId);
                          onClose();
                        }}
                        className="text-sky-400 hover:text-sky-300 flex items-center gap-1 font-mono"
                      >
                        Jump to Chapter <ExternalLink className="w-3 h-3" />
                      </button>
                    </div>
                  </div>
                );
              })
            )
          ) : filteredBookmarks.length === 0 ? (
            <div className="text-center py-12 text-slate-500 text-xs">
              <Bookmark className="w-8 h-8 mx-auto mb-2 opacity-30 text-amber-400" />
              <p>No bookmarked chapters.</p>
              <p className="mt-1 text-[11px] text-slate-600">
                Click the bookmark icon on any chapter header to save it here.
              </p>
            </div>
          ) : (
            filteredBookmarks.map((b) => (
              <div
                key={b.chapterId}
                className="p-3 rounded-xl bg-[#121622] border border-[#21262d] hover:border-[#30363d] flex items-center justify-between text-xs"
              >
                <div className="flex items-center gap-2.5">
                  <Bookmark className="w-4 h-4 text-amber-400 fill-amber-400/20 shrink-0" />
                  <div>
                    <h5 className="font-semibold text-slate-200">Chapter {b.chapterId}</h5>
                    <p className="text-[11px] text-slate-400 truncate max-w-[200px]">
                      {b.chapterTitle}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-1">
                  <button
                    onClick={() => {
                      onNavigateChapter(b.chapterId);
                      onClose();
                    }}
                    className="p-1.5 rounded hover:bg-[#1f2736] text-sky-400"
                    title="Open Chapter"
                  >
                    <ExternalLink className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={() => toggleChapterBookmark(b.chapterId, b.chapterTitle)}
                    className="p-1.5 rounded hover:bg-[#1f2736] text-rose-400"
                    title="Remove Bookmark"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
