/**
 * Client-side Storage and State Management for Bookmarks, Notes, and Multi-Color Text Highlights.
 * Decoupled from rendering and chapter logic.
 */

export type MarkerColor = 'yellow' | 'emerald' | 'purple' | 'cyan';

export interface ChapterHighlight {
  id: string;
  chapterId: string;
  chapterTitle?: string;
  text: string;
  color: MarkerColor;
  note?: string;
  createdAt: number;
}

export interface ChapterBookmark {
  chapterId: string;
  chapterTitle: string;
  createdAt: number;
}

const HIGHLIGHTS_KEY = 'cuda_lab_highlights_v1';
const BOOKMARKS_KEY = 'cuda_lab_bookmarks_v1';

function emitChange() {
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent('cuda-annotations-updated'));
  }
}

export function getBookmarks(): ChapterBookmark[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = localStorage.getItem(BOOKMARKS_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch (e) {
    console.error('Failed to read bookmarks from localStorage', e);
    return [];
  }
}

export function isChapterBookmarked(chapterId: string): boolean {
  const bookmarks = getBookmarks();
  return bookmarks.some((b) => b.chapterId === chapterId);
}

export function toggleChapterBookmark(chapterId: string, chapterTitle: string): boolean {
  const bookmarks = getBookmarks();
  const exists = bookmarks.some((b) => b.chapterId === chapterId);
  let updated: ChapterBookmark[];

  if (exists) {
    updated = bookmarks.filter((b) => b.chapterId !== chapterId);
  } else {
    updated = [
      ...bookmarks,
      {
        chapterId,
        chapterTitle,
        createdAt: Date.now(),
      },
    ];
  }

  try {
    localStorage.setItem(BOOKMARKS_KEY, JSON.stringify(updated));
    emitChange();
  } catch (e) {
    console.error('Failed to save bookmark', e);
  }

  return !exists;
}

export function getHighlights(chapterId?: string): ChapterHighlight[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = localStorage.getItem(HIGHLIGHTS_KEY);
    const list: ChapterHighlight[] = raw ? JSON.parse(raw) : [];
    if (chapterId) {
      return list.filter((h) => h.chapterId === chapterId);
    }
    return list;
  } catch (e) {
    console.error('Failed to read highlights from localStorage', e);
    return [];
  }
}

export function saveHighlight(item: Omit<ChapterHighlight, 'id' | 'createdAt'>): ChapterHighlight {
  const list = getHighlights();
  const newHighlight: ChapterHighlight = {
    ...item,
    id: 'hl_' + Math.random().toString(36).substring(2, 9) + '_' + Date.now(),
    createdAt: Date.now(),
  };

  const updated = [newHighlight, ...list];
  try {
    localStorage.setItem(HIGHLIGHTS_KEY, JSON.stringify(updated));
    emitChange();
  } catch (e) {
    console.error('Failed to save highlight', e);
  }

  return newHighlight;
}

export function removeHighlight(id: string): void {
  const list = getHighlights();
  const updated = list.filter((h) => h.id !== id);
  try {
    localStorage.setItem(HIGHLIGHTS_KEY, JSON.stringify(updated));
    emitChange();
  } catch (e) {
    console.error('Failed to remove highlight', e);
  }
}

export function updateHighlightNote(id: string, note: string): void {
  const list = getHighlights();
  const updated = list.map((h) => (h.id === id ? { ...h, note } : h));
  try {
    localStorage.setItem(HIGHLIGHTS_KEY, JSON.stringify(updated));
    emitChange();
  } catch (e) {
    console.error('Failed to update highlight note', e);
  }
}
