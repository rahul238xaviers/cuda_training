import { NextRequest, NextResponse } from 'next/server';
import { getDb } from '@/lib/db';

export async function GET(request: NextRequest) {
  try {
    const db = getDb();
    const { searchParams } = new URL(request.url);
    const chapterId = searchParams.get('chapterId');

    const bookmarksQuery = db.prepare('SELECT chapter_id as chapterId, chapter_title as chapterTitle, created_at as createdAt FROM bookmarks ORDER BY created_at DESC');
    const bookmarks = bookmarksQuery.all();

    let highlights;
    if (chapterId) {
      const highlightsQuery = db.prepare('SELECT id, chapter_id as chapterId, chapter_title as chapterTitle, text, color, note, created_at as createdAt FROM highlights WHERE chapter_id = ? ORDER BY created_at DESC');
      highlights = highlightsQuery.all(chapterId);
    } else {
      const highlightsQuery = db.prepare('SELECT id, chapter_id as chapterId, chapter_title as chapterTitle, text, color, note, created_at as createdAt FROM highlights ORDER BY created_at DESC');
      highlights = highlightsQuery.all();
    }

    return NextResponse.json({
      success: true,
      bookmarks,
      highlights,
    });
  } catch (error: any) {
    console.error('Failed to query SQLite annotations:', error);
    return NextResponse.json(
      { success: false, error: error.message || 'Database error' },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const db = getDb();
    const body = await request.json();
    const { type, data } = body;

    if (type === 'bookmark') {
      const { chapterId, chapterTitle } = data;
      // Toggle bookmark
      const checkStmt = db.prepare('SELECT chapter_id FROM bookmarks WHERE chapter_id = ?');
      const existing = checkStmt.get(chapterId);

      if (existing) {
        const delStmt = db.prepare('DELETE FROM bookmarks WHERE chapter_id = ?');
        delStmt.run(chapterId);
        return NextResponse.json({ success: true, action: 'removed', bookmarked: false });
      } else {
        const insStmt = db.prepare('INSERT INTO bookmarks (chapter_id, chapter_title, created_at) VALUES (?, ?, ?)');
        insStmt.run(chapterId, chapterTitle, Date.now());
        return NextResponse.json({ success: true, action: 'added', bookmarked: true });
      }
    }

    if (type === 'highlight') {
      const { id, chapterId, chapterTitle, text, color, note } = data;
      const highlightId = id || 'hl_' + Math.random().toString(36).substring(2, 9) + '_' + Date.now();
      const createdAt = Date.now();

      const insStmt = db.prepare(`
        INSERT INTO highlights (id, chapter_id, chapter_title, text, color, note, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          note = excluded.note,
          color = excluded.color
      `);
      insStmt.run(highlightId, chapterId, chapterTitle || '', text, color, note || null, createdAt);

      return NextResponse.json({
        success: true,
        highlight: {
          id: highlightId,
          chapterId,
          chapterTitle,
          text,
          color,
          note,
          createdAt,
        },
      });
    }

    return NextResponse.json({ success: false, error: 'Invalid entity type' }, { status: 400 });
  } catch (error: any) {
    console.error('Failed to save SQLite annotation:', error);
    return NextResponse.json(
      { success: false, error: error.message || 'Database error' },
      { status: 500 }
    );
  }
}

export async function DELETE(request: NextRequest) {
  try {
    const db = getDb();
    const body = await request.json();
    const { type, id } = body;

    if (type === 'highlight') {
      const stmt = db.prepare('DELETE FROM highlights WHERE id = ?');
      stmt.run(id);
      return NextResponse.json({ success: true, action: 'deleted' });
    }

    if (type === 'bookmark') {
      const stmt = db.prepare('DELETE FROM bookmarks WHERE chapter_id = ?');
      stmt.run(id);
      return NextResponse.json({ success: true, action: 'deleted' });
    }

    return NextResponse.json({ success: false, error: 'Invalid delete target' }, { status: 400 });
  } catch (error: any) {
    console.error('Failed to delete SQLite annotation:', error);
    return NextResponse.json(
      { success: false, error: error.message || 'Database error' },
      { status: 500 }
    );
  }
}
