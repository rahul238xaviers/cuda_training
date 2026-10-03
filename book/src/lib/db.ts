import path from 'path';
import fs from 'fs';

// Node 22+ built-in SQLite database engine
let dbInstance: any = null;

export function getDb(): any {
  if (!dbInstance) {
    const dataDir = path.join(process.cwd(), 'data');
    if (!fs.existsSync(dataDir)) {
      fs.mkdirSync(dataDir, { recursive: true });
    }
    const dbPath = path.join(dataDir, 'user_notebook.db');

    // Dynamic load for Node's built-in sqlite module
    const { DatabaseSync } = eval('require')('node:sqlite');
    dbInstance = new DatabaseSync(dbPath);

    // Initialize database tables if not already present
    dbInstance.exec(`
      CREATE TABLE IF NOT EXISTS bookmarks (
        chapter_id TEXT PRIMARY KEY,
        chapter_title TEXT NOT NULL,
        created_at INTEGER NOT NULL
      );

      CREATE TABLE IF NOT EXISTS highlights (
        id TEXT PRIMARY KEY,
        chapter_id TEXT NOT NULL,
        chapter_title TEXT,
        text TEXT NOT NULL,
        color TEXT NOT NULL,
        note TEXT,
        created_at INTEGER NOT NULL
      );

      CREATE INDEX IF NOT EXISTS idx_highlights_chapter ON highlights(chapter_id);
    `);
  }
  return dbInstance;
}
