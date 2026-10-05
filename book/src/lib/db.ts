import Database from 'better-sqlite3';
import fs from 'fs';
import path from 'path';

export const WORKSPACE_ROOT = path.resolve(process.cwd(), '..');
export const USER_DATA_DIR = path.join(WORKSPACE_ROOT, '.user_data');
export const DB_PATH = path.join(USER_DATA_DIR, 'cuda_lab.db');

let _db: Database.Database | null = null;

export function getDatabase(): Database.Database {
  if (!_db) {
    if (!fs.existsSync(USER_DATA_DIR)) {
      fs.mkdirSync(USER_DATA_DIR, { recursive: true });
    }

    _db = new Database(DB_PATH);
    _db.pragma('journal_mode = WAL');
    _db.pragma('foreign_keys = ON');

    initSchema(_db);
  }
  return _db;
}

export const getDb = getDatabase;

function initSchema(db: Database.Database) {
  db.exec(`
    CREATE TABLE IF NOT EXISTS canonical_problems (
      problem_id TEXT PRIMARY KEY,
      topic_id TEXT NOT NULL,
      volume_id TEXT NOT NULL,
      tier TEXT NOT NULL,
      problem_num INTEGER NOT NULL,
      title TEXT NOT NULL,
      status_label TEXT,
      stub_code TEXT,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_canonical_topic_tier 
      ON canonical_problems(topic_id, tier);

    CREATE TABLE IF NOT EXISTS user_problem_status (
      problem_id TEXT PRIMARY KEY,
      status TEXT NOT NULL DEFAULT 'pending',
      last_run_at DATETIME,
      throughput_gbs REAL DEFAULT -1.0,
      duration_ms INTEGER DEFAULT 0,
      error_log TEXT,
      FOREIGN KEY(problem_id) REFERENCES canonical_problems(problem_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS user_annotations (
      id TEXT PRIMARY KEY,
      chapter_id TEXT NOT NULL,
      type TEXT NOT NULL,
      content TEXT,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS bookmarks (
      chapter_id TEXT PRIMARY KEY,
      chapter_title TEXT NOT NULL,
      created_at INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS highlights (
      id TEXT PRIMARY KEY,
      chapter_id TEXT NOT NULL,
      chapter_title TEXT NOT NULL,
      text TEXT NOT NULL,
      color TEXT NOT NULL,
      note TEXT,
      created_at INTEGER NOT NULL
    );

    CREATE VIEW IF NOT EXISTS v_tier_summary AS
    SELECT 
      cp.topic_id,
      cp.volume_id,
      cp.tier,
      COUNT(cp.problem_id) AS total_problems,
      SUM(CASE WHEN ups.status = 'passed' THEN 1 ELSE 0 END) AS passed_problems,
      CASE 
        WHEN COUNT(cp.problem_id) > 0 AND SUM(CASE WHEN ups.status = 'passed' THEN 1 ELSE 0 END) = COUNT(cp.problem_id)
        THEN 'passed'
        ELSE 'pending'
      END AS tier_status
    FROM canonical_problems cp
    LEFT JOIN user_problem_status ups ON cp.problem_id = ups.problem_id
    GROUP BY cp.topic_id, cp.tier;
  `);
}

export interface CanonicalProblem {
  problem_id: string;
  topic_id: string;
  volume_id: string;
  tier: 'beginner' | 'intermediate' | 'champion';
  problem_num: number;
  title: string;
  status_label?: string;
  stub_code?: string;
}

export interface UserProblemRecord {
  problem_id: string;
  topic_id: string;
  tier: 'beginner' | 'intermediate' | 'champion';
  problem_num: number;
  title: string;
  status: 'passed' | 'pending' | 'failed';
  last_run_at?: string;
  throughput_gbs?: number;
  duration_ms?: number;
}

export interface TierSummary {
  topic_id: string;
  volume_id: string;
  tier: 'beginner' | 'intermediate' | 'champion';
  total_problems: number;
  passed_problems: number;
  tier_status: 'passed' | 'pending';
}

/**
 * Inserts or updates a canonical problem parsed from exercise/.
 */
export function upsertCanonicalProblem(p: CanonicalProblem): void {
  const db = getDatabase();
  const stmt = db.prepare(`
    INSERT INTO canonical_problems (problem_id, topic_id, volume_id, tier, problem_num, title, status_label, stub_code, updated_at)
    VALUES (@problem_id, @topic_id, @volume_id, @tier, @problem_num, @title, @status_label, @stub_code, CURRENT_TIMESTAMP)
    ON CONFLICT(problem_id) DO UPDATE SET
      title = excluded.title,
      status_label = excluded.status_label,
      stub_code = excluded.stub_code,
      updated_at = CURRENT_TIMESTAMP
  `);
  stmt.run({
    problem_id: p.problem_id,
    topic_id: p.topic_id,
    volume_id: p.volume_id,
    tier: p.tier,
    problem_num: p.problem_num,
    title: p.title,
    status_label: p.status_label || null,
    stub_code: p.stub_code || null,
  });
}

/**
 * Gets all canonical problems for a specific chapter and tier.
 */
export function getCanonicalProblems(topicId: string, tier: string): CanonicalProblem[] {
  const db = getDatabase();
  const stmt = db.prepare(`
    SELECT problem_id, topic_id, volume_id, tier, problem_num, title, status_label, stub_code
    FROM canonical_problems
    WHERE topic_id = ? AND tier = ?
    ORDER BY problem_num ASC
  `);
  return stmt.all(topicId, tier) as CanonicalProblem[];
}

/**
 * Gets problem statuses for a specific chapter and tier.
 */
export function getUserProblemsWithStatus(topicId: string, tier: string): UserProblemRecord[] {
  const db = getDatabase();
  const stmt = db.prepare(`
    SELECT 
      cp.problem_id,
      cp.topic_id,
      cp.tier,
      cp.problem_num,
      cp.title,
      COALESCE(ups.status, 'pending') AS status,
      ups.last_run_at,
      ups.throughput_gbs,
      ups.duration_ms
    FROM canonical_problems cp
    LEFT JOIN user_problem_status ups ON cp.problem_id = ups.problem_id
    WHERE cp.topic_id = ? AND cp.tier = ?
    ORDER BY cp.problem_num ASC
  `);
  return stmt.all(topicId, tier) as UserProblemRecord[];
}

/**
 * Updates a specific problem status for the user.
 */
export function recordProblemStatus(
  problemId: string,
  status: 'passed' | 'pending' | 'failed',
  throughputGbs: number = -1.0,
  durationMs: number = 0,
  errorLog?: string
): void {
  const db = getDatabase();
  const stmt = db.prepare(`
    INSERT INTO user_problem_status (problem_id, status, last_run_at, throughput_gbs, duration_ms, error_log)
    VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?, ?)
    ON CONFLICT(problem_id) DO UPDATE SET
      status = excluded.status,
      last_run_at = CURRENT_TIMESTAMP,
      throughput_gbs = excluded.throughput_gbs,
      duration_ms = excluded.duration_ms,
      error_log = excluded.error_log
  `);
  stmt.run(problemId, status, throughputGbs, durationMs, errorLog || null);
}

/**
 * Retrieves the tier summary for a topic and tier.
 */
export function getTierSummary(topicId: string, tier: string): TierSummary | null {
  const db = getDatabase();
  const stmt = db.prepare(`
    SELECT topic_id, volume_id, tier, total_problems, passed_problems, tier_status
    FROM v_tier_summary
    WHERE topic_id = ? AND tier = ?
  `);
  const row = stmt.get(topicId, tier) as TierSummary | undefined;
  return row || null;
}

/**
 * Retrieves the global course-wide test statistics.
 */
export function getCourseStats(): { totalTests: number; testsPassed: number; percent: number } {
  const db = getDatabase();
  const stmt = db.prepare(`
    SELECT 
      COUNT(cp.problem_id) AS totalTests,
      SUM(CASE WHEN ups.status = 'passed' THEN 1 ELSE 0 END) AS testsPassed
    FROM canonical_problems cp
    LEFT JOIN user_problem_status ups ON cp.problem_id = ups.problem_id
  `);
  const row = stmt.get() as { totalTests: number; testsPassed: number } | undefined;
  const totalTests = row?.totalTests || 0;
  const testsPassed = row?.testsPassed || 0;
  const percent = totalTests > 0 ? Math.round((testsPassed / totalTests) * 100) : 0;
  return { totalTests, testsPassed, percent };
}
