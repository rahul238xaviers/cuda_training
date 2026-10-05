import fs from 'fs';
import path from 'path';
import {
  upsertCanonicalProblem,
  CanonicalProblem,
  recordProblemStatus,
} from './db';

export interface ExtractedProblem {
  problem_num: number;
  title: string;
  status_label: string;
  code_block: string;
}

/**
 * Parses an exercise file and extracts all canonical problem blocks.
 */
export function parseExerciseProblems(filePath: string): ExtractedProblem[] {
  if (!fs.existsSync(filePath)) return [];

  const content = fs.readFileSync(filePath, 'utf-8');
  const lines = content.split('\n');
  const problems: ExtractedProblem[] = [];

  // Match: // PROBLEM 1: Title, // Problem 2 - Title, etc.
  const headerRegex = /^\s*\/\/\s*(?:[-=*#]+\s*)?(?:PROBLEM|Problem)\s+(\d+)\s*[:\-—]\s*(.+)$/;

  interface HeaderMatch {
    lineIdx: number;
    problem_num: number;
    title: string;
  }

  const matches: HeaderMatch[] = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const match = line.match(headerRegex);
    if (match) {
      matches.push({
        lineIdx: i,
        problem_num: parseInt(match[1], 10),
        title: match[2].trim(),
      });
    }
  }

  if (matches.length === 0) {
    // Fallback: Check for reportStatus calls if explicit // PROBLEM headers aren't used
    const reportRegex = /reportStatus\s*\(\s*["']([^"']+)["']/g;
    let rMatch;
    let idx = 1;
    while ((rMatch = reportRegex.exec(content)) !== null) {
      problems.push({
        problem_num: idx,
        title: rMatch[1].trim(),
        status_label: rMatch[1].trim(),
        code_block: '',
      });
      idx++;
    }
    return problems;
  }

  // Find start of scorecard or main end to bound the last problem
  let endOfProblemsLine = lines.length;
  for (let i = lines.length - 1; i >= 0; i--) {
    if (
      lines[i].includes('--- SCORECARD ---') ||
      lines[i].includes('SCORECARD') ||
      lines[i].includes('return (passed == total)') ||
      lines[i].includes('return passed == total') ||
      lines[i].includes('return 0;')
    ) {
      endOfProblemsLine = i;
      // Walk back any leading divider comments
      while (endOfProblemsLine > 0 && lines[endOfProblemsLine - 1].trim().startsWith('//')) {
        endOfProblemsLine--;
      }
      break;
    }
  }

  for (let i = 0; i < matches.length; i++) {
    const curr = matches[i];
    // Start of block: back up to include any decorative divider line like // -----
    let startLine = curr.lineIdx;
    if (startLine > 0 && lines[startLine - 1].trim().startsWith('//') && lines[startLine - 1].includes('---')) {
      startLine = startLine - 1;
    }

    const nextStartLine = i + 1 < matches.length ? matches[i + 1].lineIdx : endOfProblemsLine;
    let endLine = nextStartLine;

    // Back up from next header's decorative comment
    if (i + 1 < matches.length && endLine > 0 && lines[endLine - 1].trim().startsWith('//') && lines[endLine - 1].includes('---')) {
      endLine = endLine - 1;
    }

    const blockLines = lines.slice(startLine, endLine);
    const codeBlock = blockLines.join('\n').trim();

    // Extract status label from reportStatus within this block
    const statusMatch = codeBlock.match(/reportStatus\s*\(\s*["']([^"']+)["']/);
    const status_label = statusMatch ? statusMatch[1].trim() : curr.title;

    problems.push({
      problem_num: curr.problem_num,
      title: curr.title,
      status_label,
      code_block: codeBlock,
    });
  }

  return problems;
}

const PROBLEMS_MANIFEST_PATH = path.join(process.cwd(), 'src', 'config', 'problems.json');

export function getManifestProblems(
  topicId: string,
  tier: string
): Array<{ problem_num: number; title: string; label?: string }> | null {
  try {
    if (fs.existsSync(PROBLEMS_MANIFEST_PATH)) {
      const data = JSON.parse(fs.readFileSync(PROBLEMS_MANIFEST_PATH, 'utf-8'));
      if (data?.chapters?.[topicId]?.[tier]) {
        return data.chapters[topicId][tier];
      }
    }
  } catch {}
  return null;
}

/**
 * Scans an exercise file, registers all canonical problems in SQLite,
 * and initializes default statuses if not already present.
 */
export function registerCanonicalExerciseProblems(
  topicId: string,
  volumeId: string,
  tier: 'beginner' | 'intermediate' | 'champion',
  exerciseFilePath: string
): CanonicalProblem[] {
  const extracted = parseExerciseProblems(exerciseFilePath);
  const manifestDefs = getManifestProblems(topicId, tier);
  const result: CanonicalProblem[] = [];

  if (manifestDefs && manifestDefs.length > 0) {
    for (const m of manifestDefs) {
      const problem_id = `${topicId}_${tier}_p${m.problem_num}`;
      const ext = extracted.find((e) => e.problem_num === m.problem_num);
      const record: CanonicalProblem = {
        problem_id,
        topic_id: topicId,
        volume_id: volumeId,
        tier,
        problem_num: m.problem_num,
        title: m.title,
        status_label: m.label || ext?.status_label || m.title,
        stub_code: ext ? ext.code_block : '',
      };
      upsertCanonicalProblem(record);
      result.push(record);
    }
    return result;
  }

  for (const p of extracted) {
    const problem_id = `${topicId}_${tier}_p${p.problem_num}`;
    const record: CanonicalProblem = {
      problem_id,
      topic_id: topicId,
      volume_id: volumeId,
      tier,
      problem_num: p.problem_num,
      title: p.title,
      status_label: p.status_label,
      stub_code: p.code_block,
    };
    upsertCanonicalProblem(record);
    result.push(record);
  }

  return result;
}
