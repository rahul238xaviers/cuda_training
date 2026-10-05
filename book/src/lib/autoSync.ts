import fs from 'fs';
import path from 'path';
import { parseExerciseProblems, ExtractedProblem } from './problemParser';

export interface AutoSyncResult {
  merged: boolean;
  addedProblems: number[];
  backupPath?: string;
  solutionPath: string;
}

/**
 * Automatically detects new problems in exercise file that are missing
 * from the student's solution file, and injects the new problem stubs seamlessly.
 */
export function autoSyncWorkbookSolution(
  exerciseFilePath: string,
  solutionFilePath: string
): AutoSyncResult {
  if (!fs.existsSync(exerciseFilePath) || !fs.existsSync(solutionFilePath)) {
    return {
      merged: false,
      addedProblems: [],
      solutionPath: solutionFilePath,
    };
  }

  // 1. Get canonical problems from exercise
  const canonicalProblems = parseExerciseProblems(exerciseFilePath);
  if (canonicalProblems.length === 0) {
    return {
      merged: false,
      addedProblems: [],
      solutionPath: solutionFilePath,
    };
  }

  // 2. Read student's solution file and detect existing problem numbers
  const solutionContent = fs.readFileSync(solutionFilePath, 'utf-8');
  const solutionLines = solutionContent.split('\n');

  const headerRegex = /^\s*\/\/\s*(?:[-=*#]+\s*)?(?:PROBLEM|Problem)\s+(\d+)\s*[:\-—]/;
  const existingNums = new Set<number>();

  for (const line of solutionLines) {
    const match = line.match(headerRegex);
    if (match) {
      existingNums.add(parseInt(match[1], 10));
    }
  }

  // 3. Find missing problems
  const missingProblems = canonicalProblems.filter(
    (p) => !existingNums.has(p.problem_num)
  );

  if (missingProblems.length === 0) {
    return {
      merged: false,
      addedProblems: [],
      solutionPath: solutionFilePath,
    };
  }

  // 4. Create safe backup of existing solution file
  const backupPath = `${solutionFilePath}.bak`;
  try {
    fs.copyFileSync(solutionFilePath, backupPath);
  } catch (err) {
    console.error('Failed to create backup before auto-sync:', err);
  }

  // 5. Find injection point: right before the SCORECARD banner or return statement
  let injectionLineIdx = -1;
  for (let i = solutionLines.length - 1; i >= 0; i--) {
    const line = solutionLines[i];
    if (
      line.includes('--- SCORECARD ---') ||
      line.includes('SCORECARD') ||
      line.includes('return (passed == total)') ||
      line.includes('return passed == total') ||
      line.includes('return 0;')
    ) {
      injectionLineIdx = i;
      // Walk back past any decorative comments before scorecard
      while (injectionLineIdx > 0 && solutionLines[injectionLineIdx - 1].trim().startsWith('//')) {
        injectionLineIdx--;
      }
      break;
    }
  }

  // Fallback: If no scorecard found, inject before last closing brace
  if (injectionLineIdx === -1) {
    for (let i = solutionLines.length - 1; i >= 0; i--) {
      if (solutionLines[i].trim() === '}') {
        injectionLineIdx = i;
        break;
      }
    }
  }

  if (injectionLineIdx === -1) {
    injectionLineIdx = solutionLines.length;
  }

  // 6. Build the injection block
  const injectionBlocks: string[] = [
    '',
    '  // =========================================================================',
    '  // [AUTO-SYNC] NEW PROBLEM STUBS (Synchronized from updated exercise)',
    '  // =========================================================================',
    '',
  ];

  for (const p of missingProblems) {
    injectionBlocks.push(p.code_block);
    injectionBlocks.push('');
  }

  const beforeLines = solutionLines.slice(0, injectionLineIdx);
  const afterLines = solutionLines.slice(injectionLineIdx);

  const mergedContent = [...beforeLines, ...injectionBlocks, ...afterLines].join('\n');
  fs.writeFileSync(solutionFilePath, mergedContent, 'utf-8');

  return {
    merged: true,
    addedProblems: missingProblems.map((p) => p.problem_num),
    backupPath,
    solutionPath: solutionFilePath,
  };
}
