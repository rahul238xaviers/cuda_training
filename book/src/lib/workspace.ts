import path from 'path';
import fs from 'fs';
import { spawnSync } from 'child_process';
import { loadUserProgress } from './progress';
import { loadCurriculumManifest } from './curriculum';
import { getCanonicalProblems, getUserProblemsWithStatus } from './db';
import { registerCanonicalExerciseProblems } from './problemParser';
import { autoSyncWorkbookSolution } from './autoSync';

export const WORKSPACE_ROOT = path.resolve(process.cwd(), '..');
export const SRC_DIR = path.join(WORKSPACE_ROOT, 'src');
export const MODULE1_DIR = path.join(SRC_DIR, 'module1');
export const KERNELS_DIR = path.join(SRC_DIR, 'kernels');
export const SANDBOX_DIR = path.join(WORKSPACE_ROOT, 'sandbox', 'runs');

export const PLAYGROUND_CPP_PATH = path.join(WORKSPACE_ROOT, 'playground.cpp');
export const PLAYGROUND_CUDA_PATH = path.join(WORKSPACE_ROOT, 'playground.cu');
export const PLAYGROUND_KERNEL_PATH = path.join(WORKSPACE_ROOT, 'playground_kernel.cu');
export const PLAYGROUND_PATH = PLAYGROUND_CPP_PATH;

export type PlaygroundType = 'cpp' | 'cuda' | 'kernel';

export interface PlaygroundInfo {
  type: PlaygroundType;
  filename: string;
  filePath: string;
  title: string;
  subtitle: string;
  language: 'cpp' | 'cuda';
  ext: 'cpp' | 'cu';
}

export function getPlaygroundInfo(type: PlaygroundType = 'cpp'): PlaygroundInfo {
  if (type === 'cuda') {
    return {
      type: 'cuda',
      filename: 'playground.cu',
      filePath: PLAYGROUND_CUDA_PATH,
      title: 'CUDA GPU Scratchpad',
      subtitle: 'Hardware warp primitives & GPU device kernels',
      language: 'cuda',
      ext: 'cu',
    };
  }
  if (type === 'kernel') {
    return {
      type: 'kernel',
      filename: 'playground_kernel.cu',
      filePath: PLAYGROUND_KERNEL_PATH,
      title: 'Kernel Benchmark Lab',
      subtitle: 'GPU kernel verification & roofline profiling sandbox',
      language: 'cuda',
      ext: 'cu',
    };
  }
  return {
    type: 'cpp',
    filename: 'playground.cpp',
    filePath: PLAYGROUND_CPP_PATH,
    title: 'C++ Systems Playground',
    subtitle: 'Zero-cost abstractions & low-level memory playground',
    language: 'cpp',
    ext: 'cpp',
  };
}

export interface TierMeta {
  status: 'passed' | 'pending';
  tests: number;
  testsPassed?: number;
  newProblemsCount?: number;
}

export interface ChapterMeta {
  id: string;
  folder: string;
  title: string;
  completed: boolean;
  hasCheatSheet: boolean;
  totalTests: number;
  passedTests: number;
  tiers: {
    beginner: TierMeta;
    intermediate: TierMeta;
    champion: TierMeta;
  };
}

export interface MetalKernelMeta {
  id: number;
  name: string;
  category: string;
  metalFile: string;
  cudaFile: string;
  status: 'planned' | 'in_progress' | 'completed';
}

/**
 * Dynamically counts tests or problems defined in a C++ or CUDA workbook.
 * Inspects both reportStatus(...) assertions, // PROBLEM X headers, and evaluation criteria.
 */
export function countTestsInWorkbook(filePath: string): number {
  if (!fs.existsSync(filePath)) return 0;
  try {
    const content = fs.readFileSync(filePath, 'utf-8');

    // Count problem sections first
    const problemMatches = content.match(/\/\/\s*PROBLEM\s+\d+:/gi);
    if (problemMatches && problemMatches.length > 0) {
      return problemMatches.length;
    }

    // Fallback to reportStatus calls
    const reportMatches = content.match(/reportStatus\s*\(/g);
    if (reportMatches && reportMatches.length > 0) {
      return reportMatches.length;
    }

    // Kernel evaluation criteria or assertions
    const verifyMatches = content.match(/assert\(|std::abs|max_diff|CUDA_CHECK/g);
    if (verifyMatches && verifyMatches.length > 0) {
      return Math.min(verifyMatches.length, 5);
    }

    // Default to at least 1 if file is non-empty
    return content.trim().length > 100 ? 3 : 0;
  } catch {
    return 0;
  }
}

/**
 * Extracts a human-readable title from theory.md or folder slug
 */
function extractChapterTitle(folderPath: string, folderName: string): string {
  const theoryPath = path.join(folderPath, 'theory.md');
  if (fs.existsSync(theoryPath)) {
    try {
      const firstLines = fs.readFileSync(theoryPath, 'utf-8').split('\n').slice(0, 10);
      for (const line of firstLines) {
        const trimmed = line.trim();
        if (trimmed.startsWith('# ')) {
          let title = trimmed
            .replace(/^#\s*((Module\s+)?(Chapter|Stage)\s*[\d\.]+:?\s*|Module\s*[\d\.]+\s*[-—:]\s*|[\d\.]+\s*(Stage\s*\d+:?\s*)?[-—:]?\s*|[-—]\s*)?/i, '')
            .replace(/`([^`]+)`/g, '$1')
            .trim();
          if (title) return title;
        }
      }
    } catch {}
  }

  // Fallback to formatting folder name: "1.2_strides_indirection" -> "Strides & Indirection"
  const slug = folderName.replace(/^[\d\.]+_+/, '');
  return slug
    .split('_')
    .map((w) => (w === 'and' ? '&' : w.charAt(0).toUpperCase() + w.slice(1)))
    .join(' ');
}

/**
 * Dynamically scans src/module1 on the filesystem and builds the flat C++ chapter list.
 */
export function scanDynamicChapters(): ChapterMeta[] {
  if (!fs.existsSync(MODULE1_DIR)) {
    return [];
  }

  const entries = fs.readdirSync(MODULE1_DIR, { withFileTypes: true });
  const chapterFolders = entries.filter(
    (e) => e.isDirectory() && /^1\.\d+(_.*)?$/.test(e.name)
  );

  // Sort naturally by chapter number: 1.1, 1.2, ..., 1.10, ..., 1.20
  chapterFolders.sort((a, b) => {
    const numA = parseFloat(a.name.match(/^1\.(\d+)/)?.[1] || '0');
    const numB = parseFloat(b.name.match(/^1\.(\d+)/)?.[1] || '0');
    return numA - numB;
  });

  const chapters: ChapterMeta[] = chapterFolders.map((entry) => {
    const folder = entry.name;
    const folderPath = path.join(MODULE1_DIR, folder);
    const idMatch = folder.match(/^1\.(\d+)/);
    const id = idMatch ? `1.${idMatch[1]}` : folder;
    const title = extractChapterTitle(folderPath, folder);

    const hasCheatSheet = fs.existsSync(path.join(folderPath, 'cheat_sheet.md'));

    const tiersList: Array<'beginner' | 'intermediate' | 'champion'> = [
      'beginner',
      'intermediate',
      'champion',
    ];

    const tiers: Record<'beginner' | 'intermediate' | 'champion', TierMeta> = {
      beginner: { status: 'pending', tests: 0 },
      intermediate: { status: 'pending', tests: 0 },
      champion: { status: 'pending', tests: 0 },
    };

    let totalTests = 0;
    let passedTests = 0;
    let passedTiersCount = 0;

    const userProgress = loadUserProgress();
    const chProgress = userProgress.topics[id];

    for (const tier of tiersList) {
      const solPath = path.join(folderPath, 'solution', `${tier}_workbook.cpp`);
      const exPath = path.join(folderPath, 'exercise', `${tier}_workbook.cpp`);

      const solExists = fs.existsSync(solPath);
      const exExists = fs.existsSync(exPath);

      if (exExists) {
        registerCanonicalExerciseProblems(id, 'vol1', tier, exPath);
        if (solExists) {
          autoSyncWorkbookSolution(exPath, solPath);
        }
      }

      const canonicalProblems = getCanonicalProblems(id, tier);
      const count = canonicalProblems.length > 0
        ? canonicalProblems.length
        : exExists
        ? countTestsInWorkbook(exPath)
        : solExists
        ? countTestsInWorkbook(solPath)
        : 3;

      const userProblems = getUserProblemsWithStatus(id, tier);
      let passedCount = userProblems.filter((p) => p.status === 'passed').length;

      // Migration fallback from user_progress.json
      if (userProblems.length > 0 && passedCount === 0 && chProgress?.tiers?.[tier]?.status === 'passed') {
        const legacyPassed = chProgress?.tiers?.[tier]?.testsPassed || 3;
        passedCount = Math.min(legacyPassed, count);
      } else if (userProblems.length === 0 && chProgress?.tiers?.[tier]?.status === 'passed') {
        passedCount = Math.min(chProgress?.tiers?.[tier]?.testsPassed || 3, count);
      }

      const isSolved = passedCount === count && count > 0;
      const status: 'passed' | 'pending' = isSolved ? 'passed' : 'pending';
      const newProblemsCount = Math.max(0, count - passedCount);

      tiers[tier] = {
        status,
        tests: count,
        testsPassed: passedCount,
        newProblemsCount,
      };

      totalTests += count;
      passedTests += passedCount;
      if (status === 'passed') {
        passedTiersCount++;
      }
    }

    const completed = passedTiersCount === 3;

    return {
      id,
      folder,
      title,
      completed,
      hasCheatSheet,
      totalTests,
      passedTests,
      tiers,
    };
  });

  return chapters;
}

// =============================================================================
// Volume 1: C++ Systems 5-Module Structure
// =============================================================================

export interface CppModuleMeta {
  id: string;
  displayId: string;
  displayNum: number;
  title: string;
  displayTitle: string;
  tagline: string;
  topics: ChapterMeta[];
  totalTests: number;
  passedTests: number;
  completed: boolean;
}

export function getCppModuleInfo(): Record<number, { title: string; displayTitle: string; tagline: string; startCh: number; endCh: number }> {
  const manifest = loadCurriculumManifest();
  const vol1 = manifest.volumes.find((v) => v.id === 'vol1');
  const record: Record<number, { title: string; displayTitle: string; tagline: string; startCh: number; endCh: number }> = {};
  if (vol1) {
    for (const m of vol1.modules) {
      record[m.num] = {
        title: m.title,
        displayTitle: m.displayTitle,
        tagline: m.tagline,
        startCh: m.startCh || 1,
        endCh: m.endCh || 20,
      };
    }
  }
  return record;
}

export const CPP_MODULE_INFO = getCppModuleInfo();

export function scanDynamicCppModules(): CppModuleMeta[] {
  const allChapters = scanDynamicChapters();
  const manifest = loadCurriculumManifest();
  const vol1 = manifest.volumes.find((v) => v.id === 'vol1');
  const moduleDefs = vol1?.modules || [];
  const modules: CppModuleMeta[] = [];

  for (const mDef of moduleDefs) {
    const m = mDef.num;
    const startCh = mDef.startCh || 1;
    const endCh = mDef.endCh || 20;

    const modTopics = allChapters.filter((ch) => {
      const match = ch.id.match(/^1\.(\d+)/);
      if (!match) return false;
      const num = parseInt(match[1], 10);
      return num >= startCh && num <= endCh;
    });

    let modTotalTests = 0;
    let modPassedTests = 0;
    let completedCount = 0;

    for (const t of modTopics) {
      modTotalTests += t.totalTests;
      modPassedTests += t.passedTests;
      if (t.completed) completedCount++;
    }

    modules.push({
      id: String(m),
      displayId: String(m),
      displayNum: m,
      title: mDef.title,
      displayTitle: mDef.displayTitle,
      tagline: mDef.tagline,
      topics: modTopics,
      totalTests: modTotalTests,
      passedTests: modPassedTests,
      completed: completedCount === modTopics.length && modTopics.length > 0,
    });
  }

  return modules;
}

// =============================================================================
// Volume 2: CUDA Progressive Curriculum Structure
// =============================================================================

export interface CudaTopicMeta extends ChapterMeta {
  moduleId: string;
  moduleNum: number;
  moduleFolder: string;
  ext: 'cu' | 'cpp';
  displayId: string;
  displayNum: number;
}

export interface CudaModuleMeta {
  id: string;
  displayId: string;
  displayNum: number;
  folder: string;
  title: string;
  displayTitle: string;
  tagline: string;
  topics: CudaTopicMeta[];
  totalTests: number;
  passedTests: number;
  completed: boolean;
}

export function getCudaModuleInfo(): Record<number, { title: string; displayTitle: string; tagline: string }> {
  const manifest = loadCurriculumManifest();
  const vol2 = manifest.volumes.find((v) => v.id === 'vol2');
  const record: Record<number, { title: string; displayTitle: string; tagline: string }> = {};
  if (vol2) {
    for (const m of vol2.modules) {
      const diskNum = m.diskNum || (m.num + 1);
      record[diskNum] = {
        title: m.title,
        displayTitle: m.displayTitle,
        tagline: m.tagline,
      };
    }
  }
  return record;
}

export const CUDA_MODULE_INFO = getCudaModuleInfo();

export function scanDynamicCudaModules(): CudaModuleMeta[] {
  const manifest = loadCurriculumManifest();
  const vol2 = manifest.volumes.find((v) => v.id === 'vol2');
  const moduleDefs = vol2?.modules || [];
  const modules: CudaModuleMeta[] = [];

  for (const mDef of moduleDefs) {
    const m = mDef.diskNum || mDef.num;
    const displayNum = mDef.num;
    const displayId = String(displayNum);
    const modFolder = `module${m}`;
    const modDir = path.join(SRC_DIR, modFolder);
    if (!fs.existsSync(modDir)) continue;

    const entries = fs.readdirSync(modDir, { withFileTypes: true });
    const topicFolders = entries.filter(
      (e) => e.isDirectory() && new RegExp(`^${m}\\.\\d+(_.*)?$`).test(e.name)
    );

    // Sort naturally: 2.1, 2.2, 2.3, ...
    topicFolders.sort((a, b) => {
      const numA = parseFloat(a.name.match(new RegExp(`^${m}\\.(\\d+)`))?.[1] || '0');
      const numB = parseFloat(b.name.match(new RegExp(`^${m}\\.(\\d+)`))?.[1] || '0');
      return numA - numB;
    });

    const info = {
      title: mDef.title,
      displayTitle: mDef.displayTitle,
      tagline: mDef.tagline,
    };

    let modTotalTests = 0;
    let modPassedTests = 0;
    let completedTopicsCount = 0;

    const topics: CudaTopicMeta[] = topicFolders.map((entry) => {
      const folder = entry.name;
      const folderPath = path.join(modDir, folder);
      const idMatch = folder.match(new RegExp(`^${m}\\.(\\d+)`));
      const subIdx = idMatch ? idMatch[1] : '1';
      const id = idMatch ? `${m}.${subIdx}` : folder;
      const topicDisplayId = `${displayNum}.${subIdx}`;
      const title = extractChapterTitle(folderPath, folder);
      const hasCheatSheet = fs.existsSync(path.join(folderPath, 'cheat_sheet.md'));

      let ext: 'cu' | 'cpp' = 'cu';
      const exDir = path.join(folderPath, 'exercise');
      if (fs.existsSync(exDir)) {
        const files = fs.readdirSync(exDir);
        if (files.some((f) => f.endsWith('.cpp')) && !files.some((f) => f.endsWith('.cu'))) {
          ext = 'cpp';
        }
      }

      const tiersList: Array<'beginner' | 'intermediate' | 'champion'> = [
        'beginner',
        'intermediate',
        'champion',
      ];

      const tiers: Record<'beginner' | 'intermediate' | 'champion', TierMeta> = {
        beginner: { status: 'pending', tests: 0 },
        intermediate: { status: 'pending', tests: 0 },
        champion: { status: 'pending', tests: 0 },
      };

      let totalTests = 0;
      let passedTests = 0;
      let passedTiersCount = 0;

      for (const tier of tiersList) {
        const solCu = path.join(folderPath, 'solution', `${tier}_workbook.cu`);
        const solCpp = path.join(folderPath, 'solution', `${tier}_workbook.cpp`);
        const exCu = path.join(folderPath, 'exercise', `${tier}_workbook.cu`);
        const exCpp = path.join(folderPath, 'exercise', `${tier}_workbook.cpp`);

        const solPath = fs.existsSync(solCu) ? solCu : solCpp;
        const exPath = fs.existsSync(exCu) ? exCu : exCpp;

        const solExists = fs.existsSync(solPath);
        const exExists = fs.existsSync(exPath);

        const userProgress = loadUserProgress();
        const cudaProg = userProgress.topics[id] || userProgress.topics[topicDisplayId] || userProgress.topics[`cuda-${topicDisplayId}`];

        if (exExists) {
          registerCanonicalExerciseProblems(id, 'vol2', tier, exPath);
          if (solExists) {
            autoSyncWorkbookSolution(exPath, solPath);
          }
        }

        const canonicalProblems = getCanonicalProblems(id, tier);
        const count = canonicalProblems.length > 0
          ? canonicalProblems.length
          : exExists
          ? countTestsInWorkbook(exPath)
          : solExists
          ? countTestsInWorkbook(solPath)
          : 3;

        const userProblems = getUserProblemsWithStatus(id, tier);
        let passedCount = userProblems.filter((p) => p.status === 'passed').length;

        // Migration fallback from user_progress.json
        if (userProblems.length > 0 && passedCount === 0 && cudaProg?.tiers?.[tier]?.status === 'passed') {
          const legacyPassed = cudaProg?.tiers?.[tier]?.testsPassed || 3;
          passedCount = Math.min(legacyPassed, count);
        } else if (userProblems.length === 0 && cudaProg?.tiers?.[tier]?.status === 'passed') {
          passedCount = Math.min(cudaProg?.tiers?.[tier]?.testsPassed || 3, count);
        }

        const isSolved = passedCount === count && count > 0;
        const status: 'passed' | 'pending' = isSolved ? 'passed' : 'pending';
        const newProblemsCount = Math.max(0, count - passedCount);

        tiers[tier] = {
          status,
          tests: count,
          testsPassed: passedCount,
          newProblemsCount,
        };

        totalTests += count;
        passedTests += passedCount;
        if (status === 'passed') {
          passedTiersCount++;
        }
      }

      const completed = passedTiersCount === 3;
      if (completed) completedTopicsCount++;

      modTotalTests += totalTests;
      modPassedTests += passedTests;

      return {
        id,
        folder,
        moduleId: `module${m}`,
        moduleNum: m,
        moduleFolder: modFolder,
        title,
        completed,
        hasCheatSheet,
        totalTests,
        passedTests,
        ext,
        tiers,
        displayId: topicDisplayId,
        displayNum,
      };
    });

    modules.push({
      id: String(m),
      displayId,
      displayNum,
      folder: modFolder,
      title: info.title,
      displayTitle: info.displayTitle,
      tagline: info.tagline,
      topics,
      totalTests: modTotalTests,
      passedTests: modPassedTests,
      completed: completedTopicsCount === topics.length && topics.length > 0,
    });
  }

  return modules;
}

// =============================================================================
// Volume 3: Production Kernel Curriculum Structure (7 Modules)
// =============================================================================

export interface KernelTopicMeta extends ChapterMeta {
  moduleId: string;
  moduleNum: number;
  moduleFolder: string;
  ext: 'cu';
  displayId: string;
  displayNum: number;
  category: string;
}

export interface KernelModuleMeta {
  id: string;
  displayId: string;
  displayNum: number;
  folder: string;
  title: string;
  displayTitle: string;
  tagline: string;
  topics: KernelTopicMeta[];
  totalTests: number;
  passedTests: number;
  completed: boolean;
}

export function getKernelModuleInfo(): Record<number, { title: string; displayTitle: string; tagline: string; category: string }> {
  const manifest = loadCurriculumManifest();
  const vol3 = manifest.volumes.find((v) => v.id === 'vol3');
  const record: Record<number, { title: string; displayTitle: string; tagline: string; category: string }> = {};
  if (vol3) {
    for (const m of vol3.modules) {
      const diskNum = m.diskNum || m.num;
      record[diskNum] = {
        title: m.title,
        displayTitle: m.displayTitle,
        tagline: m.tagline,
        category: m.category || 'GPU Kernels',
      };
    }
  }
  return record;
}

export const KERNEL_MODULE_INFO = getKernelModuleInfo();

export function scanDynamicKernelModules(): KernelModuleMeta[] {
  const modules: KernelModuleMeta[] = [];
  if (!fs.existsSync(KERNELS_DIR)) return modules;

  const manifest = loadCurriculumManifest();
  const vol3 = manifest.volumes.find((v) => v.id === 'vol3');
  const moduleDefs = vol3?.modules || [];

  for (const mDef of moduleDefs) {
    const m = mDef.diskNum || mDef.num;
    const modFolder = `module${m}`;
    const modDir = path.join(KERNELS_DIR, modFolder);
    if (!fs.existsSync(modDir)) continue;

    const entries = fs.readdirSync(modDir, { withFileTypes: true });
    const topicFolders = entries.filter(
      (e) => e.isDirectory() && new RegExp(`^${m}\\.\\d+(_.*)?$`).test(e.name)
    );

    // Sort naturally: 1.1, 1.2, 1.3...
    topicFolders.sort((a, b) => {
      const numA = parseFloat(a.name.match(new RegExp(`^${m}\\.(\\d+)`))?.[1] || '0');
      const numB = parseFloat(b.name.match(new RegExp(`^${m}\\.(\\d+)`))?.[1] || '0');
      return numA - numB;
    });

    const info = {
      title: mDef.title,
      displayTitle: mDef.displayTitle,
      tagline: mDef.tagline,
      category: mDef.category || 'General Kernels',
    };

    let modTotalTests = 0;
    let modPassedTests = 0;
    let completedTopicsCount = 0;

    const topics: KernelTopicMeta[] = topicFolders.map((entry) => {
      const folder = entry.name;
      const folderPath = path.join(modDir, folder);
      const idMatch = folder.match(new RegExp(`^${m}\\.(\\d+)`));
      const subIdx = idMatch ? idMatch[1] : '1';
      const id = `k${m}.${subIdx}`;
      const topicDisplayId = `${m}.${subIdx}`;
      const title = extractChapterTitle(folderPath, folder);
      const hasCheatSheet = fs.existsSync(path.join(folderPath, 'cheat_sheet.md'));

      const tiersList: Array<'beginner' | 'intermediate' | 'champion'> = [
        'beginner',
        'intermediate',
        'champion',
      ];

      const tiers: Record<'beginner' | 'intermediate' | 'champion', TierMeta> = {
        beginner: { status: 'pending', tests: 0 },
        intermediate: { status: 'pending', tests: 0 },
        champion: { status: 'pending', tests: 0 },
      };

      let totalTests = 0;
      let passedTests = 0;
      let passedTiersCount = 0;

      for (const tier of tiersList) {
        const solCu = path.join(folderPath, 'solution', `${tier}_workbook.cu`);
        const exCu = path.join(folderPath, 'exercise', `${tier}_workbook.cu`);

        const solExists = fs.existsSync(solCu);
        const exExists = fs.existsSync(exCu);

        const userProgress = loadUserProgress();
        const kernelProg = userProgress.topics[id] || userProgress.topics[`k${m}.${subIdx}`];

        if (exExists) {
          registerCanonicalExerciseProblems(id, 'vol3', tier, exCu);
          if (solExists) {
            autoSyncWorkbookSolution(exCu, solCu);
          }
        }

        const canonicalProblems = getCanonicalProblems(id, tier);
        const count = canonicalProblems.length > 0
          ? canonicalProblems.length
          : exExists
          ? countTestsInWorkbook(exCu)
          : solExists
          ? countTestsInWorkbook(solCu)
          : 3;

        const userProblems = getUserProblemsWithStatus(id, tier);
        let passedCount = userProblems.filter((p) => p.status === 'passed').length;

        // Migration fallback from user_progress.json
        if (userProblems.length > 0 && passedCount === 0 && kernelProg?.tiers?.[tier]?.status === 'passed') {
          const legacyPassed = kernelProg?.tiers?.[tier]?.testsPassed || 3;
          passedCount = Math.min(legacyPassed, count);
        } else if (userProblems.length === 0 && kernelProg?.tiers?.[tier]?.status === 'passed') {
          passedCount = Math.min(kernelProg?.tiers?.[tier]?.testsPassed || 3, count);
        }

        const isSolved = passedCount === count && count > 0;
        const status: 'passed' | 'pending' = isSolved ? 'passed' : 'pending';
        const newProblemsCount = Math.max(0, count - passedCount);

        tiers[tier] = {
          status,
          tests: count,
          testsPassed: passedCount,
          newProblemsCount,
        };

        totalTests += count;
        passedTests += passedCount;
        if (status === 'passed') {
          passedTiersCount++;
        }
      }

      const completed = passedTiersCount === 3;
      if (completed) completedTopicsCount++;

      modTotalTests += totalTests;
      modPassedTests += passedTests;

      return {
        id,
        folder,
        moduleId: `module${m}`,
        moduleNum: m,
        moduleFolder: modFolder,
        title,
        completed,
        hasCheatSheet,
        totalTests,
        passedTests,
        ext: 'cu',
        tiers,
        displayId: topicDisplayId,
        displayNum: m,
        category: info.category,
      };
    });

    modules.push({
      id: String(m),
      displayId: String(m),
      displayNum: m,
      folder: modFolder,
      title: info.title,
      displayTitle: info.displayTitle,
      tagline: info.tagline,
      topics,
      totalTests: modTotalTests,
      passedTests: modPassedTests,
      completed: completedTopicsCount === topics.length && topics.length > 0,
    });
  }

  return modules;
}

// =============================================================================
// Universal Topic Location Resolver
// =============================================================================

export interface TopicLocation {
  moduleNum: number;
  moduleFolder: string;
  moduleDir: string;
  folder: string;
  fullPath: string;
  isCuda: boolean;
  isKernel: boolean;
  ext: 'cu' | 'cpp';
  title: string;
  displayModNum: number;
  displayTopicId: string;
}

export function getTopicLocation(id: string, volumeId?: string): TopicLocation | null {
  // Case A: Kernel Topic (Volume 3)
  const isKernelExplicit =
    id.startsWith('k') ||
    id.startsWith('K') ||
    id.startsWith('kernel-') ||
    volumeId === 'vol3';

  if (isKernelExplicit) {
    const cleanId = id.replace(/^(kernel-|k|K)/, '');
    const mMatch = cleanId.match(/^(\d+)\.(\d+)/);
    if (!mMatch) return null;

    const modNum = parseInt(mMatch[1], 10);
    const topicIndex = mMatch[2];
    const targetId = `${modNum}.${topicIndex}`;
    const modFolder = `module${modNum}`;
    const modDir = path.join(KERNELS_DIR, modFolder);
    if (!fs.existsSync(modDir)) return null;

    const entries = fs.readdirSync(modDir, { withFileTypes: true });
    const folderEntry = entries.find(
      (e) => e.isDirectory() && (e.name.startsWith(`${targetId}_`) || e.name === targetId)
    );
    if (!folderEntry) return null;

    const folder = folderEntry.name;
    const fullPath = path.join(modDir, folder);
    const title = extractChapterTitle(fullPath, folder);

    return {
      moduleNum: modNum,
      moduleFolder: modFolder,
      moduleDir: modDir,
      folder,
      fullPath,
      isCuda: true,
      isKernel: true,
      ext: 'cu',
      title,
      displayModNum: modNum,
      displayTopicId: `K${targetId}`,
    };
  }

  // Case B: CUDA Topic (Volume 2)
  if (id.startsWith('cuda-') || volumeId === 'vol2' || /^[2-7]\./.test(id)) {
    let modNum: number | null = null;
    let topicIndex: string | null = null;

    if (id.startsWith('cuda-')) {
      const raw = id.replace('cuda-', '');
      const m = raw.match(/^(\d+)\.(\d+)/);
      if (m) {
        modNum = parseInt(m[1], 10) + 1; // 1.x -> module 2
        topicIndex = m[2];
      }
    } else {
      const match = id.match(/^(\d+)\.(\d+)/);
      if (match) {
        const parsedMod = parseInt(match[1], 10);
        topicIndex = match[2];
        if (volumeId === 'vol2' && parsedMod >= 1 && parsedMod <= 6) {
          modNum = parsedMod + 1;
        } else {
          modNum = parsedMod;
        }
      }
    }

    if (modNum !== null && topicIndex !== null) {
      const targetId = `${modNum}.${topicIndex}`;
      const modFolder = `module${modNum}`;
      const modDir = path.join(SRC_DIR, modFolder);
      if (fs.existsSync(modDir)) {
        const entries = fs.readdirSync(modDir, { withFileTypes: true });
        const folderEntry = entries.find(
          (e) => e.isDirectory() && (e.name.startsWith(`${targetId}_`) || e.name === targetId)
        );
        if (folderEntry) {
          const folder = folderEntry.name;
          const fullPath = path.join(modDir, folder);
          let ext: 'cu' | 'cpp' = 'cu';
          const exDir = path.join(fullPath, 'exercise');
          if (fs.existsSync(exDir)) {
            const files = fs.readdirSync(exDir);
            if (files.some((f) => f.endsWith('.cpp')) && !files.some((f) => f.endsWith('.cu'))) {
              ext = 'cpp';
            }
          }
          const title = extractChapterTitle(fullPath, folder);
          const displayModNum = modNum - 1;
          const displayTopicId = `${displayModNum}.${topicIndex}`;

          return {
            moduleNum: modNum,
            moduleFolder: modFolder,
            moduleDir: modDir,
            folder,
            fullPath,
            isCuda: true,
            isKernel: false,
            ext,
            title,
            displayModNum,
            displayTopicId,
          };
        }
      }
    }
  }

  // Case C: Volume 1 C++ Systems (module1/1.x_...)
  const match = id.match(/^1\.(\d+)/);
  if (!match) return null;

  const topicIndex = match[1];
  const targetId = `1.${topicIndex}`;
  const modFolder = 'module1';
  const modDir = path.join(SRC_DIR, modFolder);
  if (!fs.existsSync(modDir)) return null;

  const entries = fs.readdirSync(modDir, { withFileTypes: true });
  const folderEntry = entries.find(
    (e) => e.isDirectory() && (e.name.startsWith(`${targetId}_`) || e.name === targetId)
  );
  if (!folderEntry) return null;

  const folder = folderEntry.name;
  const fullPath = path.join(modDir, folder);
  const title = extractChapterTitle(fullPath, folder);

  return {
    moduleNum: 1,
    moduleFolder: modFolder,
    moduleDir: modDir,
    folder,
    fullPath,
    isCuda: false,
    isKernel: false,
    ext: 'cpp',
    title,
    displayModNum: 1,
    displayTopicId: targetId,
  };
}

export function getChapterFolder(chapterId: string): string | null {
  const loc = getTopicLocation(chapterId);
  return loc ? loc.folder : null;
}

export function getClangFormatBin(): string {
  const envPath = `/opt/homebrew/bin:/opt/homebrew/opt/llvm/bin:/usr/local/bin:${process.env.PATH || ''}`;
  try {
    const whichRes = spawnSync('which', ['clang-format'], {
      encoding: 'utf-8',
      env: { ...process.env, PATH: envPath },
    });
    if (whichRes.status === 0 && whichRes.stdout.trim()) {
      return whichRes.stdout.trim();
    }
  } catch {}

  const standardCandidates = [
    '/opt/homebrew/opt/llvm/bin/clang-format',
    '/opt/homebrew/bin/clang-format',
    '/usr/local/bin/clang-format',
    '/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang-format',
    '/Library/Developer/CommandLineTools/usr/bin/clang-format',
  ];
  for (const c of standardCandidates) {
    if (fs.existsSync(c)) return c;
  }

  try {
    const llvmCellar = '/opt/homebrew/Cellar/llvm';
    if (fs.existsSync(llvmCellar)) {
      const versions = fs.readdirSync(llvmCellar);
      for (const v of versions) {
        const bin = path.join(llvmCellar, v, 'bin/clang-format');
        if (fs.existsSync(bin)) return bin;
      }
    }
  } catch {}

  return 'clang-format';
}

export const METAL_KERNELS: MetalKernelMeta[] = (loadCurriculumManifest().metalKernels || []).map((k) => ({
  id: k.id,
  name: k.name,
  category: k.category,
  metalFile: k.metalFile,
  cudaFile: k.cudaFile,
  status: k.status,
}));

