import path from 'path';
import fs from 'fs';

export const WORKSPACE_ROOT = path.resolve(process.cwd(), '..');
export const SRC_DIR = path.join(WORKSPACE_ROOT, 'src');
export const MODULE1_DIR = path.join(SRC_DIR, 'module1');
export const PLAYGROUND_PATH = path.join(WORKSPACE_ROOT, 'playground.cpp');

export interface TierMeta {
  status: 'passed' | 'pending';
  tests: number;
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
 * Dynamically counts tests or problems defined in a C++ workbook.
 * Inspects both reportStatus(...) assertions and // PROBLEM X headers.
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
          // Clean title like "# Chapter 1.2: Strides & Pointer Indirection"
          let title = trimmed
            .replace(/^#\s*((Module\s+)?Chapter\s*[\d\.]+:?\s*|Module\s*[\d\.]+:?\s*)?/i, '')
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
 * Dynamically scans src/module1 on the filesystem and builds the curriculum tree.
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

    for (const tier of tiersList) {
      const solPath = path.join(folderPath, 'solution', `${tier}_workbook.cpp`);
      const exPath = path.join(folderPath, 'exercise', `${tier}_workbook.cpp`);

      const solExists = fs.existsSync(solPath);
      const exExists = fs.existsSync(exPath);
      let isSolved = false;

      if (solExists) {
        if (!exExists) {
          isSolved = fs.statSync(solPath).size > 200;
        } else {
          const solContent = fs.readFileSync(solPath, 'utf-8').trim();
          const exContent = fs.readFileSync(exPath, 'utf-8').trim();
          isSolved = solContent !== exContent && solContent.length > 200;
        }
      }

      const count = solExists ? countTestsInWorkbook(solPath) : countTestsInWorkbook(exPath);
      const status: 'passed' | 'pending' = isSolved ? 'passed' : 'pending';
      tiers[tier] = { status, tests: count };

      totalTests += count;
      if (status === 'passed') {
        passedTests += count;
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

export function getChapterFolder(chapterId: string): string | null {
  const chapters = scanDynamicChapters();
  const ch = chapters.find((c) => c.id === chapterId);
  return ch ? ch.folder : null;
}

export function getClangFormatBin(): string {
  const candidates = [
    '/Library/Developer/CommandLineTools/usr/bin/clang-format',
    '/opt/homebrew/Cellar/llvm/22.1.7_1/bin/clang-format',
    '/opt/homebrew/bin/clang-format',
    '/usr/local/bin/clang-format',
    '/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang-format',
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  try {
    const llvmPath = '/opt/homebrew/Cellar/llvm';
    if (fs.existsSync(llvmPath)) {
      const versions = fs.readdirSync(llvmPath);
      for (const v of versions) {
        const bin = path.join(llvmPath, v, 'bin/clang-format');
        if (fs.existsSync(bin)) return bin;
      }
    }
  } catch {}
  return 'clang-format';
}

export const METAL_KERNELS: MetalKernelMeta[] = [
  { id: 1, name: 'Embedding Forward', category: 'Embedding', metalFile: 'embedding_forward.metal', cudaFile: 'embedding_forward.cu', status: 'planned' },
  { id: 2, name: 'RMSNorm Forward', category: 'Normalization', metalFile: 'rms_norm_forward.metal', cudaFile: 'rms_norm_forward.cu', status: 'planned' },
  { id: 3, name: 'RMSNorm Backward', category: 'Normalization', metalFile: 'rms_norm_backward.metal', cudaFile: 'rms_norm_backward.cu', status: 'planned' },
  { id: 4, name: 'RoPE Forward', category: 'Positional', metalFile: 'rope_forward.metal', cudaFile: 'rope_forward.cu', status: 'planned' },
  { id: 5, name: 'RoPE Backward', category: 'Positional', metalFile: 'rope_backward.metal', cudaFile: 'rope_backward.cu', status: 'planned' },
  { id: 6, name: 'FlashAttention Forward', category: 'Attention', metalFile: 'flash_attn_fwd.metal', cudaFile: 'flash_attn_fwd.cu', status: 'planned' },
  { id: 7, name: 'Fused Attention Backward', category: 'Attention', metalFile: 'fused_attn_bwd.metal', cudaFile: 'fused_attn_bwd.cu', status: 'planned' },
  { id: 8, name: 'GEMM BF16', category: 'Matrix Multiplication', metalFile: 'gemm_bf16.metal', cudaFile: 'gemm_bf16.cu', status: 'planned' },
  { id: 9, name: 'GEMM Projection', category: 'Matrix Multiplication', metalFile: 'gemm_proj.metal', cudaFile: 'gemm_proj.cu', status: 'planned' },
  { id: 10, name: 'GEMM Projection Trans B', category: 'Matrix Multiplication', metalFile: 'gemm_proj_trans_b.metal', cudaFile: 'gemm_proj_trans_b.cu', status: 'planned' },
  { id: 11, name: 'GEMM GQA', category: 'Matrix Multiplication', metalFile: 'gemm_gqa.metal', cudaFile: 'gemm_gqa.cu', status: 'planned' },
  { id: 12, name: 'GEMM FFN', category: 'Matrix Multiplication', metalFile: 'gemm_ffn.metal', cudaFile: 'gemm_ffn.cu', status: 'planned' },
  { id: 13, name: 'GEMM Backward', category: 'Matrix Multiplication', metalFile: 'gemm_backward.metal', cudaFile: 'gemm_backward.cu', status: 'planned' },
  { id: 14, name: 'SwiGLU Forward', category: 'Activation', metalFile: 'swiglu_forward.metal', cudaFile: 'swiglu_forward.cu', status: 'planned' },
  { id: 15, name: 'SwiGLU Backward', category: 'Activation', metalFile: 'swiglu_backward.metal', cudaFile: 'swiglu_backward.cu', status: 'planned' },
  { id: 16, name: 'Fused SwiGLU GEMM', category: 'Fused Operator', metalFile: 'fused_swiglu_gemm.metal', cudaFile: 'fused_swiglu_gemm.cu', status: 'planned' },
  { id: 17, name: 'Residual Add', category: 'Residual', metalFile: 'residual_add.metal', cudaFile: 'residual_add.cu', status: 'planned' },
  { id: 18, name: 'Fused Add Norm', category: 'Fused Operator', metalFile: 'fused_add_norm.metal', cudaFile: 'fused_add_norm.cu', status: 'planned' },
  { id: 19, name: 'Fused Backward Add Norm', category: 'Fused Operator', metalFile: 'fused_backward_add_norm.metal', cudaFile: 'fused_backward_add_norm.cu', status: 'planned' },
  { id: 20, name: 'Cross Entropy', category: 'Loss', metalFile: 'cross_entropy.metal', cudaFile: 'cross_entropy.cu', status: 'planned' },
  { id: 21, name: 'Compute Loss', category: 'Loss', metalFile: 'compute_loss.metal', cudaFile: 'compute_loss.cu', status: 'planned' },
  { id: 22, name: 'AdamW Optimizer Step', category: 'Optimizer', metalFile: 'adamw_step.metal', cudaFile: 'adamw_step.cu', status: 'planned' },
];
