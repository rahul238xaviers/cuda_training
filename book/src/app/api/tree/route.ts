import { NextResponse } from 'next/server';
import { scanDynamicChapters, METAL_KERNELS } from '@/lib/workspace';

export async function GET() {
  const chapters = scanDynamicChapters();

  const totalChapters = chapters.length;
  const completedCount = chapters.filter((c) => c.completed).length;

  let totalTestsAcrossCourse = 0;
  let totalPassedTests = 0;

  for (const ch of chapters) {
    totalTestsAcrossCourse += ch.totalTests;
    totalPassedTests += ch.passedTests;
  }

  // Find first uncompleted chapter for bookmark
  const firstPending = chapters.find((c) => !c.completed);
  const currentBookmark = firstPending ? firstPending.id : '1.1';

  const percent =
    totalTestsAcrossCourse > 0
      ? Math.round((totalPassedTests / totalTestsAcrossCourse) * 100)
      : Math.round((completedCount / (totalChapters || 1)) * 100);

  return NextResponse.json({
    currentBookmark,
    stats: {
      chaptersDone: completedCount,
      totalChapters,
      percent,
      testsPassed: totalPassedTests,
      totalTests: totalTestsAcrossCourse,
    },
    volume1: {
      id: 'vol1',
      title: 'Volume 1: C++ Low-Level Systems Foundations',
      description: 'Pointer arithmetic, strides, memory alignment, cache lines, custom arenas, zero-copy buffers',
      chapters,
    },
    volume2: {
      id: 'vol2',
      title: 'Volume 2: CUDA Architecture & Execution Foundations',
      description: 'Warps, thread blocks, shared SRAM bank conflicts, register pressure, memory coalescing',
      modules: [
        { id: '2', title: 'Module 2: CUDA Execution Model (Warps, Blocks, Grids)', status: 'queued', testCount: 0 },
        { id: '3', title: 'Module 3: CUDA Memory Hierarchy (Global DRAM, Shared SRAM, Registers)', status: 'queued', testCount: 0 },
        { id: '4', title: 'Module 4: Parallel Primitives (Reductions, Scan, GEMM Tiling)', status: 'queued', testCount: 0 },
      ],
    },
    volume3: {
      id: 'vol3',
      title: 'Volume 3: The 22 Production LLM CUDA Kernels',
      description: 'Direct CUDA ports of the production Apple Silicon Metal kernels in dev/large-language-model',
      kernels: METAL_KERNELS,
    },
  });
}
