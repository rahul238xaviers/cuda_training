import { NextResponse } from 'next/server';
import { scanDynamicChapters, scanDynamicCudaModules, METAL_KERNELS } from '@/lib/workspace';

export async function GET() {
  const chapters = scanDynamicChapters();
  const cudaModules = scanDynamicCudaModules();

  const totalChapters = chapters.length;
  const completedCount = chapters.filter((c) => c.completed).length;

  let totalTestsAcrossCourse = 0;
  let totalPassedTests = 0;

  for (const ch of chapters) {
    totalTestsAcrossCourse += ch.totalTests;
    totalPassedTests += ch.passedTests;
  }

  let totalCudaTopics = 0;
  let completedCudaTopics = 0;
  let totalCudaTests = 0;
  let passedCudaTests = 0;

  for (const mod of cudaModules) {
    totalCudaTopics += mod.topics.length;
    completedCudaTopics += mod.topics.filter((t) => t.completed).length;
    totalCudaTests += mod.totalTests;
    passedCudaTests += mod.passedTests;
  }

  totalTestsAcrossCourse += totalCudaTests;
  totalPassedTests += passedCudaTests;

  // Find first uncompleted chapter for bookmark (prioritizing C++, then CUDA)
  const firstPendingCh = chapters.find((c) => !c.completed);
  let currentBookmark = firstPendingCh ? firstPendingCh.id : '1.1';
  if (!firstPendingCh) {
    for (const mod of cudaModules) {
      const pendingTopic = mod.topics.find((t) => !t.completed);
      if (pendingTopic) {
        currentBookmark = pendingTopic.id;
        break;
      }
    }
  }

  const percent =
    totalTestsAcrossCourse > 0
      ? Math.round((totalPassedTests / totalTestsAcrossCourse) * 100)
      : Math.round(((completedCount + completedCudaTopics) / ((totalChapters + totalCudaTopics) || 1)) * 100);

  return NextResponse.json({
    currentBookmark,
    stats: {
      chaptersDone: completedCount,
      totalChapters,
      cudaTopicsDone: completedCudaTopics,
      totalCudaTopics,
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
      title: 'Volume 2: CUDA Architecture & LLM Execution Model',
      description: 'Warps, thread blocks, shared SRAM bank conflicts, register pressure, memory coalescing, WMMA, and LLM inference/training primitives',
      modules: cudaModules,
    },
    volume3: {
      id: 'vol3',
      title: 'Volume 3: The 22 Production LLM CUDA Kernels',
      description: 'Direct CUDA ports of the production Apple Silicon Metal kernels in dev/large-language-model',
      kernels: METAL_KERNELS,
    },
  });
}

