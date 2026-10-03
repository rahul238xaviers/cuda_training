import { NextResponse } from 'next/server';
import {
  scanDynamicChapters,
  scanDynamicCppModules,
  scanDynamicCudaModules,
  scanDynamicKernelModules,
  METAL_KERNELS,
} from '@/lib/workspace';
import { loadCurriculumManifest } from '@/lib/curriculum';

export async function GET() {
  const chapters = scanDynamicChapters();
  const cppModules = scanDynamicCppModules();
  const cudaModules = scanDynamicCudaModules();
  const kernelModules = scanDynamicKernelModules();
  const manifest = loadCurriculumManifest();

  let totalTestsAcrossCourse = 0;
  let totalPassedTests = 0;

  // 1. C++ Stats
  let totalChapters = chapters.length;
  let completedCount = 0;
  for (const ch of chapters) {
    totalTestsAcrossCourse += ch.totalTests;
    totalPassedTests += ch.passedTests;
    if (ch.completed) completedCount++;
  }

  // 2. CUDA Stats
  let totalCudaTopics = 0;
  let completedCudaTopics = 0;
  for (const mod of cudaModules) {
    totalCudaTopics += mod.topics.length;
    completedCudaTopics += mod.topics.filter((t) => t.completed).length;
    totalTestsAcrossCourse += mod.totalTests;
    totalPassedTests += mod.passedTests;
  }

  // 3. Kernel Stats
  let totalKernelTopics = 0;
  let completedKernelTopics = 0;
  for (const mod of kernelModules) {
    totalKernelTopics += mod.topics.length;
    completedKernelTopics += mod.topics.filter((t) => t.completed).length;
    totalTestsAcrossCourse += mod.totalTests;
    totalPassedTests += mod.passedTests;
  }

  // Find first uncompleted chapter for bookmark (prioritizing C++, then CUDA, then Kernels)
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

  const allTopicsCount = totalChapters + totalCudaTopics + totalKernelTopics;
  const allCompletedTopics = completedCount + completedCudaTopics + completedKernelTopics;
  const percent =
    totalTestsAcrossCourse > 0
      ? Math.round((totalPassedTests / totalTestsAcrossCourse) * 100)
      : Math.round((allCompletedTopics / (allTopicsCount || 1)) * 100);

  const vol1Meta = manifest.volumes.find((v) => v.id === 'vol1');
  const vol2Meta = manifest.volumes.find((v) => v.id === 'vol2');
  const vol3Meta = manifest.volumes.find((v) => v.id === 'vol3');

  return NextResponse.json({
    currentBookmark,
    manifestMeta: {
      title: manifest.title,
      subtitle: manifest.subtitle,
      version: manifest.version,
    },
    stats: {
      chaptersDone: completedCount,
      totalChapters,
      cudaTopicsDone: completedCudaTopics,
      totalCudaTopics,
      kernelTopicsDone: completedKernelTopics,
      totalKernelTopics,
      percent,
      testsPassed: totalPassedTests,
      totalTests: totalTestsAcrossCourse,
    },
    volumes: manifest.volumes.map((v) => ({
      id: v.id,
      shortId: v.shortId,
      title: v.title,
      displayTitle: v.displayTitle,
      tagline: v.tagline,
      description: v.description,
      modules: v.id === 'vol1' ? cppModules : v.id === 'vol2' ? cudaModules : kernelModules,
      kernels: v.id === 'vol3' ? METAL_KERNELS : undefined,
    })),
    volume1: {
      id: 'vol1',
      title: vol1Meta?.title || 'Volume 1: C++ Low-Level Systems Foundations',
      description: vol1Meta?.description || 'Pointer arithmetic, strides, memory alignment, cache lines, custom arenas, zero-copy buffers',
      modules: cppModules,
      chapters,
    },
    volume2: {
      id: 'vol2',
      title: vol2Meta?.title || 'Volume 2: CUDA Architecture & LLM Execution Model',
      description: vol2Meta?.description || 'Warps, thread blocks, shared SRAM bank conflicts, register pressure, memory coalescing, WMMA, and LLM inference/training primitives',
      modules: cudaModules,
    },
    volume3: {
      id: 'vol3',
      title: vol3Meta?.title || 'Volume 3: Production GPU Kernels Curriculum',
      description: vol3Meta?.description || 'Dense GEMM, Sparse/MoE, Attention, Normalizations, Audio, Vision, and Quantization/Optimizers',
      modules: kernelModules,
      kernels: METAL_KERNELS,
    },
  });
}
