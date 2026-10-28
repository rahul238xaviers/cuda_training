import { NextResponse } from 'next/server';
import {
  scanDynamicChapters,
  scanDynamicCppModules,
  scanDynamicCudaModules,
  scanDynamicKernelModules,
  scanPracticeKernels,
  METAL_KERNELS,
} from '@/lib/workspace';
import { loadCurriculumManifest } from '@/lib/curriculum';

export async function GET() {
  const chapters = scanDynamicChapters();
  const cppModules = scanDynamicCppModules();
  const cudaModules = scanDynamicCudaModules();
  const kernelModules = scanDynamicKernelModules();
  const practiceKernels = scanPracticeKernels();
  const manifest = loadCurriculumManifest();

  let totalTestsAcrossCourse = 0;
  let totalPassedTests = 0;

  // 1. C++ Stats (Volume 1)
  let totalChapters = chapters.length;
  let completedCount = 0;
  let vol1TotalTests = 0;
  let vol1PassedTests = 0;
  for (const ch of chapters) {
    vol1TotalTests += ch.totalTests;
    vol1PassedTests += ch.passedTests;
    if (ch.completed) completedCount++;
  }

  // 2. CUDA Stats (Volume 2)
  let totalCudaTopics = 0;
  let completedCudaTopics = 0;
  let vol2TotalTests = 0;
  let vol2PassedTests = 0;
  for (const mod of cudaModules) {
    totalCudaTopics += mod.topics.length;
    completedCudaTopics += mod.topics.filter((t) => t.completed).length;
    vol2TotalTests += mod.totalTests;
    vol2PassedTests += mod.passedTests;
  }

  // 3. Kernel Stats (Volume 3)
  let totalKernelTopics = 0;
  let completedKernelTopics = 0;
  let vol3TotalTests = 0;
  let vol3PassedTests = 0;
  for (const mod of kernelModules) {
    totalKernelTopics += mod.topics.length;
    completedKernelTopics += mod.topics.filter((t) => t.completed).length;
    vol3TotalTests += mod.totalTests;
    vol3PassedTests += mod.passedTests;
  }

  totalTestsAcrossCourse = vol1TotalTests + vol2TotalTests + vol3TotalTests;
  totalPassedTests = vol1PassedTests + vol2PassedTests + vol3PassedTests;

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
      vol1: {
        id: 'vol1',
        title: 'Volume 1: C++ Low-Level Systems Foundations',
        shortTitle: 'C++ Systems',
        chaptersDone: completedCount,
        totalChapters,
        testsPassed: vol1PassedTests,
        totalTests: vol1TotalTests,
        percent: vol1TotalTests > 0 ? Math.round((vol1PassedTests / vol1TotalTests) * 100) : 0,
      },
      vol2: {
        id: 'vol2',
        title: 'Volume 2: CUDA Architecture & LLM Execution Model',
        shortTitle: 'CUDA Systems',
        topicsDone: completedCudaTopics,
        totalTopics: totalCudaTopics,
        testsPassed: vol2PassedTests,
        totalTests: vol2TotalTests,
        percent: vol2TotalTests > 0 ? Math.round((vol2PassedTests / vol2TotalTests) * 100) : 0,
      },
      vol3: {
        id: 'vol3',
        title: 'Volume 3: Production GPU Kernels Curriculum',
        shortTitle: 'GPU Kernels',
        topicsDone: completedKernelTopics,
        totalTopics: totalKernelTopics,
        testsPassed: vol3PassedTests,
        totalTests: vol3TotalTests,
        percent: vol3TotalTests > 0 ? Math.round((vol3PassedTests / vol3TotalTests) * 100) : 0,
      },
    },
    volumes: manifest.volumes.map((v) => {
      const volStat = v.id === 'vol1'
        ? { completedItems: completedCount, totalItems: totalChapters, testsPassed: vol1PassedTests, totalTests: vol1TotalTests, percent: vol1TotalTests > 0 ? Math.round((vol1PassedTests / vol1TotalTests) * 100) : 0 }
        : v.id === 'vol2'
        ? { completedItems: completedCudaTopics, totalItems: totalCudaTopics, testsPassed: vol2PassedTests, totalTests: vol2TotalTests, percent: vol2TotalTests > 0 ? Math.round((vol2PassedTests / vol2TotalTests) * 100) : 0 }
        : { completedItems: completedKernelTopics, totalItems: totalKernelTopics, testsPassed: vol3PassedTests, totalTests: vol3TotalTests, percent: vol3TotalTests > 0 ? Math.round((vol3PassedTests / vol3TotalTests) * 100) : 0 };
      return {
        id: v.id,
        shortId: v.shortId,
        title: v.title,
        displayTitle: v.displayTitle,
        tagline: v.tagline,
        description: v.description,
        modules: v.id === 'vol1' ? cppModules : v.id === 'vol2' ? cudaModules : kernelModules,
        kernels: v.id === 'vol3' ? METAL_KERNELS : undefined,
        stats: volStat,
      };
    }),
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
    practiceKernels,
  });
}
