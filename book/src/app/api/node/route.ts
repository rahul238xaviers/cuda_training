import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { MODULE1_DIR, PLAYGROUND_PATH, getChapterFolder, METAL_KERNELS, scanDynamicChapters } from '@/lib/workspace';

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const type = searchParams.get('type') || 'theory';
  const chapterId = searchParams.get('chapter') || '1.1';
  const tier = (searchParams.get('tier') || 'beginner') as 'beginner' | 'intermediate' | 'champion';
  const kernelId = searchParams.get('kernelId');

  if (type === 'playground') {
    const code = fs.existsSync(PLAYGROUND_PATH) ? fs.readFileSync(PLAYGROUND_PATH, 'utf-8') : '// playground.cpp\n#include <iostream>\n\nint main() {\n    std::cout << "Hello CUDA & C++!" << std::endl;\n    return 0;\n}\n';
    return NextResponse.json({
      type: 'playground',
      title: 'Interactive Playground Sandbox',
      filePath: 'playground.cpp',
      code,
    });
  }

  if (type === 'kernel') {
    const idNum = parseInt(kernelId || '1', 10);
    const kernelMeta = METAL_KERNELS.find((k) => k.id === idNum) || METAL_KERNELS[0];

    const kernelSpecs: Record<number, {
      overview: string;
      memoryPattern: string;
      threadLayout: string;
      optimizations: string;
      metalVsCuda: string;
    }> = {
      1: {
        overview: 'Maps vocabulary token IDs (int32) to dense embedding vectors of dimension d_model. Performs coalesced reads and writes across warp threads.',
        memoryPattern: 'Contiguous DRAM reads from embedding table, coalesced writes into activation tensor.',
        threadLayout: '1D Grid, 256 threads per block. Each warp handles 32 consecutive embedding dimensions.',
        optimizations: 'Vectorized float4 loads for hidden dimensions, cache hit optimization on active tokens.',
        metalVsCuda: 'Metal uses threadgroup dispatch and device pointers; CUDA uses grid-stride loop with __ldg() read-only cache loads.',
      },
      2: {
        overview: 'Root Mean Square Normalization. Computes root mean square over hidden dimension and scales elements without mean-centering.',
        memoryPattern: 'Two-pass reduction across hidden dimension: sum-of-squares reduction followed by scale multiply.',
        threadLayout: 'Block per token/row, 256-1024 threads with warp-shuffle reduction (__shfl_down_sync) in Shared Memory SRAM.',
        optimizations: 'Single-pass Welford/RMS with registers, eliminates redundant DRAM trips.',
        metalVsCuda: 'Metal uses simdgroup_matrix or simdgroup reductions; CUDA maps to __shfl_down_sync warp shuffle intrinsics.',
      },
      6: {
        overview: 'Tri-Dao style FlashAttention forward kernel using online softmax and SRAM tiling to eliminate NxN DRAM materialization.',
        memoryPattern: 'Tiled Q, K, V blocks loaded into Shared Memory SRAM; streaming online softmax keeps running max and sum in registers.',
        threadLayout: '2D/3D Grid over batch x heads x sequence chunks. Warps perform tiled matrix multiplications.',
        optimizations: 'Zero bank conflicts in shared SRAM, tensor core mma.sync or warp-level outer products.',
        metalVsCuda: 'Metal uses threadgroup memory buffers with simdgroup matrix ops; CUDA leverages cp.async and shared memory swizzling.',
      },
      8: {
        overview: 'High-performance Bfloat16 General Matrix Multiply (GEMM) using 2D block-tiling and double-buffering.',
        memoryPattern: 'A-tile and B-tile copied from Global DRAM to Shared Memory SRAM with ping-pong buffering.',
        threadLayout: '128x128 output tile per thread block, 64x64 warp tiles, 8x8 thread micro-tiles.',
        optimizations: 'Asynchronous copy (cp.async), zero bank conflict swizzling, register allocation tuning.',
        metalVsCuda: 'Metal dispatch_threadgroups maps to CUDA dim3 grid/block; shared memory tile allocation maps to __shared__ arrays.',
      },
      14: {
        overview: 'SwiGLU activation function: f(x) = (x * sigmoid(beta * x)) * gate. Widely used in modern LLMs (LLaMA, Mistral).',
        memoryPattern: 'Elementwise fused streaming read of gate and up-projection buffers, single write to activation buffer.',
        threadLayout: '1D Grid, 256 threads per block. Vectorized 128-bit loads (float4 / bfloat16_8).',
        optimizations: 'Fused math avoids intermediate memory write, fast silu approximation via hardware intrinsic.',
        metalVsCuda: 'Straightforward 1:1 translation with vectorized type matching (simd::float4 -> float4).',
      },
      22: {
        overview: 'Fused AdamW optimizer update applying first and second momentum corrections directly in GPU DRAM in a single pass.',
        memoryPattern: 'Streaming read/write of params, gradients, first moments (m), and second moments (v).',
        threadLayout: '1D Grid over all model parameters, vectorized reads/writes.',
        optimizations: 'Memory-bandwidth bound kernel; saturated through vectorized float4/half8 and coalesced 128-byte transactions.',
        metalVsCuda: 'Kernel is bandwidth bound; performance matches theoretical DRAM roofline when vectorized on both architectures.',
      },
    };

    const spec = kernelSpecs[kernelMeta.id] || {
      overview: `Production kernel implementation for ${kernelMeta.name} (${kernelMeta.category}).`,
      memoryPattern: 'Coalesced DRAM transactions, register caching, SRAM scratchpad.',
      threadLayout: 'Grid-stride loop with 256-512 threads per block.',
      optimizations: 'Vectorized memory access, ILP (instruction level parallelism), warp-synchronous intrinsics.',
      metalVsCuda: 'Porting involves translating metal threadgroup qualifiers to __shared__ and simdgroup primitives to warp shuffles.',
    };

    return NextResponse.json({
      type: 'kernel',
      title: `Kernel #${kernelMeta.id}: ${kernelMeta.name}`,
      kernel: kernelMeta,
      spec,
    });
  }

  // Chapter specific nodes
  const folder = getChapterFolder(chapterId);
  if (!folder) {
    return NextResponse.json({ error: `Chapter ${chapterId} not found` }, { status: 404 });
  }

  const chapterDir = path.join(MODULE1_DIR, folder);
  const chapters = scanDynamicChapters();
  const chMeta = chapters.find((c) => c.id === chapterId);

  if (type === 'theory') {
    const theoryPath = path.join(chapterDir, 'theory.md');
    const content = fs.existsSync(theoryPath)
      ? fs.readFileSync(theoryPath, 'utf-8')
      : `# Chapter ${chapterId}: ${chMeta?.title}\n\n*Theory module coming soon.*`;
    return NextResponse.json({
      type: 'theory',
      chapterId,
      chapterTitle: chMeta?.title,
      title: `Chapter ${chapterId}: Theory & Mental Models`,
      content,
      readTimeMin: 7,
    });
  }

  if (type === 'cheat_sheet') {
    const cheatPath = path.join(chapterDir, 'cheat_sheet.md');
    const content = fs.existsSync(cheatPath)
      ? fs.readFileSync(cheatPath, 'utf-8')
      : `# Chapter ${chapterId}: Revision Cheat Sheet\n\n*Cheat sheet coming soon.*`;
    return NextResponse.json({
      type: 'cheat_sheet',
      chapterId,
      chapterTitle: chMeta?.title,
      title: `Chapter ${chapterId}: Revision Cheat Sheet`,
      content,
    });
  }

  if (type === 'workbook') {
    const solPath = path.join(chapterDir, 'solution', `${tier}_workbook.cpp`);
    const exPath = path.join(chapterDir, 'exercise', `${tier}_workbook.cpp`);

    const solutionCode = fs.existsSync(solPath) ? fs.readFileSync(solPath, 'utf-8') : '';
    const exerciseCode = fs.existsSync(exPath) ? fs.readFileSync(exPath, 'utf-8') : '';

    return NextResponse.json({
      type: 'workbook',
      chapterId,
      chapterTitle: chMeta?.title,
      tier,
      title: `Chapter ${chapterId}: ${tier.charAt(0).toUpperCase() + tier.slice(1)} Workbook`,
      solutionCode,
      exerciseCode,
      defaultTarget: solutionCode ? 'solution' : 'exercise',
      testsCount: chMeta?.tiers[tier]?.tests || 0,
      status: chMeta?.tiers[tier]?.status || 'pending',
    });
  }

  return NextResponse.json({ error: 'Invalid node type' }, { status: 400 });
}
