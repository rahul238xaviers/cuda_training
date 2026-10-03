import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import {
  WORKSPACE_ROOT,
  getTopicLocation,
  countTestsInWorkbook,
  getPlaygroundInfo,
  PlaygroundType,
  METAL_KERNELS,
} from '@/lib/workspace';

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const type = searchParams.get('type') || 'theory';
  const chapterId = searchParams.get('chapter') || '1.1';
  const tier = (searchParams.get('tier') || 'beginner') as 'beginner' | 'intermediate' | 'champion';
  const kernelId = searchParams.get('kernelId');
  const volumeId = searchParams.get('volumeId') || undefined;
  const playgroundType = (searchParams.get('playgroundType') || searchParams.get('subType') || 'cpp') as PlaygroundType;

  // 1. Multi-Playground Support (C++, CUDA, Kernel)
  if (type === 'playground') {
    const pgInfo = getPlaygroundInfo(playgroundType);
    let code = '';
    if (fs.existsSync(pgInfo.filePath)) {
      code = fs.readFileSync(pgInfo.filePath, 'utf-8');
    } else {
      if (pgInfo.type === 'cuda') {
        code = `// playground.cu - Interactive CUDA GPU Scratchpad\n#include <iostream>\n#include <cuda_runtime.h>\n\n__global__ void hello_cuda() {\n    printf("Hello from GPU Thread %d in Block %d!\\n", threadIdx.x, blockIdx.x);\n}\n\nint main() {\n    hello_cuda<<<2, 4>>>();\n    cudaDeviceSynchronize();\n    return 0;\n}\n`;
      } else if (pgInfo.type === 'kernel') {
        code = `// playground_kernel.cu - Kernel Benchmark & Verification Lab\n#include <iostream>\n#include <cuda_runtime.h>\n\nint main() {\n    std::cout << "Kernel benchmark harness ready." << std::endl;\n    return 0;\n}\n`;
      } else {
        code = `// playground.cpp - Interactive C++ Systems Playground\n#include <iostream>\n\nint main() {\n    std::cout << "Hello C++ Memory Systems!" << std::endl;\n    return 0;\n}\n`;
      }
      fs.writeFileSync(pgInfo.filePath, code, 'utf-8');
    }

    return NextResponse.json({
      type: 'playground',
      playgroundType: pgInfo.type,
      title: pgInfo.title,
      subtitle: pgInfo.subtitle,
      filePath: pgInfo.filename,
      language: pgInfo.language,
      ext: pgInfo.ext,
      code,
      relPath: pgInfo.filename,
    });
  }

  // 2. Legacy Metal Kernel blueprint specs (if specifically requested by numeric id)
  if (type === 'kernel' && kernelId && !chapterId.startsWith('k')) {
    const idNum = parseInt(kernelId || '1', 10);
    const kernelMeta = METAL_KERNELS.find((k) => k.id === idNum) || METAL_KERNELS[0];

    return NextResponse.json({
      type: 'kernel',
      title: `Kernel #${kernelMeta.id}: ${kernelMeta.name}`,
      kernel: kernelMeta,
    });
  }

  // 3. Topic Nodes (Volume 1: C++, Volume 2: CUDA, Volume 3: Kernels)
  const loc = getTopicLocation(chapterId, volumeId);
  if (!loc) {
    return NextResponse.json({ error: `Topic or Chapter ${chapterId} not found` }, { status: 404 });
  }

  const topicDir = loc.fullPath;
  const topicTitle = loc.title;
  const isCuda = loc.isCuda;
  const isKernel = loc.isKernel;
  const ext = loc.ext;
  const displayTopicId = loc.displayTopicId;

  const prefix = isKernel ? 'Kernel' : isCuda ? 'Topic' : 'Chapter';

  if (type === 'theory') {
    const theoryPath = path.join(topicDir, 'theory.md');
    const content = fs.existsSync(theoryPath)
      ? fs.readFileSync(theoryPath, 'utf-8')
      : `# ${prefix} ${displayTopicId}: ${topicTitle}\n\n*Theory module coming soon.*`;
    return NextResponse.json({
      type: 'theory',
      chapterId,
      displayId: displayTopicId,
      chapterTitle: topicTitle,
      title: `${prefix} ${displayTopicId}: Theory & Mental Models`,
      content,
      readTimeMin: isKernel ? 9 : 7,
      isCuda,
      isKernel,
      relPath: path.relative(WORKSPACE_ROOT, theoryPath),
    });
  }

  if (type === 'cheat_sheet') {
    const cheatPath = path.join(topicDir, 'cheat_sheet.md');
    const content = fs.existsSync(cheatPath)
      ? fs.readFileSync(cheatPath, 'utf-8')
      : `# ${prefix} ${displayTopicId}: Revision Cheat Sheet\n\n*Cheat sheet coming soon.*`;
    return NextResponse.json({
      type: 'cheat_sheet',
      chapterId,
      displayId: displayTopicId,
      chapterTitle: topicTitle,
      title: `${prefix} ${displayTopicId}: Revision Cheat Sheet`,
      content,
      isCuda,
      isKernel,
      relPath: path.relative(WORKSPACE_ROOT, cheatPath),
    });
  }

  if (type === 'workbook') {
    const solDir = path.join(topicDir, 'solution');
    const solCu = path.join(solDir, `${tier}_workbook.cu`);
    const solCpp = path.join(solDir, `${tier}_workbook.cpp`);
    const exCu = path.join(topicDir, 'exercise', `${tier}_workbook.cu`);
    const exCpp = path.join(topicDir, 'exercise', `${tier}_workbook.cpp`);

    const exPath = fs.existsSync(exCu) ? exCu : exCpp;
    const solPath = fs.existsSync(solCu) ? solCu : solCpp;

    const actualExt = fs.existsSync(exCu) || fs.existsSync(solCu) ? 'cu' : 'cpp';
    const targetSolPath = path.join(solDir, `${tier}_workbook.${actualExt}`);

    const exerciseCode = fs.existsSync(exPath) ? fs.readFileSync(exPath, 'utf-8') : '';

    // If solution file does NOT exist yet or is empty, auto-initialize from exercise
    if ((!fs.existsSync(solPath) || fs.statSync(solPath).size === 0) && exerciseCode) {
      if (!fs.existsSync(solDir)) {
        fs.mkdirSync(solDir, { recursive: true });
      }
      fs.writeFileSync(targetSolPath, exerciseCode, 'utf-8');
    }

    const solutionCode = fs.existsSync(solPath) && fs.statSync(solPath).size > 0
      ? fs.readFileSync(solPath, 'utf-8')
      : exerciseCode;

    // Count tests
    const testsCount = fs.existsSync(solPath)
      ? countTestsInWorkbook(solPath)
      : fs.existsSync(exPath)
      ? countTestsInWorkbook(exPath)
      : 0;

    let isSolved = false;
    if (fs.existsSync(solPath) && fs.statSync(solPath).size > 0) {
      if (!fs.existsSync(exPath)) {
        isSolved = fs.statSync(solPath).size > 200;
      } else {
        const s = fs.readFileSync(solPath, 'utf-8').trim();
        const e = fs.readFileSync(exPath, 'utf-8').trim();
        isSolved = s !== e && s.length > 200;
      }
    }

    const chosenPath = fs.existsSync(solPath) ? solPath : targetSolPath;

    return NextResponse.json({
      type: 'workbook',
      chapterId,
      displayId: displayTopicId,
      chapterTitle: topicTitle,
      tier,
      ext: actualExt,
      isCuda,
      isKernel,
      title: `${prefix} ${displayTopicId}: ${tier.charAt(0).toUpperCase() + tier.slice(1)} Workbook`,
      solutionCode,
      exerciseCode,
      defaultTarget: 'solution',
      testsCount,
      status: isSolved ? 'passed' : 'pending',
      relPath: path.relative(WORKSPACE_ROOT, chosenPath),
    });
  }

  return NextResponse.json({ error: 'Invalid node type' }, { status: 400 });
}
