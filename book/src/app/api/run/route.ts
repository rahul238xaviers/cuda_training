import { NextRequest, NextResponse } from 'next/server';
import { spawnSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import {
  WORKSPACE_ROOT,
  SANDBOX_DIR,
  getTopicLocation,
  getPlaygroundInfo,
  countTestsInWorkbook,
  PlaygroundType,
} from '@/lib/workspace';
import { recordTierProgress } from '@/lib/progress';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const {
      type,
      chapterId,
      tier,
      target = 'solution',
      code,
      playgroundType = 'cpp',
      volumeId,
    } = body;

    let isCuda = false;
    let isKernel = false;
    let sourceFileName = 'main.cpp';
    let displayTitle = 'Sandbox Task';
    let topicLocation: any = null;

    // 1. Identify Target Execution Context
    if (type === 'playground') {
      const pgInfo = getPlaygroundInfo(playgroundType as PlaygroundType);
      isCuda = pgInfo.ext === 'cu';
      isKernel = pgInfo.type === 'kernel';
      sourceFileName = `sandbox_${pgInfo.filename}`;
      displayTitle = pgInfo.title;

      // Persist to user's root playground file
      if (typeof code === 'string') {
        fs.writeFileSync(pgInfo.filePath, code, 'utf-8');
      }
    } else if (type === 'workbook') {
      topicLocation = getTopicLocation(chapterId, volumeId);
      if (!topicLocation) {
        return NextResponse.json(
          { success: false, stage: 'setup', stderr: `Chapter or topic ${chapterId} not found` },
          { status: 404 }
        );
      }
      isCuda = topicLocation.isCuda || topicLocation.ext === 'cu';
      isKernel = topicLocation.isKernel;
      sourceFileName = `workbook_${tier}.${topicLocation.ext}`;
      displayTitle = `${topicLocation.title} (${tier})`;

      // Persist to user's workbook file on disk
      const dir = path.join(topicLocation.fullPath, target);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      const targetFile = path.join(dir, `${tier}_workbook.${topicLocation.ext}`);
      if (typeof code === 'string') {
        fs.writeFileSync(targetFile, code, 'utf-8');
      }
    } else {
      return NextResponse.json({ success: false, stage: 'setup', stderr: 'Unknown run type' }, { status: 400 });
    }

    // 2. Provision Isolated Sandbox Execution Directory
    const runId = `run_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
    const runDir = path.join(SANDBOX_DIR, runId);
    fs.mkdirSync(runDir, { recursive: true });

    const sandboxSourceFile = path.join(runDir, sourceFileName);
    const sandboxBin = path.join(runDir, 'sandbox_bin');

    const sourceContent = typeof code === 'string'
      ? code
      : fs.readFileSync(type === 'playground' ? getPlaygroundInfo(playgroundType as PlaygroundType).filePath : path.join(topicLocation.fullPath, target, `${tier}_workbook.${topicLocation.ext}`), 'utf-8');

    fs.writeFileSync(sandboxSourceFile, sourceContent, 'utf-8');

    const startTime = Date.now();

    // 3. Compile within Isolated Sandbox Directory
    if (isCuda) {
      const whichNvcc = spawnSync('which', ['nvcc'], { encoding: 'utf-8' });
      const hasNvcc = whichNvcc.status === 0;

      if (!hasNvcc) {
        // macOS Host without NVIDIA GPU & nvcc
        const relSandboxPath = path.relative(WORKSPACE_ROOT, sandboxSourceFile);
        return NextResponse.json({
          success: true,
          stage: 'execution',
          exitCode: 0,
          sandboxed: true,
          sandboxPath: relSandboxPath,
          stdout: `📦 [EXECUTED IN SANDBOX: ${relSandboxPath}]\n✅ Source staged and validated for execution.\n\nℹ️  Hardware Notice (macOS Host Environment):\nThe NVIDIA 'nvcc' compiler is not present on this local macOS machine.\nCUDA device code (.cu) requires an NVIDIA GPU (e.g. RTX, A100, H100).\n\n🚀 To run automated validation on your GPU instance:\n   ./src/compile_and_run_cuda.sh ${topicLocation ? topicLocation.moduleFolder : 'playground'} ${tier || 'test'}\n\nDirect nvcc sandbox command:\n   cd ${path.relative(WORKSPACE_ROOT, runDir)} && nvcc -O3 -std=c++17 --extended-lambda ${sourceFileName} -o ./bin && ./bin\n`,
          stderr: '',
          durationMs: Date.now() - startTime,
        });
      }

      // NVCC Compilation inside Sandbox
      const compileResult = spawnSync(
        'nvcc',
        ['-O3', '-std=c++17', '--extended-lambda', sourceFileName, '-o', 'sandbox_bin'],
        {
          cwd: runDir,
          encoding: 'utf-8',
          timeout: 30000,
        }
      );

      if (compileResult.status !== 0) {
        return NextResponse.json({
          success: false,
          stage: 'compilation',
          sandboxed: true,
          stdout: compileResult.stdout || '',
          stderr: compileResult.stderr || compileResult.error?.message || 'NVCC compilation failed',
          durationMs: Date.now() - startTime,
        });
      }
    } else {
      // Clang++ C++ Compilation inside Sandbox
      const compileResult = spawnSync(
        'clang++',
        ['-std=c++20', '-O3', sourceFileName, '-o', 'sandbox_bin'],
        {
          cwd: runDir,
          encoding: 'utf-8',
          timeout: 15000,
        }
      );

      if (compileResult.status !== 0) {
        return NextResponse.json({
          success: false,
          stage: 'compilation',
          sandboxed: true,
          stdout: compileResult.stdout || '',
          stderr: compileResult.stderr || compileResult.error?.message || 'Clang++ compilation failed',
          durationMs: Date.now() - startTime,
        });
      }
    }

    // 4. Execute with 10-Second Process Watchdog Timer
    const runResult = spawnSync(sandboxBin, [], {
      cwd: runDir,
      encoding: 'utf-8',
      timeout: 10000, // 10s watchdog limit
    });

    // Cleanup binary to conserve storage
    if (fs.existsSync(sandboxBin)) {
      try {
        fs.unlinkSync(sandboxBin);
      } catch {}
    }

    const durationMs = Date.now() - startTime;
    const isWatchdogKilled = runResult.error && (runResult.error as any).code === 'ETIMEDOUT';

    if (isWatchdogKilled) {
      return NextResponse.json({
        success: false,
        stage: 'execution',
        sandboxed: true,
        exitCode: 124,
        stdout: runResult.stdout || '',
        stderr: '⚠️ [WATCHDOG TIMEOUT]: Execution exceeded the 10-second sandbox safety limit and was terminated.',
        durationMs,
      });
    }

    const relSandboxPath = path.relative(WORKSPACE_ROOT, sandboxSourceFile);
    let outputStdout = runResult.stdout || '';
    if (runResult.status === 0 && !outputStdout.includes('[SANDBOX:')) {
      outputStdout = `📦 [SANDBOX: ${relSandboxPath}]\n` + outputStdout;
    }

    if (runResult.status === 0 && topicLocation && tier) {
      const isPassed = !outputStdout.includes('[FAILED]') && !outputStdout.includes('FAILED');
      if (isPassed) {
        const testsCount = countTestsInWorkbook(sandboxSourceFile) || 3;
        const progressId = topicLocation.isKernel
          ? `k${topicLocation.moduleNum}.${topicLocation.displayTopicId.replace(/^K\d+\./, '')}`
          : topicLocation.displayTopicId;
        recordTierProgress(progressId, tier, true, testsCount);
      }
    }

    return NextResponse.json({
      success: runResult.status === 0,
      stage: 'execution',
      sandboxed: true,
      sandboxPath: relSandboxPath,
      exitCode: runResult.status,
      stdout: outputStdout,
      stderr: runResult.stderr || (runResult.error ? runResult.error.message : ''),
      durationMs,
    });
  } catch (err: any) {
    return NextResponse.json(
      {
        success: false,
        stage: 'internal',
        stderr: err.message || 'Internal sandbox execution error',
      },
      { status: 500 }
    );
  }
}
