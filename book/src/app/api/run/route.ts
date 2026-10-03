import { NextRequest, NextResponse } from 'next/server';
import { spawnSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { WORKSPACE_ROOT, MODULE1_DIR, PLAYGROUND_PATH, getTopicLocation } from '@/lib/workspace';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { type, chapterId, tier, target = 'solution', code } = body;

    let sourceFile: string;
    let tempBin: string;
    let isCuda = false;
    let topicLocation: any = null;

    if (type === 'playground') {
      sourceFile = PLAYGROUND_PATH;
      if (typeof code === 'string') {
        fs.writeFileSync(sourceFile, code, 'utf-8');
      }
      tempBin = path.join(WORKSPACE_ROOT, `pg_bin_${Date.now()}`);
    } else if (type === 'workbook') {
      topicLocation = getTopicLocation(chapterId);
      if (!topicLocation) {
        return NextResponse.json(
          { success: false, stage: 'setup', stderr: `Chapter or topic ${chapterId} not found` },
          { status: 404 }
        );
      }
      const dir = path.join(topicLocation.fullPath, target);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      sourceFile = path.join(dir, `${tier}_workbook.${topicLocation.ext}`);
      if (typeof code === 'string') {
        fs.writeFileSync(sourceFile, code, 'utf-8');
      }
      tempBin = path.join(WORKSPACE_ROOT, `wb_bin_${Date.now()}`);
      isCuda = topicLocation.isCuda || topicLocation.ext === 'cu';
    } else {
      return NextResponse.json({ success: false, stage: 'setup', stderr: 'Unknown run type' }, { status: 400 });
    }

    const startTime = Date.now();

    // Check if nvcc is available for CUDA files
    if (isCuda) {
      const whichNvcc = spawnSync('which', ['nvcc'], { encoding: 'utf-8' });
      const hasNvcc = whichNvcc.status === 0;

      if (!hasNvcc) {
        // macOS / host without NVIDIA GPU & nvcc
        return NextResponse.json({
          success: true,
          stage: 'execution',
          exitCode: 0,
          stdout: `✅ [CUDA WORKBOOK SAVED]\nLocation: ${sourceFile}\n\nℹ️  Hardware Notice (macOS Host Environment):\nThe NVIDIA 'nvcc' compiler is not present on this local macOS machine.\nCUDA device code (.cu) requires an NVIDIA GPU (e.g. RTX, A100, H100).\n\n🚀 To run automated validation on your GPU instance:\n   ./src/compile_and_run_cuda.sh ${topicLocation.moduleFolder} ${tier}\n\nDirect nvcc command:\n   nvcc -O3 -std=c++17 --extended-lambda ${sourceFile} -o /tmp/bin && /tmp/bin\n`,
          stderr: '',
          durationMs: Date.now() - startTime,
        });
      }

      // NVCC available
      const compileResult = spawnSync(
        'nvcc',
        ['-O3', '-std=c++17', '--extended-lambda', sourceFile, '-o', tempBin],
        {
          cwd: WORKSPACE_ROOT,
          encoding: 'utf-8',
          timeout: 30000,
        }
      );

      if (compileResult.status !== 0) {
        if (fs.existsSync(tempBin)) fs.unlinkSync(tempBin);
        return NextResponse.json({
          success: false,
          stage: 'compilation',
          stdout: compileResult.stdout || '',
          stderr: compileResult.stderr || compileResult.error?.message || 'NVCC compilation failed',
          durationMs: Date.now() - startTime,
        });
      }
    } else {
      // 1. Compile C++ with clang++
      const compileResult = spawnSync('clang++', ['-std=c++20', '-O3', sourceFile, '-o', tempBin], {
        cwd: WORKSPACE_ROOT,
        encoding: 'utf-8',
        timeout: 15000,
      });

      if (compileResult.status !== 0) {
        if (fs.existsSync(tempBin)) fs.unlinkSync(tempBin);
        return NextResponse.json({
          success: false,
          stage: 'compilation',
          stdout: compileResult.stdout || '',
          stderr: compileResult.stderr || compileResult.error?.message || 'Compilation failed',
          durationMs: Date.now() - startTime,
        });
      }
    }

    // 2. Execute
    const execStartTime = Date.now();
    const runResult = spawnSync(tempBin, [], {
      cwd: WORKSPACE_ROOT,
      encoding: 'utf-8',
      timeout: 10000,
    });

    if (fs.existsSync(tempBin)) {
      try { fs.unlinkSync(tempBin); } catch {}
    }

    const durationMs = Date.now() - startTime;

    return NextResponse.json({
      success: runResult.status === 0,
      stage: 'execution',
      exitCode: runResult.status,
      stdout: runResult.stdout || '',
      stderr: runResult.stderr || (runResult.error ? runResult.error.message : ''),
      durationMs,
    });
  } catch (err: any) {
    return NextResponse.json({
      success: false,
      stage: 'internal',
      stderr: err.message || 'Internal server error during compilation',
    }, { status: 500 });
  }
}
