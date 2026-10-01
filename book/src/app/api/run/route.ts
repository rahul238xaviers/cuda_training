import { NextRequest, NextResponse } from 'next/server';
import { spawnSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { WORKSPACE_ROOT, MODULE1_DIR, PLAYGROUND_PATH, getChapterFolder } from '@/lib/workspace';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { type, chapterId, tier, target = 'solution', code } = body;

    let sourceFile: string;
    let tempBin: string;

    if (type === 'playground') {
      sourceFile = PLAYGROUND_PATH;
      if (typeof code === 'string') {
        fs.writeFileSync(sourceFile, code, 'utf-8');
      }
      tempBin = path.join(WORKSPACE_ROOT, `pg_bin_${Date.now()}`);
    } else if (type === 'workbook') {
      const folder = getChapterFolder(chapterId);
      if (!folder) {
        return NextResponse.json({ success: false, stage: 'setup', stderr: `Chapter ${chapterId} not found` }, { status: 404 });
      }
      const dir = path.join(MODULE1_DIR, folder, target);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      sourceFile = path.join(dir, `${tier}_workbook.cpp`);
      if (typeof code === 'string') {
        fs.writeFileSync(sourceFile, code, 'utf-8');
      }
      tempBin = path.join(WORKSPACE_ROOT, `wb_bin_${Date.now()}`);
    } else {
      return NextResponse.json({ success: false, stage: 'setup', stderr: 'Unknown run type' }, { status: 400 });
    }

    const startTime = Date.now();

    // 1. Compile
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
