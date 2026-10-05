import { NextRequest, NextResponse } from 'next/server';
import { spawnSync } from 'child_process';
import { getClangFormatBin, WORKSPACE_ROOT } from '@/lib/workspace';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { code } = body;

    if (typeof code !== 'string') {
      return NextResponse.json({ success: false, error: 'Invalid code input' }, { status: 400 });
    }

    const clangFormat = getClangFormatBin();
    const envPath = `/opt/homebrew/bin:/opt/homebrew/opt/llvm/bin:/usr/local/bin:${process.env.PATH || ''}`;
    const formatResult = spawnSync(
      clangFormat,
      ['-style={BasedOnStyle: Google, IndentWidth: 4, ColumnLimit: 100, AccessModifierOffset: -4}'],
      {
        input: code,
        cwd: WORKSPACE_ROOT,
        encoding: 'utf-8',
        timeout: 5000,
        env: {
          ...process.env,
          PATH: envPath,
        },
      }
    );

    if (formatResult.status === 0 && typeof formatResult.stdout === 'string') {
      return NextResponse.json({
        success: true,
        formatted: formatResult.stdout,
      });
    }

    return NextResponse.json({
      success: false,
      error: formatResult.stderr || 'Clang-format failed',
      original: code,
    });
  } catch (err: any) {
    return NextResponse.json({
      success: false,
      error: err.message || 'Internal format error',
    }, { status: 500 });
  }
}
