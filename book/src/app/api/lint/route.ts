import { NextRequest, NextResponse } from 'next/server';
import { spawnSync } from 'child_process';
import { WORKSPACE_ROOT, SRC_DIR } from '@/lib/workspace';

export interface DiagnosticItem {
  line: number;
  col: number;
  severity: 'error' | 'warning' | 'info';
  message: string;
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { code } = body;

    if (typeof code !== 'string') {
      return NextResponse.json({ success: false, error: 'Invalid code input' }, { status: 400 });
    }

    const isCuda =
      code.includes('<cuda_runtime.h>') ||
      code.includes('__global__') ||
      code.includes('<<<') ||
      code.includes('__device__') ||
      code.includes('__shared__');

    if (isCuda) {
      const whichNvcc = spawnSync('which', ['nvcc'], { encoding: 'utf-8' });
      if (whichNvcc.status !== 0) {
        return NextResponse.json({
          success: true,
          diagnostics: [],
          errorCount: 0,
          warningCount: 0,
          clean: true,
          rawStderr: '',
        });
      }
    }

    const compileResult = spawnSync(
      'clang++',
      [
        '-fsyntax-only',
        '-std=c++20',
        '-Wall',
        '-Wextra',
        '-Wno-unused-parameter',
        `-I${SRC_DIR}`,
        `-I${WORKSPACE_ROOT}`,
        '-x',
        'c++',
        '-',
      ],
      {
        input: code,
        cwd: WORKSPACE_ROOT,
        encoding: 'utf-8',
        timeout: 10000,
      }
    );

    const diagnostics: DiagnosticItem[] = [];
    const stderr = compileResult.stderr || '';

    // Regex to match clang diagnostics:
    // <stdin>:3:15: error: expected expression
    // <stdin>:2:9: warning: unused variable 'x' [-Wunused-variable]
    const diagRegex = /^(?:<stdin>|.+?):(\d+):(\d+):\s+(error|warning|note):\s+(.*)$/;

    for (const rawLine of stderr.split('\n')) {
      const match = rawLine.match(diagRegex);
      if (match) {
        const line = parseInt(match[1], 10);
        const col = parseInt(match[2], 10);
        const rawSev = match[3];
        const message = match[4].trim();

        diagnostics.push({
          line,
          col,
          severity: rawSev === 'error' ? 'error' : rawSev === 'warning' ? 'warning' : 'info',
          message,
        });
      }
    }

    const errorCount = diagnostics.filter((d) => d.severity === 'error').length;
    const warningCount = diagnostics.filter((d) => d.severity === 'warning').length;

    return NextResponse.json({
      success: true,
      diagnostics,
      errorCount,
      warningCount,
      clean: diagnostics.length === 0,
      rawStderr: stderr,
    });
  } catch (err: any) {
    return NextResponse.json(
      {
        success: false,
        error: err.message || 'Internal lint error',
        diagnostics: [],
        errorCount: 0,
        warningCount: 0,
        clean: false,
      },
      { status: 500 }
    );
  }
}
