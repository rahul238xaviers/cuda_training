import { NextRequest, NextResponse } from 'next/server';
import { spawnSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { WORKSPACE_ROOT } from '@/lib/workspace';
import { TEACHER_CLI_CONFIG } from '@/config/teacher/constants';
import { buildTeacherPrompt } from '@/lib/teacher/contextBuilder';

export async function POST(request: NextRequest) {
  try {
    const { prompt, conversationId, context } = await request.json();
    const homeDir = process.env.HOME || process.env.USERPROFILE || '';
    const agyPath = homeDir ? path.join(homeDir, '.local', 'bin', 'agy') : 'agy';

    if (!fs.existsSync(agyPath)) {
      return NextResponse.json({
        success: false,
        response: `Antigravity CLI (agy) was not found at ${agyPath}. Ensure it is installed in your ~/.local/bin path.`,
      });
    }

    // Assemble comprehensive prompt using modular contextBuilder & AGENTS.md rules
    const { fullPrompt, debugContext } = buildTeacherPrompt(prompt, context);

    const args = ['--print', fullPrompt, '--output-format', TEACHER_CLI_CONFIG.OUTPUT_FORMAT];
    if (conversationId && typeof conversationId === 'string' && conversationId.trim().length > 0) {
      args.push('--conversation', conversationId.trim());
    }

    const child = spawnSync(agyPath, args, {
      cwd: WORKSPACE_ROOT,
      encoding: 'utf-8',
      timeout: TEACHER_CLI_CONFIG.TIMEOUT_MS,
      maxBuffer: TEACHER_CLI_CONFIG.MAX_BUFFER_BYTES,
    });

    if (child.status !== 0) {
      return NextResponse.json({
        success: false,
        response: child.stderr || child.error?.message || 'Error executing Antigravity CLI',
      });
    }

    try {
      const parsed = JSON.parse(child.stdout);
      return NextResponse.json({
        success: true,
        response: parsed.response || child.stdout,
        conversationId: parsed.conversation_id || conversationId || null,
        debugContext,
      });
    } catch {
      return NextResponse.json({
        success: true,
        response: child.stdout,
        conversationId: conversationId || null,
        debugContext,
      });
    }
  } catch (err: any) {
    return NextResponse.json(
      {
        success: false,
        response: err.message || 'Internal server error while contacting Antigravity',
      },
      { status: 500 }
    );
  }
}
