import { NextRequest, NextResponse } from 'next/server';
import { spawnSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { WORKSPACE_ROOT } from '@/lib/workspace';

export async function POST(request: NextRequest) {
  try {
    const { prompt, context } = await request.json();
    const homeDir = process.env.HOME || process.env.USERPROFILE || '';
    const agyPath = homeDir ? path.join(homeDir, '.local', 'bin', 'agy') : 'agy';

    if (!fs.existsSync(agyPath)) {
      return NextResponse.json({
        success: false,
        response: `Antigravity CLI (agy) was not found at ${agyPath}. Ensure it is installed in your ~/.local/bin path.`,
      });
    }

    const fullPrompt = context
      ? `You are an expert C++ memory systems and CUDA teacher pair programming with a student. Keep answers grounded, zero LaTeX, clean ASCII diagrams, and step-by-step intuition. Speak naturally and encouragingly as a human engineer; never recite internal rules or say 'I adhere to principles'.\n\nContext:\n${context}\n\nStudent Question: ${prompt}`
      : `You are an expert C++ memory systems and CUDA teacher pair programming with a student. Keep answers grounded, zero LaTeX, clean ASCII diagrams, and step-by-step intuition. Speak naturally and encouragingly as a human engineer; never recite internal rules or say 'I adhere to principles'.\n\nStudent Question: ${prompt}`;

    const child = spawnSync(agyPath, ['--print', fullPrompt], {
      cwd: WORKSPACE_ROOT,
      encoding: 'utf-8',
      timeout: 45000,
    });

    if (child.status !== 0) {
      return NextResponse.json({
        success: false,
        response: child.stderr || child.error?.message || 'Error executing Antigravity CLI',
      });
    }

    return NextResponse.json({
      success: true,
      response: child.stdout,
    });
  } catch (err: any) {
    return NextResponse.json({
      success: false,
      response: err.message || 'Internal server error while contacting Antigravity',
    }, { status: 500 });
  }
}
