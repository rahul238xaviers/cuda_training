import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getTopicLocation, getPlaygroundInfo, PlaygroundType } from '@/lib/workspace';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { type, chapterId, tier, target = 'solution', code, playgroundType = 'cpp', volumeId } = body;

    // 1. Save Playground (C++, CUDA, Kernel)
    if (type === 'playground') {
      const pgInfo = getPlaygroundInfo(playgroundType as PlaygroundType);
      fs.writeFileSync(pgInfo.filePath, code, 'utf-8');
      return NextResponse.json({ success: true, path: pgInfo.filename, playgroundType: pgInfo.type });
    }

    // 2. Fork to Sandbox Action
    if (type === 'fork_to_sandbox') {
      const targetPg: PlaygroundType = body.isKernel ? 'kernel' : body.isCuda ? 'cuda' : 'cpp';
      const pgInfo = getPlaygroundInfo(targetPg);

      // Safeguard: backup existing playground content if non-empty
      if (fs.existsSync(pgInfo.filePath)) {
        const existing = fs.readFileSync(pgInfo.filePath, 'utf-8');
        if (existing.trim() && existing !== code) {
          const backupPath = pgInfo.filePath.replace(/\.(cpp|cu)$/, '.backup.$1');
          fs.writeFileSync(backupPath, existing, 'utf-8');
        }
      }

      fs.writeFileSync(pgInfo.filePath, code, 'utf-8');
      return NextResponse.json({
        success: true,
        path: pgInfo.filename,
        playgroundType: pgInfo.type,
        message: `Successfully forked to ${pgInfo.filename}!`,
      });
    }

    // 3. Save Workbook
    if (type === 'workbook') {
      const loc = getTopicLocation(chapterId, volumeId);
      if (!loc) {
        return NextResponse.json({ success: false, error: `Topic or Chapter ${chapterId} not found` }, { status: 404 });
      }
      const dir = path.join(loc.fullPath, target);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      const filePath = path.join(dir, `${tier}_workbook.${loc.ext}`);
      fs.writeFileSync(filePath, code, 'utf-8');
      return NextResponse.json({ success: true, path: filePath });
    }

    return NextResponse.json({ success: false, error: 'Invalid save type' }, { status: 400 });
  } catch (err: any) {
    return NextResponse.json({ success: false, error: err.message }, { status: 500 });
  }
}
