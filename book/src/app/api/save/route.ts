import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { MODULE1_DIR, PLAYGROUND_PATH, getChapterFolder } from '@/lib/workspace';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { type, chapterId, tier, target = 'solution', code } = body;

    if (type === 'playground') {
      fs.writeFileSync(PLAYGROUND_PATH, code, 'utf-8');
      return NextResponse.json({ success: true, path: 'playground.cpp' });
    }

    if (type === 'workbook') {
      const folder = getChapterFolder(chapterId);
      if (!folder) {
        return NextResponse.json({ success: false, error: `Chapter ${chapterId} not found` }, { status: 404 });
      }
      const dir = path.join(MODULE1_DIR, folder, target);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      const filePath = path.join(dir, `${tier}_workbook.cpp`);
      fs.writeFileSync(filePath, code, 'utf-8');
      return NextResponse.json({ success: true, path: filePath });
    }

    return NextResponse.json({ success: false, error: 'Invalid save type' }, { status: 400 });
  } catch (err: any) {
    return NextResponse.json({ success: false, error: err.message }, { status: 500 });
  }
}
