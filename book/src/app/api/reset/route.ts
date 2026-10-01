import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { MODULE1_DIR, getChapterFolder } from '@/lib/workspace';

export async function POST(request: NextRequest) {
  try {
    const { chapterId, tier } = await request.json();

    if (!chapterId || !tier) {
      return NextResponse.json({ success: false, error: 'chapterId and tier are required' }, { status: 400 });
    }

    const folder = getChapterFolder(chapterId);
    if (!folder) {
      return NextResponse.json({ success: false, error: `Chapter ${chapterId} not found` }, { status: 404 });
    }

    const chapterDir = path.join(MODULE1_DIR, folder);
    const exPath = path.join(chapterDir, 'exercise', `${tier}_workbook.cpp`);
    const solDir = path.join(chapterDir, 'solution');
    const solPath = path.join(solDir, `${tier}_workbook.cpp`);

    if (!fs.existsSync(exPath)) {
      return NextResponse.json({ success: false, error: `Exercise template not found at ${exPath}` }, { status: 404 });
    }

    const starterCode = fs.readFileSync(exPath, 'utf-8');

    // Create solution directory if needed
    if (!fs.existsSync(solDir)) {
      fs.mkdirSync(solDir, { recursive: true });
    }

    // Overwrite solution file with fresh starter exercise code
    fs.writeFileSync(solPath, starterCode, 'utf-8');

    return NextResponse.json({
      success: true,
      message: `Reset ${tier}_workbook.cpp to original exercise template`,
      code: starterCode,
    });
  } catch (err: any) {
    return NextResponse.json({ success: false, error: err.message }, { status: 500 });
  }
}
