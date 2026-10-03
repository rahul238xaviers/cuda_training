import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getTopicLocation } from '@/lib/workspace';

export async function POST(request: NextRequest) {
  try {
    const { chapterId, tier, volumeId } = await request.json();

    if (!chapterId || !tier) {
      return NextResponse.json({ success: false, error: 'chapterId and tier are required' }, { status: 400 });
    }

    const loc = getTopicLocation(chapterId, volumeId);
    if (!loc) {
      return NextResponse.json({ success: false, error: `Topic or Chapter ${chapterId} not found` }, { status: 404 });
    }

    const topicDir = loc.fullPath;
    const exCu = path.join(topicDir, 'exercise', `${tier}_workbook.cu`);
    const exCpp = path.join(topicDir, 'exercise', `${tier}_workbook.cpp`);
    const exPath = fs.existsSync(exCu) ? exCu : exCpp;

    const actualExt = fs.existsSync(exCu) ? 'cu' : 'cpp';
    const solDir = path.join(topicDir, 'solution');
    const solPath = path.join(solDir, `${tier}_workbook.${actualExt}`);

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
      message: `Reset ${tier}_workbook.${actualExt} to original exercise template`,
      code: starterCode,
    });
  } catch (err: any) {
    return NextResponse.json({ success: false, error: err.message }, { status: 500 });
  }
}
