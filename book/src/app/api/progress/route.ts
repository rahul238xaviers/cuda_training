import { NextRequest, NextResponse } from 'next/server';
import { loadUserProgress, recordTierProgress, updateLastPosition } from '@/lib/progress';

export async function GET() {
  const data = loadUserProgress();
  return NextResponse.json(data);
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, topicId, tier, passed, testsCount, activeTopic, activeVolume, activePlayground } = body;

    if (action === 'record') {
      if (!topicId || !tier) {
        return NextResponse.json({ success: false, error: 'topicId and tier are required' }, { status: 400 });
      }
      const updated = recordTierProgress(topicId, tier, !!passed, testsCount || 3);
      return NextResponse.json({ success: true, progress: updated });
    }

    if (action === 'set_position') {
      updateLastPosition(activeTopic, activeVolume, activePlayground);
      return NextResponse.json({ success: true });
    }

    return NextResponse.json({ success: false, error: 'Unknown action' }, { status: 400 });
  } catch (err: any) {
    return NextResponse.json({ success: false, error: err.message }, { status: 500 });
  }
}
