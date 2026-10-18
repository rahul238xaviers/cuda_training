import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getTopicLocation, WORKSPACE_ROOT } from '@/lib/workspace';

const MIME_TYPES: Record<string, string> = {
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.gif': 'image/gif',
};

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const chapterId = searchParams.get('chapter') || '1.1';
  const volumeId = searchParams.get('volumeId') || undefined;
  const filePath = searchParams.get('file') || searchParams.get('path');

  if (!filePath) {
    return new NextResponse('File parameter missing', { status: 400 });
  }

  // Prevent directory traversal attacks
  const safeRelative = path.normalize(filePath).replace(/^(\.\.[\/\\])+/, '');

  let resolvedPath = '';

  // 1. Try resolving relative to topic directory
  const loc = getTopicLocation(chapterId, volumeId);
  if (loc) {
    const candidate = path.join(loc.fullPath, safeRelative);
    if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) {
      resolvedPath = candidate;
    }
  }

  // 2. Try resolving relative to WORKSPACE_ROOT
  if (!resolvedPath) {
    const candidateRoot = path.join(WORKSPACE_ROOT, safeRelative);
    if (fs.existsSync(candidateRoot) && fs.statSync(candidateRoot).isFile()) {
      resolvedPath = candidateRoot;
    }
  }

  if (!resolvedPath) {
    return new NextResponse(`Asset not found: ${filePath}`, { status: 404 });
  }

  const ext = path.extname(resolvedPath).toLowerCase();
  const contentType = MIME_TYPES[ext] || 'application/octet-stream';
  const fileBuffer = fs.readFileSync(resolvedPath);

  return new NextResponse(fileBuffer, {
    status: 200,
    headers: {
      'Content-Type': contentType,
      'Cache-Control': 'public, max-age=3600, immutable',
    },
  });
}
