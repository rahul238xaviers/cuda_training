import path from 'path';
import fs from 'fs';

export interface CurriculumManifestModule {
  num: number;
  diskNum?: number;
  title: string;
  displayTitle: string;
  tagline: string;
  category?: string;
  startCh?: number;
  endCh?: number;
}

export interface CurriculumManifestVolume {
  id: string;
  shortId: 'cpp' | 'cuda' | 'kernel' | string;
  title: string;
  displayTitle: string;
  tagline: string;
  description: string;
  sourceDir: string;
  folderPrefix?: string;
  language: 'cpp' | 'cuda';
  ext: 'cpp' | 'cu';
  modules: CurriculumManifestModule[];
}

export interface MetalKernelDefinition {
  id: number;
  name: string;
  category: string;
  metalFile: string;
  cudaFile: string;
  status: 'planned' | 'in_progress' | 'completed';
}

export interface CurriculumManifest {
  version: string;
  title: string;
  subtitle: string;
  volumes: CurriculumManifestVolume[];
  metalKernels: MetalKernelDefinition[];
}

const CONFIG_PATH = path.join(process.cwd(), 'src', 'config', 'curriculum.json');

let cachedManifest: CurriculumManifest | null = null;

export function loadCurriculumManifest(): CurriculumManifest {
  if (cachedManifest) return cachedManifest;

  try {
    if (fs.existsSync(CONFIG_PATH)) {
      const data = fs.readFileSync(CONFIG_PATH, 'utf-8');
      cachedManifest = JSON.parse(data) as CurriculumManifest;
      return cachedManifest;
    }
  } catch (err) {
    console.error('Failed to load curriculum.json, falling back to minimal defaults:', err);
  }

  // Graceful fallback if file read fails
  return {
    version: '1.0.0',
    title: 'C++ & CUDA Systems Mastery',
    subtitle: 'From Low-Level Memory to High-Performance GPU Kernels',
    volumes: [],
    metalKernels: [],
  };
}

export function getVolumeConfig(volumeId: string): CurriculumManifestVolume | undefined {
  const manifest = loadCurriculumManifest();
  return manifest.volumes.find((v) => v.id === volumeId || v.shortId === volumeId);
}
