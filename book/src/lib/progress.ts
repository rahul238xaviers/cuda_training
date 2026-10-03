import fs from 'fs';
import path from 'path';

export const WORKSPACE_ROOT = path.resolve(process.cwd(), '..');
export const PROGRESS_FILE_PATH = path.join(WORKSPACE_ROOT, 'user_progress.json');

export interface TierProgress {
  status: 'passed' | 'pending';
  testsPassed: number;
  totalTests: number;
  passedAt?: string;
}

export interface TopicProgress {
  tiers: {
    beginner: TierProgress;
    intermediate: TierProgress;
    champion: TierProgress;
  };
  completed: boolean;
}

export interface UserProgressData {
  version: number;
  updatedAt: string;
  activeTopic?: string;
  activeVolume?: string;
  activePlayground?: string;
  topics: Record<string, TopicProgress>;
}

const DEFAULT_INITIAL_PROGRESS: UserProgressData = {
  version: 1,
  updatedAt: new Date().toISOString(),
  activeTopic: '1.2',
  activeVolume: 'vol1',
  activePlayground: 'cpp',
  topics: {
    '1.1': {
      completed: true,
      tiers: {
        beginner: { status: 'passed', testsPassed: 3, totalTests: 3, passedAt: new Date().toISOString() },
        intermediate: { status: 'passed', testsPassed: 3, totalTests: 3, passedAt: new Date().toISOString() },
        champion: { status: 'passed', testsPassed: 3, totalTests: 3, passedAt: new Date().toISOString() },
      },
    },
    '1.2': {
      completed: false,
      tiers: {
        beginner: { status: 'passed', testsPassed: 3, totalTests: 3, passedAt: new Date().toISOString() },
        intermediate: { status: 'pending', testsPassed: 0, totalTests: 3 },
        champion: { status: 'pending', testsPassed: 0, totalTests: 3 },
      },
    },
  },
};

/**
 * Loads user progress from user_progress.json.
 * If file does not exist, creates it with default initial progress.
 */
export function loadUserProgress(): UserProgressData {
  try {
    if (fs.existsSync(PROGRESS_FILE_PATH)) {
      const raw = fs.readFileSync(PROGRESS_FILE_PATH, 'utf-8');
      const data = JSON.parse(raw);
      if (data && typeof data === 'object' && data.topics) {
        return data;
      }
    }
  } catch (err) {
    console.error('Failed to read user_progress.json, reinitializing:', err);
  }

  // Initialize file
  saveUserProgress(DEFAULT_INITIAL_PROGRESS);
  return DEFAULT_INITIAL_PROGRESS;
}

/**
 * Saves user progress to user_progress.json atomically.
 */
export function saveUserProgress(data: UserProgressData): void {
  try {
    data.updatedAt = new Date().toISOString();
    fs.writeFileSync(PROGRESS_FILE_PATH, JSON.stringify(data, null, 2), 'utf-8');
  } catch (err) {
    console.error('Failed to save user_progress.json:', err);
  }
}

/**
 * Updates a specific tier outcome for a chapter/topic.
 */
export function recordTierProgress(
  topicId: string,
  tier: 'beginner' | 'intermediate' | 'champion',
  passed: boolean,
  testsCount: number = 3
): UserProgressData {
  const data = loadUserProgress();
  if (!data.topics[topicId]) {
    data.topics[topicId] = {
      completed: false,
      tiers: {
        beginner: { status: 'pending', testsPassed: 0, totalTests: testsCount },
        intermediate: { status: 'pending', testsPassed: 0, totalTests: testsCount },
        champion: { status: 'pending', testsPassed: 0, totalTests: testsCount },
      },
    };
  }

  const topic = data.topics[topicId];
  topic.tiers[tier] = {
    status: passed ? 'passed' : 'pending',
    testsPassed: passed ? testsCount : 0,
    totalTests: testsCount,
    passedAt: passed ? new Date().toISOString() : undefined,
  };

  topic.completed =
    topic.tiers.beginner.status === 'passed' &&
    topic.tiers.intermediate.status === 'passed' &&
    topic.tiers.champion.status === 'passed';

  saveUserProgress(data);
  return data;
}

/**
 * Updates the user's last visited position.
 */
export function updateLastPosition(activeTopic?: string, activeVolume?: string, activePlayground?: string): void {
  const data = loadUserProgress();
  if (activeTopic) data.activeTopic = activeTopic;
  if (activeVolume) data.activeVolume = activeVolume;
  if (activePlayground) data.activePlayground = activePlayground;
  saveUserProgress(data);
}
