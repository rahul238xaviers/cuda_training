/**
 * Antigravity AI Teacher - Configuration & Storage Constants
 */

export const TEACHER_STORAGE_KEYS = {
  SESSIONS_V2: 'antigravity_teacher_sessions_v2',
  ACTIVE_SESSION_ID: 'antigravity_active_session_id',
  DRAWER_WIDTH: 'antigravity_teacher_drawer_width',
  LEGACY_CONV_ID: 'antigravity_teacher_conv_id',
  LEGACY_MESSAGES: 'antigravity_teacher_messages',
} as const;

export const TEACHER_DRAWER_CONFIG = {
  DEFAULT_WIDTH_PX: 460,
  MIN_WIDTH_PX: 360,
  MAX_WIDTH_PX: 1200,
  MAX_VIEWPORT_RATIO: 0.85,
} as const;

export const TEACHER_CLI_CONFIG = {
  TIMEOUT_MS: 120000,
  MAX_BUFFER_BYTES: 10 * 1024 * 1024,
  MAX_CODE_CHARS: 4000,
  MAX_RUN_OUTPUT_CHARS: 2000,
  OUTPUT_FORMAT: 'json',
} as const;

export interface QuickPromptChip {
  id: string;
  label: string;
  prompt: string;
  icon?: string;
  colorClass?: string;
}

export const TEACHER_QUICK_CHIPS: QuickPromptChip[] = [
  {
    id: 'review-active',
    label: '🔍 Review Active Code',
    prompt: 'Can you review my active solution step-by-step and verify my pointer arithmetic and boundary logic?',
    colorClass: 'text-sky-300',
  },
  {
    id: 'give-hint',
    label: '💡 Give Me a Hint',
    prompt: 'Can you give me a small hint for this problem without giving away the full answer?',
    colorClass: 'text-amber-300',
  },
  {
    id: 'hardware-intuition',
    label: '⚡ Hardware Intuition',
    prompt: 'Explain the hardware mental model and memory layout for this chapter from first principles.',
    colorClass: 'text-purple-300',
  },
  {
    id: 'update-code',
    label: '📝 Fix & Update Code',
    prompt: 'Can you fix the bug in my active code and provide the complete drop-in replacement snippet so I can apply it to the editor?',
    colorClass: 'text-emerald-300',
  },
];
