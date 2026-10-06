/**
 * Antigravity AI Teacher - Context Addition & Prompt Assembly Engine
 */

import { TEACHER_CLI_CONFIG } from '@/config/teacher/constants';
import { TEACHER_PEDAGOGY_RULES } from '@/config/teacher/prompts';
import { TEACHER_SKILLS } from '@/config/teacher/skills';

export interface StudentContextInput {
  chapterId?: string;
  title?: string;
  tier?: string;
  mode?: string;
  activeFile?: string;
  activeCode?: string;
  runOutput?: string;
  runStatus?: string;
}

export interface AssembledTeacherPrompt {
  fullPrompt: string;
  contextSection: string;
  debugContext: {
    activeFile: string;
    lineCount: number;
    codeBytes: number;
    hasRunOutput: boolean;
    runStatus: string;
    rulesEnforced: string[];
    activeSkills: string[];
    promptBytes: number;
    rawContextPayload: string;
  };
}

/**
 * Builds the comprehensive prompt payload injected into Antigravity CLI (agy)
 */
export function buildTeacherPrompt(
  studentPrompt: string,
  context?: StudentContextInput | string
): AssembledTeacherPrompt {
  let contextSection = '';
  let activeFile = 'playground.cpp';
  let lineCount = 0;
  let codeBytes = 0;
  let hasRunOutput = false;
  let runStatus = 'idle';

  if (context) {
    if (typeof context === 'string') {
      contextSection = `[STUDENT CONTEXT]\n${context}\n`;
    } else {
      activeFile = context.activeFile || 'playground.cpp';
      runStatus = context.runStatus || 'idle';
      hasRunOutput = Boolean(context.runOutput && context.runOutput.trim().length > 0);

      const headerParts: string[] = [];
      if (context.chapterId) headerParts.push(`Chapter: ${context.chapterId} (${context.title || ''})`);
      if (context.tier) headerParts.push(`Tier: ${context.tier}`);
      if (context.mode) headerParts.push(`Mode: ${context.mode}`);
      if (context.activeFile) headerParts.push(`File: ${context.activeFile}`);

      contextSection = `[STUDENT WORKSPACE CONTEXT]\n${headerParts.join(' | ')}\n`;

      if (context.activeCode && typeof context.activeCode === 'string') {
        lineCount = context.activeCode.split('\n').length;
        codeBytes = context.activeCode.length;
        const codeSnippet =
          context.activeCode.length > TEACHER_CLI_CONFIG.MAX_CODE_CHARS
            ? context.activeCode.substring(0, TEACHER_CLI_CONFIG.MAX_CODE_CHARS) + '\n... [truncated]'
            : context.activeCode;
        contextSection += `\n[CODE IN STUDENT EDITOR]:\n\`\`\`cpp\n${codeSnippet}\n\`\`\`\n`;
      }

      if (context.runOutput && typeof context.runOutput === 'string') {
        const runSnippet =
          context.runOutput.length > TEACHER_CLI_CONFIG.MAX_RUN_OUTPUT_CHARS
            ? context.runOutput.substring(0, TEACHER_CLI_CONFIG.MAX_RUN_OUTPUT_CHARS) + '\n... [truncated]'
            : context.runOutput;
        contextSection += `\n[LATEST RUN / COMPILER OUTPUT]:\n${runSnippet}\n`;
      }
    }
  }

  // Active skills summary injected into system prompt
  const skillsSummary = `[ACTIVE AGENT SKILLS]:\n${TEACHER_SKILLS.filter((s) => s.enabled)
    .map((s) => `- ${s.name}: ${s.description}`)
    .join('\n')}`;

  const fullPrompt = contextSection
    ? `${TEACHER_PEDAGOGY_RULES}\n\n${skillsSummary}\n\n${contextSection}\nStudent Question:\n${studentPrompt}`
    : `${TEACHER_PEDAGOGY_RULES}\n\n${skillsSummary}\n\nStudent Question:\n${studentPrompt}`;

  return {
    fullPrompt,
    contextSection,
    debugContext: {
      activeFile,
      lineCount,
      codeBytes,
      hasRunOutput,
      runStatus,
      rulesEnforced: ['NO_LATEX', 'ONE_STEP_AT_A_TIME', 'HARDWARE_FIRST', 'PLAYGROUND_FIRST', 'DIRECT_CODE_APPLY'],
      activeSkills: TEACHER_SKILLS.filter((s) => s.enabled).map((s) => s.id),
      promptBytes: fullPrompt.length,
      rawContextPayload: contextSection || 'No workspace context attached',
    },
  };
}
