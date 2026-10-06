/**
 * Antigravity AI Teacher - Registered Skills & Capabilities Registry
 */

export interface TeacherSkill {
  id: string;
  name: string;
  description: string;
  category: 'memory' | 'cuda' | 'debugging' | 'pedagogy' | 'code-generation';
  enabled: boolean;
}

export const TEACHER_SKILLS: TeacherSkill[] = [
  {
    id: 'memory-layout-visualization',
    name: 'Memory Layout & Alignment Visualizer',
    description: 'Traces memory alignments, 64-byte cache line bounds, and struct padding from first principles.',
    category: 'memory',
    enabled: true,
  },
  {
    id: 'cuda-stride-reduction',
    name: 'Tree Reduction & Stride Analyzer',
    description: 'Analyzes tree reductions, odd/even stride boundaries, non-power-of-2 tails, and warp divergence.',
    category: 'cuda',
    enabled: true,
  },
  {
    id: 'compiler-diagnostic-explainer',
    name: 'Compiler & Diagnostic Explainer',
    description: 'Interprets clang++ and nvcc compilation errors, template instantiations, and linker issues in real time.',
    category: 'debugging',
    enabled: true,
  },
  {
    id: 'monaco-code-updater',
    name: 'Direct Monaco Code Updater',
    description: 'Generates drop-in replacement snippets with 1-click sync into the active editor buffer.',
    category: 'code-generation',
    enabled: true,
  },
  {
    id: 'agents-md-pedagogy',
    name: 'AGENTS.md Pedagogy Guard',
    description: 'Enforces ASCII math, micro-steps pair programming, and hardware-first intuition on all turns.',
    category: 'pedagogy',
    enabled: true,
  },
];
