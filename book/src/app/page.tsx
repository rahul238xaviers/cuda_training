'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { SidebarTree, ActiveNode } from '@/components/SidebarTree';
import { HeaderBar } from '@/components/HeaderBar';
import { ReaderView } from '@/components/ReaderView';
import { LabEditorView } from '@/components/LabEditorView';
import { KernelSpecView } from '@/components/KernelSpecView';
import { TeacherDrawer } from '@/components/TeacherDrawer';
import { OrientationGuide } from '@/components/OrientationGuide';
import { Loader2 } from 'lucide-react';
import { getStoredTheme, applyTheme } from '@/lib/theme';

export default function BookPlatform() {
  const [treeData, setTreeData] = useState<any>(null);
  const [activeNode, setActiveNode] = useState<ActiveNode>({
    type: 'theory',
    volumeId: 'vol1',
    chapterId: '1.1',
  });
  const [nodeData, setNodeData] = useState<any>(null);
  const [activeCode, setActiveCode] = useState<string>('');
  const [target, setTarget] = useState<'solution' | 'exercise'>('solution');
  const [isLoadingNode, setIsLoadingNode] = useState<boolean>(true);
  const [isHydrated, setIsHydrated] = useState<boolean>(false);


  // Live Disk Save State
  const [saveStatus, setSaveStatus] = useState<'saved' | 'saving' | 'unsaved'>('saved');
  const autoSaveTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Terminal & Execution State
  const [terminalOutput, setTerminalOutput] = useState<string>('');
  const [runStatus, setRunStatus] = useState<'idle' | 'running' | 'success' | 'error'>('idle');
  const [durationMs, setDurationMs] = useState<number | undefined>(undefined);
  const [exitCode, setExitCode] = useState<number | null>(null);

  // Teacher Drawer State
  const [isTeacherOpen, setIsTeacherOpen] = useState<boolean>(false);

  // Navigation Sidebar Resizing & Collapse State
  // Always start with SSR-safe defaults, then hydrate from localStorage in useEffect.
  // Using localStorage in useState initializers causes hydration mismatches because
  // the server renders with the fallback while the client immediately gets the stored value.
  const [sidebarWidth, setSidebarWidth] = useState<number>(290);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState<boolean>(false);

  // Hydrate persisted sidebar prefs after mount (client-only)
  useEffect(() => {
    try {
      const savedWidth = localStorage.getItem('sidebar_width');
      if (savedWidth) setSidebarWidth(Math.max(220, Math.min(520, parseInt(savedWidth, 10))));
      const savedCollapsed = localStorage.getItem('sidebar_collapsed');
      if (savedCollapsed !== null) setIsSidebarCollapsed(savedCollapsed === 'true');
    } catch {}
  }, []);

  const handleResizeSidebar = (w: number) => {
    setSidebarWidth(w);
    try {
      localStorage.setItem('sidebar_width', String(w));
    } catch {}
  };

  const handleToggleSidebar = useCallback(() => {
    setIsSidebarCollapsed((prev) => {
      const next = !prev;
      try {
        localStorage.setItem('sidebar_collapsed', String(next));
      } catch {}
      return next;
    });
  }, []);

  // Load Tree on mount
  const loadTree = async () => {
    try {
      const res = await fetch('/api/tree');
      const data = await res.json();
      setTreeData(data);
    } catch (err) {
      console.error('Failed to load tree:', err);
    }
  };

  useEffect(() => {
    const saved = getStoredTheme();
    applyTheme(saved);
    loadTree();

    // Hydrate activeNode and target from URL query params or localStorage
    try {
      const searchParams = new URLSearchParams(window.location.search);
      const urlType = searchParams.get('type') as any;

      if (urlType) {
        const nodeFromUrl: ActiveNode = {
          type: urlType,
          chapterId: searchParams.get('chapter') || undefined,
          volumeId: searchParams.get('volumeId') || 'vol1',
          tier: (searchParams.get('tier') as any) || undefined,
          kernelId: searchParams.get('kernelId') ? parseInt(searchParams.get('kernelId')!, 10) : undefined,
          playgroundType: (searchParams.get('playgroundType') as any) || undefined,
          practiceKernel: searchParams.get('practiceKernel') || undefined,
        };
        setActiveNode(nodeFromUrl);
      } else {
        const savedNode = localStorage.getItem('active_node');
        if (savedNode) {
          const parsed = JSON.parse(savedNode);
          if (parsed && parsed.type) {
            setActiveNode(parsed);
          }
        }
      }

      const savedTarget = localStorage.getItem('active_target');
      if (savedTarget === 'solution' || savedTarget === 'exercise') {
        setTarget(savedTarget);
      }
    } catch {}

    setIsHydrated(true);
  }, []);

  // Persist activeNode and sync with browser URL
  useEffect(() => {
    if (!isHydrated || !activeNode) return;
    try {
      localStorage.setItem('active_node', JSON.stringify(activeNode));

      const params = new URLSearchParams();
      params.set('type', activeNode.type);
      if (activeNode.chapterId) params.set('chapter', activeNode.chapterId);
      if (activeNode.volumeId) params.set('volumeId', activeNode.volumeId);
      if (activeNode.tier) params.set('tier', activeNode.tier);
      if (activeNode.kernelId) params.set('kernelId', String(activeNode.kernelId));
      if (activeNode.playgroundType) params.set('playgroundType', activeNode.playgroundType);
      if (activeNode.practiceKernel) params.set('practiceKernel', activeNode.practiceKernel);

      const newUrl = `${window.location.pathname}?${params.toString()}`;
      window.history.replaceState(null, '', newUrl);
    } catch {}
  }, [activeNode, isHydrated]);

  // Persist target preference
  useEffect(() => {
    if (!isHydrated) return;
    try {
      localStorage.setItem('active_target', target);
    } catch {}
  }, [target, isHydrated]);


  // Fetch node content whenever activeNode changes
  const fetchNodeContent = useCallback(async (node: ActiveNode, currentTarget: 'solution' | 'exercise') => {
    setIsLoadingNode(true);
    setRunStatus('idle');
    setSaveStatus('saved');

    try {
      const params = new URLSearchParams();
      params.set('type', node.type);
      if (node.chapterId) params.set('chapter', node.chapterId);
      if (node.tier) params.set('tier', node.tier);
      if (node.kernelId) params.set('kernelId', String(node.kernelId));
      if (node.volumeId) params.set('volumeId', node.volumeId);
      if (node.playgroundType) params.set('playgroundType', node.playgroundType);
      if (node.practiceKernel) params.set('practiceKernel', node.practiceKernel);

      const res = await fetch(`/api/node?${params.toString()}`);
      const data = await res.json();
      setNodeData(data);

      if (node.type === 'workbook') {
        const code = currentTarget === 'solution' ? data.solutionCode : data.exerciseCode;
        setActiveCode(code || '');
        const ext = data.ext || 'cpp';
        const compiler = ext === 'cu' ? 'nvcc -O3 -std=c++17 --extended-lambda' : 'clang++ -std=c++20 -O3';
        setTerminalOutput(`Ready to compile ${node.tier}_workbook.${ext} with ${compiler}.`);
      } else if (node.type === 'practice') {
        setActiveCode(data.code || '');
        setTerminalOutput(`Ready to compile kernels/practice/${data.filePath || node.practiceKernel} with nvcc -O3 -std=c++17 --extended-lambda.`);
      } else if (node.type === 'playground') {
        setActiveCode(data.code || '');
        const compiler =
          data.playgroundType === 'cuda'
            ? 'nvcc -O3 -std=c++17 --extended-lambda'
            : data.playgroundType === 'kernel'
            ? 'nvcc -O3 -std=c++17 (Kernel Benchmark Lab)'
            : 'clang++ -std=c++20 -O3';
        setTerminalOutput(`Ready to compile ${data.filePath} with ${compiler}.`);
      }

      // Persist active position to user_progress.json
      fetch('/api/progress', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'set_position',
          activeTopic: node.chapterId,
          activeVolume: node.volumeId,
          activePlayground: node.playgroundType,
        }),
      }).catch(() => {});
    } catch (err) {
      console.error('Failed to load node content:', err);
    } finally {
      setIsLoadingNode(false);
    }
  }, []);

  useEffect(() => {
    if (!isHydrated) return;
    fetchNodeContent(activeNode, target);
  }, [activeNode, fetchNodeContent, target, isHydrated]);


  const handleTargetChange = async (newTarget: 'solution' | 'exercise') => {
    if (newTarget === target) return;

    // 1. If currently on 'solution' with unsaved work, auto-save before leaving!
    if (target === 'solution' && saveStatus === 'unsaved') {
      await handleSave(activeCode);
    }

    // 2. If switching to starter, show confirmation & reassurance prompt
    if (newTarget === 'exercise') {
      const ok = window.confirm(
        "Switch to Starter view?\n\n" +
        "Your solution is safely saved on disk.\n" +
        "The starter template is read-only for reference.\n\n" +
        "Click OK to view Starter (switch back to 'Solution' anytime to edit & run)."
      );
      if (!ok) return;
    }

    setTarget(newTarget);
    if (nodeData && activeNode.type === 'workbook') {
      if (target === 'solution') {
        nodeData.solutionCode = activeCode;
      } else {
        nodeData.exerciseCode = activeCode;
      }
      const code = newTarget === 'solution' ? nodeData.solutionCode : nodeData.exerciseCode;
      setActiveCode(code || '');
      setSaveStatus('saved');
    }
  };

  // Save code directly to local file on disk
  const handleSave = useCallback(
    async (codeToSave?: string) => {
      const code = codeToSave !== undefined ? codeToSave : activeCode;
      setSaveStatus('saving');

      try {
        const payload: any = {
          type: activeNode.type,
          code,
        };

        if (activeNode.type === 'practice') {
          payload.practiceKernel = activeNode.practiceKernel;
        } else if (activeNode.type === 'playground') {
          payload.playgroundType = activeNode.playgroundType || 'cpp';
        } else if (activeNode.type === 'workbook') {
          payload.chapterId = activeNode.chapterId;
          payload.tier = activeNode.tier;
          payload.target = target;
          payload.volumeId = activeNode.volumeId;
        }

        const res = await fetch('/api/save', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (data.success) {
          setSaveStatus('saved');
        } else {
          setSaveStatus('unsaved');
          console.error('Save failed:', data.error);
        }
      } catch (err: any) {
        setSaveStatus('unsaved');
        console.error('Save error:', err.message);
      }
    },
    [activeCode, activeNode, target]
  );

  // Debounced auto-save on code changes
  const handleCodeChange = (newCode: string) => {
    setActiveCode(newCode);
    setSaveStatus('unsaved');

    if (autoSaveTimerRef.current) {
      clearTimeout(autoSaveTimerRef.current);
    }

    autoSaveTimerRef.current = setTimeout(() => {
      handleSave(newCode);
    }, 1200);
  };

  // Format code via clang-format (returns formatted code or null)
  const handleFormat = useCallback(async (): Promise<string | null> => {
    if (!activeCode) return null;
    try {
      const res = await fetch('/api/format', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: activeCode }),
      });
      const data = await res.json();
      if (data.success && data.formatted) {
        handleCodeChange(data.formatted);
        return data.formatted;
      }
    } catch (err) {
      console.error('Format failed:', err);
    }
    return null;
  }, [activeCode]);

  // Global keydown listener to intercept Cmd+S and Cmd+B
  useEffect(() => {
    const handleGlobalKeyDown = async (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 's') {
        e.preventDefault();
        e.stopPropagation();
        // Format first on Cmd+S, then save
        const formatted = await handleFormat();
        handleSave(formatted || undefined);
      } else if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'b') {
        e.preventDefault();
        e.stopPropagation();
        handleToggleSidebar();
      }
    };

    window.addEventListener('keydown', handleGlobalKeyDown, { capture: true });
    return () => {
      window.removeEventListener('keydown', handleGlobalKeyDown, { capture: true });
    };
  }, [handleFormat, handleSave, handleToggleSidebar]);

  // Compile and run active node in isolated sandbox
  const handleRun = async () => {
    await handleSave();

    const isCuda = nodeData?.isCuda || nodeData?.ext === 'cu' || activeNode.type === 'practice' || activeNode.playgroundType === 'cuda' || activeNode.playgroundType === 'kernel';
    const compiler = isCuda ? 'nvcc -O3 -std=c++17 --extended-lambda' : 'clang++ -std=c++20 -O3';
    setTerminalOutput(`📦 Preparing isolated sandbox...\nCompiling with ${compiler}...\nExecuting binary with 10s watchdog limit...`);
    setDurationMs(undefined);
    setExitCode(null);

    try {
      const payload: any = {
        type: activeNode.type,
        code: activeCode,
      };

      if (activeNode.type === 'practice') {
        payload.practiceKernel = activeNode.practiceKernel;
      } else if (activeNode.type === 'playground') {
        payload.playgroundType = activeNode.playgroundType || 'cpp';
      } else if (activeNode.type === 'workbook') {
        payload.chapterId = activeNode.chapterId;
        payload.tier = activeNode.tier;
        payload.target = target;
        payload.volumeId = activeNode.volumeId;
      }

      const res = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const result = await res.json();

      setDurationMs(result.durationMs);
      setExitCode(result.exitCode ?? (result.success ? 0 : 1));

      if (result.problemBreakdown && result.problemBreakdown.length > 0) {
        setNodeData((prev: any) => {
          if (!prev) return prev;
          return {
            ...prev,
            problems: result.problemBreakdown,
            testsPassed: result.testsPassed ?? prev.testsPassed,
            testsCount: result.testsCount ?? prev.testsCount,
            newProblemsCount: Math.max(0, (result.testsCount ?? prev.testsCount) - (result.testsPassed ?? prev.testsPassed)),
          };
        });
        loadTree();
      }

      if (result.success) {
        setRunStatus('success');
        setTerminalOutput(result.stdout || '(Executed in sandbox with return code 0)');
        loadTree();
      } else {
        setRunStatus('error');
        const parts = [];
        if (result.stdout) parts.push(result.stdout);
        if (result.stderr) parts.push(result.stderr);
        const combined = parts.join('\n\n') || '(No output)';
        const errText =
          result.stage === 'compilation'
            ? `[SANDBOX COMPILATION ERROR]:\n${combined}`
            : `[SANDBOX RUNTIME ERROR (Exit ${result.exitCode})]:\n${combined}`;
        setTerminalOutput(errText);
      }
    } catch (err: any) {
      setRunStatus('error');
      setTerminalOutput(`[SANDBOX EXECUTION FAILED]: ${err.message}`);
    }
  };

  // Fork to Sandbox: copies current code into playground file and switches to sandbox mode
  const handleForkToSandbox = async () => {
    if (!activeCode) return;
    try {
      const isKernel = activeNode.volumeId === 'vol3' || activeNode.chapterId?.startsWith('k') || activeNode.type === 'kernel';
      const isCuda = isKernel || activeNode.volumeId === 'vol2' || /^[2-7]\./.test(activeNode.chapterId || '') || nodeData?.ext === 'cu';

      const res = await fetch('/api/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: 'fork_to_sandbox',
          code: activeCode,
          isKernel,
          isCuda,
        }),
      });
      const data = await res.json();
      if (data.success) {
        setActiveNode({
          type: 'playground',
          volumeId: 'sandbox',
          playgroundType: data.playgroundType,
        });
        setTerminalOutput(`🚀 [FORKED TO SANDBOX]: Successfully copied into ${data.path}!\nYou are now in isolated sandbox mode with live compiler access.`);
      }
    } catch (err: any) {
      console.error('Fork to sandbox failed:', err);
    }
  };

  // Start Over: Reset to original exercise template
  const handleStartOver = async () => {
    if (activeNode.type !== 'workbook' || !activeNode.chapterId || !activeNode.tier) return;

    const confirmed = window.confirm(
      `Start over ${activeNode.chapterId} (${activeNode.tier})?\n\nThis will reset your local working file back to the clean exercise starter template and reset passed problem progress.`
    );
    if (!confirmed) return;

    try {
      const res = await fetch('/api/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chapterId: activeNode.chapterId,
          tier: activeNode.tier,
          volumeId: activeNode.volumeId,
        }),
      });
      const data = await res.json();
      if (data.success) {
        setActiveCode(data.code || '');
        setSaveStatus('saved');
        setRunStatus('idle');
        setDurationMs(undefined);
        setExitCode(null);
        setTerminalOutput(`🔄 Workbook reset to fresh exercise template. Ready to begin!`);

        // Immediately update local nodeData so problem status pills and counts reflect 0 passed
        setNodeData((prev: any) => {
          if (!prev) return prev;
          const freshProblems = (data.problems || prev.problems || []).map((p: any) => ({
            ...p,
            status: 'pending',
          }));
          const totalCount = data.testsCount ?? prev.testsCount ?? freshProblems.length;
          return {
            ...prev,
            problems: freshProblems,
            testsPassed: 0,
            status: 'pending',
            newProblemsCount: totalCount,
          };
        });

        // Sync fresh tree stats and reload full node content
        await loadTree();
        await fetchNodeContent(activeNode, target);
      } else {
        alert(`Failed to reset: ${data.error}`);
      }
    } catch (err: any) {
      alert(`Reset error: ${err.message}`);
    }
  };

  const handleReset = () => {
    if (nodeData && activeNode.type === 'workbook') {
      const original = target === 'solution' ? nodeData.solutionCode : nodeData.exerciseCode;
      setActiveCode(original || '');
      setSaveStatus('saved');
    }
  };

  const handleClearConsole = () => {
    setTerminalOutput('');
    setRunStatus('idle');
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[var(--bg-app)] text-[var(--text-primary)] app-root-container">
      {/* 1. Left Tree Navigation Sidebar */}
      <SidebarTree
        treeData={treeData}
        activeNode={activeNode}
        onSelectNode={(node) => setActiveNode(node)}
        onRefreshTree={loadTree}
        width={sidebarWidth}
        onResize={handleResizeSidebar}
        isCollapsed={isSidebarCollapsed}
        onToggleCollapse={handleToggleSidebar}
      />

      {/* 2. Main Experience Container */}
      <div className="flex-1 flex flex-col h-full overflow-hidden min-w-0">
        {/* Context-Segregated Header Bar */}
        <HeaderBar
          activeNode={activeNode}
          nodeData={nodeData}
          target={target}
          onTargetChange={handleTargetChange}
          isRunning={runStatus === 'running'}
          onRun={handleRun}
          onSave={() => handleSave()}
          onFormat={handleFormat}
          onReset={handleReset}
          onStartOver={handleStartOver}
          onForkToSandbox={handleForkToSandbox}
          onClearConsole={handleClearConsole}
          onSwitchNode={(node) => setActiveNode(node)}
          onToggleTeacher={() => setIsTeacherOpen((prev) => !prev)}
          isTeacherOpen={isTeacherOpen}
          isSidebarCollapsed={isSidebarCollapsed}
          onToggleSidebar={handleToggleSidebar}
        />

        {/* Dynamic Facade Viewport */}
        <main className="flex-1 flex overflow-hidden relative">
          {isLoadingNode ? (
            <div className="flex-1 flex items-center justify-center text-slate-400 gap-2">
              <Loader2 className="w-5 h-5 animate-spin text-sky-400" />
              <span className="text-xs font-mono">Loading node content from disk...</span>
            </div>
          ) : (
            <>
              {/* Orientation & Learning Guide View (Dynamic AST Pipeline) */}
              {activeNode.type === 'guide' && (
                <OrientationGuide
                  onNavigate={(node) => setActiveNode(node)}
                  content={nodeData?.content}
                />
              )}

              {/* Theory or Cheat Sheet Reader View */}
              {(activeNode.type === 'theory' || activeNode.type === 'cheat_sheet') && (
                <ReaderView
                  title={nodeData?.title || 'Documentation'}
                  content={nodeData?.content || ''}
                  chapterId={activeNode.chapterId}
                  type={activeNode.type}
                />
              )}

              {/* Live Workbook, Playground, or Open Practice Lab View */}
              {(activeNode.type === 'workbook' || activeNode.type === 'playground' || activeNode.type === 'practice') && (
                <LabEditorView
                  filePath={
                    activeNode.type === 'practice'
                      ? `kernels/practice/${nodeData?.filePath || activeNode.practiceKernel || 'gpu_check.cu'}`
                      : activeNode.type === 'playground'
                      ? nodeData?.filePath ||
                        (activeNode.playgroundType === 'cuda'
                          ? 'playground.cu'
                          : activeNode.playgroundType === 'kernel'
                          ? 'playground_kernel.cu'
                          : 'playground.cpp')
                      : nodeData?.relPath || `workbook.${nodeData?.ext || 'cpp'}`
                  }
                  code={activeCode}
                  onChange={handleCodeChange}
                  onSave={(savedCode) => handleSave(savedCode)}
                  saveStatus={saveStatus}
                  output={terminalOutput}
                  runStatus={runStatus}
                  durationMs={durationMs}
                  exitCode={exitCode}
                  readOnly={target === 'exercise' && activeNode.type === 'workbook'}
                  problems={nodeData?.problems}
                  testsCount={nodeData?.testsCount}
                  testsPassed={nodeData?.testsPassed}
                  newProblemsCount={nodeData?.newProblemsCount}
                />
              )}

              {/* Legacy Kernel Blueprint View (if explicitly opened by id) */}
              {activeNode.type === 'kernel' && !activeNode.chapterId && (
                <KernelSpecView kernelData={nodeData} />
              )}
            </>
          )}
        </main>
      </div>

      {/* 3. Antigravity AI Teacher Assistant Drawer */}
      <TeacherDrawer
        isOpen={isTeacherOpen}
        onClose={() => setIsTeacherOpen(false)}
        chapterId={activeNode.chapterId}
        activeNode={activeNode}
        activeCode={activeCode}
        runOutput={terminalOutput}
        runStatus={runStatus}
        nodeData={nodeData}
        onApplyCode={(newCode) => {
          handleCodeChange(newCode);
          setTerminalOutput((prev) => `${prev}\n[Teacher]: Code successfully applied to editor buffer.`);
        }}
      />
    </div>
  );
}
