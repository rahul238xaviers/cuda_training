'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { SidebarTree, ActiveNode } from '@/components/SidebarTree';
import { HeaderBar } from '@/components/HeaderBar';
import { ReaderView } from '@/components/ReaderView';
import { LabEditorView } from '@/components/LabEditorView';
import { KernelSpecView } from '@/components/KernelSpecView';
import { TeacherDrawer } from '@/components/TeacherDrawer';
import { Loader2 } from 'lucide-react';

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
    loadTree();
  }, []);

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

      const res = await fetch(`/api/node?${params.toString()}`);
      const data = await res.json();
      setNodeData(data);

      if (node.type === 'workbook') {
        const code = currentTarget === 'solution' ? data.solutionCode : data.exerciseCode;
        setActiveCode(code || '');
        const ext = data.ext || 'cpp';
        const compiler = ext === 'cu' ? 'nvcc -O3 -std=c++17 --extended-lambda' : 'clang++ -std=c++20 -O3';
        setTerminalOutput(`Ready to compile ${node.tier}_workbook.${ext} with ${compiler}.`);
      } else if (node.type === 'playground') {
        setActiveCode(data.code || '');
        setTerminalOutput('Ready to compile playground.cpp with clang++ -std=c++20 -O3.');
      }
    } catch (err) {
      console.error('Failed to load node content:', err);
    } finally {
      setIsLoadingNode(false);
    }
  }, []);

  useEffect(() => {
    fetchNodeContent(activeNode, target);
  }, [activeNode, fetchNodeContent, target]);

  const handleTargetChange = (newTarget: 'solution' | 'exercise') => {
    setTarget(newTarget);
    if (nodeData && activeNode.type === 'workbook') {
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

        if (activeNode.type === 'workbook') {
          payload.chapterId = activeNode.chapterId;
          payload.tier = activeNode.tier;
          payload.target = target;
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

  // Global keydown listener to intercept Cmd+S / Ctrl+S and prevent browser Download dialog
  useEffect(() => {
    const handleGlobalKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 's') {
        e.preventDefault();
        e.stopPropagation();
        handleSave();
      }
    };

    window.addEventListener('keydown', handleGlobalKeyDown, { capture: true });
    return () => {
      window.removeEventListener('keydown', handleGlobalKeyDown, { capture: true });
    };
  }, [handleSave]);

  // Compile and run active node
  const handleRun = async () => {
    // Save to disk first
    await handleSave();

    const isCuda = nodeData?.isCuda || nodeData?.ext === 'cu';
    const compiler = isCuda ? 'nvcc -O3 -std=c++17 --extended-lambda' : 'clang++ -std=c++20 -O3';
    setTerminalOutput(`Compiling with ${compiler}...\nExecuting binary...`);
    setDurationMs(undefined);
    setExitCode(null);

    try {
      const payload: any = {
        type: activeNode.type,
        code: activeCode,
      };

      if (activeNode.type === 'workbook') {
        payload.chapterId = activeNode.chapterId;
        payload.tier = activeNode.tier;
        payload.target = target;
      }

      const res = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const result = await res.json();

      setDurationMs(result.durationMs);
      setExitCode(result.exitCode ?? (result.success ? 0 : 1));

      if (result.success) {
        setRunStatus('success');
        setTerminalOutput(result.stdout || '(Executed with return code 0 and no standard output)');
        loadTree();
      } else {
        setRunStatus('error');
        const errText =
          result.stage === 'compilation'
            ? `[COMPILATION ERROR]:\n${result.stderr || result.stdout}`
            : `[RUNTIME ERROR (Exit ${result.exitCode})]:\n${result.stderr || result.stdout}`;
        setTerminalOutput(errText);
      }
    } catch (err: any) {
      setRunStatus('error');
      setTerminalOutput(`[EXECUTION FAILED]: ${err.message}`);
    }
  };

  // Start Over: Reset to original exercise template
  const handleStartOver = async () => {
    if (activeNode.type !== 'workbook' || !activeNode.chapterId || !activeNode.tier) return;

    const confirmed = window.confirm(
      `Start over Chapter ${activeNode.chapterId} (${activeNode.tier})?\n\nThis will reset your local working file back to the original exercise starter template.`
    );
    if (!confirmed) return;

    try {
      const res = await fetch('/api/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chapterId: activeNode.chapterId,
          tier: activeNode.tier,
        }),
      });
      const data = await res.json();
      if (data.success) {
        setActiveCode(data.code || '');
        setSaveStatus('saved');
        setRunStatus('idle');
        setTerminalOutput(`🔄 Workbook reset to fresh exercise template. Ready to begin!`);
        loadTree();
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

  const handleFormat = async () => {
    if (!activeCode) return;
    try {
      const res = await fetch('/api/format', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: activeCode }),
      });
      const data = await res.json();
      if (data.success && data.formatted) {
        handleCodeChange(data.formatted);
      }
    } catch (err) {
      console.error('Format failed:', err);
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#080b11]">
      {/* 1. Left Tree Navigation Sidebar */}
      <SidebarTree
        treeData={treeData}
        activeNode={activeNode}
        onSelectNode={(node) => setActiveNode(node)}
        onRefreshTree={loadTree}
      />

      {/* 2. Main Experience Container */}
      <div className="flex-1 flex flex-col h-full overflow-hidden">
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
          onClearConsole={handleClearConsole}
          onSwitchNode={(node) => setActiveNode(node)}
          onToggleTeacher={() => setIsTeacherOpen((prev) => !prev)}
          isTeacherOpen={isTeacherOpen}
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
              {/* Theory or Cheat Sheet Reader View */}
              {(activeNode.type === 'theory' || activeNode.type === 'cheat_sheet') && (
                <ReaderView
                  title={nodeData?.title || 'Documentation'}
                  content={nodeData?.content || ''}
                  chapterId={activeNode.chapterId}
                  type={activeNode.type}
                />
              )}

              {/* Live Workbook or Playground Lab View */}
              {(activeNode.type === 'workbook' || activeNode.type === 'playground') && (
                <LabEditorView
                  filePath={
                    activeNode.type === 'playground'
                      ? 'playground.cpp'
                      : nodeData?.relPath || `src/module1/${activeNode.chapterId}/${target}/${activeNode.tier}_workbook.cpp`
                  }
                  code={activeCode}
                  onChange={handleCodeChange}
                  onSave={() => handleSave()}
                  saveStatus={saveStatus}
                  output={terminalOutput}
                  runStatus={runStatus}
                  durationMs={durationMs}
                  exitCode={exitCode}
                />
              )}

              {/* Volume 3 Kernel Blueprint View */}
              {activeNode.type === 'kernel' && (
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
        context={activeCode.substring(0, 300)}
      />
    </div>
  );
}
