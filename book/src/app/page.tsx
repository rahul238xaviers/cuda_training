'use client';

import React, { useState, useEffect, useCallback } from 'react';
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
        setTerminalOutput(`Ready to compile ${node.tier}_workbook.cpp with clang++ -std=c++20 -O3.`);
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
    }
  };

  // Compile and run active node
  const handleRun = async () => {
    setRunStatus('running');
    setTerminalOutput(`Compiling with clang++ -std=c++20 -O3...\nExecuting binary...`);
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
      } else {
        setRunStatus('error');
        const errText = result.stage === 'compilation'
          ? `[COMPILATION ERROR]:\n${result.stderr || result.stdout}`
          : `[RUNTIME ERROR (Exit ${result.exitCode})]:\n${result.stderr || result.stdout}`;
        setTerminalOutput(errText);
      }
    } catch (err: any) {
      setRunStatus('error');
      setTerminalOutput(`[EXECUTION FAILED]: ${err.message}`);
    }
  };

  // Save current code
  const handleSave = async () => {
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

      const res = await fetch('/api/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (data.success) {
        alert('File saved successfully to disk!');
      } else {
        alert(`Failed to save: ${data.error}`);
      }
    } catch (err: any) {
      alert(`Save error: ${err.message}`);
    }
  };

  const handleStartOver = async () => {
    if (activeNode.type !== 'workbook' || !activeNode.chapterId || !activeNode.tier) return;

    const confirmed = window.confirm(
      `Start over Chapter ${activeNode.chapterId} (${activeNode.tier})?\n\nThis will reset your working code back to the original exercise starter template.`
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
    }
  };

  const handleClearConsole = () => {
    setTerminalOutput('');
    setRunStatus('idle');
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
          onSave={handleSave}
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
                      : `src/module1/${activeNode.chapterId}/${target}/${activeNode.tier}_workbook.cpp`
                  }
                  code={activeCode}
                  onChange={(newCode) => setActiveCode(newCode)}
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
