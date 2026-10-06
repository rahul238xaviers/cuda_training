'use client';

import React, { useState, useEffect, useRef } from 'react';
import {
  X,
  Sparkles,
  Send,
  Bot,
  User,
  Loader2,
  RotateCcw,
  Code,
  CheckCircle2,
  AlertCircle,
  ChevronDown,
  Plus,
  Trash2,
  Copy,
  Check,
  ShieldCheck,
  ArrowRight,
  Terminal,
  FileCode,
  History,
  Layers,
  Wrench,
} from 'lucide-react';
import { marked } from 'marked';
import hljs from 'highlight.js';

import {
  TEACHER_STORAGE_KEYS,
  TEACHER_DRAWER_CONFIG,
  TEACHER_QUICK_CHIPS,
} from '@/config/teacher/constants';
import {
  getTeacherGreeting,
  TEACHER_ENFORCED_RULES_META,
} from '@/config/teacher/prompts';
import { TEACHER_SKILLS } from '@/config/teacher/skills';

export interface TeacherSession {
  id: string;
  conversationId?: string | null;
  title: string;
  chapterId?: string;
  createdAt: number;
  updatedAt: number;
  messages: Message[];
}

export interface TeacherDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  chapterId?: string;
  context?: string;
  activeNode?: any;
  activeCode?: string;
  runOutput?: string;
  runStatus?: string;
  nodeData?: any;
  onApplyCode?: (code: string) => void;
}

export interface Message {
  role: 'user' | 'assistant';
  content: string;
}

const createDefaultGreeting = (chapterId?: string): Message => ({
  role: 'assistant',
  content: getTeacherGreeting(chapterId),
});

export function TeacherDrawer({
  isOpen,
  onClose,
  chapterId,
  context,
  activeNode,
  activeCode,
  runOutput,
  runStatus,
  nodeData,
  onApplyCode,
}: TeacherDrawerProps) {
  const [sessions, setSessions] = useState<TeacherSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string>('');
  const [isSessionDropdownOpen, setIsSessionDropdownOpen] = useState(false);
  const [showContextInspector, setShowContextInspector] = useState(false);
  const [inputPrompt, setInputPrompt] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [drawerWidth, setDrawerWidth] = useState<number>(TEACHER_DRAWER_CONFIG.DEFAULT_WIDTH_PX);
  const [isResizing, setIsResizing] = useState(false);
  const [feedbackMsg, setFeedbackMsg] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Load width & sessions from localStorage
  useEffect(() => {
    try {
      const savedWidth = localStorage.getItem(TEACHER_STORAGE_KEYS.DRAWER_WIDTH);
      if (savedWidth) {
        const parsed = parseInt(savedWidth, 10);
        if (
          !isNaN(parsed) &&
          parsed >= TEACHER_DRAWER_CONFIG.MIN_WIDTH_PX &&
          parsed <= TEACHER_DRAWER_CONFIG.MAX_WIDTH_PX
        ) {
          setDrawerWidth(parsed);
        }
      }

      const rawSessions = localStorage.getItem(TEACHER_STORAGE_KEYS.SESSIONS_V2);
      const savedActiveId = localStorage.getItem(TEACHER_STORAGE_KEYS.ACTIVE_SESSION_ID);
      let loaded: TeacherSession[] = [];
      if (rawSessions) {
        loaded = JSON.parse(rawSessions);
      }

      if (!loaded || loaded.length === 0) {
        // Check legacy storage
        const legacyConvId = localStorage.getItem(TEACHER_STORAGE_KEYS.LEGACY_CONV_ID);
        const legacyMsgs = localStorage.getItem(TEACHER_STORAGE_KEYS.LEGACY_MESSAGES);
        const initialMsgs = legacyMsgs ? JSON.parse(legacyMsgs) : [createDefaultGreeting(chapterId)];
        const initialSession: TeacherSession = {
          id: `sess_${Date.now()}`,
          conversationId: legacyConvId || null,
          title: `Chapter ${chapterId || '1.1'} Mentoring`,
          chapterId: chapterId || '1.1',
          createdAt: Date.now(),
          updatedAt: Date.now(),
          messages: Array.isArray(initialMsgs) && initialMsgs.length > 0 ? initialMsgs : [createDefaultGreeting(chapterId)],
        };
        loaded = [initialSession];
      }

      setSessions(loaded);
      const active = loaded.find((s) => s.id === savedActiveId) || loaded[0];
      setCurrentSessionId(active.id);
    } catch {
      const fallback: TeacherSession = {
        id: `sess_${Date.now()}`,
        conversationId: null,
        title: `Chapter ${chapterId || '1.1'} Mentoring`,
        chapterId: chapterId || '1.1',
        createdAt: Date.now(),
        updatedAt: Date.now(),
        messages: [createDefaultGreeting(chapterId)],
      };
      setSessions([fallback]);
      setCurrentSessionId(fallback.id);
    }
  }, [chapterId]);

  // Click outside to close sessions dropdown
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setIsSessionDropdownOpen(false);
      }
    };
    if (isSessionDropdownOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isSessionDropdownOpen]);

  // Handle Resizing via dragging left border
  const handleResizeStart = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
    const startX = e.clientX;
    const startWidth = drawerWidth;

    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';

    const handleMouseMove = (moveEvent: MouseEvent) => {
      const delta = startX - moveEvent.clientX;
      const minWidth = TEACHER_DRAWER_CONFIG.MIN_WIDTH_PX;
      const maxWidth = Math.min(
        window.innerWidth * TEACHER_DRAWER_CONFIG.MAX_VIEWPORT_RATIO,
        TEACHER_DRAWER_CONFIG.MAX_WIDTH_PX
      );
      const newWidth = Math.max(minWidth, Math.min(maxWidth, startWidth + delta));
      setDrawerWidth(newWidth);
    };

    const handleMouseUp = () => {
      setIsResizing(false);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      setDrawerWidth((w) => {
        try {
          localStorage.setItem(TEACHER_STORAGE_KEYS.DRAWER_WIDTH, w.toString());
        } catch {}
        return w;
      });
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  };

  // Helper to persist updated sessions
  const saveSessions = (updated: TeacherSession[], activeId?: string) => {
    setSessions(updated);
    if (activeId) setCurrentSessionId(activeId);
    try {
      localStorage.setItem(TEACHER_STORAGE_KEYS.SESSIONS_V2, JSON.stringify(updated));
      if (activeId) localStorage.setItem(TEACHER_STORAGE_KEYS.ACTIVE_SESSION_ID, activeId);
    } catch {}
  };

  const currentSession = sessions.find((s) => s.id === currentSessionId) || sessions[0];
  const messages = currentSession?.messages || [];
  const conversationId = currentSession?.conversationId || null;

  // Auto-scroll
  useEffect(() => {
    if (isOpen) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isOpen, isLoading]);

  if (!isOpen) return null;

  // Session Handlers
  const handleCreateNewSession = () => {
    const newSession: TeacherSession = {
      id: `sess_${Date.now()}`,
      conversationId: null,
      title: `Chapter ${activeNode?.chapterId || chapterId || '1.1'} Session`,
      chapterId: activeNode?.chapterId || chapterId || '1.1',
      createdAt: Date.now(),
      updatedAt: Date.now(),
      messages: [createDefaultGreeting(activeNode?.chapterId || chapterId)],
    };
    const updated = [newSession, ...sessions];
    saveSessions(updated, newSession.id);
    setIsSessionDropdownOpen(false);
  };

  const handleSelectSession = (id: string) => {
    setCurrentSessionId(id);
    setIsSessionDropdownOpen(false);
    try {
      localStorage.setItem(TEACHER_STORAGE_KEYS.ACTIVE_SESSION_ID, id);
    } catch {}
  };

  const handleDeleteSession = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    const filtered = sessions.filter((s) => s.id !== id);
    if (filtered.length === 0) {
      handleCreateNewSession();
    } else {
      const nextActiveId = currentSessionId === id ? filtered[0].id : currentSessionId;
      saveSessions(filtered, nextActiveId);
    }
  };

  // Click delegation inside message history for "Apply to Editor" and "Copy"
  const handleHistoryClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const target = e.target as HTMLElement;

    // Apply to Editor
    const applyBtn = target.closest('.apply-code-btn') as HTMLElement;
    if (applyBtn && applyBtn.dataset.code) {
      const code = decodeURIComponent(applyBtn.dataset.code);
      if (onApplyCode) {
        onApplyCode(code);
        setFeedbackMsg('✓ Code successfully applied to Monaco Editor!');
        const originalText = applyBtn.innerHTML;
        applyBtn.innerHTML = '<span>✓ Applied!</span>';
        setTimeout(() => {
          applyBtn.innerHTML = originalText;
          setFeedbackMsg(null);
        }, 2500);
      }
      return;
    }

    // Copy to clipboard
    const copyBtn = target.closest('.copy-btn') as HTMLElement;
    if (copyBtn && copyBtn.dataset.code) {
      const code = decodeURIComponent(copyBtn.dataset.code);
      navigator.clipboard.writeText(code);
      const originalText = copyBtn.innerText;
      copyBtn.innerText = 'Copied!';
      setTimeout(() => {
        copyBtn.innerText = originalText;
      }, 1500);
    }
  };

  const handleSend = async (customPrompt?: string) => {
    const promptToSend = (customPrompt || inputPrompt).trim();
    if (!promptToSend || isLoading || !currentSession) return;

    const userMsg: Message = { role: 'user', content: promptToSend };
    const updatedMessages = [...messages, userMsg];

    // Auto-update session title from first question if default
    let newTitle = currentSession.title;
    if (messages.filter((m) => m.role === 'user').length === 0) {
      newTitle = promptToSend.length > 32 ? promptToSend.substring(0, 32) + '...' : promptToSend;
    }

    const updatedSession: TeacherSession = {
      ...currentSession,
      title: newTitle,
      updatedAt: Date.now(),
      messages: updatedMessages,
    };

    const newSessions = sessions.map((s) => (s.id === currentSession.id ? updatedSession : s));
    saveSessions(newSessions);

    if (!customPrompt) {
      setInputPrompt('');
    }
    setIsLoading(true);

    const fullContext = {
      chapterId: activeNode?.chapterId || chapterId || '1.1',
      title: nodeData?.title || '',
      tier: activeNode?.tier || '',
      mode: activeNode?.type || 'workbook',
      activeFile: activeNode?.filename || (activeNode?.tier ? `${activeNode.tier}_workbook.cpp` : 'playground.cpp'),
      activeCode: activeCode || context || '',
      runOutput: runOutput || '',
      runStatus: runStatus || 'idle',
    };

    try {
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: promptToSend,
          conversationId,
          context: fullContext,
        }),
      });
      const data = await res.json();

      const assistantMsg: Message = {
        role: 'assistant',
        content: data.response || 'No response returned from Antigravity.',
      };

      const finalSession: TeacherSession = {
        ...updatedSession,
        conversationId: data.conversationId || updatedSession.conversationId || null,
        updatedAt: Date.now(),
        messages: [...updatedMessages, assistantMsg],
      };

      const finalSessions = sessions.map((s) => (s.id === currentSession.id ? finalSession : s));
      saveSessions(finalSessions);
    } catch (err: any) {
      const errMsg: Message = {
        role: 'assistant',
        content: `Error calling Antigravity: ${err.message}`,
      };
      const finalSession: TeacherSession = {
        ...updatedSession,
        updatedAt: Date.now(),
        messages: [...updatedMessages, errMsg],
      };
      const finalSessions = sessions.map((s) => (s.id === currentSession.id ? finalSession : s));
      saveSessions(finalSessions);
    } finally {
      setIsLoading(false);
    }
  };

  // Find latest code from assistant for quick "Apply" chip
  const latestAssistantMsg = [...messages].reverse().find((m) => m.role === 'assistant');
  const codeMatch = latestAssistantMsg?.content.match(/```(?:cpp|cu|cuda|c\+\+)?\n([\s\S]*?)```/);
  const latestCodeSnippet = codeMatch ? codeMatch[1].trim() : null;

  const lineCount = activeCode ? activeCode.split('\n').length : 0;
  const activeFileName = activeNode?.filename || (activeNode?.tier ? `${activeNode.tier}_workbook.cpp` : 'playground.cpp');

  return (
    <aside
      style={{ width: `${drawerWidth}px` }}
      className="teacher-drawer-aside fixed inset-y-0 right-0 bg-[#131720] border-l border-[#21262d] shadow-2xl z-50 flex flex-col select-none"
    >
      {/* Draggable Resize Handle on Left Border */}
      <div
        onMouseDown={handleResizeStart}
        title="Drag left/right to resize Ask Teacher panel"
        className={`teacher-resize-handle absolute left-0 top-0 bottom-0 w-2.5 -translate-x-1/2 cursor-col-resize z-50 group flex items-center justify-center transition-colors ${
          isResizing ? 'active bg-sky-500/30' : 'hover:bg-sky-500/20'
        }`}
      >
        <div
          className={`teacher-resize-indicator w-1 rounded-full transition-all ${
            isResizing ? 'h-24 bg-sky-400 w-1.5' : 'h-12 bg-slate-500/60 group-hover:h-20 group-hover:bg-sky-400 group-hover:w-1.5'
          }`}
        />
      </div>

      {/* Drawer Header */}
      <div className="teacher-drawer-header h-14 px-3 bg-[#161b24] border-b border-[#21262d] flex items-center justify-between shrink-0 gap-2">
        {/* Left: App Logo & Session Dropdown Selector */}
        <div className="flex items-center gap-2 min-w-0 flex-1 relative" ref={dropdownRef}>
          <div className="w-7 h-7 rounded-lg bg-[#21262d] text-indigo-400 flex items-center justify-center border border-[#30363d] shrink-0">
            <Sparkles className="w-4 h-4 text-indigo-400" />
          </div>

          {/* Session Trigger Button */}
          <button
            onClick={() => setIsSessionDropdownOpen(!isSessionDropdownOpen)}
            className="teacher-session-btn flex items-center gap-1.5 px-2 py-1 rounded-md bg-[#1e2533] hover:bg-[#273245] border border-[#2d3748] text-left transition-colors cursor-pointer truncate max-w-[200px]"
            title="Switch or manage conversation sessions"
          >
            <div className="truncate">
              <div className="text-[11px] font-bold text-slate-100 truncate flex items-center gap-1">
                <span className="truncate">{currentSession?.title || 'Active Session'}</span>
              </div>
              <div className="text-[9px] text-slate-400 truncate">
                {sessions.length} {sessions.length === 1 ? 'session' : 'sessions'} • Chapter {currentSession?.chapterId || '1.1'}
              </div>
            </div>
            <ChevronDown className={`w-3 h-3 text-slate-400 shrink-0 transition-transform ${isSessionDropdownOpen ? 'rotate-180' : ''}`} />
          </button>

          {/* Sessions Dropdown Menu */}
          {isSessionDropdownOpen && (
            <div className="teacher-session-dropdown absolute left-0 top-12 w-72 bg-[#161d2b] border border-[#2d3748] rounded-lg shadow-2xl py-1.5 z-50 text-xs animate-in fade-in zoom-in-95 duration-100">
              <div className="px-3 py-1 border-b border-[#242e42] flex items-center justify-between">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Past Sessions</span>
                <button
                  onClick={handleCreateNewSession}
                  className="teacher-session-new-btn flex items-center gap-1 px-2 py-0.5 rounded bg-sky-600 hover:bg-sky-500 text-white text-[10px] font-semibold transition-colors cursor-pointer"
                >
                  <Plus className="w-3 h-3" />
                  New Chat
                </button>
              </div>

              <div className="max-h-60 overflow-y-auto divide-y divide-[#1e2638] py-1">
                {sessions.map((sess) => {
                  const isActive = sess.id === currentSessionId;
                  const dateStr = new Date(sess.updatedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                  return (
                    <div
                      key={sess.id}
                      onClick={() => handleSelectSession(sess.id)}
                      className={`teacher-session-item px-3 py-2 flex items-center justify-between gap-2 hover:bg-[#20293d] transition-colors cursor-pointer ${
                        isActive ? 'bg-[#25324c] border-l-2 border-sky-400 font-medium' : ''
                      }`}
                    >
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-1.5">
                          <span className={`text-[11px] truncate ${isActive ? 'text-sky-300 font-bold' : 'text-slate-200'}`}>
                            {sess.title}
                          </span>
                        </div>
                        <div className="flex items-center gap-2 text-[9.5px] text-slate-400 mt-0.5">
                          <span>Ch {sess.chapterId || '1.1'}</span>
                          <span>•</span>
                          <span>{sess.messages.length} msgs</span>
                          <span>•</span>
                          <span>{dateStr}</span>
                        </div>
                      </div>

                      <button
                        onClick={(e) => handleDeleteSession(e, sess.id)}
                        title="Delete Session"
                        className="p-1 rounded text-slate-500 hover:text-rose-400 hover:bg-rose-500/10 transition-colors cursor-pointer shrink-0"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Right Header Actions */}
        <div className="flex items-center gap-1 shrink-0">
          <button
            onClick={handleCreateNewSession}
            title="Start New Session"
            className="teacher-header-btn p-1.5 rounded-md text-slate-400 hover:text-slate-100 hover:bg-[#21262d] transition-colors cursor-pointer"
          >
            <Plus className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={onClose}
            title="Close Drawer"
            className="teacher-header-btn p-1.5 rounded-md text-slate-400 hover:text-slate-100 hover:bg-[#21262d] transition-colors cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Workspace Context & Rules Strip */}
      <div className="teacher-context-strip px-3 py-1.5 bg-[#0d1117] border-b border-[#21262d] flex items-center justify-between text-[10.5px] font-mono text-slate-400 shrink-0 gap-2">
        <div className="flex items-center gap-2 truncate">
          <span className="flex items-center gap-1 text-sky-400 font-semibold truncate">
            <Code className="w-3 h-3 text-sky-400 shrink-0" />
            {activeFileName}
          </span>
          <span className="text-slate-600">•</span>
          <span className="text-slate-400">{lineCount} lines</span>
        </div>

        <div className="flex items-center gap-1.5 shrink-0">
          {runStatus === 'success' && (
            <span className="flex items-center gap-1 text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/20 text-[9.5px]">
              <CheckCircle2 className="w-2.5 h-2.5" /> Passed
            </span>
          )}
          {runStatus === 'error' && (
            <span className="flex items-center gap-1 text-rose-400 bg-rose-500/10 px-1.5 py-0.5 rounded border border-rose-500/20 text-[9.5px]">
              <AlertCircle className="w-2.5 h-2.5" /> Failed
            </span>
          )}
          {runStatus === 'running' && (
            <span className="flex items-center gap-1 text-amber-400 animate-pulse text-[9.5px]">
              <Loader2 className="w-2.5 h-2.5 animate-spin" /> Compiling
            </span>
          )}

          {/* Context & Rules Toggle */}
          <button
            onClick={() => setShowContextInspector(!showContextInspector)}
            className="teacher-context-toggle-btn flex items-center gap-1 px-1.5 py-0.5 rounded text-[9.5px] bg-[#161d2b] hover:bg-[#1e273a] text-slate-300 border border-[#253043] transition-colors cursor-pointer font-sans"
            title="Inspect workspace context, active prompts and AGENTS.md rules"
          >
            <ShieldCheck className="w-3 h-3 text-indigo-400" />
            <span>Rules & Context</span>
            <ChevronDown className={`w-2.5 h-2.5 transition-transform ${showContextInspector ? 'rotate-180' : ''}`} />
          </button>
        </div>
      </div>

      {/* Collapsible Context & Rules Inspector */}
      {showContextInspector && (
        <div className="teacher-context-inspector p-3 bg-[#0a0e17] border-b border-[#21262d] text-xs space-y-2.5 shrink-0 max-h-64 overflow-y-auto animate-in fade-in duration-100">
          <div className="flex items-center justify-between text-[11px] font-semibold text-slate-200">
            <span className="flex items-center gap-1.5">
              <ShieldCheck className="w-3.5 h-3.5 text-indigo-400" />
              Active System Context & Pedagogy (AGENTS.md)
            </span>
            <span className="text-[10px] text-slate-400 font-mono">Real-time Injection</span>
          </div>

          <div className="grid grid-cols-2 gap-2 text-[10px]">
            <div className="teacher-context-box p-2 rounded bg-[#121824] border border-[#20293a]">
              <div className="text-slate-400 font-semibold mb-1 flex items-center gap-1">
                <FileCode className="w-3 h-3 text-sky-400" />
                Active Buffer
              </div>
              <div className="text-slate-300 font-mono">{activeFileName}</div>
              <div className="text-slate-400">{lineCount} lines ({activeCode?.length || 0} bytes)</div>
            </div>

            <div className="teacher-context-box p-2 rounded bg-[#121824] border border-[#20293a]">
              <div className="text-slate-400 font-semibold mb-1 flex items-center gap-1">
                <Terminal className="w-3 h-3 text-amber-400" />
                Compiler Diagnostics
              </div>
              <div className="text-slate-300 capitalize">{runStatus || 'idle'}</div>
              <div className="text-slate-400 truncate">{runOutput ? `${runOutput.substring(0, 30)}...` : 'No errors'}</div>
            </div>
          </div>

          {/* Enforced Pedagogy Rules */}
          <div className="teacher-context-box p-2 rounded bg-[#121824] border border-[#20293a] space-y-1 text-[10px]">
            <div className="text-slate-400 font-semibold flex items-center gap-1">
              <ShieldCheck className="w-3 h-3 text-emerald-400" />
              Enforced AGENTS.md Directives:
            </div>
            <div className="flex flex-wrap gap-1">
              {TEACHER_ENFORCED_RULES_META.map((rule) => (
                <span
                  key={rule.id}
                  className="teacher-context-rule-badge px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                  title={rule.description}
                >
                  {rule.badge}
                </span>
              ))}
            </div>
          </div>

          {/* Registered Skills */}
          <div className="teacher-context-box p-2 rounded bg-[#121824] border border-[#20293a] space-y-1 text-[10px]">
            <div className="text-slate-400 font-semibold flex items-center gap-1">
              <Wrench className="w-3 h-3 text-sky-400" />
              Active Skills & Capabilities:
            </div>
            <div className="grid grid-cols-1 gap-1">
              {TEACHER_SKILLS.map((skill) => (
                <div key={skill.id} className="flex items-center justify-between text-slate-300">
                  <span className="font-mono text-[9.5px]">• {skill.name}</span>
                  <span className="text-[9px] text-slate-500 font-sans uppercase">{skill.category}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Feedback Toast if code applied */}
      {feedbackMsg && (
        <div className="px-3 py-1.5 bg-emerald-500/20 border-b border-emerald-500/30 text-emerald-300 text-[11px] font-semibold flex items-center gap-2">
          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
          <span>{feedbackMsg}</span>
        </div>
      )}

      {/* Message History */}
      <div
        onClick={handleHistoryClick}
        className="teacher-message-history flex-1 overflow-y-auto p-4 space-y-4 select-text"
      >
        {messages.map((m, idx) => (
          <div
            key={idx}
            className={`flex gap-3 text-xs leading-relaxed ${
              m.role === 'user' ? 'justify-end' : 'justify-start'
            }`}
          >
            {m.role === 'assistant' && (
              <div className="w-6 h-6 rounded-md bg-indigo-600/30 text-indigo-400 flex items-center justify-center shrink-0 border border-indigo-500/30 mt-0.5">
                <Bot className="w-3.5 h-3.5" />
              </div>
            )}
            <div
              className={`teacher-message-bubble p-3 rounded-lg max-w-[88%] ${
                m.role === 'user'
                  ? 'teacher-user-bubble bg-sky-600 text-white rounded-tr-none'
                  : 'teacher-assistant-bubble bg-[#121827] text-slate-200 border border-[#1e293b] rounded-tl-none prose-dark'
              }`}
            >
              {m.role === 'assistant' ? (
                <div
                  dangerouslySetInnerHTML={{
                    __html: marked.parse(m.content, {
                      renderer: Object.assign(new marked.Renderer(), {
                        code({ text, lang }: { text: string; lang?: string }) {
                          const language = lang && hljs.getLanguage(lang) ? lang : undefined;
                          const highlighted = language
                            ? hljs.highlight(text, { language }).value
                            : hljs.highlightAuto(text).value;
                          return `
                            <div class="my-2.5 rounded border border-[#21262d] overflow-hidden code-block-card">
                              <div class="flex items-center justify-between px-2.5 py-1 bg-[#161b24] border-b border-[#21262d] text-[10px] text-slate-400 select-none">
                                <span class="font-mono font-bold uppercase tracking-wider">${(lang || 'CODE')}</span>
                                <div class="flex items-center gap-1.5">
                                  <button class="apply-code-btn px-2 py-0.5 rounded text-[10px] bg-emerald-600/20 text-emerald-300 hover:bg-emerald-600/30 border border-emerald-500/30 transition-colors cursor-pointer flex items-center gap-1 font-semibold" data-code="${encodeURIComponent(text)}" title="Replace/update active editor code with this snippet">
                                    <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"/></svg>
                                    Apply to Editor
                                  </button>
                                  <button class="copy-btn px-2 py-0.5 rounded text-[10px] bg-[#21262d] text-slate-300 hover:text-white transition-colors cursor-pointer" data-code="${encodeURIComponent(text)}">
                                    Copy
                                  </button>
                                </div>
                              </div>
                              <pre class="p-2.5 bg-[#090d16] overflow-x-auto text-[11px] leading-5"><code class="hljs ${language || ''}">${highlighted}</code></pre>
                            </div>
                          `;
                        },
                      }),
                    }) as string,
                  }}
                />
              ) : (
                <p className="whitespace-pre-wrap">{m.content}</p>
              )}
            </div>
            {m.role === 'user' && (
              <div className="w-6 h-6 rounded-md bg-sky-600/30 text-sky-400 flex items-center justify-center shrink-0 border border-sky-500/30 mt-0.5">
                <User className="w-3.5 h-3.5" />
              </div>
            )}
          </div>
        ))}

        {/* Dynamic Execution Steps Indicator */}
        {isLoading && (
          <div className="teacher-steps-card p-3 rounded-lg border border-[#21262d] bg-[#0d121c] space-y-2 select-none text-xs ml-9">
            <div className="flex items-center justify-between text-[11px] font-semibold text-slate-300">
              <span className="flex items-center gap-1.5 text-indigo-400">
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                Antigravity Pair Programmer
              </span>
              <span className="text-[10px] text-slate-500 font-mono">AGY Session</span>
            </div>

            <div className="space-y-1.5 pt-1 text-[11px]">
              <div className="flex items-center gap-2 text-emerald-400">
                <CheckCircle2 className="w-3 h-3 text-emerald-400 shrink-0" />
                <span>Loaded {activeFileName} ({lineCount} lines)</span>
              </div>
              <div className="flex items-center gap-2 text-sky-400">
                <span className="w-2.5 h-2.5 rounded-full border-2 border-sky-400 border-t-transparent animate-spin shrink-0" />
                <span>Applying AGENTS.md rules & formulating hardware intuition...</span>
              </div>
              <div className="text-[10px] text-slate-500 pl-5">
                • Clean ASCII math • One step at a time • Memory layout first
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Quick Action Chips Bar */}
      <div className="teacher-chips-bar px-3 py-1.5 bg-[#0e131d] border-t border-[#21262d] flex items-center gap-1.5 overflow-x-auto text-[10px] shrink-0">
        {latestCodeSnippet && onApplyCode && (
          <button
            onClick={() => {
              onApplyCode(latestCodeSnippet);
              setFeedbackMsg('✓ Applied latest code snippet to editor!');
              setTimeout(() => setFeedbackMsg(null), 2500);
            }}
            disabled={isLoading}
            className="teacher-chip-btn px-2.5 py-1 rounded bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-300 border border-emerald-500/30 whitespace-nowrap transition-colors cursor-pointer font-semibold flex items-center gap-1"
          >
            ✨ Apply Solution to Editor
          </button>
        )}

        {TEACHER_QUICK_CHIPS.map((chip) => (
          <button
            key={chip.id}
            onClick={() => handleSend(chip.prompt)}
            disabled={isLoading}
            className={`teacher-chip-btn px-2.5 py-1 rounded bg-[#161d2b] hover:bg-[#1e273a] text-slate-300 border border-[#253043] whitespace-nowrap transition-colors cursor-pointer ${
              chip.colorClass ? `hover:${chip.colorClass}` : ''
            }`}
          >
            {chip.label}
          </button>
        ))}
      </div>

      {/* Input Box */}
      <div className="teacher-drawer-footer p-3 bg-[#131720] border-t border-[#21262d] shrink-0">
        <div className="relative">
          <textarea
            value={inputPrompt}
            onChange={(e) => setInputPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder="Ask about memory layouts, pointer strides, or kernels..."
            rows={2}
            className="teacher-textarea w-full bg-[#0d1117] text-xs text-slate-100 placeholder-slate-500 p-2.5 pr-10 rounded-lg border border-[#30363d] focus:outline-none focus:border-[#1f6feb]/50 resize-none transition-colors"
          />
          <button
            onClick={() => handleSend()}
            disabled={isLoading || !inputPrompt.trim()}
            className="teacher-send-btn absolute right-2 bottom-2 p-1.5 rounded-md bg-[#1f6feb] text-white hover:bg-[#388bfd] disabled:opacity-40 disabled:cursor-not-allowed transition-colors cursor-pointer"
          >
            <Send className="w-3.5 h-3.5" />
          </button>
        </div>
        <p className="mt-1 text-[10px] text-slate-500 text-center">
          Press Enter to send, Shift+Enter for new line
        </p>
      </div>
    </aside>
  );
}
