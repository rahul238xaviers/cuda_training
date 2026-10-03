'use client';

import React, { useState } from 'react';
import { X, Sparkles, Send, Bot, User, Loader2 } from 'lucide-react';
import { marked } from 'marked';
import hljs from 'highlight.js';

interface TeacherDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  chapterId?: string;
  context?: string;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export function TeacherDrawer({ isOpen, onClose, chapterId, context }: TeacherDrawerProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: `Welcome to the lab! I'm Antigravity, your pair programming companion for C++ memory systems and CUDA kernel development.\n\nWe're currently exploring **Chapter ${chapterId || '1.1'}**. Whether you want to visualize memory layouts, reason through cache line alignment, or walk through an exercise step-by-step, feel free to ask anytime.\n\nWhat would you like to dive into?`,
    },
  ]);
  const [inputPrompt, setInputPrompt] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  if (!isOpen) return null;

  const handleSend = async () => {
    const prompt = inputPrompt.trim();
    if (!prompt || isLoading) return;

    const userMsg: Message = { role: 'user', content: prompt };
    setMessages((prev) => [...prev, userMsg]);
    setInputPrompt('');
    setIsLoading(true);

    try {
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          context: `Active Chapter: ${chapterId || '1.1'}. Context code/concept: ${context || ''}`,
        }),
      });
      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.response || 'No response returned from Antigravity.',
        },
      ]);
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `Error calling Antigravity: ${err.message}`,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <aside className="fixed inset-y-0 right-0 w-96 bg-[#131720] border-l border-[#21262d] shadow-2xl z-50 flex flex-col select-none">
      {/* Drawer Header */}
      <div className="h-14 px-4 bg-[#161b24] border-b border-[#21262d] flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-[#21262d] text-indigo-400 flex items-center justify-center border border-[#30363d]">
            <Sparkles className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-xs font-bold text-slate-100 flex items-center gap-1.5">
              Antigravity AI Teacher
            </h3>
            <p className="text-[10px] text-slate-400">Pair Programmer & Mentor</p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="p-1 rounded-md text-slate-400 hover:text-slate-100 hover:bg-[#21262d] transition-colors cursor-pointer"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Message History */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 select-text">
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
              className={`p-3 rounded-lg max-w-[85%] ${
                m.role === 'user'
                  ? 'bg-sky-600 text-white rounded-tr-none'
                  : 'bg-[#121827] text-slate-200 border border-[#1e293b] rounded-tl-none prose-dark'
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
                          return `<pre class="my-2 p-2.5 rounded bg-[#090d16] border border-[#1e293b] overflow-x-auto text-[11px]"><code class="hljs ${language || ''}">${highlighted}</code></pre>`;
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

        {isLoading && (
          <div className="flex items-center gap-2 text-xs text-slate-400 pl-9">
            <Loader2 className="w-3.5 h-3.5 animate-spin text-indigo-400" />
            <span>Thinking with Antigravity...</span>
          </div>
        )}
      </div>

      {/* Input Box */}
      <div className="p-3 bg-[#131720] border-t border-[#21262d]">
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
            placeholder="Ask about pointers, cache lines, or kernels..."
            rows={2}
            className="w-full bg-[#0d1117] text-xs text-slate-100 placeholder-slate-500 p-2.5 pr-10 rounded-lg border border-[#30363d] focus:outline-none focus:border-[#1f6feb]/50 resize-none transition-colors"
          />
          <button
            onClick={handleSend}
            disabled={isLoading || !inputPrompt.trim()}
            className="absolute right-2 bottom-2 p-1.5 rounded-md bg-[#1f6feb] text-white hover:bg-[#388bfd] disabled:opacity-40 disabled:cursor-not-allowed transition-colors cursor-pointer"
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
