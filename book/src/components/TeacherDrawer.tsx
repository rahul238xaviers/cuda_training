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
      content: `Hello! I am Antigravity, your patient C++ and CUDA teacher. I adhere strictly to our principles: zero LaTeX, plain text intuition, and step-by-step hardware models.\n\nWhat would you like to explore about Chapter ${chapterId || '1.1'}?`,
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
    <aside className="fixed inset-y-0 right-0 w-96 bg-[#0c101a] border-l border-[#1e293b] shadow-2xl z-50 flex flex-col select-none">
      {/* Drawer Header */}
      <div className="h-14 px-4 bg-[#0f1524] border-b border-[#1e293b] flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-indigo-500/20 text-indigo-400 flex items-center justify-center border border-indigo-500/30">
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
          className="p-1 rounded-md text-slate-400 hover:text-slate-100 hover:bg-[#1a2333] transition-colors"
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
      <div className="p-3 bg-[#0f1524] border-t border-[#1e293b]">
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
            className="w-full bg-[#080b11] text-xs text-slate-100 placeholder-slate-500 p-2.5 pr-10 rounded-lg border border-[#1e293b] focus:outline-none focus:border-indigo-500/50 resize-none transition-colors"
          />
          <button
            onClick={handleSend}
            disabled={isLoading || !inputPrompt.trim()}
            className="absolute right-2 bottom-2 p-1.5 rounded-md bg-indigo-600 text-white hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
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
