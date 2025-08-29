'use client';

import { useEffect, useRef, useState } from 'react';
import { PaperPlaneIcon } from '@radix-ui/react-icons';

interface ChatbotSectionProps {
  height?: string;
  title?: string;
  subtitle?: string;
  prompts?: string[];
}

export default function ChatbotSection({
  height = 'h-full',
  title = 'Hiring Assistant',
  subtitle = 'Ask questions related to the hiring process or candidate stages.',
  prompts = [
    'How many candidates are in each stage?',
    'Which candidates moved from assessment to interview?',
    'Suggest top candidates for interviews',
    'Show drop-off rate in assessment stage',
  ],
}: ChatbotSectionProps) {
  const [messages, setMessages] = useState<{ role: 'user' | 'assistant'; content: string }[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [starterPrompts, setStarterPrompts] = useState(prompts);

  const chatRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleStarterClick = (prompt: string) => {
    setMessages((prev) => [...prev, { role: 'user', content: prompt }]);
    setStarterPrompts([]);
    simulateBotResponse(prompt);
  };

  const handleSend = async () => {
    if (!input.trim()) return;
    const question = input.trim();
    setMessages((prev) => [...prev, { role: 'user', content: question }]);
    setInput('');
    setLoading(true);
    simulateBotResponse(question);
  };

  const simulateBotResponse = (prompt: string) => {
    setTimeout(() => {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `This is a sample response for: "${prompt}".` },
      ]);
      setLoading(false);
    }, 1000);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className={`relative border border-gray-200 rounded-xl bg-white shadow-md flex flex-col ${height}`}>
      <div className="px-4 pt-4 pb-2">
        <h2 className="text-lg font-semibold text-gray-800">{title}</h2>
        <p className="text-sm text-gray-500">{subtitle}</p>
      </div>

      {/* Message display */}
      <div ref={chatRef} className="flex-1 px-4 pb-4 overflow-y-auto space-y-3">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`max-w-[75%] text-gray-800 text-sm px-4 py-2 rounded-lg ${
              msg.role === 'user' ? 'ml-auto bg-gray-100 text-right' : 'bg-purple-50 text-left'
            }`}
          >
            {msg.content}
          </div>
        ))}
        {loading && <div className="text-sm text-gray-600 italic">Thinking...</div>}
      </div>

      {/* Starter Prompts */}
      {starterPrompts.length > 0 && (
        <div className="flex flex-wrap gap-2 px-4 pb-2">
          {starterPrompts.map((prompt) => (
            <button
              key={prompt}
              className="px-3 py-1 text-sm bg-gray-100 rounded-full text-gray-700 hover:bg-purple-50"
              onClick={() => handleStarterClick(prompt)}
            >
              {prompt}
            </button>
          ))}
        </div>
      )}

      {/* Input section */}
      <div className="flex items-center px-4 py-3 border-t">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask your question..."
          className="flex-1 border text-gray-600 border-gray-300 rounded-lg px-4 py-2 text-sm mr-2"
        />
        <button
          onClick={handleSend}
          disabled={loading || !input.trim()}
          className="text-purple-600 hover:text-purple-800 disabled:opacity-50"
        >
          <PaperPlaneIcon className="w-5 h-5" />
        </button>
      </div>
    </div>
  );
}