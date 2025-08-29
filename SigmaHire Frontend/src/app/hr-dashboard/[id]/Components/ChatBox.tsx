'use client';

import { useEffect, useRef, useState } from 'react';
import { PaperPlaneIcon } from '@radix-ui/react-icons';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';

interface ChatBoxProps {
  title: string;
  subtitle: string;
  prompts?: string[];
}

export default function ChatBox({
  title,
  subtitle,
  prompts = [],
}: ChatBoxProps) {
  const [query, setQuery] = useState('');
  const [chatHistory, setChatHistory] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const chatBottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (chatBottomRef.current) {
      chatBottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [chatHistory, loading]);

  const sendMessage = async (message?: string) => {
    const finalQuery = message || query.trim();
    if (!finalQuery) return;

    setChatHistory((prev) => [...prev, `🧑‍💼 ${finalQuery}`, '']);
    setQuery('');
    setLoading(true);

    try {
      const res = await fetch('http://localhost:8000/chat-stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: finalQuery }), // ✅ only query now
      });

      const reader = res.body?.getReader();
      const decoder = new TextDecoder();
      let fullAnswer = '';

      while (true) {
        const { done, value } = await reader!.read();
        if (done) break;

        fullAnswer += decoder.decode(value);
        setChatHistory((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = fullAnswer;
          return updated;
        });
      }
    } catch (error) {
      setChatHistory((prev) => [...prev.slice(0, -1), '⚠️ Error fetching response.']);
    }

    setLoading(false);
  };

  return (
    <div className="relative h-[90vh] flex flex-col bg-white rounded-xl shadow overflow-hidden">

      {/* Header */}
      <div className="p-4 border-b">
        <h2 className="text-md font-semibold text-gray-800">{title}</h2>
        <p className="text-sm text-gray-600">{subtitle}</p>

        {/* Prompt buttons */}
        <div className="flex flex-wrap gap-2 mt-2">
          {prompts.map((p, i) => (
            <button
              key={i}
              onClick={() => sendMessage(p)}
              className="text-xs text-gray-800 bg-gray-100 hover:bg-gray-200 px-3 py-1 rounded-full"
            >
              {p}
            </button>
          ))}
        </div>
      </div>

      {/* Chat history (scrollable) */}
      <div className="flex-1 overflow-y-auto bg-gray-50 p-4 space-y-3">
        {chatHistory.map((msg, i) => (
          <div key={i} className={`mb-1 ${msg.startsWith('🧑‍💼') ? 'text-right' : 'text-left'}`}>
            {msg.startsWith('🧑‍💼') ? (
              <div className="inline-block px-4 py-2 rounded-lg bg-purple-100 text-purple-800 max-w-[80%] ml-auto whitespace-pre-wrap break-words">
                {msg.replace(/^🧑‍💼 /, '')}
              </div>
            ) : (
              <div className="bg-white text-gray-800 max-w-full w-full prose prose-sm prose-p:mb-2 prose-li:mb-1 prose-hr:my-2 prose-h2:mb-2">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeRaw]}
                  components={{
                    hr: () => <hr className="my-2 border-t border-gray-300" />,
                    h1: ({ node, ...props }) => <h1 className="text-xl font-bold mb-2" {...props} />,
                    h2: ({ node, ...props }) => <h2 className="text-lg font-semibold mb-1" {...props} />,
                    ul: ({ node, ...props }) => <ul className="list-disc list-inside pl-4" {...props} />,
                    li: ({ node, ...props }) => <li className="mb-1" {...props} />,
                    p: ({ node, ...props }) => <p className="mb-2" {...props} />,
                  }}
                >
                  {msg}
                </ReactMarkdown>
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="text-sm italic text-gray-500">Assistant is typing...</div>
        )}
        <div ref={chatBottomRef} />
      </div>

      {/* Fixed input at bottom */}
      <div className="border-t px-4 py-3 bg-white sticky bottom-0">
        <div className="flex items-center border rounded-lg overflow-hidden">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
            className="flex-1 px-4 py-2 text-sm text-gray-700 focus:outline-none"
            placeholder="Ask your question..."
          />
          <button
            onClick={() => sendMessage()}
            className="px-4 py-2 bg-purple-500 text-white hover:bg-purple-600 transition flex items-center gap-1"
          >
            <PaperPlaneIcon className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
