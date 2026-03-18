'use client';
import React from 'react';
import { apiAskStream, apiFeedback } from '../lib/api';
import type { Citation, Chunk, Message } from '../lib/types';

export default function Chat() {
  const [messages, setMessages] = React.useState<Message[]>([]);
  const [q, setQ] = React.useState('');
  const [loading, setLoading] = React.useState(false);
  const [phase, setPhase] = React.useState<'retrieving' | 'generating' | 'done'>('done');
  const [feedbackMap, setFeedbackMap] = React.useState<Record<number, 'up' | 'down'>>({});
  const bottomRef = React.useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when messages change or loading indicator appears
  React.useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const send = async () => {
    if (!q.trim() || loading) return;
    const query = q.trim();
    setMessages((m) => [...m, { role: 'user', content: query }]);
    setLoading(true);
    setQ('');

    // Add empty assistant placeholder
    setMessages((m) => [...m, { role: 'assistant', content: '' }]);
    setPhase('retrieving');

    await apiAskStream(query, 4, {
      onCitations: (citations: Citation[], chunks: Chunk[], pii_redacted: boolean) => {
        setPhase('generating');
        setMessages((m) => {
          const updated = [...m];
          updated[updated.length - 1] = {
            ...updated[updated.length - 1],
            citations,
            chunks,
            pii_redacted,
          };
          return updated;
        });
      },
      onToken: (token: string) => {
        setMessages((m) => {
          const updated = [...m];
          const last = updated[updated.length - 1];
          updated[updated.length - 1] = { ...last, content: last.content + token };
          return updated;
        });
      },
      onDone: (_metrics: Record<string, number>, llm_fallback: boolean, llm_fallback_provider: string) => {
        setPhase('done');
        setLoading(false);
        if (llm_fallback) {
          setMessages((m) => {
            const updated = [...m];
            updated[updated.length - 1] = {
              ...updated[updated.length - 1],
              llm_fallback: true,
              llm_fallback_provider,
            };
            return updated;
          });
        }
      },
      onError: (message: string) => {
        setMessages((m) => {
          const updated = [...m];
          updated[updated.length - 1] = { role: 'assistant', content: `Error: ${message}` };
          return updated;
        });
        setPhase('done');
        setLoading(false);
      },
    });
  };

  const sendFeedback = async (msg: Message, rating: 'up' | 'down') => {
    if (!msg.content) return;
    const idx = messages.indexOf(msg);
    setFeedbackMap((prev) => ({ ...prev, [idx]: rating }));
    const userMsg = idx > 0 ? messages[idx - 1] : null;
    await apiFeedback(userMsg?.content ?? '', msg.content, rating);
  };

  const handleKey = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) send();
  };

  return (
    <div className="card">
      <h2>Chat</h2>

      {/* Message list */}
      <div
        role="log"
        aria-label="Conversation history"
        aria-live="polite"
        style={{
          maxHeight: 420,
          overflowY: 'auto',
          padding: 8,
          border: '1px solid #eee',
          borderRadius: 8,
          marginBottom: 12,
        }}
      >
        {messages.length === 0 && (
          <p style={{ color: '#9ca3af', fontSize: 13, textAlign: 'center', margin: '24px 0' }}>
            Ask a question about company policy or products.
          </p>
        )}

        {messages.map((m, i) => (
          <div key={i} style={{ margin: '10px 0' }}>
            {/* Role label */}
            <div
              style={{
                fontSize: 11,
                fontWeight: 600,
                color: m.role === 'user' ? '#2563eb' : '#059669',
                marginBottom: 2,
              }}
            >
              {m.role === 'user' ? 'You' : 'Assistant'}
              {m.cached && (
                <span style={{ marginLeft: 6, color: '#9ca3af', fontWeight: 400 }}>
                  (cached)
                </span>
              )}
              {m.pii_redacted && (
                <span style={{ marginLeft: 6, color: '#f59e0b', fontWeight: 400 }}>
                  ⚠ PII redacted
                </span>
              )}
              {m.llm_fallback && (
                <span style={{ marginLeft: 6, color: '#ef4444', fontWeight: 400 }}>
                  ⚠ {m.llm_fallback_provider?.startsWith('openrouter') ? 'OpenRouter' : m.llm_fallback_provider?.startsWith('ollama') ? 'Ollama' : 'LLM'} unavailable — stub response
                </span>
              )}
            </div>

            {/* Answer text — show phase status while streaming */}
            {m.role === 'assistant' && loading && i === messages.length - 1 && !m.content ? (
              <div style={{ color: '#6b7280', fontStyle: 'italic', fontSize: 13 }}>
                {phase === 'retrieving' ? '🔍 Retrieving sources…' : '✍️ Generating answer…'}
              </div>
            ) : (
              <div style={{ whiteSpace: 'pre-wrap' }}>{m.content}</div>
            )}

            {/* LLM fallback warning */}
            {m.llm_fallback && (
              <div
                style={{
                  marginTop: 8,
                  padding: '8px 12px',
                  background: '#fef2f2',
                  border: '1px solid #fecaca',
                  borderRadius: 6,
                  fontSize: 12,
                  color: '#b91c1c',
                }}
              >
                {m.llm_fallback_provider?.startsWith('openrouter') ? (
                  <>
                    <strong>OpenRouter failed to respond.</strong> The answer above was generated by the offline stub and may not be accurate.
                    <br />
                    To fix: check that <code>OPENROUTER_API_KEY</code> is set correctly in <code>.env</code> and that your account has credits.
                    See the <strong>README.md → LLM Providers</strong> section for instructions.
                  </>
                ) : m.llm_fallback_provider?.startsWith('ollama') ? (
                  <>
                    <strong>Ollama failed to respond.</strong> The answer above was generated by the offline stub and may not be accurate.
                    <br />
                    To fix: ensure Ollama is running (<code>ollama serve</code>) and the model is downloaded (<code>ollama pull llama3.2</code>), then retry.
                    See the <strong>README.md → LLM Providers</strong> section for instructions.
                  </>
                ) : (
                  <>
                    <strong>LLM failed to respond.</strong> The answer above was generated by the offline stub and may not be accurate.
                    <br />
                    See the <strong>README.md → LLM Providers</strong> section to configure a working LLM provider.
                  </>
                )}
              </div>
            )}

            {/* Citation badges */}
            {m.citations && m.citations.length > 0 && (
              <div style={{ marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                {m.citations.map((c, idx) => (
                  <span
                    key={idx}
                    className="badge"
                    title={c.section ? `${c.title} — ${c.section}` : c.title}
                    aria-label={`Source: ${c.title}${c.section ? `, section: ${c.section}` : ''}`}
                  >
                    {c.title.replace(/\.md$/, '')}
                  </span>
                ))}
              </div>
            )}

            {/* Expandable source chunks */}
            {m.chunks && m.chunks.length > 0 && (
              <details style={{ marginTop: 6 }}>
                <summary
                  style={{ cursor: 'pointer', fontSize: 13, color: '#6b7280' }}
                  aria-label="Expand source chunks"
                >
                  View {m.chunks.length} supporting chunk
                  {m.chunks.length > 1 ? 's' : ''}
                </summary>
                {m.chunks.map((c, idx) => (
                  <div
                    key={idx}
                    style={{
                      background: '#f9fafb',
                      border: '1px solid #e5e7eb',
                      borderRadius: 6,
                      padding: '8px 10px',
                      marginTop: 6,
                    }}
                  >
                    <div style={{ fontWeight: 600, fontSize: 12 }}>
                      {c.title.replace(/\.md$/, '')}
                      {c.section ? ` — ${c.section}` : ''}
                    </div>
                    <div
                      style={{
                        whiteSpace: 'pre-wrap',
                        fontSize: 12,
                        marginTop: 4,
                        color: '#374151',
                      }}
                    >
                      {c.text}
                    </div>
                  </div>
                ))}
              </details>
            )}

            {/* Feedback thumbs — only on completed assistant messages */}
            {m.role === 'assistant' && m.content && !loading && (
              <div style={{ marginTop: 6, display: 'flex', gap: 4 }}>
                <button
                  onClick={() => sendFeedback(m, 'up')}
                  aria-label="Thumbs up — helpful answer"
                  title="Helpful"
                  style={{
                    background: feedbackMap[i] === 'up' ? '#dcfce7' : 'none',
                    border: `1px solid ${feedbackMap[i] === 'up' ? '#16a34a' : '#d1d5db'}`,
                    borderRadius: 6,
                    padding: '2px 8px',
                    cursor: feedbackMap[i] ? 'default' : 'pointer',
                    fontSize: 13,
                  }}
                >
                  👍
                </button>
                <button
                  onClick={() => sendFeedback(m, 'down')}
                  aria-label="Thumbs down — unhelpful answer"
                  title="Not helpful"
                  style={{
                    background: feedbackMap[i] === 'down' ? '#fee2e2' : 'none',
                    border: `1px solid ${feedbackMap[i] === 'down' ? '#dc2626' : '#d1d5db'}`,
                    borderRadius: 6,
                    padding: '2px 8px',
                    cursor: feedbackMap[i] ? 'default' : 'pointer',
                    fontSize: 13,
                  }}
                >
                  👎
                </button>
              </div>
            )}
          </div>
        ))}

        <div ref={bottomRef} />
      </div>

      {/* Input row */}
      <div style={{ display: 'flex', gap: 8 }}>
        <input
          placeholder="Ask about policy or products…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={handleKey}
          disabled={loading}
          aria-label="Type your question here"
          style={{
            flex: 1,
            padding: 10,
            borderRadius: 8,
            border: '1px solid #ddd',
            opacity: loading ? 0.6 : 1,
          }}
        />
        <button
          onClick={send}
          disabled={loading || !q.trim()}
          aria-label="Send message"
          style={{
            padding: '10px 14px',
            borderRadius: 8,
            border: '1px solid #111',
            background: '#111',
            color: '#fff',
            cursor: loading || !q.trim() ? 'not-allowed' : 'pointer',
          }}
        >
          {loading ? 'Thinking…' : 'Send'}
        </button>
      </div>
    </div>
  );
}
