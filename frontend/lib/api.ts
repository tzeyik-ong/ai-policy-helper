import type { AskResponse, MetricsData, StreamEvent, StreamHandlers } from './types';

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

async function handleResponse<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let detail = `HTTP ${r.status}`;
    try {
      const body = await r.json();
      detail = body.detail ?? body.message ?? JSON.stringify(body);
    } catch {
      detail = (await r.text()) || detail;
    }
    throw new Error(detail);
  }
  return r.json() as Promise<T>;
}

export async function apiAsk(query: string, k: number = 4): Promise<AskResponse> {
  const r = await fetch(`${API_BASE}/api/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, k }),
  });
  return handleResponse<AskResponse>(r);
}

export async function apiAskStream(
  query: string,
  k: number = 4,
  handlers: StreamHandlers,
): Promise<void> {
  let r: Response;
  try {
    r = await fetch(`${API_BASE}/api/ask/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, k }),
    });
  } catch (e: unknown) {
    handlers.onError(e instanceof Error ? e.message : 'Network error');
    return;
  }

  if (!r.ok || !r.body) {
    let detail = `HTTP ${r.status}`;
    try { detail = (await r.json()).detail || detail; } catch { /* ignore */ }
    handlers.onError(detail);
    return;
  }

  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      try {
        const event = JSON.parse(line.slice(6)) as StreamEvent;
        if (event.type === 'citations') {
          handlers.onCitations(event.citations, event.chunks, event.pii_redacted);
        } else if (event.type === 'token') {
          handlers.onToken(event.content);
        } else if (event.type === 'done') {
          handlers.onDone(event.metrics, event.llm_fallback ?? false, event.llm_fallback_provider ?? '');
        } else if (event.type === 'error') {
          handlers.onError(event.message);
        }
      } catch { /* malformed SSE line — skip */ }
    }
  }
}

export async function apiIngest(): Promise<{ indexed_docs: number; indexed_chunks: number }> {
  const r = await fetch(`${API_BASE}/api/ingest`, { method: 'POST' });
  return handleResponse(r);
}

export async function apiMetrics(): Promise<MetricsData> {
  const r = await fetch(`${API_BASE}/api/metrics`);
  return handleResponse<MetricsData>(r);
}

export async function apiFeedback(
  query: string,
  answer: string,
  rating: 'up' | 'down',
  comment?: string,
): Promise<void> {
  await fetch(`${API_BASE}/api/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, answer, rating, comment }),
  });
}
