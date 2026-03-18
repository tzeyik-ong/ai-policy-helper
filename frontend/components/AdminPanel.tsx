'use client';
import React from 'react';
import { apiIngest, apiMetrics } from '../lib/api';
import type { MetricsData } from '../lib/types';

const STAT_FIELDS: { label: string; key: keyof MetricsData; suffix?: string }[] = [
  { label: 'Documents', key: 'total_docs' },
  { label: 'Chunks', key: 'total_chunks' },
  { label: 'Queries', key: 'query_count' },
  { label: 'Avg Retrieval', key: 'avg_retrieval_latency_ms', suffix: ' ms' },
  { label: 'Avg Generation', key: 'avg_generation_latency_ms', suffix: ' ms' },
  { label: 'Embedding Model', key: 'embedding_model' },
  { label: 'LLM Model', key: 'llm_model' },
];

export default function AdminPanel() {
  const [metrics, setMetrics] = React.useState<MetricsData | null>(null);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  const refresh = async () => {
    try {
      const m = await apiMetrics();
      setMetrics(m);
      setError(null);
    } catch (e: any) {
      setError(e.message);
    }
  };

  const ingest = async () => {
    setBusy(true);
    setError(null);
    try {
      await apiIngest();
      await refresh();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  React.useEffect(() => { refresh(); }, []);

  return (
    <div className="card">
      <h2>Admin</h2>
      <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
        <button
          onClick={ingest}
          disabled={busy}
          aria-label="Ingest sample documents into the vector store"
          style={{ padding: '8px 12px', borderRadius: 8, border: '1px solid #111', background: '#fff', cursor: busy ? 'not-allowed' : 'pointer' }}
        >
          {busy ? 'Indexing…' : 'Ingest sample docs'}
        </button>
        <button
          onClick={refresh}
          aria-label="Refresh metrics from the backend"
          style={{ padding: '8px 12px', borderRadius: 8, border: '1px solid #111', background: '#fff', cursor: 'pointer' }}
        >
          Refresh metrics
        </button>
      </div>

      {error && (
        <div role="alert" style={{ color: '#b91c1c', marginBottom: 8, fontSize: 13 }}>
          {error}
        </div>
      )}

      {metrics && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: 8 }}>
          {STAT_FIELDS.map(({ label, key, suffix }) => (
            <div
              key={key}
              style={{ background: '#f9fafb', border: '1px solid #e5e7eb', borderRadius: 8, padding: '10px 14px' }}
            >
              <div style={{ fontSize: 11, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                {label}
              </div>
              <div style={{ fontSize: 15, fontWeight: 600, marginTop: 4, wordBreak: 'break-word' }}>
                {typeof metrics[key] === 'number'
                  ? `${(metrics[key] as number).toFixed(typeof metrics[key] === 'number' && String(metrics[key]).includes('.') ? 1 : 0)}${suffix ?? ''}`
                  : String(metrics[key])}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
