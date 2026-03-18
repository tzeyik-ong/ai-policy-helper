export interface Citation {
  title: string;
  section?: string;
}

export interface Chunk {
  title: string;
  section?: string;
  text: string;
}

export interface Message {
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  chunks?: Chunk[];
  cached?: boolean;
  pii_redacted?: boolean;
  llm_fallback?: boolean;
  llm_fallback_provider?: string;
}

export interface AskResponse {
  query: string;
  answer: string;
  citations: Citation[];
  chunks: Chunk[];
  metrics: Record<string, number>;
  cached: boolean;
  pii_redacted: boolean;
  llm_fallback: boolean;
  llm_fallback_provider: string;
}

export interface MetricsData {
  total_docs: number;
  total_chunks: number;
  query_count: number;
  avg_retrieval_latency_ms: number;
  avg_generation_latency_ms: number;
  embedding_model: string;
  llm_model: string;
}

export type StreamEvent =
  | { type: 'citations'; citations: Citation[]; chunks: Chunk[]; pii_redacted: boolean }
  | { type: 'token'; content: string }
  | { type: 'done'; metrics: Record<string, number>; llm_fallback?: boolean; llm_fallback_provider?: string }
  | { type: 'error'; message: string };

export interface StreamHandlers {
  onCitations: (citations: Citation[], chunks: Chunk[], pii_redacted: boolean) => void;
  onToken: (token: string) => void;
  onDone: (metrics: Record<string, number>, llm_fallback: boolean, llm_fallback_provider: string) => void;
  onError: (message: string) => void;
}
