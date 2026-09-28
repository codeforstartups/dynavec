export interface Metrics {
  total: number;
  qpm: number;
  p50: number;
  p95: number;
  p99: number;
  cache_hit_rate: number | null;
  cache_hits: number;
  cache_total: number;
  error_rate: number;
  avg_results: number;
  op_mix: Record<string, number>;
  namespaces: Record<string, number>;
  histogram: number[];
  bucket_width_s: number;
  window_start: number;
  window_seconds: number;
}

export interface TraceEvent {
  id: string;
  ts: number;
  op: string;
  namespace: string;
  latency_ms: number;
  n_results: number;
  top_k: number | null;
  cache_hit: boolean | null;
  filtered: boolean;
  rescore: string | null;
  rerank: string | null;
  score_top: number | null;
  score_mean: number | null;
  status: string;
  error: string | null;
  query_preview: string | null;
  eval_faithfulness?: number | null;
  eval_relevance?: number | null;
  eval_recall?: number | null;
  eval_mrr?: number | null;
  eval_ndcg?: number | null;
}

export interface TraceFilters {
  op?: string;
  status?: string;
  namespace?: string;
}

export interface ResourcesInfo {
  table: string;
  table_status: string;
  table_item_count: number;
  vector_bucket: string;
  index: string;
  dimension: number;
  distance_metric: string;
  namespaces: string[];
}

export interface GraphNode {
  id: string;
  [key: string]: any;
}

export interface GraphLink {
  source: string;
  target: string;
  label: string;
}

export interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}
