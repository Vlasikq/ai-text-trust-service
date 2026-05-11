// Fetch-обёртка с auto-refresh на 401.
// Access-токен — в памяти (через AuthProvider). Refresh — в localStorage (компромисс MVP).
// Для прода после защиты — refresh в httpOnly+SameSite=Strict cookie через BFF-route.
//
// API_URL:
//   - Same-origin prod (PWA на той же VM что и API): пустая строка → запрос идёт на /api/...
//   - Локальный dev (next dev на :3000, API на YC): NEXT_PUBLIC_API_URL=https://89.169.141.35.sslip.io
//   - Cross-origin (Vercel и т.п.): NEXT_PUBLIC_API_URL=https://api.example.com

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "";

const REFRESH_KEY = "aitrust_refresh";

let accessToken: string | null = null;
const accessListeners = new Set<(t: string | null) => void>();

export function setAccessToken(token: string | null) {
  accessToken = token;
  accessListeners.forEach((l) => l(token));
}

export function getAccessToken() {
  return accessToken;
}

export function subscribeAccess(cb: (t: string | null) => void): () => void {
  accessListeners.add(cb);
  return () => {
    accessListeners.delete(cb);
  };
}

export function setRefreshToken(token: string | null) {
  if (typeof window === "undefined") return;
  if (token) localStorage.setItem(REFRESH_KEY, token);
  else localStorage.removeItem(REFRESH_KEY);
}

export function getRefreshToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(REFRESH_KEY);
}

// ── Internal: refresh access using stored refresh ─────────────

let refreshInflight: Promise<string | null> | null = null;

async function refreshAccess(): Promise<string | null> {
  if (refreshInflight) return refreshInflight;
  const refresh = getRefreshToken();
  if (!refresh) return null;

  refreshInflight = (async () => {
    try {
      const res = await fetch(`${API_URL}/auth/refresh`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh_token: refresh }),
      });
      if (!res.ok) {
        // Refresh не удался — чистим всё, заставляем переавторизоваться.
        setRefreshToken(null);
        setAccessToken(null);
        return null;
      }
      const data = await res.json();
      setAccessToken(data.access_token);
      setRefreshToken(data.refresh_token);
      return data.access_token as string;
    } finally {
      refreshInflight = null;
    }
  })();

  return refreshInflight;
}

// ── Public: typed fetch with retry on 401 ─────────────────────

export interface ApiOptions extends RequestInit {
  // Если true, не пытаемся auto-refresh (используется самим /auth/refresh, чтобы не зациклиться).
  skipAuth?: boolean;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    public detail: string,
    public body?: unknown,
  ) {
    super(`API ${status}: ${detail}`);
  }
}

async function _fetch(
  path: string,
  opts: ApiOptions,
  attemptedRefresh = false,
): Promise<Response> {
  const headers = new Headers(opts.headers);
  if (!headers.has("Content-Type") && opts.body) {
    headers.set("Content-Type", "application/json");
  }
  if (!opts.skipAuth && accessToken) {
    headers.set("Authorization", `Bearer ${accessToken}`);
  }

  const res = await fetch(`${API_URL}${path}`, { ...opts, headers });

  // 401 от защищённого endpoint'а — пробуем рефрешнуть один раз и повторить.
  if (res.status === 401 && !opts.skipAuth && !attemptedRefresh) {
    const newAccess = await refreshAccess();
    if (newAccess) return _fetch(path, opts, true);
  }
  return res;
}

export async function api<T = unknown>(
  path: string,
  opts: ApiOptions = {},
): Promise<T> {
  const res = await _fetch(path, opts);
  const text = await res.text();
  let body: unknown = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = text;
  }
  if (!res.ok) {
    const detail =
      (body && typeof body === "object" && "detail" in body
        ? String((body as { detail: unknown }).detail)
        : null) || res.statusText;
    throw new ApiError(res.status, detail, body);
  }
  return body as T;
}

// ── Domain types (subset of OpenAPI) ──────────────────────────

export type Verdict = "ai" | "human";
export type RiskLevel = "HIGH" | "MEDIUM" | "LOW";
export type AnalysisStatus = "SUCCESS" | "NO_DECISION" | "ERROR";

export interface AnalyzeResponse {
  request_id: string;
  status: AnalysisStatus;
  verdict: Verdict | null;
  confidence: number | null;
  risk_level: RiskLevel | null;
  text_length: number;
  was_truncated: boolean;
  processing_time_ms: number;
  warnings: string[];
  disclaimer: string;
  explanation?: {
    top_markers: Array<{
      feature: string;
      value: number;
      human_baseline: number;
      deviation_percent: number;
      direction: "above" | "below";
    }>;
    summary: string;
  } | null;
}

export interface UserPublic {
  id: string;
  email: string;
  role: string;
  status: string;
  created_at: string;
  last_login_at: string | null;
}

export interface TokenPair {
  access_token: string;
  refresh_token: string;
  token_type: string;
  access_expires_in: number;
  refresh_expires_in: number;
}

export interface RegisterResponse {
  user: UserPublic;
  tokens: TokenPair;
}

export interface SegmentScoreItem {
  start_sent: number;
  end_sent: number;
  start_char: number;
  end_char: number;
  prob_ai: number;
}

export interface SegmentSummary {
  n_segments: number;
  mean_prob: number | null;
  min_prob: number | null;
  max_prob: number | null;
  n_high: number;
  n_medium: number;
  n_low: number;
  transition_detected: boolean;
}

export interface SegmentScoringResponse {
  text_length: number;
  window_size: number;
  step: number;
  segments: SegmentScoreItem[];
  summary: SegmentSummary;
}

export interface AnalysisHistoryItem {
  id: string;
  request_id: string;
  status: AnalysisStatus;
  verdict: Verdict | null;
  confidence: number | null;
  risk_level: RiskLevel | null;
  detector_used: string;
  text_length: number;
  requested_at: string;
}

export interface AnalysisHistoryPage {
  items: AnalysisHistoryItem[];
  next_cursor: string | null;
}

// Detailed scan view (/me/analyses/{id}). Расширяет AnalysisHistoryItem полями,
// которые сохраняются в БД: cascade_path, inference_ms, cached, warnings, explanation.
// Текст не возвращается (privacy-by-design); was_truncated тоже отсутствует.
export interface AnalysisDetailResponse extends AnalysisHistoryItem {
  inference_ms: number;
  cached: boolean;
  cascade_path: string | null;
  warnings: string[];
  explanation: AnalyzeResponse["explanation"];
  disclaimer: string;
}

// Extract text (/api/v1/extract/text).
export interface ExtractTextResponse {
  text: string;
  source_format: "txt" | "docx" | "pdf";
  char_count: number;
  warnings: string[];
}

// Batch (/api/v1/batch).
export type BatchStatus = "PENDING" | "PROCESSING" | "COMPLETED";

export interface BatchSkippedRow {
  row_num: number;
  reason: string;
  external_id: string | null;
}

export interface BatchCreatedResponse {
  batch_id: string;
  n_accepted: number;
  n_skipped: number;
  skipped: BatchSkippedRow[];
  poll_url: string;
}

export interface BatchStatusResponse {
  batch_id: string;
  file_name: string | null;
  status: BatchStatus;
  n_total: number;
  n_completed: number;
  n_failed: number;
  detector_type: string;
  explain: boolean;
  created_at: string;
  completed_at: string | null;
  skipped: BatchSkippedRow[];
  results_csv_url: string | null;
  results_json_url: string | null;
}

export interface BatchListItem {
  batch_id: string;
  file_name: string | null;
  status: BatchStatus;
  n_total: number;
  n_completed: number;
  n_failed: number;
  created_at: string;
  completed_at: string | null;
}

export interface BatchListPage {
  items: BatchListItem[];
  next_cursor: string | null;
}

// Stats (/me/stats). Daily zero-padded на бэке.
export type StatsPeriod = "7d" | "30d" | "90d" | "all";

export interface DailyBucket {
  date: string; // YYYY-MM-DD
  total: number;
  ai: number;
  human: number;
}

export interface StatsResponse {
  period: StatsPeriod;
  period_start: string;
  period_end: string;
  total_scans: number;
  by_verdict: Record<string, number>;
  by_risk_level: Record<string, number>;
  by_detector: Record<string, number>;
  avg_confidence: number | null;
  daily: DailyBucket[];
}

// ── Async jobs ───────────────────────────────────────────────

export type JobStatusValue = "PENDING" | "PROCESSING" | "SUCCESS" | "ERROR";

export interface JobCreatedResponse {
  job_id: string;
  request_id: string;
  status: JobStatusValue;
  poll_url: string;
}

export interface JobStatusResponse {
  job_id: string;
  request_id: string;
  status: JobStatusValue;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  result: AnalyzeResponse | null;
  error: string | null;
}

// ── Endpoints ─────────────────────────────────────────────────

export const apiEndpoints = {
  analyze: (text: string, explain = false) =>
    api<AnalyzeResponse>("/api/v1/analyze", {
      method: "POST",
      body: JSON.stringify({ text, explain }),
    }),
  createJob: (text: string, explain = false) =>
    api<JobCreatedResponse>("/api/v1/jobs", {
      method: "POST",
      body: JSON.stringify({ text, explain }),
    }),
  getJob: (jobId: string) =>
    api<JobStatusResponse>(`/api/v1/jobs/${jobId}`),
  analyzeSegments: (text: string, window_size = 3, step = 2) =>
    api<SegmentScoringResponse>("/api/v1/analyze/segments", {
      method: "POST",
      body: JSON.stringify({ text, window_size, step }),
    }),
  extractText: async (file: File): Promise<ExtractTextResponse> => {
    // multipart upload — общий `api()` шлёт только JSON, поэтому отдельный fetch.
    const fd = new FormData();
    fd.append("file", file);
    const headers = new Headers();
    if (accessToken) headers.set("Authorization", `Bearer ${accessToken}`);
    const res = await fetch(`${API_URL}/api/v1/extract/text`, {
      method: "POST",
      headers,
      body: fd,
    });
    const text = await res.text();
    let body: unknown = null;
    try {
      body = text ? JSON.parse(text) : null;
    } catch {
      body = text;
    }
    if (!res.ok) {
      const detail =
        (body && typeof body === "object" && "detail" in body
          ? String((body as { detail: unknown }).detail)
          : null) || res.statusText;
      throw new ApiError(res.status, detail, body);
    }
    return body as ExtractTextResponse;
  },
  register: (email: string, password: string) =>
    api<RegisterResponse>("/auth/register", {
      method: "POST",
      body: JSON.stringify({ email, password }),
      skipAuth: true,
    }),
  login: (email: string, password: string) =>
    api<TokenPair>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
      skipAuth: true,
    }),
  logout: (refresh_token: string) =>
    api<void>("/auth/logout", {
      method: "POST",
      body: JSON.stringify({ refresh_token }),
      skipAuth: true,
    }),
  me: () => api<UserPublic>("/me"),
  myAnalyses: (limit = 20, cursor?: string) => {
    const qs = new URLSearchParams({ limit: String(limit) });
    if (cursor) qs.set("cursor", cursor);
    return api<AnalysisHistoryPage>(`/me/analyses?${qs}`);
  },
  getAnalysis: (id: string) =>
    api<AnalysisDetailResponse>(`/me/analyses/${encodeURIComponent(id)}`),
  getStats: (period: StatsPeriod = "30d") =>
    api<StatsResponse>(`/me/stats?period=${encodeURIComponent(period)}`),

  // ── Batch ────────────────────────────────────────────────────
  // multipart upload: API_URL + Bearer-токен подставляется вручную, потому
  // что общий `api()` шлёт только JSON.
  createBatch: async (
    file: File,
    detectorType: "tfidf" | "cascade" = "tfidf",
    explain = false,
  ): Promise<BatchCreatedResponse> => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("detector_type", detectorType);
    fd.append("explain", explain ? "true" : "false");
    const headers = new Headers();
    if (accessToken) headers.set("Authorization", `Bearer ${accessToken}`);
    const res = await fetch(`${API_URL}/api/v1/batch`, {
      method: "POST",
      headers,
      body: fd,
    });
    const text = await res.text();
    let body: unknown = null;
    try {
      body = text ? JSON.parse(text) : null;
    } catch {
      body = text;
    }
    if (!res.ok) {
      const detail =
        (body && typeof body === "object" && "detail" in body
          ? String((body as { detail: unknown }).detail)
          : null) || res.statusText;
      throw new ApiError(res.status, detail, body);
    }
    return body as BatchCreatedResponse;
  },
  getBatchStatus: (id: string) =>
    api<BatchStatusResponse>(`/api/v1/batch/${encodeURIComponent(id)}`),
  myBatches: (limit = 20, cursor?: string) => {
    const qs = new URLSearchParams({ limit: String(limit) });
    if (cursor) qs.set("cursor", cursor);
    return api<BatchListPage>(`/me/batches?${qs}`);
  },
  // Скачивание отчёта: <a href={resultsUrl(...)} download> неудобно (нет
  // Bearer-заголовка). Тянем через fetch+Blob и отдаём URL для <a>.
  downloadBatchResult: async (id: string, format: "csv" | "json") => {
    const headers = new Headers();
    if (accessToken) headers.set("Authorization", `Bearer ${accessToken}`);
    const res = await fetch(
      `${API_URL}/api/v1/batch/${encodeURIComponent(id)}/results.${format}`,
      { headers },
    );
    if (!res.ok) {
      throw new ApiError(res.status, res.statusText);
    }
    const blob = await res.blob();
    return URL.createObjectURL(blob);
  },
};
