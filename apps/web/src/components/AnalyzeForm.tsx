"use client";

import { useRef, useState, useSyncExternalStore } from "react";
import {
  apiEndpoints,
  ApiError,
  type AnalyzeResponse,
  type JobStatusResponse,
  type RiskLevel,
  type SegmentScoringResponse,
  type Verdict,
} from "@/lib/api";
import { VERDICT_LABEL_SHORT, verdictDotClass } from "@/lib/labels";
import { ResultCard } from "./ResultCard";
import { SegmentedText } from "./SegmentedText";
import demoSamples from "@/lib/demo-samples.json";

// Сессионная история анализов в localStorage. Полный текст не сохраняем —
// храним короткий preview, чтобы PII не оседал в браузере.
interface SessionEntry {
  id: string;
  preview: string;
  result: AnalyzeResponse;
  timestamp: number;
}

// Старый ключ хранил полный текст пользователя — мигрируем и удаляем.
const SESSION_KEY_V1 = "aitrust:session_history";
const SESSION_KEY = "aitrust:session:v2";
const SESSION_MAX = 8;
const PREVIEW_LEN = 120;

interface LegacyEntryV1 {
  id?: string;
  text?: string;
  result?: AnalyzeResponse;
  timestamp?: number;
}

function makePreview(text: string): string {
  if (text.length <= PREVIEW_LEN) return text;
  return text.slice(0, PREVIEW_LEN).trimEnd() + "…";
}

function migrateLegacy(): SessionEntry[] | null {
  const raw = localStorage.getItem(SESSION_KEY_V1);
  if (!raw) return null;
  localStorage.removeItem(SESSION_KEY_V1);
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter(
        (e: unknown): e is LegacyEntryV1 =>
          typeof e === "object" && e !== null && "result" in e,
      )
      .map((e): SessionEntry => ({
        id: e.id ?? `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`,
        preview: makePreview(e.text ?? ""),
        result: e.result as AnalyzeResponse,
        timestamp: e.timestamp ?? Date.now(),
      }))
      .slice(0, SESSION_MAX);
  } catch {
    return [];
  }
}

function loadSession(): SessionEntry[] {
  if (typeof window === "undefined") return [];
  try {
    const migrated = migrateLegacy();
    if (migrated && migrated.length > 0) {
      saveSession(migrated);
      return migrated;
    }
    const raw = localStorage.getItem(SESSION_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.slice(0, SESSION_MAX) : [];
  } catch {
    return [];
  }
}

// Кэш + подписчики: useSyncExternalStore требует стабильный snapshot между
// вызовами getSnapshot, поэтому держим один массив и инвалидируем при write.
let sessionCache: SessionEntry[] | null = null;
const sessionListeners = new Set<() => void>();
const EMPTY_SESSION: SessionEntry[] = [];

function getSessionSnapshot(): SessionEntry[] {
  if (sessionCache === null) sessionCache = loadSession();
  return sessionCache;
}

function getServerSessionSnapshot(): SessionEntry[] {
  return EMPTY_SESSION;
}

function subscribeSession(cb: () => void): () => void {
  sessionListeners.add(cb);
  const onStorage = (e: StorageEvent) => {
    if (e.key === SESSION_KEY || e.key === SESSION_KEY_V1) {
      sessionCache = null;
      cb();
    }
  };
  window.addEventListener("storage", onStorage);
  return () => {
    sessionListeners.delete(cb);
    window.removeEventListener("storage", onStorage);
  };
}

function saveSession(entries: SessionEntry[]) {
  if (typeof window === "undefined") return;
  const next = entries.slice(0, SESSION_MAX);
  try {
    localStorage.setItem(SESSION_KEY, JSON.stringify(next));
    sessionCache = next;
    for (const l of sessionListeners) l();
  } catch {
    /* quota exceeded — пишем только в кэш, чтобы UI не расходился */
    sessionCache = next;
    for (const l of sessionListeners) l();
  }
}

const MIN_CHARS = 300;
const MAX_CHARS = 8000;

// Длина текста, после которой sync /analyze может уйти в INFERENCE_TIMEOUT
// (3 сек на cascade серой зоне = 5–15 сек CPU). Уходим сразу в /jobs + polling.
const ASYNC_LENGTH_THRESHOLD = 1500;

// Polling параметры. Cascade на CPU в худшем случае ~15 сек, плюс запас.
const POLL_INTERVAL_MS = 1000;
const POLL_TIMEOUT_MS = 60_000;

type Stage = "idle" | "sync" | "queued" | "processing";

function isTimeoutResponse(r: AnalyzeResponse): boolean {
  return r.status === "ERROR" && r.warnings.includes("INFERENCE_TIMEOUT");
}

async function pollJob(jobId: string): Promise<AnalyzeResponse> {
  const startedAt = Date.now();
  while (Date.now() - startedAt < POLL_TIMEOUT_MS) {
    const status: JobStatusResponse = await apiEndpoints.getJob(jobId);
    if (status.status === "SUCCESS" && status.result) return status.result;
    if (status.status === "ERROR") {
      throw new Error(status.error || "Job завершился с ошибкой");
    }
    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
  }
  throw new Error("Превышено время ожидания результата (60 сек)");
}

export function AnalyzeForm() {
  const [text, setText] = useState("");
  const [explain, setExplain] = useState(true);
  const [perSegment, setPerSegment] = useState(false);
  const [stage, setStage] = useState<Stage>("idle");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [analyzedText, setAnalyzedText] = useState<string>("");
  const [segments, setSegments] = useState<SegmentScoringResponse | null>(null);
  // Drag-and-drop состояние.
  const [dragOver, setDragOver] = useState(false);
  const [extracting, setExtracting] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<{ name: string; format: string } | null>(null);
  // Сессионная история — external store вокруг localStorage. На SSR пустой,
  // на клиенте подтягивается без hydration mismatch (useSyncExternalStore).
  const session = useSyncExternalStore(
    subscribeSession,
    getSessionSnapshot,
    getServerSessionSnapshot,
  );
  // id текущего показанного анализа из истории (null = только что выполненный).
  const [activeId, setActiveId] = useState<string | null>(null);
  // Чтобы быстро отменять polling при размонтировании / новом сабмите.
  const submitTokenRef = useRef(0);

  function pushToSession(entry: SessionEntry) {
    saveSession([entry, ...session]);
  }

  function selectFromSession(id: string) {
    const entry = session.find((e) => e.id === id);
    if (!entry) return;
    setResult(entry.result);
    // Полного текста в истории нет (приватность) — подсветка не рендерится.
    setAnalyzedText("");
    setSegments(null);
    setActiveId(id);
    setError(null);
  }

  function clearSession() {
    saveSession([]);
    setActiveId(null);
  }

  const len = text.length;
  const tooShort = len > 0 && len < MIN_CHARS;
  const tooLong = len > MAX_CHARS;
  const loading = stage !== "idle";
  const canSubmit = len >= 1 && !loading;

  async function runAsync(myToken: number): Promise<AnalyzeResponse> {
    const created = await apiEndpoints.createJob(text, explain);
    if (submitTokenRef.current !== myToken) throw new Error("cancelled");
    setStage("processing");
    return await pollJob(created.job_id);
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);
    setSegments(null);
    if (!canSubmit) return;

    const myToken = ++submitTokenRef.current;

    try {
      let r: AnalyzeResponse;

      if (text.length >= ASYNC_LENGTH_THRESHOLD) {
        // Длинный текст → сразу в очередь. Cascade на CPU здесь точно не уложится в 3с.
        setStage("queued");
        r = await runAsync(myToken);
      } else {
        // Короткий → пробуем sync. На INFERENCE_TIMEOUT — fallback в очередь.
        setStage("sync");
        r = await apiEndpoints.analyze(text, explain);
        if (submitTokenRef.current !== myToken) return;

        if (isTimeoutResponse(r)) {
          setStage("queued");
          r = await runAsync(myToken);
        }
      }

      if (submitTokenRef.current !== myToken) return;
      setResult(r);
      setAnalyzedText(text);

      // Опциональный посегментный запрос — независимый от основного, не
      // блокирующий: если упадёт, основной результат уже на экране.
      if (perSegment && r.status === "SUCCESS") {
        try {
          const seg = await apiEndpoints.analyzeSegments(text);
          if (submitTokenRef.current === myToken) setSegments(seg);
        } catch {
          /* segments — best-effort, ошибку не показываем */
        }
      }

      // Сохраняем в сессионную историю только успешные результаты с вердиктом.
      if (r.status === "SUCCESS" && r.verdict) {
        const entry: SessionEntry = {
          id:
            typeof crypto !== "undefined" && "randomUUID" in crypto
              ? crypto.randomUUID()
              : `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`,
          preview: makePreview(text),
          result: r,
          timestamp: Date.now(),
        };
        pushToSession(entry);
        setActiveId(entry.id);
      }
    } catch (e) {
      if (submitTokenRef.current !== myToken) return;
      const msg = e instanceof ApiError
        ? e.detail
        : e instanceof Error && e.message !== "cancelled"
        ? e.message
        : "Ошибка соединения. Попробуйте позже.";
      setError(msg);
    } finally {
      if (submitTokenRef.current === myToken) setStage("idle");
    }
  }

  const submitLabel =
    stage === "queued"
      ? "Поставлено в очередь..."
      : stage === "processing"
      ? "Обрабатываем сложный текст..."
      : stage === "sync"
      ? "Анализирую..."
      : "Проанализировать";

  function loadDemo(kind: "human" | "ai") {
    setText(demoSamples[kind].text);
    setResult(null);
    setSegments(null);
    setError(null);
    setUploadedFile(null);
  }

  // Скрыть результат и посегментную карточку, не трогая textarea —
  // вызывается из крестика на ResultCard.
  function dismissResult() {
    setResult(null);
    setSegments(null);
    setActiveId(null);
    setError(null);
  }

  async function handleDroppedFile(f: File) {
    // Если в textarea что-то набрано — подтверждение, чтобы не потерять работу.
    if (text.trim().length > 0) {
      const ok = window.confirm(
        `Заменить текущий текст содержимым файла «${f.name}»?`,
      );
      if (!ok) return;
    }
    setExtracting(true);
    setError(null);
    try {
      const res = await apiEndpoints.extractText(f);
      setText(res.text);
      setUploadedFile({ name: f.name, format: res.source_format });
      setResult(null);
      setSegments(null);
    } catch (e) {
      const msg =
        e instanceof ApiError
          ? e.detail
          : e instanceof Error
          ? e.message
          : "Не удалось распарсить файл";
      setError(msg);
    } finally {
      setExtracting(false);
    }
  }

  function onTextareaDragOver(e: React.DragEvent) {
    if (Array.from(e.dataTransfer.items).some((i) => i.kind === "file")) {
      e.preventDefault();
      setDragOver(true);
    }
  }
  function onTextareaDragLeave() {
    setDragOver(false);
  }
  function onTextareaDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) void handleDroppedFile(f);
  }

  return (
    <div className="space-y-4">
      <form onSubmit={onSubmit} className="space-y-3">
        <label className="block">
          <div className="flex items-baseline justify-between mb-1 gap-2 flex-wrap">
            <span className="block text-sm font-medium">Текст для анализа</span>
            <div className="flex items-center gap-2 text-xs">
              <span className="text-[var(--muted)]">Загрузить пример:</span>
              <button
                type="button"
                onClick={() => loadDemo("human")}
                disabled={loading}
                className="px-2 py-1 rounded border border-[var(--border)] hover:bg-[var(--border)]/40 transition-colors disabled:opacity-50 font-medium"
                title={`Человеческий текст (${demoSamples.human.domain}, ${demoSamples.human.text.length} симв.)`}
              >
                человек
              </button>
              <button
                type="button"
                onClick={() => loadDemo("ai")}
                disabled={loading}
                className="px-2 py-1 rounded border border-[var(--border)] hover:bg-[var(--border)]/40 transition-colors disabled:opacity-50 font-medium"
                title={`AI-сгенерированный текст (${demoSamples.ai.source}, ${demoSamples.ai.text.length} симв.)`}
              >
                AI ({demoSamples.ai.source})
              </button>
            </div>
          </div>
          <div
            onDragOver={onTextareaDragOver}
            onDragLeave={onTextareaDragLeave}
            onDrop={onTextareaDrop}
            className={`relative rounded-md transition-colors ${
              dragOver
                ? "ring-2 ring-[var(--primary)] bg-[var(--primary)]/5"
                : ""
            }`}
          >
            <textarea
              value={text}
              onChange={(e) => {
                setText(e.target.value);
                if (uploadedFile) setUploadedFile(null);
              }}
              placeholder={`Вставьте русскоязычный текст (${MIN_CHARS}–${MAX_CHARS} символов) или перетащите TXT/DOCX/PDF...`}
              rows={10}
              aria-label="Текст для анализа"
              aria-describedby="analyze-text-counter"
              aria-invalid={tooLong || undefined}
              className="w-full px-3 py-2 rounded-md border border-[var(--border)] bg-[var(--background)] focus:outline-none focus:ring-2 focus:ring-[var(--primary)] resize-y font-mono text-sm"
            />
            {dragOver && (
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none rounded-md bg-[var(--primary)]/10 border-2 border-dashed border-[var(--primary)] text-sm font-medium text-[var(--primary)]">
                Отпустите файл, чтобы извлечь текст
              </div>
            )}
            {extracting && (
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none rounded-md bg-[var(--background)]/80 text-sm">
                Извлекаем текст…
              </div>
            )}
          </div>
          {uploadedFile && (
            <div className="mt-1 inline-flex items-center gap-2 text-xs px-2 py-1 rounded-full bg-[var(--border)]/40 border border-[var(--border)]">
              <span aria-hidden>📄</span>
              <span className="font-medium">{uploadedFile.name}</span>
              <span className="text-[var(--muted)]">
                · {uploadedFile.format.toUpperCase()} · {text.length} символов
              </span>
              <button
                type="button"
                onClick={() => setUploadedFile(null)}
                className="text-[var(--muted)] hover:text-[var(--foreground)] ml-1"
                title="Скрыть"
              >
                ×
              </button>
            </div>
          )}
          <div className="flex items-center justify-between text-xs mt-1">
            <span
              id="analyze-text-counter"
              className={`${
                tooShort
                  ? "text-[var(--warn)]"
                  : tooLong
                  ? "text-[var(--danger)]"
                  : "text-[var(--muted)]"
              }`}
            >
              {len} символов
              {tooShort && ` · мин. ${MIN_CHARS} для уверенного результата`}
              {tooLong && ` · превышен лимит ${MAX_CHARS}, текст будет обрезан`}
              {!tooShort && !tooLong && len >= ASYNC_LENGTH_THRESHOLD && (
                <span className="text-[var(--muted)]">
                  {" "}· длинный текст обработается через очередь (до ~15 сек)
                </span>
              )}
            </span>
            <button
              type="button"
              onClick={() => {
                setText("");
                setUploadedFile(null);
                dismissResult();
              }}
              className="text-[var(--muted)] hover:underline disabled:opacity-50"
              disabled={(!text && !result) || loading}
            >
              Очистить
            </button>
          </div>
        </label>

        <div className="flex items-center justify-between gap-3 flex-wrap">
          <div className="flex flex-col gap-1">
            <label className="flex items-center gap-2 text-sm cursor-pointer select-none">
              <input
                type="checkbox"
                checked={explain}
                onChange={(e) => setExplain(e.target.checked)}
                className="w-4 h-4 accent-[var(--primary)]"
                disabled={loading}
              />
              <span>Показать объяснение (стилеметрические признаки)</span>
            </label>
            <label className="flex items-center gap-2 text-sm cursor-pointer select-none">
              <input
                type="checkbox"
                checked={perSegment}
                onChange={(e) => setPerSegment(e.target.checked)}
                className="w-4 h-4 accent-[var(--primary)]"
                disabled={loading}
              />
              <span>
                Посегментный анализ{" "}
                <span className="text-[var(--muted)] text-xs">(экспериментально)</span>
              </span>
            </label>
          </div>
          <button
            type="submit"
            disabled={!canSubmit}
            className="px-6 py-2.5 rounded-md bg-[var(--primary)] text-[var(--primary-fg)] font-medium hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
          >
            {submitLabel}
          </button>
        </div>
      </form>

      {stage === "queued" || stage === "processing" ? (
        <div className="p-3 rounded-md bg-[var(--border)]/30 border border-[var(--border)] text-sm text-[var(--muted)]">
          Текст отправлен на каскадную обработку (TF-IDF → ruRoBERTa). Это
          может занять до 15 секунд на CPU; страница обновится автоматически.
        </div>
      ) : null}

      {error && (
        <div className="p-3 rounded-md bg-red-50 border border-red-200 text-[var(--danger)] text-sm dark:bg-red-950/30">
          {error}
        </div>
      )}

      {session.length > 0 && (
        <SessionHistoryTabs
          entries={session}
          activeId={activeId}
          onSelect={selectFromSession}
          onClear={clearSession}
        />
      )}

      {result && (
        <ResultCard
          result={result}
          text={analyzedText}
          onClose={dismissResult}
        />
      )}

      {result && segments && (
        <div className="rounded-lg border border-[var(--border)] bg-[var(--background)] overflow-hidden">
          <SegmentedText text={analyzedText} segments={segments.segments} />
        </div>
      )}
    </div>
  );
}

/** Горизонтальные вкладки последних SESSION_MAX анализов сессии. */
function SessionHistoryTabs({
  entries,
  activeId,
  onSelect,
  onClear,
}: {
  entries: SessionEntry[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onClear: () => void;
}) {
  return (
    <div className="border border-[var(--border)] rounded-md bg-[var(--border)]/15 p-3 space-y-2">
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <div className="text-xs uppercase tracking-wider text-[var(--muted)] font-semibold">
          История сессии · {entries.length}
        </div>
        <button
          type="button"
          onClick={onClear}
          className="text-xs text-[var(--muted)] hover:text-[var(--danger)] underline"
        >
          Очистить
        </button>
      </div>

      <div className="flex gap-2 overflow-x-auto pb-1 -mb-1">
        {entries.map((entry) => {
          const isActive = entry.id === activeId;
          const verdict = entry.result.verdict;
          const risk = entry.result.risk_level;
          const shortPreview =
            entry.preview.length > 36
              ? entry.preview.slice(0, 36).trim() + "…"
              : entry.preview;

          return (
            <button
              key={entry.id}
              type="button"
              onClick={() => onSelect(entry.id)}
              className={`shrink-0 text-left px-3 py-2 rounded-md border text-xs transition-colors min-w-[180px] max-w-[240px] ${
                isActive
                  ? "bg-[var(--background)] border-[var(--primary)] shadow-sm"
                  : "bg-[var(--background)]/60 border-[var(--border)] hover:border-[var(--primary)]/60"
              }`}
              title={entry.preview}
            >
              <div className="flex items-center gap-1.5 mb-1">
                <VerdictDot verdict={verdict} risk={risk} />
                <span className="font-semibold">
                  {verdict ? VERDICT_LABEL_SHORT[verdict] : "—"}
                </span>
                <span className="text-[var(--muted)] font-mono">
                  {entry.result.confidence !== null
                    ? `${(entry.result.confidence * 100).toFixed(0)}%`
                    : "—"}
                </span>
                <span className="text-[var(--muted)] ml-auto text-[10px]">
                  {formatRelative(entry.timestamp)}
                </span>
              </div>
              <div className="truncate text-[var(--muted)] font-normal">
                {shortPreview}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function VerdictDot({
  verdict,
  risk,
}: {
  verdict: Verdict | null | undefined;
  risk: RiskLevel | null | undefined;
}) {
  return (
    <span
      className={`inline-block w-2 h-2 rounded-full ${verdictDotClass(verdict, risk)}`}
      aria-hidden
    />
  );
}

function formatRelative(ts: number): string {
  const diffSec = Math.max(0, Math.floor((Date.now() - ts) / 1000));
  if (diffSec < 60) return `${diffSec}с назад`;
  const m = Math.floor(diffSec / 60);
  if (m < 60) return `${m}мин`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}ч`;
  const d = Math.floor(h / 24);
  return `${d}д`;
}
