"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  apiEndpoints,
  ApiError,
  type BatchStatusResponse,
  type BatchSkippedRow,
} from "@/lib/api";
import { useAuth } from "@/lib/auth";

const POLL_INTERVAL_MS = 2000;

type Stage = "idle" | "uploading" | "processing" | "completed" | "error";

export default function BatchPage() {
  const { user, loading: authLoading } = useAuth();
  const router = useRouter();

  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<{
    headers: string[];
    sampleRows: string[][];
    totalRows: number;
  } | null>(null);
  const [detectorType, setDetectorType] = useState<"tfidf" | "cascade">(
    "tfidf",
  );
  const [explain, setExplain] = useState(false);

  const [stage, setStage] = useState<Stage>("idle");
  const [error, setError] = useState<string | null>(null);
  const [batch, setBatch] = useState<BatchStatusResponse | null>(null);

  // Recursive setTimeout: следующий тик стартует только после возврата
  // предыдущего, иначе на slow network копились бы параллельные запросы.
  const pollRef = useRef<number | null>(null);
  const cancelledRef = useRef(false);

  useEffect(() => {
    if (!authLoading && !user) router.replace("/login");
  }, [authLoading, user, router]);

  useEffect(() => {
    return () => {
      cancelledRef.current = true;
      if (pollRef.current) window.clearTimeout(pollRef.current);
    };
  }, []);

  async function handleFile(f: File) {
    setError(null);
    setBatch(null);
    setStage("idle");
    setFile(f);

    // Превью первых 5 строк. Простой парсер, не учитывает экранированные кавычки —
    // достаточно для иллюстрации перед загрузкой; серверный парсер строгий.
    try {
      const text = await f.text();
      const lines = text.split(/\r?\n/).filter((l) => l.trim());
      if (lines.length === 0) throw new Error("Файл пуст");
      const delim = detectDelimiter(lines[0]);
      const headers = lines[0].split(delim);
      const sampleRows = lines.slice(1, 6).map((l) => l.split(delim));
      setPreview({
        headers,
        sampleRows,
        totalRows: lines.length - 1,
      });
    } catch {
      setPreview(null);
    }
  }

  function detectDelimiter(headerLine: string): string {
    const counts = {
      ",": (headerLine.match(/,/g) ?? []).length,
      ";": (headerLine.match(/;/g) ?? []).length,
      "\t": (headerLine.match(/\t/g) ?? []).length,
    };
    return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0] || ",";
  }

  function clearPolling() {
    cancelledRef.current = true;
    if (pollRef.current) {
      window.clearTimeout(pollRef.current);
      pollRef.current = null;
    }
  }

  async function pollLoop(batchId: string) {
    if (cancelledRef.current) return;
    let shouldContinue = true;
    try {
      const s = await apiEndpoints.getBatchStatus(batchId);
      if (cancelledRef.current) return;
      setBatch(s);
      if (s.status === "COMPLETED") {
        setStage("completed");
        shouldContinue = false;
      }
    } catch (e) {
      if (cancelledRef.current) return;
      if (e instanceof ApiError && e.status === 404) {
        setError("Batch не найден");
        setStage("error");
        shouldContinue = false;
      }
      // Прочие ошибки (transient network / 5xx) — продолжаем тикать.
    }
    if (shouldContinue && !cancelledRef.current) {
      pollRef.current = window.setTimeout(
        () => void pollLoop(batchId),
        POLL_INTERVAL_MS,
      );
    }
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return;
    setError(null);
    setStage("uploading");

    try {
      const created = await apiEndpoints.createBatch(file, detectorType, explain);
      setStage("processing");
      cancelledRef.current = false;
      // Стартовый тик: подтягиваем первый статус и рекурсивно планируем следующие.
      void pollLoop(created.batch_id);
    } catch (e) {
      setStage("error");
      if (e instanceof ApiError) {
        setError(e.detail || `Ошибка ${e.status}`);
      } else {
        setError("Ошибка соединения");
      }
    }
  }

  async function downloadResult(format: "csv" | "json") {
    if (!batch) return;
    try {
      const url = await apiEndpoints.downloadBatchResult(batch.batch_id, format);
      const a = document.createElement("a");
      a.href = url;
      a.download = `batch_${batch.batch_id}.${format}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Не удалось скачать");
    }
  }

  function reset() {
    clearPolling();
    setFile(null);
    setPreview(null);
    setBatch(null);
    setError(null);
    setStage("idle");
  }

  if (authLoading || !user) {
    return <div className="text-[var(--muted)]">Загрузка…</div>;
  }

  return (
    <div className="space-y-5">
      <header>
        <h1 className="text-2xl font-bold">Пакетная обработка (CSV)</h1>
        <p className="text-sm text-[var(--muted)] mt-1">
          Загрузите CSV с колонкой <code>text</code> (опционально <code>id</code>,{" "}
          <code>name</code>). Лимиты: до 100 строк, до 2 МБ, текст 300–8000 символов.
        </p>
      </header>

      {error && (
        <div className="p-3 rounded-md bg-red-50 border border-red-200 text-[var(--danger)] text-sm dark:bg-red-950/30">
          {error}
        </div>
      )}

      {stage === "idle" || stage === "error" ? (
        <UploadForm
          file={file}
          preview={preview}
          detectorType={detectorType}
          explain={explain}
          onFile={handleFile}
          onDetector={setDetectorType}
          onExplain={setExplain}
          onSubmit={onSubmit}
          onReset={reset}
        />
      ) : (
        <ProgressView
          stage={stage}
          batch={batch}
          onDownload={downloadResult}
          onReset={reset}
        />
      )}
    </div>
  );
}

// ── upload form ───────────────────────────────────────────────

function UploadForm({
  file,
  preview,
  detectorType,
  explain,
  onFile,
  onDetector,
  onExplain,
  onSubmit,
  onReset,
}: {
  file: File | null;
  preview: { headers: string[]; sampleRows: string[][]; totalRows: number } | null;
  detectorType: "tfidf" | "cascade";
  explain: boolean;
  onFile: (f: File) => void;
  onDetector: (d: "tfidf" | "cascade") => void;
  onExplain: (b: boolean) => void;
  onSubmit: (e: React.FormEvent) => void;
  onReset: () => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  return (
    <form onSubmit={onSubmit} className="space-y-3">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          const f = e.dataTransfer.files[0];
          if (f) onFile(f);
        }}
        onClick={() => inputRef.current?.click()}
        className={`border-2 border-dashed rounded-md p-6 text-center cursor-pointer transition-colors ${
          dragOver
            ? "border-[var(--primary)] bg-[var(--primary)]/10"
            : "border-[var(--border)] hover:border-[var(--primary)]/60"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".csv,text/csv"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) onFile(f);
          }}
        />
        {file ? (
          <div className="text-sm">
            <div className="font-medium">{file.name}</div>
            <div className="text-xs text-[var(--muted)] mt-1">
              {(file.size / 1024).toFixed(1)} КБ ·{" "}
              {preview ? `${preview.totalRows} строк (без header)` : "нет превью"}
            </div>
          </div>
        ) : (
          <div className="text-sm text-[var(--muted)]">
            Перетащите CSV сюда или нажмите для выбора
          </div>
        )}
      </div>

      {preview && (
        <div className="border border-[var(--border)] rounded-md p-3 overflow-x-auto">
          <div className="text-xs uppercase tracking-wider text-[var(--muted)] font-semibold mb-2">
            Превью первых строк
          </div>
          <table className="text-xs w-full">
            <thead>
              <tr>
                {preview.headers.map((h, i) => (
                  <th
                    key={i}
                    className="text-left px-2 py-1 border-b border-[var(--border)] font-semibold"
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {preview.sampleRows.map((row, ri) => (
                <tr key={ri} className="even:bg-[var(--border)]/20">
                  {row.map((cell, ci) => (
                    <td key={ci} className="px-2 py-1 max-w-[280px] truncate" title={cell}>
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="grid sm:grid-cols-2 gap-3">
        <label className="text-sm">
          <span className="block text-xs uppercase tracking-wider text-[var(--muted)] font-semibold mb-1">
            Детектор
          </span>
          <select
            value={detectorType}
            onChange={(e) => onDetector(e.target.value as "tfidf" | "cascade")}
            className="w-full px-3 py-2 rounded-md border border-[var(--border)] bg-[var(--background)] focus:outline-none focus:ring-2 focus:ring-[var(--primary)]"
          >
            <option value="tfidf">TF-IDF (быстро, рекомендуется)</option>
            <option value="cascade">Cascade (точнее, но медленнее)</option>
          </select>
        </label>
        <label className="flex items-center gap-2 self-end pb-1.5 text-sm cursor-pointer select-none">
          <input
            type="checkbox"
            checked={explain}
            onChange={(e) => onExplain(e.target.checked)}
            className="w-4 h-4 accent-[var(--primary)]"
          />
          <span>Включить объяснимость</span>
        </label>
      </div>

      <div className="flex gap-2 justify-end">
        {file && (
          <button
            type="button"
            onClick={onReset}
            className="px-4 py-2 rounded-md border border-[var(--border)] hover:bg-[var(--border)]/40"
          >
            Очистить
          </button>
        )}
        <button
          type="submit"
          disabled={!file}
          className="px-6 py-2 rounded-md bg-[var(--primary)] text-[var(--primary-fg)] font-medium hover:opacity-90 disabled:opacity-50"
        >
          Запустить обработку
        </button>
      </div>
    </form>
  );
}

// ── progress ──────────────────────────────────────────────────

function ProgressView({
  stage,
  batch,
  onDownload,
  onReset,
}: {
  stage: Stage;
  batch: BatchStatusResponse | null;
  onDownload: (f: "csv" | "json") => void;
  onReset: () => void;
}) {
  if (!batch) {
    return <div className="text-[var(--muted)] text-sm">Загружаем статус…</div>;
  }
  const done = batch.n_completed + batch.n_failed;
  const pct = batch.n_total > 0 ? (done / batch.n_total) * 100 : 0;

  return (
    <div className="space-y-4">
      <div className="border border-[var(--border)] rounded-md p-4 space-y-3">
        <div className="flex items-baseline justify-between gap-3 flex-wrap">
          <div>
            <div className="text-xs uppercase tracking-wider text-[var(--muted)] font-semibold">
              {batch.file_name || "Batch"}
            </div>
            <div className="text-2xl font-bold mt-0.5">
              {batch.status === "COMPLETED"
                ? "Готово"
                : stage === "uploading"
                ? "Загрузка…"
                : "Обработка…"}
            </div>
          </div>
          <div className="font-mono text-sm">
            {done} / {batch.n_total}
            {batch.n_failed > 0 && (
              <span className="text-[var(--danger)] ml-2">
                ({batch.n_failed} ошибок)
              </span>
            )}
          </div>
        </div>

        <div className="h-2 rounded-full bg-[var(--border)] overflow-hidden">
          <div
            className="h-full bg-[var(--primary)] transition-all"
            style={{ width: `${pct}%` }}
          />
        </div>

        {batch.status === "COMPLETED" && (
          <div className="flex gap-2 pt-1">
            <button
              type="button"
              onClick={() => onDownload("csv")}
              className="px-4 py-2 rounded-md bg-[var(--primary)] text-[var(--primary-fg)] font-medium hover:opacity-90 text-sm"
            >
              Скачать CSV
            </button>
            <button
              type="button"
              onClick={() => onDownload("json")}
              className="px-4 py-2 rounded-md border border-[var(--border)] hover:bg-[var(--border)]/40 text-sm"
            >
              Скачать JSON
            </button>
            <button
              type="button"
              onClick={onReset}
              className="px-4 py-2 rounded-md border border-[var(--border)] hover:bg-[var(--border)]/40 text-sm ml-auto"
            >
              Новый batch
            </button>
          </div>
        )}
      </div>

      {batch.skipped.length > 0 && <SkippedSummary rows={batch.skipped} />}
    </div>
  );
}

function SkippedSummary({ rows }: { rows: BatchSkippedRow[] }) {
  return (
    <details className="border border-[var(--border)] rounded-md p-3">
      <summary className="cursor-pointer text-sm font-semibold">
        Пропущено строк: {rows.length}
      </summary>
      <ul className="mt-2 text-xs space-y-1 text-[var(--muted)]">
        {rows.map((r, i) => (
          <li key={i}>
            Строка {r.row_num}
            {r.external_id ? ` (id=${r.external_id})` : ""}: {r.reason}
          </li>
        ))}
      </ul>
    </details>
  );
}
