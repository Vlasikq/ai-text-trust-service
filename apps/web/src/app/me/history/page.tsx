"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  apiEndpoints,
  type AnalysisHistoryItem,
  type AnalysisHistoryPage,
} from "@/lib/api";
import { useAuth } from "@/lib/auth";

const VERDICT_LABEL = { ai: "ИИ", human: "Человек" } as const;
const RISK_CLS: Record<string, string> = {
  HIGH: "bg-red-100 text-red-800 dark:bg-red-950/40 dark:text-red-300",
  MEDIUM:
    "bg-amber-100 text-amber-800 dark:bg-amber-950/40 dark:text-amber-300",
  LOW: "bg-green-100 text-green-800 dark:bg-green-950/40 dark:text-green-300",
};

export default function HistoryPage() {
  const { user, loading: authLoading } = useAuth();
  const router = useRouter();
  const [items, setItems] = useState<AnalysisHistoryItem[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [hasMore, setHasMore] = useState(false);

  useEffect(() => {
    if (!authLoading && !user) {
      router.replace("/login");
    }
  }, [authLoading, user, router]);

  useEffect(() => {
    if (!user) return;
    let cancelled = false;
    (async () => {
      try {
        const r: AnalysisHistoryPage = await apiEndpoints.myAnalyses(20);
        if (!cancelled) {
          setItems(r.items);
          setCursor(r.next_cursor);
          setHasMore(!!r.next_cursor);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [user]);

  async function loadMore() {
    if (!cursor) return;
    setLoading(true);
    try {
      const r = await apiEndpoints.myAnalyses(20, cursor);
      setItems((prev) => [...prev, ...r.items]);
      setCursor(r.next_cursor);
      setHasMore(!!r.next_cursor);
    } finally {
      setLoading(false);
    }
  }

  if (authLoading || (!user && loading)) {
    return <div className="text-[var(--muted)]">Загрузка...</div>;
  }

  return (
    <div className="space-y-4">
      <header>
        <h1 className="text-2xl font-bold">История анализов</h1>
        <p className="text-sm text-[var(--muted)] mt-1">
          Ваши последние запросы. Текст не сохраняется — только результат и
          метаданные.
        </p>
      </header>

      {items.length === 0 && !loading ? (
        <div className="text-sm text-[var(--muted)] py-12 text-center border border-dashed border-[var(--border)] rounded-md">
          Пока нет анализов. Перейдите на главную и проанализируйте первый
          текст.
        </div>
      ) : (
        <ul className="space-y-2">
          {items.map((it) => (
            <li key={it.id}>
              <Link
                href={`/me/history/${it.id}`}
                className="block border border-[var(--border)] rounded-md p-3 flex items-center justify-between gap-3 flex-wrap hover:border-[var(--primary)] hover:bg-[var(--border)]/30 transition-colors"
              >
                <div className="space-y-0.5">
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-sm">
                      {it.verdict ? VERDICT_LABEL[it.verdict] : "—"}
                    </span>
                    {it.risk_level && (
                      <span
                        className={`text-xs px-1.5 py-0.5 rounded font-medium ${
                          RISK_CLS[it.risk_level] || ""
                        }`}
                      >
                        {it.risk_level}
                      </span>
                    )}
                    <span className="text-xs text-[var(--muted)]">
                      {it.detector_used}
                    </span>
                  </div>
                  <div className="text-xs text-[var(--muted)]">
                    {new Date(it.requested_at).toLocaleString("ru-RU")} ·{" "}
                    {it.text_length} симв.
                  </div>
                </div>
                {it.confidence !== null && it.verdict && (
                  <div
                    className="text-sm font-mono"
                    title={`P(AI) = ${(it.confidence * 100).toFixed(1)}%`}
                  >
                    {(
                      (it.verdict === "ai"
                        ? it.confidence
                        : 1 - it.confidence) * 100
                    ).toFixed(1)}
                    %
                  </div>
                )}
              </Link>
            </li>
          ))}
        </ul>
      )}

      {hasMore && (
        <button
          onClick={loadMore}
          disabled={loading}
          className="w-full px-4 py-2 rounded-md border border-[var(--border)] hover:bg-[var(--border)]/50 disabled:opacity-50 text-sm"
        >
          {loading ? "Загрузка..." : "Загрузить ещё"}
        </button>
      )}
    </div>
  );
}
