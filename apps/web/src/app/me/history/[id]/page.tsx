"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import {
  apiEndpoints,
  ApiError,
  type AnalysisDetailResponse,
  type AnalyzeResponse,
} from "@/lib/api";
import { ResultCard } from "@/components/ResultCard";
import { useAuth } from "@/lib/auth";

/**
 * Detailed scan view: /me/history/[id].
 * Текст в БД не сохраняется (privacy-by-design) — карточка рендерится без подсветки
 * текста, с явным баннером. ResultCard принимает AnalyzeResponse — мы маппим
 * Detail-схему в совместимую (inference_ms → processing_time_ms, was_truncated=false).
 */
export default function AnalysisDetailPage() {
  const { user, loading: authLoading } = useAuth();
  const router = useRouter();
  const params = useParams<{ id: string }>();
  const id = params.id;

  const [detail, setDetail] = useState<AnalysisDetailResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!authLoading && !user) {
      router.replace("/login");
    }
  }, [authLoading, user, router]);

  useEffect(() => {
    if (!user || !id) return;
    let cancelled = false;
    (async () => {
      try {
        const r = await apiEndpoints.getAnalysis(id);
        if (!cancelled) setDetail(r);
      } catch (e) {
        if (cancelled) return;
        if (e instanceof ApiError && e.status === 404) {
          setError("Анализ не найден или принадлежит другому пользователю.");
        } else if (e instanceof ApiError) {
          setError(e.detail);
        } else {
          setError("Не удалось загрузить карточку.");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [user, id]);

  if (authLoading || (!user && loading)) {
    return <div className="text-[var(--muted)]">Загрузка…</div>;
  }

  return (
    <div className="space-y-4">
      <nav className="text-sm text-[var(--muted)]">
        <Link href="/me/history" className="hover:underline">
          История
        </Link>
        <span className="px-1.5">›</span>
        <span>
          {detail
            ? `Анализ от ${new Date(detail.requested_at).toLocaleString("ru-RU")}`
            : "Анализ"}
        </span>
      </nav>

      {error ? (
        <div className="p-4 rounded-md bg-red-50 border border-red-200 text-[var(--danger)] text-sm dark:bg-red-950/30">
          {error}
          <div className="mt-2">
            <Link
              href="/me/history"
              className="underline text-[var(--foreground)]"
            >
              ← Вернуться к истории
            </Link>
          </div>
        </div>
      ) : loading ? (
        <div className="text-[var(--muted)]">Загрузка карточки…</div>
      ) : detail ? (
        <>
          <PrivacyBanner />
          <ResultCard result={detailToAnalyzeResponse(detail)} />
        </>
      ) : null}
    </div>
  );
}

function PrivacyBanner() {
  return (
    <div className="rounded-md border border-[var(--border)] bg-[var(--border)]/20 p-3 text-xs text-[var(--muted)] leading-relaxed">
      Текст анализа не сохраняется в системе по соображениям приватности —
      показаны только метаданные, объяснение и предупреждения.
    </div>
  );
}

function detailToAnalyzeResponse(d: AnalysisDetailResponse): AnalyzeResponse {
  return {
    request_id: d.request_id,
    status: d.status,
    verdict: d.verdict,
    confidence: d.confidence,
    risk_level: d.risk_level,
    text_length: d.text_length,
    was_truncated: false,
    processing_time_ms: d.inference_ms,
    warnings: d.warnings,
    disclaimer: d.disclaimer,
    explanation: d.explanation ?? null,
  };
}
