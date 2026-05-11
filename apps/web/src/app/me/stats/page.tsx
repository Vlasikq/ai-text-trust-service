"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { apiEndpoints, type StatsPeriod, type StatsResponse } from "@/lib/api";
import { useAuth } from "@/lib/auth";

// recharts тянет ~150KB JS; грузим его только когда пользователь дошёл до
// /me/stats и есть данные для отрисовки. ssr:false — графики чисто клиентские.
const StatsCharts = dynamic(() => import("./StatsCharts"), {
  ssr: false,
  loading: () => (
    <div className="text-[var(--muted)] text-sm py-8 text-center">
      Загружаем графики…
    </div>
  ),
});

const PERIOD_LABEL: Record<StatsPeriod, string> = {
  "7d": "7 дней",
  "30d": "30 дней",
  "90d": "90 дней",
  all: "Всё время",
};

export default function StatsPage() {
  const { user, loading: authLoading } = useAuth();
  const router = useRouter();
  const [period, setPeriod] = useState<StatsPeriod>("30d");
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!authLoading && !user) {
      router.replace("/login");
    }
  }, [authLoading, user, router]);

  useEffect(() => {
    if (!user) return;
    let cancelled = false;
    (async () => {
      setLoading(true);
      setError(null);
      try {
        const r = await apiEndpoints.getStats(period);
        if (!cancelled) setStats(r);
      } catch (e) {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : "Ошибка загрузки.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [user, period]);

  if (authLoading || (!user && loading)) {
    return <div className="text-[var(--muted)]">Загрузка…</div>;
  }

  return (
    <div className="space-y-5">
      <header className="flex items-baseline justify-between gap-3 flex-wrap">
        <div>
          <h1 className="text-2xl font-bold">Статистика</h1>
          <p className="text-sm text-[var(--muted)] mt-1">
            Агрегированная сводка по вашим анализам.
          </p>
        </div>
        <div className="flex gap-1 text-xs">
          {(Object.keys(PERIOD_LABEL) as StatsPeriod[]).map((p) => (
            <button
              key={p}
              type="button"
              onClick={() => setPeriod(p)}
              className={`px-2.5 py-1 rounded-md border transition-colors ${
                period === p
                  ? "bg-[var(--primary)] text-[var(--primary-fg)] border-[var(--primary)]"
                  : "border-[var(--border)] hover:bg-[var(--border)]/40"
              }`}
            >
              {PERIOD_LABEL[p]}
            </button>
          ))}
        </div>
      </header>

      {error && (
        <div className="p-3 rounded-md bg-red-50 border border-red-200 text-[var(--danger)] text-sm dark:bg-red-950/30">
          {error}
        </div>
      )}

      {loading && !stats ? (
        <div className="text-[var(--muted)]">Загрузка…</div>
      ) : stats && stats.total_scans === 0 ? (
        <div className="text-sm text-[var(--muted)] py-12 text-center border border-dashed border-[var(--border)] rounded-md">
          Пока нет анализов за выбранный период.{" "}
          <Link href="/" className="underline text-[var(--foreground)]">
            Сделайте первый
          </Link>
          .
        </div>
      ) : stats ? (
        <>
          <KpiRow stats={stats} />
          <StatsCharts stats={stats} />
        </>
      ) : null}
    </div>
  );
}

function KpiRow({ stats }: { stats: StatsResponse }) {
  const ai = stats.by_verdict.ai ?? 0;
  const human = stats.by_verdict.human ?? 0;
  const avg =
    stats.avg_confidence !== null
      ? `${(stats.avg_confidence * 100).toFixed(1)}%`
      : "—";
  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
      <Kpi label="Всего" value={stats.total_scans} />
      <Kpi label="ИИ" value={ai} accent="text-red-600 dark:text-red-400" />
      <Kpi
        label="Человек"
        value={human}
        accent="text-green-600 dark:text-green-400"
      />
      <Kpi label="Средн. P(AI)" value={avg} />
    </div>
  );
}

function Kpi({
  label,
  value,
  accent,
}: {
  label: string;
  value: string | number;
  accent?: string;
}) {
  return (
    <div className="border border-[var(--border)] rounded-md p-3">
      <div className="text-xs text-[var(--muted)] uppercase tracking-wider">
        {label}
      </div>
      <div className={`text-2xl font-bold mt-0.5 ${accent ?? ""}`}>{value}</div>
    </div>
  );
}
