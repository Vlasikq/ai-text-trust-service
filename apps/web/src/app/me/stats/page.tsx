"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { apiEndpoints, type StatsPeriod, type StatsResponse } from "@/lib/api";
import { useAuth } from "@/lib/auth";

const PERIOD_LABEL: Record<StatsPeriod, string> = {
  "7d": "7 дней",
  "30d": "30 дней",
  "90d": "90 дней",
  all: "Всё время",
};

const VERDICT_COLOR: Record<string, string> = {
  ai: "#ef4444",
  human: "#22c55e",
};
const RISK_COLOR: Record<string, string> = {
  HIGH: "#ef4444",
  MEDIUM: "#f59e0b",
  LOW: "#22c55e",
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
    setLoading(true);
    setError(null);
    (async () => {
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

  const verdictPie = useMemo(() => {
    if (!stats) return [];
    return Object.entries(stats.by_verdict).map(([key, value]) => ({
      name: key === "ai" ? "ИИ" : key === "human" ? "Человек" : key,
      value,
      key,
    }));
  }, [stats]);

  const riskBars = useMemo(() => {
    if (!stats) return [];
    const order: Array<keyof typeof RISK_COLOR> = ["HIGH", "MEDIUM", "LOW"];
    return order
      .filter((k) => k in stats.by_risk_level)
      .map((k) => ({ name: k, value: stats.by_risk_level[k] }));
  }, [stats]);

  const detectorBars = useMemo(() => {
    if (!stats) return [];
    return Object.entries(stats.by_detector).map(([k, v]) => ({
      name: k,
      value: v,
    }));
  }, [stats]);

  const dailyBars = useMemo(() => {
    if (!stats) return [];
    return stats.daily.map((d) => ({
      // короткий день (DD.MM) для тика, full date для tooltip-а
      label: d.date.slice(5).replace("-", "."),
      date: d.date,
      total: d.total,
      ai: d.ai,
      human: d.human,
    }));
  }, [stats]);

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

          <Section title="Анализы по дням">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={dailyBars}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis
                  dataKey="label"
                  fontSize={10}
                  stroke="var(--muted)"
                  interval="preserveStartEnd"
                  tickMargin={4}
                />
                <YAxis
                  allowDecimals={false}
                  fontSize={10}
                  stroke="var(--muted)"
                />
                <Tooltip
                  contentStyle={tooltipStyle}
                  labelFormatter={(_, payload) => {
                    const item = payload?.[0]?.payload;
                    return item ? new Date(item.date).toLocaleDateString("ru-RU") : "";
                  }}
                />
                <Bar dataKey="ai" stackId="a" fill={VERDICT_COLOR.ai} name="ИИ" />
                <Bar
                  dataKey="human"
                  stackId="a"
                  fill={VERDICT_COLOR.human}
                  name="Человек"
                />
              </BarChart>
            </ResponsiveContainer>
          </Section>

          <div className="grid sm:grid-cols-2 gap-4">
            <Section title="Распределение вердиктов">
              {verdictPie.length === 0 ? (
                <EmptyChart />
              ) : (
                <ResponsiveContainer width="100%" height={220}>
                  <PieChart>
                    <Pie
                      data={verdictPie}
                      dataKey="value"
                      nameKey="name"
                      innerRadius={50}
                      outerRadius={80}
                      label={(entry: { name?: string; value?: number }) =>
                        `${entry.name ?? ""}: ${entry.value ?? 0}`
                      }
                    >
                      {verdictPie.map((entry) => (
                        <Cell
                          key={entry.key}
                          fill={VERDICT_COLOR[entry.key] || "#94a3b8"}
                        />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={tooltipStyle} />
                  </PieChart>
                </ResponsiveContainer>
              )}
            </Section>

            <Section title="Уровни риска">
              {riskBars.length === 0 ? (
                <EmptyChart />
              ) : (
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={riskBars}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="name" fontSize={11} stroke="var(--muted)" />
                    <YAxis
                      allowDecimals={false}
                      fontSize={10}
                      stroke="var(--muted)"
                    />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Bar dataKey="value" name="Анализов">
                      {riskBars.map((entry) => (
                        <Cell
                          key={entry.name}
                          fill={RISK_COLOR[entry.name] || "#94a3b8"}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              )}
            </Section>
          </div>

          <Section title="Использованные детекторы">
            {detectorBars.length === 0 ? (
              <EmptyChart />
            ) : (
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={detectorBars} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis
                    type="number"
                    allowDecimals={false}
                    fontSize={10}
                    stroke="var(--muted)"
                  />
                  <YAxis
                    type="category"
                    dataKey="name"
                    fontSize={11}
                    stroke="var(--muted)"
                    width={80}
                  />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Bar dataKey="value" name="Анализов" fill="#3b82f6" />
                </BarChart>
              </ResponsiveContainer>
            )}
          </Section>
        </>
      ) : null}
    </div>
  );
}

const tooltipStyle = {
  background: "var(--background)",
  border: "1px solid var(--border)",
  borderRadius: 6,
  fontSize: 12,
} as const;

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="border border-[var(--border)] rounded-md p-4 bg-[var(--background)]">
      <div className="text-xs uppercase tracking-wider text-[var(--muted)] font-semibold mb-3">
        {title}
      </div>
      {children}
    </div>
  );
}

function EmptyChart() {
  return (
    <div className="text-xs text-[var(--muted)] py-8 text-center">
      Нет данных
    </div>
  );
}

function KpiRow({ stats }: { stats: StatsResponse }) {
  const ai = stats.by_verdict.ai ?? 0;
  const human = stats.by_verdict.human ?? 0;
  const avg = stats.avg_confidence !== null ? `${(stats.avg_confidence * 100).toFixed(1)}%` : "—";
  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
      <Kpi label="Всего" value={stats.total_scans} />
      <Kpi label="ИИ" value={ai} accent="text-red-600 dark:text-red-400" />
      <Kpi label="Человек" value={human} accent="text-green-600 dark:text-green-400" />
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
