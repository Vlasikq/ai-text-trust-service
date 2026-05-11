"use client";

// Тяжёлый блок recharts вынесен в отдельный модуль и подгружается ленивым
// импортом из page.tsx — это убирает recharts из main chunk страницы.

import { useMemo } from "react";
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
import type { StatsResponse } from "@/lib/api";
import {
  RISK_HEX,
  VERDICT_HEX,
  VERDICT_LABEL_SHORT,
} from "@/lib/labels";

const tooltipStyle = {
  background: "var(--background)",
  border: "1px solid var(--border)",
  borderRadius: 6,
  fontSize: 12,
} as const;

export default function StatsCharts({ stats }: { stats: StatsResponse }) {
  const verdictPie = useMemo(
    () =>
      Object.entries(stats.by_verdict).map(([key, value]) => ({
        name:
          key === "ai" || key === "human"
            ? VERDICT_LABEL_SHORT[key]
            : key,
        value,
        key,
      })),
    [stats],
  );

  const riskBars = useMemo(() => {
    const order: Array<keyof typeof RISK_HEX> = ["HIGH", "MEDIUM", "LOW"];
    return order
      .filter((k) => k in stats.by_risk_level)
      .map((k) => ({ name: k, value: stats.by_risk_level[k] }));
  }, [stats]);

  const detectorBars = useMemo(
    () =>
      Object.entries(stats.by_detector).map(([k, v]) => ({
        name: k,
        value: v,
      })),
    [stats],
  );

  const dailyBars = useMemo(
    () =>
      stats.daily.map((d) => ({
        // DD.MM для тика, full date для tooltip-а
        label: d.date.slice(5).replace("-", "."),
        date: d.date,
        total: d.total,
        ai: d.ai,
        human: d.human,
      })),
    [stats],
  );

  return (
    <>
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
                return item
                  ? new Date(item.date).toLocaleDateString("ru-RU")
                  : "";
              }}
            />
            <Bar dataKey="ai" stackId="a" fill={VERDICT_HEX.ai} name="ИИ" />
            <Bar
              dataKey="human"
              stackId="a"
              fill={VERDICT_HEX.human}
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
                      fill={
                        entry.key === "ai" || entry.key === "human"
                          ? VERDICT_HEX[entry.key]
                          : "#94a3b8"
                      }
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
                      fill={RISK_HEX[entry.name] || "#94a3b8"}
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
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
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
