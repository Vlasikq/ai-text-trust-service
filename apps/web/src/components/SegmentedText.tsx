"use client";

import type { SegmentScoreItem } from "@/lib/api";

/**
 * Посегментная подсветка текста: каждому диапазону start_char..end_char
 * присваивается цвет фона по P(AI). Перекрытия окон обрабатываются
 * усреднением — цвет каждого пикселя текста = среднее P(AI) перекрывающих
 * сегментов. Альтернатива (последний-выигрывает) даёт визуальные «полосы»
 * на стыках; усреднение выглядит плавнее.
 *
 * Это POC из § 3.5.4: отдельный компонент (не расширение `TextHighlights`),
 * чтобы не ломать существующую подсветку шаблонных оборотов AI-стиля.
 */
export function SegmentedText({
  text,
  segments,
}: {
  text: string;
  segments: SegmentScoreItem[];
}) {
  if (!text || segments.length === 0) return null;

  // Для каждого char вычисляем средний P(AI) по перекрывающим окнам.
  const probAtChar = computeCharProbs(text.length, segments);

  // Группируем последовательные char'ы с одинаковым (округлённым) prob — иначе
  // получим N тысяч <span>, что тяжело для DOM на длинных текстах.
  const runs: { start: number; end: number; prob: number | null }[] = [];
  let curStart = 0;
  let curProb = bucket(probAtChar[0]);
  for (let i = 1; i <= text.length; i++) {
    const p = i === text.length ? null : bucket(probAtChar[i]);
    if (p !== curProb) {
      runs.push({ start: curStart, end: i, prob: curProb });
      curStart = i;
      curProb = p;
    }
  }

  return (
    <div className="px-4 pb-4 border-b border-[var(--border)]">
      <div className="text-xs uppercase tracking-wider text-[var(--muted)] font-semibold mb-2">
        Посегментная подсветка
      </div>
      <p className="text-sm leading-relaxed font-mono whitespace-pre-wrap break-words">
        {runs.map((r, i) => {
          const slice = text.slice(r.start, r.end);
          if (r.prob === null) {
            return <span key={i}>{slice}</span>;
          }
          const color = probToBg(r.prob);
          const tooltip = `P(AI) ≈ ${(r.prob * 100).toFixed(0)}%`;
          return (
            <span
              key={i}
              title={tooltip}
              style={{ backgroundColor: color }}
              className="rounded-sm transition-colors"
            >
              {slice}
            </span>
          );
        })}
      </p>
      <Legend />
      <p className="text-xs text-[var(--muted)] mt-2 leading-relaxed">
        Экспериментальная функция: P(AI) считается по скользящим окнам из{" "}
        нескольких предложений (без калибровки). Не подходит для одиночных коротких фраз.
      </p>
    </div>
  );
}

function computeCharProbs(
  textLength: number,
  segments: SegmentScoreItem[],
): number[] {
  const sumProb = new Float64Array(textLength);
  const counts = new Uint32Array(textLength);
  for (const s of segments) {
    const start = Math.max(0, s.start_char);
    const end = Math.min(textLength, s.end_char);
    for (let i = start; i < end; i++) {
      sumProb[i] += s.prob_ai;
      counts[i] += 1;
    }
  }
  const out = new Array<number>(textLength);
  for (let i = 0; i < textLength; i++) {
    out[i] = counts[i] > 0 ? sumProb[i] / counts[i] : NaN;
  }
  return out;
}

// Дискретизуем prob в бакеты: красить с шагом 0.05 — иначе будет слишком
// много `<span>` (после русского run-length группирования).
function bucket(p: number): number | null {
  if (Number.isNaN(p)) return null;
  return Math.round(p * 20) / 20;
}

// Цвет фона: зелёный (low) → жёлтый (mid) → красный (high). С прозрачностью
// 35% чтобы текст оставался читаемым на тёмной теме.
function probToBg(prob: number): string {
  // Линейная интерполяция HSL: hue 120 (green) → 60 (yellow) → 0 (red).
  const hue = 120 - 120 * prob;
  return `hsla(${hue}, 70%, 55%, 0.35)`;
}

function Legend() {
  return (
    <div className="flex items-center gap-3 text-xs text-[var(--muted)] mt-3">
      <span className="flex items-center gap-1">
        <span
          className="inline-block w-3 h-3 rounded-sm"
          style={{ backgroundColor: probToBg(0.05) }}
        />
        P(AI) ≈ 0
      </span>
      <span className="flex items-center gap-1">
        <span
          className="inline-block w-3 h-3 rounded-sm"
          style={{ backgroundColor: probToBg(0.5) }}
        />
        неопределённо
      </span>
      <span className="flex items-center gap-1">
        <span
          className="inline-block w-3 h-3 rounded-sm"
          style={{ backgroundColor: probToBg(0.95) }}
        />
        P(AI) ≈ 1
      </span>
    </div>
  );
}
