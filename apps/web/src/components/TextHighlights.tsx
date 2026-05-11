"use client";

// Token-level подсветка: шаблонные обороты + значок у слишком длинных
// предложений. Эвристика на тех же признаках, которые видит StyleExplainer
// на бэке (длина предложения + formal_connector_ratio).

import { useMemo } from "react";

// Локальный словарь — fallback на случай, если бэк не вернул matched_phrases
// (например, запись из истории, сделанная до появления поля).
const FALLBACK_CONNECTORS = [
  "стали неотъемлемой частью",
  "является неотъемлемой частью",
  "представляет собой",
  "представляют собой",
  "распределённую координацию",
  "распределенную координацию",
  "существенно повысить",
  "стандартизированными",
  "стандартизированные",
  "перспективным направлением",
  "что делает их",
  "что делает",
  "позволяющий",
  "позволяющая",
  "позволяют",
  "позволяет",
  "обладают специализированными",
  "обладают",
  "обладает",
  "следует отметить",
  "следует подчеркнуть",
  "стоит отметить",
  "можно отметить",
  "важно отметить",
  "важно подчеркнуть",
  "помимо этого",
  "более того",
  "в свою очередь",
  "таким образом",
  "в результате",
  "следовательно",
  "несомненно",
  "безусловно",
  "необходимо",
  "комплексный",
  "комплексные",
  "комплексным",
  "благодаря",
  "ввиду",
  "вследствие",
  "является",
  "являются",
  "становится",
  "становятся",
  "позволяя",
  "масштабируемость",
];

// ── matching ──────────────────────────────────────────────────

interface Match {
  start: number;
  end: number;
  term: string;
}

function findMatches(text: string, terms: string[]): Match[] {
  const lower = text.toLowerCase();
  const raw: Match[] = [];
  for (const term of terms) {
    const t = term.toLowerCase();
    let from = 0;
    while (from < lower.length) {
      const idx = lower.indexOf(t, from);
      if (idx < 0) break;
      // Граница слова: предыдущий и следующий символ — не буква (или начало/конец).
      const prev = idx > 0 ? lower[idx - 1] : " ";
      const next = idx + t.length < lower.length ? lower[idx + t.length] : " ";
      const isWordBoundary = (c: string) => !/[а-яёa-z]/i.test(c);
      if (isWordBoundary(prev) && isWordBoundary(next)) {
        raw.push({ start: idx, end: idx + t.length, term });
      }
      from = idx + Math.max(1, t.length);
    }
  }
  // Сортировка по start; при коллизии — берём самый длинный (он содержит остальные).
  raw.sort((a, b) => a.start - b.start || b.end - b.start - (a.end - a.start));
  const out: Match[] = [];
  let lastEnd = -1;
  for (const m of raw) {
    if (m.start >= lastEnd) {
      out.push(m);
      lastEnd = m.end;
    }
  }
  return out;
}

// ── sentence-level: только flag, без визуальной подсветки сплошняком ─────

function splitSentencesWithSpans(text: string): Array<{
  text: string;
  start: number;
  end: number;
  words: number;
}> {
  const out: Array<{ text: string; start: number; end: number; words: number }> =
    [];
  const re = /[^.!?…]+(?:[.!?…]+|$)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    const t = m[0];
    const trimmed = t.trim();
    if (!trimmed) continue;
    const start = m.index + (t.length - t.trimStart().length);
    const end = start + trimmed.length;
    const words = trimmed.split(/\s+/).filter(Boolean).length;
    out.push({ text: trimmed, start, end, words });
  }
  return out;
}

// ── render ────────────────────────────────────────────────────

interface Segment {
  text: string;
  highlight?: string; // term для tooltip
}

function buildSegments(text: string, matches: Match[]): Segment[] {
  if (matches.length === 0) return [{ text }];
  const out: Segment[] = [];
  let cursor = 0;
  for (const m of matches) {
    if (m.start > cursor) {
      out.push({ text: text.slice(cursor, m.start) });
    }
    out.push({ text: text.slice(m.start, m.end), highlight: m.term });
    cursor = m.end;
  }
  if (cursor < text.length) {
    out.push({ text: text.slice(cursor) });
  }
  return out;
}

export function TextHighlights({
  text,
  matchedPhrases,
}: {
  text: string;
  matchedPhrases?: string[] | null;
}) {
  const { segments, longSentences, totalConnectors } = useMemo(() => {
    const terms =
      matchedPhrases && matchedPhrases.length > 0
        ? matchedPhrases
        : FALLBACK_CONNECTORS;
    const matches = findMatches(text, terms);
    const sentences = splitSentencesWithSpans(text);
    const avg =
      sentences.length > 0
        ? sentences.reduce((a, s) => a + s.words, 0) / sentences.length
        : 0;
    const longThr = Math.max(25, avg * 1.2);
    const longs = sentences.filter((s) => s.words >= longThr);
    return {
      segments: buildSegments(text, matches),
      longSentences: longs,
      totalConnectors: matches.length,
    };
  }, [text, matchedPhrases]);

  if (text.length === 0) return null;

  const nothingFound = totalConnectors === 0 && longSentences.length === 0;

  return (
    <details
      open
      className="border-b border-[var(--border)] bg-[var(--border)]/10"
    >
      <summary className="p-4 cursor-pointer select-none flex items-center justify-between font-semibold text-sm hover:bg-[var(--border)]/30">
        <span>Подсветка в тексте</span>
        <span className="text-xs text-[var(--muted)] font-normal">
          {totalConnectors > 0 && `${totalConnectors} оборот${plural(totalConnectors)}`}
          {totalConnectors > 0 && longSentences.length > 0 && " · "}
          {longSentences.length > 0 &&
            `${longSentences.length} длинн${plural2(longSentences.length)} предлож${plural2b(longSentences.length)}`}
        </span>
      </summary>
      <div className="px-4 pb-4 space-y-3">
        <div className="flex items-center gap-3 text-xs text-[var(--muted)] flex-wrap">
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-3 h-3 rounded bg-amber-300 dark:bg-amber-700" />
            шаблонный оборот
          </span>
          {longSentences.length > 0 && (
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-0.5 bg-red-500" />
              длинное предложение (≥25 слов)
            </span>
          )}
        </div>

        <p className="text-sm leading-relaxed">
          {segments.map((s, i) =>
            s.highlight ? (
              <span
                key={i}
                title={`шаблонный оборот: «${s.highlight}»`}
                className="bg-amber-200/80 dark:bg-amber-900/50 rounded px-0.5 cursor-help"
              >
                {s.text}
              </span>
            ) : (
              <span key={i}>{s.text}</span>
            ),
          )}
        </p>

        {longSentences.length > 0 && (
          <div className="text-xs text-[var(--muted)] space-y-1.5 pt-2 border-t border-[var(--border)]/50">
            <div className="font-medium">Длинные предложения:</div>
            <ul className="space-y-1">
              {longSentences.slice(0, 5).map((s, i) => (
                <li
                  key={i}
                  className="border-l-2 border-red-500 pl-2 leading-relaxed"
                >
                  <span className="text-[var(--muted)]">{s.words} слов: </span>
                  <span className="text-[var(--foreground)]">
                    {s.text.length > 120 ? s.text.slice(0, 120) + "..." : s.text}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {nothingFound && (
          <p className="text-xs text-[var(--muted)]">
            Шаблонные обороты и длинные предложения не обнаружены — модель
            опиралась на другие признаки (распределение пунктуации, лексическое
            разнообразие и т.п.). Они показаны выше.
          </p>
        )}

        <p className="text-xs text-[var(--muted)] leading-relaxed">
          Подсветка — <em>эвристическая</em>: размечены конкретные шаблонные
          обороты (по словарю типичных GPT-конструкций) и длинные предложения.
          Это не полная карта внимания модели, а индикативная визуализация.
        </p>
      </div>
    </details>
  );
}

function plural(n: number): string {
  const m10 = n % 10,
    m100 = n % 100;
  if (m10 === 1 && m100 !== 11) return "";
  if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return "а";
  return "ов";
}

function plural2(n: number): string {
  // длинное / длинных
  return n === 1 ? "ое" : "ых";
}
function plural2b(n: number): string {
  // предложение / предложений
  return n === 1 ? "ение" : "ений";
}
