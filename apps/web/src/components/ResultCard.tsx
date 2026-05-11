"use client";

import type { AnalyzeResponse } from "@/lib/api";
import { TextHighlights } from "./TextHighlights";

const VERDICT_LABEL = {
  ai: "Сгенерировано ИИ",
  human: "Написано человеком",
} as const;

const RISK_STYLES: Record<string, { label: string; cls: string }> = {
  HIGH: {
    label: "ВЫСОКИЙ риск",
    cls: "bg-red-50 text-red-700 border-red-200 dark:bg-red-950/30 dark:text-red-300",
  },
  MEDIUM: {
    label: "СРЕДНИЙ риск",
    cls: "bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-950/30 dark:text-amber-300",
  },
  LOW: {
    label: "НИЗКИЙ риск",
    cls: "bg-green-50 text-green-700 border-green-200 dark:bg-green-950/30 dark:text-green-300",
  },
};

const WARNING_LABEL: Record<string, string> = {
  TEXT_TOO_SHORT: "Текст короткий — точность снижена",
  TEXT_TRUNCATED: "Текст обрезан до 8000 символов",
  TRANSFORMER_UNAVAILABLE: "Доступен только TF-IDF (трансформер недоступен)",
  LOW_CONFIDENCE: "Низкая уверенность",
  CALIBRATION_UNAVAILABLE: "Калибровка отключена",
  INFERENCE_TIMEOUT: "Превышено время инференса",
  STYLE_EXPLANATION_HEURISTIC: "Объяснение — эвристическое",
};

// Локализация имён стилеметрических признаков. Ключи — ровно те, что отдаёт
// scripts/stylometric_features.py (см. _extract_single).
const FEATURE_INFO: Record<string, { label: string; hint?: string }> = {
  // base
  text_len: { label: "Длина текста" },
  word_count: { label: "Количество слов" },
  sentence_count: { label: "Количество предложений" },
  avg_word_len: {
    label: "Средняя длина слова",
    hint: "AI чаще выбирает длинные книжные слова.",
  },
  avg_sent_len: {
    label: "Средняя длина предложения",
    hint: "AI обычно пишет более длинные и однородные фразы.",
  },
  comma_density: {
    label: "Плотность запятых",
    hint: "AI часто перегружает текст обособлениями.",
  },
  question_marks: {
    label: "Вопросительные знаки ( ? )",
    hint: "Вопросы и диалог чаще у людей.",
  },
  exclamation_marks: {
    label: "Восклицательные знаки ( ! )",
    hint: "Эмоциональный знак, у AI почти не встречается.",
  },
  ttr: {
    label: "Лексическое разнообразие (TTR)",
    hint: "Доля уникальных слов. У AI часто ниже — словарь повторяется.",
  },
  ends_formulaic: {
    label: "Шаблонная концовка",
    hint: "«Таким образом…», «подводя итог…», «в заключение…» — типично для AI.",
  },

  // lexical
  std_word_len: { label: "Разброс длины слов" },
  max_word_len: { label: "Самое длинное слово" },
  hapax_ratio: {
    label: "Доля уникальных слов",
    hint: "Слова, встречающиеся в тексте всего один раз.",
  },
  long_word_ratio: {
    label: "Доля длинных слов (>10 букв)",
    hint: "AI чаще использует длинные термины и абстрактные слова.",
  },
  short_word_ratio: {
    label: "Доля коротких слов (<3 букв)",
    hint: "В живой речи много коротких слов: предлоги, частицы, междометия.",
  },
  cyrillic_ratio: { label: "Доля кириллицы" },
  digit_ratio: {
    label: "Доля цифр",
    hint: "Конкретные числа и даты — чаще в живых текстах.",
  },

  // punctuation
  semicolons: {
    label: "Точки с запятой ( ; )",
    hint: "Книжный знак; у AI чаще нормы.",
  },
  colons: {
    label: "Двоеточия ( : )",
    hint: "Перечисления и пояснения. Встречаются и у людей, и у AI.",
  },
  dashes: {
    label: "Длинные тире ( — )",
    hint: "Признак живой авторской речи; AI ставит реже.",
  },
  parens: {
    label: "Скобки ( ... )",
    hint: "Уточнения в скобках характерны для живой речи.",
  },
  quotes_guillemet: {
    label: "Кавычки-«ёлочки» («»)",
    hint: "Характерны для русскоязычных живых текстов; AI часто их избегает.",
  },
  ellipsis: {
    label: "Многоточия (...)",
    hint: "Незавершённость мысли — признак живого автора.",
  },
  punct_density: {
    label: "Плотность пунктуации",
    hint: "Сколько знаков препинания на символ.",
  },

  // structure
  sentence_len_std: {
    label: "Разброс длин предложений",
    hint: "У живых текстов чередуются короткие и длинные; AI пишет ровнее.",
  },
  first_sentence_len: { label: "Длина первого предложения" },
  last_sentence_len: {
    label: "Длина последнего предложения",
    hint: "AI часто завершает длинной обобщающей фразой.",
  },
  max_sent_len: { label: "Самое длинное предложение" },

  // ngram entropy
  char_bigram_entropy: {
    label: "Энтропия биграмм символов",
    hint: "Разнообразие пар букв. У AI часто ниже — стиль однообразнее.",
  },
  char_trigram_entropy: {
    label: "Энтропия триграмм символов",
    hint: "То же, но на тройках букв.",
  },
  word_bigram_entropy: {
    label: "Энтропия биграмм слов",
    hint: "Насколько разнообразны пары соседних слов.",
  },

  // repetition
  self_repetition_rate: {
    label: "Самоповтор (n-граммы)",
    hint: "Сколько кусков текста повторяются. У AI часто выше.",
  },
  mattr_50: {
    label: "Скользящий TTR (MATTR-50)",
    hint: "Лексическое разнообразие на окнах по 50 слов. AI часто ниже.",
  },
};

// Маркер «функциональное слово» (fw_*): автоматически генерируем подпись.
function fwLabel(key: string): { label: string; hint: string } | null {
  if (!key.startsWith("fw_")) return null;
  const word = key.slice(3).replace(/_/g, " ");
  return {
    label: `Маркер «${word}»`,
    hint: "Частотные слова-индикаторы AI-стиля. AI чаще использует обобщающие конструкции.",
  };
}

function getInfo(key: string): { label: string; hint?: string } | null {
  if (FEATURE_INFO[key]) return FEATURE_INFO[key];
  const fw = fwLabel(key);
  if (fw) return fw;
  return null;
}

function featureLabel(key: string): string {
  const info = getInfo(key);
  if (info) return info.label;
  // Fallback: snake_case → "Snake case".
  const human = key.replace(/_/g, " ");
  return human.charAt(0).toUpperCase() + human.slice(1);
}

function featureHint(key: string): string | undefined {
  return getInfo(key)?.hint;
}

export function ResultCard({
  result,
  text,
  onClose,
}: {
  result: AnalyzeResponse;
  text?: string;
  // Опциональный колбэк закрытия. Если передан — рендерим крестик в углу.
  // На странице /me/history/[id] не передаётся (там карточка — единственный
  // контент, закрывается переходом по крошкам).
  onClose?: () => void;
}) {
  const isNoDecision = result.status !== "SUCCESS";
  const risk = result.risk_level ? RISK_STYLES[result.risk_level] : null;
  const explanation = result.explanation;

  return (
    <div className="relative rounded-lg border border-[var(--border)] bg-[var(--background)] overflow-hidden">
      {onClose && (
        <button
          type="button"
          onClick={onClose}
          aria-label="Закрыть результат"
          title="Скрыть результат"
          className="absolute top-2 right-2 z-10 w-7 h-7 flex items-center justify-center rounded-full text-[var(--muted)] hover:bg-[var(--border)]/60 hover:text-[var(--foreground)] transition-colors text-lg leading-none"
        >
          ×
        </button>
      )}
      <div className="p-4 border-b border-[var(--border)]">
        {isNoDecision ? (
          <div className="text-sm pr-8">
            <div className="font-semibold text-[var(--warn)]">
              Решение не принято
            </div>
            <div className="text-[var(--muted)] mt-1">
              Сервис не смог уверенно классифицировать текст. Проверьте
              предупреждения ниже.
            </div>
          </div>
        ) : (
          <div className="flex items-start gap-3 flex-wrap pr-8">
            <div className="flex-1 min-w-0">
              <div className="text-xs text-[var(--muted)] uppercase tracking-wider">
                Вердикт
              </div>
              <div className="text-2xl font-bold mt-1">
                {result.verdict ? VERDICT_LABEL[result.verdict] : "—"}
              </div>
              {result.confidence !== null && result.verdict && (
                <div className="text-sm text-[var(--muted)] mt-1">
                  Уверенность в вердикте:{" "}
                  {(
                    (result.verdict === "ai"
                      ? result.confidence
                      : 1 - result.confidence) * 100
                  ).toFixed(1)}
                  %{" "}
                  <span className="text-xs">
                    (P(AI) = {(result.confidence * 100).toFixed(1)}%)
                  </span>
                </div>
              )}
            </div>
            {risk && (
              <span
                className={`px-3 py-1 rounded-full border text-xs font-semibold uppercase tracking-wider ${risk.cls}`}
              >
                {risk.label}
              </span>
            )}
          </div>
        )}
      </div>

      {!isNoDecision && result.confidence !== null && (
        <ProbabilityScale prob={result.confidence} />
      )}

      <div className="p-4 grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs border-b border-[var(--border)]">
        <Field label="Длина" value={`${result.text_length} симв.`} />
        <Field
          label="Время"
          value={`${result.processing_time_ms.toFixed(1)} мс`}
        />
        <Field label="Обрезано" value={result.was_truncated ? "Да" : "Нет"} />
      </div>

      {explanation && explanation.top_markers.length > 0 && (
        <ExplanationSection
          summary={explanation.summary}
          markers={explanation.top_markers}
          isAI={result.verdict === "ai"}
        />
      )}
      {/* summary от бэка пока не показываем — он дублирует список markers */}

      {text && result.verdict === "ai" && (
        <TextHighlights text={text} markers={explanation?.top_markers ?? []} />
      )}

      {result.warnings.length > 0 && (
        <div className="p-4 border-b border-[var(--border)] bg-amber-50/50 dark:bg-amber-950/10">
          <div className="text-xs font-semibold text-[var(--warn)] uppercase tracking-wider mb-2">
            Предупреждения
          </div>
          <ul className="space-y-1 text-sm">
            {result.warnings.map((w) => (
              <li key={w}>· {WARNING_LABEL[w] || w}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="p-4 text-xs text-[var(--muted)] leading-relaxed">
        {result.disclaimer}
      </div>
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[var(--muted)] uppercase tracking-wider">
        {label}
      </div>
      <div className="font-medium mt-0.5">{value}</div>
    </div>
  );
}

/**
 * Визуальная шкала P(AI) с тремя зонами:
 *   LOW    [0.00, 0.30) — зелёная (вероятно человек)
 *   MEDIUM [0.30, 0.70) — жёлтая (зона неопределённости каскада)
 *   HIGH   [0.70, 1.00] — красная (вероятно AI)
 *
 * Треугольный маркер показывает где оказалась оценка.
 * Пороги совпадают с зонами каскадного детектора и трёхуровневой моделью риска.
 */
function ProbabilityScale({ prob }: { prob: number }) {
  const pct = Math.max(0, Math.min(1, prob)) * 100;
  return (
    <div className="px-4 pt-4 pb-5 border-b border-[var(--border)]">
      <div className="flex items-baseline justify-between mb-2 text-xs">
        <span className="uppercase tracking-wider text-[var(--muted)]">
          Шкала вероятности AI
        </span>
        <span className="font-mono font-semibold text-sm">
          P(AI) = {pct.toFixed(1)}%
        </span>
      </div>

      {/* Шкала: 30%/40%/30% — LOW/MEDIUM/HIGH (совпадает с порогами каскада) */}
      <div className="relative h-3 rounded-full overflow-hidden flex">
        <div className="bg-green-500/70" style={{ width: "30%" }} />
        <div className="bg-amber-400/70" style={{ width: "40%" }} />
        <div className="bg-red-500/70" style={{ width: "30%" }} />

        {/* Маркер положения */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-[var(--foreground)]"
          style={{ left: `calc(${pct}% - 1px)` }}
          aria-hidden
        />
      </div>

      {/* Треугольный индикатор под маркером */}
      <div className="relative h-2 mt-0.5">
        <div
          className="absolute -top-0.5"
          style={{
            left: `calc(${pct}% - 6px)`,
            width: 0,
            height: 0,
            borderLeft: "6px solid transparent",
            borderRight: "6px solid transparent",
            borderBottom: "6px solid var(--foreground)",
          }}
          aria-hidden
        />
      </div>

      {/* Подписи зон */}
      <div className="flex text-[10px] uppercase tracking-wider text-[var(--muted)] mt-1.5">
        <div className="text-left" style={{ width: "30%" }}>
          0&nbsp;– 30
          <div className="text-green-700 dark:text-green-400 font-semibold normal-case tracking-normal">
            вероятно человек
          </div>
        </div>
        <div className="text-center" style={{ width: "40%" }}>
          30&nbsp;– 70
          <div className="text-amber-700 dark:text-amber-400 font-semibold normal-case tracking-normal">
            ручная проверка
          </div>
        </div>
        <div className="text-right" style={{ width: "30%" }}>
          70&nbsp;– 100
          <div className="text-red-700 dark:text-red-400 font-semibold normal-case tracking-normal">
            вероятно AI
          </div>
        </div>
      </div>
    </div>
  );
}

interface Marker {
  feature: string;
  value: number;
  human_baseline: number;
  deviation_percent: number;
  direction: "above" | "below";
}

function ExplanationSection({
  markers,
  isAI,
}: {
  summary: string;
  markers: Marker[];
  isAI: boolean;
}) {
  // Сортируем по силе отклонения, чтобы самые яркие признаки были сверху.
  const sorted = [...markers].sort(
    (a, b) => Math.abs(b.deviation_percent) - Math.abs(a.deviation_percent),
  );

  return (
    <details
      open
      className="border-b border-[var(--border)] bg-[var(--border)]/20"
    >
      <summary className="p-4 cursor-pointer select-none flex items-center justify-between font-semibold text-sm hover:bg-[var(--border)]/40">
        <span>
          {isAI ? "Почему это похоже на ИИ-текст" : "Стилеметрический анализ"}
        </span>
        <span className="text-xs text-[var(--muted)] font-normal">
          {markers.length} признак{plural(markers.length)}
        </span>
      </summary>

      <div className="px-4 pb-4 space-y-4">
        <p className="text-xs text-[var(--muted)] leading-relaxed">
          Сравниваем стиль текста со среднестатистическим человеком на
          русскоязычном корпусе. Полоска показывает, насколько признак
          отличается от нормы. Признаки отсортированы по силе отклонения.
        </p>

        <ul className="space-y-3">
          {sorted.map((m) => (
            <MarkerBar key={m.feature} marker={m} />
          ))}
        </ul>

        <div className="flex items-center justify-center gap-4 text-xs text-[var(--muted)] pt-2 border-t border-[var(--border)]/50">
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-3 h-3 rounded bg-blue-500/70" />
            ниже нормы
          </span>
          <span className="font-medium text-[var(--foreground)]">·норма·</span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-3 h-3 rounded bg-red-500/70" />
            выше нормы
          </span>
        </div>
      </div>
    </details>
  );
}

function plural(n: number): string {
  const mod10 = n % 10;
  const mod100 = n % 100;
  if (mod10 === 1 && mod100 !== 11) return "";
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return "а";
  return "ов";
}

function MarkerBar({ marker }: { marker: Marker }) {
  const dev = marker.deviation_percent; // может быть >0 (above) или <0 (below)
  const sign = marker.direction === "above" ? 1 : -1;
  // Нормируем отклонение в шкалу [0..1] от центра. 100% = край.
  const magnitude = Math.min(Math.abs(dev), 200) / 200; // 0..1
  // Ширина «крыла» бара от центра.
  const wingPct = Math.max(4, magnitude * 50); // от центра до края = 50%

  const isAbove = sign > 0;
  const barColor = isAbove ? "bg-red-500" : "bg-blue-500";

  const hint = featureHint(marker.feature);

  return (
    <li className="space-y-1.5">
      <div className="flex items-baseline justify-between gap-2 text-sm">
        <span className="font-medium">{featureLabel(marker.feature)}</span>
        <span className="text-xs font-mono shrink-0 text-[var(--muted)]">
          {marker.value.toFixed(2)} <span className="opacity-60">/</span>{" "}
          норма {marker.human_baseline.toFixed(2)}
        </span>
      </div>

      {/* Centered bar: норма по центру, отклонение растёт в сторону */}
      <div className="relative h-2 rounded-full bg-[var(--border)] overflow-hidden">
        {/* Центральная риска */}
        <div
          aria-hidden
          className="absolute top-0 bottom-0 left-1/2 w-px bg-[var(--background)]/60"
        />
        {/* Сегмент */}
        <div
          className={`absolute top-0 bottom-0 ${barColor} transition-all`}
          style={
            isAbove
              ? { left: "50%", width: `${wingPct}%` }
              : { right: "50%", width: `${wingPct}%` }
          }
        />
      </div>

      <div className="flex items-baseline justify-between gap-2 text-xs">
        {hint ? (
          <span className="text-[var(--muted)] leading-snug pr-2">{hint}</span>
        ) : (
          <span />
        )}
        <span
          className={`font-mono shrink-0 font-medium ${
            isAbove ? "text-red-600 dark:text-red-400" : "text-blue-600 dark:text-blue-400"
          }`}
        >
          {dev > 0 ? "+" : ""}
          {dev.toFixed(0)}%
        </span>
      </div>
    </li>
  );
}
