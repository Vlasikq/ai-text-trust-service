// Подписи и цвета verdict/risk/warning. Регистр ключей повторяет значения API.

import type { Verdict, RiskLevel } from "./api";

export const VERDICT_LABEL_FULL: Record<Verdict, string> = {
  ai: "Сгенерировано ИИ",
  human: "Написано человеком",
};

export const VERDICT_LABEL_SHORT: Record<Verdict, string> = {
  ai: "ИИ",
  human: "Человек",
};

// Hex для recharts — SVG-fill не понимает CSS-переменные.
export const VERDICT_HEX: Record<Verdict, string> = {
  ai: "#ef4444",
  human: "#22c55e",
};

export const RISK_HEX: Record<RiskLevel, string> = {
  HIGH: "#ef4444",
  MEDIUM: "#f59e0b",
  LOW: "#22c55e",
};

export const RISK_STYLE: Record<RiskLevel, { label: string; cls: string }> = {
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

// Компактный бейдж риска без рамки — для списков.
export const RISK_BADGE_CLS: Record<RiskLevel, string> = {
  HIGH: "bg-red-100 text-red-800 dark:bg-red-950/40 dark:text-red-300",
  MEDIUM: "bg-amber-100 text-amber-800 dark:bg-amber-950/40 dark:text-amber-300",
  LOW: "bg-green-100 text-green-800 dark:bg-green-950/40 dark:text-green-300",
};

// Цветная точка-индикатор: сначала по риску, в no-decision случае — по вердикту.
export function verdictDotClass(
  verdict: Verdict | null | undefined,
  risk: RiskLevel | null | undefined,
): string {
  if (risk === "HIGH") return "bg-red-500";
  if (risk === "MEDIUM") return "bg-amber-500";
  if (risk === "LOW") return "bg-green-500";
  if (verdict === "ai") return "bg-red-500";
  if (verdict === "human") return "bg-green-500";
  return "bg-gray-400";
}

// Ключи совпадают со значениями WarningCode enum на бэке.
export const WARNING_LABEL: Record<string, string> = {
  TEXT_TOO_SHORT: "Текст короткий — точность снижена",
  TEXT_TRUNCATED: "Текст обрезан до 8000 символов",
  TRANSFORMER_UNAVAILABLE: "Доступен только TF-IDF (трансформер недоступен)",
  LOW_CONFIDENCE: "Низкая уверенность",
  CALIBRATION_UNAVAILABLE: "Калибровка отключена",
  INFERENCE_TIMEOUT: "Превышено время инференса",
  STYLE_EXPLANATION_HEURISTIC: "Объяснение — эвристическое",
};
