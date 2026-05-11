"""
Style-based explanation: compare stylometric features against human baselines.

Вердикт (human/ai) задаётся детектором (TF-IDF / каскад / трансформер).
Объяснение — эвристика относительно средних по in-domain human-корпусу;
на текстах вне распределения обучения маркеры могут быть неинформативны.
API добавляет предупреждение STYLE_EXPLANATION_HEURISTIC при explain=true.
"""

import json
import logging
from pathlib import Path

from app.explanation._phrases import find_matched_phrases
from app.features.stylometric import extract_all
from app.schemas import Explanation, StyleMarker

log = logging.getLogger(__name__)

# Локализация имён признаков для summary. Должна быть в точности согласована
# с ResultCard.tsx: FEATURE_INFO — иначе UI и summary будут говорить разными
# словами о тех же признаках.
# fw_*-маркеры обрабатываются отдельно через `_fw_label` (динамика по словарю).
_FEATURE_LABELS: dict[str, str] = {
    # base
    "text_len": "длина текста",
    "word_count": "количество слов",
    "sentence_count": "количество предложений",
    "avg_word_len": "средняя длина слова",
    "avg_sent_len": "средняя длина предложения",
    "comma_density": "плотность запятых",
    "question_marks": "вопросительные знаки",
    "exclamation_marks": "восклицательные знаки",
    "ttr": "лексическое разнообразие (TTR)",
    "ends_formulaic": "шаблонная концовка",
    # lexical
    "std_word_len": "разброс длины слов",
    "max_word_len": "максимальная длина слова",
    "hapax_ratio": "доля уникальных слов",
    "long_word_ratio": "доля длинных слов (>10 букв)",
    "short_word_ratio": "доля коротких слов (<3 букв)",
    "cyrillic_ratio": "доля кириллицы",
    "digit_ratio": "доля цифр",
    # punctuation
    "semicolons": "точки с запятой",
    "colons": "двоеточия",
    "dashes": "длинные тире",
    "parens": "скобки",
    "quotes_guillemet": "кавычки-«ёлочки»",
    "ellipsis": "многоточия",
    "punct_density": "плотность пунктуации",
    # structure
    "sentence_len_std": "разброс длин предложений",
    "first_sentence_len": "длина первого предложения",
    "last_sentence_len": "длина последнего предложения",
    "max_sent_len": "длина самого длинного предложения",
    # ngram entropy
    "char_bigram_entropy": "энтропия биграмм символов",
    "char_trigram_entropy": "энтропия триграмм символов",
    "word_bigram_entropy": "энтропия биграмм слов",
    # repetition
    "self_repetition_rate": "самоповтор (n-граммы)",
    "mattr_50": "скользящий TTR (MATTR-50)",
}

# Признаки с очень низкой человеческой частотой (baseline < этого порога) и
# value=0 в анализируемом тексте — нулевая пунктуация в коротком тексте даёт
# отклонение -100% и забивает топ-маркеры мусором (см. SECURITY review).
_LOW_FREQ_BASELINE = 0.05


def _fw_label(feature: str) -> str | None:
    """Подпись для функциональных маркеров `fw_<слово>`."""
    if not feature.startswith("fw_"):
        return None
    word = feature[3:].replace("_", " ")
    return f"маркер «{word}»"


def _human_label(feature: str) -> str:
    """Человекочитаемая подпись признака; fallback — само имя."""
    if feature in _FEATURE_LABELS:
        return _FEATURE_LABELS[feature]
    fw = _fw_label(feature)
    if fw is not None:
        return fw
    return feature


class StyleExplainer:
    def __init__(self, baselines_path: Path):
        self._baselines: dict[str, float] = {}
        self._baselines_path = baselines_path

    def load(self) -> None:
        if not self._baselines_path.exists():
            log.warning("Human baselines not found at %s", self._baselines_path)
            return
        with open(self._baselines_path, encoding="utf-8") as f:
            self._baselines = json.load(f)
        log.info("Loaded %d human baselines", len(self._baselines))

    def is_ready(self) -> bool:
        return len(self._baselines) > 0

    def explain(self, text: str, top_n: int = 5) -> Explanation:
        """Извлечь признаки, сравнить с baseline, вернуть top-N отклонений.

        Фильтр шумовых маркеров: признак с `value == 0` и низким baseline
        (низкочастотная пунктуация типа `colons`, `dashes`, `parens`) даёт
        -100% отклонение и не несёт сигнала — его выкидываем, чтобы не
        забивать UI. Признак `ends_formulaic` бинарный, поэтому 0/-100%
        для него — валидный сигнал и пропускается через явный whitelist.
        """
        features_df = extract_all(text)
        features = features_df.iloc[0].to_dict()

        deviations: list[StyleMarker] = []
        for feat, value in features.items():
            baseline = self._baselines.get(feat)
            if baseline is None or baseline == 0:
                continue

            value_f = float(value)
            # Пропускаем «пустой» сигнал по низкочастотным признакам.
            if value_f == 0 and abs(baseline) < _LOW_FREQ_BASELINE and feat != "ends_formulaic":
                continue

            dev_pct = ((value_f - baseline) / abs(baseline)) * 100
            deviations.append(
                StyleMarker(
                    feature=feat,
                    value=round(value_f, 4),
                    human_baseline=round(baseline, 4),
                    deviation_percent=round(dev_pct, 1),
                    direction="above" if value_f > baseline else "below",
                )
            )

        # Сортировка по абсолютному отклонению; топ-N — для UI и summary.
        deviations.sort(key=lambda m: abs(m.deviation_percent), reverse=True)
        top = deviations[:top_n]

        summary = self._build_summary(top)
        matched_phrases = find_matched_phrases(text)
        return Explanation(
            top_markers=top,
            summary=summary,
            matched_phrases=matched_phrases,
        )

    def _build_summary(self, markers: list[StyleMarker]) -> str:
        if not markers:
            return "Стилометрических отклонений не обнаружено."

        parts = []
        for m in markers[:3]:
            label = _human_label(m.feature)
            direction = "выше" if m.direction == "above" else "ниже"
            parts.append(f"{label} на {abs(m.deviation_percent):.0f}% {direction} нормы")

        return "Ключевые отклонения: " + "; ".join(parts) + "."
