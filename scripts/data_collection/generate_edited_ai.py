"""
Генерация post-edited AI текстов для оценки robustness к редактированию.

Берём AI-тексты из data/splits/test_tm.jsonl, просим LLM выполнить
редактирование разной глубины. Цель — проверить, как cascade реагирует
на текст, который AI-сгенерировал, а человек (или LLM-симуляция) отредактировал.

4 уровня:
  - raw:        исходный AI-текст (контроль)
  - light_edit: исправить 2-3 шероховатости, не менять структуру
  - heavy_edit: переписать 30-40% предложений своими словами
  - mixed:      первые 50% — human текст, вторые 50% — AI на ту же тему

Результат: data/edited_ai/*.jsonl
"""

import argparse
import json
import logging
import random
import sys
import time
import uuid
from difflib import SequenceMatcher
from pathlib import Path

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
OUTPUT_DIR = DATA_DIR / "edited_ai"

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / ".env.local", override=True)  # personal API keys

# llm_backends живёт рядом — auto-resolution через sys.path[0] при запуске скрипта.
from llm_backends import create_backend

# segment_scoring живёт в scripts/ (notebook-обёртка) — добавляем родительскую папку.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from segment_scoring import split_sentences  # noqa: E402

SEED = 42

# ---------------------------------------------------------------------------
# Промпты для уровней редактирования
# ---------------------------------------------------------------------------

EDIT_TEMPLATES = {
    "light_edit": {
        "system": (
            "Ты редактор. Тебе дают черновик — нужно сделать минимальную правку. "
            "Исправь 2-3 неловкие формулировки, убери канцеляризмы если есть. "
            "НЕ переписывай текст целиком. Структура, порядок абзацев и >90% слов "
            "должны остаться как есть."
        ),
        "user": (
            "Сделай лёгкую редакторскую правку этого текста. "
            "Исправь только самые заметные шероховатости (2-3 места). "
            "Не переписывай, не добавляй новые абзацы. Верни только отредактированный текст.\n\n"
            "ТЕКСТ:\n{text}"
        ),
    },
    "heavy_edit": {
        "system": (
            "Ты опытный редактор. Тебе дают текст — нужно серьёзно переработать его. "
            "Перепиши 30-40% предложений другими словами, измени порядок некоторых абзацев, "
            "добавь свои связки между мыслями. Текст должен звучать как твой."
        ),
        "user": (
            "Серьёзно отредактируй этот текст:\n"
            "- Перепиши 30-40% предложений своими словами\n"
            "- Измени порядок подачи где уместно\n"
            "- Добавь свои переходы между мыслями\n"
            "- Убери шаблонные обороты\n"
            "Верни только итоговый текст.\n\n"
            "ТЕКСТ:\n{text}"
        ),
    },
}


# ---------------------------------------------------------------------------
# Загрузка исходных текстов
# ---------------------------------------------------------------------------


def load_ai_texts(split_path: Path, max_count: int = 200) -> list[dict]:
    """Загрузить AI-тексты (label=1) из split файла."""
    texts = []
    with open(split_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["label"] == 1:
                texts.append(rec)
    random.seed(SEED)
    random.shuffle(texts)
    return texts[:max_count]


def load_human_texts_by_domain(split_path: Path) -> dict[str, list[str]]:
    """Загрузить human-тексты (label=0) сгруппированные по домену."""
    by_domain: dict[str, list[str]] = {}
    with open(split_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["label"] == 0:
                by_domain.setdefault(rec["domain"], []).append(rec["text"])
    return by_domain


# ---------------------------------------------------------------------------
# Генерация
# ---------------------------------------------------------------------------


def generate_edit(backend, text: str, edit_level: str, temperature: float = 0.7) -> str | None:
    """Применить LLM-редактирование заданного уровня."""
    template = EDIT_TEMPLATES[edit_level]
    return backend.generate(
        system=template["system"],
        user=template["user"].format(text=text),
        temperature=temperature,
    )


def apply_sentence_replacement(
    text: str, backend, replace_ratio: float, sleep_s: float = 1.0,
) -> str:
    """Заменить ровно replace_ratio предложений LLM-перефразированием.

    Механическая замена: текст разбивается на предложения, случайно
    выбирается N% из них, каждое отправляется в LLM по отдельности.
    Контроль степени изменения точный и воспроизводимый.
    """
    sents = split_sentences(text)
    if len(sents) < 3:
        return text

    n_replace = max(1, round(len(sents) * replace_ratio))
    indices = sorted(random.sample(range(len(sents)), min(n_replace, len(sents))))

    result = list(sents)
    for idx in indices:
        rewritten = backend.generate(
            system=(
                "Перепиши одно предложение другими словами. "
                "Сохрани смысл и длину. Верни только переписанное предложение, "
                "без кавычек и пояснений."
            ),
            user=sents[idx],
            temperature=0.7,
        )
        if rewritten and len(rewritten.strip()) > 10:
            result[idx] = rewritten.strip()
        time.sleep(sleep_s)

    return " ".join(result)


def generate_mixed(human_text: str, ai_text: str) -> str:
    """Собрать mixed-текст: первая половина human, вторая половина AI.

    label остаётся 1 (AI) — для eval смотрим на confidence distribution,
    а не на бинарную accuracy. Mixed-тексты должны попадать в MEDIUM-зону.
    """
    human_sents = split_sentences(human_text)
    ai_sents = split_sentences(ai_text)
    mid_h = len(human_sents) // 2
    mid_a = len(ai_sents) // 2
    mixed = " ".join(human_sents[:mid_h] + ai_sents[mid_a:])
    return mixed


def make_record(
    text: str, edit_level: str, editor_model: str, original: dict, **extra
) -> dict:
    edit_ratio = SequenceMatcher(None, original["text"], text).ratio()
    return {
        "text": text,
        "label": 1,
        "domain": original["domain"],
        "source_model": original.get("source_model", "unknown"),
        "source_name": f"edited_{edit_level}",
        "doc_id": f"edited_{uuid.uuid4().hex[:8]}",
        "meta": {
            "edit_level": edit_level,
            "editor_model": editor_model,
            "original_doc_id": original["doc_id"],
            "original_length": len(original["text"]),
            "edited_length": len(text),
            "edit_ratio": round(edit_ratio, 4),
            **extra,
        },
    }


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Saved %d records → %s", len(records), path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate post-edited AI texts")
    parser.add_argument("--backend", required=True, help="LLM backend (gigachat, openai, ...)")
    parser.add_argument("--model", default=None, help="Model name override")
    parser.add_argument(
        "--edit-level",
        choices=["light_edit", "full_rewrite", "replace_10", "replace_30", "replace_50", "mixed", "all"],
        default="all",
    )
    parser.add_argument("--count", type=int, default=100, help="Texts per level")
    parser.add_argument("--split", type=Path, default=SPLITS_DIR / "test_tm.jsonl")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--sleep", type=float, default=1.0, help="Delay between API calls")
    parser.add_argument("--base-url", default=None, help="Custom API base URL (for Groq etc.)")
    parser.add_argument("--api-key", default=None, help="API key override")
    args = parser.parse_args()

    random.seed(SEED)

    backend = create_backend(args.backend, args.model, args.base_url, args.api_key)
    model_name = args.model or args.backend

    ai_texts = load_ai_texts(args.split, args.count)
    human_by_domain = load_human_texts_by_domain(args.split)

    ALL_LEVELS = ["light_edit", "replace_10", "replace_30", "replace_50", "full_rewrite", "mixed"]
    levels = ALL_LEVELS if args.edit_level == "all" else [args.edit_level]

    # Маппинг replace_N → ratio
    REPLACE_RATIOS = {"replace_10": 0.10, "replace_30": 0.30, "replace_50": 0.50}

    for level in levels:
        log.info("=== Level: %s, model: %s ===", level, model_name)
        records = []

        for i, orig in enumerate(ai_texts):
            if level == "mixed":
                domain_humans = human_by_domain.get(orig["domain"], [])
                if not domain_humans:
                    continue
                human_text = random.choice(domain_humans)
                edited = generate_mixed(human_text, orig["text"])
            elif level in REPLACE_RATIOS:
                edited = apply_sentence_replacement(
                    orig["text"], backend, REPLACE_RATIOS[level], sleep_s=args.sleep,
                )
            elif level == "full_rewrite":
                edited = generate_edit(backend, orig["text"], "heavy_edit")
            else:
                edited = generate_edit(backend, orig["text"], level)

            if not edited or len(edited) < 100:
                log.warning("Skip %d: empty or too short", i)
                continue

            records.append(make_record(edited, level, model_name, orig))
            log.info("[%d/%d] %s: %d → %d chars (ratio=%.3f)",
                     i + 1, len(ai_texts), level,
                     len(orig["text"]), len(edited),
                     records[-1]["meta"]["edit_ratio"])

            if level not in ("mixed",) and level not in REPLACE_RATIOS:
                time.sleep(args.sleep)

        save_jsonl(records, args.output_dir / f"edited_{level}_{model_name}.jsonl")

    log.info("Done.")


if __name__ == "__main__":
    main()
