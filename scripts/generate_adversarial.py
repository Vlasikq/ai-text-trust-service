"""
Генерация adversarial test set, парафраз AI-текстов через LLM.

Берём AI-тексты из data/splits/test.jsonl, просим LLM переписать их
"как человек" чтобы проверить устойчивость детектора к обходу.

"""

import argparse
import json
import logging
import random
import sys
import time
import uuid
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
ADV_DIR = DATA_DIR / "adversarial"

load_dotenv(PROJECT_ROOT / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_backends import LLMBackend, create_backend  # noqa: E402

SEED = 42


PARAPHRASE_TEMPLATES = [
    {
        "system": "Ты редактор-фрилансер. Тебе дают черновик — ты переписываешь его своим языком.",
        "user": (
            "Перепиши этот текст полностью своими словами. "
            "Сохрани смысл и основные факты, но измени структуру, формулировки и порядок абзацев. "
            "Добавь немного разговорности — как будто объясняешь другу. "
            "Не используй списки и эмодзи. Пиши только переписанный текст.\n\n"
            "ТЕКСТ:\n{text}"
        ),
    },
    {
        "system": "Ты студент, готовишь пересказ статьи для семинара.",
        "user": (
            "Перескажи этот текст своими словами для выступления на семинаре. "
            "Передай основные идеи, но используй свои формулировки. "
            "Можно упростить сложные места. Пиши естественно, не как робот. "
            "Только текст пересказа, без вступлений вроде «Вот пересказ».\n\n"
            "ТЕКСТ:\n{text}"
        ),
    },
    {
        "system": "Ты журналист. Переписываешь чужой материал, чтобы он звучал оригинально.",
        "user": (
            "Перепиши этот текст так, чтобы он звучал как твой авторский материал. "
            "Измени структуру предложений, замени слова синонимами, "
            "переставь абзацы местами. Смысл должен остаться прежним. "
            "Пиши только готовый текст.\n\n"
            "ТЕКСТ:\n{text}"
        ),
    },
    {
        "system": "Ты копирайтер. Клиент прислал текст и просит сделать рерайт.",
        "user": (
            "Сделай глубокий рерайт этого текста. "
            "Не просто замени слова — перестрой предложения, "
            "измени способ подачи информации, добавь переходы между мыслями. "
            "Текст должен звучать свежо и естественно. "
            "Без списков и заголовков. Только финальный текст.\n\n"
            "ТЕКСТ:\n{text}"
        ),
    },
]

BACK_TRANSLATION_TEMPLATE = {
    "system": "Ты профессиональный переводчик.",
    "user_to_en": (
        "Переведи следующий русский текст на английский язык. "
        "Переводи литературно, не дословно. Только перевод, без пояснений.\n\n"
        "ТЕКСТ:\n{text}"
    ),
    "user_to_ru": (
        "Переведи следующий английский текст на русский язык. "
        "Переводи литературно и естественно. Только перевод, без пояснений.\n\n"
        "TEXT:\n{text}"
    ),
}



def load_ai_texts_from_test() -> list[dict]:
    path = SPLITS_DIR / "test.jsonl"
    return [
        row for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
        if row["label"] == 1
    ]


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            oid = json.loads(line).get("meta", {}).get("original_doc_id")
            if oid:
                done.add(oid)
        except Exception:
            continue
    return done


def generate_paraphrase(
    llm: LLMBackend,
    texts: list[dict],
    count: int,
    method: str,
    sleep_s: float,
) -> Path:
    ADV_DIR.mkdir(parents=True, exist_ok=True)

    out_path = ADV_DIR / f"adversarial_{method}_{llm.model_name}.jsonl"
    done_ids = load_done_ids(out_path)
    rng = random.Random(SEED)
    rng.shuffle(texts)
    remaining = [t for t in texts if t.get("doc_id", "") not in done_ids][:count]
    log.info(f"Done: {len(done_ids)}, remaining: {len(remaining)}")

    if not remaining:
        return out_path

    generated = 0
    errors = 0

    with out_path.open("a", encoding="utf-8") as f:
        for i, src in enumerate(remaining):
            original_text = src["text"]

            if method == "paraphrase":
                template = rng.choice(PARAPHRASE_TEMPLATES)
                system = template["system"]
                user = template["user"].format(text=original_text[:6000])
                result = llm.generate(system, user, temperature=0.9)

            elif method == "back-translation":
                # Шаг 1: ru → en
                bt = BACK_TRANSLATION_TEMPLATE
                en_text = llm.generate(bt["system"], bt["user_to_en"].format(text=original_text[:6000]), temperature=0.3)
                if not en_text:
                    errors += 1
                    continue
                time.sleep(sleep_s)
                # Шаг 2: en → ru
                result = llm.generate(bt["system"], bt["user_to_ru"].format(text=en_text[:6000]), temperature=0.3)
            else:
                raise ValueError(f"Unknown method: {method}")

            if not result or len(result) < 200:
                errors += 1
                log.warning(f"[{i+1}/{len(remaining)}] SKIP: empty or too short")
                continue

            row = {
                "text": result,
                "label": "ai",
                "domain": src["domain"],
                "source_model": f"adversarial_{method}",
                "source_name": llm.model_name,
                "doc_id": f"adv_{uuid.uuid4()}",
                "created_at": str(date.today()),
                "meta": {
                    "method": method,
                    "paraphrase_model": llm.model_name,
                    "original_source_model": src.get("source_model", "unknown"),
                    "original_doc_id": src.get("doc_id", ""),
                    "original_len": len(original_text),
                    "adversarial_len": len(result),
                },
            }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            generated += 1

            if (i + 1) % 25 == 0 or (i + 1) == len(remaining):
                log.info(f"[{i+1}/{len(remaining)}] Generated: {generated}, Errors: {errors}")

            time.sleep(sleep_s)

    log.info(f"Done: {generated} generated, {errors} errors -> {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate adversarial test set via LLM paraphrase")
    p.add_argument("--backend", choices=["yandex", "gigachat", "openai", "gemini"], default="yandex")
    p.add_argument("--model", default=None)
    p.add_argument("--method", choices=["paraphrase", "back-translation"], default="paraphrase")
    p.add_argument("--count", type=int, default=200, help="Number of texts to paraphrase")
    p.add_argument("--sleep", type=float, default=0.5)
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    llm = create_backend(args.backend, args.model, args.base_url, args.api_key)
    ai_texts = load_ai_texts_from_test()
    log.info(f"{llm.model_name} / {args.method} / {args.count} texts ({len(ai_texts)} in test)")

    if not ai_texts:
        log.error("No AI texts in test split!")
        return 1

    generate_paraphrase(llm, ai_texts, args.count, args.method, args.sleep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
