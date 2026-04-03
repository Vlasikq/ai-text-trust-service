"""
Topic-matched генерация AI-текстов на темах из human-корпуса.

Устраняет topic confound (DEC-008): AI и human пишут об одном и том же,
разница = чистый стиль генерации.

Для каждого домена:
  1. Извлекает тему из human-текста (заголовок / первое предложение)
  2. Генерирует AI-текст на ту же тему
  3. Сохраняет пару (human_doc_id, topic) в meta

Использование:
  python scripts/generate_topic_matched.py --backend gigachat --domain all --count 2000
  python scripts/generate_topic_matched.py --backend yandex --model yandexgpt-lite --domain news
"""

import argparse
import json
import logging
import random
import re
import sys
import time
import uuid
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_backends import create_backend  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HUMAN_DIR = DATA_DIR / "human"
GEN_DIR = DATA_DIR / "gen_topic_matched"

load_dotenv(PROJECT_ROOT / ".env")

SEED = 42

# ── Извлечение тем из human-данных ──────────────────────────────


def _first_sentence(text: str) -> str:
    """Извлекает первое предложение (до первой точки/!/?)."""
    m = re.match(r"(.+?[.!?])\s", text)
    if m and len(m.group(1)) >= 20:
        return m.group(1).strip()
    # fallback: первые 150 символов до последнего пробела
    chunk = text[:150]
    last_space = chunk.rfind(" ")
    if last_space > 50:
        return chunk[:last_space].strip()
    return chunk.strip()


def _wiki_title(text: str) -> str:
    """Извлекает название статьи Wikipedia из начала текста.

    Формат: 'Название (пояснение) — описание...'
    """
    # убираем ударения и спецсимволы в начале
    clean = re.sub(r"[́̀]", "", text[:300])
    # ищем паттерн 'Слово (что-то) —' или 'Слово —'
    m = re.match(r"^(.+?)\s*[\—\-–]\s", clean)
    if m and len(m.group(1).strip()) >= 3:
        title = m.group(1).strip()
        # оставляем пояснение в скобках если название слишком короткое
        short_title = re.sub(r"\s*\([^)]+\)\s*", " ", title).strip()
        if len(short_title) >= 5:
            return short_title
        return title
    # fallback: первое предложение
    return _first_sentence(text)


def load_human_topics(domain: str) -> list[dict]:
    """Загружает human-тексты и извлекает темы."""
    topics = []

    if domain == "news":
        path = HUMAN_DIR / "taiga_nplus1_with_time.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            topic = _first_sentence(row["text"])
            topics.append({
                "topic": topic,
                "human_doc_id": row["doc_id"],
                "human_source": row["source_name"],
            })

    elif domain == "essay":
        path = HUMAN_DIR / "habr_essays.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            title = row.get("meta", {}).get("title", "")
            if title and len(title) >= 5:
                topic = title
            else:
                topic = _first_sentence(row["text"])
            topics.append({
                "topic": topic,
                "human_doc_id": row["doc_id"],
                "human_source": row["source_name"],
            })

    elif domain == "wiki":
        path = HUMAN_DIR / "wiki_articles.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            topic = _wiki_title(row["text"])
            topics.append({
                "topic": topic,
                "human_doc_id": row["doc_id"],
                "human_source": row["source_name"],
            })

    # фильтруем слишком короткие темы — LLM не сможет написать осмысленный текст
    min_topic_len = 10
    before = len(topics)
    topics = [t for t in topics if len(t["topic"]) >= min_topic_len]
    filtered = before - len(topics)
    if filtered:
        log.info(f"Filtered {filtered} topics shorter than {min_topic_len} chars")

    log.info(f"Loaded {len(topics)} human topics for domain={domain}")
    return topics


# ── Промпт-шаблоны для topic-matched генерации ─────────────────

NEWS_TM_TEMPLATES = [
    {
        "system": (
            "Ты журналист научно-популярного издания."
            " Пишешь информативные новости для широкой аудитории."
        ),
        "user": (
            "Напиши научно-популярную новость на тему, описанную ниже."
            " Не копируй текст дословно — напиши свою версию новости.\n\n"
            "ТЕМА: {topic}\n"
            "ОБЪЁМ: {target_chars}–{target_chars_hi} символов.\n"
            "Формат: сплошной текст без заголовков, списков и markdown."
        ),
    },
    {
        "system": (
            "Ты редактор новостного портала."
            " Пишешь краткие и точные новости."
        ),
        "user": (
            "Напиши новостную заметку по следующей теме."
            " Изложи факты своими словами.\n\n"
            "ТЕМА: {topic}\n"
            "ОБЪЁМ: {target_chars}–{target_chars_hi} символов.\n"
            "Формат: сплошной текст без markdown-разметки."
        ),
    },
    {
        "system": (
            "Ты научный корреспондент."
            " Объясняешь сложные темы простым языком."
        ),
        "user": (
            "Напиши научную новость на указанную тему."
            " Добавь контекст и значение открытия/события.\n\n"
            "ТЕМА: {topic}\n"
            "ОБЪЁМ: {target_chars}–{target_chars_hi} символов.\n"
            "Без заголовков, списков и форматирования."
        ),
    },
]

ESSAY_TM_TEMPLATES = [
    {
        "system": (
            "Ты автор аналитических статей для IT-аудитории."
            " Пишешь вдумчивые обзоры с личным мнением."
        ),
        "user": (
            "Напиши статью-обзор на тему: «{topic}».\n"
            "Раскрой тему с разных сторон, приведи аргументы.\n"
            "ОБЪЁМ: {target_chars}–{target_chars_hi} символов.\n"
            "Формат: сплошной текст без markdown."
        ),
    },
    {
        "system": (
            "Ты блогер на техническом портале."
            " Пишешь длинные посты с примерами из практики."
        ),
        "user": (
            "Напиши пост на тему: «{topic}».\n"
            "Поделись мыслями, опытом, наблюдениями.\n"
            "ОБЪЁМ: {target_chars}–{target_chars_hi} символов.\n"
            "Без заголовков, списков и форматирования."
        ),
    },
    {
        "system": (
            "Ты колумнист технологического издания."
            " Пишешь экспертные колонки."
        ),
        "user": (
            "Напиши экспертную колонку на тему: «{topic}».\n"
            "Сформулируй позицию и аргументируй её.\n"
            "ОБЪЁМ: {target_chars}–{target_chars_hi} символов.\n"
            "Сплошной текст, без markdown-разметки."
        ),
    },
]

WIKI_TM_TEMPLATES = [
    {
        "system": (
            "Ты автор энциклопедических статей."
            " Пишешь нейтрально и информативно."
        ),
        "user": (
            "Напиши энциклопедическую статью о: «{topic}».\n"
            "Стиль: нейтральный, фактологический, без оценок.\n"
            "ОБЪЁМ: {target_chars}–{target_chars_hi} символов.\n"
            "Сплошной текст без списков, заголовков и markdown."
        ),
    },
    {
        "system": (
            "Ты составитель справочных материалов."
            " Пишешь структурированные статьи для учебных целей."
        ),
        "user": (
            "Напиши справочную статью о: «{topic}».\n"
            "Изложи основные факты, историю, значение.\n"
            "ОБЪЁМ: {target_chars}–{target_chars_hi} символов.\n"
            "Формат: сплошной текст без форматирования."
        ),
    },
    {
        "system": (
            "Ты научный редактор."
            " Пишешь точные и проверенные статьи."
        ),
        "user": (
            "Напиши научно-справочную статью о: «{topic}».\n"
            "Опиши суть, контекст и ключевые факты.\n"
            "ОБЪЁМ: {target_chars}–{target_chars_hi} символов.\n"
            "Без markdown, заголовков и списков."
        ),
    },
]

DOMAIN_TM_CONFIG = {
    "news": {
        "templates": NEWS_TM_TEMPLATES,
        "char_range": (1000, 3500),
    },
    "wiki": {
        "templates": WIKI_TM_TEMPLATES,
        "char_range": (2000, 5500),
    },
    "essay": {
        "templates": ESSAY_TM_TEMPLATES,
        "char_range": (1500, 4000),
    },
}


# ── Генерация ───────────────────────────────────────────────────


def load_done_ids(path: Path) -> set[str]:
    """Загружает уже сгенерированные prompt_id для resume."""
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            pid = json.loads(line).get("meta", {}).get("prompt_id")
            if pid:
                done.add(pid)
        except Exception:
            continue
    return done


def build_prompt_table(
    domain: str,
    count: int,
    human_topics: list[dict],
) -> list[dict]:
    """Строит таблицу промптов, привязанных к human-темам."""
    cfg = DOMAIN_TM_CONFIG[domain]
    templates = cfg["templates"]
    lo, hi = cfg["char_range"]

    rng = random.Random(SEED)
    # перемешиваем human-темы, берём первые count
    shuffled = human_topics.copy()
    rng.shuffle(shuffled)
    selected = shuffled[:count]

    prompts = []
    for i, ht in enumerate(selected):
        target_chars = rng.randint(lo, hi)
        temperature = rng.choice([0.7, 0.8, 0.9, 1.0])
        template = rng.choice(templates)

        user_msg = template["user"].format(
            topic=ht["topic"],
            target_chars=target_chars,
            target_chars_hi=target_chars + 500,
        )

        prompt_id = f"tm_{domain}_{i:05d}"
        prompts.append({
            "prompt_id": prompt_id,
            "domain": domain,
            "topic": ht["topic"],
            "human_doc_id": ht["human_doc_id"],
            "human_source": ht["human_source"],
            "target_chars": target_chars,
            "temperature": temperature,
            "system": template["system"],
            "user": user_msg,
        })

    return prompts


def generate_topic_matched(
    llm,
    domain: str,
    count: int,
    human_topics: list[dict],
    sleep_s: float = 0.3,
) -> Path:
    """Генерирует topic-matched AI-тексты и сохраняет в JSONL."""
    GEN_DIR.mkdir(parents=True, exist_ok=True)

    model_safe = llm.model_name.replace("/", "_").replace(":", "_")
    out_path = GEN_DIR / f"{model_safe}_{domain}.jsonl"

    done_ids = load_done_ids(out_path)
    prompts = build_prompt_table(domain, count, human_topics)
    remaining = [p for p in prompts if p["prompt_id"] not in done_ids]

    log.info(
        f"[topic-matched] {model_safe}/{domain}: "
        f"total={len(prompts)}, done={len(done_ids)}, remaining={len(remaining)}"
    )

    if not remaining:
        log.info("Nothing to generate — all done.")
        return out_path

    generated = 0
    errors = 0

    with out_path.open("a", encoding="utf-8") as f:
        for i, p in enumerate(remaining):
            text = llm.generate(p["system"], p["user"], p["temperature"])

            # убираем <think>...</think> блок (Qwen, DeepSeek и др.)
            if text and re.search(r"<think>", text):
                text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text).strip()

            if text is None or len(text) < 200:
                errors += 1
                log.warning(
                    f"[{i+1}/{len(remaining)}] SKIP {p['prompt_id']}: "
                    f"empty or too short"
                )
                continue

            row = {
                "text": text,
                "label": "ai",
                "domain": p["domain"],
                "source_name": llm.model_name,
                "doc_id": f"ai_tm_{uuid.uuid4()}",
                "created_at": str(date.today()),
                "meta": {
                    "prompt_id": p["prompt_id"],
                    "topic": p["topic"],
                    "human_doc_id": p["human_doc_id"],
                    "human_source": p["human_source"],
                    "target_chars": p["target_chars"],
                    "temperature": p["temperature"],
                    "actual_chars": len(text),
                    "topic_matched": True,
                },
            }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            generated += 1

            if (i + 1) % 50 == 0 or (i + 1) == len(remaining):
                log.info(
                    f"[{i+1}/{len(remaining)}] Generated: {generated}, "
                    f"Errors: {errors}, Last len: {len(text)}"
                )

            time.sleep(sleep_s)

    log.info(
        f"Done: {generated} generated, {errors} errors -> {out_path}"
    )
    return out_path


# ── CLI ─────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate topic-matched AI texts from human corpus topics",
    )
    p.add_argument(
        "--backend",
        choices=["openai", "gigachat", "yandex", "gemini"],
        default="gigachat",
    )
    p.add_argument("--model", default=None)
    p.add_argument(
        "--domain",
        choices=["news", "wiki", "essay", "all"],
        default="all",
    )
    p.add_argument(
        "--count",
        type=int,
        default=2000,
        help="Texts per domain (default: 2000)",
    )
    p.add_argument("--sleep", type=float, default=0.3)
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-url", default=None, help="Custom API base URL (for Groq etc.)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    domains = (
        ["news", "wiki", "essay"]
        if args.domain == "all"
        else [args.domain]
    )

    llm = create_backend(args.backend, args.model, args.base_url, args.api_key)
    log.info(f"Backend: {llm.model_name}")

    for domain in domains:
        human_topics = load_human_topics(domain)
        if not human_topics:
            log.error(f"No human topics found for domain={domain}")
            continue

        actual_count = min(args.count, len(human_topics))
        log.info(
            f"=== {llm.model_name} / {domain} / "
            f"{actual_count} topic-matched texts ==="
        )
        generate_topic_matched(
            llm, domain, actual_count, human_topics, sleep_s=args.sleep,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
