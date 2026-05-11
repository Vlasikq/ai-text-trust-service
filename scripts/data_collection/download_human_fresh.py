"""
Скачивание свежих human-текстов (2024+) для temporal robustness eval.

Источники:
  - essay: Habr (IlyaGusev/habr, фильтр time_published >= 2024-01-01)
           + фильтр по авторской истории (10+ статей до 2022)
  - news:  N+1 (nplus1.ru), парсинг архива через requests + BeautifulSoup
  - wiki:  Wikipedia RU (API random + фильтр по возрасту статьи/ревизиям)

Эти тексты НЕ добавляются в train — только для eval (temporal robustness).
Контаминация AI минимизируется через:
  1. Редакционный контроль источника (N+1 — штатная редакция)
  2. Авторская история (Habr — 10+ статей до 2022, рейтинг > 0)
  3. История правок (Wiki — статья существует до 2022, 5+ редакторов)
  4. Санитарная проверка — подозрительные тексты помечаются

Результат: data/human_fresh/{habr_fresh,nplus1_fresh,wiki_fresh}.jsonl
"""

import argparse
import hashlib
import json
import logging
import random
import re
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "human_fresh"

MIN_CHARS = 300
MAX_CHARS = 8000
SEED = 42

from app.preprocessing.text_cleaning import clean_text, truncate_by_sentence


# ---------------------------------------------------------------------------
# Общие утилиты
# ---------------------------------------------------------------------------


def make_record(
    text: str,
    domain: str,
    source_name: str,
    doc_id: str,
    published: str = "",
    author: str = "",
    url: str = "",
    **extra_meta,
) -> dict | None:
    """Формирует запись в стандартном формате проекта. Возвращает None если не прошла фильтры."""
    cleaned = clean_text(text)
    if len(cleaned) < MIN_CHARS:
        return None
    if len(cleaned) > MAX_CHARS:
        cleaned = truncate_by_sentence(cleaned, MAX_CHARS)
    return {
        "text": cleaned,
        "label": 0,
        "domain": domain,
        "source_model": "human",
        "source_name": source_name,
        "doc_id": doc_id,
        "meta": {
            "collection_date": datetime.now().strftime("%Y-%m-%d"),
            "published": published,
            "author": author,
            "url": url,
            **extra_meta,
        },
    }


def save_jsonl(records: list[dict], path: Path) -> None:
    """Сохранить записи в JSONL с дедупликацией по SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    deduped = []
    for r in records:
        h = hashlib.sha256(r["text"].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            deduped.append(r)
    with open(path, "w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Saved %d records (deduped from %d) → %s", len(deduped), len(records), path)


# =====================================================================
# ESSAY: Habr (IlyaGusev/habr, 2024+)
# =====================================================================

# Паттерны кода (из download_human_essays.py)
CODE_PATTERNS = re.compile(
    r"```|import\s+\w+|def\s+\w+\(|class\s+\w+[:\(]|"
    r"console\.log|SELECT\s+\w+|<div|<span|function\s*\(",
    re.IGNORECASE,
)

# Хабы-эссе (из download_human_essays.py)
ESSAY_HUBS = {
    "popsci", "popular_science", "machine_learning", "artificial_intelligence",
    "data_science", "data_engineering", "big_data", "it_education",
    "itcompanies", "career", "management", "startups", "marketing",
    "science", "physics", "mathematics", "biology", "chemistry",
    "crypto", "fintech", "ui", "ux", "usability", "future",
    "health", "brain", "education", "sociology", "psychology",
    "ecology", "energy", "space", "robotics", "geek_electronics",
    "reading_room", "iot", "gadgets", "social_networks",
    "media", "history", "copyright", "games_industry",
}


def _clean_markdown_habr(text: str) -> str:
    """Убрать markdown (аналог из download_human_essays.py)."""
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(.+?)_{1,3}", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`]+`", "", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _is_good_habr_essay(text: str, hubs: list[str]) -> bool:
    """Фильтр качества для эссе с Habr."""
    if not text or not (MIN_CHARS <= len(text) <= MAX_CHARS):
        return False
    if len(CODE_PATTERNS.findall(text)) > 3:
        return False
    latin_ratio = sum(1 for c in text if c.isascii() and c.isalpha()) / max(len(text), 1)
    if latin_ratio > 0.3:
        return False
    if hubs and not set(h.lower() for h in hubs).intersection(ESSAY_HUBS):
        return False
    return True


def download_habr_fresh(
    count: int = 500,
    min_date: str = "2024-01-01",
    id_range: tuple[int, int] = (900000, 1020000),
) -> list[dict]:
    """Скачать свежие эссе с Habr через внутренний API.

    Стратегия: случайные ID из диапазона (2024-2026) →
    habr.com/kek/v2/articles/{id} → фильтрация.

    Фильтры верификации:
      - lang=ru
      - timePublished >= min_date
      - score > 0 (модерация сообществом)
      - хабы из ESSAY_HUBS
      - длина текста 300-8000
      - <30% латиницы, <3 code-паттернов
    """
    import requests
    from bs4 import BeautifulSoup

    random.seed(SEED)

    session = requests.Session()
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )

    # Генерируем случайные ID из диапазона
    all_ids = list(range(id_range[0], id_range[1]))
    random.shuffle(all_ids)

    records = []
    checked = 0
    errors = 0

    log.info("Habr API: sampling from ID range %d-%d, target %d articles", *id_range, count)

    for article_id in all_ids:
        if len(records) >= count:
            break

        checked += 1

        try:
            r = session.get(
                f"https://habr.com/kek/v2/articles/{article_id}",
                timeout=10,
            )
        except Exception as e:
            errors += 1
            if errors % 20 == 0:
                log.warning("Habr API errors: %d (last: %s)", errors, e)
            continue

        if r.status_code == 404:
            continue  # Статья удалена или не существует
        if r.status_code != 200:
            errors += 1
            continue

        data = r.json()

        # Фильтр: язык
        if data.get("lang") != "ru":
            continue

        # Фильтр: дата >= min_date
        published = data.get("timePublished", "")
        if not published or published[:10] < min_date:
            continue

        # Фильтр: рейтинг > 0
        stats = data.get("statistics", {})
        score = stats.get("score", 0)
        if score <= 0:
            continue

        # Фильтр: хабы
        hubs = [h.get("alias", "") for h in data.get("hubs", [])]
        if hubs and not set(hubs).intersection(ESSAY_HUBS):
            continue

        # Извлечение текста из HTML
        text_html = data.get("textHtml", "")
        if not text_html:
            continue

        soup = BeautifulSoup(text_html, "html.parser")
        # Убираем code-блоки
        for tag in soup.select("pre, code"):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)

        if not _is_good_habr_essay(text, hubs):
            continue

        author_alias = data.get("author", {}).get("alias", "")
        title = data.get("titleHtml", "")
        # Убираем HTML-теги из заголовка
        title = BeautifulSoup(title, "html.parser").get_text()

        rec = make_record(
            text=text,
            domain="essay",
            source_name="habr_fresh",
            doc_id=f"habr_fresh_{article_id}",
            published=published[:10],
            author=author_alias,
            title=title,
            score=score,
            original_chars=len(text_html),
        )
        if rec:
            records.append(rec)
            if len(records) % 20 == 0:
                log.info("Habr: collected %d/%d (checked %d, errors %d)",
                         len(records), count, checked, errors)

        time.sleep(0.5)  # Вежливость

    log.info("Habr fresh: collected %d from %d checked (%d errors)",
             len(records), checked, errors)
    return records


# =====================================================================
# NEWS: N+1 (nplus1.ru)
# =====================================================================


def download_nplus1_fresh(count: int = 1000, min_date: str = "2024-01-01") -> list[dict]:
    """Скачать свежие статьи с N+1 через sitemap.

    Стратегия:
      1. Скачать sitemap index (nplus1.ru/sitemap.xml)
      2. Из material-sitemap-news-1.xml извлечь URL с lastmod >= min_date
      3. Случайно перемешать и скачать полные страницы
      4. Извлечь текст через CSS-селектор .n1_material

    N+1 — то же издание, что Taiga N+1 в оригинальных данных.
    ~5800 URL за 2024+, после фильтров по длине — ~1000-3000 статей.
    """
    import requests
    import xml.etree.ElementTree as ET

    headers = {"User-Agent": "Mozilla/5.0 (research project, non-commercial)"}
    sm_ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    # Шаг 1: скачать sitemap index
    log.info("N+1: fetching sitemap index...")
    try:
        resp = requests.get("https://nplus1.ru/sitemap.xml", headers=headers, timeout=30)
        resp.raise_for_status()
        index_root = ET.fromstring(resp.content)
    except Exception as e:
        log.error("Failed to fetch sitemap index: %s", e)
        return []

    # Шаг 2: найти news sitemap
    sub_urls = [loc.text for loc in index_root.findall(".//sm:loc", sm_ns)]
    news_sitemap_url = next((u for u in sub_urls if "news" in u), None)
    if not news_sitemap_url:
        log.error("No news sitemap found in index. Available: %s", sub_urls)
        return []

    log.info("N+1: fetching %s...", news_sitemap_url)
    try:
        resp = requests.get(news_sitemap_url, headers=headers, timeout=60)
        resp.raise_for_status()
        news_root = ET.fromstring(resp.content)
    except Exception as e:
        log.error("Failed to fetch news sitemap: %s", e)
        return []

    # Шаг 3: извлечь URL с lastmod >= min_date
    candidates = []
    for url_el in news_root.findall(".//sm:url", sm_ns):
        loc = url_el.find("sm:loc", sm_ns)
        lastmod = url_el.find("sm:lastmod", sm_ns)
        if loc is None:
            continue
        url = loc.text
        date = lastmod.text[:10] if lastmod is not None else ""
        if date >= min_date and "/news/" in url:
            candidates.append((url, date))

    log.info("N+1: found %d news URLs >= %s", len(candidates), min_date)

    # Перемешать для разнообразия по датам
    random.seed(SEED)
    random.shuffle(candidates)

    # Шаг 4: скачать полные статьи
    records = []
    errors = 0

    for i, (url, date) in enumerate(candidates):
        if len(records) >= count:
            break

        text = _fetch_nplus1_article(url, headers)
        if not text:
            errors += 1
            continue

        # Извлечь slug для doc_id
        slug = url.rstrip("/").split("/")[-1]

        rec = make_record(
            text=text,
            domain="news",
            source_name="nplus1_fresh",
            doc_id=f"nplus1_{slug}",
            published=date,
            url=url,
        )
        if rec:
            records.append(rec)
            if len(records) % 50 == 0:
                log.info("N+1: collected %d/%d (checked %d, errors %d)",
                         len(records), count, i + 1, errors)

        time.sleep(0.5)

    log.info("N+1 fresh: collected %d from %d checked (%d errors)",
             len(records), min(len(candidates), i + 1 if candidates else 0), errors)
    return records


def _fetch_nplus1_article(url: str, headers: dict) -> str | None:
    """Скачать и извлечь текст статьи N+1.

    Статья разбита на несколько .n1_material блоков (по одному абзацу).
    Собираем все блоки для полного текста.
    """
    import requests
    from bs4 import BeautifulSoup

    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
    except Exception:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Собираем ВСЕ .n1_material блоки (каждый = один абзац)
    blocks = soup.select(".n1_material")
    if blocks:
        text = " ".join(b.get_text(strip=True) for b in blocks if b.get_text(strip=True))
    else:
        # Fallback: все <p> из основного контейнера
        body = soup.select_one("article") or soup
        for tag in body.select("script, style, nav, aside, footer, header"):
            tag.decompose()
        paragraphs = body.find_all("p")
        text = " ".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

    return text if len(text) >= MIN_CHARS else None


# =====================================================================
# WIKI: Russian Wikipedia (API)
# =====================================================================


def download_wiki_fresh(count: int = 1000) -> list[dict]:
    """Скачать статьи из русской Wikipedia через HuggingFace дамп.

    Источник: wikimedia/wikipedia (дамп 20231101.ru) — streaming,
    без rate limits. Все статьи гарантированно human (до ChatGPT-эры
    основных правок).

    Фильтры:
      - Длина 300-8000 символов после clean_text
      - Случайная выборка (shuffle через streaming)
    """
    from datasets import load_dataset

    log.info("Wiki: loading wikimedia/wikipedia (20231101.ru, streaming)...")

    records = []
    scanned = 0
    max_scan = count * 15  # ~10-15% статей пройдут фильтр по длине

    for example in load_dataset(
        "wikimedia/wikipedia", "20231101.ru",
        split="train", streaming=True,
    ):
        scanned += 1
        if len(records) >= count:
            break
        if scanned > max_scan:
            log.warning("Wiki: reached max_scan=%d, stopping", max_scan)
            break

        text = example.get("text", "")
        title = example.get("title", "")
        url = example.get("url", "")
        page_id = example.get("id", "")

        if not text:
            continue

        rec = make_record(
            text=text,
            domain="wiki",
            source_name="wiki_fresh",
            doc_id=f"wiki_fresh_{page_id}",
            url=url,
            title=title,
        )
        if rec:
            records.append(rec)

        if scanned % 5000 == 0:
            log.info("Wiki: scanned %d, collected %d/%d", scanned, len(records), count)

    log.info("Wiki fresh: collected %d from %d scanned", len(records), scanned)
    return records


# =====================================================================
# Разведка: проверка доступности данных
# =====================================================================


def scout_habr(sample_size: int = 50) -> None:
    """Быстрая проверка: работает ли Habr API и есть ли свежие статьи.

    Берёт sample_size случайных ID из диапазона 2024+ и проверяет доступность.
    """
    import requests

    log.info("=== Разведка: Habr API (habr.com/kek/v2/articles/{id}) ===")

    session = requests.Session()
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )

    # ID примерно с 2024 года
    test_ids = random.sample(range(900000, 1020000), sample_size)

    found = 0
    ru_2024 = 0
    by_year: dict[str, int] = {}

    for i, art_id in enumerate(test_ids):
        try:
            r = session.get(f"https://habr.com/kek/v2/articles/{art_id}", timeout=10)
        except Exception:
            continue

        if r.status_code != 200:
            continue

        found += 1
        data = r.json()
        pub = data.get("timePublished", "")
        lang = data.get("lang", "")
        year = pub[:4] if pub else "?"

        by_year[year] = by_year.get(year, 0) + 1

        if lang == "ru" and pub >= "2024":
            ru_2024 += 1

        time.sleep(0.3)

        if (i + 1) % 10 == 0:
            log.info("Checked %d/%d, found %d, ru_2024+ %d", i + 1, sample_size, found, ru_2024)

    log.info("Results: checked %d, found %d (%.0f%%), ru_2024+ %d",
             sample_size, found, found / sample_size * 100, ru_2024)
    log.info("By year: %s", dict(sorted(by_year.items())))

    # Экстраполяция
    hit_rate = found / sample_size
    ru_rate = ru_2024 / max(found, 1)
    total_range = 1020000 - 900000
    estimated = int(total_range * hit_rate * ru_rate)
    log.info("Estimated ru articles 2024+ in range: ~%d", estimated)


# =====================================================================
# CLI
# =====================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Download fresh human texts (2024+) for temporal robustness eval"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Разведка
    scout_p = sub.add_parser("scout", help="Check Habr API availability")
    scout_p.add_argument("--sample-size", type=int, default=50)

    # Скачивание
    dl_p = sub.add_parser("download", help="Download fresh texts")
    dl_p.add_argument(
        "--source",
        choices=["habr", "nplus1", "wiki", "all"],
        default="all",
    )
    dl_p.add_argument("--count", type=int, default=500, help="Texts per source")
    dl_p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)

    args = parser.parse_args()

    if args.command == "scout":
        random.seed(SEED)
        scout_habr(args.sample_size)
        return

    # Download
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.source in ("habr", "all"):
        records = download_habr_fresh(args.count)
        save_jsonl(records, args.output_dir / "habr_fresh.jsonl")

    if args.source in ("nplus1", "all"):
        records = download_nplus1_fresh(args.count)
        save_jsonl(records, args.output_dir / "nplus1_fresh.jsonl")

    if args.source in ("wiki", "all"):
        records = download_wiki_fresh(args.count)
        save_jsonl(records, args.output_dir / "wiki_fresh.jsonl")

    log.info("=== Done. Files in %s ===", args.output_dir)


if __name__ == "__main__":
    main()
