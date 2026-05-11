"""
Скачивание человеческих эссе/статей из открытых источников для третьего домена.

Источник: IlyaGusev/habr на HuggingFace — >600k статей с Хабра.
Фильтруем: русскоязычные, длинные (1500-6000 символов), не кодовые.

Результат: JSONL с полями text, label, domain, source_name, doc_id, created_at
"""

import argparse
import io
import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

MIN_CHARS = 1500
MAX_CHARS = 6000
HABR_JSONL_ZST_URL = (
    "https://huggingface.co/datasets/IlyaGusev/habr/resolve/main/habr.jsonl.zst?download=true"
)

# Диапазон дат: 2018-2022
MIN_YEAR = 2018
MAX_TIMESTAMP = 1667260800  # 2022-11-01 00:00:00 UTC
MIN_TIMESTAMP = 1514764800  # 2018-01-01 00:00:00 UTC
# Хабы которые больше похожи на эссе/аналитику, а не чисто код
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

# Паттерны, указывающие на код-тяжёлый текст
CODE_PATTERNS = re.compile(
    r"```|import\s+\w+|def\s+\w+\(|class\s+\w+[:\(]|"
    r"console\.log|SELECT\s+\w+|<div|<span|function\s*\(",
    re.IGNORECASE,
)


def clean_markdown(text: str) -> str:
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


def is_good_essay(text: str, hubs: list[str]) -> bool:
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


def iter_habr_examples():
    try:
        from datasets import load_dataset
        yield from load_dataset("IlyaGusev/habr", split="train", streaming=True)
        return
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc):
            raise
        log.warning("datasets fallback: %s; switching to JSONL.ZST stream", exc)
    except ImportError:
        raise ImportError("pip install datasets zstandard jsonlines pysimdjson")

    import requests
    import zstandard as zstd

    with requests.get(HABR_JSONL_ZST_URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        response.raw.decode_content = False
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(response.raw) as reader:
            text_reader = io.TextIOWrapper(reader, encoding="utf-8")
            try:
                for line in text_reader:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
            finally:
                text_reader.close()


def download_habr(count: int, output_path: Path) -> int:
    log.info(f"Загрузка IlyaGusev/habr (streaming)... Цель: {count} статей")

    collected = 0
    scanned = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for example in iter_habr_examples():
            scanned += 1
            if scanned % 10000 == 0:
                log.info(f"Scanned: {scanned}, Collected: {collected}/{count}")
            if collected >= count:
                break
            if example.get("language") != "ru":
                continue

            ts = example.get("time_published")
            if not ts or ts < MIN_TIMESTAMP or ts > MAX_TIMESTAMP:
                continue

            raw_text = example.get("text_markdown", "") or ""
            hubs = example.get("hubs", []) or []
            text = clean_markdown(raw_text)

            if not is_good_essay(text, hubs):
                continue

            try:
                created_at = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            except Exception:
                created_at = ""

            row = {
                "text": text,
                "label": "human",
                "domain": "essay",
                "source_name": "habr",
                "doc_id": f"habr_{example.get('id', uuid.uuid4())}",
                "created_at": created_at,
                "meta": {
                    "title": example.get("title", ""),
                    "hubs": hubs[:5],
                    "score": example.get("statistics", {}).get("score", 0),
                    "original_chars": len(raw_text),
                    "clean_chars": len(text),
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            collected += 1

    log.info(f"Done: {collected} collected, {scanned} scanned -> {output_path}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download human essays from Habr (HuggingFace)")
    p.add_argument("--count", type=int, default=4000, help="Number of essays to collect")
    p.add_argument(
        "--output",
        type=str,
        default=str(DATA_DIR / "habr_essays.jsonl"),
        help="Output JSONL path",
    )
    p.add_argument("--min-year", type=int, default=2018,
        help="Earliest publication year (default: 2018)",
    )
    p.add_argument("--max-date", type=str, default="2022-11-01",
        help="Latest publication date YYYY-MM-DD (default: 2022-11-01, before ChatGPT)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    global MIN_TIMESTAMP, MAX_TIMESTAMP
    MIN_TIMESTAMP = int(datetime(args.min_year, 1, 1).timestamp())
    MAX_TIMESTAMP = int(datetime.strptime(args.max_date, "%Y-%m-%d").timestamp())

    log.info(f"Date filter: {args.min_year}-01-01 to {args.max_date}")
    return download_habr(args.count, Path(args.output))


if __name__ == "__main__":
    raise SystemExit(main())
