"""
Сборка финального датасета

Загружает human и AI тексты, дедуплицирует, фильтрует по длине,
балансирует per-domain, создаёт стратифицированные splits

"""

import argparse
import hashlib
import json
import logging
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HUMAN_DIR = DATA_DIR / "human"
GEN_DIR = DATA_DIR / "gen"
SPLITS_DIR = DATA_DIR / "splits"

SEED = 42
DEFAULT_MIN_CHARS = 500
DEFAULT_MAX_CHARS = 8000

HUMAN_SOURCES = [
    ("taiga_nplus1_with_time.jsonl", "news", "taiga_nplus1"),
    ("wiki_articles.jsonl", "wiki", "wikipedia"),
    ("habr_essays.jsonl", "essay", "habr"),
]


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def text_hash(text: str) -> str:
    normalized = " ".join(text.split()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_human_data() -> list[dict]:
    rows = []
    for filename, default_domain, default_source in HUMAN_SOURCES:
        path = HUMAN_DIR / filename
        if not path.exists():
            log.warning(f"Not found: {path}")
            continue
        count = 0
        for row in iter_jsonl(path):
            rows.append({
                "text": row["text"],
                "label": 0,
                "domain": row.get("domain", default_domain),
                "source_model": "human",
                "source_name": row.get("source_name", default_source),
                "doc_id": row.get("doc_id", ""),
            })
            count += 1
        log.info(f"  {filename}: {count} texts (domain={default_domain})")
    return rows


def load_ai_data() -> list[dict]:
    rows = []
    if not GEN_DIR.exists():
        log.warning(f"No gen directory: {GEN_DIR}")
        return rows
    for fpath in sorted(GEN_DIR.glob("*.jsonl")):
        count = 0
        for row in iter_jsonl(fpath):
            if row.get("source_name") == "local-template":
                continue
            rows.append({
                "text": row["text"],
                "label": 1,
                "domain": row.get("domain", "unknown"),
                "source_model": row.get("source_name", "unknown"),
                "source_name": row.get("source_name", "unknown"),
                "doc_id": row.get("doc_id", ""),
            })
            count += 1
        if count > 0:
            log.info(f"  {fpath.name}: {count} texts")
    return rows


def filter_by_length(data: list[dict], min_chars: int, max_chars: int, label_name: str) -> list[dict]:
    before = len(data)
    filtered = [r for r in data if min_chars <= len(r["text"]) <= max_chars]
    removed = before - len(filtered)
    if removed > 0:
        log.info(f"  Length filter ({label_name}): {before} → {len(filtered)} (removed {removed})")
    return filtered


def deduplicate(data: list[dict], label_name: str) -> list[dict]:
    seen = set()
    unique = []
    dupes = 0
    for row in data:
        h = text_hash(row["text"])
        if h in seen:
            dupes += 1
            continue
        seen.add(h)
        unique.append(row)
    log.info(f"  Dedup ({label_name}): {len(data)} → {len(unique)} (removed {dupes})")
    return unique


def make_splits(
    human: list[dict],
    ai: list[dict],
    holdout_model: Optional[str] = None,
    test_size: float = 0.15,
    val_size: float = 0.10,
) -> dict[str, list[dict]]:
    rng = random.Random(SEED)
    splits = {}

    main_human = human
    main_ai = ai

    # 1. Holdout (unseen generator)
    if holdout_model:
        holdout_ai = [r for r in ai if r["source_model"] == holdout_model]
        main_ai = [r for r in ai if r["source_model"] != holdout_model]
        if holdout_ai:
            log.info(f"Holdout ({holdout_model}): {len(holdout_ai)} AI texts")
            holdout_human_n = min(len(holdout_ai), len(human) // 5)
            holdout_human = rng.sample(human, holdout_human_n)
            holdout_human_ids = {id(r) for r in holdout_human}
            main_human = [r for r in human if id(r) not in holdout_human_ids]
            splits["holdout_unseen"] = holdout_ai + holdout_human
            rng.shuffle(splits["holdout_unseen"])
        else:
            log.warning(f"No texts from holdout model '{holdout_model}'")

    # 2. Per-domain балансировка
    human_by_domain = defaultdict(list)
    ai_by_domain = defaultdict(list)
    for r in main_human:
        human_by_domain[r["domain"]].append(r)
    for r in main_ai:
        ai_by_domain[r["domain"]].append(r)

    balanced_data = []
    for domain in sorted(set(human_by_domain) | set(ai_by_domain)):
        h = human_by_domain.get(domain, [])
        a = ai_by_domain.get(domain, [])
        if not h or not a:
            log.warning(f"Domain '{domain}': human={len(h)}, ai={len(a)} — skipping")
            continue
        n = min(len(h), len(a))
        if len(h) > n:
            h = rng.sample(h, n)
        if len(a) > n:
            a = rng.sample(a, n)
        log.info(f"  {domain}: {n} human + {n} ai = {2 * n}")
        balanced_data.extend(h)
        balanced_data.extend(a)

    rng.shuffle(balanced_data)

    # 3. Стратифицированный split по label, domain
    groups = defaultdict(list)
    for r in balanced_data:
        groups[(r["label"], r["domain"])].append(r)

    train, val, test = [], [], []
    for key, items in groups.items():
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, int(n * test_size))
        n_val = max(1, int(n * val_size))
        test.extend(items[:n_test])
        val.extend(items[n_test:n_test + n_val])
        train.extend(items[n_test + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    splits["train"] = train
    splits["val"] = val
    splits["test"] = test
    return splits


def save_splits(splits: dict[str, list[dict]]):
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    for name, data in splits.items():
        path = SPLITS_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.info(f"Saved {name}: {len(data)} rows")

    # Summary
    summary = {}
    for name, data in splits.items():
        lens = [len(r["text"]) for r in data] if data else [0]
        summary[name] = {
            "total": len(data),
            "labels": dict(Counter(r["label"] for r in data)),
            "domains": dict(Counter(r["domain"] for r in data)),
            "source_models": dict(Counter(r["source_model"] for r in data)),
            "text_length": {
                "min": min(lens), "max": max(lens),
                "median": int(statistics.median(lens)),
                "mean": int(statistics.mean(lens)),
            },
        }

    (SPLITS_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assemble dataset from all sources")
    p.add_argument("--holdout-model", default=None,
                    help="Model to hold out for unseen-generator test")
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--val-size", type=float, default=0.10)
    p.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    p.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    p.add_argument("--stats", action="store_true", help="Only print stats")
    p.add_argument("--no-length-filter", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    human = load_human_data()
    ai = load_ai_data()
    log.info(f"Raw: human={len(human)}, ai={len(ai)}")

    human = deduplicate(human, "human")
    ai = deduplicate(ai, "AI")

    if not args.no_length_filter:
        human = filter_by_length(human, args.min_chars, args.max_chars, "human")
        ai = filter_by_length(ai, args.min_chars, args.max_chars, "AI")

    log.info(f"Clean: human={len(human)}, ai={len(ai)}")

    if args.stats:
        return 0

    if not ai:
        log.error("No AI data. Run generation scripts first.")
        return 1

    splits = make_splits(human, ai,
                         holdout_model=args.holdout_model,
                         test_size=args.test_size,
                         val_size=args.val_size)
    save_splits(splits)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
