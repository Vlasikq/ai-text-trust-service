"""
Сборка финального датасета из raw-данных.

Два режима:
  --mode original       Оригинальные AI-тексты (data/gen/), topic-aware split
  --mode topic_matched  Topic-matched AI-тексты (data/gen_topic_matched/)
                        split по human_doc_id (без утечки тем)

Pipeline (оба режима):
  1. Загрузка human / AI / adversarial
  2. strip_markdown + нормализация whitespace
  3. Дедупликация (SHA-256)
  4. Фильтрация по длине, truncate длинных по границе предложения
  5. Holdout unseen generator
  6. Per-domain балансировка
  7. Stratified split (topic-aware или по human_doc_id)
  8. Adversarial -> только test
"""

import argparse
import hashlib
import json
import logging
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from app.preprocessing.text_cleaning import clean_text, truncate_by_sentence

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"

SEED = 42
DEFAULT_MIN_CHARS = 300
DEFAULT_MAX_CHARS = 8000

HUMAN_SOURCES = [
    ("taiga_nplus1_with_time.jsonl", "news", "taiga_nplus1"),
    ("wiki_articles.jsonl", "wiki", "wikipedia"),
    ("habr_essays.jsonl", "essay", "habr"),
]


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_human() -> list[dict]:
    rows = []
    for filename, domain, source in HUMAN_SOURCES:
        path = DATA_DIR / "human" / filename
        if not path.exists():
            log.warning(f"Не найден: {path}")
            continue
        count = 0
        for row in iter_jsonl(path):
            rows.append({
                "text": row["text"],
                "label": 0,
                "domain": row.get("domain", domain),
                "source_model": "human",
                "source_name": row.get("source_name", source),
                "doc_id": row.get("doc_id", ""),
                "topic": None,  # human не имеют контролируемых тем
            })
            count += 1
        log.info(f"  human/{filename}: {count}")
    return rows


def load_ai() -> list[dict]:
    rows = []
    gen_dir = DATA_DIR / "gen"
    if not gen_dir.exists():
        return rows
    for fpath in sorted(gen_dir.glob("*.jsonl")):
        count = 0
        for row in iter_jsonl(fpath):
            meta = row.get("meta", {}) or {}
            topic = meta.get("topic") if isinstance(meta, dict) else None
            rows.append({
                "text": row["text"],
                "label": 1,
                "domain": row.get("domain", "unknown"),
                "source_model": row.get("source_name", "unknown"),
                "source_name": row.get("source_name", "unknown"),
                "doc_id": row.get("doc_id", ""),
                "topic": topic,
            })
            count += 1
        if count:
            log.info(f"  gen/{fpath.name}: {count}")
    return rows


def load_ai_topic_matched() -> list[dict]:
    """Загрузка topic-matched AI-текстов из data/gen_topic_matched/."""
    rows = []
    gen_dir = DATA_DIR / "gen_topic_matched"
    if not gen_dir.exists():
        return rows
    for fpath in sorted(gen_dir.glob("*.jsonl")):
        count = 0
        for row in iter_jsonl(fpath):
            meta = row.get("meta", {}) or {}
            rows.append({
                "text": row["text"],
                "label": 1,
                "domain": row.get("domain", "unknown"),
                "source_model": row.get("source_name", "unknown"),
                "source_name": row.get("source_name", "unknown"),
                "doc_id": row.get("doc_id", ""),
                "topic": meta.get("topic"),
                "human_doc_id": meta.get("human_doc_id"),
            })
            count += 1
        if count:
            log.info(f"  gen_topic_matched/{fpath.name}: {count}")
    return rows


def load_adversarial() -> list[dict]:
    rows = []
    adv_dir = DATA_DIR / "adversarial"
    if not adv_dir.exists():
        return rows
    for fpath in sorted(adv_dir.glob("*.jsonl")):
        count = 0
        for row in iter_jsonl(fpath):
            rows.append({
                "text": row["text"],
                "label": 1,  # adversarial = AI
                "domain": row.get("domain", "unknown"),
                "source_model": "adversarial",
                "source_name": row.get("source_name", "unknown"),
                "doc_id": row.get("doc_id", ""),
                "topic": None,
            })
            count += 1
        if count:
            log.info(f"  adv/{fpath.name}: {count}")
    return rows


# ── dedup & filter ───────────────────────────────────────────────


def text_hash(text: str) -> str:
    normalized = " ".join(text.split()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def deduplicate(data: list[dict], name: str) -> list[dict]:
    seen = set()
    unique = []
    for row in data:
        h = text_hash(row["text"])
        if h in seen:
            continue
        seen.add(h)
        unique.append(row)
    removed = len(data) - len(unique)
    if removed:
        log.info(f"  Dedup ({name}): {len(data)} → {len(unique)} (−{removed})")
    return unique


def apply_cleaning(data: list[dict], name: str) -> list[dict]:
    """strip_markdown + normalize_whitespace для всех текстов."""
    for row in data:
        row["text"] = clean_text(row["text"])
    log.info(f"  Очистка ({name}): {len(data)} текстов")
    return data


def filter_by_length(data: list[dict], min_chars: int, max_chars: int, name: str) -> list[dict]:
    result = []
    truncated = 0
    removed = 0
    for row in data:
        text_len = len(row["text"])
        if text_len < min_chars:
            removed += 1
            continue
        if text_len > max_chars:
            row["text"] = truncate_by_sentence(row["text"], max_chars)
            truncated += 1
        result.append(row)
    log.info(f"  Длина ({name}): {len(data)} → {len(result)} "
             f"(удалено {removed}, truncated {truncated})")
    return result


# ── split logic ─────────────────────────────────────────────────


def make_splits(
    human: list[dict],
    ai: list[dict],
    adversarial: list[dict],
    holdout_model: str | None = None,
    test_size: float = 0.15,
    val_size: float = 0.10,
) -> dict[str, list[dict]]:
    rng = random.Random(SEED)
    splits = {}

    main_human = list(human)
    main_ai = list(ai)

    # 1. Holdout unseen generator
    if holdout_model:
        holdout_ai = [r for r in ai if r["source_model"] == holdout_model]
        main_ai = [r for r in ai if r["source_model"] != holdout_model]
        if holdout_ai:
            # пропорциональная выборка human для holdout
            holdout_human_n = min(len(holdout_ai), len(human) // 5)
            holdout_human = rng.sample(human, holdout_human_n)
            holdout_ids = {r["doc_id"] for r in holdout_human}
            main_human = [r for r in human if r["doc_id"] not in holdout_ids]
            splits["holdout_unseen"] = holdout_ai + holdout_human
            rng.shuffle(splits["holdout_unseen"])
            log.info(f"Holdout ({holdout_model}): {len(holdout_ai)} AI + "
                     f"{holdout_human_n} human = {len(splits['holdout_unseen'])}")
        else:
            log.warning(f"Holdout model '{holdout_model}' не найдена")

    # 2. Per-domain балансировка
    human_by_dom = defaultdict(list)
    ai_by_dom = defaultdict(list)
    for r in main_human:
        human_by_dom[r["domain"]].append(r)
    for r in main_ai:
        ai_by_dom[r["domain"]].append(r)

    balanced = []
    for dom in sorted(set(human_by_dom) | set(ai_by_dom)):
        h = human_by_dom.get(dom, [])
        a = ai_by_dom.get(dom, [])
        if not h or not a:
            log.warning(f"  {dom}: human={len(h)}, ai={len(a)} — пропускаем")
            continue
        n = min(len(h), len(a))
        if len(h) > n:
            h = rng.sample(h, n)
        if len(a) > n:
            a = rng.sample(a, n)
        log.info(f"  {dom}: {n} human + {n} AI = {2 * n}")
        balanced.extend(h)
        balanced.extend(a)

    # 3. Topic-aware stratified split: группируем по (domain, label, topic).
    # AI: topic из meta. Human: каждый текст — своя "группа" (random split)
    topic_groups = defaultdict(list)  # (domain, label, topic) → [rows]
    for r in balanced:
        topic = r.get("topic")
        if topic is None:
            # human: уникальная группа для каждого текста (= random split)
            topic = f"_human_{r['doc_id']}"
        topic_groups[(r["domain"], r["label"], topic)].append(r)

    # Собираем уникальные topic-группы для split-а
    # Стратифицируем по (domain, label)
    strata = defaultdict(list)  # (domain, label) → [topic_key, ...]
    for key in topic_groups:
        dom, lbl, topic = key
        strata[(dom, lbl)].append(key)

    train, val, test = [], [], []

    for (dom, lbl), keys in strata.items():
        rng.shuffle(keys)

        # для AI: split по topic-группам
        # для human: каждый текст отдельная группа, эффективно random split
        all_rows = []
        for k in keys:
            all_rows.append(topic_groups[k])

        n = len(all_rows)
        n_test = max(1, int(n * test_size))
        n_val = max(1, int(n * val_size))

        test_groups = all_rows[:n_test]
        val_groups = all_rows[n_test:n_test + n_val]
        train_groups = all_rows[n_test + n_val:]

        for group in test_groups:
            test.extend(group)
        for group in val_groups:
            val.extend(group)
        for group in train_groups:
            train.extend(group)

    # 4. Adversarial → только test
    if adversarial:
        test.extend(adversarial)
        log.info(f"Adversarial → test: {len(adversarial)}")

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    splits["train"] = train
    splits["val"] = val
    splits["test"] = test
    return splits


def make_splits_topic_matched(
    human: list[dict],
    ai_tm: list[dict],
    adversarial: list[dict],
    holdout_model: str = "qwen/qwen3-32b",
    test_size: float = 0.15,
    val_size: float = 0.10,
) -> dict[str, list[dict]]:
    """Сборка splits для topic-matched данных.

    Split по human_doc_id: если human X в train, все AI-пары X тоже в train.
    Holdout = Qwen3 (unseen generator).
    """
    rng = random.Random(SEED)
    splits = {}

    # 1. Holdout: отделяем unseen generator
    holdout_ai = [r for r in ai_tm if r["source_model"] == holdout_model]
    main_ai = [r for r in ai_tm if r["source_model"] != holdout_model]
    log.info(f"Topic-matched AI: main={len(main_ai)}, holdout ({holdout_model})={len(holdout_ai)}")

    # 2. Индексируем human по doc_id
    human_by_id = {}
    for r in human:
        human_by_id[r["doc_id"]] = r

    # 3. Собираем пары: human_doc_id -> {human_row, ai_rows}
    #    "Paired" = human тексты, для которых есть topic-matched AI
    paired_ids = set()
    ai_by_human_id = defaultdict(list)
    for r in main_ai:
        hid = r.get("human_doc_id")
        if hid and hid in human_by_id:
            paired_ids.add(hid)
            ai_by_human_id[hid].append(r)

    unpaired_human = [r for r in human if r["doc_id"] not in paired_ids]
    log.info(f"Paired human: {len(paired_ids)}, unpaired human: {len(unpaired_human)}")

    # 4. Split paired human_doc_ids по доменам
    paired_by_domain = defaultdict(list)
    for hid in paired_ids:
        dom = human_by_id[hid]["domain"]
        paired_by_domain[dom].append(hid)

    train_data, val_data, test_data = [], [], []

    for dom in sorted(paired_by_domain):
        ids = paired_by_domain[dom]
        rng.shuffle(ids)

        n = len(ids)
        n_test = max(1, int(n * test_size))
        n_val = max(1, int(n * val_size))

        test_ids = set(ids[:n_test])
        val_ids = set(ids[n_test:n_test + n_val])
        train_ids = set(ids[n_test + n_val:])

        for hid, split_list in [(test_ids, test_data), (val_ids, val_data),
                                (train_ids, train_data)]:
            for doc_id in hid:
                split_list.append(human_by_id[doc_id])
                split_list.extend(ai_by_human_id[doc_id])

        log.info(f"  {dom}: {len(train_ids)} train + {len(val_ids)} val + "
                 f"{len(test_ids)} test paired groups")

    # 5. Per-domain балансировка внутри каждого split
    for split_name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        human_by_dom = defaultdict(list)
        ai_by_dom = defaultdict(list)
        for r in data:
            if r["label"] == 0:
                human_by_dom[r["domain"]].append(r)
            else:
                ai_by_dom[r["domain"]].append(r)

        balanced = []
        for dom in sorted(set(human_by_dom) | set(ai_by_dom)):
            h = human_by_dom.get(dom, [])
            a = ai_by_dom.get(dom, [])
            if not h or not a:
                continue
            n = min(len(h), len(a))
            if len(h) > n:
                h = rng.sample(h, n)
            if len(a) > n:
                a = rng.sample(a, n)
            balanced.extend(h)
            balanced.extend(a)

        data.clear()
        data.extend(balanced)

    # 6. Распределяем unpaired human пропорционально (дополняем human-класс)
    rng.shuffle(unpaired_human)
    unpaired_by_dom = defaultdict(list)
    for r in unpaired_human:
        unpaired_by_dom[r["domain"]].append(r)

    for split_name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        ai_count_by_dom = defaultdict(int)
        human_count_by_dom = defaultdict(int)
        for r in data:
            if r["label"] == 1:
                ai_count_by_dom[r["domain"]] += 1
            else:
                human_count_by_dom[r["domain"]] += 1

        for dom in ai_count_by_dom:
            deficit = ai_count_by_dom[dom] - human_count_by_dom[dom]
            if deficit > 0 and unpaired_by_dom[dom]:
                extra = unpaired_by_dom[dom][:deficit]
                unpaired_by_dom[dom] = unpaired_by_dom[dom][deficit:]
                data.extend(extra)

    # 7. Adversarial -> только test
    if adversarial:
        test_data.extend(adversarial)
        log.info(f"Adversarial -> test: {len(adversarial)}")

    rng.shuffle(train_data)
    rng.shuffle(val_data)
    rng.shuffle(test_data)

    splits["train_tm"] = train_data
    splits["val_tm"] = val_data
    splits["test_tm"] = test_data

    # 8. Holdout unseen (Qwen3)
    if holdout_ai:
        # human для holdout: пропорциональная выборка из оставшихся unpaired
        remaining_unpaired = []
        for dom_list in unpaired_by_dom.values():
            remaining_unpaired.extend(dom_list)
        holdout_human_n = min(len(holdout_ai), len(remaining_unpaired))
        holdout_human = (
            rng.sample(remaining_unpaired, holdout_human_n)
            if remaining_unpaired else []
        )
        holdout_data = holdout_ai + holdout_human
        rng.shuffle(holdout_data)
        splits["holdout_tm_unseen"] = holdout_data
        log.info(f"Holdout TM ({holdout_model}): {len(holdout_ai)} AI + "
                 f"{holdout_human_n} human = {len(holdout_data)}")

    return splits


# ── save ─────────────────────────────────────────────────────────


def save_splits(splits: dict[str, list[dict]]):
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # удаляем служебное поле topic перед записью
    for name, data in splits.items():
        path = SPLITS_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in data:
                out = {k: v for k, v in row.items()
                       if k not in ("topic", "human_doc_id")}
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        log.info(f"Saved {name}: {len(data)} rows → {path}")

    # summary
    summary = {}
    for name, data in splits.items():
        lens = [len(r["text"]) for r in data] if data else [0]
        summary[name] = {
            "total": len(data),
            "labels": dict(Counter(r["label"] for r in data)),
            "domains": dict(Counter(r["domain"] for r in data)),
            "source_models": dict(Counter(r["source_model"] for r in data)),
            "text_length": {
                "min": min(lens),
                "max": max(lens),
                "median": int(statistics.median(lens)),
                "mean": int(statistics.mean(lens)),
            },
        }

    summary_path = SPLITS_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info(f"Summary → {summary_path}")

    # итоговая сводка в консоль
    print("\n" + "=" * 60)
    print("СВОДКА SPLITS")
    print("=" * 60)
    for name, data in splits.items():
        labels = Counter(r["label"] for r in data)
        domains = Counter(r["domain"] for r in data)
        models = Counter(r["source_model"] for r in data)
        print(f"\n{name}: {len(data)} текстов")
        print(f"  labels: {dict(labels)}")
        print(f"  domains: {dict(domains)}")
        print(f"  models: {dict(models)}")
        lens = [len(r["text"]) for r in data]
        print(f"  длина: min={min(lens)}, median={int(statistics.median(lens))}, "
              f"max={max(lens)}")


# ── CLI ──────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Сборка датасета из raw-данных")
    p.add_argument("--mode", choices=["original", "topic_matched"],
                   default="original",
                   help="original = data/gen/, topic_matched = data/gen_topic_matched/")
    p.add_argument("--holdout-model", default=None,
                   help="Модель для holdout (unseen generator test)")
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--val-size", type=float, default=0.10)
    p.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    p.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    p.add_argument("--no-clean", action="store_true",
                   help="Пропустить strip_markdown + normalize (для отладки)")
    p.add_argument("--stats", action="store_true",
                   help="Только статистика, без сохранения")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    log.info(f"Режим: {args.mode}")
    log.info("Загрузка данных...")
    human = load_human()
    adversarial = load_adversarial()

    if args.mode == "topic_matched":
        ai = load_ai_topic_matched()
        log.info(f"Raw: human={len(human)}, AI (topic-matched)={len(ai)}, "
                 f"adversarial={len(adversarial)}")
    else:
        ai = load_ai()
        log.info(f"Raw: human={len(human)}, AI={len(ai)}, "
                 f"adversarial={len(adversarial)}")

    # очистка
    if not args.no_clean:
        log.info("Очистка текстов (strip_markdown + normalize)...")
        human = apply_cleaning(human, "human")
        ai = apply_cleaning(ai, "AI")
        adversarial = apply_cleaning(adversarial, "adversarial")

    # дедупликация
    log.info("Дедупликация...")
    human = deduplicate(human, "human")
    ai = deduplicate(ai, "AI")
    adversarial = deduplicate(adversarial, "adversarial")

    # фильтрация по длине
    log.info("Фильтрация по длине...")
    human = filter_by_length(human, args.min_chars, args.max_chars, "human")
    ai = filter_by_length(ai, args.min_chars, args.max_chars, "AI")
    adversarial = filter_by_length(adversarial, args.min_chars, args.max_chars, "adversarial")

    log.info(f"После очистки: human={len(human)}, AI={len(ai)}, "
             f"adversarial={len(adversarial)}")

    if args.stats:
        return 0

    if not ai:
        log.error("Нет AI-данных. Сначала запустите скрипты генерации.")
        return 1

    # сборка splits
    log.info("Сборка splits...")
    if args.mode == "topic_matched":
        holdout = args.holdout_model or "qwen/qwen3-32b"
        splits = make_splits_topic_matched(
            human, ai, adversarial,
            holdout_model=holdout,
            test_size=args.test_size,
            val_size=args.val_size,
        )
    else:
        splits = make_splits(
            human, ai, adversarial,
            holdout_model=args.holdout_model,
            test_size=args.test_size,
            val_size=args.val_size,
        )

    save_splits(splits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
