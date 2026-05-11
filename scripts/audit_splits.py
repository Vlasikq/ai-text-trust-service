"""Аудит целостности сплитов: пересечения текстов и doc_id, близость n-грамм."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
OUT_PATH = PROJECT_ROOT / "artifacts" / "split_integrity_report.json"

TM_NAMES = ["train_tm", "val_tm", "test_tm", "holdout_tm_unseen"]
MAIN_NAMES = ["train", "val", "test", "holdout_unseen"]


def _normalize_text(s: str) -> str:
    return " ".join(s.lower().split())


def _load_split(name: str) -> pd.DataFrame | None:
    path = SPLITS_DIR / f"{name}.jsonl"
    if not path.exists():
        return None
    return pd.read_json(path, lines=True)


def _pair_overlaps(
    splits: dict[str, pd.DataFrame], keys: list[tuple[str, str]]
) -> list[dict]:
    rows = []
    for a, b in keys:
        if a not in splits or b not in splits:
            continue
        ta, tb = splits[a], splits[b]
        text_ov = set(ta["text"]) & set(tb["text"])
        doc_ov = set(ta["doc_id"]) & set(tb["doc_id"])
        na = {_normalize_text(t) for t in ta["text"]}
        nb = {_normalize_text(t) for t in tb["text"]}
        norm_ov = na & nb
        rows.append(
            {
                "pair": f"{a}__{b}",
                "exact_text_overlap": len(text_ov),
                "normalized_text_overlap": len(norm_ov),
                "doc_id_overlap": len(doc_ov),
            }
        )
    return rows


def _char_shingles(text: str, k: int = 5) -> set[str]:
    t = _normalize_text(text).replace(" ", "")
    if len(t) < k:
        return {t} if t else set()
    return {t[i : i + k] for i in range(len(t) - k + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _near_duplicate_scan(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    sample_a: int = 80,
    sample_b: int = 400,
    seed: int = 42,
) -> dict:
    """Для подвыборки текстов из A — макс. Jaccard по k-граммам к подвыборке B."""
    rng = __import__("random").Random(seed)
    idx_a = list(range(len(df_a)))
    idx_b = list(range(len(df_b)))
    rng.shuffle(idx_a)
    rng.shuffle(idx_b)
    idx_a = idx_a[: min(sample_a, len(idx_a))]
    idx_b = idx_b[: min(sample_b, len(idx_b))]
    shingles_b = [_char_shingles(str(df_b.iloc[i]["text"])) for i in idx_b]

    max_j = 0.0
    p95_list = []
    for i in idx_a:
        sa = _char_shingles(str(df_a.iloc[i]["text"]))
        best = max((_jaccard(sa, sb) for sb in shingles_b), default=0.0)
        p95_list.append(best)
        max_j = max(max_j, best)
    p95_list.sort()
    p95 = p95_list[int(0.95 * (len(p95_list) - 1))] if p95_list else 0.0
    return {
        "char5gram_max_jaccard_sample": round(max_j, 4),
        "char5gram_p95_best_jaccard_sample": round(p95, 4),
        "sample_a": len(idx_a),
        "sample_b": len(idx_b),
    }


def _domain_label_tables(df: pd.DataFrame) -> dict:
    ct = pd.crosstab(df["domain"], df["label"])
    return {str(k): ct.loc[k].to_dict() for k in ct.index}


def _chi2_homogeneity_crosstab(ct_a: pd.DataFrame, ct_b: pd.DataFrame) -> dict | None:
    """Хи-квадрат однородности по ячейкам domain×label между val и test."""
    try:
        from scipy.stats import chi2_contingency
    except ImportError:
        return None
    rows = sorted(set(ct_a.index) | set(ct_b.index))
    cols = sorted(set(ct_a.columns) | set(ct_b.columns))
    if not rows or not cols:
        return None

    def flat_counts(ct: pd.DataFrame) -> list[float]:
        ct = ct.reindex(index=rows, columns=cols, fill_value=0)
        return [float(ct.loc[r, c]) for r in rows for c in cols]

    v = flat_counts(ct_a)
    t = flat_counts(ct_b)
    chi2, p, dof, _ = chi2_contingency([v, t])
    return {"chi2": float(chi2), "p_value": float(p), "dof": int(dof)}


def _sha256_short(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    tm: dict[str, pd.DataFrame] = {}
    for n in TM_NAMES:
        df = _load_split(n)
        if df is not None:
            tm[n] = df

    main: dict[str, pd.DataFrame] = {}
    for n in MAIN_NAMES:
        df = _load_split(n)
        if df is not None:
            main[n] = df

    tm_pairs = [
        ("train_tm", "val_tm"),
        ("train_tm", "test_tm"),
        ("train_tm", "holdout_tm_unseen"),
        ("val_tm", "test_tm"),
        ("val_tm", "holdout_tm_unseen"),
        ("test_tm", "holdout_tm_unseen"),
    ]
    main_pairs = [
        ("train", "val"),
        ("train", "test"),
        ("train", "holdout_unseen"),
        ("val", "test"),
        ("val", "holdout_unseen"),
        ("test", "holdout_unseen"),
    ]

    report: dict = {
        "tm_pairwise": _pair_overlaps(tm, tm_pairs),
        "main_pairwise": _pair_overlaps(main, main_pairs),
    }

    if "train_tm" in tm and "test_tm" in tm:
        report["tm_near_duplicate_train_vs_test"] = _near_duplicate_scan(
            tm["train_tm"], tm["test_tm"]
        )

    if "val_tm" in tm and "test_tm" in tm:
        va = pd.crosstab(tm["val_tm"]["domain"], tm["val_tm"]["label"])
        te = pd.crosstab(tm["test_tm"]["domain"], tm["test_tm"]["label"])
        report["tm_domain_label_val"] = _domain_label_tables(tm["val_tm"])
        report["tm_domain_label_test"] = _domain_label_tables(tm["test_tm"])
        report["tm_val_vs_test_domain_label_chi2"] = _chi2_homogeneity_crosstab(va, te)

    if "train_tm" in tm:
        dup_hashes = Counter(_sha256_short(t) for t in tm["train_tm"]["text"])
        dups = sum(1 for c in dup_hashes.values() if c > 1)
        report["train_tm_duplicate_hash_prefix_count"] = dups

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Written {OUT_PATH}")


if __name__ == "__main__":
    main()
