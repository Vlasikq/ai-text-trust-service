"""
Подбор порогов cascade_lo / cascade_hi по val_tm с фиксированными TF-IDF и Transformer.

Метрика: бинарный F1 на val_tm (каскад: в «серой зоне» берётся трансформер).
Сравнение с TF-IDF-only на holdout_tm_unseen.

Требуются артефакты в artifacts/service и artifacts/transformer/model.
Результат: artifacts/cascade_threshold_sweep.json
"""

from __future__ import annotations

import itertools
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
OUT_PATH = PROJECT_ROOT / "artifacts" / "cascade_threshold_sweep.json"
# Полный holdout + трансформер в серой зоне = часы; для отчёта — подвыборка
HOLDOUT_MAX_SAMPLES = 800


def _labels_from_prob(probs: list[float], thresh: float = 0.5) -> list[int]:
    return [1 if p >= thresh else 0 for p in probs]


def cascade_probs(
    texts: list[str],
    fast,
    slow,
    lo: float,
    hi: float,
) -> tuple[list[float], dict]:
    """Для каждого текста: fast вне (lo,hi) иначе slow."""
    probs = []
    paths = {"fast": 0, "slow": 0}
    t0 = time.perf_counter()
    for t in texts:
        fr = fast.predict(t)
        p = fr.prob_ai
        if p <= lo or p >= hi:
            probs.append(p)
            paths["fast"] += 1
        else:
            sr = slow.predict(t)
            probs.append(sr.prob_ai)
            paths["slow"] += 1
    elapsed = time.perf_counter() - t0
    return probs, {**paths, "total_s": round(elapsed, 2)}


def main() -> None:
    from app.config import Settings
    from app.detectors.tfidf import TFIDFDetector
    from app.detectors.transformer import TransformerDetector

    settings = Settings()
    val_path = SPLITS_DIR / "val_tm.jsonl"
    hold_path = SPLITS_DIR / "holdout_tm_unseen.jsonl"
    if not val_path.exists():
        log.error("Нет %s", val_path)
        return

    val_df = pd.read_json(val_path, lines=True)
    texts_val = val_df["text"].tolist()
    y_val = val_df["label"].astype(int).tolist()

    hold_df = None
    if hold_path.exists():
        hold_df = pd.read_json(hold_path, lines=True)

    tfidf = TFIDFDetector(settings.artifacts_dir)
    tfidf.load()
    if not settings.transformer_dir.exists():
        log.error("Нет директории трансформера: %s", settings.transformer_dir)
        return
    transformer = TransformerDetector(settings.transformer_dir)
    transformer.load()

    # TF-IDF-only baseline
    p_tf_val = [tfidf.predict(t).prob_ai for t in texts_val]
    f1_tf_val = f1_score(y_val, _labels_from_prob(p_tf_val), average="binary")

    lo_grid = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    hi_grid = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    results = []
    best = None

    for lo, hi in itertools.product(lo_grid, hi_grid):
        if lo >= hi - 0.05:
            continue
        probs, paths = cascade_probs(texts_val, tfidf, transformer, lo, hi)
        f1 = f1_score(y_val, _labels_from_prob(probs), average="binary")
        frac_slow = paths["slow"] / len(texts_val) if texts_val else 0
        row = {
            "cascade_lo": lo,
            "cascade_hi": hi,
            "f1_val_tm": round(f1, 4),
            "pct_slow_path": round(100 * frac_slow, 1),
            "eval_wall_time_s": paths["total_s"],
        }
        results.append(row)
        if best is None or f1 > best["f1_val_tm"]:
            best = row

    assert best is not None
    log.info("Best on val_tm: %s", best)

    out: dict = {
        "tfidf_only_f1_val_tm": round(float(f1_tf_val), 4),
        "best_cascade_val_tm": best,
        "grid_results": sorted(results, key=lambda r: -r["f1_val_tm"])[:20],
    }

    if hold_df is not None:
        hdf = hold_df
        if len(hdf) > HOLDOUT_MAX_SAMPLES:
            n_per = max(1, HOLDOUT_MAX_SAMPLES // 2)
            chunks = []
            for lbl in sorted(hdf["label"].unique()):
                g = hdf[hdf["label"] == lbl]
                chunks.append(g.sample(n=min(n_per, len(g)), random_state=42))
            hdf = pd.concat(chunks, ignore_index=True)
            log.info(
                "Holdout subsample for cascade eval: %d rows (cap=%d)",
                len(hdf),
                HOLDOUT_MAX_SAMPLES,
            )
        texts_h = hdf["text"].tolist()
        y_h = hdf["label"].astype(int).tolist()
        p_tf_h = [tfidf.predict(t).prob_ai for t in texts_h]
        f1_tf_h = f1_score(y_h, _labels_from_prob(p_tf_h), average="binary")
        lo_b, hi_b = best["cascade_lo"], best["cascade_hi"]
        p_c_h, _ = cascade_probs(texts_h, tfidf, transformer, lo_b, hi_b)
        f1_c_h = f1_score(y_h, _labels_from_prob(p_c_h), average="binary")
        out["holdout_tm_unseen"] = {
            "eval_rows": len(hdf),
            "holdout_subsampled": len(hold_df) > HOLDOUT_MAX_SAMPLES,
            "tfidf_f1": round(float(f1_tf_h), 4),
            "cascade_f1_best_val_thresholds": round(float(f1_c_h), 4),
            "cascade_lo": lo_b,
            "cascade_hi": hi_b,
        }

    out["default_settings_cascade"] = {
        "cascade_lo": settings.cascade_lo,
        "cascade_hi": settings.cascade_hi,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Written %s", OUT_PATH)


if __name__ == "__main__":
    main()
