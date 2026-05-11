"""
Latency benchmark: measure inference time across 3 detector modes.

Samples 100 texts from test_tm (P10-P90 by length) and runs each
through tfidf, transformer, and cascade detectors.

Output: artifacts/latency_benchmark.json — median, p95, p99 per mode.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "splits"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OUTPUT_PATH = ARTIFACTS_DIR / "latency_benchmark.json"


def _sample_texts(path: Path, n: int = 100, seed: int = 42) -> list[str]:
    """Sample n texts spanning P10-P90 by length."""
    df = pd.read_json(path, lines=True)
    lens = df["text"].str.len()
    p10, p90 = lens.quantile(0.1), lens.quantile(0.9)
    subset = df[(lens >= p10) & (lens <= p90)]
    sample = subset.sample(n=min(n, len(subset)), random_state=seed)
    log.info(
        "Sampled %d texts (P10=%d, P90=%d chars)",
        len(sample),
        int(p10),
        int(p90),
    )
    return sample["text"].tolist()


def _measure(predict_fn, texts: list[str], warmup: int = 3) -> dict:
    """Run predict_fn on texts, return latency stats in ms."""
    # Warmup
    for t in texts[:warmup]:
        predict_fn(t)

    latencies = []
    for t in texts:
        t0 = time.perf_counter()
        predict_fn(t)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

    arr = np.array(latencies)
    return {
        "n": len(arr),
        "median_ms": round(float(np.median(arr)), 2),
        "mean_ms": round(float(np.mean(arr)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2),
    }


def main():

    texts = _sample_texts(DATA_DIR / "test_tm.jsonl", n=100)
    results = {}

    # ── TF-IDF ────────────────────────────────────────────────
    log.info("Loading TF-IDF detector...")
    from app.config import Settings
    from app.detectors.tfidf import TFIDFDetector

    settings = Settings()
    tfidf = TFIDFDetector(settings.artifacts_dir)
    tfidf.load()

    t0 = time.perf_counter()
    tfidf.predict(texts[0])
    cold_tfidf_ms = (time.perf_counter() - t0) * 1000

    log.info("Benchmarking TF-IDF (%d texts)...", len(texts))
    results["tfidf"] = _measure(lambda t: tfidf.predict(t), texts)
    results["tfidf"]["cold_start_first_predict_ms"] = round(cold_tfidf_ms, 2)
    log.info("TF-IDF: %s", results["tfidf"])

    # ── Transformer ───────────────────────────────────────────
    transformer_dir = settings.transformer_dir
    if transformer_dir.exists() and (transformer_dir / "model.safetensors").exists():
        log.info("Loading Transformer detector...")
        from app.detectors.transformer import TransformerDetector

        transformer = TransformerDetector(transformer_dir)
        transformer.load()

        t0 = time.perf_counter()
        transformer.predict(texts[0])
        cold_tr_ms = (time.perf_counter() - t0) * 1000

        log.info("Benchmarking Transformer (%d texts)...", len(texts))
        results["transformer"] = _measure(lambda t: transformer.predict(t), texts)
        results["transformer"]["cold_start_first_predict_ms"] = round(cold_tr_ms, 2)
        log.info("Transformer: %s", results["transformer"])

        # ── Cascade ───────────────────────────────────────────
        log.info("Benchmarking Cascade (lo=0.30, hi=0.70)...")
        from app.detectors.cascade import CascadeDetector

        cascade = CascadeDetector(
            fast=tfidf,
            slow=transformer,
            cascade_lo=settings.cascade_lo,
            cascade_hi=settings.cascade_hi,
        )

        t0 = time.perf_counter()
        cascade.predict(texts[0])
        cold_cascade_ms = (time.perf_counter() - t0) * 1000

        cascade_latencies = []
        cascade_paths = {"fast": 0, "slow": 0}
        for t in texts:
            t0 = time.perf_counter()
            r = cascade.predict(t)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            cascade_latencies.append(elapsed_ms)
            path = r.metadata.get("cascade_path", "fast")
            cascade_paths[path] = cascade_paths.get(path, 0) + 1

        arr = np.array(cascade_latencies)
        results["cascade"] = {
            "n": len(arr),
            "median_ms": round(float(np.median(arr)), 2),
            "mean_ms": round(float(np.mean(arr)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
            "min_ms": round(float(np.min(arr)), 2),
            "max_ms": round(float(np.max(arr)), 2),
            "paths": cascade_paths,
            "pct_fast": round(cascade_paths["fast"] / len(arr) * 100, 1),
            "cold_start_first_predict_ms": round(cold_cascade_ms, 2),
        }
        log.info("Cascade: %s", results["cascade"])
    else:
        log.warning("Transformer model not found — skipping transformer/cascade benchmarks")

    # ── Save ──────────────────────────────────────────────────
    output = {
        "benchmark_date": pd.Timestamp.now().isoformat(),
        "n_texts": len(texts),
        "text_length_range": (
            f"P10-P90 ({min(len(t) for t in texts)}"
            f"-{max(len(t) for t in texts)} chars)"
        ),
        "results": results,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    log.info("Saved → %s", OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("LATENCY BENCHMARK")
    print("=" * 60)
    print(f"{'Режим':<15} {'Median':>8} {'P95':>8} {'P99':>8}")
    print("-" * 45)
    for mode in ["tfidf", "transformer", "cascade"]:
        if mode in results:
            r = results[mode]
            med, p95, p99 = r["median_ms"], r["p95_ms"], r["p99_ms"]
            print(f"{mode:<15} {med:>7.1f}ms {p95:>7.1f}ms {p99:>7.1f}ms")
    if "cascade" in results:
        print(f"\nCascade fast path: {results['cascade']['pct_fast']}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
