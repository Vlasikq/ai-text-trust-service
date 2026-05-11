"""
Prod-VM latency benchmark.

Сокращённая версия benchmark_latency.py для запуска внутри aitrust-api контейнера на prod-VM:
  - читает test_tm.jsonl из /tmp/test_tm.jsonl (а не data/splits)
  - пишет результат в /tmp/prod_latency.json
  - меряет TF-IDF, трансформер, каскад на тех же 100 текстах P10-P90
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

INPUT_PATH = Path("/tmp/test_tm.jsonl")
OUTPUT_PATH = Path("/tmp/prod_latency.json")


def sample_texts(path: Path, n: int = 100, seed: int = 42) -> list[str]:
    df = pd.read_json(path, lines=True)
    lens = df["text"].str.len()
    p10, p90 = lens.quantile(0.1), lens.quantile(0.9)
    subset = df[(lens >= p10) & (lens <= p90)]
    sample = subset.sample(n=min(n, len(subset)), random_state=seed)
    log.info("Sampled %d texts (P10=%d, P90=%d chars)", len(sample), int(p10), int(p90))
    return sample["text"].tolist()


def measure(predict_fn, texts: list[str], warmup: int = 3) -> dict:
    for t in texts[:warmup]:
        predict_fn(t)
    latencies = []
    for t in texts:
        t0 = time.perf_counter()
        predict_fn(t)
        latencies.append((time.perf_counter() - t0) * 1000)
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
    import platform
    import psutil

    texts = sample_texts(INPUT_PATH, n=100)
    avg_chars = int(np.mean([len(t) for t in texts]))

    env = {
        "platform": platform.platform(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_mb": int(psutil.virtual_memory().total / 1024 / 1024),
        "python_version": platform.python_version(),
        "n_texts": len(texts),
        "avg_chars": avg_chars,
        "warmup": 3,
    }
    log.info("Env: %s", env)

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

    log.info("Benchmarking TF-IDF...")
    results["tfidf"] = measure(lambda t: tfidf.predict(t), texts)
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

        log.info("Benchmarking Transformer (this may take several minutes on CPU)...")
        results["transformer"] = measure(lambda t: transformer.predict(t), texts)
        results["transformer"]["cold_start_first_predict_ms"] = round(cold_tr_ms, 2)
        log.info("Transformer: %s", results["transformer"])

        # ── Cascade ───────────────────────────────────────────
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
            cascade_latencies.append((time.perf_counter() - t0) * 1000)
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

    output = {
        "benchmark_date": pd.Timestamp.now().isoformat(),
        "env": env,
        "text_length_range": (
            f"P10-P90 ({min(len(t) for t in texts)}-{max(len(t) for t in texts)} chars)"
        ),
        "results": results,
    }

    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Saved -> %s", OUTPUT_PATH)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
