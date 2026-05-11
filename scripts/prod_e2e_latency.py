"""
End-to-end latency через публичный API сервиса.

Меряет user-perceived latency через HTTPS из локальной сети,
для двух режимов: tfidf и cascade. Включает SSL handshake,
network roundtrip, парсинг JSON, FastAPI middleware, валидация,
preprocessing, inference, response serialization.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data" / "splits" / "test_tm.jsonl"
OUT = PROJECT_ROOT / "artifacts" / "prod_e2e_latency.json"

URL = "https://89.169.141.35.sslip.io/api/v1/analyze"
N_TEXTS = 30
WARMUP = 3
SEED = 42


def sample_texts(n: int = N_TEXTS) -> list[str]:
    df = pd.read_json(DATA, lines=True)
    lens = df["text"].str.len()
    p10, p90 = lens.quantile(0.1), lens.quantile(0.9)
    subset = df[(lens >= p10) & (lens <= p90)]
    sample = subset.sample(n=min(n, len(subset)), random_state=SEED)
    return sample["text"].tolist()


def measure_endpoint(texts: list[str], detector: str, session: requests.Session) -> dict:
    # Warmup
    for t in texts[:WARMUP]:
        session.post(URL, json={"text": t, "detector_type": detector}, timeout=120)
    latencies = []
    failures = 0
    for t in texts:
        t0 = time.perf_counter()
        try:
            r = session.post(URL, json={"text": t, "detector_type": detector}, timeout=120)
            r.raise_for_status()
        except Exception as e:
            failures += 1
            print(f"  fail: {e}")
            continue
        latencies.append((time.perf_counter() - t0) * 1000)
    arr = np.array(latencies)
    return {
        "n_success": int(len(arr)),
        "n_fail": failures,
        "median_ms": round(float(np.median(arr)), 1),
        "mean_ms": round(float(np.mean(arr)), 1),
        "p95_ms": round(float(np.percentile(arr, 95)), 1),
        "p99_ms": round(float(np.percentile(arr, 99)), 1),
        "min_ms": round(float(np.min(arr)), 1),
        "max_ms": round(float(np.max(arr)), 1),
    }


def measure_health(session: requests.Session, n: int = 10) -> dict:
    """Сетевая задержка (нижняя оценка)."""
    url = "https://89.169.141.35.sslip.io/health"
    for _ in range(2):
        session.get(url, timeout=10)
    lat = []
    for _ in range(n):
        t0 = time.perf_counter()
        session.get(url, timeout=10)
        lat.append((time.perf_counter() - t0) * 1000)
    arr = np.array(lat)
    return {
        "median_ms": round(float(np.median(arr)), 1),
        "p95_ms": round(float(np.percentile(arr, 95)), 1),
    }


def main() -> None:
    texts = sample_texts()
    avg_chars = int(np.mean([len(t) for t in texts]))
    print(f"Sampled {len(texts)} texts (avg {avg_chars} chars)")
    print(f"URL: {URL}")
    print(f"Warmup: {WARMUP}")

    session = requests.Session()

    print("\n=== Network baseline (/health) ===")
    health = measure_health(session)
    print(f"  /health: p50={health['median_ms']} ms, p95={health['p95_ms']} ms")

    results = {"network_baseline_health": health}

    print("\n=== TF-IDF (detector_type=tfidf) ===")
    results["tfidf"] = measure_endpoint(texts, "tfidf", session)
    print(f"  {results['tfidf']}")

    print("\n=== Cascade (detector_type=cascade) ===")
    results["cascade"] = measure_endpoint(texts, "cascade", session)
    print(f"  {results['cascade']}")

    output = {
        "benchmark_date": pd.Timestamp.now().isoformat(),
        "url": URL,
        "n_texts": len(texts),
        "avg_chars": avg_chars,
        "warmup": WARMUP,
        "note": (
            "End-to-end latency через публичный HTTPS из локальной сети. "
            "Включает: SSL handshake (на первом запросе), network roundtrip, FastAPI middleware, "
            "preprocessing, inference, response serialization. "
            "Для оценки чистой prod-inference latency вычесть network_baseline_health.median_ms."
        ),
        "results": results,
    }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
