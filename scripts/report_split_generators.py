"""Таблица генераторов (source_model) по сплитам. Пишет artifacts/split_generators.json."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "splits"
OUT_JSON = ROOT / "artifacts" / "split_generators.json"

SPLITS_MAIN = ["train", "val", "test", "holdout_unseen"]
SPLITS_TM = ["train_tm", "val_tm", "test_tm", "holdout_tm_unseen"]


def _counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    df = pd.read_json(path, lines=True)
    return df["source_model"].value_counts().to_dict()


def main() -> None:
    out: dict = {"main": {}, "topic_matched": {}, "notes": []}
    for name in SPLITS_MAIN:
        p = SPLITS / f"{name}.jsonl"
        out["main"][name] = _counts(p)
    for name in SPLITS_TM:
        p = SPLITS / f"{name}.jsonl"
        out["topic_matched"][name] = _counts(p)
    out["notes"] = [
        "holdout_unseen: ожидается только yandexgpt среди AI (см. tests/test_splits.py).",
        "holdout_tm_unseen: ожидается только qwen/qwen3-32b среди AI.",
    ]
    text = json.dumps(out, ensure_ascii=False, indent=2)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(text, encoding="utf-8")
    print(text)
    print(f"(also written {OUT_JSON})")


if __name__ == "__main__":
    main()
