"""
Полная оценка калибровки трансформера и TF-IDF на трёх срезах.

Pipeline:
  1. Прогнать трансформер на val_tm (если нет в кэше) -> получить raw probs
  2. Загрузить cached_predictions для test_tm и holdout_tm
  3. Загрузить TF-IDF probs (через pipeline сервиса) на val_tm/test_tm/holdout_tm
  4. Обучить Platt-калибраторы на val_tm для каждой модели
  5. Применить калибраторы -> получить platt-probs
  6. Посчитать ECE/Brier на test_tm и holdout_tm (НЕ на val_tm — там Platt fit)
  7. Сохранить calibration_summary.csv + calibration_eval.json
  8. Построить reliability diagram (test_tm vs holdout_tm для трансформера)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
ARTIFACTS = PROJECT_ROOT / "artifacts"
TRANSFORMER_DIR = ARTIFACTS / "transformer" / "model"
CACHED_PREDS_PATH = ARTIFACTS / "transformer" / "cached_predictions.json"
SERVICE_DIR = ARTIFACTS / "service"

OUT_CSV = ARTIFACTS / "calibration_summary.csv"
OUT_JSON = ARTIFACTS / "calibration_eval.json"
OUT_RELIABILITY = ARTIFACTS / "reliability_diagram_transformer.png"


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error with proper right-edge handling for last bin."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    val = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_prob >= edges[i]) & (y_prob <= edges[i + 1])
        else:
            mask = (y_prob >= edges[i]) & (y_prob < edges[i + 1])
        if not mask.any():
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        val += (mask.sum() / len(y_prob)) * abs(bin_acc - bin_conf)
    return float(val)


def fit_platt(probs_val: np.ndarray, labels_val: np.ndarray) -> LogisticRegression:
    """Logistic regression on log-odds (Platt scaling)."""
    eps = 1e-10
    logits = np.log(np.clip(probs_val, eps, 1 - eps) / np.clip(1 - probs_val, eps, 1 - eps))
    lr = LogisticRegression(max_iter=1000)
    lr.fit(logits.reshape(-1, 1), labels_val)
    return lr


def apply_platt(lr: LogisticRegression, probs: np.ndarray) -> np.ndarray:
    eps = 1e-10
    logits = np.log(np.clip(probs, eps, 1 - eps) / np.clip(1 - probs, eps, 1 - eps))
    return lr.predict_proba(logits.reshape(-1, 1))[:, 1]


def predict_transformer_on_val_tm() -> tuple[np.ndarray, np.ndarray]:
    """Прогон ruRoBERTa-large на val_tm. Возвращает (probs_ai, labels)."""
    print("[transformer] loading model from", TRANSFORMER_DIR)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[transformer] device = {device}")
    tok = AutoTokenizer.from_pretrained(str(TRANSFORMER_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(TRANSFORMER_DIR)).to(device)
    model.eval()

    df = pd.read_json(SPLITS_DIR / "val_tm.jsonl", lines=True)
    print(f"[transformer] val_tm: {len(df)} texts")
    probs = np.zeros(len(df), dtype=np.float32)
    labels = df["label"].to_numpy()

    t0 = time.perf_counter()
    with torch.no_grad():
        for i, text in enumerate(df["text"].tolist()):
            inputs = tok(
                text, return_tensors="pt", truncation=True, max_length=512, padding=True
            ).to(device)
            logits = model(**inputs).logits
            p = torch.softmax(logits, dim=-1).cpu().numpy()[0, 1]
            probs[i] = float(p)
            if (i + 1) % 100 == 0:
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / elapsed
                eta = (len(df) - (i + 1)) / rate
                print(f"  [{i+1}/{len(df)}] {rate:.1f} text/s, ETA {eta:.0f}s")
    print(f"[transformer] done in {time.perf_counter() - t0:.1f}s")
    return probs, labels


def predict_tfidf_on_split(name: str) -> tuple[np.ndarray, np.ndarray]:
    """TF-IDF через сервисный артефакт на указанном сплите."""
    print(f"[tfidf] predicting on {name}...")
    df = pd.read_json(SPLITS_DIR / f"{name}.jsonl", lines=True)
    char_vec = joblib.load(SERVICE_DIR / "tfidf_char_tm.joblib")
    word_vec = joblib.load(SERVICE_DIR / "tfidf_word_tm.joblib")
    clf = joblib.load(SERVICE_DIR / "model_C_logreg.joblib")

    from scipy.sparse import hstack

    X_char = char_vec.transform(df["text"])
    X_word = word_vec.transform(df["text"])
    X = hstack([X_char, X_word])
    probs = clf.predict_proba(X)[:, 1]
    labels = df["label"].to_numpy()
    return probs.astype(np.float32), labels


def main() -> None:
    summary_rows: list[dict] = []
    full_results: dict = {}

    # ------------------------------------------------------------------
    # 1. Загрузка cached predictions трансформера (test_tm, holdout, adv)
    # ------------------------------------------------------------------
    print("\n=== Loading cached transformer predictions ===")
    cached = json.loads(CACHED_PREDS_PATH.read_text(encoding="utf-8"))
    t_probs = {}
    t_labels = {}
    for split in ("test_tm", "holdout_tm", "adversarial"):
        t_probs[split] = np.array(cached[split]["probs"], dtype=np.float32)
        t_labels[split] = np.array(cached[split]["labels"], dtype=int)
        print(f"  transformer {split}: n={len(t_probs[split])}")

    # ------------------------------------------------------------------
    # 2. val_tm predictions трансформера (из кэша или новый прогон)
    # ------------------------------------------------------------------
    if "val_tm" in cached:
        print("\n=== val_tm transformer predictions: using cached ===")
        val_probs_t = np.array(cached["val_tm"]["probs"], dtype=np.float32)
        val_labels_t = np.array(cached["val_tm"]["labels"], dtype=int)
        print(f"  cached val_tm: n={len(val_probs_t)}")
    else:
        print("\n=== Transformer on val_tm (for Platt fit) ===")
        val_probs_t, val_labels_t = predict_transformer_on_val_tm()
        cached["val_tm"] = {
            "probs": val_probs_t.tolist(),
            "labels": val_labels_t.tolist(),
            "preds": (val_probs_t >= 0.5).astype(int).tolist(),
        }
        CACHED_PREDS_PATH.write_text(
            json.dumps(cached, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  saved val_tm predictions back to {CACHED_PREDS_PATH}")
    t_probs["val_tm"] = val_probs_t
    t_labels["val_tm"] = val_labels_t

    # ------------------------------------------------------------------
    # 3. TF-IDF на val_tm/test_tm/holdout_tm
    # ------------------------------------------------------------------
    print("\n=== TF-IDF predictions ===")
    c_probs = {}
    c_labels = {}
    for split in ("val_tm", "test_tm", "holdout_tm_unseen"):
        c_probs[split], c_labels[split] = predict_tfidf_on_split(split)

    # alias для согласованности с трансформером
    c_probs["holdout_tm"] = c_probs.pop("holdout_tm_unseen")
    c_labels["holdout_tm"] = c_labels.pop("holdout_tm_unseen")

    # ------------------------------------------------------------------
    # 4. Обучить Platt на val_tm
    # ------------------------------------------------------------------
    print("\n=== Fit Platt on val_tm ===")
    platt_t = fit_platt(t_probs["val_tm"], t_labels["val_tm"])
    platt_c = fit_platt(c_probs["val_tm"], c_labels["val_tm"])
    print("  Platt fitted for transformer and TF-IDF")

    # ------------------------------------------------------------------
    # 5. Применить Platt + посчитать ECE/Brier на test_tm и holdout_tm
    # ------------------------------------------------------------------
    print("\n=== ECE/Brier on test_tm and holdout_tm (independent of Platt fit) ===")
    for model_name, probs_dict, labels_dict, platt in [
        ("transformer", t_probs, t_labels, platt_t),
        ("tfidf", c_probs, c_labels, platt_c),
    ]:
        for split in ("test_tm", "holdout_tm"):
            probs_raw = probs_dict[split]
            labels = labels_dict[split]
            probs_platt = apply_platt(platt, probs_raw)

            ece_raw = ece_score(labels, probs_raw)
            ece_pl = ece_score(labels, probs_platt)
            brier_raw = float(brier_score_loss(labels, probs_raw))
            brier_pl = float(brier_score_loss(labels, probs_platt))

            row = {
                "model": model_name,
                "split": split,
                "n": int(len(labels)),
                "ece_raw": round(ece_raw, 4),
                "ece_platt": round(ece_pl, 4),
                "brier_raw": round(brier_raw, 4),
                "brier_platt": round(brier_pl, 4),
            }
            summary_rows.append(row)
            print(
                f"  {model_name:12s} {split:12s} n={row['n']:6d}  "
                f"ECE: {ece_raw:.4f} -> {ece_pl:.4f}  "
                f"Brier: {brier_raw:.4f} -> {brier_pl:.4f}"
            )

    # Diagnostic only: ECE on val_tm (для информации, не для оценки)
    print("\n=== Reference ECE on val_tm (where Platt was fit — not an honest evaluation) ===")
    for model_name, probs_dict, labels_dict, platt in [
        ("transformer", t_probs, t_labels, platt_t),
        ("tfidf", c_probs, c_labels, platt_c),
    ]:
        probs_raw = probs_dict["val_tm"]
        labels = labels_dict["val_tm"]
        probs_platt = apply_platt(platt, probs_raw)
        ece_raw = ece_score(labels, probs_raw)
        ece_pl = ece_score(labels, probs_platt)
        print(
            f"  {model_name:12s} val_tm     ECE: {ece_raw:.4f} -> {ece_pl:.4f}  (reference only)"
        )
        full_results[f"{model_name}_val_tm_ece_raw"] = round(ece_raw, 4)
        full_results[f"{model_name}_val_tm_ece_platt"] = round(ece_pl, 4)

    # ------------------------------------------------------------------
    # 6. Сохранить артефакты
    # ------------------------------------------------------------------
    df = pd.DataFrame(summary_rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved {OUT_CSV}")

    full_results["splits"] = summary_rows
    full_results["_methodology"] = (
        "Platt scaling fit on val_tm; ECE/Brier reported on independent test_tm and "
        "holdout_tm. ECE n_bins=10 with right-edge inclusive last bin. "
        "Brier = mean squared error between probability and binary label."
    )
    OUT_JSON.write_text(json.dumps(full_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {OUT_JSON}")

    # ------------------------------------------------------------------
    # 7. Reliability diagram для трансформера (test_tm vs holdout_tm)
    # ------------------------------------------------------------------
    print("\n=== Reliability diagram ===")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (split, color) in zip(axes, [("test_tm", "tab:blue"), ("holdout_tm", "tab:red")]):
        probs_raw = t_probs[split]
        probs_platt = apply_platt(platt_t, probs_raw)
        labels = t_labels[split]

        for probs, label, ls in [
            (probs_raw, "raw", "-"),
            (probs_platt, "after Platt", "--"),
        ]:
            n_bins = 10
            edges = np.linspace(0, 1, n_bins + 1)
            mids = (edges[:-1] + edges[1:]) / 2
            accs = []
            confs = []
            ws = []
            for i in range(n_bins):
                if i == n_bins - 1:
                    mask = (probs >= edges[i]) & (probs <= edges[i + 1])
                else:
                    mask = (probs >= edges[i]) & (probs < edges[i + 1])
                if mask.any():
                    accs.append(labels[mask].mean())
                    confs.append(probs[mask].mean())
                    ws.append(mask.sum())
                else:
                    accs.append(np.nan)
                    confs.append(mids[i])
                    ws.append(0)

            ax.plot(confs, accs, marker="o", linestyle=ls, label=label)

        ax.plot([0, 1], [0, 1], "k:", alpha=0.5, label="perfect")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted probability (confidence)")
        ax.set_ylabel("Empirical accuracy")
        ax.set_title(f"ruRoBERTa on {split}")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")

    fig.suptitle("Reliability diagram: ruRoBERTa-large, test_tm (in-distribution) vs holdout_tm (Qwen3 OOD)")
    fig.tight_layout()
    fig.savefig(OUT_RELIABILITY, dpi=120, bbox_inches="tight")
    print(f"Saved {OUT_RELIABILITY}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
