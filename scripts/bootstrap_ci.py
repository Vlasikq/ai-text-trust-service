"""
Bootstrap 95% CI для ключевых метрик:
  - Qwen3-holdout F1 (TF-IDF, ruRoBERTa, Cascade)
  - Adversarial v1 recall (TF-IDF, ruRoBERTa, Cascade)
  - Adversarial v2 (homoglyph) RoBERTa Δrecall vs base (опционально)

Метод: stratified resampling по классу AI/Human, n_iter=1000 итераций.

Входы:
  - artifacts/transformer/cached_predictions.json (трансформер на test_tm/holdout/adv/val)
  - artifacts/service/tfidf_*.joblib + model_C_logreg.joblib (TF-IDF)

Выход:
  - artifacts/bootstrap_ci.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.metrics import f1_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
ARTIFACTS = PROJECT_ROOT / "artifacts"
SERVICE_DIR = ARTIFACTS / "service"
CACHED_PREDS = ARTIFACTS / "transformer" / "cached_predictions.json"
ADV_PATH = PROJECT_ROOT / "data" / "adversarial" / "adversarial_paraphrase_GigaChat.jsonl"
OUT_CSV = ARTIFACTS / "bootstrap_ci.csv"

CASCADE_LO = 0.30
CASCADE_HI = 0.70

N_ITER = 1000
RNG_SEED = 42


def bootstrap_ci(
    metric_fn,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_iter: int = N_ITER,
    seed: int = RNG_SEED,
) -> tuple[float, float, float]:
    """Stratified bootstrap CI для метрики. Возвращает (estimate, ci_lo, ci_hi)."""
    rng = np.random.default_rng(seed)
    estimate = metric_fn(y_true, y_pred)

    # Стратификация по классу
    idx_pos = np.where(y_true == 1)[0]
    idx_neg = np.where(y_true == 0)[0]

    samples = np.zeros(n_iter)
    for i in range(n_iter):
        b_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
        b_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
        b_idx = np.concatenate([b_pos, b_neg])
        samples[i] = metric_fn(y_true[b_idx], y_pred[b_idx])

    ci_lo = float(np.percentile(samples, 2.5))
    ci_hi = float(np.percentile(samples, 97.5))
    return float(estimate), ci_lo, ci_hi


def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")


def recall_ai(y_true, y_pred):
    """Recall на классе AI (label=1)."""
    if (y_true == 1).sum() == 0:
        return float("nan")
    return recall_score(y_true, y_pred, pos_label=1)


def predict_tfidf(texts: list[str]) -> np.ndarray:
    char_vec = joblib.load(SERVICE_DIR / "tfidf_char_tm.joblib")
    word_vec = joblib.load(SERVICE_DIR / "tfidf_word_tm.joblib")
    clf = joblib.load(SERVICE_DIR / "model_C_logreg.joblib")
    X = hstack([char_vec.transform(texts), word_vec.transform(texts)])
    return clf.predict_proba(X)[:, 1]


def cascade_combine(tfidf_probs: np.ndarray, transformer_probs: np.ndarray) -> np.ndarray:
    """Если TF-IDF prob в зоне неопределённости [lo, hi] -> берём трансформер."""
    in_grey = (tfidf_probs > CASCADE_LO) & (tfidf_probs < CASCADE_HI)
    return np.where(in_grey, transformer_probs, tfidf_probs)


def main() -> None:
    rows: list[dict] = []
    cached = json.loads(CACHED_PREDS.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # 1. Qwen3-holdout: F1 для TF-IDF, ruRoBERTa, Cascade
    # ------------------------------------------------------------------
    print("\n=== Qwen3-holdout F1 ===")
    holdout_df = pd.read_json(SPLITS_DIR / "holdout_tm_unseen.jsonl", lines=True)
    y_holdout = holdout_df["label"].to_numpy()

    tfidf_probs_holdout = predict_tfidf(holdout_df["text"].tolist())
    transformer_probs_holdout = np.array(cached["holdout_tm"]["probs"])
    cascade_probs_holdout = cascade_combine(tfidf_probs_holdout, transformer_probs_holdout)

    for name, probs in [
        ("TF-IDF", tfidf_probs_holdout),
        ("ruRoBERTa", transformer_probs_holdout),
        ("Cascade", cascade_probs_holdout),
    ]:
        preds = (probs >= 0.5).astype(int)
        est, lo, hi = bootstrap_ci(f1_macro, y_holdout, preds)
        rows.append(
            {
                "split": "Qwen3-holdout",
                "n": len(y_holdout),
                "method": name,
                "metric": "F1-macro",
                "estimate": round(est, 4),
                "ci_lo_95": round(lo, 4),
                "ci_hi_95": round(hi, 4),
            }
        )
        print(f"  {name:12s} F1 = {est:.4f}  [95% CI: {lo:.4f}, {hi:.4f}]")

    # ------------------------------------------------------------------
    # 2. Adversarial v1 (n=511): recall для TF-IDF, ruRoBERTa, Cascade
    # ------------------------------------------------------------------
    print("\n=== Adversarial v1 recall ===")
    adv_transformer_probs = np.array(cached["adversarial"]["probs"])
    adv_labels = np.array(cached["adversarial"]["labels"])
    print(f"  adversarial cached: n={len(adv_labels)}")

    # Прочитаем тексты adversarial для прогона TF-IDF
    # Адверсариал v1 = парафразы AI (label=1 везде)
    adv_texts: list[str] = []
    adv_labels_check: list[int] = []
    for path in [
        PROJECT_ROOT / "data" / "adversarial" / "adversarial_paraphrase_GigaChat.jsonl",
        PROJECT_ROOT / "data" / "adversarial" / "adversarial_paraphrase_yandexgpt-lite.jsonl",
    ]:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            adv_texts.append(obj["text"])
            adv_labels_check.append(int(obj.get("label", 1)))
    adv_texts = adv_texts[: len(adv_labels)]
    adv_labels_check = np.array(adv_labels_check[: len(adv_labels)])
    print(f"  adversarial texts loaded: {len(adv_texts)}")

    tfidf_probs_adv = predict_tfidf(adv_texts)
    cascade_probs_adv = cascade_combine(tfidf_probs_adv, adv_transformer_probs)

    for name, probs in [
        ("TF-IDF", tfidf_probs_adv),
        ("ruRoBERTa", adv_transformer_probs),
        ("Cascade", cascade_probs_adv),
    ]:
        preds = (probs >= 0.5).astype(int)
        est, lo, hi = bootstrap_ci(recall_ai, adv_labels, preds)
        rows.append(
            {
                "split": "adversarial_v1_paraphrase",
                "n": len(adv_labels),
                "method": name,
                "metric": "recall_AI",
                "estimate": round(est, 4),
                "ci_lo_95": round(lo, 4),
                "ci_hi_95": round(hi, 4),
            }
        )
        print(f"  {name:12s} recall = {est:.4f}  [95% CI: {lo:.4f}, {hi:.4f}]")

    # ------------------------------------------------------------------
    # 3. Adversarial v2: пропускаем, raw per-text predictions недоступны
    # (artifacts/adversarial_v2_eval_raw.csv содержит агрегаты, не per-text)
    # ------------------------------------------------------------------
    print("\n[skip] adversarial v2 per-text probabilities not cached - skipping CI")

    # ------------------------------------------------------------------
    # 4. test_tm: F1 для всех (диагностика)
    # ------------------------------------------------------------------
    print("\n=== test_tm F1 (diagnostic) ===")
    test_df = pd.read_json(SPLITS_DIR / "test_tm.jsonl", lines=True)
    # Удалим adversarial из test_tm для честного сравнения
    test_main = test_df[test_df["source_model"] != "adversarial"].reset_index(drop=True)
    y_test = test_main["label"].to_numpy()
    tfidf_probs_test = predict_tfidf(test_main["text"].tolist())
    # Cached transformer test_tm имеет n=1800 - совпадает с test_main
    transformer_probs_test = np.array(cached["test_tm"]["probs"])
    if len(transformer_probs_test) != len(y_test):
        print(
            f"  WARN: transformer test_tm n={len(transformer_probs_test)} vs "
            f"test_main n={len(y_test)} -- skipping diagnostic"
        )
    else:
        cascade_probs_test = cascade_combine(tfidf_probs_test, transformer_probs_test)
        for name, probs in [
            ("TF-IDF", tfidf_probs_test),
            ("ruRoBERTa", transformer_probs_test),
            ("Cascade", cascade_probs_test),
        ]:
            preds = (probs >= 0.5).astype(int)
            est, lo, hi = bootstrap_ci(f1_macro, y_test, preds)
            rows.append(
                {
                    "split": "test_tm",
                    "n": len(y_test),
                    "method": name,
                    "metric": "F1-macro",
                    "estimate": round(est, 4),
                    "ci_lo_95": round(lo, 4),
                    "ci_hi_95": round(hi, 4),
                }
            )
            print(f"  {name:12s} F1 = {est:.4f}  [95% CI: {lo:.4f}, {hi:.4f}]")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved {OUT_CSV}")


if __name__ == "__main__":
    main()
