"""
Экспорт артефактов для production-сервиса (Фаза 0).

Переобучает Baseline C на topic-matched данных, вычисляет human baselines,
обучает калибраторы (Platt scaling), сохраняет model_manifest.json.

Вход:  data/splits/train_tm.jsonl, val_tm.jsonl
Выход: artifacts/service/
  ├── tfidf_char_tm.joblib
  ├── tfidf_word_tm.joblib
  ├── model_C_logreg.joblib
  ├── human_baselines.json
  ├── model_manifest.json
  └── calibration/
      ├── tfidf_calibrator.joblib
      ├── tfidf_diagnostics.json
      ├── transformer_calibrator.joblib
      └── transformer_diagnostics.json
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from app.features.stylometric import extract_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "service"
TRANSFORMER_DIR = PROJECT_ROOT / "artifacts" / "transformer"


# ---------------------------------------------------------------------------
# Step 1: Baseline C (TF-IDF + LogReg)
# ---------------------------------------------------------------------------


def train_baseline_c(train_path: Path, val_path: Path, output_dir: Path) -> dict:
    """Переобучить TF-IDF + LogReg на topic-matched данных.

    Параметры из 03_baseline_model.ipynb:
      - char_wb (3,5), max_features=30000, sublinear_tf=True
      - word (1,2), max_features=20000, sublinear_tf=True, token_pattern для кириллицы
      - LogReg: C=50.0, solver=saga, max_iter=1000
      - Обучение на train_tm + val_tm (как в ноутбуке)
    """
    log.info("Loading train_tm + val_tm...")
    train = pd.read_json(train_path, lines=True)
    val = pd.read_json(val_path, lines=True)
    combined = pd.concat([train, val], ignore_index=True)

    log.info("Train: %d, Val: %d, Combined: %d", len(train), len(val), len(combined))

    # TF-IDF vectorizers
    tfidf_char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=30_000,
        sublinear_tf=True,
    )
    tfidf_word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=20_000,
        sublinear_tf=True,
        token_pattern=r"(?i)[а-яёa-z]{2,}",
    )

    # Fit on combined (train + val)
    log.info("Fitting TF-IDF vectorizers...")
    X_char = tfidf_char.fit_transform(combined["text"])
    X_word = tfidf_word.fit_transform(combined["text"])
    X = hstack([X_char, X_word])
    y = combined["label"].values

    log.info(
        "Feature matrix: %d samples × %d features",
        X.shape[0],
        X.shape[1],
    )

    # LogReg
    log.info("Training LogReg (C=50.0, saga)...")
    model = LogisticRegression(
        C=50.0,
        solver="saga",
        max_iter=1000,
        random_state=42,
    )
    model.fit(X, y)

    # Evaluate on val (для метрик в manifest)
    X_val_char = tfidf_char.transform(val["text"])
    X_val_word = tfidf_word.transform(val["text"])
    X_val = hstack([X_val_char, X_val_word])
    y_val = val["label"].values
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    metrics = {
        "f1_macro": round(float(f1_score(y_val, y_pred, average="macro")), 4),
        "auc": round(float(roc_auc_score(y_val, y_prob)), 4),
        "accuracy": round(float((y_pred == y_val).mean()), 4),
        "ece": round(float(_ece_score(y_val, y_prob)), 4),
        "n_train": len(combined),
        "n_val": len(val),
    }
    log.info("Baseline C metrics: %s", metrics)

    joblib.dump(tfidf_char, output_dir / "tfidf_char_tm.joblib")
    joblib.dump(tfidf_word, output_dir / "tfidf_word_tm.joblib")
    joblib.dump(model, output_dir / "model_C_logreg.joblib")
    log.info("Saved TF-IDF + LogReg → %s", output_dir)

    return metrics


# ---------------------------------------------------------------------------
# Step 2: Human baselines
# ---------------------------------------------------------------------------


def compute_human_baselines(train_path: Path, output_dir: Path) -> None:
    """Вычислить медианы стилометрических признаков для human-текстов."""
    log.info("Loading human texts from train_tm...")
    train = pd.read_json(train_path, lines=True)
    human = train[train["label"] == 0]
    log.info("Human texts: %d", len(human))

    log.info("Extracting 45 stylometric features (this takes ~30s)...")
    feat_df = extract_all(human)

    medians = {}
    for col in feat_df.columns:
        medians[col] = round(float(feat_df[col].median()), 6)

    path = output_dir / "human_baselines.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(medians, f, ensure_ascii=False, indent=2)
    log.info("Saved %d baselines → %s", len(medians), path)


# ---------------------------------------------------------------------------
# Step 3: Calibrators (Platt scaling)
# ---------------------------------------------------------------------------


def train_calibrators(val_path: Path, output_dir: Path) -> None:
    """Обучить Platt scaling калибраторы на val_tm."""
    log.info("Loading val_tm for calibration...")
    val = pd.read_json(val_path, lines=True)
    y_val = val["label"].values
    cal_dir = output_dir / "calibration"

    # TF-IDF calibrator
    tfidf_char = joblib.load(output_dir / "tfidf_char_tm.joblib")
    tfidf_word = joblib.load(output_dir / "tfidf_word_tm.joblib")
    model_c = joblib.load(output_dir / "model_C_logreg.joblib")

    X_char = tfidf_char.transform(val["text"])
    X_word = tfidf_word.transform(val["text"])
    X_val = hstack([X_char, X_word])
    tfidf_probs = model_c.predict_proba(X_val)[:, 1]

    ece_before = _ece_score(y_val, tfidf_probs)
    log.info("TF-IDF ECE before calibration: %.4f", ece_before)

    if ece_before < 0.05:
        log.info("TF-IDF already well-calibrated — saving passthrough diagnostics")
        _save_diagnostics(cal_dir, "tfidf", ece_before, ece_before, passthrough=True)
    else:
        tfidf_logits = np.log(
            np.clip(tfidf_probs, 1e-10, 1 - 1e-10)
            / np.clip(1 - tfidf_probs, 1e-10, 1 - 1e-10)
        ).reshape(-1, 1)

        cal = LogisticRegression(solver="lbfgs", max_iter=1000)
        cal.fit(tfidf_logits, y_val)
        cal_probs = cal.predict_proba(tfidf_logits)[:, 1]
        ece_after = _ece_score(y_val, cal_probs)

        joblib.dump(cal, cal_dir / "tfidf_calibrator.joblib")
        _save_diagnostics(cal_dir, "tfidf", ece_before, ece_after)
        log.info("TF-IDF ECE after calibration: %.4f", ece_after)

    # Transformer calibrator (from cached predictions if available)
    cached_path = TRANSFORMER_DIR / "cached_predictions.json"
    if cached_path.exists():
        log.info("Loading transformer cached predictions...")
        with open(cached_path) as f:
            cached = json.load(f)

        # Ищем предсказания для val_tm
        if "val_tm" in cached:
            t_probs = np.array(cached["val_tm"]["probs"])
            t_labels = np.array(cached["val_tm"]["labels"])

            t_ece_before = _ece_score(t_labels, t_probs)
            log.info("Transformer ECE before calibration: %.4f", t_ece_before)

            if t_ece_before < 0.05:
                _save_diagnostics(
                    cal_dir, "transformer",
                    t_ece_before, t_ece_before, passthrough=True,
                )
            else:
                t_logits = np.log(
                    np.clip(t_probs, 1e-10, 1 - 1e-10)
                    / np.clip(1 - t_probs, 1e-10, 1 - 1e-10)
                ).reshape(-1, 1)
                t_cal = LogisticRegression(solver="lbfgs", max_iter=1000)
                t_cal.fit(t_logits, t_labels)
                t_cal_probs = t_cal.predict_proba(t_logits)[:, 1]
                t_ece_after = _ece_score(t_labels, t_cal_probs)

                joblib.dump(t_cal, cal_dir / "transformer_calibrator.joblib")
                _save_diagnostics(cal_dir, "transformer", t_ece_before, t_ece_after)
                log.info("Transformer ECE after calibration: %.4f", t_ece_after)
        else:
            log.warning(
                "No val_tm in cached_predictions.json — skip transformer"
            )
    else:
        log.warning("No cached_predictions.json — skipping transformer calibrator")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ece_score(y_true, y_prob, n_bins=10):
    """Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() / len(y_prob) * abs(bin_acc - bin_conf)
    return float(ece)


def _save_diagnostics(cal_dir, method, ece_before, ece_after, passthrough=False):
    diag = {
        "method": method,
        "ece_before": round(ece_before, 4),
        "ece_after": round(ece_after, 4),
        "passthrough": passthrough,
        "calibrated_at": datetime.now().isoformat(),
    }
    path = cal_dir / f"{method}_diagnostics.json"
    with open(path, "w") as f:
        json.dump(diag, f, indent=2)


def save_manifest(output_dir: Path, metrics: dict) -> None:
    """Создать model_manifest.json с метаданными."""
    manifest = {
        "version": "1.0.0",
        "created_at": datetime.now().isoformat(),
        "training_data": "data/splits/train_tm.jsonl",
        "validation_data": "data/splits/val_tm.jsonl",
        "methods": {
            "tfidf": {
                "artifacts": [
                    "tfidf_char_tm.joblib",
                    "tfidf_word_tm.joblib",
                    "model_C_logreg.joblib",
                ],
                "params": {
                    "char_ngram": [3, 5],
                    "word_ngram": [1, 2],
                    "char_max_features": 30000,
                    "word_max_features": 20000,
                    "C": 50.0,
                    "solver": "saga",
                },
            },
            "transformer": {
                "artifacts": ["../transformer/model/"],
                "source": "artifacts/transformer/model",
                "model_name": "ai-forever/ruRoBERTa-large",
            },
        },
        "calibration": {
            "method": "platt_scaling",
            "artifacts": [
                "calibration/tfidf_calibrator.joblib",
                "calibration/transformer_calibrator.joblib",
            ],
        },
        "thresholds": {
            "cascade_lo": 0.30,
            "cascade_hi": 0.70,
        },
        "metrics": metrics,
    }
    path = output_dir / "model_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log.info("Manifest saved → %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Export service artifacts (Phase 0)")
    parser.add_argument("--train", type=Path, default=DATA_DIR / "train_tm.jsonl")
    parser.add_argument("--val", type=Path, default=DATA_DIR / "val_tm.jsonl")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "calibration").mkdir(exist_ok=True)

    log.info("=== Phase 0: Export service artifacts ===")

    # 1. Baseline C
    log.info("Step 1: Training Baseline C on topic-matched data...")
    metrics = train_baseline_c(args.train, args.val, args.output_dir)

    # 2. Human baselines
    log.info("Step 2: Computing human baselines...")
    compute_human_baselines(args.train, args.output_dir)

    # 3. Calibrators
    log.info("Step 3: Training calibrators...")
    train_calibrators(args.val, args.output_dir)

    # 4. Manifest
    log.info("Step 4: Saving manifest...")
    save_manifest(args.output_dir, metrics)

    log.info("=== Done. Artifacts in %s ===", args.output_dir)


if __name__ == "__main__":
    main()
