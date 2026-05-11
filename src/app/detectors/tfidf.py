"""Детектор на TF-IDF и логистической регрессии.

Загружает из `joblib` два TF-IDF-векторизатора (символьный и словный) плюс
обученную `LogisticRegression`. Быстрый путь: около 50 мс на текст.
"""

import logging
import time
from pathlib import Path

import joblib
from scipy.sparse import hstack

from app.detectors.base import BaseDetector, DetectionResult

log = logging.getLogger(__name__)


class TFIDFDetector(BaseDetector):
    def __init__(self, artifacts_dir: Path):
        self._artifacts_dir = artifacts_dir
        self._tfidf_char = None
        self._tfidf_word = None
        self._model = None

    def load(self) -> None:
        char_path = self._artifacts_dir / "tfidf_char_tm.joblib"
        word_path = self._artifacts_dir / "tfidf_word_tm.joblib"
        model_path = self._artifacts_dir / "model_C_logreg.joblib"

        # Запасной путь: артефакты сервиса ещё не экспортированы — берём
        # модели, обученные на не-topic-matched сплитах.
        if not char_path.exists():
            fallback = self._artifacts_dir.parent
            char_path = fallback / "tfidf_char.joblib"
            word_path = fallback / "tfidf_word.joblib"
            model_path = fallback / "model_A_logreg.joblib"
            log.warning("Service artifacts not found, falling back to %s", fallback)

        self._tfidf_char = joblib.load(char_path)
        self._tfidf_word = joblib.load(word_path)
        self._model = joblib.load(model_path)
        log.info("TFIDFDetector loaded from %s", self._artifacts_dir)

    def predict(self, text: str) -> DetectionResult:
        t0 = time.perf_counter()

        X_char = self._tfidf_char.transform([text])
        X_word = self._tfidf_word.transform([text])
        X = hstack([X_char, X_word])

        prob_ai = float(self._model.predict_proba(X)[0, 1])
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return DetectionResult(
            prob_ai=prob_ai,
            method="tfidf",
            inference_ms=elapsed_ms,
        )

    def is_ready(self) -> bool:
        return self._model is not None
