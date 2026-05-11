"""Юнит-тесты TFIDFDetector. Используем реальные artefacts из artifacts/service/
— для tfidf они весят ~2.5 МБ и доступны in-tree."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.detectors.tfidf import TFIDFDetector

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "service"

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS_DIR / "model_C_logreg.joblib").exists(),
    reason="TF-IDF artifacts отсутствуют, тесты требуют artifacts/service/",
)


@pytest.fixture
def loaded_detector() -> TFIDFDetector:
    det = TFIDFDetector(ARTIFACTS_DIR)
    det.load()
    return det


def test_is_ready_false_before_load():
    det = TFIDFDetector(ARTIFACTS_DIR)
    assert det.is_ready() is False


def test_is_ready_true_after_load(loaded_detector):
    assert loaded_detector.is_ready() is True


def test_load_raises_when_artifacts_absent(tmp_path):
    det = TFIDFDetector(tmp_path / "no_such_dir")
    with pytest.raises(FileNotFoundError):
        det.load()


def test_predict_returns_method_tfidf(loaded_detector):
    res = loaded_detector.predict("Это короткий тестовый текст.")
    assert res.method == "tfidf"
    assert 0.0 <= res.prob_ai <= 1.0


def test_predict_deterministic(loaded_detector):
    text = "Один и тот же текст для проверки детерминизма."
    a = loaded_detector.predict(text)
    b = loaded_detector.predict(text)
    assert a.prob_ai == b.prob_ai


def test_predict_records_inference_ms(loaded_detector):
    res = loaded_detector.predict("Текст с замером времени.")
    assert res.inference_ms >= 0.0


def test_predict_empty_string(loaded_detector):
    # TF-IDF на пустой строке даёт нулевую матрицу — predict не должен падать.
    res = loaded_detector.predict("")
    assert 0.0 <= res.prob_ai <= 1.0


def test_predict_handles_unicode(loaded_detector):
    res = loaded_detector.predict("Привет 🌍 emoji и кириллица.")
    assert 0.0 <= res.prob_ai <= 1.0


def test_predict_handles_long_text(loaded_detector):
    # Длиннее обычного max_chars (8000) — детектор сам по себе не должен
    # ограничивать длину, обрезка делается в preprocessing pipeline.
    text = ("Длинное предложение на русском языке. " * 500).strip()
    res = loaded_detector.predict(text)
    assert 0.0 <= res.prob_ai <= 1.0
