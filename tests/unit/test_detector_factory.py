"""Тесты фабрики детектора: выбор реализации и fallback каскада на TF-IDF."""

from pathlib import Path

import pytest

from app.config import DetectorType, Settings
from app.detectors.cascade import CascadeDetector
from app.detectors.factory import build_detector
from app.detectors.tfidf import TFIDFDetector
from app.detectors.transformer import TransformerDetector


@pytest.fixture
def _no_op_load(monkeypatch):
    """Заглушка load() для всех трёх детекторов, чтобы тесты не зависели от артефактов."""

    def _noop(self):
        return None

    monkeypatch.setattr(TFIDFDetector, "load", _noop)
    monkeypatch.setattr(TransformerDetector, "load", _noop)
    monkeypatch.setattr(CascadeDetector, "load", _noop)


def _settings(detector_type: DetectorType, transformer_exists: bool, tmp_path: Path) -> Settings:
    if transformer_exists:
        transformer_dir = tmp_path / "transformer-present"
        transformer_dir.mkdir()
    else:
        transformer_dir = tmp_path / "transformer-missing"  # не создаём
    return Settings(
        detector_type=detector_type,
        artifacts_dir=tmp_path,
        transformer_dir=transformer_dir,
    )


class TestBuildDetector:
    def test_tfidf_returns_tfidf(self, tmp_path, _no_op_load):
        det = build_detector(_settings(DetectorType.tfidf, False, tmp_path))
        assert isinstance(det, TFIDFDetector)

    def test_transformer_returns_transformer(self, tmp_path, _no_op_load):
        det = build_detector(_settings(DetectorType.transformer, True, tmp_path))
        assert isinstance(det, TransformerDetector)

    def test_transformer_without_artifact_raises(self, tmp_path, _no_op_load):
        with pytest.raises(FileNotFoundError, match="Transformer required"):
            build_detector(_settings(DetectorType.transformer, False, tmp_path))

    def test_cascade_full_returns_cascade(self, tmp_path, _no_op_load):
        det = build_detector(_settings(DetectorType.cascade, True, tmp_path))
        assert isinstance(det, CascadeDetector)
        # Каскад собран с обоими ветвями.
        assert isinstance(det._fast, TFIDFDetector)
        assert isinstance(det._slow, TransformerDetector)

    def test_cascade_without_transformer_fallbacks_to_tfidf_only(
        self, tmp_path, _no_op_load, caplog
    ):
        """Каскад без артефакта трансформера деградирует до TF-IDF, не падает."""
        det = build_detector(_settings(DetectorType.cascade, False, tmp_path))
        assert isinstance(det, CascadeDetector)
        assert isinstance(det._fast, TFIDFDetector)
        assert det._slow is None
        # Предупреждение залогировано — оператор должен это видеть.
        assert any("Transformer dir not found" in r.message for r in caplog.records)
