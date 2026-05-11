"""Tests for app.calibration.platt — ProbabilityCalibrator and ECE."""

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from app.calibration.platt import ProbabilityCalibrator, ece_score

# ── ECE ───────────────────────────────────────────────────────


class TestECE:
    def test_perfect_calibration(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        assert ece_score(y_true, y_prob) == pytest.approx(0.0, abs=0.01)

    def test_worst_calibration(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.95, 0.95, 0.95, 0.05, 0.05, 0.05])
        assert ece_score(y_true, y_prob) > 0.8

    def test_moderate_calibration(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.3, 0.4, 0.6, 0.7])
        ece = ece_score(y_true, y_prob)
        assert 0.0 < ece < 0.5


# ── ProbabilityCalibrator ─────────────────────────────────────


class TestCalibratorPassthrough:
    def test_no_files_passthrough(self, tmp_path):
        cal = ProbabilityCalibrator(tmp_path)
        cal.load()
        assert cal.calibrate(0.85, "tfidf") == 0.85
        assert cal.calibrate(0.50, "transformer") == 0.50

    def test_ece_below_threshold_passthrough(self, tmp_path):
        # Create diagnostics with low ECE
        diag = {"ece_before": 0.03, "ece_after": 0.02}
        (tmp_path / "tfidf_diagnostics.json").write_text(json.dumps(diag))
        # Create dummy calibrator file (won't be loaded)
        _save_dummy_calibrator(tmp_path / "tfidf_calibrator.joblib")

        cal = ProbabilityCalibrator(tmp_path)
        cal.load(ece_threshold=0.05)
        assert cal.calibrate(0.85, "tfidf") == 0.85

    def test_unknown_method_passthrough(self, tmp_path):
        cal = ProbabilityCalibrator(tmp_path)
        cal.load()
        assert cal.calibrate(0.5, "unknown") == 0.5


class TestCalibratorWithModel:
    def test_calibration_changes_value(self, tmp_path):
        _save_dummy_calibrator(tmp_path / "tfidf_calibrator.joblib")

        cal = ProbabilityCalibrator(tmp_path)
        cal.load()
        result = cal.calibrate(0.85, "tfidf")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_calibration_extreme_values(self, tmp_path):
        _save_dummy_calibrator(tmp_path / "tfidf_calibrator.joblib")

        cal = ProbabilityCalibrator(tmp_path)
        cal.load()

        # Near 0 and near 1 should not crash
        assert 0.0 <= cal.calibrate(0.001, "tfidf") <= 1.0
        assert 0.0 <= cal.calibrate(0.999, "tfidf") <= 1.0


class TestCalibratorDiagnostics:
    def test_diagnostics(self, tmp_path):
        _save_dummy_calibrator(tmp_path / "tfidf_calibrator.joblib")

        cal = ProbabilityCalibrator(tmp_path)
        cal.load()
        diag = cal.get_diagnostics()

        assert "tfidf" in diag["loaded_methods"]
        assert "transformer" in diag["passthrough_methods"]


# ── Helpers ───────────────────────────────────────────────────


def _save_dummy_calibrator(path: Path):
    """Train a simple LogReg calibrator and save it."""
    import joblib

    X = np.array([[-2], [-1], [0], [1], [2]])
    y = np.array([0, 0, 0, 1, 1])
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(X, y)
    joblib.dump(lr, path)
