"""
Probability calibration via Platt scaling (logistic regression on raw scores).

Workflow:
  1. export_service_artifacts.py trains calibrators on val_tm
  2. Service loads calibrator joblib at startup
  3. At inference: raw prob_ai → calibrated prob_ai

If ECE < 0.05 on validation set, calibrator is passthrough (no distortion).
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np

log = logging.getLogger(__name__)


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE).

    Partitions predictions into equal-width bins and computes
    weighted average of |accuracy - confidence| per bin.
    """
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


class ProbabilityCalibrator:
    def __init__(self, calibration_dir: Path):
        self._calibration_dir = calibration_dir
        self._calibrators: dict[str, object] = {}  # method → fitted LogReg
        self._passthrough: set[str] = set()  # methods where ECE < threshold

    def load(self, ece_threshold: float = 0.05) -> None:
        """Load calibrators from joblib files.

        If a companion *_diagnostics.json file exists, check ECE.
        If ECE_before < threshold, mark method as passthrough.
        """
        for method in ("tfidf", "transformer"):
            cal_path = self._calibration_dir / f"{method}_calibrator.joblib"
            diag_path = self._calibration_dir / f"{method}_diagnostics.json"

            if not cal_path.exists():
                log.info("No calibrator for %s — will passthrough", method)
                self._passthrough.add(method)
                continue

            # Check if calibration is even needed
            if diag_path.exists():
                with open(diag_path) as f:
                    diag = json.load(f)
                if diag.get("ece_before", 1.0) < ece_threshold:
                    ece_val = diag["ece_before"]
                    log.info("%s ECE=%.4f < %.2f — passthrough", method, ece_val, ece_threshold)
                    self._passthrough.add(method)
                    continue

            self._calibrators[method] = joblib.load(cal_path)
            log.info("Loaded calibrator for %s", method)

    def calibrate(self, prob_ai: float, method: str) -> float:
        """Apply Platt scaling to a raw probability."""
        if method in self._passthrough:
            return prob_ai

        cal = self._calibrators.get(method)
        if cal is None:
            return prob_ai

        # Platt scaling: LogReg.predict_proba on log-odds
        logit = np.log(max(prob_ai, 1e-10) / max(1 - prob_ai, 1e-10))
        calibrated = float(cal.predict_proba(np.array([[logit]]))[0, 1])
        return calibrated

    def get_diagnostics(self) -> dict:
        return {
            "loaded_methods": list(self._calibrators.keys()),
            "passthrough_methods": list(self._passthrough),
        }
