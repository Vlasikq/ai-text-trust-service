"""Фабрика детектора: один источник правил выбора реализации и fallback каскада."""

from __future__ import annotations

import logging

from app.config import DetectorType, Settings
from app.detectors.base import BaseDetector
from app.detectors.cascade import CascadeDetector
from app.detectors.tfidf import TFIDFDetector
from app.detectors.transformer import TransformerDetector

log = logging.getLogger(__name__)


def build_detector(settings: Settings) -> BaseDetector:
    """Собирает детектор согласно `settings.detector_type`.

    Правила fallback:
      - `tfidf`        → только TF-IDF;
      - `transformer`  → только трансформер; если артефакт отсутствует — `FileNotFoundError`;
      - `cascade`      → TF-IDF + трансформер; если артефакта трансформера нет,
                         каскад деградирует до «TF-IDF only» (`slow=None`) с предупреждением.
    """
    tfidf = TFIDFDetector(settings.artifacts_dir)

    if settings.detector_type == DetectorType.tfidf:
        tfidf.load()
        return tfidf

    if not settings.transformer_dir.exists():
        log.warning("Transformer dir not found: %s", settings.transformer_dir)
        if settings.detector_type == DetectorType.transformer:
            raise FileNotFoundError(
                f"Transformer required but not found: {settings.transformer_dir}"
            )
        tfidf.load()
        return CascadeDetector(fast=tfidf, slow=None)

    transformer = TransformerDetector(settings.transformer_dir)

    if settings.detector_type == DetectorType.transformer:
        transformer.load()
        return transformer

    cascade = CascadeDetector(
        fast=tfidf,
        slow=transformer,
        cascade_lo=settings.cascade_lo,
        cascade_hi=settings.cascade_hi,
    )
    cascade.load()
    return cascade
