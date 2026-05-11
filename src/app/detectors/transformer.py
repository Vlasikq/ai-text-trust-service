"""Детектор на трансформере ruRoBERTa-large.

Это медленный путь: около 500 мс на текст на GPU и 2–3 секунды на CPU.
Загружается, только если `detector_type` равен `transformer` или `cascade`.
"""

import logging
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.detectors.base import BaseDetector, DetectionResult

log = logging.getLogger(__name__)


class TransformerDetector(BaseDetector):
    def __init__(self, model_dir: Path, max_length: int = 512):
        self._model_dir = model_dir
        self._max_length = max_length
        self._tokenizer = None
        self._model = None
        self._device = None

    def load(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_dir))
        self._model = AutoModelForSequenceClassification.from_pretrained(str(self._model_dir)).to(
            self._device
        )
        self._model.eval()
        log.info("TransformerDetector loaded from %s (device=%s)", self._model_dir, self._device)

    def predict(self, text: str) -> DetectionResult:
        t0 = time.perf_counter()

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        prob_ai = float(probs[1])
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return DetectionResult(
            prob_ai=prob_ai,
            method="transformer",
            inference_ms=elapsed_ms,
            metadata={"device": str(self._device)},
        )

    def is_ready(self) -> bool:
        return self._model is not None
