"""Каскадный детектор: сначала TF-IDF, трансформер только при неопределённости.

Логика основана на ноутбуке `06_robustness`. TF-IDF выдаёт `prob_ai`. Если значение
лежит ниже `cascade_lo` или выше `cascade_hi`, считаем ответ уверенным и возвращаем
его. Иначе текст попадает в «серую зону» и уходит на трансформер. Если трансформер
недоступен, возвращаем TF-IDF и добавляем предупреждение.

Около 92% запросов отрабатывает TF-IDF по быстрому пути.
"""

import logging

from app.detectors.base import BaseDetector, DetectionResult
from app.schemas import WarningCode

log = logging.getLogger(__name__)


class CascadeDetector(BaseDetector):
    def __init__(
        self,
        fast: BaseDetector,
        slow: BaseDetector | None,
        cascade_lo: float = 0.30,
        cascade_hi: float = 0.70,
    ):
        self._fast = fast
        self._slow = slow
        self._cascade_lo = cascade_lo
        self._cascade_hi = cascade_hi

    def load(self) -> None:
        self._fast.load()
        if self._slow is not None:
            try:
                self._slow.load()
            except Exception:
                log.warning("Slow detector failed to load, cascade will use fast-only fallback")
                self._slow = None

    def predict(self, text: str) -> DetectionResult:
        fast_result = self._fast.predict(text)

        if fast_result.prob_ai <= self._cascade_lo or fast_result.prob_ai >= self._cascade_hi:
            fast_result.metadata["cascade_path"] = "fast"
            return fast_result

        if self._slow is not None and self._slow.is_ready():
            slow_result = self._slow.predict(text)
            slow_result.inference_ms += fast_result.inference_ms
            slow_result.metadata["cascade_path"] = "slow"
            slow_result.metadata["fast_prob"] = fast_result.prob_ai
            return slow_result

        # Трансформер недоступен: возвращаем результат TF-IDF и помечаем запасной путь.
        fast_result.metadata["cascade_path"] = "fallback"
        fast_result.metadata["warning"] = WarningCode.transformer_unavailable.value
        return fast_result

    def is_ready(self) -> bool:
        return self._fast.is_ready()

    def get_info(self) -> dict:
        return {
            "type": "CascadeDetector",
            "ready": self.is_ready(),
            "fast_ready": self._fast.is_ready(),
            "slow_ready": self._slow.is_ready() if self._slow else False,
            "cascade_lo": self._cascade_lo,
            "cascade_hi": self._cascade_hi,
        }
