"""Общий контракт для всех детекторов и value-объект `DetectionResult`."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class DetectionResult:
    prob_ai: float
    method: str
    inference_ms: float
    metadata: dict = field(default_factory=dict)


class BaseDetector(ABC):
    @abstractmethod
    def load(self) -> None:
        """Загружает артефакты модели в память."""

    @abstractmethod
    def predict(self, text: str) -> DetectionResult:
        """Делает инференс по одному предобработанному тексту."""

    @abstractmethod
    def is_ready(self) -> bool:
        """Возвращает `True`, когда артефакты загружены и можно делать инференс."""

    def get_info(self) -> dict:
        """Описание детектора для /ready и логов."""
        return {"type": type(self).__name__, "ready": self.is_ready()}
