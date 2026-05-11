"""Логика анализа без HTTP- и БД-слоёв.

Эти функции вызывает синхронный `/api/v1/analyze` через `routers/analyze.py`
и асинхронный воркер из `worker.py`. Путь анализа в обоих случаях одинаковый.

Тесты: `tests/test_api.py` покрывает синхронный путь end-to-end и заодно
этот модуль; точечные проверки лежат в `tests/test_detection_service.py`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from app.cache import DetectionCache
from app.calibration.platt import ProbabilityCalibrator
from app.config import Settings
from app.detectors.base import BaseDetector, DetectionResult
from app.explanation.markers import StyleExplainer
from app.preprocessing.pipeline import PreprocessResult, TextPreprocessor
from app.schemas import (
    AnalyzeResponse,
    MethodScore,
    RiskLevel,
    Status,
    Verdict,
    WarningCode,
)

log = logging.getLogger(__name__)


@dataclass
class DetectionContext:
    """DI-контейнер для логики анализа. Собирается один раз в lifespan."""

    settings: Settings
    preprocessor: TextPreprocessor
    detector: BaseDetector
    calibrator: ProbabilityCalibrator | None = None
    explainer: StyleExplainer | None = None
    cache: DetectionCache | None = None


@dataclass
class DetectionOutcome:
    """Что возвращает `run_detection` вызывающему коду.

    response       — готовый Pydantic-объект для ответа клиенту.
    detection      — сырой результат детектора (нужен, чтобы сохранить в БД).
    preprocessing  — результат препроцессинга: `text_hash`, был ли обрезан текст, исходная длина.
    cached_hit     — был ли запрос обслужен из кеша.
    """

    response: AnalyzeResponse
    detection: DetectionResult | None
    preprocessing: PreprocessResult
    cached_hit: bool


def _risk_level(confidence: float, lo: float, hi: float) -> RiskLevel:
    if confidence >= hi:
        return RiskLevel.high
    if confidence >= lo:
        return RiskLevel.medium
    return RiskLevel.low


async def run_detection(
    ctx: DetectionContext,
    text: str,
    *,
    request_id: str,
    explain: bool = False,
    debug: bool = False,
    timeout_s: float | None = None,
) -> DetectionOutcome:
    """Прогоняет полный путь анализа: препроцессинг, инференс, калибровка, вердикт, объяснение.

    Параметры:
      ctx          — DI-контейнер с настройками и детектором.
      text         — сырой текст пользователя.
      request_id   — идентификатор для логов и persist (correlation_id из middleware).
      explain      — включить стилометрические маркеры (медленнее).
      debug        — добавить method_scores в ответ.
      timeout_s    — таймаут на инференс в секундах. `None` означает без таймаута
                     и используется в воркере, где каскад в серой зоне может занять
                     больше десяти секунд. Синхронный endpoint передаёт
                     `settings.cascade_timeout_ms / 1000`.

    Функция не ходит в БД, и Prometheus-метрики тоже не пишет. За `record_request()`
    отвечает вызывающий код, потому что он знает контекст: HTTP-запрос или job.
    """
    t0 = time.perf_counter()

    # 1. Препроцессинг.
    prep = ctx.preprocessor(text)
    warnings = list(prep.warnings)

    # Если текст слишком короткий — модель не запускаем.
    if WarningCode.text_too_short in warnings:
        elapsed = (time.perf_counter() - t0) * 1000
        response = AnalyzeResponse(
            request_id=request_id,
            status=Status.no_decision,
            text_length=prep.original_length,
            processing_time_ms=round(elapsed, 2),
            warnings=warnings,
            disclaimer=ctx.settings.disclaimer_text,
        )
        return DetectionOutcome(response=response, detection=None, preprocessing=prep, cached_hit=False)

    # 2. Кеш.
    cached_hit = False
    result: DetectionResult | None = ctx.cache.get(prep.text) if ctx.cache else None
    if result is not None:
        cached_hit = True
    else:
        try:
            if timeout_s is None:
                # Без таймаута: режим воркера, где каскад в серой зоне идёт долго.
                result = await asyncio.to_thread(ctx.detector.predict, prep.text)
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(ctx.detector.predict, prep.text),
                    timeout=timeout_s,
                )
        except asyncio.TimeoutError:
            elapsed = (time.perf_counter() - t0) * 1000
            log.error("Inference timeout after %.0f ms", elapsed)
            response = AnalyzeResponse(
                request_id=request_id,
                status=Status.error,
                text_length=prep.original_length,
                processing_time_ms=round(elapsed, 2),
                warnings=[WarningCode.inference_timeout],
                disclaimer=ctx.settings.disclaimer_text,
            )
            return DetectionOutcome(response=response, detection=None, preprocessing=prep, cached_hit=False)

        if ctx.cache:
            ctx.cache.put(prep.text, result)

    # Если каскад уронил трансформер и сработал запасной путь, прокидываем предупреждение.
    cascade_warning = result.metadata.get("warning")
    if cascade_warning:
        warnings.append(WarningCode(cascade_warning))

    # 3. Калибровка.
    confidence = result.prob_ai
    calibrated: float | None = None
    if ctx.calibrator is not None and ctx.settings.calibration_enabled:
        try:
            calibrated = ctx.calibrator.calibrate(result.prob_ai, result.method)
            confidence = calibrated
        except Exception:
            log.warning("Calibration failed for request %s", request_id, exc_info=True)
            warnings.append(WarningCode.calibration_unavailable)

    # 4. Уровень риска и вердикт.
    risk = _risk_level(confidence, ctx.settings.risk_thresh_low, ctx.settings.risk_thresh_high)
    verdict = Verdict.ai if confidence >= ctx.settings.verdict_threshold else Verdict.human

    if risk == RiskLevel.medium:
        warnings.append(WarningCode.low_confidence)

    # 5. Объяснение (опционально, дороже по времени).
    explanation = None
    if explain and ctx.explainer is not None:
        try:
            explanation = ctx.explainer.explain(prep.text)
            if explanation is not None:
                warnings.append(WarningCode.style_explanation_heuristic)
        except Exception:
            log.warning("Explanation failed for request %s", request_id, exc_info=True)

    # 6. Debug method scores
    method_scores = None
    if debug:
        method_scores = [
            MethodScore(
                method=result.method,
                raw_score=result.prob_ai,
                calibrated_score=calibrated,
                inference_ms=result.inference_ms,
            )
        ]

    elapsed = (time.perf_counter() - t0) * 1000

    response = AnalyzeResponse(
        request_id=request_id,
        status=Status.success,
        verdict=verdict,
        confidence=round(confidence, 4),
        risk_level=risk,
        explanation=explanation,
        method_scores=method_scores,
        text_length=prep.original_length,
        was_truncated=prep.was_truncated,
        processing_time_ms=round(elapsed, 2),
        warnings=warnings,
        disclaimer=ctx.settings.disclaimer_text,
    )

    return DetectionOutcome(
        response=response,
        detection=result,
        preprocessing=prep,
        cached_hit=cached_hit,
    )
