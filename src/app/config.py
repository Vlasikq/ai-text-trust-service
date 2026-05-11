"""Настройки приложения. Грузятся из ENV и `.env`-файлов.

Все параметры собраны в одном месте: режим детектора, пороги, пути, логирование.
"""

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings

from app.schemas import DEFAULT_DISCLAIMER

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class DetectorType(str, Enum):
    tfidf = "tfidf"
    transformer = "transformer"
    cascade = "cascade"


class Settings(BaseSettings):
    # Метаданные сервиса.
    # service_version должен совпадать с тегом образа aitrust-api:<service_version>
    # и значением в pyproject.toml. Попадает в /health, /ready, Swagger UI
    # и поле Analysis.service_version.
    service_version: str = "0.3.0"
    model_version: str = "ru-detector-0.1.0"
    # В проде включается через ENV IS_PRODUCTION=true.
    # При is_production=true lifespan не стартует с дефолтным JWT_SECRET.
    is_production: bool = False

    # Детектор. По умолчанию TF-IDF — на holdout он часто устойчивее каскада.
    # Каскад включается через DETECTOR_TYPE=cascade.
    detector_type: DetectorType = DetectorType.tfidf
    artifacts_dir: Path = PROJECT_ROOT / "artifacts" / "service"
    transformer_dir: Path = PROJECT_ROOT / "artifacts" / "transformer" / "model"

    # Ограничения по длине текста.
    min_chars: int = 300
    max_chars: int = 8000

    # Пороги риска и вердикта.
    risk_thresh_low: float = 0.30
    risk_thresh_high: float = 0.70
    verdict_threshold: float = 0.50

    # Каскад.
    cascade_lo: float = 0.30
    cascade_hi: float = 0.70
    cascade_timeout_ms: int = Field(
        default=3000,
        validation_alias=AliasChoices("CASCADE_TIMEOUT_MS", "REQUEST_TIMEOUT_MS"),
    )

    # Калибровка.
    calibration_enabled: bool = True

    # Наблюдаемость.
    log_level: str = "INFO"
    log_text: bool = Field(
        default=False, description="Log raw text (privacy risk, use only for debug)"
    )

    # База данных.
    database_url: str = "postgresql+asyncpg://aitrust:aitrust@localhost:5432/aitrust"
    db_pool_size: int = 5
    db_pool_overflow: int = 5
    db_fallback_log_path: Path = PROJECT_ROOT / "logs" / "db_fallback.jsonl"

    # Кеш.
    cache_maxsize: int = 1024

    # Дисклеймер. Подставляется в каждый AnalyzeResponse и в описание Swagger.
    # Переопределяется через ENV DISCLAIMER_TEXT для разных каналов
    # (PWA, Telegram-бот, API).
    disclaimer_text: str = DEFAULT_DISCLAIMER

    # Авторизация. JWT-секрет используется для access-токенов; в проде
    # обязательно переопределяется через ENV. Refresh-токены — непрозрачные
    # 256 случайных бит, в БД лежит sha256.
    jwt_secret: str = Field(
        default="dev-only-secret-change-me-in-production",
        validation_alias=AliasChoices("JWT_SECRET", "AUTH_JWT_SECRET"),
    )
    jwt_algorithm: str = "HS256"
    # iss/aud по RFC 7519 §4.1.1/§4.1.3 (OWASP ASVS V3.5): подписав токеном с другим
    # `iss`/`aud`, атакующий не сможет переиспользовать его в нашем API.
    jwt_issuer: str = "aitrust"
    jwt_audience: str = "aitrust-api"
    access_token_ttl_minutes: int = 15  # короткоживущий access JWT
    refresh_token_ttl_days: int = 30  # долгий refresh, ротируется на каждом /auth/refresh
    # CORS_ORIGINS — список origin'ов через запятую. Пустая строка означает same-origin.
    cors_origins: str = Field(default="", validation_alias=AliasChoices("CORS_ORIGINS"))

    # Rate-limit. slowapi считает лимиты по ключу: анонимы — по IP,
    # авторизованные — по префиксу access-токена. Brute-force защита /auth/login
    # сделана отдельным более жёстким лимитом. Лимиты в нотации slowapi: "N/period".
    # На endpoint висит один общий лимит, разделение клиентов идёт через key_func.
    rate_limit_analyze: str = "60/minute"
    rate_limit_jobs: str = "20/minute"
    rate_limit_feedback: str = "30/minute"
    rate_limit_me: str = "60/minute"
    rate_limit_login: str = "10/minute"
    rate_limit_register: str = "5/hour"
    rate_limit_refresh: str = "60/minute"
    rate_limit_logout: str = "10/minute"
    # Batch — отдельный жёсткий лимит: 100 текстов на загрузку, 5 загрузок в час.
    # Этого хватает для типового сценария (преподаватель проверяет работы).
    rate_limit_batch: str = "5/hour"
    # Extract — предпросмотр при drag-and-drop файла. Лимит мягче, чем у batch,
    # но строго: endpoint не должен заменять собой полноценный анализ.
    rate_limit_extract: str = "30/minute"

    # .env — общие dev-настройки (версия, пороги, эндпоинты LLM).
    # .env.local — личные креды конкретного разработчика; перекрывает значения
    # из .env, потому что последний файл в кортеже имеет приоритет.
    model_config = {
        "env_file": (".env", ".env.local"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Возвращает синглтон `Settings`: парсим ENV ровно один раз за процесс.

    Без кеша каждый `Depends(get_settings)` на защищённом маршруте
    переинициализировал бы `Settings`. Pydantic тут не дорог, но это горячий
    путь. Тесты, которым нужны нестандартные настройки, создают `Settings(...)`
    напрямую и не вызывают эту функцию.
    """
    return Settings()
