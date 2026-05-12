"""Настройки приложения. Грузятся из ENV и `.env`-файлов.

Все параметры собраны в одном месте: режим детектора, пороги, пути, логирование.
"""

from enum import Enum
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Self

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings

from app.schemas import DEFAULT_DISCLAIMER

# JWT_SECRET с этим префиксом считается дефолтным dev-секретом — в prod запрещён.
# Префикс вынесен сюда, чтобы валидатор Settings ловил его до bootstrap'а воркера,
# а не только в lifespan API.
_DEFAULT_JWT_SECRET_PREFIX = "dev-only-secret"

# Признаки локального DSN: prod-инстанс к localhost подключаться не должен.
_LOCAL_DB_HOSTS = ("localhost", "127.0.0.1", "::1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_service_version() -> str:
    """Достать версию пакета из единого источника — pyproject.toml.

    Сначала пробуем installed metadata (быстро, работает после `pip install`).
    Fallback — прямое чтение pyproject.toml в репо и Docker-образе.
    """
    try:
        return metadata.version("ai-text-trust-service")
    except metadata.PackageNotFoundError:
        pass

    import tomllib

    for candidate in (PROJECT_ROOT / "pyproject.toml", Path("/app/pyproject.toml")):
        if candidate.is_file():
            with open(candidate, "rb") as f:
                data = tomllib.load(f)
            version = data.get("project", {}).get("version")
            if version:
                return str(version)
    return "0.0.0-dev"


class DetectorType(str, Enum):
    tfidf = "tfidf"
    transformer = "transformer"
    cascade = "cascade"


class Settings(BaseSettings):
    # Метаданные сервиса.
    # Источник правды — pyproject.toml. ENV SERVICE_VERSION оставлен как
    # override для случаев, когда тег образа отличается от значения в pyproject
    # (например, hotfix-build без bump'а версии).
    service_version: str = Field(default_factory=_resolve_service_version)
    model_version: str = "ru-detector-0.1.0"
    # Дефолт True — fail-closed: забытая переменная не должна тихо отключить
    # prod-инварианты. Локалка и тесты явно ставят IS_PRODUCTION=false
    # (см. .env.example и tests/conftest.py).
    is_production: bool = True

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

    @model_validator(mode="after")
    def _check_prod_invariants(self) -> Self:
        """Прод-инварианты по 12-factor: ловим конфиг-ошибки при создании Settings,
        до lifespan. Иначе воркер (worker.py) стартует с дефолтным JWT_SECRET,
        потому что валидация раньше жила только в FastAPI lifespan.
        """
        if not self.is_production:
            return self

        errors: list[str] = []

        if self.jwt_secret.startswith(_DEFAULT_JWT_SECRET_PREFIX):
            errors.append(
                "JWT_SECRET остался дефолтным dev-значением — задайте 32+ случайных байт"
            )

        if any(host in self.database_url for host in _LOCAL_DB_HOSTS):
            errors.append("DATABASE_URL указывает на localhost в production")

        origins = [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        if "*" in origins:
            errors.append("CORS_ORIGINS='*' недопустим с allow_credentials=True")
        for origin in origins:
            if origin.startswith("http://"):
                errors.append(f"CORS_ORIGINS содержит http:// origin без TLS: {origin}")

        if errors:
            raise ValueError("Production config rejected: " + "; ".join(errors))
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Возвращает синглтон `Settings`: парсим ENV ровно один раз за процесс.

    Без кеша каждый `Depends(get_settings)` на защищённом маршруте
    переинициализировал бы `Settings`. Pydantic тут не дорог, но это горячий
    путь. Тесты, которым нужны нестандартные настройки, создают `Settings(...)`
    напрямую и не вызывают эту функцию.
    """
    return Settings()
