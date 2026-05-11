"""Модели запросов и ответов API, плюс справочные перечисления.

`WarningCode` сделан перечислением, а не строкой: опечатки в коде падают сразу,
а не превращаются в новое тихое значение.
"""

from enum import Enum

from pydantic import BaseModel, EmailStr, Field

DEFAULT_DISCLAIMER = (
    "Результат носит вероятностный характер и не является доказательством авторства. "
    "Сервис не проверяет факты и не оценивает качество текста. "
    "Точность снижается на коротких фрагментах, пост-отредактированных AI-текстах и "
    "генераторах, не представленных в обучающей выборке. "
    "Решения с правовыми или дисциплинарными последствиями должен принимать человек."
)

# Перечисления.


class RiskLevel(str, Enum):
    high = "HIGH"
    medium = "MEDIUM"
    low = "LOW"


class Verdict(str, Enum):
    human = "human"
    ai = "ai"


class Status(str, Enum):
    success = "SUCCESS"
    no_decision = "NO_DECISION"
    error = "ERROR"


class WarningCode(str, Enum):
    text_too_short = "TEXT_TOO_SHORT"
    text_truncated = "TEXT_TRUNCATED"
    transformer_unavailable = "TRANSFORMER_UNAVAILABLE"
    low_confidence = "LOW_CONFIDENCE"
    calibration_unavailable = "CALIBRATION_UNAVAILABLE"
    inference_timeout = "INFERENCE_TIMEOUT"
    style_explanation_heuristic = "STYLE_EXPLANATION_HEURISTIC"


# Запрос на анализ.


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50_000)
    explain: bool = False
    debug: bool = False


# Под-модели ответа.


class StyleMarker(BaseModel):
    feature: str
    value: float
    human_baseline: float
    deviation_percent: float
    direction: str  # "above" | "below"


class Explanation(BaseModel):
    top_markers: list[StyleMarker]
    summary: str
    # Шаблонные обороты, найденные в тексте. Пустой список = ничего не нашли;
    # клиент использует это поле для подсветки, не дублируя словарь у себя.
    matched_phrases: list[str] = []


class MethodScore(BaseModel):
    method: str
    raw_score: float
    calibrated_score: float | None = None
    inference_ms: float


# Ответ на анализ.


class AnalyzeResponse(BaseModel):
    request_id: str
    status: Status
    verdict: Verdict | None = None
    confidence: float | None = None
    risk_level: RiskLevel | None = None
    explanation: Explanation | None = None
    method_scores: list[MethodScore] | None = None
    text_length: int
    was_truncated: bool = False
    processing_time_ms: float
    warnings: list[WarningCode] = Field(default_factory=list)
    disclaimer: str = DEFAULT_DISCLAIMER


# Асинхронные задачи.


class JobStatusValue(str, Enum):
    pending = "PENDING"
    processing = "PROCESSING"
    success = "SUCCESS"
    error = "ERROR"


class JobCreateRequest(BaseModel):
    """Тело POST /api/v1/jobs — то же что AnalyzeRequest, но обрабатывается async."""

    text: str = Field(..., min_length=1, max_length=50_000)
    explain: bool = False


class JobCreatedResponse(BaseModel):
    """Ответ на POST /api/v1/jobs — job создан, polling-url для статуса."""

    job_id: str
    request_id: str
    status: JobStatusValue = JobStatusValue.pending
    poll_url: str  # /api/v1/jobs/{job_id}


class JobStatusResponse(BaseModel):
    """Ответ на GET /api/v1/jobs/{id} — статус + результат если SUCCESS."""

    job_id: str
    request_id: str
    status: JobStatusValue
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    result: AnalyzeResponse | None = None
    error: str | None = None


# Авторизация.


class RegisterRequest(BaseModel):
    email: EmailStr
    # Минимум 8 символов; верхняя граница защищает от argon2 DoS (длинная строка → CPU).
    password: str = Field(..., min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., min_length=10, max_length=512)


class TokenPair(BaseModel):
    """Возвращается после login и refresh."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    access_expires_in: int  # секунд до expiry access-токена
    refresh_expires_in: int  # секунд до expiry refresh-токена


class UserPublic(BaseModel):
    """Безопасное представление пользователя — без password_hash."""

    id: str
    email: str
    role: str
    status: str
    created_at: str
    last_login_at: str | None = None


class RegisterResponse(BaseModel):
    """Ответ на /auth/register: пользователь + сразу пара токенов (auto-login после регистрации)."""

    user: UserPublic
    tokens: TokenPair


class AnalysisHistoryItem(BaseModel):
    """Элемент истории /me/analyses."""

    id: str
    request_id: str
    status: Status
    verdict: Verdict | None = None
    confidence: float | None = None
    risk_level: RiskLevel | None = None
    detector_used: str
    text_length: int
    requested_at: str


class AnalysisHistoryPage(BaseModel):
    """Курсорная пагинация: cursor — id последней записи, итератор по requested_at DESC."""

    items: list[AnalysisHistoryItem]
    next_cursor: str | None = None


class AnalysisDetailResponse(AnalysisHistoryItem):
    """Полная карточка анализа для /me/analyses/{id}.

    Расширяет историю полями, которые сохраняются в БД (см. модель `Analysis`):
    explanation (JSONB), warnings (JSONB), inference_ms, cascade_path, cached.
    `was_truncated` в БД не сохраняется — поле опущено намеренно. Текст также
    не сохраняется (privacy-by-design) и в карточке не возвращается.
    """

    inference_ms: float
    cached: bool
    cascade_path: str | None = None
    warnings: list[WarningCode] = Field(default_factory=list)
    explanation: Explanation | None = None
    disclaimer: str = DEFAULT_DISCLAIMER


# Batch-обработка (/api/v1/batch).


class BatchStatusValue(str, Enum):
    pending = "PENDING"
    processing = "PROCESSING"
    completed = "COMPLETED"


class BatchSkippedRow(BaseModel):
    """Строка CSV, которая не была отправлена в обработку (короткий/длинный текст и т.п.)."""

    row_num: int
    reason: str
    external_id: str | None = None


class BatchCreatedResponse(BaseModel):
    """Ответ POST /api/v1/batch — batch создан, jobs поставлены в очередь."""

    batch_id: str
    n_accepted: int  # сколько строк ушло в обработку
    n_skipped: int  # сколько отвергнуто на parsing-фазе
    skipped: list[BatchSkippedRow]
    poll_url: str


class BatchStatusResponse(BaseModel):
    """Статус batch'а: прогресс + skipped из parsing-фазы."""

    batch_id: str
    file_name: str | None
    status: BatchStatusValue
    n_total: int
    n_completed: int
    n_failed: int
    detector_type: str
    explain: bool
    created_at: str
    completed_at: str | None
    skipped: list[BatchSkippedRow]
    results_csv_url: str | None  # доступен только после COMPLETED
    results_json_url: str | None


class BatchListItem(BaseModel):
    """Одна запись в /me/batches."""

    batch_id: str
    file_name: str | None
    status: BatchStatusValue
    n_total: int
    n_completed: int
    n_failed: int
    created_at: str
    completed_at: str | None


class BatchListPage(BaseModel):
    items: list[BatchListItem]
    next_cursor: str | None = None


# Посегментная оценка (/api/v1/analyze/segments).


class SegmentScoreItem(BaseModel):
    """Один скользящий сегмент: позиции в исходном тексте + P(AI)."""

    start_sent: int
    end_sent: int
    start_char: int
    end_char: int
    prob_ai: float


class SegmentSummary(BaseModel):
    n_segments: int
    mean_prob: float | None = None
    min_prob: float | None = None
    max_prob: float | None = None
    n_high: int = 0
    n_medium: int = 0
    n_low: int = 0
    transition_detected: bool = False


class SegmentScoringRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50_000)
    window_size: int = Field(default=3, ge=1, le=10)
    step: int = Field(default=2, ge=1, le=10)


class SegmentScoringResponse(BaseModel):
    """Сегменты + сводка. Дисклеймер про экспериментальность — в UI/Главе 3.5.4."""

    text_length: int
    window_size: int
    step: int
    segments: list[SegmentScoreItem]
    summary: SegmentSummary


# Статистика (/me/stats).


class StatsPeriod(str, Enum):
    p7d = "7d"
    p30d = "30d"
    p90d = "90d"
    p_all = "all"


class DailyBucket(BaseModel):
    """Один день в bar-chart. count=0 для пустых дней (zero-padding обязательна)."""

    date: str  # YYYY-MM-DD
    total: int
    ai: int
    human: int


class StatsResponse(BaseModel):
    """Персональная статистика пользователя за период.

    Агрегаты считаются по `Analysis.user_id == current_user.id`. Для периодов
    `7d/30d/90d` `period_start` рассчитывается от `now()`; для `all` —
    timestamp самого раннего анализа пользователя (или `period_end` если данных нет).
    `daily` всегда содержит каждую дату из окна (zero-padding).
    """

    period: StatsPeriod
    period_start: str
    period_end: str
    total_scans: int
    by_verdict: dict[str, int]
    by_risk_level: dict[str, int]
    by_detector: dict[str, int]
    avg_confidence: float | None
    daily: list[DailyBucket]
