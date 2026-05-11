"""ORM-модели сервиса: analyses, feedback, model_deployments и связанные таблицы.

Принятые решения по схеме:
  - warnings и explanation хранятся как JSONB в analyses, а не отдельными таблицами;
  - первичные ключи — UUID v7, что даёт time-sortable значения и убирает фрагментацию B-tree;
  - сырой текст не сохраняется (приватность по умолчанию), в БД лежит только text_hash.
"""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from uuid_utils import uuid7


def _uuid7() -> str:
    return uuid7()


class Base(DeclarativeBase):
    pass


# Таблица analyses.


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid7)
    request_id: Mapped[str] = mapped_column(UUID(as_uuid=True), unique=True, nullable=False)
    text_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    text_length: Mapped[int] = mapped_column(Integer, nullable=False)

    # Detection result
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    verdict: Mapped[str | None] = mapped_column(String(10), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    risk_level: Mapped[str | None] = mapped_column(String(10), nullable=True)

    # Detector info
    detector_used: Mapped[str] = mapped_column(String(20), nullable=False)
    cascade_path: Mapped[str | None] = mapped_column(String(20), nullable=True)
    inference_ms: Mapped[float] = mapped_column(Float, nullable=False)
    cached: Mapped[bool] = mapped_column(Boolean, default=False)

    # Versioning
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    service_version: Mapped[str] = mapped_column(String(20), nullable=False)
    deployment_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("model_deployments.id"),
        nullable=True,
    )
    # Анонимные запросы — user_id NULL; авторизованные привязаны к пользователю.
    user_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    # JSONB fields (instead of separate tables)
    warnings: Mapped[list] = mapped_column(JSONB, default=list)
    explanation: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Timestamps
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    persisted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default="now()")

    # Relationships
    deployment: Mapped["ModelDeployment | None"] = relationship(
        back_populates="analyses", lazy="selectin"
    )
    feedback_items: Mapped[list["Feedback"]] = relationship(
        back_populates="analysis", lazy="selectin"
    )
    user: Mapped["User | None"] = relationship(back_populates="analyses", lazy="selectin")

    __table_args__ = (
        CheckConstraint(
            "status IN ('SUCCESS', 'NO_DECISION', 'ERROR')",
            name="ck_analyses_status",
        ),
        CheckConstraint(
            "verdict IS NULL OR verdict IN ('human', 'ai')",
            name="ck_analyses_verdict",
        ),
        CheckConstraint(
            "risk_level IS NULL OR risk_level IN ('HIGH', 'MEDIUM', 'LOW')",
            name="ck_analyses_risk_level",
        ),
        Index("ix_analyses_requested_at", "requested_at"),
        Index("ix_analyses_risk_level", "risk_level"),
        Index("ix_analyses_text_hash", "text_hash"),
        Index("ix_analyses_user_id_requested_at", "user_id", "requested_at"),
    )


# Таблица feedback.


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid7)
    analysis_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analyses.id", ondelete="SET NULL"),
        nullable=True,
    )
    text_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_verdict: Mapped[str] = mapped_column(String(10), nullable=False)
    source: Mapped[str] = mapped_column(String(20), default="api")
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default="now()")

    # Relationships
    analysis: Mapped["Analysis | None"] = relationship(back_populates="feedback_items")

    __table_args__ = (
        CheckConstraint(
            "user_verdict IN ('human', 'ai')",
            name="ck_feedback_user_verdict",
        ),
        CheckConstraint(
            "source IN ('api', 'gradio', 'manual')",
            name="ck_feedback_source",
        ),
        Index("ix_feedback_analysis_id", "analysis_id"),
        Index("ix_feedback_text_hash", "text_hash"),
    )


# Таблица analysis_jobs.


class JobStatus:
    """String constants для status field в analysis_jobs."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"


class BatchStatus:
    """String constants для status field в batch_jobs."""
    PENDING = "PENDING"  # ни один child-job ещё не закончился
    PROCESSING = "PROCESSING"  # хотя бы один child-job в работе/завершён, но не все
    COMPLETED = "COMPLETED"  # все child-job завершены (успешно или с ошибкой)


class AnalysisJob(Base):
    """Async-job для cascade-detection в серой зоне (трансформер 5–15 сек на CPU).

    Privacy: `input_text` хранится только пока job inflight (PENDING/PROCESSING),
    обнуляется при завершении. text_hash остаётся для cache lookup и аналитики.
    """

    __tablename__ = "analysis_jobs"

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid7)
    request_id: Mapped[str] = mapped_column(UUID(as_uuid=True), unique=True, nullable=False)

    # Inflight payload — обнуляется после finished (см. JobRepository.mark_success/_error)
    input_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    text_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    text_length: Mapped[int] = mapped_column(Integer, nullable=False)
    detector_type: Mapped[str] = mapped_column(String(20), nullable=False, default="cascade")
    explain: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Lifecycle
    status: Mapped[str] = mapped_column(String(20), nullable=False, default=JobStatus.PENDING)
    worker_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Result link — FK к Analysis после SUCCESS
    analysis_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analyses.id", ondelete="SET NULL"),
        nullable=True,
    )

    # User attribution — anonymous-режим оставляет NULL (как и в analyses).
    user_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Группировка под batch (P1). Без FK constraint — иначе CASCADE удалил бы
    # child analysis_jobs при удалении batch_jobs, ломая цепочку к analyses.
    batch_id: Mapped[str | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    # Идентификатор и метка из исходного CSV пользователя — для матчинга в отчёте.
    batch_external_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    batch_external_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default="now()", nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        CheckConstraint(
            "status IN ('PENDING', 'PROCESSING', 'SUCCESS', 'ERROR')",
            name="ck_analysis_jobs_status",
        ),
        Index("ix_analysis_jobs_status_created_at", "status", "created_at"),
        Index("ix_analysis_jobs_text_hash", "text_hash"),
        Index("ix_analysis_jobs_user_id_batch_id", "user_id", "batch_id"),
    )


# Таблица batch_jobs.


class BatchJob(Base):
    """Batch-режим: один CSV → один BatchJob ⇄ N AnalysisJob.

    Счётчики `n_completed` / `n_failed` инкрементятся атомарно через SQL
    `UPDATE … SET n_completed = n_completed + 1` (см. worker._on_child_done) —
    иначе при `--scale worker=2` race condition теряет события.

    Partial unique index `uq_one_active_batch_per_user` (см. миграцию 004)
    гарантирует «не более одного активного batch'а на пользователя» атомарно
    на уровне БД. На втором POST endpoint ловит IntegrityError → 409.
    """

    __tablename__ = "batch_jobs"

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid7)
    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    file_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    n_total: Mapped[int] = mapped_column(Integer, nullable=False)
    n_completed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # skipped_summary — массив {row_num, reason, external_id} для строк CSV,
    # которые не создали analysis_jobs (короткий/длинный/пустой текст, лимит).
    skipped_summary: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)

    status: Mapped[str] = mapped_column(String(20), nullable=False, default="PENDING")
    detector_type: Mapped[str] = mapped_column(String(20), nullable=False, default="tfidf")
    explain: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default="now()", nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        CheckConstraint(
            "status IN ('PENDING', 'PROCESSING', 'COMPLETED')",
            name="ck_batch_jobs_status",
        ),
        CheckConstraint("n_total > 0", name="ck_batch_jobs_n_total_positive"),
        CheckConstraint(
            "n_completed >= 0 AND n_failed >= 0 AND n_completed + n_failed <= n_total",
            name="ck_batch_jobs_counters",
        ),
        Index("ix_batch_jobs_user_id_created_at", "user_id", "created_at"),
        # Partial unique: гарантирует не более одного активного batch'а на user.
        # postgresql_where позволяет SQLAlchemy `metadata.create_all` (тесты
        # с testcontainers) видеть тот же index, что прописан в alembic-миграции.
        Index(
            "uq_one_active_batch_per_user",
            "user_id",
            unique=True,
            postgresql_where="status IN ('PENDING', 'PROCESSING')",
        ),
    )


# Таблицы users и refresh_tokens.


class UserStatus:
    ACTIVE = "active"
    DISABLED = "disabled"


class UserRole:
    USER = "user"
    ADMIN = "admin"


class User(Base):
    """Зарегистрированный пользователь сервиса.

    Email — уникальный логин; password_hash — argon2id (см. app.auth.passwords).
    Анонимный режим сохраняется: `analyses.user_id` nullable, sync `/analyze` без auth работает.
    """

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid7)
    email: Mapped[str] = mapped_column(String(254), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default=UserStatus.ACTIVE)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default=UserRole.USER)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default="now()", nullable=False
    )
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    refresh_tokens: Mapped[list["RefreshToken"]] = relationship(
        back_populates="user", lazy="selectin", cascade="all, delete-orphan"
    )
    analyses: Mapped[list["Analysis"]] = relationship(back_populates="user", lazy="noload")

    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'disabled')",
            name="ck_users_status",
        ),
        CheckConstraint(
            "role IN ('user', 'admin')",
            name="ck_users_role",
        ),
        # Expression-index по lower(email) — case-insensitive uniqueness.
        Index(
            "ix_users_email_lower",
            text("lower(email)"),
            unique=True,
        ),
    )


class RefreshToken(Base):
    """Refresh-токен с поддержкой rotation + reuse detection.

    Хранится только sha256 от токена (token_hash). При refresh старый токен
    помечается revoked=True и поле replaced_by_id указывает на новый — это
    позволяет детектировать reuse украденного токена и инвалидировать всю цепь.
    """

    __tablename__ = "refresh_tokens"

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid7)
    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    replaced_by_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("refresh_tokens.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default="now()", nullable=False
    )
    user_agent: Mapped[str | None] = mapped_column(String(255), nullable=True)
    ip: Mapped[str | None] = mapped_column(String(45), nullable=True)

    user: Mapped["User"] = relationship(back_populates="refresh_tokens")

    __table_args__ = (
        Index("ix_refresh_tokens_token_hash", "token_hash", unique=True),
        Index("ix_refresh_tokens_user_id", "user_id"),
    )


# Таблица model_deployments.


class ModelDeployment(Base):
    __tablename__ = "model_deployments"

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid7)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    service_version: Mapped[str] = mapped_column(String(20), nullable=False)
    artifacts_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    code_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    detector_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    sanity_check_passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    deployed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default="now()")
    config_snapshot: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Relationships
    analyses: Mapped[list["Analysis"]] = relationship(back_populates="deployment", lazy="selectin")
