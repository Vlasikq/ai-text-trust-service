"""004 batch_jobs + analysis_jobs.batch_id для группового режима.

Revision ID: 004_batch_jobs
Revises: 003_users
Create Date: 2026-05-10

P1 фича: загрузка CSV → пакет анализов. Один batch_jobs ⇄ N analysis_jobs.

Архитектурные решения (см. PLAN_P1_FEATURES.md):
  - `analysis_jobs.batch_id` — БЕЗ FK на batch_jobs (CASCADE удалил бы child
    analyses, ломая audit-цепочку). Просто UUID-колонка с индексом.
  - Partial unique index на (user_id) WHERE status IN ('PENDING','PROCESSING')
    атомарно гарантирует «не более одного активного batch'а у пользователя» —
    `IntegrityError` на втором POST безопаснее, чем check-then-insert.
  - Счётчики `n_completed/n_failed` инкрементятся атомарно через SQL
    `UPDATE ... SET n_completed = n_completed + 1` (см. worker.py). Поэтому
    тут просто int-колонки с дефолтом 0.
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision = "004_batch_jobs"
down_revision = "003_users"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "batch_jobs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("file_name", sa.String(255), nullable=True),
        sa.Column("n_total", sa.Integer, nullable=False),
        sa.Column("n_completed", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("n_failed", sa.Integer, nullable=False, server_default=sa.text("0")),
        # skipped_summary хранит rows, которые не создали analysis_jobs (text_too_short
        # / text_too_long / empty_text / too_many_rows из csv_parser). Возвращается
        # клиенту в /batch/{id} статусе и в JSON-отчёте.
        sa.Column("skipped_summary", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column(
            "status",
            sa.String(20),
            nullable=False,
            server_default=sa.text("'PENDING'"),
        ),
        sa.Column("detector_type", sa.String(20), nullable=False, server_default=sa.text("'tfidf'")),
        sa.Column("explain", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('PENDING', 'PROCESSING', 'COMPLETED')",
            name="ck_batch_jobs_status",
        ),
        sa.CheckConstraint("n_total > 0", name="ck_batch_jobs_n_total_positive"),
        sa.CheckConstraint(
            "n_completed >= 0 AND n_failed >= 0 AND n_completed + n_failed <= n_total",
            name="ck_batch_jobs_counters",
        ),
    )

    # История batch'ей пользователя — `/me/batches` сортирует по created_at DESC.
    op.create_index(
        "ix_batch_jobs_user_id_created_at",
        "batch_jobs",
        ["user_id", "created_at"],
    )

    # Атомарный single-active-batch lock. Postgres-specific (partial index).
    # При второй попытке создать batch у того же пользователя — IntegrityError.
    op.execute(
        "CREATE UNIQUE INDEX uq_one_active_batch_per_user "
        "ON batch_jobs (user_id) WHERE status IN ('PENDING', 'PROCESSING')"
    )

    # ── analysis_jobs: batch grouping ─────────────────────────
    # batch_id БЕЗ FK constraint: CASCADE-удаление batch_jobs не должно
    # затрагивать child analysis_jobs (а значит и analyses через analysis_id).
    # external_id/name — то, что прислал пользователь в CSV (колонки 'id'/'name').
    # Возвращаются в отчёт, чтобы клиент мог сматчить результат со своими данными.
    op.add_column(
        "analysis_jobs",
        sa.Column("batch_id", UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "analysis_jobs",
        sa.Column("batch_external_id", sa.String(128), nullable=True),
    )
    op.add_column(
        "analysis_jobs",
        sa.Column("batch_external_name", sa.String(255), nullable=True),
    )
    op.create_index(
        "ix_analysis_jobs_user_id_batch_id",
        "analysis_jobs",
        ["user_id", "batch_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_analysis_jobs_user_id_batch_id", table_name="analysis_jobs")
    op.drop_column("analysis_jobs", "batch_external_name")
    op.drop_column("analysis_jobs", "batch_external_id")
    op.drop_column("analysis_jobs", "batch_id")

    op.execute("DROP INDEX IF EXISTS uq_one_active_batch_per_user")
    op.drop_index("ix_batch_jobs_user_id_created_at", table_name="batch_jobs")
    op.drop_table("batch_jobs")
