"""002 analysis_jobs table for async processing.

Revision ID: 002_analysis_jobs
Revises: 001_initial
Create Date: 2026-05-05

Worker process опрашивает таблицу через `SELECT ... FOR UPDATE SKIP LOCKED`.
Privacy: `input_text` хранится только пока job в PENDING/PROCESSING,
обнуляется при завершении (см. JobRepository.mark_success / mark_error).
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

revision = "002_analysis_jobs"
down_revision = "001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "analysis_jobs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("request_id", UUID(as_uuid=True), unique=True, nullable=False),
        # Privacy: input_text хранится только inflight (PENDING/PROCESSING),
        # обнуляется в SUCCESS/ERROR. text_hash остаётся для cache lookup.
        sa.Column("input_text", sa.Text, nullable=True),
        sa.Column("text_hash", sa.String(64), nullable=False),
        sa.Column("text_length", sa.Integer, nullable=False),
        sa.Column("detector_type", sa.String(20), nullable=False, server_default=sa.text("'cascade'")),
        sa.Column("explain", sa.Boolean, server_default=sa.false(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default=sa.text("'PENDING'")),
        sa.Column("worker_id", sa.String(100), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column(
            "analysis_id",
            UUID(as_uuid=True),
            sa.ForeignKey("analyses.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('PENDING', 'PROCESSING', 'SUCCESS', 'ERROR')",
            name="ck_analysis_jobs_status",
        ),
    )
    # Worker query path: WHERE status='PENDING' ORDER BY created_at LIMIT 1.
    # Композитный индекс делает SELECT FOR UPDATE SKIP LOCKED моментальным.
    op.create_index(
        "ix_analysis_jobs_status_created_at",
        "analysis_jobs",
        ["status", "created_at"],
    )
    op.create_index("ix_analysis_jobs_text_hash", "analysis_jobs", ["text_hash"])


def downgrade() -> None:
    op.drop_index("ix_analysis_jobs_text_hash", table_name="analysis_jobs")
    op.drop_index("ix_analysis_jobs_status_created_at", table_name="analysis_jobs")
    op.drop_table("analysis_jobs")
