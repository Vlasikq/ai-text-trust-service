"""003 users + refresh_tokens, analyses.user_id FK.

Revision ID: 003_users
Revises: 002_analysis_jobs
Create Date: 2026-05-06

Auth: свой JWT + refresh-rotation + argon2id.
Анонимный режим сохраняется: `analyses.user_id` nullable, ON DELETE SET NULL.
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

revision = "003_users"
down_revision = "002_analysis_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── users ─────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(254), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default=sa.text("'active'")),
        sa.Column("role", sa.String(20), nullable=False, server_default=sa.text("'user'")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("status IN ('active', 'disabled')", name="ck_users_status"),
        sa.CheckConstraint("role IN ('user', 'admin')", name="ck_users_role"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    # ── refresh_tokens ────────────────────────────────────────
    op.create_table(
        "refresh_tokens",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("token_hash", sa.String(64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "replaced_by_id",
            UUID(as_uuid=True),
            sa.ForeignKey("refresh_tokens.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("user_agent", sa.String(255), nullable=True),
        sa.Column("ip", sa.String(45), nullable=True),
    )
    op.create_index(
        "ix_refresh_tokens_token_hash", "refresh_tokens", ["token_hash"], unique=True
    )
    op.create_index("ix_refresh_tokens_user_id", "refresh_tokens", ["user_id"])

    # ── analyses.user_id FK ───────────────────────────────────
    op.add_column(
        "analyses",
        sa.Column("user_id", UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_analyses_user_id_users",
        "analyses",
        "users",
        ["user_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_analyses_user_id_requested_at",
        "analyses",
        ["user_id", "requested_at"],
    )

    # ── analysis_jobs.user_id FK ──────────────────────────────
    op.add_column(
        "analysis_jobs",
        sa.Column("user_id", UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_analysis_jobs_user_id_users",
        "analysis_jobs",
        "users",
        ["user_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_analysis_jobs_user_id_users", "analysis_jobs", type_="foreignkey"
    )
    op.drop_column("analysis_jobs", "user_id")

    op.drop_index("ix_analyses_user_id_requested_at", table_name="analyses")
    op.drop_constraint("fk_analyses_user_id_users", "analyses", type_="foreignkey")
    op.drop_column("analyses", "user_id")

    op.drop_index("ix_refresh_tokens_user_id", table_name="refresh_tokens")
    op.drop_index("ix_refresh_tokens_token_hash", table_name="refresh_tokens")
    op.drop_table("refresh_tokens")

    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
