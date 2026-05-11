"""001 Initial schema: analyses, feedback, model_deployments.

Revision ID: 001_initial
Revises:
Create Date: 2026-04-07
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── model_deployments (referenced by analyses FK) ─────────
    op.create_table(
        "model_deployments",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("model_version", sa.String(50), nullable=False),
        sa.Column("service_version", sa.String(20), nullable=False),
        sa.Column("artifacts_hash", sa.String(64), nullable=True),
        sa.Column("code_hash", sa.String(64), nullable=True),
        sa.Column("detector_type", sa.String(20), nullable=True),
        sa.Column("sanity_check_passed", sa.Boolean, nullable=True),
        sa.Column(
            "deployed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.Column("config_snapshot", JSONB, nullable=True),
    )

    # ── analyses ──────────────────────────────────────────────
    op.create_table(
        "analyses",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("request_id", UUID(as_uuid=True), unique=True, nullable=False),
        sa.Column("text_hash", sa.String(64), nullable=False),
        sa.Column("text_length", sa.Integer, nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("verdict", sa.String(10), nullable=True),
        sa.Column("confidence", sa.Float, nullable=True),
        sa.Column("risk_level", sa.String(10), nullable=True),
        sa.Column("detector_used", sa.String(20), nullable=False),
        sa.Column("cascade_path", sa.String(20), nullable=True),
        sa.Column("inference_ms", sa.Float, nullable=False),
        sa.Column("cached", sa.Boolean, default=False),
        sa.Column("model_version", sa.String(50), nullable=False),
        sa.Column("service_version", sa.String(20), nullable=False),
        sa.Column(
            "deployment_id",
            UUID(as_uuid=True),
            sa.ForeignKey("model_deployments.id"),
            nullable=True,
        ),
        sa.Column("warnings", JSONB, server_default=sa.text("'[]'::jsonb")),
        sa.Column("explanation", JSONB, nullable=True),
        sa.Column("requested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "persisted_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        # Check constraints
        sa.CheckConstraint(
            "status IN ('SUCCESS', 'NO_DECISION', 'ERROR')",
            name="ck_analyses_status",
        ),
        sa.CheckConstraint(
            "verdict IS NULL OR verdict IN ('human', 'ai')",
            name="ck_analyses_verdict",
        ),
        sa.CheckConstraint(
            "risk_level IS NULL OR risk_level IN ('HIGH', 'MEDIUM', 'LOW')",
            name="ck_analyses_risk_level",
        ),
    )
    op.create_index("ix_analyses_requested_at", "analyses", ["requested_at"])
    op.create_index("ix_analyses_risk_level", "analyses", ["risk_level"])
    op.create_index("ix_analyses_text_hash", "analyses", ["text_hash"])

    # ── feedback ──────────────────────────────────────────────
    op.create_table(
        "feedback",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "analysis_id",
            UUID(as_uuid=True),
            sa.ForeignKey("analyses.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("text_hash", sa.String(64), nullable=True),
        sa.Column("user_verdict", sa.String(10), nullable=False),
        sa.Column("source", sa.String(20), server_default=sa.text("'api'")),
        sa.Column("comment", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(
            "user_verdict IN ('human', 'ai')",
            name="ck_feedback_user_verdict",
        ),
        sa.CheckConstraint(
            "source IN ('api', 'gradio', 'manual')",
            name="ck_feedback_source",
        ),
    )
    op.create_index("ix_feedback_analysis_id", "feedback", ["analysis_id"])
    op.create_index("ix_feedback_text_hash", "feedback", ["text_hash"])


def downgrade() -> None:
    op.drop_table("feedback")
    op.drop_table("analyses")
    op.drop_table("model_deployments")
