"""005 case-insensitive uniqueness для users.email."""


from alembic import op

revision = "005_email_lower_index"
down_revision = "004_batch_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("ix_users_email", table_name="users")
    op.execute(
        "CREATE UNIQUE INDEX ix_users_email_lower ON users (lower(email))"
    )


def downgrade() -> None:
    op.drop_index("ix_users_email_lower", table_name="users")
    op.create_index("ix_users_email", "users", ["email"], unique=True)
