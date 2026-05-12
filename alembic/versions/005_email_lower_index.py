"""005 case-insensitive uniqueness для users.email.

CONCURRENTLY: на пустой users эффекта нет, на проде с растущей таблицей
ACCESS EXCLUSIVE заменяется на SHARE UPDATE EXCLUSIVE — логины не блокируются
на время перестройки индекса. autocommit_block обязателен: CONCURRENTLY не
работает внутри транзакции.
"""


from alembic import op

revision = "005_email_lower_index"
down_revision = "004_batch_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_users_email")
        op.execute(
            "CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS "
            "ix_users_email_lower ON users (lower(email))"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_users_email_lower")
        op.execute(
            "CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS "
            "ix_users_email ON users (email)"
        )
