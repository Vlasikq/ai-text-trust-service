"""
Unit of Work — lightweight transaction boundary.

Usage:
    async with UnitOfWork() as uow:
        await uow.analyses.save(analysis)
        await uow.feedback.save(fb)
        await uow.commit()
    # auto-rollback on exception, auto-close always
"""

from app.database import engine as engine_mod
from app.database.repositories import (
    AnalysisRepository,
    BatchJobRepository,
    DeploymentRepository,
    FeedbackRepository,
    JobRepository,
    RefreshTokenRepository,
    UserRepository,
)


class UnitOfWork:
    async def __aenter__(self):
        if engine_mod.async_session is None:
            raise RuntimeError("Database not initialized")
        self.session = engine_mod.async_session()
        self.analyses = AnalysisRepository(self.session)
        self.feedback = FeedbackRepository(self.session)
        self.deployments = DeploymentRepository(self.session)
        self.jobs = JobRepository(self.session)
        self.batches = BatchJobRepository(self.session)
        self.users = UserRepository(self.session)
        self.refresh_tokens = RefreshTokenRepository(self.session)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            await self.session.rollback()
        await self.session.close()

    async def commit(self):
        await self.session.commit()
