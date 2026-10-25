"""Shared code the web runs for a user reaches only that user's jobs."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

import openscientist.database.engine as engine_module
import openscientist.database.session as session_module
from openscientist.database import get_session_ctx
from openscientist.database.models import Job, User
from openscientist.database.rls import set_current_user
from openscientist.database.session import thread_safe_user_session
from openscientist.job_chat import load_job_context
from openscientist.knowledge_state import KnowledgeState
from openscientist.orchestrator.iteration import update_job_status


@dataclass(frozen=True)
class _Fixture:
    job_id: str
    owner: UUID
    stranger: UUID


@pytest_asyncio.fixture
async def alices_job(test_engine: AsyncEngine) -> AsyncGenerator[_Fixture, None]:
    """Alice's job with one finding, and Bob, who has no share in it."""
    engine_module._engine = None
    session_module._session_factory = None
    async with AsyncSession(test_engine, expire_on_commit=False) as session:
        alice = User(email=f"alice-{uuid4().hex}@scoped.test", name="Alice")
        bob = User(email=f"bob-{uuid4().hex}@scoped.test", name="Bob")
        session.add_all([alice, bob])
        await session.flush()
        job = Job(
            owner_id=alice.id, research_question="Alice's question", status="awaiting_feedback"
        )
        session.add(job)
        await session.commit()
    ks = KnowledgeState(str(job.id), "Alice's question", 3)
    ks.add_finding("Alice's finding", "evidence")
    await ks.save_to_database(str(job.id))
    yield _Fixture(job_id=str(job.id), owner=alice.id, stranger=bob.id)
    async with test_engine.begin() as conn:
        await conn.execute(text("DELETE FROM jobs WHERE id = :id"), {"id": job.id})
        await conn.execute(
            text("DELETE FROM users WHERE id = ANY(:ids)"), {"ids": [alice.id, bob.id]}
        )
    if engine_module._engine is not None:
        await engine_module._engine.dispose()
    engine_module._engine = None
    session_module._session_factory = None


async def _count(test_engine: AsyncEngine, sql: str, job_id: str) -> int:
    async with test_engine.connect() as conn:
        return int((await conn.execute(text(sql), {"id": job_id})).scalar_one())


async def _status(test_engine: AsyncEngine, job_id: str) -> str:
    async with test_engine.connect() as conn:
        result = await conn.execute(text("SELECT status FROM jobs WHERE id = :id"), {"id": job_id})
        return str(result.scalar_one())


def test_a_user_loads_the_knowledge_of_their_own_job_only(alices_job: _Fixture) -> None:
    mine = KnowledgeState.load_from_database_sync(
        alices_job.job_id, open_session=partial(thread_safe_user_session, alices_job.owner)
    )
    assert [f["title"] for f in mine.data["findings"]] == ["Alice's finding"]
    with pytest.raises(ValueError, match="not found"):
        KnowledgeState.load_from_database_sync(
            alices_job.job_id,
            open_session=partial(thread_safe_user_session, alices_job.stranger),
        )


@pytest.mark.asyncio
async def test_a_user_cannot_save_knowledge_into_someone_elses_job(
    alices_job: _Fixture, test_engine: AsyncEngine
) -> None:
    ks = await KnowledgeState.load_from_database(alices_job.job_id)
    ks.add_finding("Planted finding", "evidence")
    with pytest.raises(DBAPIError):
        ks.save_to_database_sync(
            alices_job.job_id,
            open_session=partial(thread_safe_user_session, alices_job.stranger),
        )
    planted = "SELECT count(*) FROM findings WHERE job_id = :id AND text = 'Planted finding'"
    assert await _count(test_engine, planted, alices_job.job_id) == 0


@pytest.mark.asyncio
async def test_continuing_after_feedback_changes_only_the_users_own_job(
    alices_job: _Fixture, test_engine: AsyncEngine, tmp_path: Path
) -> None:
    job_dir = tmp_path / alices_job.job_id
    await update_job_status(
        job_dir, "running", open_session=partial(thread_safe_user_session, alices_job.stranger)
    )
    assert await _status(test_engine, alices_job.job_id) == "awaiting_feedback"
    await update_job_status(
        job_dir, "running", open_session=partial(thread_safe_user_session, alices_job.owner)
    )
    assert await _status(test_engine, alices_job.job_id) == "running"


@pytest.mark.asyncio
async def test_chat_context_comes_only_from_the_chatting_users_jobs(
    alices_job: _Fixture,
) -> None:
    async with get_session_ctx() as session:
        await set_current_user(session, alices_job.owner)
        own = await load_job_context(alices_job.job_id, session)
    async with get_session_ctx() as session:
        await set_current_user(session, alices_job.stranger)
        foreign = await load_job_context(alices_job.job_id, session)
    assert "Alice's question" in own
    assert "Alice's finding" in own
    assert "Alice" not in foreign
