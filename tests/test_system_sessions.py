"""The web's system paths (worker threads, notifications) act without a user and
must not depend on the web login's own rights: they run through the admin login.

Each test points DATABASE_URL at a login with no privileges at all, as the web
login will be once it stops being a superuser, so a path that still used it
fails here.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

import openscientist.database.engine as engine_module
import openscientist.database.session as session_module
from openscientist.database.models import Job, User
from openscientist.job.types import JobStatus
from openscientist.job_manager import (
    _db_delete_job,
    _db_get_job,
    _db_get_notification_target,
    _db_list_jobs,
    _db_update_job_status,
)
from openscientist.ntfy import ensure_user_has_topic, get_user_ntfy_settings
from openscientist.settings import get_settings

_PASSWORD = "system-sessions-test-password"


def _reset_database_state() -> None:
    get_settings.cache_clear()
    engine_module._engine = None
    engine_module._admin_engine = None
    session_module._session_factory = None
    session_module._admin_session_factory = None


@pytest_asyncio.fixture
async def web_login_without_rights(
    test_engine: AsyncEngine, test_database_url: str
) -> AsyncGenerator[None, None]:
    web_role = f"web_{uuid4().hex}"
    async with test_engine.begin() as conn:
        await conn.execute(text(f"CREATE ROLE {web_role} LOGIN PASSWORD '{_PASSWORD}'"))
    saved = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = (
        make_url(test_database_url)
        .set(username=web_role, password=_PASSWORD)
        .render_as_string(hide_password=False)
    )
    _reset_database_state()
    yield
    if saved is None:
        os.environ.pop("DATABASE_URL", None)
    else:
        os.environ["DATABASE_URL"] = saved
    _reset_database_state()
    async with test_engine.begin() as conn:
        await conn.execute(text(f"DROP ROLE {web_role}"))


@dataclass(frozen=True)
class _OwnedJob:
    job_id: str
    owner_id: UUID


@pytest_asyncio.fixture
async def owned_job(test_engine: AsyncEngine) -> AsyncGenerator[_OwnedJob, None]:
    async with AsyncSession(test_engine, expire_on_commit=False) as session:
        owner = User(email=f"owner-{uuid4().hex}@system.test", name="Owner")
        session.add(owner)
        await session.flush()
        job = Job(owner_id=owner.id, research_question="System paths", status="queued")
        session.add(job)
        await session.commit()
    yield _OwnedJob(job_id=str(job.id), owner_id=owner.id)
    async with test_engine.begin() as conn:
        await conn.execute(text("DELETE FROM jobs WHERE id = :id"), {"id": job.id})
        await conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": owner.id})


async def _status(test_engine: AsyncEngine, job_id: str) -> str | None:
    async with test_engine.connect() as conn:
        result = await conn.execute(text("SELECT status FROM jobs WHERE id = :id"), {"id": job_id})
        value = result.scalar_one_or_none()
        return None if value is None else str(value)


@pytest.mark.asyncio
async def test_worker_paths_reach_a_users_job_without_the_web_login(
    web_login_without_rights: None, owned_job: _OwnedJob, test_engine: AsyncEngine
) -> None:
    queued = [job.id for job in await _db_list_jobs(status=JobStatus.QUEUED)]
    assert UUID(owned_job.job_id) in queued

    target = await _db_update_job_status(owned_job.job_id, JobStatus.RUNNING)
    assert target.owner_id == str(owned_job.owner_id)
    assert await _status(test_engine, owned_job.job_id) == "running"

    job = await _db_get_job(owned_job.job_id)
    assert job is not None and job.status == "running"
    assert (await _db_get_notification_target(owned_job.job_id)).owner_id == str(owned_job.owner_id)


@pytest.mark.asyncio
async def test_a_job_is_deleted_without_a_user_or_the_web_login(
    web_login_without_rights: None, owned_job: _OwnedJob, test_engine: AsyncEngine
) -> None:
    await _db_delete_job(owned_job.job_id)
    assert await _status(test_engine, owned_job.job_id) is None


@pytest.mark.asyncio
async def test_notification_lookups_work_without_the_web_login(
    web_login_without_rights: None, owned_job: _OwnedJob, test_engine: AsyncEngine
) -> None:
    assert await get_user_ntfy_settings(owned_job.owner_id) == (True, None)
    topic = await ensure_user_has_topic(owned_job.owner_id)
    assert topic is not None
    async with test_engine.connect() as conn:
        stored = await conn.execute(
            text("SELECT ntfy_topic FROM users WHERE id = :id"), {"id": owned_job.owner_id}
        )
        assert stored.scalar_one() == topic
