"""Per-container database logins, against the real test database."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from uuid import UUID

import asyncpg  # type: ignore[import-untyped]
import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from openscientist.database.models import Job
from openscientist.job_container.db_login import create_agent_login, drop_agent_login


async def _connect(database_url: str) -> asyncpg.Connection:
    url = make_url(database_url)
    return await asyncpg.connect(
        host=url.host,
        port=url.port,
        database=url.database,
        user=url.username,
        password=url.password,
    )


@pytest_asyncio.fixture
async def job_id(test_engine: AsyncEngine) -> AsyncGenerator[UUID, None]:
    async with AsyncSession(test_engine, expire_on_commit=False) as session:
        job = Job(owner_id=None, research_question="Login lifecycle")
        session.add(job)
        await session.commit()
    yield job.id
    async with test_engine.begin() as conn:
        await conn.execute(text("DELETE FROM jobs WHERE id = :id"), {"id": job.id})


@pytest.mark.asyncio
async def test_a_login_reaches_its_job_until_it_is_dropped(
    job_id: UUID, test_engine: AsyncEngine
) -> None:
    login = await create_agent_login(str(job_id), lifetime_seconds=300)
    conn = await _connect(login.database_url)
    try:
        assert {row["id"] for row in await conn.fetch("SELECT id FROM jobs")} == {job_id}
    finally:
        await conn.close()
    async with test_engine.connect() as engine_conn:
        valid_until = (
            await engine_conn.execute(
                text("SELECT rolvaliduntil FROM pg_roles WHERE rolname = :role"),
                {"role": login.role},
            )
        ).scalar_one()
    # Valid for the container's lifetime plus a short grace, and no longer.
    assert datetime.now(UTC) + timedelta(minutes=5) < valid_until
    assert valid_until < datetime.now(UTC) + timedelta(minutes=20)

    await drop_agent_login(login.role)

    with pytest.raises(asyncpg.exceptions.InvalidAuthorizationSpecificationError):
        await _connect(login.database_url)
    async with test_engine.connect() as engine_conn:
        mapped = (
            await engine_conn.execute(
                text("SELECT count(*) FROM agent_job_roles WHERE role_name = :role"),
                {"role": login.role},
            )
        ).scalar_one()
    assert mapped == 0


@pytest.mark.asyncio
async def test_dropping_a_login_ends_its_open_sessions(job_id: UUID) -> None:
    login = await create_agent_login(str(job_id), lifetime_seconds=300)
    conn = await _connect(login.database_url)
    try:
        await drop_agent_login(login.role)
        with pytest.raises(
            (
                asyncpg.exceptions.AdminShutdownError,
                asyncpg.exceptions.ConnectionDoesNotExistError,
                asyncpg.exceptions.InterfaceError,
            )
        ):
            await conn.fetch("SELECT id FROM jobs")
    finally:
        if not conn.is_closed():
            conn.terminate()


@pytest.mark.asyncio
async def test_only_agent_logins_can_be_dropped() -> None:
    with pytest.raises(ValueError, match="not an agent login"):
        await drop_agent_login("openscientist_admin")
