"""Database logins for agent containers.

Every agent container gets its own login, a member of openscientist_agent mapped
to its job in agent_job_roles, so the policies hold it to that job's rows. The
login is created right before the container starts and dropped when the
container is removed. It expires on its own a little after the longest the
container can run, so a login that teardown misses stops working anyway.
"""

from __future__ import annotations

import logging
import re
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.engine import make_url

from openscientist.async_tasks import run_sync
from openscientist.database.session import AsyncSessionLocal
from openscientist.settings import get_settings

logger = logging.getLogger(__name__)

#: Container label that carries the container's login name.
AGENT_LOGIN_LABEL = "openscientist.db_login"

_ROLE_PATTERN = re.compile(r"agent_[0-9a-f]{32}")
# Room for teardown and clock skew past the container's own deadline.
_GRACE = timedelta(minutes=10)


@dataclass(frozen=True)
class AgentLogin:
    role: str
    database_url: str


async def create_agent_login(job_id: str, *, lifetime_seconds: int) -> AgentLogin:
    """Create a login for one container of ``job_id`` and map it to the job."""
    role = f"agent_{secrets.token_hex(16)}"
    password = secrets.token_urlsafe(32)
    valid_until = datetime.now(UTC) + timedelta(seconds=lifetime_seconds) + _GRACE
    async with AsyncSessionLocal(thread_safe=True) as session:
        # CREATE ROLE takes no bind parameters, so the server quotes the values.
        statement = (
            await session.execute(
                text(
                    "SELECT format('CREATE ROLE %I LOGIN PASSWORD %L VALID UNTIL %L "
                    "IN ROLE openscientist_agent', CAST(:role AS text), "
                    "CAST(:password AS text), CAST(:valid_until AS text))"
                ),
                {"role": role, "password": password, "valid_until": valid_until.isoformat()},
            )
        ).scalar_one()
        await (await session.connection()).exec_driver_sql(statement)
        await session.execute(
            text("INSERT INTO agent_job_roles (role_name, job_id) VALUES (:role, :job)"),
            {"role": role, "job": UUID(job_id)},
        )
        await session.commit()
    url = make_url(get_settings().database.effective_database_url)
    url = url.set(username=role, password=password)
    return AgentLogin(role=role, database_url=url.render_as_string(hide_password=False))


async def drop_agent_login(role: str) -> None:
    """Unmap the login, end its sessions and drop it.

    The mapping goes first, so the login reaches nothing even if a later step
    fails.
    """
    if not _ROLE_PATTERN.fullmatch(role):
        raise ValueError(f"{role!r} is not an agent login")
    async with AsyncSessionLocal(thread_safe=True) as session:
        await session.execute(
            text("DELETE FROM agent_job_roles WHERE role_name = :role"), {"role": role}
        )
        await session.commit()
        await session.execute(
            text("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE usename = :role"),
            {"role": role},
        )
        statement = (
            await session.execute(
                text("SELECT format('DROP ROLE IF EXISTS %I', CAST(:role AS text))"),
                {"role": role},
            )
        ).scalar_one()
        await (await session.connection()).exec_driver_sql(statement)
        await session.commit()


def open_agent_login(job_id: str, *, lifetime_seconds: int) -> AgentLogin:
    """Synchronous :func:`create_agent_login`, for the container runner."""
    try:
        return run_sync(create_agent_login(job_id, lifetime_seconds=lifetime_seconds))
    except Exception as error:
        raise RuntimeError(
            f"Could not create the database login for an agent container of job {job_id}: {error}"
        ) from error


def close_agent_login(role: str) -> None:
    """Synchronous :func:`drop_agent_login` that only logs a failure, because it
    runs during teardown. The login still expires on its own."""
    try:
        run_sync(drop_agent_login(role))
    except Exception as error:
        logger.error("Could not drop agent database login %s: %s", role, error)
