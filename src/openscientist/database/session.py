"""
Database session management for OpenScientist.

Provides async sessions and dependency injection for FastAPI/NiceGUI. Every
session binds the pooled engine of the loop it is opened on, so it works in
any thread with a running loop.

Session Types:
    - get_session(): Standard session with RLS enforced as openscientist_app
    - get_admin_session(): Admin session that bypasses RLS via elevated role
    - async_session_local(): Low-level session, with no role or user of its own

get_session() connects as the main database user, and every transaction it runs
switches to openscientist_app, a non-superuser NOLOGIN role subject to RLS.
"""

import logging
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from .engine import get_admin_engine, get_engine
from .rls import set_current_user, use_app_role

logger = logging.getLogger(__name__)


def _session(engine: AsyncEngine) -> AsyncSession:
    return AsyncSession(engine, expire_on_commit=False, autoflush=False)


def thread_safe_admin_session() -> AsyncSession:
    """An admin session for system work in the web process that acts for no user.

    Connects with ``ADMIN_DATABASE_URL`` (``openscientist_admin``, BYPASSRLS).
    """
    return _session(get_admin_engine())


@asynccontextmanager
async def thread_safe_user_session(user_id: UUID) -> AsyncIterator[AsyncSession]:
    """A session whose every transaction runs as ``user_id`` under RLS."""
    async with _session(get_engine()) as session:
        await use_app_role(session)
        await set_current_user(session, user_id)
        yield session


#: Opens the caller's session inside the coroutine that uses it.
SessionOpener = Callable[[], AbstractAsyncContextManager[AsyncSession]]


def async_session_local() -> AsyncSession:
    """Create a new async session.

    Returns:
        A new AsyncSession instance.

    Note:
        The returned session runs as the login role. Callers that act for a user
        call use_app_role() and set_current_user(), or use get_session() instead.
    """
    return _session(get_engine())


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to provide a database session with RLS enforced.

    Connects as the main database user, and runs every transaction as
    openscientist_app, which is subject to RLS policies, so queries only return
    rows the current user is authorized to see.

    Callers must still call set_current_user(session, user_id) to set the
    RLS context for the specific user.

    Usage in FastAPI (as dependency):
        @app.get("/items")
        async def get_items(session: AsyncSession = Depends(get_session)):
            ...

    Usage in NiceGUI or standalone (as context manager):
        async with get_session_ctx() as session:
            await set_current_user(session, user.id)
            ...

    Yields:
        AsyncSession: A SQLAlchemy async session with RLS enforced.
    """
    async with _session(get_engine()) as session:
        try:
            await use_app_role(session)
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Context manager wrapper for get_session (for NiceGUI/standalone use).
# FastAPI Depends uses get_session directly (plain async generator).
get_session_ctx = asynccontextmanager(get_session)


@asynccontextmanager
async def get_admin_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to provide a database session with admin privileges.

    Sessions from this factory bypass RLS policies by connecting with
    the openscientist_admin PostgreSQL role which has BYPASSRLS privilege.

    The role is created by docker/postgres/bootstrap.sql, which conftest also runs.

    Use this only for:
    - Background schedulers (no user context available)
    - Admin-authenticated endpoints (verified by @require_admin)
    - Migrations and schema changes
    - Test fixtures that need to create data across tenants

    WARNING: This grants full database access. Never expose to user-facing APIs.

    Usage:
        async with get_admin_session() as session:
            # RLS bypassed via BYPASSRLS role privilege
            all_jobs = await session.execute(select(Job))
            ...

    Yields:
        AsyncSession: A SQLAlchemy async session with admin privileges.
    """
    async with _session(get_admin_engine()) as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
