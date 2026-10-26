"""
Database engine configuration for OpenScientist.

An asyncpg connection belongs to the event loop that opened it, so each loop
gets its own pooled engines, one per database login, created on first use.
When the loop shuts down (``asyncio.run``, ``asyncio.Runner``, uvicorn and
pytest-asyncio all shut down its async generators before closing it), its
engines are disposed and its pooled connections closed.

Engine Types:
    - get_engine(): the login of ``DATABASE_URL``, subject to RLS
    - get_admin_engine(): the login of ``ADMIN_DATABASE_URL``, which bypasses RLS

ADMIN_DATABASE_URL must connect with a PostgreSQL role that has elevated
privileges (table owner or role with BYPASSRLS). The admin engine refuses to
start without it.
"""

import asyncio
import threading
import weakref
from collections.abc import AsyncGenerator

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from openscientist.database.retry import guard_connect
from openscientist.settings import get_settings

# Load environment variables from .env file
load_dotenv()


class _LoopEngines:
    """One loop's engines by URL, and the async generator that disposes them
    when the loop shuts its async generators down."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.by_url: dict[str, AsyncEngine] = {}
        self.keeper = _dispose_at_shutdown(loop, self.by_url)
        # The keeper awaits nothing before its yield, so one step runs it there
        # without the loop, and that first step registers it with this loop.
        try:
            self.keeper.asend(None).send(None)
        except StopIteration:
            return
        raise AssertionError("the engine keeper awaited before reaching its yield")


_loop_engines: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _LoopEngines] = (
    weakref.WeakKeyDictionary()
)
_loop_engines_lock = threading.Lock()


async def _dispose_at_shutdown(
    loop: asyncio.AbstractEventLoop, by_url: dict[str, AsyncEngine]
) -> AsyncGenerator[None, None]:
    try:
        yield
    finally:
        with _loop_engines_lock:
            _loop_engines.pop(loop, None)
        for engine in by_url.values():
            await engine.dispose()


def _engine_for(database_url: str) -> AsyncEngine:
    """The running loop's pooled engine for ``database_url``."""
    loop = asyncio.get_running_loop()
    with _loop_engines_lock:
        engines = _loop_engines.get(loop)
        if engines is None:
            engines = _loop_engines[loop] = _LoopEngines(loop)
        engine = engines.by_url.get(database_url)
        if engine is None:
            settings = get_settings()
            engine = engines.by_url[database_url] = guard_connect(
                create_async_engine(
                    database_url,
                    pool_size=20,
                    max_overflow=10,
                    pool_timeout=30,
                    pool_recycle=3600,
                    pool_pre_ping=True,
                    echo=settings.database.sql_echo,
                ),
                retry_for=settings.database.connect_retry_seconds,
            )
    return engine


def get_engine() -> AsyncEngine:
    """
    The running loop's engine for standard operations.

    It connects with ``DATABASE_URL``, whose login works under RLS policies.

    Returns:
        AsyncEngine: The SQLAlchemy async engine instance.

    Raises:
        RuntimeError: No event loop runs in this thread.
    """
    return _engine_for(get_settings().database.effective_database_url)


def get_admin_engine() -> AsyncEngine:
    """
    The running loop's engine for privileged operations.

    It connects with ``ADMIN_DATABASE_URL``, whose role bypasses RLS policies
    (table owner or BYPASSRLS grant).

    Use this engine only for:
    - Background schedulers (no user context available)
    - Admin-authenticated endpoints (verified by @require_admin)
    - Migrations and schema changes
    - Test fixtures that need to create data across tenants

    Returns:
        AsyncEngine: The admin SQLAlchemy async engine instance.

    Raises:
        ValueError: ``ADMIN_DATABASE_URL`` is not configured.
        RuntimeError: No event loop runs in this thread.
    """
    return _engine_for(get_settings().database.required_admin_database_url())
