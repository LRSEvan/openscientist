"""The migration that gives ``openscientist_admin`` its password from
``POSTGRES_ADMIN_PASSWORD``, so no deployment keeps a password fixed in the
repository."""

from __future__ import annotations

import importlib.util
from collections.abc import AsyncGenerator
from pathlib import Path
from types import ModuleType

import asyncpg  # type: ignore[import-untyped]
import pytest
import pytest_asyncio
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import Connection, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine

from tests.db_logins import ADMIN_PASSWORD

_MIGRATION = (
    Path(__file__).parent.parent
    / "src/openscientist/database/migrations/versions"
    / "2026-10-24_1200_admin_password_from_env.py"
)

# Quotes, a backslash, a percent sign, a colon and a dollar sign: each one
# breaks a different way of splicing the value into ALTER ROLE.
_AWKWARD_PASSWORD = """it's 50% "admin":$1\\x"""


def _migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location("admin_password_from_env", _MIGRATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def _upgrade(engine: AsyncEngine) -> None:
    module = _migration()

    def run(connection: Connection) -> None:
        with Operations.context(MigrationContext.configure(connection)):
            module.upgrade()

    async with engine.begin() as conn:
        await conn.run_sync(run)


async def _login_as_admin(database_url: str, password: str) -> str:
    url = make_url(database_url)
    conn = await asyncpg.connect(
        host=url.host,
        port=url.port,
        database=url.database,
        user="openscientist_admin",
        password=password,
    )
    try:
        user: str = await conn.fetchval("SELECT current_user")
        return user
    finally:
        await conn.close()


@pytest_asyncio.fixture
async def restores_admin_password(test_engine: AsyncEngine) -> AsyncGenerator[None, None]:
    """The test changes the admin password. Put back the one conftest bootstrapped."""
    yield
    async with test_engine.begin() as conn:
        await conn.execute(text(f"ALTER ROLE openscientist_admin PASSWORD '{ADMIN_PASSWORD}'"))


@pytest.mark.asyncio
async def test_admin_logs_in_with_the_configured_password(
    test_engine: AsyncEngine,
    test_database_url: str,
    restores_admin_password: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", test_database_url)
    monkeypatch.setenv("POSTGRES_ADMIN_PASSWORD", _AWKWARD_PASSWORD)

    await _upgrade(test_engine)

    assert await _login_as_admin(test_database_url, _AWKWARD_PASSWORD) == "openscientist_admin"
    with pytest.raises(asyncpg.exceptions.InvalidPasswordError):
        await _login_as_admin(test_database_url, "openscientist_dev_password")


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, ""])
async def test_refuses_to_run_without_a_password(
    test_engine: AsyncEngine,
    test_database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    value: str | None,
) -> None:
    """Running on would leave the role with whatever password it had before."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", test_database_url)
    if value is None:
        monkeypatch.delenv("POSTGRES_ADMIN_PASSWORD", raising=False)
    else:
        monkeypatch.setenv("POSTGRES_ADMIN_PASSWORD", value)

    with pytest.raises(RuntimeError, match="POSTGRES_ADMIN_PASSWORD"):
        await _upgrade(test_engine)
