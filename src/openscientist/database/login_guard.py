"""Startup checks that the web runs with its own narrow database login."""

from __future__ import annotations

from collections.abc import Mapping

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

#: Passwords that must never reach the web, and the file each belongs in.
DATABASE_SECRETS = {"POSTGRES_PASSWORD": ".env.db", "POSTGRES_OWNER_PASSWORD": ".env.owner"}


def leaked_database_secrets(environ: Mapping[str, str]) -> list[str]:
    """The superuser and owner password variables set in ``environ``."""
    return [name for name in DATABASE_SECRETS if environ.get(name)]


async def database_login_problem(database_url: str) -> str | None:
    """Why ``database_url``'s login is too powerful for the web, or None."""
    engine = create_async_engine(database_url, poolclass=NullPool)
    try:
        async with engine.connect() as conn:
            row = (
                await conn.execute(
                    text(
                        "SELECT rolname, rolsuper, rolbypassrls FROM pg_catalog.pg_roles "
                        "WHERE rolname = current_user"
                    )
                )
            ).one()
    finally:
        await engine.dispose()
    if row.rolsuper:
        return f"DATABASE_URL logs in as {row.rolname}, a superuser. Use openscientist_web."
    if row.rolbypassrls:
        return f"DATABASE_URL logs in as {row.rolname}, which has BYPASSRLS. Use openscientist_web."
    return None
