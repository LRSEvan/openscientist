"""
Row-Level Security (RLS) utilities for OpenScientist.

Provides middleware and utilities for PostgreSQL Row-Level Security enforcement.
RLS policies control access to database rows based on the current user context.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

from sqlalchemy import event, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session, SessionTransaction

#: The role user sessions run as. The policies on user data are written for it.
APP_ROLE = "openscientist_app"
_ROLE_KEY = "openscientist.rls_app_role"
_USER_KEY = "openscientist.rls_user_id"


@event.listens_for(Session, "after_begin")
def _apply_rls_to_transaction(
    session: Session, _transaction: SessionTransaction, connection: Connection
) -> None:
    """Give every transaction the session's role and user. Both are transaction-local,
    so a commit cannot shed them and a pooled connection never keeps them."""
    if session.info.get(_ROLE_KEY):
        connection.exec_driver_sql(f"SET LOCAL ROLE {APP_ROLE}")
    if _USER_KEY in session.info:
        connection.execute(
            text("SELECT set_config('app.current_user_id', :user_id, true)"),
            {"user_id": session.info[_USER_KEY]},
        )


async def use_app_role(session: AsyncSession) -> None:
    """Run every transaction of ``session`` as ``openscientist_app``, the open one included."""
    session.info[_ROLE_KEY] = True
    if session.in_transaction():
        await session.execute(text(f"SET LOCAL ROLE {APP_ROLE}"))


async def set_current_user(session: AsyncSession, user_id: UUID | None) -> None:
    """
    Set the user that RLS policies see as ``app.current_user_id``.

    The user applies to every transaction of the session, the open one included,
    and to no other session that later gets the same pooled connection.

    Args:
        session: SQLAlchemy async session
        user_id: UUID of the current user, or None to clear

    Example:
        >>> async with get_session_ctx() as session:
        ...     await set_current_user(session, user.id)
        ...     # All queries in this session now respect RLS for this user
        ...     jobs = await session.execute(select(Job))
    """
    value = None if user_id is None else str(user_id)
    session.info[_USER_KEY] = value
    if session.in_transaction():
        await session.execute(
            text("SELECT set_config('app.current_user_id', :user_id, true)"),
            {"user_id": value},
        )


async def get_current_user(session: AsyncSession) -> UUID | None:
    """
    Get the current user ID from the PostgreSQL session.

    Args:
        session: SQLAlchemy async session

    Returns:
        UUID of the current user, or None if not set

    Example:
        >>> async with get_session_ctx() as session:
        ...     user_id = await get_current_user(session)
        ...     print(f"Current user: {user_id}")
    """
    result = await session.execute(text("SELECT current_setting('app.current_user_id', true)"))
    user_id_str = result.scalar_one()

    if user_id_str is None or user_id_str == "":
        return None

    return UUID(user_id_str)


@asynccontextmanager
async def session_with_user(
    session: AsyncSession,
    user_id: UUID | None,
) -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager that sets user context for RLS and cleans up automatically.

    This is a convenience wrapper that combines session management with user
    context setting. It ensures the user ID is properly set before queries
    and cleared afterward.

    Args:
        session: SQLAlchemy async session
        user_id: UUID of the current user, or None for no RLS context

    Yields:
        The same session with user context applied

    Example:
        >>> async with get_session_ctx() as session:
        ...     async with session_with_user(session, user.id) as user_session:
        ...         jobs = await user_session.execute(select(Job))
        ...         # Only sees jobs visible to user
    """
    # Store the original user ID
    original_user_id = await get_current_user(session)

    try:
        # Set the new user context
        await set_current_user(session, user_id)
        yield session
    finally:
        # Restore the original user context
        await set_current_user(session, original_user_id)


async def verify_rls_enabled(session: AsyncSession, table_name: str) -> bool:
    """
    Check if RLS is enabled on a specific table.

    Useful for debugging and verification during development and testing.

    Args:
        session: SQLAlchemy async session
        table_name: Name of the table to check

    Returns:
        True if RLS is enabled on the table, False otherwise

    Example:
        >>> async with get_session_ctx() as session:
        ...     is_protected = await verify_rls_enabled(session, "jobs")
        ...     print(f"Jobs table RLS: {is_protected}")
    """
    result = await session.execute(
        text(
            """
            SELECT relrowsecurity
            FROM pg_class
            WHERE relname = :table_name
        """
        ),
        {"table_name": table_name},
    )
    row = result.first()
    return row is not None and row[0] is True


async def list_rls_policies(session: AsyncSession, table_name: str) -> list[dict[str, Any]]:
    """
    List all RLS policies defined on a table.

    Returns detailed information about each policy including name, command,
    and definition. Useful for debugging and documentation.

    Args:
        session: SQLAlchemy async session
        table_name: Name of the table to check

    Returns:
        List of dictionaries containing policy information

    Example:
        >>> async with get_session_ctx() as session:
        ...     policies = await list_rls_policies(session, "jobs")
        ...     for policy in policies:
        ...         print(f"{policy['name']}: {policy['cmd']}")
    """
    result = await session.execute(
        text(
            """
            SELECT
                polname as name,
                polcmd as cmd,
                pg_get_expr(polqual, polrelid) as using_expr,
                pg_get_expr(polwithcheck, polrelid) as check_expr
            FROM pg_policy
            JOIN pg_class ON pg_class.oid = pg_policy.polrelid
            WHERE pg_class.relname = :table_name
            ORDER BY polname
        """
        ),
        {"table_name": table_name},
    )

    return [
        {
            "name": row[0],
            "cmd": row[1],  # '*' (ALL), 'r' (SELECT), 'w' (UPDATE), 'a' (INSERT), 'd' (DELETE)
            "using": row[2],
            "check": row[3],
        }
        for row in result
    ]
