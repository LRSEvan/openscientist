"""
Database package for OpenScientist.

Provides SQLAlchemy ORM models, session management, and database utilities.

Dual-Engine Pattern:
    - get_session(): Standard session with RLS enforced by PostgreSQL role
    - get_admin_session(): Admin session that bypasses RLS via BYPASSRLS role

ADMIN_DATABASE_URL must name a PostgreSQL role that has the BYPASSRLS privilege
(created by docker/postgres/bootstrap.sql). Admin sessions never fall back to
DATABASE_URL.
"""

from .base import Base, UUIDv7Mixin
from .engine import get_admin_engine, get_engine
from .rls import list_rls_policies, set_current_user, verify_rls_enabled
from .session import AsyncSessionLocal, get_admin_session, get_session, get_session_ctx

__all__ = [
    "AsyncSessionLocal",
    "Base",
    "UUIDv7Mixin",
    "get_admin_engine",
    "get_admin_session",
    "get_engine",
    "get_session",
    "get_session_ctx",
    "list_rls_policies",
    "set_current_user",
    "verify_rls_enabled",
]
