"""Sign-in checks and logout under a real database outage
(plans/auth-db-errors.md)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from openscientist.auth import middleware
from openscientist.auth.fastapi_routes import router
from openscientist.database.models import Session, User
from tests.db_outage import Forwarder


@pytest_asyncio.fixture
async def session_token(test_engine: AsyncEngine) -> AsyncGenerator[UUID, None]:
    """A signed-in visitor's valid session."""
    async with AsyncSession(test_engine, expire_on_commit=False) as db:
        user = User(email=f"auth-{uuid4().hex}@outage.test", name="Visitor", is_approved=True)
        db.add(user)
        await db.flush()
        login = Session(user_id=user.id, expires_at=datetime.now(UTC) + timedelta(days=1))
        db.add(login)
        await db.commit()
    yield login.id
    async with AsyncSession(test_engine) as db:
        await db.execute(delete(User).where(User.id == user.id))
        await db.commit()


@pytest.fixture
def visitor(session_token: UUID, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """The NiceGUI context of a visitor carrying ``session_token``. Returns
    their stored session, which a sign-out would clear."""
    storage: dict[str, object] = {"session_token": str(session_token), "user_id": "kept"}
    ui = MagicMock()
    ui.context.client.request.cookies = {"session_token": str(session_token)}
    ui.context.client.page.path = "/jobs"
    monkeypatch.setattr(middleware, "app", SimpleNamespace(storage=SimpleNamespace(user=storage)))
    monkeypatch.setattr(middleware, "ui", ui)
    return storage


async def test_a_page_during_an_outage_keeps_the_visitor_signed_in(
    database: Forwarder, visitor: dict[str, object]
) -> None:
    ran = False

    async def page() -> None:
        nonlocal ran
        ran = True

    stored = dict(visitor)
    database.down()
    await middleware.require_auth(page)()
    assert not ran
    assert visitor == stored
    middleware.ui.navigate.to.assert_not_called()  # type: ignore[attr-defined]


def test_a_sync_page_during_an_outage_keeps_the_visitor_signed_in(
    database: Forwarder, visitor: dict[str, object]
) -> None:
    ran = False

    def page() -> None:
        nonlocal ran
        ran = True

    stored = dict(visitor)
    database.down()
    middleware.require_auth(page)()
    assert not ran
    assert visitor == stored
    middleware.ui.navigate.to.assert_not_called()  # type: ignore[attr-defined]


async def _session_exists(test_engine: AsyncEngine, token: UUID) -> bool:
    async with test_engine.connect() as conn:
        return await conn.scalar(select(Session.id).where(Session.id == token)) is not None


async def test_logout_during_an_outage_signs_out_now_and_revokes_later(
    database: Forwarder,
    test_engine: AsyncEngine,
    session_token: UUID,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pages fall back to the token NiceGUI keeps per browser, so signing out
    clears it too, or the next page would still find the visitor signed in."""
    from openscientist.auth import fastapi_routes

    stored: dict[str, object] = {"session_token": str(session_token), "authenticated": True}
    monkeypatch.setattr(
        fastapi_routes, "app", SimpleNamespace(storage=SimpleNamespace(user=stored))
    )
    app = FastAPI()
    app.include_router(router)
    database.down()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        cookies={"session_token": str(session_token)},
    ) as client:
        response = await client.get("/auth/logout")
    assert response.headers["location"] == "/login"
    assert 'session_token=""' in response.headers["set-cookie"]
    assert await _session_exists(test_engine, session_token)
    assert stored == {}
    database.up()
    for _ in range(100):
        if not await _session_exists(test_engine, session_token):
            break
        await asyncio.sleep(0.2)
    assert not await _session_exists(test_engine, session_token)
