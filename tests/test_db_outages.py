"""Job-state paths under a real database outage (plans/swallowed-db-errors.md).

The app reaches the test database through a TCP forwarder that a test closes,
so connects are refused, and reopens to end the outage."""

from __future__ import annotations

import asyncio
import os
import socket
import threading
import time
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

import openscientist.database.engine as engine_module
import openscientist.database.session as session_module
from openscientist.database.models import Job, User
from openscientist.database.retry import DatabaseUnavailableError
from openscientist.job_manager import JobManager, JobStatus
from openscientist.orchestrator.discovery import _assert_job_running
from openscientist.orchestrator.iteration import update_job_status
from openscientist.settings import get_settings


class _Forwarder:
    """Forwards a local port to the database until :meth:`down`, and again after :meth:`up`."""

    def __init__(self, target: tuple[str, int]) -> None:
        self._target = target
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        self.port = probe.getsockname()[1]
        probe.close()
        self._listener: socket.socket | None = None
        self._open: set[socket.socket] = set()
        self._lock = threading.Lock()
        self.up()

    def up(self) -> None:
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", self.port))
        listener.listen()
        listener.settimeout(0.1)
        self._listener = listener
        threading.Thread(target=self._accept, args=(listener,), daemon=True).start()

    def down(self) -> None:
        listener, self._listener = self._listener, None
        if listener is not None:
            listener.close()
        with self._lock:
            for sock in self._open:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                sock.close()
            self._open.clear()

    def _accept(self, listener: socket.socket) -> None:
        while self._listener is listener:
            try:
                client, _ = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            server = socket.create_connection(self._target)
            with self._lock:
                self._open.update((client, server))
            for src, dst in ((client, server), (server, client)):
                threading.Thread(target=self._pipe, args=(src, dst), daemon=True).start()

    @staticmethod
    def _pipe(src: socket.socket, dst: socket.socket) -> None:
        try:
            while data := src.recv(65536):
                dst.sendall(data)
        except OSError:
            pass
        finally:
            for sock in (src, dst):
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass


def _reset_database_state() -> None:
    get_settings.cache_clear()
    engine_module._engine = None
    engine_module._admin_engine = None
    session_module._session_factory = None
    session_module._admin_session_factory = None


@pytest.fixture
def database(test_engine: AsyncEngine) -> Iterator[_Forwarder]:
    """Point every app login at the forwarder, which starts up."""
    names = ("DATABASE_URL", "ADMIN_DATABASE_URL")
    saved = {name: os.environ[name] for name in names}
    target = make_url(saved["DATABASE_URL"])
    forwarder = _Forwarder((target.host or "127.0.0.1", target.port or 5432))
    for name in names:
        url = make_url(saved[name]).set(host="127.0.0.1", port=forwarder.port)
        os.environ[name] = url.render_as_string(hide_password=False)
    _reset_database_state()
    yield forwarder
    forwarder.down()
    os.environ.update(saved)
    _reset_database_state()


@pytest.fixture(autouse=True)
def _no_knowledge_state_progress() -> Iterator[None]:
    with patch(
        "openscientist.job_manager._load_progress_from_knowledge_state", return_value=(0, 0)
    ):
        yield


@pytest_asyncio.fixture
async def job_in(test_engine: AsyncEngine) -> AsyncGenerator[dict[JobStatus, UUID], None]:
    """One job in each status, created straight on the test database."""
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    async with factory() as session:
        await session.execute(delete(Job).where(Job.status == JobStatus.QUEUED.value))
        jobs = {status: Job(research_question="Q?", status=status.value) for status in JobStatus}
        session.add_all(jobs.values())
        await session.commit()
    yield {status: job.id for status, job in jobs.items()}
    async with factory() as session:
        await session.execute(delete(Job).where(Job.id.in_([job.id for job in jobs.values()])))
        await session.commit()


async def _status(test_engine: AsyncEngine, job_id: UUID) -> str:
    async with test_engine.connect() as conn:
        return str(await conn.scalar(select(Job.status).where(Job.id == job_id)))


async def _down_for(database: _Forwarder, seconds: float) -> None:
    await asyncio.sleep(seconds)
    database.up()


async def test_a_container_write_outlives_a_short_outage(
    database: _Forwarder, test_engine: AsyncEngine, job_in: dict[JobStatus, UUID], tmp_path: Path
) -> None:
    job = job_in[JobStatus.RUNNING]
    database.down()
    write = asyncio.create_task(
        update_job_status(tmp_path / str(job), "completed", allowed_from=("running",))
    )
    await _down_for(database, 2)
    assert await asyncio.wait_for(write, 20)
    assert await _status(test_engine, job) == "completed"


async def test_a_container_write_fails_once_the_outage_outlasts_its_window(
    database: _Forwarder, test_engine: AsyncEngine, job_in: dict[JobStatus, UUID], tmp_path: Path
) -> None:
    job = job_in[JobStatus.RUNNING]
    database.down()
    started = time.monotonic()
    with pytest.raises(DatabaseUnavailableError):
        await update_job_status(
            tmp_path / str(job), "completed", allowed_from=("running",), retry_for=2
        )
    assert time.monotonic() - started < 10
    database.up()
    assert await _status(test_engine, job) == "running"


async def test_a_container_status_read_outlives_a_short_outage(
    database: _Forwarder, job_in: dict[JobStatus, UUID]
) -> None:
    database.down()
    read = asyncio.create_task(_assert_job_running(str(job_in[JobStatus.RUNNING])))
    await _down_for(database, 2)
    await asyncio.wait_for(read, 20)


async def test_any_container_read_waits_out_a_short_outage(
    database: _Forwarder,
    job_in: dict[JobStatus, UUID],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Agent containers run with a connect window, so a read that is not a
    status call, like loading the job's runtime context, survives too."""
    from openscientist.database.retry import CONTAINER_DB_RETRY_SECONDS
    from openscientist.orchestrator.discovery import _load_runtime_context

    monkeypatch.setenv("OPENSCIENTIST_DB_CONNECT_RETRY_SECONDS", str(CONTAINER_DB_RETRY_SECONDS))
    _reset_database_state()
    database.down()
    read = asyncio.create_task(_load_runtime_context(tmp_path / str(job_in[JobStatus.RUNNING])))
    await _down_for(database, 2)
    context = await asyncio.wait_for(read, 20)
    assert context["research_question"] == "Q?"


async def test_an_error_other_than_a_lost_connection_is_not_retried(
    test_engine: AsyncEngine, job_in: dict[JobStatus, UUID], tmp_path: Path
) -> None:
    @asynccontextmanager
    async def read_only() -> AsyncIterator[AsyncSession]:
        async with AsyncSession(test_engine) as session:
            await session.execute(text("SET LOCAL ROLE pg_read_all_data"))
            yield session

    started = time.monotonic()
    with pytest.raises(DBAPIError, match="permission denied"):
        await update_job_status(
            tmp_path / str(job_in[JobStatus.RUNNING]),
            "failed",
            allowed_from=("running",),
            open_session=read_only,
        )
    assert time.monotonic() - started < 1


async def test_a_job_read_during_an_outage_raises_instead_of_not_found(
    database: _Forwarder, job_in: dict[JobStatus, UUID], tmp_path: Path
) -> None:
    manager = JobManager(jobs_dir=tmp_path)
    database.down()
    with pytest.raises(DatabaseUnavailableError):
        await asyncio.to_thread(manager.get_job, str(job_in[JobStatus.RUNNING]))


async def test_the_api_answers_503_during_an_outage(
    database: _Forwarder, job_in: dict[JobStatus, UUID]
) -> None:
    from fastapi import FastAPI

    from openscientist.api.auth import get_current_user_from_api_key
    from openscientist.api.errors import register_database_error_handlers
    from openscientist.api.router import api_router

    user = User(id=uuid4(), email="outage@test.local", name="Outage")
    app = FastAPI()
    app.dependency_overrides[get_current_user_from_api_key] = lambda: user
    app.include_router(api_router)
    register_database_error_handlers(app)
    database.down()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(f"/api/v1/jobs/{job_in[JobStatus.RUNNING]}")
    assert response.status_code == 503
    assert response.json() == {"detail": "Database unavailable"}


async def test_a_worker_write_applies_once_the_database_returns(
    database: _Forwarder, test_engine: AsyncEngine, job_in: dict[JobStatus, UUID], tmp_path: Path
) -> None:
    manager = JobManager(jobs_dir=tmp_path)
    job = job_in[JobStatus.RUNNING]
    database.down()
    write = asyncio.create_task(
        asyncio.to_thread(
            manager._update_job_status,
            str(job),
            JobStatus.FAILED,
            allowed_from=JobManager.STARTED_STATUSES,
            error_message="Agent container exited with code 1",
            notify=False,
            retry=True,
        )
    )
    await _down_for(database, 2)
    assert await asyncio.wait_for(write, 20)
    assert await _status(test_engine, job) == "failed"


async def test_startup_recovery_runs_once_the_database_returns(
    database: _Forwarder, test_engine: AsyncEngine, job_in: dict[JobStatus, UUID], tmp_path: Path
) -> None:
    manager = JobManager(jobs_dir=tmp_path, max_concurrent=5)
    database.down()
    with (
        patch("openscientist.job_container.JobContainerRunner", return_value=MagicMock()),
        patch.object(manager, "_run_job"),
    ):
        recovery = asyncio.create_task(asyncio.to_thread(manager.recover_after_restart))
        await _down_for(database, 2)
        await asyncio.wait_for(recovery, 20)
    assert await _status(test_engine, job_in[JobStatus.RUNNING]) == "cancelled"
    assert str(job_in[JobStatus.QUEUED]) in manager._running_jobs


async def test_a_pick_that_meets_an_outage_starts_the_job_after_it(
    database: _Forwarder, job_in: dict[JobStatus, UUID], tmp_path: Path
) -> None:
    manager = JobManager(jobs_dir=tmp_path)
    queued = str(job_in[JobStatus.QUEUED])
    with patch.object(manager, "_run_job"):
        database.down()
        assert not await asyncio.to_thread(manager._start_next_queued_job)
        await _down_for(database, 2)
        for _ in range(100):
            if queued in manager._running_jobs:
                break
            await asyncio.sleep(0.2)
    assert queued in manager._running_jobs


async def test_a_crashed_container_hands_its_error_to_the_job(
    test_engine: AsyncEngine, job_in: dict[JobStatus, UUID], tmp_path: Path
) -> None:
    """The worker sees only the exit code, so the job's error comes from the
    file the container leaves in the job directory, and a stale one is ignored."""
    from openscientist.job.agent_error import write_agent_error

    job = str(job_in[JobStatus.QUEUED])
    (tmp_path / job).mkdir()
    write_agent_error(tmp_path / job, "a previous run's error")
    runner = MagicMock()
    runner.get_exit_code.return_value = 1
    runner.launch.side_effect = lambda job_id, job_dir, **_: write_agent_error(
        job_dir, "DatabaseUnavailableError: Connect call failed"
    )
    manager = JobManager(jobs_dir=tmp_path)
    with (
        patch("openscientist.job_container.JobContainerRunner", return_value=runner),
        patch("openscientist.job_manager.time.sleep"),
    ):
        await asyncio.to_thread(manager._run_job_in_container, job)
    async with test_engine.connect() as conn:
        error = await conn.scalar(select(Job.error_message).where(Job.id == UUID(job)))
    assert error == (
        "Agent container exited with code 1: DatabaseUnavailableError: Connect call failed"
    )
