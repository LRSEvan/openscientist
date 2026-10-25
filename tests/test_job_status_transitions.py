"""Job status writes apply only from the statuses the transition table allows
(plans/job-status-transitions.md), on real database rows."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Iterator
from pathlib import Path
from typing import Protocol
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
import pytest_asyncio
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from openscientist.database.models import Job
from openscientist.job_manager import JobManager, JobStatus, _db_update_job_status
from openscientist.orchestrator.discovery import _persist_final_status, _ReportOutcome


class MakeJob(Protocol):
    async def __call__(self, status: JobStatus, error: str | None = None) -> UUID: ...


@pytest.fixture(autouse=True)
def _no_knowledge_state_progress():
    with patch(
        "openscientist.job_manager._load_progress_from_knowledge_state", return_value=(0, 0)
    ):
        yield


@pytest_asyncio.fixture
async def make_job(test_engine: AsyncEngine) -> AsyncGenerator[MakeJob, None]:
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    created: list[UUID] = []

    async def _make(status: JobStatus, error: str | None = None) -> UUID:
        job = Job(research_question=f"In {status.value}?", status=status.value)
        job.error_message = error
        async with factory() as session:
            session.add(job)
            await session.commit()
        created.append(job.id)
        return job.id

    yield _make
    async with factory() as session:
        await session.execute(delete(Job).where(Job.id.in_(created)))
        await session.commit()


async def _status(test_engine: AsyncEngine, job_id: UUID) -> str:
    return (await _ending(test_engine, job_id))[0]


async def _ending(test_engine: AsyncEngine, job_id: UUID) -> tuple[str, str | None]:
    async with test_engine.connect() as conn:
        row = (
            await conn.execute(select(Job.status, Job.error_message).where(Job.id == job_id))
        ).one()
    return str(row.status), row.error_message


@pytest.fixture
def runner() -> Iterator[MagicMock]:
    """The container runner, which launches nothing. A launched container
    reports a crash at once, so a worker that should not have launched ends fast."""
    fake = MagicMock()
    fake.get_exit_code.return_value = 1
    with (
        patch("openscientist.job_container.JobContainerRunner", return_value=fake),
        patch("openscientist.job_manager.time.sleep"),
    ):
        yield fake


def _manager(tmp_path: Path, max_concurrent: int = 1) -> JobManager:
    return JobManager(jobs_dir=tmp_path, max_concurrent=max_concurrent)


async def test_building_a_job_manager_changes_no_job(
    make_job: MakeJob, test_engine: AsyncEngine, tmp_path: Path, runner: MagicMock
) -> None:
    """A CLI command or a dev reload builds one while the web's jobs run."""
    running = await make_job(JobStatus.RUNNING)
    _manager(tmp_path)
    assert await _status(test_engine, running) == "running"
    runner.cleanup.assert_not_called()


async def test_a_job_cancelled_before_its_worker_begins_is_not_launched(
    make_job: MakeJob, test_engine: AsyncEngine, tmp_path: Path, runner: MagicMock
) -> None:
    manager = _manager(tmp_path)
    job = await make_job(JobStatus.QUEUED)
    manager.cancel_job(str(job))
    with patch.object(manager, "_start_next_queued_job"):
        manager._run_job_in_container(str(job))
    runner.launch.assert_not_called()
    assert await _status(test_engine, job) == "cancelled"


async def test_cancelling_a_started_queued_job_keeps_its_slot_and_stops_it(
    make_job: MakeJob, tmp_path: Path, runner: MagicMock
) -> None:
    """The job's thread is live, so it keeps counting against max_concurrent."""
    manager = _manager(tmp_path)
    job = await make_job(JobStatus.QUEUED)
    manager._running_jobs[str(job)] = MagicMock()
    with patch.object(manager, "_start_next_queued_job"):
        manager.cancel_job(str(job))
    assert str(job) in manager._running_jobs
    runner.stop.assert_called_once_with(str(job))


async def test_a_started_queued_job_is_not_picked_again(
    make_job: MakeJob, tmp_path: Path, runner: MagicMock
) -> None:
    manager = _manager(tmp_path, max_concurrent=2)
    job = await make_job(JobStatus.QUEUED)
    own_thread = MagicMock()
    manager._running_jobs[str(job)] = own_thread
    with patch.object(manager, "_run_job"):
        manager._start_next_queued_job()
    assert manager._running_jobs[str(job)] is own_thread


async def test_the_oldest_queued_job_starts_first(
    make_job: MakeJob, tmp_path: Path, runner: MagicMock
) -> None:
    manager = _manager(tmp_path)
    older = await make_job(JobStatus.QUEUED)
    newer = await make_job(JobStatus.QUEUED)
    with patch.object(manager, "_run_job"):
        manager._start_next_queued_job()
    assert str(older) in manager._running_jobs
    assert str(newer) not in manager._running_jobs


async def test_a_job_waiting_for_feedback_can_be_cancelled(
    make_job: MakeJob, test_engine: AsyncEngine, tmp_path: Path, runner: MagicMock
) -> None:
    manager = _manager(tmp_path)
    job = await make_job(JobStatus.AWAITING_FEEDBACK)
    manager._running_jobs[str(job)] = MagicMock()
    with patch.object(manager, "_start_next_queued_job"):
        manager.cancel_job(str(job))
    assert await _status(test_engine, job) == "cancelled"
    runner.stop.assert_called_once_with(str(job))


@pytest.mark.parametrize("finished", [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED])
async def test_start_job_refuses_a_finished_job(
    make_job: MakeJob, test_engine: AsyncEngine, tmp_path: Path, finished: JobStatus
) -> None:
    manager = _manager(tmp_path)
    job = await make_job(finished)
    with pytest.raises(ValueError, match="cannot be started"):
        manager.start_job(str(job))
    assert str(job) not in manager._running_jobs
    assert await _status(test_engine, job) == finished.value


@pytest.mark.parametrize(
    ("web_status", "web_error", "outcome"),
    [
        (JobStatus.CANCELLED, None, _ReportOutcome(success=True, error="")),
        (JobStatus.FAILED, "Job timed out", _ReportOutcome(success=False, error="report")),
    ],
)
async def test_the_containers_result_does_not_overwrite_the_webs_ending(
    make_job: MakeJob,
    test_engine: AsyncEngine,
    tmp_path: Path,
    web_status: JobStatus,
    web_error: str | None,
    outcome: _ReportOutcome,
) -> None:
    """A cancel or a web timeout lands before the container's final write."""
    job = await make_job(web_status, web_error)
    await _persist_final_status(tmp_path / str(job), outcome)
    assert await _ending(test_engine, job) == (web_status.value, web_error)


async def test_the_webs_late_failure_does_not_overwrite_a_completed_job(
    make_job: MakeJob, test_engine: AsyncEngine
) -> None:
    """The container finished just before the web's timeout or crash check."""
    job = await make_job(JobStatus.COMPLETED)
    result = await _db_update_job_status(
        str(job),
        JobStatus.FAILED,
        allowed_from=JobManager.STARTED_STATUSES,
        error_message="Job timed out",
    )
    assert not result.applied
    assert await _status(test_engine, job) == "completed"


async def test_startup_recovery_keeps_waiting_jobs_and_ends_started_ones(
    make_job: MakeJob, test_engine: AsyncEngine, tmp_path: Path, runner: MagicMock
) -> None:
    pending = await make_job(JobStatus.PENDING)
    queued = await make_job(JobStatus.QUEUED)
    running = await make_job(JobStatus.RUNNING)
    paused = await make_job(JobStatus.AWAITING_FEEDBACK)
    completed = await make_job(JobStatus.COMPLETED)
    manager = _manager(tmp_path, max_concurrent=5)
    with patch.object(manager, "_run_job"):
        manager.recover_after_restart()
    assert await _status(test_engine, running) == "cancelled"
    assert await _status(test_engine, paused) == "cancelled"
    assert await _status(test_engine, completed) == "completed"
    assert await _status(test_engine, pending) == "queued"
    assert await _status(test_engine, queued) == "queued"
    assert {str(pending), str(queued)} <= manager._running_jobs.keys()
    cleaned = {call.args[0] for call in runner.cleanup.call_args_list}
    assert {str(running), str(paused)} <= cleaned
