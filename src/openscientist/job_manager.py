"""
Job manager for OpenScientist discovery jobs.

Handles job lifecycle, status tracking, and cleanup.
"""

import argparse
import json
import logging
import shutil
import threading
import time
from collections.abc import Coroutine
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from openscientist.container_manager import get_container_manager
from openscientist.database.models import User
from openscientist.database.models.job import Job as JobModel
from openscientist.database.models.job_share import JobShare
from openscientist.database.rls import set_current_user
from openscientist.database.session import AsyncSessionLocal
from openscientist.exceptions import ProviderError
from openscientist.job.types import JobInfo, JobStatus, JobStatusUpdateResult
from openscientist.knowledge_state import KnowledgeState
from openscientist.ntfy import notify_job_status_change
from openscientist.orchestrator import create_job
from openscientist.providers import get_provider
from openscientist.version import get_version_string

logger = logging.getLogger(__name__)


def _effective_model(settings: Any) -> str | None:
    """Resolve the model the active provider will actually use, for recording
    on the job so the UI can show a model badge.

    ``OPENSCIENTIST_MODEL`` (and the anthropic default) are checked first, but a
    provider may carry its model in a provider default (Ollama's, for example), so
    when those are unset we ask the provider itself. Returns None when nothing
    resolves (for example codex on the account default), leaving the job with a
    provider badge only.
    """
    model = settings.provider.model or settings.provider.anthropic_default_sonnet_model
    if model:
        return str(model)
    try:
        return get_provider().effective_model_name()
    except Exception as exc:  # provider misconfigured, unavailable, etc.
        logger.debug("Could not resolve provider model for job: %s", exc)
    return None


# Database helper functions for async operations


async def _apply_rls_context(session: AsyncSession, user_id: UUID | None) -> None:
    """Apply user-scoped RLS context when a user id is provided."""
    if user_id is None:
        return
    await session.execute(text("SET ROLE openscientist_app"))
    await set_current_user(session, user_id)


def _derive_progress_from_db(status: str, current_iteration: int) -> int:
    """Derive iterations_completed from Job model columns (no KS load needed)."""
    if status in ("running", "awaiting_feedback"):
        return current_iteration - 1 if current_iteration > 1 else 0
    return current_iteration


def _load_progress_from_knowledge_state(
    job_id: str,
    status: str,
    default_iterations: int,
    default_findings: int,
) -> tuple[int, int]:
    """Load iteration and findings progress from persisted knowledge state."""
    try:
        ks = KnowledgeState.load_from_database_sync(job_id).data
        findings_count = len(ks.get("findings", []))
        ks_iteration = int(ks.get("iteration", 1))

        if status in ("running", "awaiting_feedback"):
            iterations_completed = ks_iteration - 1 if ks_iteration > 1 else 0
        else:
            iterations_completed = ks_iteration

        return iterations_completed, findings_count
    except Exception as e:
        logger.warning("Failed to load KS for job %s: %s", job_id, e)
        return default_iterations, default_findings


async def _db_create_job(
    job_id: str,
    research_question: str,
    max_iterations: int,
    use_hypotheses: bool = False,
    investigation_mode: str = "autonomous",
    owner_id: UUID | None = None,
    short_title: str | None = None,
    description: str | None = None,
    pdb_code: str | None = None,
    space_group: str | None = None,
    llm_provider: str | None = None,
    llm_config: dict[str, Any] | None = None,
) -> JobModel:
    """Create a job in the database (thread-safe for worker threads).

    Args:
        job_id: UUID string for the job (used as primary key)
        research_question: The full research question; drives the agent prompt
        max_iterations: Maximum iterations allowed
        use_hypotheses: Whether hypothesis tracking tools are enabled for this job
        investigation_mode: Investigation mode ('autonomous' or 'coinvestigate')
        owner_id: UUID of the job owner (optional)
        short_title: Optional short display label (truncated to 100 chars)
        description: Optional job description
        pdb_code: Optional PDB code
        space_group: Optional crystal space group

    Returns:
        The created JobModel instance
    """
    async with AsyncSessionLocal(thread_safe=True) as session:
        await _apply_rls_context(session, owner_id)

        job = JobModel(
            id=UUID(job_id),
            owner_id=owner_id,
            research_question=research_question,
            short_title=short_title[:100] if short_title else None,
            description=description,
            use_hypotheses=use_hypotheses,
            investigation_mode=investigation_mode,
            status=JobStatus.PENDING.value,
            max_iterations=max_iterations,
            current_iteration=0,
            pdb_code=pdb_code,
            space_group=space_group,
            llm_provider=llm_provider,
            llm_config=llm_config,
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
        return job


async def _db_get_job(job_id: str, user_id: UUID | None = None) -> JobModel | None:
    """Get a job from the database (thread-safe for worker threads).

    When user_id is provided, drops to openscientist_app role so RLS policies
    are enforced. Without user_id, runs as the connection user (superuser)
    which bypasses RLS — use only for internal/system operations.
    """
    async with AsyncSessionLocal(thread_safe=True) as session:
        await _apply_rls_context(session, user_id)

        stmt = select(JobModel).where(JobModel.id == UUID(job_id))
        result = await session.execute(stmt)
        return result.scalar_one_or_none()


async def _db_get_share_permission(job_id: str, user_id: UUID) -> str | None:
    """Return the share permission level for a user on a job.

    Returns 'view', 'edit', or None if the user has no share.
    """
    async with AsyncSessionLocal(thread_safe=True) as session:
        await _apply_rls_context(session, user_id)

        stmt = select(JobShare.permission_level).where(
            JobShare.job_id == UUID(job_id),
            JobShare.shared_with_user_id == user_id,
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()


async def _db_update_job_status(
    job_id: str,
    status: JobStatus,
    error_message: str | None = None,
    user_id: UUID | None = None,
    cancellation_reason: str | None = None,
) -> JobStatusUpdateResult:
    """Update job status in the database (thread-safe for worker threads).

    Returns:
        JobStatusUpdateResult with the job owner's ntfy settings for notifications.
    """
    result = JobStatusUpdateResult()

    async with AsyncSessionLocal(thread_safe=True) as session:
        await _apply_rls_context(session, user_id)

        stmt = select(JobModel).where(JobModel.id == UUID(job_id))
        db_result = await session.execute(stmt)
        job = db_result.scalar_one_or_none()

        if job:
            job.status = status.value
            if error_message:
                job.error_message = error_message
            if cancellation_reason:
                job.cancellation_reason = cancellation_reason
            await session.commit()

            result.owner_id = str(job.owner_id) if job.owner_id else None
            result.job_title = job.short_title or job.research_question
            result.current_iteration = job.current_iteration

            # Fetch owner's ntfy settings for notifications
            if job.owner_id:
                user_stmt = select(User.ntfy_enabled, User.ntfy_topic).where(
                    User.id == job.owner_id
                )
                user_result = await session.execute(user_stmt)
                user_row = user_result.first()
                if user_row:
                    result.ntfy_enabled = user_row.ntfy_enabled
                    result.ntfy_topic = user_row.ntfy_topic

    return result


async def _db_get_job_statuses(job_ids: list[str]) -> dict[str, str]:
    """Batch-fetch job statuses by ID."""
    async with AsyncSessionLocal(thread_safe=True) as session:
        stmt = select(JobModel.id, JobModel.status).where(
            JobModel.id.in_([UUID(jid) for jid in job_ids])
        )
        result = await session.execute(stmt)
        return {str(row.id): row.status for row in result}


async def _db_list_jobs(
    status: JobStatus | None = None,
    limit: int | None = None,
    user_id: UUID | None = None,
) -> list[JobModel]:
    """List jobs from the database (thread-safe for worker threads)."""
    async with AsyncSessionLocal(thread_safe=True) as session:
        await _apply_rls_context(session, user_id)

        stmt = select(JobModel).order_by(JobModel.created_at.desc())

        if status:
            stmt = stmt.where(JobModel.status == status.value)

        if limit:
            stmt = stmt.limit(limit)

        result = await session.execute(stmt)
        return list(result.scalars().all())


_LAST_SERVED_LOOKBACK = timedelta(days=90)
"""How far back ``_select_next_queued_job`` looks for an owner's
last-served job. Bounds the aggregate to a recency window instead of a
user's entire job history, which would otherwise scan ~100% of a growing
jobs table on every pick (a plain index can't narrow a filter that excludes
almost nothing — verified via EXPLAIN against ~50k synthetic rows before
adding this bound). An owner whose last job predates this window is
treated as never-served — a deliberate tradeoff: a long-dormant user gets
welcomed back promptly rather than penalized, which isn't really unfair in
practice.
"""


async def _select_next_queued_job(
    session: AsyncSession, *, privileged_only: bool | None = None
) -> JobModel | None:
    """The fairness query: least-recently-served owner first, that owner's
    oldest queued job as the tiebreak (FIFO within an owner, rotation
    across owners). Factored out so both the self-opening and
    test-injected-session paths in ``_db_get_next_queued_job`` share it.

    A dedicated query rather than reusing ``_db_list_jobs`` — that function's
    newest-first ordering is its own documented contract (``list_jobs()``),
    relied on elsewhere; this scheduler pick is a different, unrelated
    question ("who goes next") from "list jobs for display."

    "Last served" is approximated as the most recent ``updated_at`` among an
    owner's jobs that have left the queue within ``_LAST_SERVED_LOOKBACK``
    (i.e. started at least once, recently) — jobs are marked ``running``
    before launch (see ``_run_job_in_container``), so ``updated_at`` is
    stamped at start. This is the same proxy ``JobInfo.started_at`` already
    relies on (``job/types.py``), so it needs no new column. An owner with
    no such jobs (never served, including orphaned/``NULL``-owner jobs, or
    only served outside the lookback window) sorts first via ``NULLS
    FIRST``.

    Known limitation: ``NULL = NULL`` is never true in SQL, so the owner
    join can never match one orphaned (``owner_id IS NULL``) job's "last
    served" row against another orphaned job — every orphaned job always
    looks never-served, even if a different orphaned job already ran.
    Orphaned jobs just fall back to plain FIFO-by-``created_at`` among
    themselves as a result, never getting real owner-rotation. Acceptable:
    orphaned ("legacy", per the Job model) jobs have no real owner for the
    per-owner fairness policy to apply to in the first place.

    Known limitation: a job cancelled while still queued/pending never ran,
    but still counts as "served" here, because there is no reliable,
    schema-free signal that distinguishes "cancelled before ever starting"
    from "cancelled after running." ``current_iteration`` looked promising
    but isn't: it's seeded to 1 at job *creation* time, before the job is
    even queued (``KnowledgeState.__init__`` -> ``orchestrator/setup.py``'s
    ``create_job()`` -> ``save_to_database_sync``, synchronously, at
    submission), not after the first real iteration — so it can't tell
    "never queued-to-run" apart from "ran once." Closing this gap for real
    would need a genuine ``started_at``-style column, which is out of scope
    here.

    ``privileged_only=None`` (the default) ranks across all queued jobs.
    ``True``/``False`` additionally restricts to jobs owned by a
    privileged/regular ``User``, scoping the identical ranking rule to one
    pool instead of duplicating it — ``_db_get_next_queued_job`` uses this
    to let privileged owners' jobs always run first. A ``NULL`` owner
    (orphaned job) is treated as regular.
    """
    last_served_cutoff = datetime.now(UTC) - _LAST_SERVED_LOOKBACK
    last_served = (
        select(
            JobModel.owner_id.label("owner_id"),
            func.max(JobModel.updated_at).label("last_served"),
        )
        .where(
            JobModel.status.not_in([JobStatus.QUEUED.value, JobStatus.PENDING.value]),
            JobModel.updated_at > last_served_cutoff,
        )
        .group_by(JobModel.owner_id)
        .subquery()
    )
    stmt = (
        select(JobModel)
        # NULL = NULL is never true, so this never matches one orphaned job
        # against another's "last served" row — see the known-limitation
        # note on _db_get_next_queued_job above.
        .outerjoin(last_served, last_served.c.owner_id == JobModel.owner_id)
        .where(JobModel.status == JobStatus.QUEUED.value)
    )
    if privileged_only is not None:
        privileged_owner_ids = select(User.id).where(User.privileged.is_(True)).scalar_subquery()
        if privileged_only:
            stmt = stmt.where(JobModel.owner_id.in_(privileged_owner_ids))
        else:
            # owner_id NOT IN (...) is NULL (never true) for orphaned jobs,
            # so they'd be silently excluded from the regular pool too
            # without this explicit IS NULL case.
            stmt = stmt.where(
                (JobModel.owner_id.is_(None)) | (JobModel.owner_id.not_in(privileged_owner_ids))
            )
    stmt = stmt.order_by(
        last_served.c.last_served.asc().nulls_first(), JobModel.created_at.asc()
    ).limit(1)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def _pick_privileged_or_regular(session: AsyncSession) -> JobModel | None:
    """A privileged owner's queued job always wins; otherwise falls back to
    the regular pool. Both pools reuse ``_select_next_queued_job``'s
    ranking unchanged, so an owner's position within their own pool is
    unaffected by privilege."""
    privileged_pick = await _select_next_queued_job(session, privileged_only=True)
    if privileged_pick is not None:
        return privileged_pick
    return await _select_next_queued_job(session, privileged_only=False)


async def _db_get_next_queued_job(session: AsyncSession | None = None) -> JobModel | None:
    """Pick the next queued job (see ``_pick_privileged_or_regular``). With
    no privileged users (true for every user before this feature), this
    degenerates to exactly #268's original single-pool behavior.

    Accepts an optional ``session`` (same idiom as
    ``KnowledgeState.save_to_database``) so tests — and the admin Queue
    panel's ``_simulate_privileged_queue_order``, which is production code,
    not a test — can run the real query inside their own transaction;
    ``_start_next_queued_job`` always calls this with no argument, opening
    its own session exactly like ``_db_list_jobs``.
    """
    if session is not None:
        return await _pick_privileged_or_regular(session)
    async with AsyncSessionLocal(thread_safe=True) as session:
        return await _pick_privileged_or_regular(session)


_NEVER_SERVED = datetime.min.replace(tzinfo=UTC)
"""Sort-first sentinel standing in for SQL's NULLS FIRST in the pure-Python
re-simulation below (datetime.min sorts before every real timestamp)."""


async def _simulate_privileged_queue_order(session: AsyncSession) -> list[dict[str, Any]]:
    """Non-destructively compute the full order the scheduler
    (``_db_get_next_queued_job``) would drain the current queue in, for the
    admin queue-visibility panel.

    Computed as a pure in-memory re-simulation of ``_select_next_queued_job``'s
    ranking rule over data fetched once — not by repeatedly calling the
    real pick-and-mark-served function against the database. Doing it that
    way would require a SAVEPOINT held for the whole drain (so each pick's
    "mark as served" write stays visible to the next pick), which would
    keep every picked job's row locked for as long as the whole simulation
    takes — on a long queue, that could stall the real scheduler trying to
    start one of those same jobs while an admin merely has this panel
    open. This version never writes anything, so there's nothing to lock
    or roll back.

    Caution: this re-implements ``_select_next_queued_job``'s ranking
    rule rather than calling it, so if that rule ever changes, this must
    be updated to match — ``TestSimulatePrivilegedQueueOrder`` exists to
    catch drift between the two.
    """
    cutoff = datetime.now(UTC) - _LAST_SERVED_LOOKBACK
    last_served_stmt = (
        select(JobModel.owner_id, func.max(JobModel.updated_at))
        .where(
            JobModel.owner_id.is_not(None),
            JobModel.status.not_in([JobStatus.QUEUED.value, JobStatus.PENDING.value]),
            JobModel.updated_at > cutoff,
        )
        .group_by(JobModel.owner_id)
    )
    # owner_id is NOT NULL here by the query's own filter above; the
    # column's nullable type is just a static-typing artifact of the FK.
    last_served_by_owner: dict[UUID, datetime] = {
        owner_id: last_served
        for owner_id, last_served in (await session.execute(last_served_stmt)).all()
        if owner_id is not None
    }

    queued_stmt = (
        select(JobModel, User)
        .outerjoin(User, User.id == JobModel.owner_id)
        .where(JobModel.status == JobStatus.QUEUED.value)
        .order_by(JobModel.created_at.asc())
    )
    queued_rows = (await session.execute(queued_stmt)).all()

    # Group by pool, keyed by owner_id for a real owner (so last_served is
    # looked up once per owner) or by the job's own id for an orphaned job
    # — mirroring SQL's NULL != NULL, every orphaned job is independently
    # never-served rather than grouped with other orphaned jobs.
    privileged_pool: dict[object, list[tuple[JobModel, User | None]]] = {}
    regular_pool: dict[object, list[tuple[JobModel, User | None]]] = {}
    for job, owner in queued_rows:
        target_pool = privileged_pool if (owner is not None and owner.privileged) else regular_pool
        key: object = owner.id if owner is not None else job.id
        target_pool.setdefault(key, []).append((job, owner))

    # Sort key per key is (generation, value): generation 0 means "not yet
    # picked in this simulation" (value = real last_served, or _NEVER_SERVED),
    # generation 1 means "picked at virtual_clock=value" — any generation-1
    # key sorts after every generation-0 key regardless of value, and
    # higher virtual_clock sorts later, which is exactly "simulated-serving
    # is always more recent than real history, and later picks are more
    # recent than earlier ones." Avoids mixing real and synthetic
    # timestamps in one comparable type.
    order: list[tuple[JobModel, User | None]] = []
    for pool in (privileged_pool, regular_pool):
        remaining = {key: list(jobs) for key, jobs in pool.items()}
        last_served: dict[object, tuple[int, datetime | int]] = {}
        for key, jobs in remaining.items():
            owner = jobs[0][1]
            seed = last_served_by_owner.get(owner.id, _NEVER_SERVED) if owner else _NEVER_SERVED
            last_served[key] = (0, seed)
        virtual_clock = 0
        while remaining:
            next_key = min(
                remaining,
                key=lambda k: (last_served[k], remaining[k][0][0].created_at),
            )
            job, owner = remaining[next_key].pop(0)
            order.append((job, owner))
            virtual_clock += 1
            last_served[next_key] = (1, virtual_clock)
            if not remaining[next_key]:
                del remaining[next_key]

    return [
        {
            "rank": i + 1,
            "pool": "privileged" if (owner is not None and owner.privileged) else "regular",
            "owner_name": owner.name if owner is not None else "(orphaned, no owner)",
            "job_id": str(job.id),
            "research_question": job.research_question,
            "created_at": job.created_at,
        }
        for i, (job, owner) in enumerate(order)
    ]


async def _db_delete_job(job_id: str, user_id: UUID | None = None) -> None:
    """Delete a job from the database (thread-safe for worker threads)."""
    async with AsyncSessionLocal(thread_safe=True) as session:
        await _apply_rls_context(session, user_id)

        stmt = select(JobModel).where(JobModel.id == UUID(job_id))
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

        if job:
            await session.delete(job)
            await session.commit()


def _run_async[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run async code from sync context.

    Thin wrapper around :func:`openscientist.async_tasks.run_sync` kept for
    internal compatibility within this module.
    """
    from openscientist.async_tasks import run_sync

    return run_sync(coro)


class JobManager:
    """
    Manages OpenScientist discovery jobs.

    Features:
    - Create and queue jobs
    - Run jobs asynchronously
    - Track job status
    - List and query jobs
    - Clean up old jobs
    """

    def __init__(self, jobs_dir: Path = Path("jobs"), max_concurrent: int = 1):
        """
        Initialize job manager.

        Args:
            jobs_dir: Base directory for jobs
            max_concurrent: Maximum concurrent jobs
        """
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

        self.max_concurrent = max_concurrent
        self._running_jobs: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

        # Clean up any stale running/queued jobs from previous restart
        self._cleanup_stale_jobs()

        logger.info(
            "JobManager initialized: %s, max_concurrent=%d",
            self.jobs_dir,
            max_concurrent,
        )

    def _cleanup_stale_jobs(self) -> None:
        """
        Mark any running/queued/awaiting_feedback jobs as cancelled on startup.

        This handles the case where the server restarts while jobs are in progress.
        Since the orchestrator process died, these jobs will never complete.
        """
        stale_count = 0
        for job_info in self._list_operational_jobs():
            job_id = job_info.job_id
            if job_info.status in [
                JobStatus.RUNNING,
                JobStatus.QUEUED,
                JobStatus.AWAITING_FEEDBACK,
                JobStatus.GENERATING_REPORT,
            ]:
                logger.warning(
                    "Marking stale job %s as cancelled (was %s)",
                    job_id,
                    job_info.status.value,
                )
                # Update database
                try:
                    owner_id = UUID(job_info.owner_id) if job_info.owner_id else None
                    _run_async(
                        _db_update_job_status(
                            job_id,
                            JobStatus.CANCELLED,
                            user_id=owner_id,
                            cancellation_reason="Server restarted while job was running",
                        )
                    )
                except Exception as e:
                    logger.error("Failed to update job %s in database: %s", job_id, e)

                # Stop any orphaned agent container for this stale job.
                try:
                    from openscientist.job_container import JobContainerRunner

                    JobContainerRunner().cleanup(job_id)
                except Exception as e:
                    logger.warning("Failed to cleanup container for stale job %s: %s", job_id, e)

                stale_count += 1

        if stale_count > 0:
            logger.info("Cleaned up %d stale job(s) from previous run", stale_count)

    def _ensure_job_not_exists(self, job_id: str) -> None:
        if self.get_job(job_id) is not None:
            raise ValueError(f"Job {job_id} already exists")

    def _check_budget_before_creation(self) -> None:
        try:
            provider = get_provider()
            budget_check = provider.check_budget_limits()
        except ProviderError as e:
            # Keep job creation available if provider cost endpoint is temporarily unavailable.
            logger.warning("Budget check unavailable: %s", e)
            budget_check = {"can_proceed": True}

        if budget_check.get("can_proceed", True):
            return

        errors = budget_check.get("errors", [])
        error_msg = "; ".join(errors) if errors else "Budget limit exceeded"
        raise ValueError(f"Cannot create job: {error_msg}")

    def _create_db_job_record(
        self,
        *,
        job_id: str,
        research_question: str,
        max_iterations: int,
        use_hypotheses: bool,
        investigation_mode: str,
        owner_id: str | None,
        short_title: str | None,
        description: str | None,
        pdb_code: str | None,
        space_group: str | None,
        llm_provider: str | None = None,
        llm_config: dict[str, Any] | None = None,
    ) -> UUID | None:
        owner_uuid = UUID(owner_id) if owner_id else None
        try:
            _run_async(
                _db_create_job(
                    job_id,
                    research_question,
                    max_iterations,
                    use_hypotheses=use_hypotheses,
                    investigation_mode=investigation_mode,
                    owner_id=owner_uuid,
                    short_title=short_title,
                    description=description,
                    pdb_code=pdb_code,
                    space_group=space_group,
                    llm_provider=llm_provider,
                    llm_config=llm_config,
                )
            )
        except Exception as e:
            logger.error("Failed to create job in database: %s", e)
            raise ValueError(f"Failed to create job in database: {e}") from e
        return owner_uuid

    def _rollback_failed_job_creation(self, job_id: str, owner_uuid: UUID | None) -> None:
        try:
            _run_async(_db_delete_job(job_id, owner_uuid))
        except Exception as cleanup_error:
            logger.error("Failed to rollback DB job %s: %s", job_id, cleanup_error)
        try:
            job_dir = self.jobs_dir / job_id
            if job_dir.exists():
                shutil.rmtree(job_dir)
        except OSError:
            logger.warning("Failed to cleanup job directory after failure: %s", job_id)

    def _create_job_files(
        self,
        *,
        job_id: str,
        research_question: str,
        data_files: list[Path],
        max_iterations: int,
        owner_id: str | None,
        owner_uuid: UUID | None,
    ) -> None:
        try:
            create_job(
                job_id=job_id,
                research_question=research_question,
                data_files=data_files,
                max_iterations=max_iterations,
                jobs_dir=self.jobs_dir,
                owner_id=owner_id,
            )
        except Exception as e:
            logger.error("Failed to initialize filesystem for job %s: %s", job_id, e)
            # Compensating action: remove partially-created DB row/files to avoid split-brain.
            self._rollback_failed_job_creation(job_id, owner_uuid)
            raise ValueError(f"Failed to initialize job files: {e}") from e

    def create_job(
        self,
        job_id: str,
        research_question: str,
        data_files: list[Path],
        max_iterations: int = 10,
        use_hypotheses: bool = False,
        auto_start: bool = True,
        investigation_mode: str = "autonomous",
        owner_id: str | None = None,
        short_title: str | None = None,
        description: str | None = None,
        pdb_code: str | None = None,
        space_group: str | None = None,
    ) -> JobInfo:
        """
        Create a new discovery job.

        Args:
            job_id: Unique job identifier
            research_question: The full research question; drives the agent prompt
            data_files: List of data file paths (can be empty for literature-only jobs)
            max_iterations: Maximum iterations
            use_hypotheses: Whether to enable hypothesis tracking tools
            auto_start: Whether to start job immediately
            investigation_mode: "autonomous" (default) or "coinvestigate"
            owner_id: UUID of the job owner (optional, for orphaned jobs)
            short_title: Optional short display label (truncated to 100 chars)
            description: Optional job description
            pdb_code: Optional PDB code metadata
            space_group: Optional crystal space group metadata

        Returns:
            JobInfo object

        Raises:
            ValueError: If job_id already exists
            BudgetExceededError: If insufficient budget
        """
        self._ensure_job_not_exists(job_id)
        self._check_budget_before_creation()

        # Capture provider & model from current settings
        from openscientist.settings import get_settings

        settings = get_settings()
        llm_provider = settings.provider.provider_id.lower()
        model = _effective_model(settings)
        llm_config = {"model": model} if model else None

        # Create job in database
        logger.info("Creating job %s in database", job_id)
        owner_uuid = self._create_db_job_record(
            job_id=job_id,
            research_question=research_question,
            max_iterations=max_iterations,
            use_hypotheses=use_hypotheses,
            investigation_mode=investigation_mode,
            owner_id=owner_id,
            short_title=short_title,
            description=description,
            pdb_code=pdb_code,
            space_group=space_group,
            llm_provider=llm_provider,
            llm_config=llm_config,
        )

        # Create job directory and files
        logger.info("Creating job %s filesystem structure", job_id)
        self._create_job_files(
            job_id=job_id,
            research_question=research_question,
            data_files=data_files,
            max_iterations=max_iterations,
            owner_id=owner_id,
            owner_uuid=owner_uuid,
        )

        # Load job info
        job_info = self._load_job_info(job_id)
        if job_info is None:
            raise ValueError(f"Failed to load newly created job {job_id}")

        # Auto-start if requested
        if auto_start:
            self.start_job(job_id)

        return job_info

    def start_job(self, job_id: str) -> None:
        """
        Start a job asynchronously.

        Args:
            job_id: Job ID

        Raises:
            ValueError: If job not found or already running
        """
        with self._lock:
            # Check job exists
            job_info = self.get_job(job_id)
            if job_info is None:
                raise ValueError(f"Job {job_id} not found")

            # Check if already running
            if job_id in self._running_jobs:
                raise ValueError(f"Job {job_id} is already running")

            # Check concurrent limit (awaiting_feedback jobs don't count)
            if self._get_active_job_count() >= self.max_concurrent:
                # Queue the job
                self._update_job_status(job_id, JobStatus.QUEUED)
                logger.info("Job %s queued (max concurrent reached)", job_id)
                return

            # Start the job
            logger.info("Starting job %s", job_id)
            thread = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
            self._running_jobs[job_id] = thread
            thread.start()

    def regenerate_report(self, job_id: str) -> None:
        """Re-run only the report-generation phase for a completed job.

        Backs the admin "Regenerate report" action. Launches the agent
        container in report-only mode (no discovery iterations: the findings in
        the persisted KnowledgeState are reused). The job must exist, be
        completed, and not currently running, and a concurrency slot must be
        free. Unlike ``start_job`` this does NOT queue: report regeneration is a
        one-off admin action, so it fails loudly rather than waiting.

        Raises:
            ValueError: If the job is missing, not completed, already running,
                or the max concurrent limit is reached.
        """
        with self._lock:
            job_info = self.get_job(job_id)
            if job_info is None:
                raise ValueError(f"Job {job_id} not found")
            if job_info.status != JobStatus.COMPLETED:
                raise ValueError(f"Job {job_id} is not completed (status: {job_info.status.value})")
            if job_id in self._running_jobs:
                raise ValueError(f"Job {job_id} is already running")
            if self._get_active_job_count() >= self.max_concurrent:
                raise ValueError("Cannot regenerate report: maximum concurrent jobs reached")

            logger.info("Regenerating report for job %s", job_id)
            thread = threading.Thread(
                target=self._run_job,
                args=(job_id,),
                kwargs={"run_mode": "report_only"},
                daemon=True,
            )
            self._running_jobs[job_id] = thread
            thread.start()

    def _run_job(self, job_id: str, run_mode: str = "discovery") -> None:
        """Run a job (internal, called by thread)."""
        self._run_job_in_container(job_id, run_mode=run_mode)

    def _run_job_in_container(self, job_id: str, run_mode: str = "discovery") -> None:
        """Launch an agent container for the job and block until it reaches a terminal status.

        ``run_mode`` is "discovery" for the full loop or "report_only" to
        re-run just the report-generation phase against the persisted findings.
        """
        from openscientist.job_container import JobContainerRunner

        poll_interval = 5
        terminal_statuses = {"completed", "failed", "cancelled"}

        job_dir = self.jobs_dir / job_id
        runner = JobContainerRunner()

        try:
            self._update_job_status(job_id, JobStatus.RUNNING)
            runner.launch(job_id, job_dir, run_mode=run_mode)
            logger.info("Agent container launched for job %s (mode=%s)", job_id, run_mode)

            # Poll the database until the container's agent writes a terminal status.
            # Also check if the container has exited unexpectedly (crash before DB write).
            from openscientist.settings import get_settings

            timeout_seconds = get_settings().container.agent_timeout
            timeout_hours = timeout_seconds / 3600
            elapsed = 0
            while elapsed < timeout_seconds:
                time.sleep(poll_interval)
                elapsed += poll_interval
                try:
                    job_info = self._load_job_info(job_id)
                    if job_info and job_info.status.value in terminal_statuses:
                        logger.info(
                            "Container job %s reached terminal status: %s",
                            job_id,
                            job_info.status.value,
                        )
                        return
                except Exception as poll_err:
                    logger.warning("DB poll failed for job %s: %s", job_id, poll_err)

                # If the container has exited but the DB still shows running,
                # the agent crashed before writing a terminal status — fail fast.
                exit_code = runner.get_exit_code(job_id)
                if exit_code is not None and exit_code != 0:
                    logger.error(
                        (
                            "Agent container for job %s exited with code %d before "
                            "writing terminal status"
                        ),
                        job_id,
                        exit_code,
                    )
                    self._update_job_status(
                        job_id,
                        JobStatus.FAILED,
                        error_message=f"Agent container exited with code {exit_code}",
                    )
                    return

            # Hard timeout reached.
            logger.error("Container job %s timed out after %.1f hours", job_id, timeout_hours)
            self._update_job_status(
                job_id,
                JobStatus.FAILED,
                error_message=f"Job timed out after {timeout_hours:.1f} hours",
            )

        except Exception as e:
            logger.error(
                "Container job %s failed [%s]: %s", job_id, get_version_string(), e, exc_info=True
            )
            self._update_job_status(job_id, JobStatus.FAILED, error_message=str(e))

        finally:
            runner.cleanup(job_id, log_dir=job_dir)
            with self._lock:
                self._running_jobs.pop(job_id, None)
            self._start_next_queued_job()

    def _start_next_queued_job(self) -> None:
        """Start the next queued job if slots available."""
        with self._lock:
            if self._get_active_job_count() >= self.max_concurrent:
                return

            try:
                next_job = _run_async(_db_get_next_queued_job())
            except Exception as e:
                # Matches _list_operational_jobs's own swallow-and-log: a
                # transient DB error here must not propagate out of
                # cancel_job() (which calls this unguarded) and turn an
                # already-committed cancellation into an apparent failure.
                logger.warning("Failed to pick next queued job: %s", e)
                return
            if next_job is None:
                return

            job_id = str(next_job.id)

            # Claim the job by flipping it out of "queued" here, while
            # still holding self._lock, rather than leaving that to the
            # spawned thread below. _db_get_next_queued_job's query only
            # considers status == "queued" rows; without this, the picked
            # job stays "queued" in the database until that thread gets
            # around to its own status update, and another
            # _start_next_queued_job call landing in that window (e.g.
            # from a second job finishing or being cancelled moments
            # later) could pick this same job again and launch it twice.
            # Using the raw DB write (not the notification-sending
            # self._update_job_status wrapper) keeps this idempotent and
            # side-effect-free -- the thread's own self._update_job_status
            # call below still fires the one real "job started" notification.
            try:
                _run_async(_db_update_job_status(job_id, JobStatus.RUNNING))
            except Exception as e:
                logger.warning("Failed to claim job %s before starting: %s", job_id, e)
                return

            logger.info("Starting queued job %s", job_id)
            thread = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
            self._running_jobs[job_id] = thread
            thread.start()

    def cancel_job(self, job_id: str) -> None:
        """
        Cancel a pending, running, or queued job.

        Args:
            job_id: Job ID

        Raises:
            ValueError: If job not found
        """
        job_info = self.get_job(job_id)
        if job_info is None:
            raise ValueError(f"Job {job_id} not found")

        if job_info.status not in [JobStatus.PENDING, JobStatus.RUNNING, JobStatus.QUEUED]:
            raise ValueError(f"Job {job_id} is not pending, running, or queued")

        # Update status with reason
        self._update_job_status(
            job_id, JobStatus.CANCELLED, cancellation_reason="Cancelled by user"
        )

        # For running jobs, keep the thread tracked until it exits so
        # active-slot accounting stays accurate.
        with self._lock:
            if job_info.status in [JobStatus.PENDING, JobStatus.QUEUED]:
                self._running_jobs.pop(job_id, None)

        # Send SIGTERM to the agent container immediately so it doesn't
        # keep burning resources until the polling loop notices the DB change.
        if job_info.status == JobStatus.RUNNING:
            try:
                from openscientist.job_container import JobContainerRunner

                JobContainerRunner().stop(job_id)
            except Exception as e:
                logger.warning("Failed to stop container for cancelled job %s: %s", job_id, e)

        logger.info("Job %s cancelled", job_id)

        # Start next queued job if any
        self._start_next_queued_job()

    def get_job(self, job_id: str) -> JobInfo | None:
        """
        Get job information.

        Args:
            job_id: Job ID

        Returns:
            JobInfo or None if not found
        """
        return self._load_job_info(job_id)

    def list_jobs(
        self,
        status: JobStatus | None = None,
        limit: int | None = None,
    ) -> list[JobInfo]:
        """
        List all jobs from the database (no user filtering).

        For user-facing queries, use a database session with RLS instead.

        Args:
            status: Filter by status
            limit: Maximum number of jobs to return

        Returns:
            List of JobInfo objects, sorted by created_at (newest first)
        """
        db_jobs = _run_async(_db_list_jobs(status=status, limit=limit))
        return [self._db_model_to_job_info(m) for m in db_jobs]

    def delete_job(self, job_id: str) -> None:
        """
        Delete a job and its files.

        Args:
            job_id: Job ID

        Raises:
            ValueError: If job not found or still running
        """
        job_info = self.get_job(job_id)
        if job_info is None:
            raise ValueError(f"Job {job_id} not found")

        if job_info.status == JobStatus.RUNNING:
            raise ValueError(f"Cannot delete running job {job_id}")

        # Delete from database
        try:
            owner_id = UUID(job_info.owner_id) if job_info.owner_id else None
            _run_async(_db_delete_job(job_id, owner_id))
        except Exception as e:
            logger.error("Failed to delete job from database: %s", e)
            raise ValueError(f"Failed to delete job {job_id} from database: {e}") from e

        # Clean up any executor containers for this job
        try:
            container_manager = get_container_manager()
            if container_manager.is_available():
                removed = container_manager.cleanup_job_containers(job_id)
                if removed > 0:
                    logger.info("Removed %d executor container(s) for job %s", removed, job_id)
        except Exception as e:
            logger.warning("Failed to cleanup containers for job %s: %s", job_id, e)

        # Delete job directory
        job_dir = self.jobs_dir / job_id
        if job_dir.exists():
            shutil.rmtree(job_dir)
            logger.info("Deleted job %s", job_id)

    def cleanup_old_jobs(self, days: int = 7, keep_completed: bool = True) -> int:
        """
        Clean up old jobs.

        Args:
            days: Delete jobs older than this many days
            keep_completed: Keep completed jobs regardless of age

        Returns:
            Number of jobs deleted
        """
        cutoff = datetime.now(UTC) - timedelta(days=days)
        deleted = 0

        for job_info in self._list_operational_jobs():
            job_id = job_info.job_id

            # Skip running jobs
            if job_info.status == JobStatus.RUNNING:
                continue

            # Skip completed jobs if requested
            if keep_completed and job_info.status == JobStatus.COMPLETED:
                continue

            # Check age
            created_at = datetime.fromisoformat(job_info.created_at)
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=UTC)
            if created_at < cutoff:
                try:
                    self.delete_job(job_id)
                    deleted += 1
                except (OSError, ValueError) as e:
                    logger.error("Failed to delete job %s: %s", job_id, e)

        # Also cleanup orphaned executor containers
        try:
            container_manager = get_container_manager()
            if container_manager.is_available():
                orphaned = container_manager.cleanup_orphaned_containers(max_age_hours=days * 24)
                if orphaned > 0:
                    logger.info("Removed %d orphaned executor container(s)", orphaned)
        except Exception as e:
            logger.warning("Failed to cleanup orphaned containers: %s", e)

        logger.info("Cleaned up %d old jobs", deleted)
        return deleted

    def _get_active_job_count(self) -> int:
        """
        Get count of actively running jobs (excluding awaiting_feedback).

        Jobs in AWAITING_FEEDBACK status don't count against the concurrent limit
        so scientists can take unlimited time without blocking the queue.
        """
        if not self._running_jobs:
            return 0
        job_ids = list(self._running_jobs.keys())
        statuses = _run_async(_db_get_job_statuses(job_ids))
        return sum(1 for s in statuses.values() if s != JobStatus.AWAITING_FEEDBACK.value)

    def get_coinvestigate_count(self) -> int:
        """
        Get count of jobs in coinvestigate mode (running or awaiting feedback).

        Used to limit concurrent coinvestigations to prevent resource exhaustion.
        """
        count = 0
        for job_info in self._list_operational_jobs():
            if job_info.investigation_mode == "coinvestigate" and job_info.status in [
                JobStatus.RUNNING,
                JobStatus.AWAITING_FEEDBACK,
                JobStatus.QUEUED,
            ]:
                count += 1
        return count

    def can_start_coinvestigate(self, max_coinvestigate: int = 15) -> bool:
        """
        Check if a new coinvestigate job can be started.

        Args:
            max_coinvestigate: Maximum concurrent coinvestigate jobs (default 15)

        Returns:
            True if under the limit, False otherwise
        """
        return self.get_coinvestigate_count() < max_coinvestigate

    def get_job_summary(self) -> dict[str, Any]:
        """
        Get summary of all jobs (no user filtering).

        Returns:
            Dictionary with job counts and budget info
        """
        jobs = self.list_jobs()

        status_counts = {}
        for status in JobStatus:
            status_counts[status.value] = sum(1 for j in jobs if j.status == status)

        # Get project-level cost info from provider
        try:
            provider = get_provider()
            cost_info = provider.get_cost_info(lookback_hours=24)
            budget_check = provider.evaluate_budget(cost_info)
        except (ValueError, ProviderError) as e:
            logger.warning("Could not fetch cost info: %s", e)
            cost_info = None
            budget_check = None

        return {
            "total_jobs": len(jobs),
            "status_counts": status_counts,
            "cost_info": cost_info,
            "budget_check": budget_check,
        }

    def _list_operational_jobs(
        self,
        status: JobStatus | None = None,
        limit: int | None = None,
    ) -> list[JobInfo]:
        """
        List jobs for operational workflows.

        Jobs are sourced exclusively from the database.
        """
        try:
            db_jobs = _run_async(_db_list_jobs(status=status, limit=limit))
            return [self._db_model_to_job_info(job) for job in db_jobs]
        except Exception as e:
            logger.warning("Failed to list jobs from database for operational scan: %s", e)
            return []

    def _db_model_to_job_info(self, job_model: JobModel) -> JobInfo:
        """Convert a database JobModel to JobInfo with real-time progress from KS."""
        # Load progress from persisted knowledge state for all jobs.
        job_id = str(job_model.id)
        iterations_completed, findings_count = _load_progress_from_knowledge_state(
            job_id=job_id,
            status=job_model.status,
            default_iterations=job_model.current_iteration,
            default_findings=0,
        )

        return JobInfo.from_db_model(job_model, iterations_completed, findings_count)

    def _load_job_info(self, job_id: str) -> JobInfo | None:
        """
        Load job info from database.
        """
        try:
            job_model = _run_async(_db_get_job(job_id))
            if job_model:
                return self._db_model_to_job_info(job_model)
        except Exception as e:
            logger.warning("Failed to load job %s from database: %s", job_id, e)
            return None

        return None

    def _update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        error_message: str | None = None,
        cancellation_reason: str | None = None,
    ) -> None:
        """Update job status in the database."""
        ntfy_result: JobStatusUpdateResult | None = None
        try:
            ntfy_result = _run_async(
                _db_update_job_status(job_id, status, error_message, None, cancellation_reason)
            )
        except Exception as e:
            logger.error("Failed to update job status in database: %s", e)

        if not ntfy_result or not ntfy_result.ntfy_enabled or not ntfy_result.owner_id:
            return

        iterations_completed = _derive_progress_from_db(status.value, ntfy_result.current_iteration)
        try:
            _run_async(
                notify_job_status_change(
                    user_id=UUID(ntfy_result.owner_id),
                    job_id=job_id,
                    job_title=ntfy_result.job_title or "",
                    new_status=status.value,
                    error_message=error_message,
                    cancellation_reason=cancellation_reason,
                    iteration=iterations_completed,
                    ntfy_topic=ntfy_result.ntfy_topic,
                )
            )
        except Exception as e:
            logger.warning("Failed to send ntfy notification: %s", e)


def main() -> None:
    """CLI entry point for job manager."""
    parser = argparse.ArgumentParser(description="OpenScientist Job Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # List jobs
    list_parser = subparsers.add_parser("list", help="List jobs")
    list_parser.add_argument("--status", help="Filter by status")
    list_parser.add_argument("--limit", type=int, help="Limit number of jobs")

    # Get job
    get_parser = subparsers.add_parser("get", help="Get job info")
    get_parser.add_argument("job_id", help="Job ID")

    # Delete job
    delete_parser = subparsers.add_parser("delete", help="Delete job")
    delete_parser.add_argument("job_id", help="Job ID")

    # Cleanup
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old jobs")
    cleanup_parser.add_argument("--days", type=int, default=7, help="Delete jobs older than N days")
    cleanup_parser.add_argument(
        "--delete-completed", action="store_true", help="Delete completed jobs too"
    )

    # Summary
    subparsers.add_parser("summary", help="Get job summary")

    # Bootstrap filesystem jobs into DB
    bootstrap_parser = subparsers.add_parser(
        "bootstrap",
        help="Bootstrap filesystem jobs into the database",
    )
    bootstrap_parser.add_argument(
        "--jobs-dir",
        default="jobs",
        help="Directory containing job folders (default: jobs)",
    )
    bootstrap_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report without writing database changes",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.command == "bootstrap":
        from openscientist.bootstrap import bootstrap_jobs_from_filesystem_sync

        result = bootstrap_jobs_from_filesystem_sync(
            jobs_dir=Path(args.jobs_dir),
            dry_run=args.dry_run,
        )
        print(json.dumps(result.to_dict(), indent=2))
        return

    # Create job manager
    manager = JobManager()

    # Execute command
    if args.command == "list":
        status = JobStatus(args.status) if args.status else None
        jobs = manager.list_jobs(status=status, limit=args.limit)

        print(f"{'Job ID':<20} {'Status':<12} {'Iterations':<12} {'Findings':<10} {'Created At'}")
        print("-" * 80)

        for job in jobs:
            print(
                f"{job.job_id:<20} {job.status.value:<12} "
                f"{job.iterations_completed}/{job.max_iterations:<6} "
                f"{job.findings_count:<10} {job.created_at}"
            )

    elif args.command == "get":
        job_result = manager.get_job(args.job_id)
        if job_result is None:
            print(f"Job {args.job_id} not found")
        else:
            print(json.dumps(job_result.to_dict(), indent=2))

    elif args.command == "delete":
        manager.delete_job(args.job_id)
        print(f"Deleted job {args.job_id}")

    elif args.command == "cleanup":
        deleted = manager.cleanup_old_jobs(days=args.days, keep_completed=not args.delete_completed)
        print(f"Deleted {deleted} jobs")

    elif args.command == "summary":
        summary = manager.get_job_summary()
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
