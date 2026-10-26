"""A run's cost is written as it goes, so a stop or a failed write loses little
(plans/cost-record-loss.md)."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from uuid import UUID

import pytest
import pytest_asyncio
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from openscientist.agent.base import TokenUsage
from openscientist.database.models import CostRecord, Job
from openscientist.orchestrator.discovery import _CostLedger
from tests.db_outage import Forwarder


@pytest_asyncio.fixture
async def job(test_engine: AsyncEngine) -> AsyncGenerator[UUID, None]:
    async with AsyncSession(test_engine, expire_on_commit=False) as session:
        row = Job(research_question="Q?", status="running")
        session.add(row)
        await session.commit()
    yield row.id
    async with AsyncSession(test_engine) as session:
        await session.execute(delete(Job).where(Job.id == row.id))
        await session.commit()


async def _records(test_engine: AsyncEngine, job: UUID) -> list[tuple[str, int | None, int, int]]:
    async with test_engine.connect() as conn:
        rows = await conn.execute(
            select(
                CostRecord.operation_type,
                CostRecord.iteration,
                CostRecord.input_tokens,
                CostRecord.output_tokens,
            )
            .where(CostRecord.job_id == job)
            .order_by(CostRecord.created_at)
        )
    return [tuple(row) for row in rows]


def _ledger(job: UUID, usage: TokenUsage) -> _CostLedger:
    return _CostLedger(
        str(job), usage=lambda: usage, provider_name="Anthropic", model_name="claude-sonnet-4-5"
    )


async def test_each_record_holds_only_its_own_tokens(test_engine: AsyncEngine, job: UUID) -> None:
    usage = TokenUsage()  # grows in place, as an agent's running total does
    ledger = _ledger(job, usage)
    ledger.begin("discovery", 1)
    usage += TokenUsage(input_tokens=100, output_tokens=10)
    await ledger.record()
    ledger.begin("discovery", 2)
    usage += TokenUsage(input_tokens=40, output_tokens=4)
    await ledger.record()
    ledger.begin("report", None)
    await ledger.close()
    assert await _records(test_engine, job) == [("discovery", 1, 100, 10), ("discovery", 2, 40, 4)]


async def test_a_failed_write_carries_its_tokens_to_the_next(
    database: Forwarder, test_engine: AsyncEngine, job: UUID
) -> None:
    usage = TokenUsage()
    ledger = _ledger(job, usage)
    ledger.begin("discovery", 1)
    usage += TokenUsage(input_tokens=100, output_tokens=10)
    database.down()
    await ledger.record()
    database.up()
    assert await _records(test_engine, job) == []
    ledger.begin("report", None)
    usage += TokenUsage(input_tokens=40, output_tokens=4)
    await ledger.record()
    assert await _records(test_engine, job) == [("report", None, 140, 14)]


async def test_a_failed_last_write_logs_every_count(
    database: Forwarder, job: UUID, caplog: pytest.LogCaptureFixture
) -> None:
    usage = TokenUsage()
    ledger = _ledger(job, usage)
    ledger.begin("discovery", 3)
    usage += TokenUsage(
        input_tokens=11,
        output_tokens=22,
        cache_write_tokens=33,
        cache_write_1h_tokens=44,
        cache_read_tokens=55,
        reasoning_tokens=66,
    )
    database.down()
    with caplog.at_level(logging.ERROR):
        await ledger.close()
    [message] = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    for count in ("input=11", "output=22", "cache_write=33", "cache_write_1h=44"):
        assert count in message
    for count in ("cache_read=55", "reasoning=66", "claude-sonnet-4-5", str(job)):
        assert count in message
