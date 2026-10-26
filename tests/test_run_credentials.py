"""A container run's LLM proxy and execution broker credentials work only while
that run lives (plans/token-findings.md)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from uuid import uuid4

import pytest

from openscientist.job_container.secrets import (
    RunCredentials,
    broker_accepts,
    issue_run_credentials,
    proxy_accepts,
    revoke_run_credentials,
)

StartRun = Callable[[str], tuple[str, RunCredentials]]


@pytest.fixture
def start_run() -> Iterator[StartRun]:
    """Start a run of a job, returning its login role and credentials."""
    roles: list[str] = []

    def start(job_id: str) -> tuple[str, RunCredentials]:
        role = f"agent_{uuid4().hex}"
        roles.append(role)
        return role, issue_run_credentials(job_id, role)

    yield start
    for role in roles:
        revoke_run_credentials(role)


def _accepted(job_id: str, credentials: RunCredentials) -> tuple[bool, bool]:
    return (
        proxy_accepts(credentials.llm_placeholder),
        broker_accepts(job_id, credentials.exec_token),
    )


def test_a_runs_credentials_end_with_it(start_run: StartRun) -> None:
    role, credentials = start_run("job-1")
    assert _accepted("job-1", credentials) == (True, True)
    revoke_run_credentials(role)
    assert _accepted("job-1", credentials) == (False, False)


def test_an_earlier_run_of_the_same_job_stays_refused(start_run: StartRun) -> None:
    first_role, first = start_run("job-1")
    revoke_run_credentials(first_role)
    _, second = start_run("job-1")
    assert _accepted("job-1", first) == (False, False)
    assert _accepted("job-1", second) == (True, True)


def test_two_live_runs_of_one_job_end_separately(start_run: StartRun) -> None:
    chat_role, chat = start_run("job-1")
    _, discovery = start_run("job-1")
    revoke_run_credentials(chat_role)
    assert _accepted("job-1", chat) == (False, False)
    assert _accepted("job-1", discovery) == (True, True)


def test_a_broker_token_works_only_for_its_own_job(start_run: StartRun) -> None:
    _, credentials = start_run("job-1")
    assert not broker_accepts("job-2", credentials.exec_token)


def test_credentials_made_up_for_a_job_are_refused() -> None:
    made_up = "job-1." + "0" * 64
    assert not proxy_accepts(made_up)
    assert not broker_accepts("job-1", made_up)
