"""The operator's harness login lives in a home of one container run, outside the
job directory, and goes when the run ends (plans/copied-logins.md)."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from docker import errors as docker_errors
from openscientist.agent.codex_agent import CodexAgent
from openscientist.agent.omp_agent import OmpAgent
from openscientist.job_container.db_login import AgentLogin
from openscientist.job_container.run_home import (
    remove_legacy_job_homes,
    remove_stale_run_homes,
)
from openscientist.job_container.runner import AGENT_APP_DIR, JobContainerRunner
from openscientist.providers.base import AirgapEgress, AirgapPosture

_MARKER = "operator-login-marker"


class _Provider:
    def proxied_container_env(self, **_: Any) -> dict[str, str]:
        return {}

    def prelaunch_model_context_env(self) -> dict[str, str]:
        return {}

    def airgap_egress(self) -> AirgapPosture:
        return AirgapPosture(AirgapEgress.PROXY)


@dataclass
class _Launches:
    jobs_dir: Path
    client: MagicMock
    runner: JobContainerRunner
    closed: MagicMock
    roles: list[str]

    def launch(self, job_id: str) -> tuple[str, dict[str, Any]]:
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        self.runner.launch(job_id, job_dir)
        return self.roles[-1], self.client.containers.run.call_args.kwargs

    def container_of(self, run: dict[str, Any]) -> MagicMock:
        container = MagicMock()
        container.labels = run["labels"]
        return container


def _settings(*, omp: Path | None = None, codex: Path | None = None, provider_id: str) -> Any:
    return SimpleNamespace(
        container=SimpleNamespace(
            host_project_dir=None,
            container_app_dir="/app",
            agent_network=None,
            agent_memory="8g",
            agent_cpu=2.0,
            agent_platform=None,
            agent_image="openscientist-agent:latest",
            agent_timeout=14400,
        ),
        provider=SimpleNamespace(
            provider_id=provider_id,
            omp_auth_host_path=str(omp) if omp else None,
            codex_auth_host_path=str(codex) if codex else None,
            gcp_credentials_host_path=None,
            get_container_env_vars=dict,
        ),
        database=SimpleNamespace(effective_database_url="postgresql+asyncpg://w:p@db:5432/db"),
        phenix=SimpleNamespace(phenix_host_path=None),
        secret_key="master-key",
        airgap=SimpleNamespace(enabled=False),
    )


@pytest.fixture(params=["omp", "codex"])
def harness(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[type, Any, str, str]:
    """A harness configured with an operator login: its agent class, settings,
    the name of its home inside a job, and a file of the copied login."""
    if request.param == "omp":
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "agent.db").write_text(_MARKER)
        (vault / "config.yml").write_text(_MARKER)
        return OmpAgent, _settings(omp=vault, provider_id="vllm"), ".omp-home", "agent.db"
    auth = tmp_path / "auth.json"
    auth.write_text(_MARKER)
    return CodexAgent, _settings(codex=auth, provider_id="openai"), ".codex", "auth.json"


@pytest.fixture
def launches(harness: tuple[type, Any, str, str], tmp_path: Path) -> Iterator[_Launches]:
    agent_class, settings, _, _ = harness
    roles: list[str] = []

    def open_login(job_id: str, *, lifetime_seconds: int) -> AgentLogin:
        roles.append(f"agent_{uuid4().hex}")
        return AgentLogin(role=roles[-1], database_url="postgresql+asyncpg://a:b@db:5432/db")

    client = MagicMock()
    with (
        patch("openscientist.job_container.runner.docker.from_env", return_value=client),
        patch("openscientist.job_container.runner.get_settings", return_value=settings),
        patch("openscientist.job_container.runner.get_provider", return_value=_Provider()),
        patch("openscientist.job_container.runner.open_agent_login", side_effect=open_login),
        patch("openscientist.job_container.runner.close_agent_login") as closed,
        patch.object(JobContainerRunner, "_get_network", return_value="bridge"),
        patch("openscientist.agent.factory.agent_class_for_provider_id", return_value=agent_class),
    ):
        yield _Launches(tmp_path / "jobs", client, JobContainerRunner(), closed, roles)


def _home_of(run: dict[str, Any], job_id: str, home_name: str) -> Path:
    """The host directory mounted as the harness's home in ``run``."""
    target = f"{AGENT_APP_DIR}/jobs/{job_id}/{home_name}"
    sources = [src for src, mount in run["volumes"].items() if mount["bind"] == target]
    assert len(sources) == 1, run["volumes"]
    assert run["volumes"][sources[0]]["mode"] == "rw"
    return Path(sources[0])


def test_a_runs_login_is_mounted_from_its_own_home_not_the_job_directory(
    launches: _Launches, harness: tuple[type, Any, str, str]
) -> None:
    _, _, home_name, login_file = harness
    _, run = launches.launch("job-1")
    home = _home_of(run, "job-1", home_name)
    assert (home / login_file).read_text() == _MARKER
    assert not home.is_relative_to(launches.jobs_dir / "job-1")
    copied = [p for p in (launches.jobs_dir / "job-1").rglob("*") if p.is_file()]
    assert copied == []


def test_overlapping_runs_have_separate_homes_and_end_separately(
    launches: _Launches, harness: tuple[type, Any, str, str]
) -> None:
    _, _, home_name, login_file = harness
    _, first = launches.launch("job-1")
    _, second = launches.launch("job-1")
    first_home, second_home = (
        _home_of(first, "job-1", home_name),
        _home_of(second, "job-1", home_name),
    )
    assert first_home != second_home
    launches.client.containers.list.return_value = [launches.container_of(first)]
    launches.runner.cleanup("job-1")
    assert not first_home.exists()
    assert (second_home / login_file).read_text() == _MARKER


def test_a_cleanup_by_a_fresh_runner_removes_the_runs_home(
    launches: _Launches, harness: tuple[type, Any, str, str]
) -> None:
    """Startup cleanup after a restart finds the home from the container alone."""
    _, _, home_name, _ = harness
    _, run = launches.launch("job-1")
    home = _home_of(run, "job-1", home_name)
    launches.client.containers.list.return_value = [launches.container_of(run)]
    JobContainerRunner().cleanup("job-1")
    assert not home.exists()


def test_a_launch_that_fails_removes_the_runs_home(
    launches: _Launches, harness: tuple[type, Any, str, str]
) -> None:
    launches.client.containers.run.side_effect = docker_errors.APIError("no such image")
    with pytest.raises(docker_errors.APIError):
        launches.launch("job-1")
    homes = launches.jobs_dir / ".run-homes"
    assert not homes.exists() or not any(homes.iterdir())
    launches.closed.assert_called_once_with(launches.roles[-1])


def test_a_chat_turn_removes_its_home_when_it_ends(
    launches: _Launches, harness: tuple[type, Any, str, str]
) -> None:
    _, _, home_name, _ = harness
    job_dir = launches.jobs_dir / "job-1"
    job_dir.mkdir(parents=True)
    chat = MagicMock()
    chat.wait.return_value = {"StatusCode": 0}

    def started(**kwargs: Any) -> MagicMock:
        chat.labels = kwargs["labels"]
        chat.run_kwargs = kwargs
        return chat

    launches.client.containers.run.side_effect = started
    launches.runner.run_chat_turn("job-1", job_dir, "t1")
    assert not _home_of(chat.run_kwargs, "job-1", home_name).exists()


def _aged(path: Path, age: timedelta) -> None:
    then = time.time() - age.total_seconds()
    os.utime(path, (then, then))


def test_the_reaper_removes_homes_no_container_needs(tmp_path: Path) -> None:
    homes = tmp_path / ".run-homes"
    for role in ("agent_live", "agent_gone", "agent_new"):
        (homes / role).mkdir(parents=True)
        (homes / role / "auth.json").write_text(_MARKER)
    _aged(homes / "agent_live", timedelta(hours=2))
    _aged(homes / "agent_gone", timedelta(hours=2))
    grace = timedelta(minutes=15)
    assert remove_stale_run_homes(tmp_path, None, older_than=grace) == []
    assert remove_stale_run_homes(tmp_path, {"agent_live"}, older_than=grace) == ["agent_gone"]
    assert sorted(p.name for p in homes.iterdir()) == ["agent_live", "agent_new"]


def test_legacy_copies_leave_jobs_that_no_container_runs(tmp_path: Path) -> None:
    for job in ("done", "running"):
        for name in (".codex/auth.json", ".omp-home/agent.db", "final_report.md"):
            (tmp_path / job / name).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / job / name).write_text(_MARKER)
    assert remove_legacy_job_homes(tmp_path, None) == []
    assert remove_legacy_job_homes(tmp_path, {"running"}) == ["done"]
    assert sorted(p.name for p in (tmp_path / "done").iterdir()) == ["final_report.md"]
    assert (tmp_path / "running" / ".codex" / "auth.json").exists()
    assert (tmp_path / "running" / ".omp-home" / "agent.db").exists()
