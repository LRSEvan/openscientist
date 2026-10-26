"""Secrets for the untrusted agent container: its derived settings key, and the
credentials one container run presents to the LLM proxy and the execution broker."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
from dataclasses import dataclass

_JOB_SECRET_LABEL = "job_secret:"
_SEP = "."


def derive_job_secret(master_key: str, job_id: str) -> str:
    """Return HMAC-SHA256(master_key, "job_secret:" + job_id) as a 64-char hex digest.

    Keeps the master key out of the job container while satisfying the required
    Settings.secret_key field inside it.
    """
    message = f"{_JOB_SECRET_LABEL}{job_id}".encode()
    return hmac.new(master_key.encode(), message, hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class RunCredentials:
    """What one container run presents to the LLM proxy and to the execution
    broker, each ``"<job_id>.<random token>"``."""

    llm_placeholder: str
    exec_token: str


# Kept by the web process, which runs the proxy, the broker and every container
# launch, so a restart, which ends every run, forgets them all.
_runs: dict[str, RunCredentials] = {}
_runs_lock = threading.Lock()


def issue_run_credentials(job_id: str, role: str) -> RunCredentials:
    """New credentials for the run of ``job_id`` whose database login is ``role``,
    accepted until :func:`revoke_run_credentials` ends them."""
    credentials = RunCredentials(
        llm_placeholder=f"{job_id}{_SEP}{secrets.token_hex(32)}",
        exec_token=f"{job_id}{_SEP}{secrets.token_hex(32)}",
    )
    with _runs_lock:
        _runs[role] = credentials
    return credentials


def revoke_run_credentials(role: str) -> None:
    """End the credentials of the run whose database login is ``role``."""
    with _runs_lock:
        _runs.pop(role, None)


def _live_runs() -> list[RunCredentials]:
    with _runs_lock:
        return list(_runs.values())


def _same(expected: str, presented: str) -> bool:
    return hmac.compare_digest(expected.encode(), presented.encode())


def proxy_accepts(placeholder: str) -> bool:
    """Whether ``placeholder`` is the LLM credential of a live run."""
    return any(_same(run.llm_placeholder, placeholder) for run in _live_runs())


def broker_accepts(job_id: str, token: str) -> bool:
    """Whether ``token`` is the execution credential of a live run of ``job_id``."""
    if token.rpartition(_SEP)[0] != job_id:
        return False
    return any(_same(run.exec_token, token) for run in _live_runs())
