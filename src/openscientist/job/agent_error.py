"""The error an agent container leaves for the web when it exits with a failure.

The web's worker sees only the container's exit code, so the container writes
its final error into the job directory, which both sides mount."""

from __future__ import annotations

from pathlib import Path

_AGENT_ERROR_FILE = "agent-error.txt"


def write_agent_error(job_dir: Path, error: str) -> None:
    (job_dir / _AGENT_ERROR_FILE).write_text(error, encoding="utf-8")


def take_agent_error(job_dir: Path) -> str | None:
    """The error the container left in ``job_dir``, removed so a later run
    cannot report it again."""
    path = job_dir / _AGENT_ERROR_FILE
    try:
        error = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    path.unlink(missing_ok=True)
    return error or None
