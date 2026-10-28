"""Tests for new job page helpers."""

import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from openscientist.webapp_components.pages.new_job import _build_upload_session_id, _submit_job


def test_build_upload_session_id_uses_user_and_client_id():
    """Upload session IDs should be scoped by user and websocket client."""
    client = SimpleNamespace(id="client-abc")
    session_id = _build_upload_session_id("user-123", client)
    assert session_id == "user-123:client-abc"


def test_build_upload_session_id_handles_missing_user_with_anonymous_prefix():
    """Anonymous fallback should still include client identity."""
    client = SimpleNamespace(id="client-xyz")
    session_id = _build_upload_session_id(None, client)
    assert session_id == "anonymous:client-xyz"


def test_submit_job_has_use_hypotheses_parameter():
    """_submit_job must accept use_hypotheses so the form toggle is wired in."""
    sig = inspect.signature(_submit_job)
    assert "use_hypotheses" in sig.parameters


def test_submit_job_has_coinvestigate_mode_at_top_level():
    """coinvestigate_mode must be a top-level parameter of _submit_job."""
    sig = inspect.signature(_submit_job)
    assert "coinvestigate_mode" in sig.parameters


async def test_submit_job_creates_the_job_and_opens_its_page() -> None:
    """Submitting a question creates the job for the signed-in user and
    navigates them to the job's page."""
    from openscientist.webapp_components.pages import new_job

    job_manager = SimpleNamespace(create_job=AsyncMock())
    form = SimpleNamespace(
        research_question=SimpleNamespace(value="  Why do corals bleach?  "),
        max_iterations=SimpleNamespace(value=12),
        use_hypotheses=SimpleNamespace(value=True),
        coinvestigate_mode=SimpleNamespace(value=False),
    )
    with (
        patch.object(new_job, "ui") as mock_ui,
        patch.object(new_job, "get_current_user_id", return_value="user-123"),
        patch.object(new_job, "get_uploaded_files", return_value=[]),
        patch.object(new_job, "uuid") as mock_uuid,
    ):
        mock_uuid.uuid4.return_value = "00000000-1111-2222-3333-444444444444"
        await new_job._submit_job(
            job_manager=job_manager,
            user_can_start_jobs=True,
            session_id="user-123:client-1",
            research_question=form.research_question,
            max_iterations=form.max_iterations,
            use_hypotheses=form.use_hypotheses,
            coinvestigate_mode=form.coinvestigate_mode,
        )

    job_manager.create_job.assert_awaited_once()
    assert job_manager.create_job.await_args.kwargs == {
        "job_id": "00000000-1111-2222-3333-444444444444",
        "research_question": "Why do corals bleach?",
        "data_files": [],
        "max_iterations": 12,
        "use_hypotheses": True,
        "auto_start": True,
        "investigation_mode": "autonomous",
        "owner_id": "user-123",
    }
    mock_ui.navigate.to.assert_called_with("/job/00000000-1111-2222-3333-444444444444")
