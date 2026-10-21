"""Add (updated_at, owner_id) index to speed up the recency-bounded
fairness query.

Only useful alongside the matching _LAST_SERVED_LOOKBACK bound in
job_manager.py: once the last_served aggregate is bounded to a recent
window, updated_at > cutoff becomes a genuinely selective filter, unlike
a plain status filter (status NOT IN ('queued', 'pending') matches nearly
every row, so no index on it can help). owner_id is included so the
GROUP BY owner_id right after the range filter can be satisfied from the
index itself rather than a separate heap lookup per matching row.

Revision ID: add_updated_at_idx
Revises: add_privileged
Create Date: 2026-10-20 13:30:00
"""

from collections.abc import Sequence

from alembic import op

revision: str = "add_updated_at_idx"
down_revision: str | None = "add_privileged"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index("ix_jobs_updated_at_owner_id", "jobs", ["updated_at", "owner_id"])


def downgrade() -> None:
    op.drop_index("ix_jobs_updated_at_owner_id", table_name="jobs")
