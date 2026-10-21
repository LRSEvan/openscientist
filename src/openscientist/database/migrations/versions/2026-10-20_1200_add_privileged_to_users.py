"""Add privileged flag to users.

Revision ID: add_privileged
Revises: add_version_info
Create Date: 2026-10-20 12:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "add_privileged"
down_revision: str | None = "add_version_info"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "privileged",
            sa.Boolean(),
            nullable=False,
            server_default="false",
            comment="Whether user's queued jobs are always run ahead of regular users",
        ),
    )


def downgrade() -> None:
    op.drop_column("users", "privileged")
