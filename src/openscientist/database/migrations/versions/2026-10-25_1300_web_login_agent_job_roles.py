"""Let the web login map, list and unmap agent logins.

docker/postgres/bootstrap.sql creates openscientist_web before migrations run.

Revision ID: web_login_agent_job_roles
Revises: agent_role_policies
Create Date: 2026-10-25 13:00:00
"""

from collections.abc import Sequence

from alembic import op

revision: str = "web_login_agent_job_roles"
down_revision: str | None = "agent_role_policies"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("GRANT SELECT, INSERT, DELETE ON agent_job_roles TO openscientist_web")


def downgrade() -> None:
    op.execute("REVOKE ALL ON agent_job_roles FROM openscientist_web")
