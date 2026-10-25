"""Let a user's session save their own notification settings.

users had only a SELECT policy, so the notification toggle, which updates the
user's row through openscientist_app, matched no rows and saved nothing. The
update policy opens the user's own row. A policy cannot limit columns, so the
app role's UPDATE on users narrows to the notification columns, and a user
session still cannot change its own approval or other fields.

Revision ID: users_update_own_ntfy
Revises: web_login_agent_job_roles
Create Date: 2026-10-25 14:00:00
"""

from collections.abc import Sequence

from alembic import op

revision: str = "users_update_own_ntfy"
down_revision: str | None = "web_login_agent_job_roles"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_OWN_ROW = """
    current_setting('app.current_user_id', TRUE) IS NOT NULL
    AND id::text = current_setting('app.current_user_id', TRUE)
"""


def upgrade() -> None:
    op.execute(
        f"CREATE POLICY users_update_own ON users FOR UPDATE TO openscientist_app "
        f"USING ({_OWN_ROW}) WITH CHECK ({_OWN_ROW})"
    )
    op.execute("REVOKE UPDATE ON users FROM openscientist_app")
    # updated_at because the ORM sets it on every update.
    op.execute("GRANT UPDATE (ntfy_enabled, ntfy_topic, updated_at) ON users TO openscientist_app")


def downgrade() -> None:
    op.execute(
        "REVOKE UPDATE (ntfy_enabled, ntfy_topic, updated_at) ON users FROM openscientist_app"
    )
    op.execute("GRANT UPDATE ON users TO openscientist_app")
    op.execute("DROP POLICY IF EXISTS users_update_own ON users")
