"""Set the openscientist_admin password from POSTGRES_ADMIN_PASSWORD.

This revision runs once, so a later change to POSTGRES_ADMIN_PASSWORD needs an
ALTER ROLE by hand.

Revision ID: admin_password_from_env
Revises: add_version_info
Create Date: 2026-10-24 12:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from openscientist.settings import DatabaseSettings

revision: str = "admin_password_from_env"
down_revision: str | None = "add_version_info"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    password = DatabaseSettings().postgres_admin_password
    if password is None or not password.get_secret_value():
        raise RuntimeError(
            "POSTGRES_ADMIN_PASSWORD is not set. It becomes the openscientist_admin "
            "password and must match the one in ADMIN_DATABASE_URL."
        )
    # ALTER ROLE takes no bind parameters, so the server quotes the value.
    bind = op.get_bind()
    statement = bind.execute(
        sa.text(
            "SELECT format('ALTER ROLE openscientist_admin WITH PASSWORD %L', "
            "CAST(:password AS text))"
        ),
        {"password": password.get_secret_value()},
    ).scalar_one()
    bind.exec_driver_sql(statement)


def downgrade() -> None:
    # The previous password is not recoverable, so the role keeps the current one.
    pass
