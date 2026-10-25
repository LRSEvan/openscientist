"""Add the openscientist_agent role, the per-job login mapping and its policies.

Each agent container will log in as its own role, a member of
openscientist_agent, mapped to its job in agent_job_roles. The grants cover
exactly what the container reads and writes.

The existing policies had no TO clause, so they applied to every role without
BYPASSRLS, a job login included. They trust app.current_user_id, which any login
can set for itself, and some read tables a job login may not. They are narrowed
to openscientist_app, the one role they governed. Every job table also gets a
restrictive policy, so a policy added later without a TO clause cannot widen a
job login's rows.

Revision ID: agent_role_policies
Revises: admin_password_from_env
Create Date: 2026-10-25 12:00:00
"""

from collections.abc import Sequence

from alembic import op

revision: str = "agent_role_policies"
down_revision: str | None = "admin_password_from_env"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_ROLE = "openscientist_agent"

# Table, job key column, privileges. INSERT covers the whole row, because the ORM
# writes every mapped column and the policies pin job_id. UPDATE lists only what
# the container changes, so owner, job and natural keys stay fixed. SELECT also
# lets the ORM read generated ids back with RETURNING.
_JOB_TABLES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "jobs",
        "id",
        (
            "SELECT",
            "UPDATE (status, error_message, version_info, current_iteration, data_summary, "
            "agent_status, agent_status_updated_at, consensus_answer, short_title, updated_at)",
        ),
    ),
    ("job_data_files", "job_id", ("SELECT",)),
    (
        "hypotheses",
        "job_id",
        ("SELECT", "INSERT", "UPDATE (status, test_strategy, supporting_evidence, updated_at)"),
    ),
    ("findings", "job_id", ("SELECT", "INSERT", "UPDATE (finding_type, data, updated_at)")),
    (
        "literature",
        "job_id",
        (
            "SELECT",
            "INSERT",
            "UPDATE (abstract, authors, journal, year, doi, extra_metadata, updated_at)",
        ),
    ),
    ("analysis_log", "job_id", ("SELECT", "INSERT")),
    (
        "iteration_summaries",
        "job_id",
        ("SELECT", "INSERT", "UPDATE (summary_text, strapline, updated_at)"),
    ),
    ("feedback_history", "job_id", ("SELECT", "INSERT")),
    ("cost_records", "job_id", ("SELECT", "INSERT")),
)

# Shared reference data the container reads.
_READ_ONLY_TABLES = ("skills", "pubmed_articles")


def upgrade() -> None:
    op.execute(
        f"""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '{_ROLE}') THEN
                CREATE ROLE {_ROLE} NOLOGIN;
            END IF;
        END
        $$
        """
    )
    op.execute(f"GRANT USAGE ON SCHEMA public TO {_ROLE}")

    op.execute(
        """
        DO $$
        DECLARE p record;
        BEGIN
            FOR p IN SELECT policyname, tablename FROM pg_catalog.pg_policies
                     WHERE schemaname = 'public' AND roles = '{public}'
            LOOP
                EXECUTE format('ALTER POLICY %I ON public.%I TO openscientist_app',
                               p.policyname, p.tablename);
            END LOOP;
        END
        $$
        """
    )

    op.execute(
        """
        CREATE TABLE agent_job_roles (
            role_name name PRIMARY KEY,
            job_id uuid NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
            created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    # One login per container, so a chat turn's login and the discovery login
    # can map to the same job at once.
    op.execute("CREATE INDEX ix_agent_job_roles_job_id ON agent_job_roles (job_id)")
    op.execute(
        "COMMENT ON TABLE agent_job_roles IS "
        "'Database login of each agent container, and the job it may reach'"
    )
    # The default privileges grant every new table to openscientist_app. Web
    # users have no business with the mapping. openscientist_admin keeps it.
    op.execute("REVOKE ALL ON agent_job_roles FROM openscientist_app")

    # SECURITY DEFINER, because the agent may not read agent_job_roles. Inside it
    # current_user is the function's owner, so the job login is session_user,
    # which the agent cannot change.
    op.execute(
        """
        CREATE FUNCTION agent_job_id() RETURNS uuid
        LANGUAGE sql STABLE SECURITY DEFINER
        SET search_path = pg_catalog, pg_temp
        AS $$
            SELECT job_id FROM public.agent_job_roles WHERE role_name = session_user
        $$
        """
    )
    op.execute("REVOKE ALL ON FUNCTION agent_job_id() FROM PUBLIC")
    op.execute(f"GRANT EXECUTE ON FUNCTION agent_job_id() TO {_ROLE}")

    for table, key, privileges in _JOB_TABLES:
        for privilege in privileges:
            op.execute(f"GRANT {privilege} ON {table} TO {_ROLE}")
        own_job = f"{key} = agent_job_id()"
        op.execute(
            f"CREATE POLICY {table}_agent_job ON {table} FOR ALL TO {_ROLE} "
            f"USING ({own_job}) WITH CHECK ({own_job})"
        )
        op.execute(
            f"CREATE POLICY {table}_agent_only_own_job ON {table} AS RESTRICTIVE "
            f"FOR ALL TO {_ROLE} USING ({own_job}) WITH CHECK ({own_job})"
        )

    for table in _READ_ONLY_TABLES:
        op.execute(f"GRANT SELECT ON {table} TO {_ROLE}")
    # skills_select_all requires app.current_user_id, which a job login has none of.
    op.execute(
        f"CREATE POLICY skills_agent_enabled ON skills FOR SELECT TO {_ROLE} USING (is_enabled)"
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS skills_agent_enabled ON skills")
    for table in _READ_ONLY_TABLES:
        op.execute(f"REVOKE ALL ON {table} FROM {_ROLE}")
    for table, _key, _privileges in reversed(_JOB_TABLES):
        op.execute(f"DROP POLICY IF EXISTS {table}_agent_only_own_job ON {table}")
        op.execute(f"DROP POLICY IF EXISTS {table}_agent_job ON {table}")
        op.execute(f"REVOKE ALL ON {table} FROM {_ROLE}")
    op.execute("DROP FUNCTION IF EXISTS agent_job_id()")
    op.execute("DROP TABLE IF EXISTS agent_job_roles")
    op.execute(f"REVOKE USAGE ON SCHEMA public FROM {_ROLE}")
    op.execute(f"DROP ROLE IF EXISTS {_ROLE}")
    op.execute(
        """
        DO $$
        DECLARE p record;
        BEGIN
            FOR p IN SELECT policyname, tablename FROM pg_catalog.pg_policies
                     WHERE schemaname = 'public' AND roles = '{openscientist_app}'
            LOOP
                EXECUTE format('ALTER POLICY %I ON public.%I TO PUBLIC',
                               p.policyname, p.tablename);
            END LOOP;
        END
        $$
        """
    )
