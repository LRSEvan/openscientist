"""Regression test: the ingest drops and rebuilds the search index, so it must
run as the schema owner, not the DML-only ADMIN_DATABASE_URL role that
docker-compose configures.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

import asyncpg  # type: ignore[import-untyped]
import pytest
import pytest_asyncio
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine

from openscientist.pubmed_mirror.__main__ import loader_dsn
from openscientist.pubmed_mirror.ingest import ingest_files
from openscientist.settings import MigrationSettings
from tests.db_logins import ADMIN_PASSWORD, OWNER_PASSWORD, login_url

_SAMPLE_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>901</PMID>
      <Article>
        <Journal>
          <Title>Journal of Fish Biology</Title>
          <JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue>
        </Journal>
        <ArticleTitle>Zebrafish thermal tolerance</ArticleTitle>
        <Abstract><AbstractText>Warm water acclimation.</AbstractText></Abstract>
        <AuthorList><Author><LastName>Doe</LastName></Author></AuthorList>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>902</PMID>
      <Article>
        <Journal>
          <JournalIssue><PubDate><Year>2023</Year></PubDate></JournalIssue>
        </Journal>
        <ArticleTitle>Soil microbiome depth gradients</ArticleTitle>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


@pytest.fixture
def baseline_file(tmp_path: Path) -> Path:
    """A tiny two-article baseline file (name encodes baseline year 2026)."""
    path = tmp_path / "pubmed26n0001.xml"
    path.write_text(_SAMPLE_XML)
    return path


@pytest.fixture
def admin_login_url(test_engine: AsyncEngine, test_database_url: str) -> str:
    """LOGIN DSN for openscientist_admin, as docker-compose configures it."""
    return login_url(test_database_url, "openscientist_admin", ADMIN_PASSWORD)


@pytest_asyncio.fixture
async def clean_pubmed_tables(test_engine: AsyncEngine) -> AsyncGenerator[None, None]:
    """Leave the session-shared corpus tables empty for later tests."""
    yield
    async with test_engine.begin() as conn:
        await conn.execute(text("TRUNCATE pubmed_articles, pubmed_corpus_meta"))


@pytest.mark.asyncio
async def test_ingest_over_loader_dsn_rebuilds_search_index(
    test_database_url: str,
    baseline_file: Path,
    test_engine: AsyncEngine,
    clean_pubmed_tables: None,
) -> None:
    """The DSN main() hands to ingest_files is the owner login, with the password
    filled in from POSTGRES_OWNER_PASSWORD as the migration step passes it."""
    owner_url = make_url(test_database_url).set(username="openscientist_owner", password=None)
    settings = MigrationSettings(
        OWNER_DATABASE_URL=owner_url.render_as_string(hide_password=False),
        POSTGRES_OWNER_PASSWORD=SecretStr(OWNER_PASSWORD),
    )

    total = await ingest_files([baseline_file], dsn=loader_dsn(settings))

    assert total == 2
    async with test_engine.connect() as conn:
        assert await conn.scalar(text("SELECT count(*) FROM pubmed_articles")) == 2
        assert (
            await conn.scalar(
                text(
                    "SELECT count(*) FROM pg_indexes"
                    " WHERE tablename = 'pubmed_articles'"
                    " AND indexname = 'ix_pubmed_articles_search_vector'"
                )
            )
            == 1
        )
        meta = (
            await conn.execute(
                text(
                    "SELECT baseline_year, article_count FROM pubmed_corpus_meta"
                    " ORDER BY id DESC LIMIT 1"
                )
            )
        ).one()
        assert tuple(meta) == (2026, 2)


@pytest.mark.asyncio
async def test_ingest_over_admin_dsn_reproduces_clean_install_failure(
    admin_login_url: str,
    baseline_file: Path,
    clean_pubmed_tables: None,
) -> None:
    """The admin role cannot DROP the owner's index, so ingesting over
    ADMIN_DATABASE_URL fails. Proves the sibling test is not vacuous."""
    with pytest.raises(asyncpg.exceptions.InsufficientPrivilegeError):
        await ingest_files([baseline_file], dsn=admin_login_url)
