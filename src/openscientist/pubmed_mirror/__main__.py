"""Build-time loader: download the MEDLINE baseline and load it into Postgres.

python -m openscientist.pubmed_mirror                  # full baseline
python -m openscientist.pubmed_mirror --limit-files 1  # smoke test
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from openscientist.pubmed_mirror.download import download_baseline
from openscientist.pubmed_mirror.ingest import ingest_files
from openscientist.settings import MigrationSettings


def loader_dsn(settings: MigrationSettings) -> str:
    """Ingest drops and rebuilds the search index, which only the table owner may do."""
    return settings.required_owner_database_url()


def main() -> None:
    parser = argparse.ArgumentParser(description="Load the PubMed baseline into Postgres.")
    parser.add_argument("--data-dir", default="pubmed_baseline", help="Where to stage downloads")
    parser.add_argument("--limit-files", type=int, default=None, help="Download at most N files")
    parser.add_argument("--max-articles", type=int, default=None, help="Load at most N articles")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    files = download_baseline(Path(args.data_dir), limit=args.limit_files)
    total = asyncio.run(
        ingest_files(
            files,
            dsn=loader_dsn(MigrationSettings()),
            max_articles=args.max_articles,
        )
    )
    print(f"Loaded {total} articles from {len(files)} file(s).")


if __name__ == "__main__":
    main()
