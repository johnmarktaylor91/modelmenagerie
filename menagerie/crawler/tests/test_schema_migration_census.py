"""IC-9 census half: the schema-migration census fires on schema movement.

The census validates every finalized transaction's RECORDED author result
against the candidate tree's live author-result schema, so a tightening shows
up as a pre-launch count instead of a 100+-paid-session re-author storm. It
is consumed at every wave boundary, the Wave-2 batch procedure, and the
first-generation Wave-1 boundary run; AB-30's mixed-era reader matrix is its
pairing consumer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from menagerie.crawler.artifact_transactions import ArtifactEventLedger
from menagerie.crawler.constants import AUTHOR_PROPOSAL_SCHEMA_VERSION_V3
from menagerie.crawler.tests.test_artifact_transactions import (
    _commit_private,
    _context,
    _mirrors,
)
from menagerie.crawler.tools.schema_migration_census import (
    main,
    schema_migration_census,
)

pytestmark = pytest.mark.smoke


def _store(tmp_path: Path):
    repo = tmp_path / "repo"
    canonical = repo / "menagerie" / "crawler"
    ledger_path = canonical / "records" / "artifacts" / "shard.jsonl"
    context = _context("m_one")
    mirrors = _mirrors(tmp_path)
    with ArtifactEventLedger(ledger_path) as ledger:
        _commit_private(
            "m_one",
            content=b"census bytes",
            context=context,
            mirrors=mirrors,
            ledger=ledger,
            canonical_root=canonical,
            repository_root=repo,
            evidence_id="ev-one",
        )
    return ledger_path, repo


def test_live_schema_census_reads_valid(tmp_path: Path) -> None:
    """A store recorded under the live schema reads all schema-valid."""

    ledger_path, repo = _store(tmp_path)
    rows = schema_migration_census((ledger_path,), repository_root=repo)
    assert [row["disposition"] for row in rows] == ["schema-valid"]
    assert main(
        ["--artifact-shard", str(ledger_path), "--repository-root", str(repo), "--gate"]
    ) == 0


def test_schema_movement_fires_the_census_and_the_gate(tmp_path: Path) -> None:
    """Validating recorded results against a moved schema reports every
    would-be-stale head, and the gate arm exits nonzero: a nonzero census is
    an explicit migration decision, never an ambient re-author storm."""

    ledger_path, repo = _store(tmp_path)
    rows = schema_migration_census(
        (ledger_path,),
        repository_root=repo,
        schema_version=AUTHOR_PROPOSAL_SCHEMA_VERSION_V3,
    )
    assert [row["disposition"] for row in rows] == ["schema-stale"]
    assert rows[0]["error"]
    exit_code = main(
        [
            "--artifact-shard",
            str(ledger_path),
            "--repository-root",
            str(repo),
            "--schema-version",
            AUTHOR_PROPOSAL_SCHEMA_VERSION_V3,
            "--gate",
        ]
    )
    assert exit_code == 2


def test_torn_anchor_reads_unreadable_and_gates(tmp_path: Path) -> None:
    """A finalized event whose reconstruction bytes contradict their anchor
    reads reconstruction-unreadable (disclosed, gate-refusing), never valid."""

    ledger_path, repo = _store(tmp_path)
    documents = list((repo / "menagerie" / "crawler" / "reconstruction").rglob("*.json"))
    assert documents
    documents[0].write_bytes(documents[0].read_bytes() + b" ")
    rows = schema_migration_census((ledger_path,), repository_root=repo)
    assert [row["disposition"] for row in rows] == ["reconstruction-unreadable"]
