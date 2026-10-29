"""AB-30 matrix + IC-9 batch run: author-result payload seam.

IC-9's schema-migration census is AB-30's pairing consumer: it validates
every finalized transaction's RECORDED author result against the candidate
tree's LIVE schema, so any batch movement of the author-result schema shows
up as a pre-launch schema-stale count instead of a paid re-author storm.

This module runs that census against the FROZEN pre-batch chain captured at
the Wave-2 freeze. At batch assembly (``MENAGERIE_WAVE2_ASSEMBLY=1`` -- and
at every later tree) a schema tightening turns the frozen recorded result
stale, the gate arm exits 2, and this matrix fails LOUD: the batch then owes
an explicit migration decision, never an ambient cost.

The live-records batch run (the orchestrator's §4.2 hook, same tool over the
real artifact shards) is::

    PYTHONPATH=. python -m menagerie.crawler.tools.schema_migration_census \\
        --artifact-shard <records>/artifacts/<shard>.jsonl \\
        --repository-root <campaign clone root> --gate

The 14 schema-stale finalized results already live at the Wave-1 tree
(WS6's pilot reading) are that run's inherited migration decision; they are
records-side facts, deliberately NOT frozen here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from menagerie.crawler.constants import AUTHOR_RESULT_SCHEMA_VERSION_V3
from menagerie.crawler.tests.mixed_era import (
    assert_era_outcome,
    classify_reader,
    load_manifest,
    read_fixture_bytes,
)
from menagerie.crawler.tools.schema_migration_census import main, schema_migration_census

pytestmark = pytest.mark.smoke


def _materialize_repo(tmp_path: Path) -> tuple[Path, Path]:
    """Re-materialize the frozen chain's canonical repository layout.

    The reconstruction document is committed under a portable filename; the
    manifest records its canonical anchor path (which contains ``:`` and so
    cannot live in a Windows-checkout-safe repository tree).
    """

    manifest = load_manifest()
    anchor = manifest["award_chain_anchor"]
    repo = tmp_path / "repo"
    shard = repo / "menagerie" / "crawler" / "records" / "artifacts" / "shard.jsonl"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(read_fixture_bytes("award_chain/shard.jsonl"))
    document = repo / str(anchor["anchor_path"])
    document.parent.mkdir(parents=True)
    document.write_bytes(read_fixture_bytes("award_chain/reconstruction_doc.json"))
    return shard, repo


def test_frozen_prebatch_result_is_schema_valid_at_the_live_tree(tmp_path: Path) -> None:
    """IC-9 over the frozen pre-batch chain: valid now; stale = loud decision.

    This is the committed half of the IC-9 batch run: if the assembled batch
    moves the live author-result schema, this exact frozen record flips to
    schema-stale and the assertion fails, forcing the explicit migration
    decision the census exists to extract.
    """

    shard, repo = _materialize_repo(tmp_path)
    outcome = classify_reader(
        lambda: [
            str(row["disposition"])
            for row in schema_migration_census((shard,), repository_root=repo)
        ]
    )
    assert_era_outcome(
        reader_id="ic9-schema-migration-census-gate",
        outcome=outcome,
        prebatch_expected="parsed",
    )
    assert outcome[1] == ["schema-valid"], (
        "the frozen pre-batch recorded author result went schema-stale: the batch "
        "owes an explicit migration decision (IC-9 gate), never an ambient re-author cost"
    )
    assert (
        main(["--artifact-shard", str(shard), "--repository-root", str(repo), "--gate"]) == 0
    )


def test_stale_schema_control_arm_fires_the_gate(tmp_path: Path) -> None:
    """Control: the census sees staleness (frozen result vs the retired v3)."""

    shard, repo = _materialize_repo(tmp_path)
    rows = schema_migration_census(
        (shard,), repository_root=repo, schema_version=AUTHOR_RESULT_SCHEMA_VERSION_V3
    )
    assert [str(row["disposition"]) for row in rows] == ["schema-stale"]
    assert (
        main(
            [
                "--artifact-shard",
                str(shard),
                "--repository-root",
                str(repo),
                "--schema-version",
                AUTHOR_RESULT_SCHEMA_VERSION_V3,
                "--gate",
            ]
        )
        == 2
    )


def test_tampered_reconstruction_reports_unreadable(tmp_path: Path) -> None:
    """A reconstruction contradicting its anchor reports unreadable, not valid."""

    shard, repo = _materialize_repo(tmp_path)
    manifest = load_manifest()
    document = repo / str(manifest["award_chain_anchor"]["anchor_path"])
    document.write_bytes(document.read_bytes() + b" ")
    rows = schema_migration_census((shard,), repository_root=repo)
    assert [str(row["disposition"]) for row in rows] == ["reconstruction-unreadable"]
