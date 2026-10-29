"""AB-30 matrix: run-id invocation-identity seam (D-22/DX-23, WS3/WS7).

The Wave-2 batch rotates ``run_id`` off the retired shared constant
``crawler-run`` (all 43 200a starts reused it) onto real per-invocation
identities backed by the append-only ``operational/invocations.jsonl``
ledger (schema ``menagerie.crawler.invocation-identity.v1``). The era rule
this matrix pins, reader x era:

* prior-era rows stamped with the constant stay READABLE everywhere, but
  era-scoped -- classified ``pre-invocation-ledger`` (readable, never
  attributable to one logical invocation);
* every NEW identity refuses the constant TYPED
  (:class:`LegacyRunIdentityError` on mint/append; the ledger reader
  refuses a smuggled constant-era row because the ledger is new-era only);
* an unknown invocation-ledger schema version refuses typed at read.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from menagerie.crawler.constants import InvocationOrigin
from menagerie.crawler.identity import stamp_row_digest
from menagerie.crawler.invocation_identity import (
    INVOCATION_LEDGER_ERA,
    LEGACY_RUN_ID,
    PRE_LEDGER_ERA,
    InvocationIdentityError,
    LegacyRunIdentityError,
    close_leg_citations,
    invocation_ledger_path,
    leg_valid_single_invocation,
    load_invocation_rows,
    mint_default_run_id,
    record_invocation_start,
    record_invocation_succession,
    void_interventions_by_invocation,
)
from menagerie.crawler.leg_generation import LegGeneration, open_leg_generation
from menagerie.crawler.tests.mixed_era import assert_era_outcome, classify_reader

pytestmark = pytest.mark.smoke

_SNAPSHOT_SHA = "sha256:" + "0" * 64


def _open_leg(
    records_root: Path, *, run_id: str, invocation_id: str | None = None
) -> LegGeneration:
    """Open (or join) the one matrix leg generation under one records root."""

    return open_leg_generation(
        records_root,
        campaign_id="c1-mech",
        intake_snapshot_id="snap-1",
        intake_snapshot_sha256=_SNAPSHOT_SHA,
        phase=None,
        run_id=run_id,
        opened_at="2026-08-24T00:00:00Z",
        invocation_id=invocation_id,
    )


def test_legacy_constant_rows_classify_pre_ledger_never_one_invocation(
    tmp_path: Path,
) -> None:
    """Cell 1: a prior-era ``crawler-run`` row parses era-scoped, unattributable.

    A mixed-era generation -- one historical leg stamped with the constant,
    one new-era leg carrying a real invocation row -- read through the era
    classification reader must parse with per-run era labels: the constant
    row is ``pre-invocation-ledger`` with NO invocation attribution, the
    minted row is ``invocation-ledger``, and the new-era neighbor never
    upgrades the legacy run into "one logical invocation".
    """

    invocation = record_invocation_start(
        tmp_path, machine_id="worker-a", invocation_origin=InvocationOrigin.ORDINARY_RUN
    )
    generation = _open_leg(tmp_path, run_id=LEGACY_RUN_ID)
    _open_leg(
        tmp_path, run_id=invocation["run_id"], invocation_id=invocation["invocation_id"]
    )

    outcome = classify_reader(
        lambda: close_leg_citations(tmp_path, generation.generation_id)
    )
    assert_era_outcome(
        reader_id="close-citation-era-classifier",
        outcome=outcome,
        prebatch_expected="parsed",
    )
    by_run = {citation["run_id"]: citation for citation in outcome[1]}
    assert set(by_run) == {LEGACY_RUN_ID, invocation["run_id"]}
    assert by_run[LEGACY_RUN_ID]["era"] == PRE_LEDGER_ERA
    assert by_run[LEGACY_RUN_ID]["invocation_ids"] == []
    assert by_run[invocation["run_id"]]["era"] == INVOCATION_LEDGER_ERA
    assert by_run[invocation["run_id"]]["invocation_ids"] == [
        invocation["invocation_id"]
    ]

    # Readable-unattributable, era-honest: the constant-era run keeps the
    # certification NEGATIVE; readability never becomes attribution.
    certification = leg_valid_single_invocation(tmp_path, generation.generation_id)
    assert certification.valid is False
    assert certification.reason == "unattributed-run-identity"
    assert LEGACY_RUN_ID in certification.unattributed_run_ids

    # Same era rule on the void-attribution reader: a void row stamped with
    # the constant buckets under the pre-ledger era even though the ledger
    # holds new-era rows (the constant can never name a ledger invocation).
    void_outcome = classify_reader(
        lambda: void_interventions_by_invocation(
            [
                {
                    "void_id": "sha256:v1",
                    "void_of": "sha256:g1",
                    "reason": "duplicate grant",
                    "voided_by": "jmt",
                    "created_at": "2026-08-01T00:00:00Z",
                    "run_id": LEGACY_RUN_ID,
                }
            ],
            load_invocation_rows(tmp_path),
        )
    )
    assert_era_outcome(
        reader_id="void-attribution-era-classifier",
        outcome=void_outcome,
        prebatch_expected="parsed",
    )
    assert void_outcome[1] == {PRE_LEDGER_ERA: 1}


def test_new_identity_paths_refuse_the_legacy_constant_typed(tmp_path: Path) -> None:
    """Cell 2: minting/appending a NEW row under the constant refuses typed.

    Both new-identity writers -- ``invocation_start`` and the typed
    succession -- must refuse ``run_id="crawler-run"`` with the seam's
    typed error, and a refusal must append NOTHING durable.
    """

    start_outcome = classify_reader(
        lambda: record_invocation_start(
            tmp_path,
            machine_id="worker-a",
            invocation_origin=InvocationOrigin.ORDINARY_RUN,
            run_id=LEGACY_RUN_ID,
        )
    )
    assert_era_outcome(
        reader_id="invocation-start-legacy-constant",
        outcome=start_outcome,
        prebatch_expected="refused-typed",
    )
    assert isinstance(start_outcome[1], LegacyRunIdentityError)
    assert load_invocation_rows(tmp_path) == ()

    root = record_invocation_start(
        tmp_path, machine_id="worker-a", invocation_origin=InvocationOrigin.ORDINARY_RUN
    )
    succession_outcome = classify_reader(
        lambda: record_invocation_succession(
            tmp_path,
            predecessor_invocation_id=root["invocation_id"],
            machine_id="worker-a",
            run_id=LEGACY_RUN_ID,
        )
    )
    assert_era_outcome(
        reader_id="invocation-succession-legacy-constant",
        outcome=succession_outcome,
        prebatch_expected="refused-typed",
    )
    assert isinstance(succession_outcome[1], LegacyRunIdentityError)
    assert load_invocation_rows(tmp_path) == (root,)


def test_ledger_read_refuses_a_smuggled_legacy_constant_row(tmp_path: Path) -> None:
    """Cell 2, read side: the ledger itself is new-era only.

    A constant-era row physically written into ``invocations.jsonl`` with a
    CONSISTENT content address (the identity is the sole offense) must
    refuse typed at read -- historical rows live outside this ledger, so a
    reader that parsed one would be laundering the legacy era into an
    attribution authority.
    """

    row = record_invocation_start(
        tmp_path, machine_id="worker-a", invocation_origin=InvocationOrigin.ORDINARY_RUN
    )
    smuggled = stamp_row_digest({**row, "run_id": LEGACY_RUN_ID})
    invocation_ledger_path(tmp_path).write_text(json.dumps(smuggled) + "\n")

    outcome = classify_reader(lambda: load_invocation_rows(tmp_path))
    assert_era_outcome(
        reader_id="invocation-ledger-read-legacy-constant",
        outcome=outcome,
        prebatch_expected="refused-typed",
    )
    assert isinstance(outcome[1], InvocationIdentityError)
    # The refusal is the era rule, not an incidental digest mismatch.
    assert "retired shared run identity" in str(outcome[1])


def test_minted_default_identity_rotates_and_appends(tmp_path: Path) -> None:
    """Cell 3: the default mint is off the constant and append-accepted."""

    minted = mint_default_run_id()
    assert minted != LEGACY_RUN_ID
    assert minted.startswith("run-")

    outcome = classify_reader(
        lambda: record_invocation_start(
            tmp_path,
            machine_id="worker-a",
            invocation_origin=InvocationOrigin.ORDINARY_RUN,
            run_id=minted,
        )
    )
    assert_era_outcome(
        reader_id="invocation-append-minted-identity",
        outcome=outcome,
        prebatch_expected="parsed",
    )
    row = outcome[1]
    assert row["run_id"] == minted
    assert load_invocation_rows(tmp_path) == (row,)


def test_unknown_invocation_schema_version_refuses_typed(tmp_path: Path) -> None:
    """Cell 4: an unknown ledger schema version refuses typed at read.

    The row is otherwise well-formed and digest-consistent (re-stamped
    after the mutation), so the schema vocabulary is the sole offense --
    a future-era row can never half-parse through this reader.
    """

    row = record_invocation_start(
        tmp_path, machine_id="worker-a", invocation_origin=InvocationOrigin.ORDINARY_RUN
    )
    alien = stamp_row_digest(
        {**row, "schema_version": "menagerie.crawler.invocation-identity.v999"}
    )
    invocation_ledger_path(tmp_path).write_text(json.dumps(alien) + "\n")

    outcome = classify_reader(lambda: load_invocation_rows(tmp_path))
    assert_era_outcome(
        reader_id="invocation-ledger-read-unknown-schema",
        outcome=outcome,
        prebatch_expected="refused-typed",
    )
    assert isinstance(outcome[1], InvocationIdentityError)
    assert "unknown schema" in str(outcome[1])
