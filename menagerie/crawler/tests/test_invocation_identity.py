"""Invocation-identity ledger and leg attribution (D-22/DX-23 + B3-XQ-12).

Every 200a driver start reused ``run_id="crawler-run"``, so no ledger read can
prove which process invocation wrote a row, how many invocations one leg
spanned, or whether a "single leg" actually survived a kill/resume. The
invocation ledger rotates the run_id axis to a real per-invocation identity:
one durable row per process invocation, typed succession rows for autonomous
pause/resume, and derivations that make one-leg certification, close citation,
and void-append accounting machine-checkable. Historical rows keep the legacy
constant and stay readable (era rule); only NEW identities are refused it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from menagerie.crawler.constants import InvocationOrigin
from menagerie.crawler.driver_contracts import DriverConfig
from menagerie.crawler.invocation_identity import (
    INVOCATION_IDENTITY_SCHEMA_VERSION,
    LEGACY_RUN_ID,
    PRE_LEDGER_ERA,
    InvocationIdentityError,
    LegacyRunIdentityError,
    close_leg_citations,
    invocation_ledger_path,
    leg_valid_single_invocation,
    load_invocation_rows,
    mint_default_run_id,
    mint_run_id,
    record_invocation_start,
    record_invocation_succession,
    void_interventions_by_invocation,
)
from menagerie.crawler.leg_generation import (
    LegGeneration,
    generation_run_ids,
    load_leg_events,
    open_leg_generation,
)

_SNAPSHOT_SHA = "sha256:" + "0" * 64


def _open_leg(
    records_root: Path,
    *,
    run_id: str,
    invocation_id: str | None = None,
    opened_at: str = "2026-08-24T00:00:00Z",
) -> LegGeneration:
    """Open (or continue) the one fixture leg generation."""

    return open_leg_generation(
        records_root,
        campaign_id="c1-mech",
        intake_snapshot_id="snap-1",
        intake_snapshot_sha256=_SNAPSHOT_SHA,
        phase=None,
        run_id=run_id,
        opened_at=opened_at,
        invocation_id=invocation_id,
    )


@pytest.mark.smoke
def test_mint_run_id_rotates_off_the_legacy_constant() -> None:
    first = mint_run_id(
        machine_id="worker-a", hostname="worker-a.local", pid=101, started_at="2026-08-24T00:00:00Z"
    )
    second = mint_run_id(
        machine_id="worker-a", hostname="worker-a.local", pid=102, started_at="2026-08-24T00:00:00Z"
    )
    assert first != second
    assert first.startswith("run-")
    assert first != LEGACY_RUN_ID
    # Same inputs, same identity: the mint is a digest, not entropy.
    assert first == mint_run_id(
        machine_id="worker-a", hostname="worker-a.local", pid=101, started_at="2026-08-24T00:00:00Z"
    )
    # The zero-argument mint is unique per call even within one microsecond.
    assert mint_default_run_id() != mint_default_run_id()


@pytest.mark.smoke
def test_driver_config_requires_real_run_identity() -> None:
    config = DriverConfig()
    assert config.run_id != LEGACY_RUN_ID
    assert config.run_id.startswith("run-")
    # Two constructions never share an identity: the default mints per invocation.
    assert DriverConfig().run_id != DriverConfig().run_id
    with pytest.raises(LegacyRunIdentityError):
        DriverConfig(run_id=LEGACY_RUN_ID)
    with pytest.raises(ValueError):
        DriverConfig(run_id="   ")


@pytest.mark.smoke
def test_invocation_ledger_roundtrip(tmp_path: Path) -> None:
    row = record_invocation_start(
        tmp_path, machine_id="worker-a", invocation_origin=InvocationOrigin.ORDINARY_RUN
    )
    path = invocation_ledger_path(tmp_path)
    assert path == tmp_path / "operational" / "invocations.jsonl"
    assert path.is_file()
    assert row["schema_version"] == INVOCATION_IDENTITY_SCHEMA_VERSION
    assert row["event"] == "invocation_start"
    assert row["run_id"].startswith("run-")
    assert row["invocation_id"].startswith("inv-")
    assert row["machine_id"] == "worker-a"
    assert isinstance(row["pid"], int) and row["pid"] > 0
    assert isinstance(row["hostname"], str) and row["hostname"]
    loaded = load_invocation_rows(tmp_path)
    assert loaded == (row,)
    # A torn final line is tolerated: the crashed append never executed work.
    with path.open("ab") as handle:
        handle.write(b'{"torn":')
    assert load_invocation_rows(tmp_path) == (row,)


def test_invocation_ledger_refuses_interior_corruption(tmp_path: Path) -> None:
    row = record_invocation_start(
        tmp_path, machine_id="worker-a", invocation_origin=InvocationOrigin.ORDINARY_RUN
    )
    path = invocation_ledger_path(tmp_path)
    tampered = dict(row)
    tampered["machine_id"] = "someone-else"
    path.write_bytes(json.dumps(tampered).encode("utf-8") + b"\n")
    with pytest.raises(InvocationIdentityError):
        load_invocation_rows(tmp_path)
    path.write_bytes(b"not json\n")
    with pytest.raises(InvocationIdentityError):
        load_invocation_rows(tmp_path)


def test_new_invocation_rows_refuse_the_legacy_constant(tmp_path: Path) -> None:
    with pytest.raises(LegacyRunIdentityError):
        record_invocation_start(
            tmp_path,
            machine_id="worker-a",
            invocation_origin=InvocationOrigin.ORDINARY_RUN,
            run_id=LEGACY_RUN_ID,
        )


def test_succession_requires_known_predecessor_and_closed_reason(tmp_path: Path) -> None:
    with pytest.raises(InvocationIdentityError):
        record_invocation_succession(
            tmp_path, predecessor_invocation_id="inv-absent", machine_id="worker-a"
        )
    start = record_invocation_start(
        tmp_path, machine_id="worker-a", invocation_origin=InvocationOrigin.ORDINARY_RUN
    )
    with pytest.raises(InvocationIdentityError):
        record_invocation_succession(
            tmp_path,
            predecessor_invocation_id=start["invocation_id"],
            reason="machine-reboot",
            machine_id="worker-a",
        )
    resumed = record_invocation_succession(
        tmp_path, predecessor_invocation_id=start["invocation_id"], machine_id="worker-a"
    )
    assert resumed["event"] == "invocation_succession"
    assert resumed["reason"] == "autonomous-pause-resume"
    assert resumed["predecessor_invocation_id"] == start["invocation_id"]
    # A typed resume continues the run identity it succeeds by default.
    assert resumed["run_id"] == start["run_id"]
    assert resumed["invocation_id"] != start["invocation_id"]


@pytest.mark.smoke
def test_kill_resume_x3_fails_one_leg_certification(tmp_path: Path) -> None:
    """B3-XQ-12 acceptance: kill/resume x3 => 4 rows, certification FAILS."""

    generation: LegGeneration | None = None
    for _ in range(4):
        invocation = record_invocation_start(
            tmp_path, machine_id="worker-a", invocation_origin=InvocationOrigin.MANUAL_RESUME
        )
        generation = _open_leg(
            tmp_path,
            run_id=invocation["run_id"],
            invocation_id=invocation["invocation_id"],
        )
    assert generation is not None
    assert len(load_invocation_rows(tmp_path)) == 4
    assert len(generation_run_ids(tmp_path, generation.generation_id)) == 4
    certification = leg_valid_single_invocation(tmp_path, generation.generation_id)
    assert certification.valid is False
    assert certification.reason == "untyped-restart"
    assert len(certification.invocation_ids) == 4


def test_typed_pause_resume_preserves_one_leg(tmp_path: Path) -> None:
    """B3-XQ-12: a typed autonomous pause/resume chain is still one leg."""

    root = record_invocation_start(
        tmp_path, machine_id="worker-a", invocation_origin=InvocationOrigin.ORDINARY_RUN
    )
    generation = _open_leg(
        tmp_path, run_id=root["run_id"], invocation_id=root["invocation_id"]
    )
    previous = root
    for index in range(3):
        # One resume rotates onto a fresh run identity; the rest inherit.
        fresh = (
            mint_run_id(
                machine_id="worker-a",
                hostname="worker-a.local",
                pid=4000 + index,
                started_at=f"2026-08-24T00:0{index}:00Z",
            )
            if index == 1
            else None
        )
        previous = record_invocation_succession(
            tmp_path,
            predecessor_invocation_id=previous["invocation_id"],
            machine_id="worker-a",
            run_id=fresh,
        )
        _open_leg(
            tmp_path,
            run_id=previous["run_id"],
            invocation_id=previous["invocation_id"],
        )
    assert len(load_invocation_rows(tmp_path)) == 4
    certification = leg_valid_single_invocation(tmp_path, generation.generation_id)
    assert certification.valid is True
    assert certification.reason == "typed-succession-chain"
    assert len(certification.invocation_ids) == 4
    assert certification.unattributed_run_ids == ()


def test_one_leg_certification_is_era_honest(tmp_path: Path) -> None:
    """A legacy generation (no invocation rows) is uncertifiable, never valid."""

    generation = _open_leg(tmp_path, run_id=LEGACY_RUN_ID)
    certification = leg_valid_single_invocation(tmp_path, generation.generation_id)
    assert certification.valid is False
    assert certification.reason == "unattributed-run-identity"
    assert certification.unattributed_run_ids == (LEGACY_RUN_ID,)


@pytest.mark.smoke
def test_close_names_the_legs_it_spans(tmp_path: Path) -> None:
    first = record_invocation_start(
        tmp_path, machine_id="worker-a", invocation_origin=InvocationOrigin.ORDINARY_RUN
    )
    second = record_invocation_start(
        tmp_path, machine_id="worker-a", invocation_origin=InvocationOrigin.MANUAL_RESUME
    )
    generation = _open_leg(
        tmp_path, run_id=first["run_id"], invocation_id=first["invocation_id"]
    )
    _open_leg(tmp_path, run_id=second["run_id"], invocation_id=second["invocation_id"])
    citations = close_leg_citations(tmp_path, generation.generation_id)
    assert [c["run_id"] for c in citations] == sorted([first["run_id"], second["run_id"]])
    by_run = {c["run_id"]: c for c in citations}
    assert by_run[first["run_id"]]["invocation_ids"] == [first["invocation_id"]]
    assert by_run[second["run_id"]]["invocation_ids"] == [second["invocation_id"]]
    assert all(c["era"] == "invocation-ledger" for c in citations)
    # The citation list is exactly the durable joined-run set, never a subset.
    assert {c["run_id"] for c in citations} == set(
        generation_run_ids(tmp_path, generation.generation_id)
    )


def test_close_citations_are_era_honest_for_legacy_runs(tmp_path: Path) -> None:
    generation = _open_leg(tmp_path, run_id=LEGACY_RUN_ID)
    citations = close_leg_citations(tmp_path, generation.generation_id)
    assert citations == (
        {"run_id": LEGACY_RUN_ID, "invocation_ids": [], "era": PRE_LEDGER_ERA},
    )


@pytest.mark.smoke
def test_leg_rows_carry_invocation_id_monotone(tmp_path: Path) -> None:
    _open_leg(tmp_path, run_id="run-attributed", invocation_id="inv-attributed")
    events = load_leg_events(tmp_path)
    opens = [row for row in events if row["event"] == "leg_open"]
    runs = [row for row in events if row["event"] == "leg_run"]
    assert opens and all(row["invocation_id"] == "inv-attributed" for row in opens)
    assert runs and all(row["invocation_id"] == "inv-attributed" for row in runs)
    # Versioned-monotone: an old-era root without the field still validates.
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    _open_leg(legacy_root, run_id="run-old-era")
    legacy_events = load_leg_events(legacy_root)
    assert legacy_events
    assert all("invocation_id" not in row for row in legacy_events)


def test_void_appends_count_as_interventions_by_invocation(tmp_path: Path) -> None:
    invocation = record_invocation_start(
        tmp_path, machine_id="worker-a", invocation_origin=InvocationOrigin.ORDINARY_RUN
    )
    void_rows = [
        # Legacy void: no run identity -> pre-ledger era bucket.
        {
            "void_id": "sha256:v1",
            "void_of": "sha256:g1",
            "reason": "duplicate grant",
            "voided_by": "jmt",
            "created_at": "2026-08-01T00:00:00Z",
        },
        # New-era void carrying its writing invocation's run identity.
        {
            "void_id": "sha256:v2",
            "void_of": "sha256:g2",
            "reason": "superseded",
            "voided_by": "jmt",
            "created_at": "2026-08-24T00:00:00Z",
            "run_id": invocation["run_id"],
        },
        # A grant row is not a void and is never counted.
        {"grant_id": "sha256:g3", "created_at": "2026-08-24T00:00:00Z"},
    ]
    counts = void_interventions_by_invocation(void_rows, load_invocation_rows(tmp_path))
    assert counts[invocation["invocation_id"]] == 1
    assert counts[PRE_LEDGER_ERA] == 1
    assert sum(counts.values()) == 2
