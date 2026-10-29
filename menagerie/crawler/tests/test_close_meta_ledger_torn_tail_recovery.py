"""Torn CLOSE-META ledger tails recover like every other ledger (R2 BL-7 gap).

BL-7 gave the core ``recordio`` ledgers evidence-then-truncate torn-tail
recovery and T1-07/T5-1 gave it to the eight machine-deferral valves. The three
ledgers the close/licensing spine is actually *built* from never inherited any
of it:

* ``intake/leg-events.jsonl`` -- the leg generation epoch, its census baselines,
  and every sealed selection plan;
* ``intake/closes.jsonl`` -- the close objects that license the next rung;
* ``intake/registered-rungs.jsonl`` -- the Track A registry whose rows anchor a
  staged rung to the close that licensed it.

All three were read by ``ledger_chain.decode_ledger``, which did
``path.read_bytes().splitlines()``: a torn, non-newline-terminated final line --
the signature of a SIGKILL, a power loss, or ENOSPC mid-append -- decodes as a
truncated JSON document and raises. Because ``verify_records_integrity``,
``load_leg_events``, ``load_rung_registrations``, ``load_close_objects``, and
the census in ``source_identity`` all funnel through those decoders, one torn
byte permanently REFUSED both closing and staging campaign-wide, with no
recovery path at all: exactly the step-200 blocker.

The contract these pins hold is recordio's, unchanged and not weakened: only a
torn FINAL line ever recovers, its bytes are hashed into a durable recovery
ledger BEFORE truncation, and a COMPLETE malformed line -- or a torn tail
sitting behind one -- stays hard, typed corruption preserved byte-for-byte.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pytest

from menagerie.crawler.close_object import (
    CloseObjectError,
    close_object_path,
    load_close_objects,
)
from menagerie.crawler.ledger_chain import (
    LedgerChainError,
    chain_state,
    verify_records_integrity,
)
from menagerie.crawler.leg_generation import (
    LegGenerationError,
    census_identities,
    close_ledger_sources,
    leg_events_path,
    load_leg_events,
    record_leg_plan,
    source_rows,
)
from menagerie.crawler.models import JsonObject
from menagerie.crawler.reducer import default_ledger_paths
from menagerie.crawler.rung_staging import (
    RungStagingError,
    append_rung_registration,
    build_rung_registration,
    load_rung_registrations,
    rung_registry_path,
)
from menagerie.crawler.tests.close_object_support import record_fixture_close
from menagerie.crawler.tests.test_valve_torn_tail_recovery import (
    _evidence_path,
    _tear_final_line,
)

pytestmark = pytest.mark.smoke

CAMPAIGN = "c1-mech"
SNAPSHOT = "intake-aaaa"
SNAPSHOT_SHA256 = "sha256:aaaa"


def _seed_core_ledgers(records_root: Path) -> None:
    """Materialize the empty canonical fact ledgers a fixture leg baselines."""

    ledgers = default_ledger_paths(records_root)
    for name in ("models", "attempts", "gates", "artifacts"):
        path = getattr(ledgers, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(b"")


# --------------------------------------------------------------------------
# The three close meta-ledgers, each seeded and read through its OWN writer
# and its OWN strict decoder -- never through a test-local reimplementation.
# --------------------------------------------------------------------------


def _seed_leg_event(records_root: Path, index: int) -> JsonObject:
    """Seal one real leg-plan event on the leg-event ledger."""

    return record_leg_plan(
        records_root,
        campaign_id=CAMPAIGN,
        generation_id=f"sha256:{index:064d}",
        run_id=f"run-{index}",
        admitted_ids=(f"m{index}",),
        plan_payload={"seeded_plan": index},
        created_at=f"2026-08-11T00:0{index}:00Z",
    )


def _load_leg_event_ids(records_root: Path) -> tuple[str, ...]:
    """Return every durable leg-event content address in ledger order."""

    return tuple(str(row["row_sha256"]) for row in load_leg_events(records_root))


def _seed_close(records_root: Path, index: int) -> JsonObject:
    """Close one fixture rung honestly through the production derivation.

    Only the FIRST close of the rung records member heads in its own window;
    every later close is a new adjudication epoch that honestly CARRIES the
    same members, exactly as ``test_close_object`` re-closes them.
    """

    _seed_core_ledgers(records_root)
    return record_fixture_close(
        records_root,
        campaign_id=CAMPAIGN,
        intake_snapshot_id=SNAPSHOT,
        intake_snapshot_sha256=SNAPSHOT_SHA256,
        member_ids=("m1", "m2"),
        seed_member_heads=not close_object_path(records_root).exists(),
        closed_at=f"2026-08-11T01:0{index}:00Z",
        run_id=f"pytest-close-{index}",
    )


def _load_close_ids(records_root: Path) -> tuple[str, ...]:
    """Return every persisted close content address in ledger order."""

    return tuple(
        str(row["close_sha256"]) for row in load_close_objects([close_object_path(records_root)])
    )


def _snapshot_id(index: int) -> str:
    """Return one syntactically valid intake snapshot id."""

    return f"intake-{index:020x}"


def _seed_registration(records_root: Path, index: int) -> JsonObject:
    """Register one rung on the Track A registry ledger."""

    registration = build_rung_registration(
        campaign_id=CAMPAIGN,
        snapshot_id=_snapshot_id(index),
        snapshot_sha256=f"sha256:{index:064d}",
        derived_from_snapshot_id=_snapshot_id(0),
        member_count=index,
        label=f"rung-{index}",
        registered_by="pytest",
        created_at=f"2026-08-11T02:0{index}:00Z",
    )
    staged = append_rung_registration(rung_registry_path(records_root), registration)
    return dict(staged.registration.to_row())


def _load_registration_ids(records_root: Path) -> tuple[str, ...]:
    """Return every durable registration snapshot id in ledger order."""

    return tuple(row.snapshot_id for row in load_rung_registrations(records_root))


@dataclass(frozen=True)
class MetaLedgerSpec:
    """One close meta-ledger, its writer, its strict reader, and its refusal."""

    scope: str
    path_for: Callable[[Path], Path]
    seed: Callable[[Path, int], Mapping[str, Any]]
    load_ids: Callable[[Path], Sequence[str]]
    identity_of: Callable[[Mapping[str, Any]], str]
    error: type[Exception]


META_LEDGERS: tuple[MetaLedgerSpec, ...] = (
    MetaLedgerSpec(
        scope="meta.leg-events",
        path_for=leg_events_path,
        seed=_seed_leg_event,
        load_ids=_load_leg_event_ids,
        identity_of=lambda row: str(row["row_sha256"]),
        error=LegGenerationError,
    ),
    MetaLedgerSpec(
        scope="meta.closes",
        path_for=close_object_path,
        seed=_seed_close,
        load_ids=_load_close_ids,
        identity_of=lambda row: str(row["close_sha256"]),
        error=CloseObjectError,
    ),
    MetaLedgerSpec(
        scope="meta.registry",
        path_for=rung_registry_path,
        seed=_seed_registration,
        load_ids=_load_registration_ids,
        identity_of=lambda row: str(row["snapshot_id"]),
        error=RungStagingError,
    ),
)

_META_IDS = tuple(spec.scope for spec in META_LEDGERS)


@pytest.mark.parametrize("spec", META_LEDGERS, ids=_META_IDS)
def test_a_torn_close_meta_ledger_read_recovers_with_evidence(
    spec: MetaLedgerSpec, tmp_path: Path
) -> None:
    """A crash mid-append never bricks the close spine's own read paths.

    The torn final line is evidenced (offset + tail hash, durably synced) and
    then truncated; every earlier complete row survives; the chain still
    verifies over what remains.
    """

    records_root = tmp_path / "records"
    first = spec.seed(records_root, 1)
    spec.seed(records_root, 2)
    ledger = spec.path_for(records_root)
    good_end = _tear_final_line(ledger)

    assert tuple(spec.load_ids(records_root)) == (spec.identity_of(first),)

    evidence = _evidence_path(ledger)
    assert evidence.is_file(), "a dropped torn row must be evidenced, never silent"
    records = [json.loads(line) for line in evidence.read_text().splitlines()]
    assert records[0]["byte_offset"] == good_end
    assert records[0]["byte_count"] > 0
    assert records[0]["tail_sha256"].startswith("sha256:")
    assert ledger.read_bytes().endswith(b"\n")
    assert chain_state(ledger, scope=spec.scope).rows == 1


@pytest.mark.parametrize("spec", META_LEDGERS, ids=_META_IDS)
def test_a_torn_close_meta_ledger_still_accepts_the_next_append(
    spec: MetaLedgerSpec, tmp_path: Path
) -> None:
    """The interrupted work is re-derivable: the ledger is not write-bricked.

    The append path decodes the whole ledger under its own lock before it can
    link the next chain row, so a torn tail used to refuse every future append
    forever. Recovery restores exactly one property: the next honest append
    lands, and its chain link extends the retained prefix.
    """

    records_root = tmp_path / "records"
    first = spec.seed(records_root, 1)
    spec.seed(records_root, 2)
    ledger = spec.path_for(records_root)
    _tear_final_line(ledger)

    third = spec.seed(records_root, 3)
    assert set(spec.load_ids(records_root)) == {
        spec.identity_of(first),
        spec.identity_of(third),
    }
    assert chain_state(ledger, scope=spec.scope).rows == 2


@pytest.mark.parametrize("spec", META_LEDGERS, ids=_META_IDS)
def test_newline_terminated_meta_garbage_is_still_refused(
    spec: MetaLedgerSpec, tmp_path: Path
) -> None:
    """A COMPLETE malformed line stays hard, typed corruption -- never repaired."""

    records_root = tmp_path / "records"
    spec.seed(records_root, 1)
    ledger = spec.path_for(records_root)
    with ledger.open("ab") as handle:
        handle.write(b"not-json\n")
    frozen = ledger.read_bytes()

    with pytest.raises((spec.error, LedgerChainError)):
        spec.load_ids(records_root)
    assert ledger.read_bytes() == frozen, "hard corruption must never be rewritten"
    assert not _evidence_path(ledger).exists()


@pytest.mark.parametrize("spec", META_LEDGERS, ids=_META_IDS)
def test_a_torn_meta_tail_behind_hard_corruption_is_not_truncated(
    spec: MetaLedgerSpec, tmp_path: Path
) -> None:
    """Recovery refuses a ledger whose RETAINED prefix is itself corrupt."""

    records_root = tmp_path / "records"
    spec.seed(records_root, 1)
    ledger = spec.path_for(records_root)
    with ledger.open("ab") as handle:
        handle.write(b"not-json\n")
        handle.write(b'{"torn":')
    frozen = ledger.read_bytes()

    with pytest.raises((spec.error, LedgerChainError)):
        spec.load_ids(records_root)
    assert ledger.read_bytes() == frozen
    assert not _evidence_path(ledger).exists()


def test_a_torn_meta_tail_no_longer_bricks_close_verification(tmp_path: Path) -> None:
    """The step-200 blocker itself: Track A verification recovers, not refuses.

    ``verify_records_integrity`` is the Track A authority every close and every
    staging decision consults. With a torn tail on ANY of the three close
    meta-ledgers it used to raise ``LedgerChainError`` forever -- licensing
    permanently REFUSED, with nothing an operator could run to recover. It must
    now recover each torn tail with evidence and then verify the retained
    history.
    """

    records_root = tmp_path / "records"
    _seed_leg_event(records_root, 1)
    _seed_leg_event(records_root, 2)
    _seed_close(records_root, 1)
    _seed_registration(records_root, 1)
    _seed_registration(records_root, 2)

    torn = (
        leg_events_path(records_root),
        rung_registry_path(records_root),
    )
    for ledger in torn:
        _tear_final_line(ledger)

    verify_records_integrity(records_root)

    for ledger in torn:
        assert _evidence_path(ledger).is_file()
        assert ledger.read_bytes().endswith(b"\n")

    # And staging is unblocked in the same breath: the registry accepts the
    # next honest registration, and the census the close gate re-observes is
    # derivable again.
    _seed_registration(records_root, 3)
    assert _load_registration_ids(records_root) == (_snapshot_id(1), _snapshot_id(3))
    assert set(census_identities(records_root)) == set(close_ledger_sources(records_root))


def test_a_torn_census_source_tail_recovers_before_its_identity_is_attested(
    tmp_path: Path,
) -> None:
    """A torn census SOURCE cannot poison the identity a close attests.

    ``source_identity`` counted the torn fragment as a row and hashed it into
    ``ledger_sha256``, so a crash mid-append made the close census disagree
    with every later re-observation -- the same brick from the census side. The
    census read recovers first, so the attested identity describes exactly the
    rows the campaign acknowledged: the interrupted append never landed, and it
    contributes neither a row count nor a byte to the attestation.
    """

    records_root = tmp_path / "records"
    _seed_core_ledgers(records_root)
    _seed_close(records_root, 1)

    models = close_ledger_sources(records_root)["models.shard"]
    assert models.is_file() and models.read_bytes().endswith(b"\n")
    clean = census_identities(records_root)["models.shard"]
    durable = models.read_bytes()

    # Exactly what a SIGKILL mid-append leaves: a partial, unacknowledged row.
    with models.open("ab") as handle:
        handle.write(b'{"stable_id": "m3", "record_re')

    torn_identity = census_identities(records_root)["models.shard"]
    assert torn_identity == clean, "an unacknowledged fragment must not be attested"
    assert _evidence_path(models).is_file()
    assert models.read_bytes() == durable

    # And the row-index authority agrees: the fragment never occupies a row.
    assert len(source_rows(models)) == int(clean["rows"])
