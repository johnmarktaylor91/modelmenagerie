"""D1-F5: the pending-metadata re-admission predicate is narrowed to runs records.

The re-admission arm exists for exactly one documented producer: the R1/R2
catch-up path, where a ``runs`` record with pending authored metadata awaits its
metadata_batch gate. The M1 execution-failure fidelity waiver creates a second
pending-metadata producer -- machine-dead failure terminals -- and the old
predicate (``authored_metadata_state == "pending"``, nothing else) would re-enter
every such model into the work loop on every leg, including after a rollback
from the future scheduler to this wave driver. The narrowed predicate is the
ONE spelling both consult: ``pending AND status.kind == "runs"``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from menagerie.crawler.authority import build_authority_context
from menagerie.crawler.driver import _pending_metadata_readmission
from menagerie.crawler.recordio import scan_jsonl

pytestmark = pytest.mark.smoke


def test_predicate_readmits_only_pending_runs_records() -> None:
    """Import-dead and author-dead pending terminals stay closed; runs re-enter."""

    assert _pending_metadata_readmission(
        {"authored_metadata_state": "pending", "status": {"kind": "runs"}}
    )
    for stage in ("import", "environment", "constructor", "input", "forward", "author"):
        assert not _pending_metadata_readmission(
            {
                "authored_metadata_state": "pending",
                "status": {"kind": "failed", "stage": stage, "code": f"failed:{stage}"},
            }
        )
    assert not _pending_metadata_readmission(
        {"authored_metadata_state": "accepted", "status": {"kind": "runs"}}
    )
    # Fail-closed on malformed shapes: nothing re-enters by accident.
    assert not _pending_metadata_readmission({})


def test_machine_dead_pending_terminal_is_not_readmitted_by_the_wave_driver(
    tmp_path: Path,
) -> None:
    """A waived machine-dead terminal never re-enters the author lane per leg.

    RED on the pre-narrowing predicate: the seeded failure terminal with pending
    metadata re-entered ``phase_work`` and bought a fresh author session on the
    very next leg.
    """

    from menagerie.crawler.reducer import CanonicalReducer
    from menagerie.crawler.tests.test_fidelity_waiver_kernel import (
        _execution_failed_attempt,
        _waiver_terminal,
    )
    from menagerie.crawler.tests.test_slice_f_driver import (
        FakeAuthor,
        FakeChecker,
        _driver,
        _paths,
        _snapshot,
    )

    snapshot = _snapshot(tmp_path, count=1)
    stable_id = snapshot.items[0].stable_id
    paths = _paths(tmp_path, snapshot)
    context = build_authority_context(
        active_intake_snapshot_id=snapshot.snapshot_id,
        active_intake_snapshot_sha256=snapshot.snapshot_sha256,
        intake_rows=(item.to_dict() for item in snapshot.items),
        author_model="claude",
        author_version="test",
        checker_model="codex",
        checker_version="test",
    )
    attempt = _execution_failed_attempt(
        stable_id,
        stage="environment",
        reason_code="build-failed",
        env_generation=None,
    )
    terminal = _waiver_terminal(
        stable_id,
        stage="environment",
        reason_code="build-failed",
        env_generation=None,
    )
    paths.ledgers.models.parent.mkdir(parents=True, exist_ok=True)
    with CanonicalReducer(paths.ledgers, context) as reducer:
        reducer.append_attempt(attempt)
        reducer.append_model(reducer.prepare_model(terminal))
        assert stable_id in reducer.current_records

    author = FakeAuthor()
    from menagerie.crawler.partition_proof import PartitionProofError

    with pytest.raises(PartitionProofError, match="nothing to carry from"):
        _driver(tmp_path, snapshot, author=author, checker=FakeChecker()).run()

    # The terminal partition is already complete; the leg must not re-buy the
    # author/gate lane for a closed machine-dead terminal.
    assert author.calls == {}
    models = scan_jsonl(paths.ledgers.models)
    assert len(models) == 1
    assert models[0]["status"]["code"] == "failed:environment"
    assert models[0]["authored_metadata_state"] == "pending"
