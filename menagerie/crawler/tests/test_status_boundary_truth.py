"""BD-1: the boundary view tells the LEDGER's truth, not the projection's.

The step-100 close (AUDIT100 T1 all three labs) ran the playbook's own boundary
form against 67 settled heads and got ``exit 1`` with 13-24 members reported
**missing** -- every one of them holding a recorded terminal, 7-13 of them
holding ``runs`` awards -- plus 16 of 17 terminal-refusal parks reported LIVE
for models that had terminalized hours earlier, and a funnel with
``status:runs: 0``. Nothing was lost: coverage, park release, and the deferral
accounting were all keyed on the dependency-currency PROJECTION, so a head the
projection could not vouch for read as NONEXISTENT.

The four defects pinned here, per the BLOCKING row-5 remedy:

1. Coverage comes from LEDGER heads. A member holding any recorded revision is
   never ``missing``.
2. A member holding a recorded terminal is never machine-deferral-CLAIMED; its
   valve row is at best ``superseded``.
3. Park release is evaluated against the ledger head, so a park whose model
   later recorded a revision is consumed -- no phantom park inventory.
4. Staleness is its own bucket, ``stale:<axis>``, and it still FAILS CLOSED
   (S2-opus binding rider): a non-empty stale bucket refuses with the axis
   named, because raw heads include records the projection rejects for good
   reason. The tripwire keeps firing; it stops lying about why.

The staleness census also gains ``awards_stale_on_live_basis`` (T6-opus-13):
one line that would have made the award-eating incident visible before
resume-4 launched instead of after.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler import authority as authority_module
from menagerie.crawler.cli import EXIT_OK
from menagerie.crawler.cli import main as cli_main
from menagerie.crawler.driver import DriverPaths
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.refusal_parks import (
    append_refusal_park_row,
    build_refusal_park_row,
    refusal_park_path,
)
from menagerie.crawler.identity import stable_hash
from menagerie.crawler.status import (
    REPLAY_REFUSED_STALE_AXIS,
    PartitionError,
    assert_partition,
    awards_stale_on_live_basis,
    partition_report,
    stale_axis,
)
from menagerie.crawler.tests.campaign_binding_support import bind_snapshot_to_campaign
from menagerie.crawler.tests.test_slice_f_driver import _driver, _snapshot


def _status(tmp_path: Path, snapshot_root: Path, *extra: str) -> int:
    """Run ``crawler status`` against an isolated records root."""

    return cli_main(
        [
            "--repo-root",
            str(tmp_path),
            "status",
            "--intake",
            str(snapshot_root),
            "--records-root",
            str(tmp_path / "records"),
            *extra,
        ]
    )


def _records_paths(tmp_path: Path) -> LedgerPaths:
    """Return the fixture's canonical ledger paths."""

    return LedgerPaths(
        tmp_path / "records" / "models" / "current-shard.jsonl",
        tmp_path / "records" / "attempts" / "local.jsonl",
        tmp_path / "records" / "gates" / "current-shard.jsonl",
    )


def _closed_leg(tmp_path: Path, count: int = 3) -> Any:
    """Drive one fixture leg to a complete close and return its snapshot."""

    snapshot = _snapshot(tmp_path, count=count)
    bind_snapshot_to_campaign(tmp_path, snapshot)
    paths = DriverPaths(tmp_path / "runtime", snapshot.root, _records_paths(tmp_path))
    result = _driver(tmp_path, snapshot, paths_override=paths).run()
    assert result.status == "complete"
    return snapshot


def _rotate_author_dispatcher(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rotate ONE shipped-policy axis exactly as a mid-leg hotfix does.

    ``author_dispatcher_identity`` is the content digest of the shipped
    dispatcher component, and it was the single largest stale axis in the
    step-100 census (10 of 42 stale members). Pointing the component name at a
    different shipped file reproduces "author_dispatch.py changed between legs"
    without editing any production byte -- the identity the next projection
    derives simply differs from the one every recorded vector carries.
    """

    monkeypatch.setattr(authority_module, "AUTHOR_DISPATCHER_COMPONENT", "author_pool.py")


@pytest.mark.smoke
def test_stale_axis_classifies_the_closed_vector_axes() -> None:
    """The stale reason is mapped to a CLOSED axis vocabulary, fail-closed."""

    assert stale_axis("dependency-vector: stale author_dispatcher_identity") == (
        "author_dispatcher_identity"
    )
    assert stale_axis("dependency-vector: stale environment_generation") == (
        "environment_generation"
    )
    assert stale_axis("intake: stable ID is absent from active snapshot") == "intake"
    assert stale_axis("family-authority: binding belongs to another stable ID") == (
        "family-authority"
    )
    # An axis name that is NOT in the closed dependency-vector set, and any
    # unrecognized reason, must never invent a bucket that reads as understood.
    assert stale_axis("dependency-vector: stale invented_identity") == "unclassified"
    assert stale_axis("ReductionError: something nobody enumerated") == "unclassified"
    # The arm a rotation actually trips first: the replay re-derives the vector
    # from live authority BEFORE the currency check, and names the axes inline.
    assert (
        stale_axis(
            "model dependency vector contradicts reducer derivation for m1: "
            "author_dispatcher_identity: recorded='sha256:aaa' derived='sha256:bbb'"
        )
        == "author_dispatcher_identity"
    )
    assert (
        stale_axis(
            "model dependency vector contradicts reducer derivation for m1: "
            "recorded vector is NoneType, not a mapping"
        )
        == "dependency-vector"
    )
    # The driver's own fallback label must ROUND-TRIP through the classifier:
    # a head every replayable axis calls current, that the full replay still
    # refused, is attributed to the replay -- not to an invented axis, and not
    # silently to `unclassified`.
    assert (
        stale_axis(f"{REPLAY_REFUSED_STALE_AXIS}: the dependency-currency replay refused")
        == REPLAY_REFUSED_STALE_AXIS
    )


@pytest.mark.smoke
def test_a_ledger_head_is_coverage_even_when_the_projection_drops_it() -> None:
    """T1-opus-02: a stale head is COVERED-and-stale, never missing."""

    head = {
        "stable_id": "m1",
        "record_revision": "sha256:aaa",
        "status": {"code": "runs", "kind": "runs"},
    }
    report = partition_report(
        ("m1",),
        {},
        ledger_heads={"m1": head},
        stale_reasons={"m1": "dependency-vector: stale author_dispatcher_identity"},
    )
    assert report.missing_ids == frozenset()
    assert report.stale_by_axis == {"author_dispatcher_identity": frozenset({"m1"})}
    # Fail-closed, per the S2 binding rider: covered-but-stale is NOT valid.
    assert report.valid is False


@pytest.mark.smoke
def test_a_terminal_holding_member_is_never_machine_deferred(tmp_path: Path) -> None:
    """T1-opus-02(2)/T1-opus-04: the valve row is at best superseded."""

    head = {
        "stable_id": "m1",
        "record_revision": "sha256:aaa",
        "status": {"code": "failed:author", "kind": "failed"},
    }
    park = build_refusal_park_row(
        stable_id="m1",
        work_id="w1",
        name="Example",
        campaign_id="c1-mech",
        run_id="run-1",
        machine_id="host-1",
        created_at="2026-08-13T09:12:51Z",
        refusal_class="ReductionError",
        record_error="model environment generation is absent from active context",
        record_error_type="ReductionError",
        fallback_error_type=None,
        fallback_error=None,
        intended_status_code="failed:author",
        intended_reason_code=None,
        blocked_parent_revision=None,
        candidate_sha256=None,
    )
    report = partition_report(
        ("m1",),
        {},
        machine_deferrals={"m1": park},
        ledger_heads={"m1": head},
        stale_reasons={"m1": "dependency-vector: stale publication_policy_identity"},
    )
    assert report.machine_deferred == {}
    assert report.refusal_parked_ids == frozenset()
    assert report.superseded_deferral_ids == frozenset({"m1"})
    assert report.missing_ids == frozenset()


@pytest.mark.smoke
def test_an_accounted_stale_head_is_disclosed_without_upgrading_or_crashing() -> None:
    """The refusing arm is the stale members NOTHING accounts.

    A stale head whose live valve row is the accounting for the withheld
    RE-DRIVE (the environment-rebuild deferral after a checker-digest rotation)
    keeps exactly the verdict it had before coverage moved to the ledger: the
    close is not ``complete``-clean of deferrals, the state is disclosed, and
    nothing crashes. What changed is that the row is no longer a coverage CLAIM
    and the staleness is no longer invisible -- it is in the census and, for an
    award, in ``awards_stale_on_live_basis``.

    A stale head with NO accounting row is the step-100 population (24 recorded
    terminals reported as phantom coverage holes) and must keep failing closed.
    """

    rebuild_row = {
        "stable_id": "m1",
        "disposition": "deferred:environment-rebuild-required",
        "recheck_hint": "rebuild the routed environment, then re-drive m1",
    }
    rebuild_row["row_sha256"] = stable_hash(dict(rebuild_row))
    heads = {
        "m1": {
            "stable_id": "m1",
            "record_revision": "sha256:aaa",
            "status": {"code": "runs", "kind": "runs"},
        },
        "m2": {
            "stable_id": "m2",
            "record_revision": "sha256:bbb",
            "status": {"code": "runs", "kind": "runs"},
        },
    }
    stale = {
        "m1": "dependency-vector: stale checker_prompt_identity",
        "m2": "dependency-vector: stale checker_prompt_identity",
    }
    report = partition_report(
        ("m1", "m2"),
        {},
        machine_deferrals={"m1": rebuild_row},
        ledger_heads=heads,
        stale_reasons=stale,
    )
    # Both are disclosed as stale, and both awards are counted.
    assert report.stale_ids == frozenset({"m1", "m2"})
    assert awards_stale_on_live_basis(heads, report.stale_ids) == frozenset({"m1", "m2"})
    # m1's row is neither a coverage claim nor "superseded": it defers the re-drive.
    assert report.machine_deferred == {}
    assert report.superseded_deferral_ids == frozenset()
    assert report.stale_deferred == {"deferred:environment-rebuild-required": frozenset({"m1"})}
    # Only the UNACCOUNTED one refuses.
    assert report.unaccounted_stale_ids == frozenset({"m2"})
    assert report.valid is False
    with pytest.raises(PartitionError) as refusal:
        assert_partition(
            ("m1", "m2"),
            {},
            machine_deferrals={"m1": rebuild_row},
            ledger_heads=heads,
            stale_reasons=stale,
        )
    assert "stale:checker_prompt_identity=['m2']" in str(refusal.value)
    assert "m1" not in str(refusal.value)


@pytest.mark.smoke
def test_boundary_gate_reports_a_policy_move_in_the_recheck_census_not_as_staleness(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """RC-2: a policy-axis move is neither a coverage hole nor staleness.

    Before RC-2 this gate named the rotated axis as 3 stale heads (itself a
    fix over 3 phantom missing models). Since RC-2 every settled head replays
    at its recorded basis, so the partition is VALID; the move lands in the
    recheck census (an unmeasured producer-axis move reads soundness) with
    every member listed, and nothing is missing or stale.
    """

    snapshot = _closed_leg(tmp_path)
    _rotate_author_dispatcher(monkeypatch)

    exit_code = _status(tmp_path, snapshot.root, "--verify-partition", "--require-valid-partition")
    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip().splitlines()[-1])

    assert exit_code == 0, captured.err
    assert payload["partition_valid"] is True
    assert payload["missing"] == []
    assert payload["ledger_heads"] == 3
    assert payload["stale"]["count"] == 0
    assert payload["awards_stale_on_live_basis"] == 0
    assert payload["head_status_counts"] == {"runs": 3}
    census = payload["recheck_census"]
    members = sorted(item.stable_id for item in snapshot.items)
    assert census["counts"] == {"needs_recheck": 3, "by_head_kind": {"runs": 3}, "unmeasured": 3, "review_held": 0}
    assert census["by_axis"] == {"author_dispatcher_identity": members}
    assert {row["remedy"] for row in census["needs_recheck"]} == {"re-author"}
    assert payload["configuration_history"] == []


@pytest.mark.smoke
def test_a_park_whose_model_recorded_is_never_a_live_park_again(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """T1-opus-04/T2-opus-07: release keys on the ledger head, not the projection.

    The step-100 shape exactly: a terminal-refusal park written while the
    recording path was broken, the model terminalizing hours later, and the
    park still reported LIVE at close because the head had gone stale.
    """

    snapshot = _closed_leg(tmp_path)
    parked = sorted(item.stable_id for item in snapshot.items)[0]
    append_refusal_park_row(
        refusal_park_path(tmp_path / "records"),
        build_refusal_park_row(
            stable_id=parked,
            work_id="w-earlier",
            name="Example0",
            campaign_id="c1-mech",
            run_id="run-earlier",
            machine_id="host-1",
            created_at="2026-08-13T09:12:51Z",
            refusal_class="ReductionError",
            record_error="model environment generation is absent from active context",
            record_error_type="ReductionError",
            fallback_error_type=None,
            fallback_error=None,
            intended_status_code="failed:author",
            intended_reason_code=None,
            blocked_parent_revision=None,
            candidate_sha256=None,
        ),
    )
    _rotate_author_dispatcher(monkeypatch)

    assert _status(tmp_path, snapshot.root, "--verify-partition") == EXIT_OK
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert "refusal_parks" not in payload
    assert payload["missing"] == []
