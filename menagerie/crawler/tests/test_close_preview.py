"""D-20 (LC-7): per-leg read-only close preview / shadow readiness.

8-11h of 200a was lost to end-of-leg refusals whose evidence rows had been on
disk for hours. The preview makes them legible at leg START: one per-leg
``close-preview`` operational event, derived through the DX-24 probe path --
the exact production boundary predicate, one implementation, zero drift.
Report-only in every direction: no recovery writes, never a leg killer, never
a replacement for the final replay, and per the PD-1 KERNEL_DECISION the
optional ``leg-partial.v1`` checkpoint it aligns with is never a licence
input.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from menagerie.crawler.constants import OperationalEventKind
from menagerie.crawler.driver_contracts import DriverPaths
from menagerie.crawler.intake import create_intake_snapshot
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.status import PartitionError, assert_partition
from menagerie.crawler.tests.campaign_binding_support import bind_snapshot_to_campaign
from menagerie.crawler.tests.test_slice_f_driver import _driver, _snapshot, _write_jsonl
from menagerie.crawler.tools.shadow_close_probe import run_probe


def _records_digest(records_root: Path) -> dict[str, str]:
    """Byte-level digest of every file under the records root."""

    return {
        str(path.relative_to(records_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(records_root.rglob("*"))
        if path.is_file()
    }


@pytest.mark.smoke
def test_preview_predicts_the_end_of_leg_refusal(tmp_path: Path) -> None:
    """The preview names, at leg start, the exact members the close refuses.

    Acceptance (tl-ws7 D-20): preview-predicts-end-of-leg-refusal on a root
    whose close would refuse -- here the maximal coverage hole (an intake with
    no records at all). The prediction and the close are the SAME predicate,
    so every would-refuse member the preview names appears verbatim in the
    production refusal.
    """

    snapshot = _snapshot(tmp_path, count=2)
    bind_snapshot_to_campaign(tmp_path, snapshot)
    records_root = tmp_path / "records"

    would_refuse, refusal, payload = run_probe(
        intake=snapshot.root, records_root=records_root, repo_root=tmp_path
    )
    assert would_refuse is True
    assert payload["partition_valid"] is False
    with pytest.raises(PartitionError) as caught:
        assert_partition((item.stable_id for item in snapshot.items), ())
    for item in snapshot.items:
        assert item.stable_id in str(refusal)
        assert item.stable_id in str(caught.value)


@pytest.mark.smoke
def test_preview_is_read_only_and_triggers_no_recovery_writes(tmp_path: Path) -> None:
    """Report mode leaves every byte under the records root untouched.

    The design note's stated trap: ``materialize_close_sources`` runs at leg
    OPEN and writes; the preview must reach the derivation WITHOUT the open
    path. Byte-compare the whole root across a preview run.
    """

    snapshot = _snapshot(tmp_path, count=2)
    bind_snapshot_to_campaign(tmp_path, snapshot)
    assert _driver(tmp_path, snapshot).run().status == "complete"
    records_root = tmp_path / "records"
    before = _records_digest(records_root)
    would_refuse, _refusal, _payload = run_probe(
        intake=snapshot.root, records_root=records_root, repo_root=tmp_path
    )
    assert would_refuse is False
    assert _records_digest(records_root) == before


@pytest.mark.smoke
def test_the_driver_emits_one_close_preview_event_per_leg(tmp_path: Path) -> None:
    """The per-leg instrument: one durable close-preview event at leg start.

    The event rides the operational ledger with per-occurrence identity, a
    verdict from the closed {would-close, would-refuse, preview-error} set,
    and the generation it previewed for. It is disclosure, never authority.
    """

    snapshot = _snapshot(tmp_path, count=2)
    bind_snapshot_to_campaign(tmp_path, snapshot)
    driver = _driver(tmp_path, snapshot)
    assert driver.run().status == "complete"
    operational = driver.paths.operational_ledger
    events = [
        json.loads(line)
        for line in operational.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    previews = [
        event
        for event in events
        if event.get("event_kind") == OperationalEventKind.CLOSE_PREVIEW.value
    ]
    assert [event["details"]["at"] for event in previews] == ["leg-start", "leg-exit"]
    details = previews[0]["details"]
    assert details["verdict"] in {"would-close", "would-refuse", "preview-error"}
    assert details["generation_id"], "the preview names the generation it previewed"
    # A fresh leg's members are all still missing at leg start: the honest
    # verdict is would-refuse (the leg's whole job is to resolve them), and
    # the preview discloses WHICH members -- that list going stale-nonempty
    # near leg end is the 200a signal this instrument exists for.
    assert details["verdict"] in {"would-refuse", "would-close"}
    if details["verdict"] == "would-refuse":
        assert previews[0]["status"] == "degraded"
    # DX-26: the leg-exit twin is the immutable nonterminal leg-partial.v1
    # checkpoint; a lawfully CLOSED leg carries it too ("every leg exit").
    assert previews[1]["details"]["checkpoint"] == "leg-partial.v1"
    assert previews[1]["details"]["exit_class"] == "close_minted"


@pytest.mark.smoke
def test_a_broken_preview_is_disclosed_and_never_kills_the_leg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Report-only in the failure direction too: preview-error, leg completes.

    A probe that raises must become a DEGRADED preview-error event while the
    leg closes normally -- the preview may never kill a leg.
    """

    import menagerie.crawler.tools.shadow_close_probe as probe_module

    def _boom(**_kwargs: object) -> tuple[bool, str, dict]:
        raise RuntimeError("injected preview fault")

    monkeypatch.setattr(probe_module, "run_probe", _boom)
    snapshot = _snapshot(tmp_path, count=2)
    bind_snapshot_to_campaign(tmp_path, snapshot)
    driver = _driver(tmp_path, snapshot)
    result = driver.run()
    assert result.status == "complete"
    events = [
        json.loads(line)
        for line in driver.paths.operational_ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    previews = [
        event
        for event in events
        if event.get("event_kind") == OperationalEventKind.CLOSE_PREVIEW.value
    ]
    assert [event["details"]["at"] for event in previews] == ["leg-start", "leg-exit"]
    for preview in previews:
        assert preview["details"]["verdict"] == "preview-error"
        assert preview["status"] == "degraded"


@pytest.mark.smoke
def test_preview_resolves_the_manifest_under_a_nested_production_intake(tmp_path: Path) -> None:
    """The preview sees the real partition under the production repository layout.

    P5-R3-SOL-02: production intakes live at
    ``<repo>/menagerie/crawler/records/intake``, so deriving the probe's repo
    root as ``intake_root.parent`` doubled the path
    (``.../records/intake/menagerie/crawler/records/campaigns.json``) and every
    per-leg preview degraded to a manifest-resolution error with null partition
    fields -- the instrument was blind on exactly the layout it ships under.
    The driver must hand the probe the canonical repository root, so the
    preview carries non-null partition fields and no manifest-resolution error.
    """

    records_root = tmp_path / "menagerie" / "crawler" / "records"
    master = tmp_path / "master.jsonl"
    deferred = tmp_path / "deferred.jsonl"
    _write_jsonl(
        master,
        [{"name": f"Example{index}", "zoo": "fixtures", "variant": "base"} for index in range(2)],
    )
    _write_jsonl(deferred, [])
    snapshot = create_intake_snapshot(master, deferred, records_root / "intake")
    bind_snapshot_to_campaign(tmp_path, snapshot)
    driver = _driver(
        tmp_path,
        snapshot,
        paths_override=DriverPaths(
            tmp_path / "runtime",
            snapshot.root,
            LedgerPaths(
                records_root / "models" / "current-shard.jsonl",
                records_root / "attempts" / "local.jsonl",
                records_root / "gates" / "current-shard.jsonl",
            ),
        ),
    )
    # Under the production ancestry (records/crawler/menagerie) the reducer's
    # env-generation exactness validation ARMS (it is skipped in flat fixture
    # layouts), and the fake environment lane has no committed exact artifacts
    # under <repo>/menagerie/crawler/envs -- so the members fault-quarantine
    # and the leg lawfully closes with deferrals. That fail-closed arm is not
    # this test's subject; both closed ``complete`` spellings are accepted.
    assert driver.run().status in {"complete", "complete-with-deferrals"}
    events = [
        json.loads(line)
        for line in driver.paths.operational_ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    previews = [
        event
        for event in events
        if event.get("event_kind") == OperationalEventKind.CLOSE_PREVIEW.value
    ]
    assert [event["details"]["at"] for event in previews] == ["leg-start", "leg-exit"]
    for preview in previews:
        details = preview["details"]
        assert details["verdict"] in {"would-close", "would-refuse"}, details
        assert details["partition_valid"] is not None, details
        assert "no campaign manifest" not in str(details.get("refusal") or ""), details
