"""D-37: the 200a corpus as a committed detection-regression benchmark.

Every known 200a incident replays through the REAL ``build_snapshot`` +
delivery layer, and CI asserts each is NAMED within its bounded sim-latency;
the monitor's numbers equal the close row's partition numbers at the close
instant (the assertion that would have caught the valve-omission class in
one line); and the alert stream is two-sided -- at most the message budget
overall AND at least one message inside every incident window. A monitor
change that regresses detection on 200a fails HERE, before it ships: the
200a monitor itself was gated only on fixture-shaped counting math and met
its first wedge in production.

The committed fixture is a field-projection of the 200a records root
(``tools/detection_replay.project_records``); regenerate ONLY via
``python -m menagerie.crawler.tools.detection_replay --src <root> --dest
menagerie/crawler/tests/fixtures/replay_200a`` and re-pin the digests in the
same reviewed change. The B2-X-V25 arm (env-gated on ``MENAGERIE_200A_ROOT``)
diffs real full-fidelity time-sliced snapshots against the projection, so the
reduction can never silently change what the monitor would have seen.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from menagerie.crawler.tools.detection_replay import (
    ReplayHarness,
    first_alert_in_window,
    project_records,
)

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "replay_200a"
MANIFEST = json.loads((FIXTURE_ROOT / "manifest.json").read_text(encoding="utf-8"))


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@pytest.mark.smoke
def test_fixture_digests_match_the_manifest() -> None:
    """The benchmark is digest-pinned: silent fixture edits fail loudly."""

    for relative, expected in MANIFEST["files"].items():
        observed = hashlib.sha256((FIXTURE_ROOT / relative).read_bytes()).hexdigest()
        assert observed == expected, f"fixture drift in {relative}"
    on_disk = {
        path.relative_to(FIXTURE_ROOT).as_posix()
        for path in FIXTURE_ROOT.rglob("*.jsonl")
    }
    assert on_disk == set(MANIFEST["files"]), "unmanifested fixture file(s)"


@pytest.fixture(scope="module")
def campaign_replay(tmp_path_factory) -> tuple[list, object]:
    """One full-campaign replay shared by the assertions below."""

    work = tmp_path_factory.mktemp("replay-work")
    harness = ReplayHarness(FIXTURE_ROOT, work)
    replay = MANIFEST["replay"]
    alerts, final, _state = harness.replay(
        _parse(replay["start"]),
        _parse(replay["end"]),
        step=timedelta(minutes=replay["step_minutes"]),
    )
    return alerts, final


def test_every_known_incident_is_named_within_budget(campaign_replay) -> None:
    alerts, _final = campaign_replay
    for incident in MANIFEST["incidents"]:
        alert = first_alert_in_window(
            alerts,
            kind_prefix=incident["expected_kind_prefix"],
            window_start=_parse(incident["window_start"]),
            window_end=_parse(incident["window_end"]),
        )
        assert alert is not None, (
            f"incident {incident['name']} was never named"
            f" ({incident['expected_kind_prefix']}) inside its window"
        )
        assert alert.at <= _parse(incident["named_by"]), (
            f"incident {incident['name']} named at {alert.at.isoformat()},"
            f" past the {incident['named_by']} detection budget"
        )


def test_message_stream_is_two_sided(campaign_replay) -> None:
    """<= budget overall AND >= 1 message inside every incident window.

    A monitor silenced into its budget is as broken as one that spams past
    it; both directions are pinned.
    """

    alerts, _final = campaign_replay
    budget = MANIFEST["message_budget_total"]
    assert len(alerts) <= budget, (
        f"{len(alerts)} alerts exceed the {budget} message budget --"
        " a standing condition is paging on levels, not transitions"
    )
    for incident in MANIFEST["incidents"]:
        window_start = _parse(incident["window_start"])
        window_end = _parse(incident["window_end"])
        assert any(
            window_start <= alert.at <= window_end for alert in alerts
        ), f"ZERO messages inside the {incident['name']} incident window"


def test_completion_fires_exactly_once(campaign_replay) -> None:
    alerts, _final = campaign_replay
    completions = [alert for alert in alerts if alert.kind == "completion"]
    assert len(completions) == 1


def test_monitor_numbers_equal_the_close_row_at_the_close_instant(
    tmp_path: Path,
) -> None:
    """The convergence oracle (OB-Q1): monitor == close partition numbers.

    The valve-omission class (m10502 invisible; rung_report under-counting)
    dies in this one assertion: at the close instant the live view must say
    exactly what the close row proved.
    """

    harness = ReplayHarness(FIXTURE_ROOT, tmp_path / "work")
    snapshot = harness.snapshot_at(_parse(MANIFEST["source"]["closed_at"]))
    expected = MANIFEST["close_convergence"]
    observed = {
        "roster_total": snapshot.roster_total,
        "done": snapshot.done,
        "runs": snapshot.runs,
        "failed": snapshot.failed,
        "skipped": snapshot.skipped,
        "deferred": snapshot.deferred,
        "parked": snapshot.parked,
        "in_flight": snapshot.in_flight,
        "unrecognized": snapshot.unrecognized,
        "net_terminals": snapshot.net_terminals,
        "complete": snapshot.complete,
    }
    assert observed == {
        key: expected[key] for key in observed
    }, "monitor diverged from the 200a close row at the close instant"
    cross = expected["close_row_cross_check"]
    assert observed["roster_total"] == cross["intake"]
    assert observed["parked"] == cross["machine_deferred_members"]


@pytest.mark.skipif(
    not os.environ.get("MENAGERIE_200A_ROOT"),
    reason="B2-X-V25 fidelity arm needs MENAGERIE_200A_ROOT (the full 200a records root)",
)
def test_projection_fidelity_against_the_real_corpus(tmp_path: Path) -> None:
    """B2-X-V25: real build_snapshot over full-fidelity slices == projection.

    Runs on hosts holding the real 200a root; proves the committed reduction
    changes nothing the monitor consumes at three probe instants (mid-wedge,
    mid-campaign, close).
    """

    real_root = Path(os.environ["MENAGERIE_200A_ROOT"])
    projected_real = tmp_path / "projected-real"
    digests = project_records(real_root, projected_real)
    assert digests == MANIFEST["files"], (
        "the committed fixture is not the projection of MENAGERIE_200A_ROOT"
    )
    full_harness = ReplayHarness(_full_fidelity_copy(real_root, tmp_path), tmp_path / "w1")
    fixture_harness = ReplayHarness(FIXTURE_ROOT, tmp_path / "w2")
    probes = (
        "2026-08-20T12:00:00+00:00",
        "2026-08-22T00:00:00+00:00",
        MANIFEST["source"]["closed_at"],
    )
    volatile = {
        "generated_at",
        "authority_root",
        "disk_free_gb",
        "last_activity_age_minutes",
        "notify_transport_disposition",
        "corrupt_lines",
    }
    from dataclasses import asdict

    for probe in probes:
        instant = _parse(probe)
        real = asdict(full_harness.snapshot_at(instant))
        projected = asdict(fixture_harness.snapshot_at(instant))
        for key in volatile:
            real.pop(key, None)
            projected.pop(key, None)
        assert real == projected, f"projection infidelity at {probe}"


def _full_fidelity_copy(real_root: Path, tmp_path: Path) -> Path:
    """Copy the REAL (unprojected) monitor-consumed ledgers for slicing."""

    import shutil

    dest = tmp_path / "real-full"
    for relative in MANIFEST["files"]:
        src = real_root / relative
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copyfile(src, target)
        else:
            target.write_text("", encoding="utf-8")
    return dest
