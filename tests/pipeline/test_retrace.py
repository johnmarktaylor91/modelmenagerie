"""The low-priority re-trace feed (``retrace``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from menagerie.pipeline import DataLayout, retrace

ON = {"retrace": {"enabled": True}}
B_OK = {"runners": [{"runner": "B", "action": "ok"}], "stage3_queue": {"eligible": 3, "would_queue": []}}


def test_it_feeds_only_when_b_is_ok_and_no_live_row_was_queued() -> None:
    assert retrace.hold({}, B_OK, paused=None, disk_held=False) == "off"
    assert retrace.hold(ON, B_OK, paused=None, disk_held=False) is None
    assert retrace.hold(ON, B_OK, paused="trace-host paused for TorchLens", disk_held=False).startswith("trace-host")
    assert retrace.hold(ON, B_OK, paused=None, disk_held=True) == "the disk guard holds queueing"
    held_b = {**B_OK, "runners": [{"runner": "B", "action": "held-pause"}]}
    assert retrace.hold(ON, held_b, paused=None, disk_held=False) == "runner B is held-pause"
    live = {**B_OK, "stage3_queue": {"queued": ["m1-forward-eval-x"]}}
    assert retrace.hold(ON, live, paused=None, disk_held=False) == "this pass queued 1 live jobs"
    for queue in ({"held": "disk paused"}, None):
        assert retrace.hold(ON, {**B_OK, "stage3_queue": queue}, paused=None, disk_held=False)


def test_a_bad_switch_is_refused() -> None:
    for bad in ({"retrace": True}, {"retrace": {"enabled": True, "batch": 0}}):
        with pytest.raises(ValueError, match="retrace"):
            retrace.config(bad)


def test_a_batch_skips_rows_a_lane_holds_and_moves_the_fed_ones(tmp_path: Path) -> None:
    (tmp_path / "todo.txt").write_text("".join(f"m{n}\n" for n in range(1, 21)))
    cfg = {"retrace": {"enabled": True, "batch": 4, "folder": str(tmp_path)}}
    calls: list[dict[str, Any]] = []

    def queue(lane: None, **kw: Any) -> list[str]:
        calls.append(kw)
        return [f"{i}-trace-eval-x" for i in kw["ids"] if i != "m4"]  # m4 already has its job

    out = retrace.feed(DataLayout(tmp_path / "data"), cfg, dry_run=False, waiting=lambda s: 0,
                       holds=lambda ids, data: {"m2": "run-1"}, queue=queue)
    assert out == {"fed": ["m1", "m3", "m5"], "skipped": ["m4"], "left": 16}
    assert calls[0]["again"] is True and calls[0]["spool"] == retrace.SPOOL
    assert (tmp_path / "queued.txt").read_text() == "m1\tqueued\nm3\tqueued\nm4\tskipped\nm5\tqueued\n"
    assert (tmp_path / "todo.txt").read_text().split()[:3] == ["m2", "m6", "m7"]
    busy = retrace.feed(DataLayout(tmp_path / "data"), cfg, dry_run=False, waiting=lambda s: 3,
                        holds=lambda ids, data: {}, queue=queue)
    assert busy == {"held": "spool B has 3 waiting", "left": 16} and len(calls) == 1
    dry = retrace.feed(DataLayout(tmp_path / "data"), cfg, dry_run=True, waiting=lambda s: 0,
                       holds=lambda ids, data: {}, queue=queue)
    assert dry == {"would_feed": ["m2", "m6", "m7", "m8"], "left": 16} and len(calls) == 1
    (tmp_path / "todo.txt").write_text("")
    assert retrace.feed(DataLayout(tmp_path / "data"), cfg, dry_run=False) == {"left": 0}
