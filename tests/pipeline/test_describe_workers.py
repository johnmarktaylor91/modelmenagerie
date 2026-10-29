"""Describe seats on the worker runtimes: the 2b-first room, host choice, start and ended rules."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from menagerie.paths import work_dir
from menagerie.pipeline import admission, describe_workers, mini_placement, mini_relay, tick


@pytest.fixture(autouse=True)
def work(tmp_path, monkeypatch):
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.delenv(describe_workers.WORKERS, raising=False)
    work_dir("leases").mkdir(parents=True)


def _lease(label: str, kind: str, placement: str | None = None) -> None:
    body = {"label": label, "kind": kind, "keys": [label], **({"placement": placement} if placement else {})}
    (work_dir("leases") / f"{label}.json").write_text(json.dumps(body))


def test_room_is_the_slots_2b_left_free_plus_the_describe_seats_already_there():
    _lease("describe-mini-a", "describe", "zm")
    _lease("2b-mini-a", "2b", "zm")
    free = ["mini", "zm", "zm", "zm"]
    room = describe_workers.record_room(free, ["mini", "zm"], held=False)
    assert room == {"zm": 3}  # two slots 2b left, plus the describe seat there
    saved = json.loads((work_dir("tick") / describe_workers.ROOM).read_text())
    assert saved["room"] == {"zm": 3} and describe_workers.room() == {"zm": 3}


def test_no_room_when_2b_filled_every_free_slot_or_is_held():
    _lease("describe-mini-a", "describe", "zm")
    assert describe_workers.record_room(["zm", "zm"], ["zm", "zm"], held=False) == {}
    assert describe_workers.record_room(["zm", "zm"], [], held=True) == {}
    assert describe_workers.record_room([], [], held=False) == {}
    assert describe_workers.hosts(4) == []


def test_hosts_follow_room_seat_cap_and_freshness():
    describe_workers.record_room(["mini", "zm", "zm", "zm"], [], held=False)
    assert describe_workers.hosts(4) == ["zm", "mini"]  # most room first
    _lease("describe-mini-a", "describe", "zm")
    _lease("describe-mini-b", "describe", "zm")
    _lease("describe-mini-c", "describe", "zm")
    _lease("describe-hub", "describe")  # a hub seat holds no worker room
    assert describe_workers.hosts(4) == ["mini"]
    assert describe_workers.hosts(3) == []  # the worker seat cap
    path = work_dir("tick") / describe_workers.ROOM
    old = datetime.now(timezone.utc) - timedelta(seconds=describe_workers.ROOM_FRESH_S + 60)
    path.write_text(json.dumps({"at": old.strftime("%Y-%m-%dT%H:%M:%SZ"), "room": {"mini": 1}}))
    assert describe_workers.room() == {} and describe_workers.hosts(9) == []
    path.write_text("not json")
    assert describe_workers.room() == {}


def test_a_worker_lane_has_ended_once_ingested_or_its_lease_is_gone():
    assert describe_workers.landed({"run": "r1"})  # a hub lane: fleet-run's end is enough
    lane = {"run": "r2", "placement": "zm", "label": "describe-mini-x"}
    _lease("describe-mini-x", "describe", "zm")
    assert not describe_workers.landed(lane)  # TERMINAL, the relay has not pulled it yet
    mini_relay.state_dir("describe-mini-x").mkdir(parents=True)
    (mini_relay.state_dir("describe-mini-x") / "ingested.json").write_text("{}")
    assert describe_workers.landed(lane)
    lost = {"run": "r3", "placement": "zm", "label": "describe-mini-lost"}
    assert describe_workers.landed(lost)  # lease expired and released without an ingest


def _started(host: str) -> mini_placement.Started:
    lease = {"label": f"describe-mini-{host}", "kind": "describe", "run": f"run-{host}"}
    return mini_placement.Started(host, Path("/nonexistent"), lease, ("f-a",), None)


def test_launch_tries_workers_in_turn_and_returns_none_for_the_hub(monkeypatch):
    brief = Path("/nonexistent/BRIEF.md")
    assert describe_workers.launch(brief, ["f-a"], role="model-judgment") is None  # switch off
    monkeypatch.setenv(describe_workers.WORKERS, "2")
    monkeypatch.setattr(tick, "config", lambda: {"mini_workers": {"zm": {"cap": 3}, "mini": {"cap": 1}}})
    describe_workers.record_room(["zm", "zm", "mini"], [], held=False)
    tried = []

    def start(families, brief, *, role, host, cfg, config):
        tried.append((host, config))
        if host == "zm":
            raise admission.Refused("no_admission", "worker zm: held: disk guard warn on the trace host")
        return _started(host)

    monkeypatch.setattr(mini_placement, "start_describe", start)
    monkeypatch.setattr(mini_placement, "finish", lambda started: started.lease["run"])
    assert describe_workers.launch(brief, ["f-a"], role="model-judgment") == (
        "run-mini", "mini", "describe-mini-mini")
    assert tried == [("zm", {"cap": 3}), ("mini", {"cap": 1})]  # most room first, then the next
    monkeypatch.setattr(mini_placement, "start_describe", lambda *a, **kw: (_ for _ in ()).throw(
        admission.Refused("no_admission", "refused")))
    assert describe_workers.launch(brief, ["f-a"], role="model-judgment") is None


def test_an_unproven_start_holds_the_batch_instead_of_starting_twice(monkeypatch):
    monkeypatch.setenv(describe_workers.WORKERS, "1")
    monkeypatch.setattr(tick, "config", lambda: {"mini_workers": {"zm": {"cap": 3}}})
    describe_workers.record_room(["zm"], [], held=False)

    def start(*a, **kw):
        raise subprocess.TimeoutExpired("run-on", 60)

    monkeypatch.setattr(mini_placement, "start_describe", start)
    with pytest.raises(admission.Refused, match="worker_uncertain"):
        describe_workers.launch(Path("/nonexistent/BRIEF.md"), ["f-a"], role="model-judgment")
