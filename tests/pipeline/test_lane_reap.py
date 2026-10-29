"""``lane_reap``: which run-on answers end a lane's reaping and which wait for a later pass."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from menagerie.pipeline import lane_reap

NOW = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize(("output", "want"), [
    ("REAPED\t2b-mini-a\t349298688\tfailed-ttl\n", "reaped"),
    ("KEEP:running 2b-mini-a\n", "retry"),
    ("KEEP:pull-before-end 2b-mini-a\nPULL-DUE 2b-mini-a\n", "retry"),
    ("KEEP:outputs-unverified 2b-mini-a\nHELD\t2b-mini-a\t1\toutputs-unverified\n", "KEEP:outputs-unverified"),
    ("KEEP:worktree-missing 2b-mini-a\n", "KEEP:worktree-missing"),
    ("run-on reap: another reap is running; skipping this one\n", "busy"),
    ("KEEP:timeout 2b-mini-a", "retry"),
    ("(no jobs)\n", "gone"),
    ("KEEP:running 2b-mini-ab\n", "gone"),  # another job's line says nothing about this one
])
def test_verdict(output: str, want: str) -> None:
    assert lane_reap.verdict("2b-mini-a", output) == want


def _ready(root: Path, briefs: Path, label: str, age_s: float = 60) -> None:
    folder = root / label
    folder.mkdir(parents=True)
    (folder / "staged.json").write_text(json.dumps({"paths": {str(briefs / label): "/w"}}))
    marker = folder / "ingested.json"
    marker.write_text(json.dumps({"status": "ingested"}))
    when = NOW.timestamp() - age_s
    os.utime(marker, (when, when))


def test_a_busy_reaper_ends_the_pass_and_a_retry_is_asked_again(tmp_path: Path) -> None:
    root, briefs = tmp_path / "minis", tmp_path / "2b"
    for age, label in enumerate(("2b-mini-a", "2b-mini-b", "2b-mini-c")):
        _ready(root, briefs, label, age_s=60 * (age + 1))  # newest first: a, b, c
    answers = {"2b-mini-a": "KEEP:running 2b-mini-a", "2b-mini-b": "another reap is running"}
    asked: list[str] = []

    def reap(label: str) -> tuple[int, str]:
        asked.append(label)
        return 0, answers.get(label, f"REAPED\t{label}\t1\tsettled-ttl")

    held = {"2b-mini-a", "2b-mini-b", "2b-mini-c"}
    out = lane_reap.step(root, tmp_path / "leases", briefs, set(), now=NOW, reap=reap, held=held,
                         deadline=time.monotonic() + 600)
    assert asked == ["2b-mini-a", "2b-mini-b"] and out["retry"] == 1 and out["busy"] == 1
    assert not any((root / label / lane_reap.DONE).exists() for label in asked)
    answers.clear()
    lane_reap.step(root, tmp_path / "leases", briefs, set(), now=NOW, reap=reap, held=held,
                   deadline=time.monotonic() + 600)
    assert sorted(asked[2:]) == ["2b-mini-a", "2b-mini-b", "2b-mini-c"]


def test_no_call_starts_near_the_pass_deadline(tmp_path: Path) -> None:
    root, briefs = tmp_path / "minis", tmp_path / "2b"
    _ready(root, briefs, "2b-mini-a")
    out = lane_reap.step(root, tmp_path / "leases", briefs, set(), now=NOW,
                         reap=lambda label: pytest.fail("reaped past the deadline"),
                         held={"2b-mini-a"}, deadline=time.monotonic() + 10)
    assert out == {"ready": 1, "waiting": {"lease": 0, "uncommitted": 0}}


def test_uncommitted_keys_name_the_families_and_ids_of_changed_lane_files(tmp_path: Path) -> None:
    data = tmp_path / "menagerie" / "data"
    (data / "models" / "00").mkdir(parents=True)
    (data / "models" / "00" / "m1.json").write_text(json.dumps({"id": "m1", "family": "f-one"}))
    changed = ["menagerie/data/models/00/m1.json", "menagerie/data/adapters/f-two--v2.py",
               "menagerie/data/research/f-three.json", "menagerie/data/ledger/2026-10.jsonl",
               "site/src/x.ts"]
    assert lane_reap.uncommitted_keys(tmp_path, "menagerie/data", changed) == {
        "m1", "f-one", "f-two", "f-three"}


def test_only_jobs_run_on_holds_for_their_grace_are_candidates(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLEET_STATE_DIR", str(tmp_path))
    assert lane_reap.held_by_run_on() is None
    (tmp_path / "run-on-held.json").write_text(json.dumps({"hosts": {
        "worker-c": {"2b-mini-a": {"project": "menagerie", "reason": "failed-retention"},
                     "2b-mini-b": {"project": "menagerie", "reason": "outputs-unverified"},
                     "other-x": {"project": "other", "reason": "failed-retention"}},
        "worker-b": {"2b-mini-c": {"project": "menagerie", "reason": "empty-pull-retention"}}}}))
    assert lane_reap.held_by_run_on() == {"2b-mini-a", "2b-mini-c"}
    out = lane_reap.step(tmp_path / "minis", tmp_path / "leases", tmp_path / "2b", set(), now=NOW,
                         reap=lambda label: pytest.fail("no held record"), held=None,
                         deadline=time.monotonic() + 600)
    assert "skipped" in out
