"""A tick pass reads each packet, the calibration rows and the stage 3 folders once, not per use."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

import menagerie.pipeline as pipeline
from menagerie.pipeline import DataLayout, calibration, stage3


def _age(path: Path, seconds: float = 60) -> None:
    then = time.time() - seconds
    os.utime(path, (then, then))


class _Rec:
    pass


@pytest.fixture
def reads(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    seen: list[Path] = []
    monkeypatch.delenv(pipeline.JSON_CACHE_ENV, raising=False)
    monkeypatch.setattr(pipeline, "_CACHED", {})
    monkeypatch.setattr(pipeline, "_KEPT", pipeline._Kept())
    monkeypatch.setattr(pipeline, "read_json", lambda cls, path: seen.append(path) or path.read_text())
    return seen


def test_an_unchanged_file_is_parsed_once(tmp_path: Path, reads: list[Path]) -> None:
    path = tmp_path / "p.json"
    path.write_text("{}")
    _age(path)
    first = pipeline.read_json_cached(_Rec, path)
    assert pipeline.read_json_cached(_Rec, path) is first
    assert reads == [path]


def test_a_changed_file_is_parsed_again(tmp_path: Path, reads: list[Path]) -> None:
    path = tmp_path / "p.json"
    path.write_text("{}")
    _age(path, 120)
    pipeline.read_json_cached(_Rec, path)
    path.write_text('{"a": 1}')
    _age(path, 60)
    pipeline.read_json_cached(_Rec, path)
    assert reads == [path, path]


def test_a_file_written_just_now_is_never_kept(tmp_path: Path, reads: list[Path]) -> None:
    path = tmp_path / "p.json"
    path.write_text("{}")
    pipeline.read_json_cached(_Rec, path)
    pipeline.read_json_cached(_Rec, path)
    assert reads == [path, path]


def test_the_next_process_starts_from_the_kept_parses(tmp_path: Path, reads: list[Path],
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    kept, a, b = tmp_path / "cache" / "records.pickle", tmp_path / "a.json", tmp_path / "b.json"
    for path in (a, b):
        path.write_text(path.stem)
        _age(path)
    monkeypatch.setenv(pipeline.JSON_CACHE_ENV, str(kept))
    assert [pipeline.read_json_cached(_Rec, p) for p in (a, b)] == ["a", "b"]
    pipeline._KEPT.save()
    b.unlink()

    def next_process() -> None:
        monkeypatch.setattr(pipeline, "_CACHED", {})
        monkeypatch.setattr(pipeline, "_KEPT", pipeline._Kept())

    next_process()
    assert pipeline.read_json_cached(_Rec, a) == "a" and reads == [a, b]  # not parsed again
    a.write_text("A")
    _age(a, 30)
    assert pipeline.read_json_cached(_Rec, a) == "A" and reads == [a, b, a]  # changed: parsed
    pipeline._KEPT.save()
    next_process()
    pipeline._KEPT.start()
    assert set(pipeline._CACHED) == {(f"{_Rec.__module__}.{_Rec.__qualname__}", str(a))}  # b gone
    monkeypatch.setattr(pipeline, "_records_version", lambda: "another record code")
    next_process()
    pipeline._KEPT.start()
    assert pipeline._CACHED == {}


def test_calibration_rows_are_parsed_once_per_change(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    data = DataLayout(tmp_path)
    path = tmp_path / "review" / "calibration.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"role": "r", "run": 1}) + "\n")
    _age(path)
    parsed = []
    real = json.loads
    monkeypatch.setattr(calibration.json, "loads", lambda s: parsed.append(s) or real(s))
    assert calibration.rows(data) == calibration.rows(data) == [{"role": "r", "run": 1}]
    assert len(parsed) == 1
    calibration.rows(data).clear()  # a caller's list is its own
    assert calibration.rows(data) == [{"role": "r", "run": 1}]
    path.write_text(path.read_text() + json.dumps({"role": "r", "run": 2}) + "\n")
    _age(path, 30)
    assert [row["run"] for row in calibration.rows(data)] == [1, 2]


def test_pending_from_one_listing_matches_a_fresh_look(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stage3, "work_dir", lambda name: tmp_path / name)
    sent, collected = stage3._local("sent"), stage3._local("collected")
    for name in ("m1-r1-001", "m1-r1-002", "m10-r1-003", "m2-r1-004", "m1-r2-005"):
        (sent / f"{name}.json").write_text("{}")
    (collected / "m1-r1-002.json").write_text("{}")
    (collected / "m2-r1-004.json").write_text("{}")
    open_ = stage3.open_jobs()
    assert open_ == ["m1-r1-001", "m1-r2-005", "m10-r1-003"]
    for model_id, rev in (("m1", "r1"), ("m10", "r1"), ("m2", "r1"), ("m1", "r2"), ("m3", "r1")):
        assert stage3._pending(model_id, rev, open_) == stage3._pending(model_id, rev)
    assert stage3._pending("m1", "r1", open_) == "m1-r1-001"
    assert stage3._pending("m2", "r1", open_) is None
