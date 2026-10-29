"""Vocabulary gates: fixed 50-family calibration and independent 40-family trial audit."""

import json
import re
from pathlib import Path

import pytest

from menagerie.pipeline import DataLayout, describe_jev, review
from menagerie.pipeline.records import DescribeRecord, from_dict, to_dict


def _answers(folder: Path, ids: list[str]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "jev.json").write_text(json.dumps({"state": "done", "out": "journal.jsonl"}))
    (folder / "journal.jsonl").write_text("".join(json.dumps({"id": fid, "status": "ok", "answers": {
        "domain": {"choice": "audio/speech", "probabilities": {"audio/speech": 0.95, "unknown": 0.05}},
        "is_generative": {"noul": 0.9}}}) + "\n" for fid in ids))


@pytest.mark.parametrize("agree,enabled", [(44, False), (45, True)])
def test_each_field_needs_45_of_50(tmp_path, agree, enabled):
    ids = [f"f-{n}" for n in range(50)]
    _answers(tmp_path, ids)
    (tmp_path / "labels.jsonl").write_text("".join(json.dumps({"id": fid,
        "domain": "audio/speech" if n < agree else None,
        "is_generative": False}) + "\n" for n, fid in enumerate(ids)))
    result = describe_jev.score(tmp_path, ids, wanted=50)
    assert result["fields"]["domain"] == {"agree": agree, "enabled": enabled}
    assert not result["fields"]["is_generative"]["enabled"]


def test_duplicate_or_short_draw_never_enables(tmp_path):
    ids = [f"f-{n}" for n in range(49)]
    _answers(tmp_path, ids)
    (tmp_path / "labels.jsonl").write_text("".join(json.dumps({"id": fid, "domain": "audio/speech"}) + "\n" for fid in ids))
    for picked, wanted in [(ids, 49), (ids + [ids[0]], 50), (ids, 50)]:
        assert not describe_jev.score(tmp_path, picked, wanted=wanted)["fields"]["domain"]["enabled"]


def test_uncertainty_invalid_value_or_partial_call_falls_back(tmp_path, monkeypatch):
    data = DataLayout(tmp_path / "data")
    monkeypatch.setattr(describe_jev, "enabled_fields", lambda _: frozenset(describe_jev.FIELDS))
    monkeypatch.setattr(describe_jev, "triage", lambda *a, **kw: {"state": "done"})
    monkeypatch.setattr(describe_jev, "answers", lambda _: {"f-a": {
        "domain": ("audio/speech", 0.59, 0.4), "modality": ("audio", 0.7, 0.19),
        "paradigm": (None, 0.95, 0.9), "arch_subclass": ("invented", 0.95, 0.9),
        "is_generative": (True, 0.9, 0.8)}, "outside": {"is_generative": (True, 1.0, 1.0)}})
    assert describe_jev.classify(tmp_path, [{"id": "f-a"}, {"id": "missing"}], data=data) == {
        "f-a": {"is_generative": True}}
    monkeypatch.setattr(describe_jev, "triage", lambda *a, **kw: {"state": "unavailable"})
    assert describe_jev.classify(tmp_path, [{"id": "f-a"}], data=data) == {}
    for invalid in [float("nan"), float("inf"), -0.1, 1.1, "0.9"]:
        assert describe_jev._call({"noul": invalid}) == (None, 0.0, 0.0)


def test_purposes_pass_the_broker_argument_schema():
    # The broker's jev triage operation accepts only this --purpose pattern (exit 78 otherwise).
    for purpose in (describe_jev.CLASSIFY_PURPOSE, describe_jev.DRIVE_PURPOSE):
        assert re.fullmatch(r"[A-Za-z0-9._: -]{1,128}", purpose), purpose


def _trial(tmp_path, misses):
    data = DataLayout(tmp_path / "data")
    folder = tmp_path / "trial"
    ids = [f"f-{n}" for n in range(40)]
    folder.mkdir()
    (folder / "out").mkdir()
    (folder / "trial.json").write_text(json.dumps({"families": ids}))
    (folder / "done.json").write_text("{}")
    (folder / "trial-values.json").write_text(json.dumps({fid: {"is_generative": True} for fid in ids}))
    for n, fid in enumerate(ids):
        record = from_dict(DescribeRecord, {"schema": "describe/1", "family": fid, "by": "test",
            "prompt": "0" * 8, "inputs": [], "is_generative": True, "grounding_state": "grounded"})
        path = data.describe(fid)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(to_dict(record)))
        (folder / "out" / f"{fid}.json").write_text(json.dumps({"is_generative": n >= misses}))
    path = data.root / "review/describe-drive.jsonl"
    path.parent.mkdir()
    path.write_text(json.dumps({"batch": "drive", "wanted": 50, "families": 50, "complete": True,
        "fields": {"is_generative": {"agree": 45, "enabled": True}}}) + "\n")
    return data, folder, ids


@pytest.mark.parametrize("misses,disabled", [(4, False), (5, True)])
def test_trial_disables_only_over_ten_percent_and_restores_prose(tmp_path, misses, disabled):
    data, folder, ids = _trial(tmp_path, misses)
    result = describe_jev.audit_trial(ids, data=data, folders=[folder])
    assert result["fields"]["is_generative"] == {"written": 40, "disagree": misses, "disabled": disabled}
    assert ("is_generative" in describe_jev.enabled_fields(data)) == (not disabled)
    assert json.loads(data.describe(ids[0]).read_text())["is_generative"] == (not disabled)
    assert describe_jev.audit_trial(ids, data=data, folders=[folder]) == result
    assert len((data.root / "review/describe-drive.jsonl").read_text().splitlines()) == 2


def test_trial_requires_all_40_families_and_independent_labels(tmp_path):
    data, folder, ids = _trial(tmp_path, 0)
    (folder / "out" / f"{ids[0]}.json").write_text("{}")
    result = describe_jev.audit_trial(ids, data=data, folders=[folder])
    assert result["fields"]["is_generative"]["disagree"] == 1
    with pytest.raises(review.ReviewError, match="40 distinct"):
        describe_jev.audit_trial(ids[:39], data=data, folders=[folder])
    with pytest.raises(review.ReviewError, match="duplicate trial family"):
        describe_jev.audit_trial(ids, data=data, folders=[folder, folder])


def test_interrupted_trial_fallback_reuses_saved_disable_verdict(tmp_path, monkeypatch):
    data, folder, ids = _trial(tmp_path, 5)
    fallback = describe_jev._fallback

    def partial(samples, fields, *, data):
        fallback({fid: samples[fid] for fid in ids[:2]}, fields, data=data)
        raise OSError("interrupted after two records")

    monkeypatch.setattr(describe_jev, "_fallback", partial)
    with pytest.raises(OSError, match="interrupted"):
        describe_jev.audit_trial(ids, data=data, folders=[folder])
    monkeypatch.setattr(describe_jev, "_fallback", fallback)
    result = describe_jev.audit_trial(ids, data=data, folders=[folder])
    assert result["fields"]["is_generative"]["disabled"]
    assert all(json.loads(data.describe(fid).read_text())["is_generative"] is False for fid in ids[:5])


@pytest.mark.parametrize("state", ["ungrounded", "unchecked"])
def test_failed_prose_still_records_disable_verdict(tmp_path, state):
    data, folder, ids = _trial(tmp_path, 5)
    path = data.describe(ids[-1])
    record = json.loads(path.read_text())
    path.write_text(json.dumps({**record, "grounding_state": state}))
    result = describe_jev.audit_trial(ids, data=data, folders=[folder])
    assert (result["settled"], result["grounded"]) == (40, 39)
    assert result["fields"]["is_generative"] == {"written": 39, "disagree": 5, "disabled": True}
    assert not describe_jev.enabled_fields(data)
    assert json.loads(path.read_text())["grounding_state"] == state
