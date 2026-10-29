"""Repin evidence, comparison and forward reuse must fail closed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from menagerie.pipeline import DataLayout, repin, repin_slice


def test_private_probe_snapshots_status_before_trace_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    pytest.importorskip("torch")
    from torchlens import user_funcs

    from menagerie.pipeline import repin_checks

    def validate(*args: object, **kwargs: object) -> bool:
        trace = SimpleNamespace(validation_replay_status=SimpleNamespace(state="passed"))
        kwargs["_trace_observer"](trace)
        trace.validation_replay_status.state = "available"
        return True

    monkeypatch.setattr(user_funcs, "_validate_forward_pass_torch", validate)
    result = repin_checks.private_calls()
    assert result["passed"] and result["observer_states"] == ["passed"]


def _baseline() -> list[dict[str, str]]:
    return [{"stable_id": "m1", "name": "toy", "batch": "small", "r1": "passed",
             "r1_n_ops": "4", "r1_hash": "abc", "r1_structure_only": ""}]


def test_missing_or_changed_rows_remain_unexplained() -> None:
    old = _baseline()
    assert repin_slice.compare([], old)[0]["agree"] == "unexplained"
    new = [{"stable_id": "m1", "mode": "trace", "status": "passed",
            "n_ops": 4, "graph_hash": "abc"}]
    assert repin_slice.compare(new, old)[0]["agree"] == "same"
    new[0]["n_ops"] = 5
    assert repin_slice.compare(new, old)[0]["agree"] == "unexplained"
    assert repin_slice.compare(new, old, explanations={"m1": "reproduced fix"})[0]["agree"] == "explained"
    new[0]["status"] = "infra"
    assert repin_slice.compare(new, old, explanations={"m1": "excuse"})[0]["agree"] == "unexplained"


def _proof_dir(tmp_path: Path, day: dict[str, object], slice_a: dict[str, object]) -> Path:
    import hashlib

    baseline = tmp_path / "menagerie" / "data" / "regression" / "slice_a_200_r1.tsv"
    baseline.parent.mkdir(parents=True)
    baseline.write_text("stable_id\n")
    good = {"version": "2.36.0", "source_digest": "source", "passed": True,
            "complete": True, "rows": 200, "unexplained": 0, "smoke": False, "ids": [],
            "repo_data": True,
            "baseline_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest()}
    (tmp_path / "repin-day.json").write_text(json.dumps({**good, **day}))
    (tmp_path / "repin-slice-a.json").write_text(json.dumps({**good, **slice_a}))
    (tmp_path / "repin-conv3.json").write_text(json.dumps(
        {"version": "2.36.0", "source_digest": "source", "passed": True,
         "rows": 13, "threads": 3, "sandbox": True, "repo_data": True}))
    return tmp_path


@pytest.mark.parametrize("delta", [
    {},
    {"smoke": True}, {"complete": False}, {"rows": 3}, {"unexplained": 1},
    {"source_digest": "old"}, {"version": "2.35.3"}, {"passed": False},
    {"ids": ["m1"]}, {"baseline_sha256": "0" * 64}, {"repo_data": False}, {"repo_data": None},
])
def test_incomplete_or_stale_proof_cannot_flip(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, delta: dict[str, object]) -> None:
    monkeypatch.setattr(repin, "source_digest", lambda repo: "source")
    directory = _proof_dir(tmp_path, {}, delta)
    assert bool(repin._proof("2.36.0", directory, directory)) == bool(delta)


@pytest.mark.parametrize("delta", [{"ids": ["m3638"]}, {"smoke": True}, {"smoke": None}])
def test_partial_or_smoke_day_evidence_cannot_flip(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, delta: dict[str, object]) -> None:
    monkeypatch.setattr(repin, "source_digest", lambda repo: "source")
    directory = _proof_dir(tmp_path, delta, {})
    assert repin._proof("2.36.0", directory, directory)


def test_source_digest_binds_checked_models_adapters_and_batches() -> None:
    files = repin.checked_model_files()
    assert any(f.endswith("/m3625.json") for f in files)
    assert "menagerie/data/adapters/f-beit3.py" in files
    assert "menagerie/classics/mogrifier_lstm.py" in files
    assert all((repin.REPO / f).is_file() for f in files)


def test_flip_requires_a_clean_candidate_at_its_branch_tip(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert not repin._committed_tip(tmp_path)  # not a checkout: fails closed
    monkeypatch.setattr(repin, "_checkouts", lambda: (tmp_path, tmp_path))
    monkeypatch.setattr(repin, "source_digest", lambda repo: "source")
    monkeypatch.setattr(repin, "priority_ids", lambda data, version: [])
    monkeypatch.setattr(repin, "_proof", lambda *args: [])
    result = repin.flip("2.36.0", data=None, dry_run=True, check_dir=tmp_path)
    assert any("branch tip" in problem for problem in result["blockers"])


def test_dry_run_requires_approval_and_never_merges(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(repin, "_checkouts", lambda: (tmp_path, tmp_path))
    monkeypatch.setattr(repin, "source_digest", lambda repo: "source")
    monkeypatch.setattr(repin, "priority_ids", lambda data, version: ["m1"])
    monkeypatch.setattr(repin, "_proof", lambda *args: [])
    result = repin.flip("2.36.0", data=None, dry_run=True, check_dir=tmp_path)
    assert not result["ready"] and result["trace_only"]
    with pytest.raises(repin.RecordError, match="held"):
        repin.flip("2.36.0", data=None, check_dir=tmp_path)


def test_c1_drain_does_not_accept_c2_pending_zero(tmp_path: Path) -> None:
    status = tmp_path / "status.md"
    status.write_text("## C1\nPending 1;\n## C2\nPending 0;\n")
    assert not repin._c1_drained(status)
    status.write_text("## C1\nPending 0;\n## C2\nPending 5;\n")
    assert repin._c1_drained(status)


@pytest.mark.parametrize("field", ["c1_drained", "approved"])
def test_numeric_flags_cannot_approve_a_flip(field: str) -> None:
    approval = {"version": "2.36.0", "source_digest": "source",
                "c1_drained": True, "approved": True}
    approval[field] = 1
    assert not repin._approved(approval, "2.36.0", "source")


def test_required_structure_evidence_cannot_be_explained_away() -> None:
    old = _baseline()
    old[0]["r1_structure_only"] = "validated"
    new = [{"stable_id": "m1", "mode": "trace", "status": "validated",
            "n_ops": 4, "graph_hash": "abc"}]
    assert repin_slice.compare(new, old, explanations={"m1": "fix"})[0]["agree"] == "unexplained"


def test_priority_excludes_stale_recipes_and_unrelated_dino_names(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from menagerie.pipeline import DataLayout, ledger
    from menagerie.pipeline.records import LedgerRow

    data = DataLayout(tmp_path)
    models = tmp_path / "models"
    models.mkdir()
    for sid in ("m1", "m2", "m3", "m4", "m5", "m6"):
        (models / f"{sid}.json").write_text("{}")
    monkeypatch.setattr(repin, "read_json", lambda kind, path: SimpleNamespace(
        id=path.stem, name="unrelated_dino", recipe=SimpleNamespace(rev="a" * 12)))
    monkeypatch.setattr(repin, "CARRIED_IDS", ("m2",))
    rows = [LedgerRow(id=sid, stage="trace", at="2026-10-08T12:00:00+00:00",
                      host="worker", env="torch-main@00000000", recipe_rev=rev,
                      outcome=outcome, cls=cls, tl=pin, mode="eval", capture="full",
                      validation_state="passed" if outcome == "pass" else "failed",
                      n_ops=4, graph_hash="0" * 64)
            for sid, rev, outcome, cls, pin in (
                ("m1", "a" * 12, "fail", "torchlens_gap:replay", "2.35.3"),
                ("m3", "a" * 12, "pass", None, "2.35.3"),
                ("m4", "b" * 12, "pass", None, "2.35.3"),
                ("m5", "a" * 12, "pass", None, "2.36.0"))]
    monkeypatch.setattr(ledger, "read_all", lambda **kw: rows)
    assert repin.priority_ids(data, "2.36.0") == ["m1", "m2", "m3"]


def test_expired_deadline_preserves_a_partial_report(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    monkeypatch.setattr(repin_slice, "_selection", lambda *args: _baseline())

    def stopped(*args, **kw):
        raise TimeoutError("deadline")

    monkeypatch.setattr(repin_slice, "_batches", stopped)
    monkeypatch.setattr(repin_slice.regression, "collect", lambda spool: [])
    result = repin_slice.run(data=None, out=tmp_path, smoke=True, deadline=time.time())
    assert result["unexplained"] == 1 and not result["passed"]
    assert (tmp_path / "slice.json").is_file()
    assert (tmp_path / "slice_a_200_r3.tsv").is_file()


@pytest.mark.parametrize("document", [[], None, "wrong type"])
def test_malformed_worker_records_cannot_qualify_a_flip(
        tmp_path: Path, document: object) -> None:
    (tmp_path / "repin-day.json").write_text(json.dumps(document))
    assert "invalid repin-day.json" in repin._proof("2.36.0", tmp_path, tmp_path)


def test_live_lease_preflight_fails_before_the_pin_is_merged(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import stage3

    changes = []
    go = tmp_path / "go.json"
    go.write_text(json.dumps({"version": "2.36.0", "source_digest": "source",
                              "c1_drained": True, "approved": True}))
    monkeypatch.setattr(repin, "_checkouts", lambda: (tmp_path, tmp_path))
    monkeypatch.setattr(repin, "source_digest", lambda repo: "source")
    monkeypatch.setattr(repin, "_proof", lambda *args: [])
    monkeypatch.setattr(repin, "_c1_drained", lambda path: True)
    monkeypatch.setattr(repin, "_committed_tip", lambda candidate: "abc")
    monkeypatch.setattr(repin, "priority_ids", lambda data, version: ["m1"])
    monkeypatch.setattr(repin, "_merge", lambda main, tip: changes.append("merge"))
    monkeypatch.setattr(stage3, "queue", lambda *args, **kw: changes.append("queue"))

    def held(*args):
        raise repin.RecordError("live lease")

    monkeypatch.setattr(stage3, "_committed", held)
    with pytest.raises(repin.RecordError, match="live lease"):
        repin.flip("2.36.0", data=None, go=go, check_dir=tmp_path)
    assert not changes


@pytest.mark.parametrize("delta", [{}, {"threads": 6}, {"status": "failed:replay"},
                                   {"tl": "2.35.3"}])
def test_conv_retyping_requires_every_three_thread_sandbox_pass(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, delta: dict[str, object]) -> None:
    from menagerie.pipeline import repin_convs

    monkeypatch.setattr(repin, "source_digest", lambda: "source")
    rows = [{"stable_id": sid, "status": "validated", "tl": "2.36.0", "threads": 3}
            for sid in repin_convs.IDS]
    rows[0].update(delta)
    data = DataLayout(repin.REPO / "menagerie" / "data")
    result = repin_convs._receipt(rows, tmp_path, data)
    notes = json.loads((tmp_path / "confirmed-conv-explanations.json").read_text())
    assert result["passed"] == (not delta)
    assert len(notes["explanations"]) == (0 if delta else 13)
    assert result["version"] == "2.36.0" and result["results"] == rows
    assert result["repo_data"] is True


def test_conv_confirmation_overrides_threads_without_changing_original_batches(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import repin_convs

    calls = []
    baseline = [{"stable_id": sid, "batch": "small" if i < 10 else "large"}
                for i, sid in enumerate(repin_convs.IDS)]
    monkeypatch.setattr(repin_slice, "_wave", lambda ids, batch, mode, **kw:
                        calls.append((ids, batch, kw["threads"])))
    repin_convs._run(baseline, None, tmp_path, None)
    assert len(calls) == 8 and all(threads == 3 for _, _, threads in calls)
    assert all(len(ids) == (2 if batch == "small" else 1) for ids, batch, _ in calls)


def test_evidence_from_another_data_root_is_not_bound(tmp_path: Path) -> None:
    assert repin.repo_data(DataLayout(repin.REPO / "menagerie" / "data"))
    assert not repin.repo_data(DataLayout(tmp_path))


def test_flip_holds_when_the_candidate_moves_after_validation(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import stage3

    changes, tips = [], iter(["abc", "def"])
    go = tmp_path / "go.json"
    go.write_text(json.dumps({"version": "2.36.0", "source_digest": "source",
                              "c1_drained": True, "approved": True}))
    monkeypatch.setattr(repin, "_checkouts", lambda: (tmp_path, tmp_path))
    monkeypatch.setattr(repin, "source_digest", lambda repo: "source")
    monkeypatch.setattr(repin, "_proof", lambda *args: [])
    monkeypatch.setattr(repin, "_c1_drained", lambda path: True)
    monkeypatch.setattr(repin, "_committed_tip", lambda candidate: next(tips))
    monkeypatch.setattr(repin, "priority_ids", lambda data, version: ["m1"])
    monkeypatch.setattr(repin, "_merge", lambda main, tip: changes.append("merge"))
    monkeypatch.setattr(stage3, "_committed", lambda *args: None)
    with pytest.raises(repin.RecordError, match="moved"):
        repin.flip("2.36.0", data=None, go=go, check_dir=tmp_path)
    assert not changes


def test_merge_takes_the_validated_commit_not_the_moving_branch(tmp_path: Path) -> None:
    import subprocess

    def git(*args: str) -> str:
        return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                              cwd=tmp_path, check=True, capture_output=True, text=True).stdout.strip()

    git("init", "-q")
    git("symbolic-ref", "HEAD", "refs/heads/main")  # older git has no init -b
    git("commit", "-q", "--allow-empty", "-m", "base")
    git("branch", repin.BRANCH)
    git("checkout", "-q", repin.BRANCH)
    git("commit", "-q", "--allow-empty", "-m", "checked")
    checked = git("rev-parse", "HEAD")
    git("commit", "-q", "--allow-empty", "-m", "later")
    git("checkout", "-q", "main")
    repin._merge(tmp_path, checked)
    assert git("rev-parse", "HEAD") == checked
