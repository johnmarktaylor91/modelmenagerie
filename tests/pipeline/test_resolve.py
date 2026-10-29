"""Router evidence is resumable; verdicts never settle roster rows."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from menagerie.pipeline import DataLayout, resolve, resolve_confirm
from menagerie.pipeline.records import RecordError


def row(stable_id: str = "m123", name: str = "ExampleNet", **fields: object) -> dict:
    return {"stable_id": stable_id, "name": name, "zoo": "discovered-pytorch",
            "found_by": ["legacy:c2-disco"], **fields}


def data_at(tmp_path: Path, rows: list[dict] | None = None) -> DataLayout:
    data = DataLayout(tmp_path / "data")
    data.root.mkdir()
    data.roster.write_text("".join(json.dumps(r) + "\n" for r in rows or [row()]))
    return data


def cand(url: str = "https://arxiv.org/abs/2001.00001", **fields: object) -> dict:
    return {"title": "ExampleNet", "url": url, "kind": "paper", "via": "arxiv",
            "year": "2020", "matching_text": "Introducing ExampleNet", "evidence": ["s-example"],
            **fields}


def test_archive_response_parse() -> None:
    fixture = Path(__file__).parents[2] / "menagerie/data/resolve/fixtures/openalex.json"
    got = resolve.parse("openalex", fixture.read_bytes())
    assert len(got) == 3
    assert all(c["title"] and c["url"] and c["kind"] == "paper" for c in got)


def test_combined_top_three_deduplicates_versions_and_keeps_evidence() -> None:
    results = [cand(evidence=["s-one"]), cand("http://arxiv.org/abs/2001.00001v2", evidence=["s-two"]),
               *(cand(f"https://example.org/paper{i}") for i in range(5))]
    got = resolve.top_three(results)
    assert len(got) == 3
    assert got[0]["evidence"] == ["s-one", "s-two"]
    assert got[0]["url"] == "https://arxiv.org/abs/2001.00001"


def test_resume_only_skips_successful_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = data_at(tmp_path)
    calls = []
    def search(source: str, name: str, **kwargs: object) -> dict:
        calls.append((source, name))
        return {"complete": True, "candidates": [cand()], "calls": []}
    monkeypatch.setattr(resolve, "search", search)
    monkeypatch.setenv("MENAGERIE_RESOLVE_SOURCES", "arxiv,hf")
    first = resolve.run(None, data=data)
    second = resolve.run(None, data=data)
    assert first["searched"] == 2 and second["searched"] == 0
    assert len(calls) == 2
    record = json.loads(Path(second["candidates"]).read_text().splitlines()[0])
    assert len(record["candidates"]) == 1
    assert data.roster.read_text() == json.dumps(row()) + "\n"
    assert not (data.root / "ledger").exists()


def test_partial_checkpoint_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = data_at(tmp_path)
    outcomes = iter([False, True])
    monkeypatch.setattr(resolve, "search", lambda *a, **k: {
        "complete": next(outcomes), "candidates": [], "calls": []})
    monkeypatch.setenv("MENAGERIE_RESOLVE_SOURCES", "arxiv")
    assert resolve.run(None, data=data)["incomplete"] == 1
    assert resolve.run(None, data=data)["incomplete"] == 0


def test_unknown_id_is_refused_before_network(tmp_path: Path) -> None:
    with pytest.raises(RecordError, match="unknown"):
        resolve.run(["m999"], data=data_at(tmp_path))


def test_dead_census_rows_are_inputs(tmp_path: Path) -> None:
    data = data_at(tmp_path, [row(source_url="https://github.com/gone/model")])
    census = data.root / "sweeps/census-2026-10"
    census.mkdir(parents=True)
    (census / "census.jsonl").write_text(json.dumps({"id": "m123", "alive": False, "archived": False}) + "\n")
    assert [r.stable_id for r in resolve.inputs(None, data=data)] == ["m123"]


def test_classic_intake_is_selected_by_provenance(tmp_path: Path) -> None:
    data = data_at(tmp_path, [row(zoo="unregistered-classics-pytorch", found_by=["legacy:c3-classics"])])
    assert [r.stable_id for r in resolve.inputs(None, data=data)] == ["m123"]


def test_missing_identity_is_never_silently_skipped(tmp_path: Path) -> None:
    data = data_at(tmp_path, [row(stable_id=None)])
    with pytest.raises(RecordError, match="silently skip"):
        resolve.inputs(None, data=data)


def candidates_file(data: DataLayout, tmp_path: Path, count: int = 1) -> Path:
    path = tmp_path / "candidates.jsonl"
    path.write_text("".join(json.dumps({"id": f"m{i + 123}", "row": row(f"m{i + 123}"),
        "candidates": [cand()], "complete": True}) + "\n" for i in range(count)))
    return path


@pytest.mark.parametrize("verdict", ["yes", "no", "unsure"])
def test_verdict_side_effects_and_idempotency(verdict: str, tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    data = data_at(tmp_path)
    events = []
    monkeypatch.setattr("menagerie.pipeline.drive.event", lambda *a, **k: events.append((a, k)))
    records = [{"id": "m123", "row": row(), "candidates": [cand()]}]
    answers = [{"id": "m123", "candidate": cand()["url"], "verdict": verdict}]
    resolve_confirm.apply(answers, records, data=data, lane="trial", role="model-judgment")
    resolve_confirm.apply(answers, records, data=data, lane="trial", role="model-judgment")
    assert len(events) == 1
    assert events[0][0][2] == ("hint:confirmed" if verdict == "yes" else "hint:rejected")
    hints = resolve_confirm.hints(["m123"], data=data)
    assert len(hints) == (1 if verdict == "yes" else 0)
    assert data.roster.read_text() == json.dumps(row()) + "\n"
    assert not (data.root / "ledger").exists()
    assert not (data.root / "research").exists()


def test_answer_validation_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = data_at(tmp_path)
    events = []
    monkeypatch.setattr("menagerie.pipeline.drive.event", lambda *a, **k: events.append(k))
    rows = [{"id": "m123", "candidates": [cand()]}]
    answers = [{"id": "m123", "candidate": cand()["url"], "verdict": "yes"},
               {"id": "m123", "candidate": "https://unlisted.example/", "verdict": "yes"}]
    with pytest.raises(RecordError):
        resolve_confirm.apply(answers, rows, data=data, lane="trial", role="model-judgment")
    assert events == []
    assert resolve_confirm.hints(["m123"], data=data) == []


def test_fast_classify_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RecordError, match="model-judgment"):
        resolve_confirm.confirm(data=data_at(tmp_path), role="fast-classify")


def test_hint_in_ordinary_2a_brief(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import research
    data = data_at(tmp_path)
    monkeypatch.setattr("menagerie.pipeline.drive.event", lambda *a, **k: None)
    resolve_confirm.apply([{ "id": "m123", "candidate": cand()["url"], "verdict": "yes"}],
        [{"id": "m123", "candidates": [cand()]}], data=data, lane="trial", role="model-judgment")
    text = research.brief({"label": "trial", "families": [{"ids": ["m123"]}]},
                          data=data, lane_dir=tmp_path / "lane")
    assert "Hints (untrusted)" in text and cand()["url"] in text and "s-example" in text
    assert "Rows (members to confirm)" in text


@pytest.mark.parametrize("damaged", ["events.jsonl", "hints.jsonl"])
@pytest.mark.parametrize("content", ["invalid JSON\n", "null\n", "{}\n"])
def test_damaged_hint_data_cannot_block_ordinary_2a(tmp_path: Path, damaged: str,
                                                  content: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import research

    data = data_at(tmp_path)
    monkeypatch.setattr("menagerie.pipeline.drive.event", lambda *a, **k: None)
    resolve_confirm.apply([{"id": "m123", "candidate": cand()["url"], "verdict": "yes"}],
        [{"id": "m123", "candidates": [cand()]}], data=data, lane="trial", role="model-judgment")
    (data.root / "resolve" / damaged).write_text(content)
    text = research.brief({"label": "trial", "families": [{"ids": ["m123"]}]},
                          data=data, lane_dir=tmp_path / "lane")
    assert "Rows (members to confirm)" in text and "ExampleNet" in text
    assert "Continue ordinary 2a" in text and "could not be read safely" in text
    assert cand()["url"] not in text
    assert not (data.root / "ledger").exists() and not (data.root / "research").exists()


def test_confirm_batches_forty_and_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = data_at(tmp_path, [row(f"m{i + 123}") for i in range(81)])
    path = candidates_file(data, tmp_path, 81)
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setattr(resolve_confirm, "_capacity", lambda: True)
    monkeypatch.setattr("menagerie.pipeline.admission.lane_cap", lambda k: 4)
    leases, launches = [], []
    monkeypatch.setattr("menagerie.pipeline.drive.lanes", lambda **k: leases)
    def launch(kind: str, role: str, brief: Path, **kwargs: object) -> str:
        run = f"trial{len(launches)}"
        leases.append({"run": run, "kind": kind, "label": kwargs["label"]})
        launches.append(json.loads(brief.read_text().split("```json\n")[1].split("\n```")[0]))
        return run
    monkeypatch.setattr("menagerie.pipeline.drive.launch", launch)
    got = resolve_confirm.confirm(data=data, candidates=path)
    assert [r["rows"] for r in got] == [40, 40, 1]
    assert len(launches) == 3
    assert all(r["status"] == "running" for r in resolve_confirm.confirm(data=data, candidates=path))
    assert len(launches) == 3


def test_critical_capacity_launches_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = data_at(tmp_path)
    path = candidates_file(data, tmp_path)
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setattr("menagerie.pipeline.drive.lanes", lambda **k: [])
    monkeypatch.setattr(resolve_confirm, "_capacity", lambda: False)
    got = resolve_confirm.confirm(data=data, candidates=path)
    assert got[0]["status"] == "held"


def test_incomplete_api_envelopes_are_not_negative_results() -> None:
    with pytest.raises(ValueError):
        resolve.parse("github", b'{"items":[],"incomplete_results":true}')
    with pytest.raises(TypeError):
        resolve.parse("openalex", b'{}')
    with pytest.raises(ValueError):
        resolve.parse("arxiv", b'<html/>')


def test_openalex_always_uses_local_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline.fetch import Response
    calls = []
    def wrapper(url: str) -> Response:
        calls.append(url)
        return Response(200, url, "application/json", b'{"results":[]}')
    monkeypatch.setattr("menagerie.pipeline.fetch._openalex", wrapper)
    monkeypatch.setattr("requests.get", lambda *a, **k: pytest.fail("direct OpenAlex request"))
    assert resolve._request("openalex", "Example Net")[0] == 200
    assert "search=Example+Net" in calls[0]


def test_introduction_requires_all_three_checks(tmp_path: Path) -> None:
    root = tmp_path / "resolve"
    (root / "checkpoints").mkdir(parents=True)
    hypothesis = {"id": "m123", "title": "Introducing ExampleNet",
                  "wikipedia": "https://en.wikipedia.org/wiki/ExampleNet", "role": "model-judgment"}
    (root / "introductions.jsonl").write_text(json.dumps(hypothesis) + "\n")
    for source in ("openalex", "crossref", "wikipedia"):
        assert resolve._introduced("m123", root) == []
        item = cand(title=hypothesis["title"], evidence=["s-" + source],
                    references="A citation: " + hypothesis["title"] + " (2020).")
        (root / "checkpoints" / f"m123-intro-{source}.json").write_text(json.dumps({
            "complete": True, "hypothesis": hypothesis, "candidates": [item]}))
    got = resolve._introduced("m123", root)
    assert got[0]["via"] == "introduced-by" and len(got[0]["evidence"]) == 3
    assert len(resolve.top_three(got)) == 1


def test_wikipedia_body_is_not_reference_evidence() -> None:
    payload = {"parse": {"title": "ExampleNet", "text": {"*":
        '<p>Introducing ExampleNet</p><ol class="references"><li>Another paper</li></ol>'}}}
    got = resolve.parse("wikipedia", json.dumps(payload).encode())[0]
    assert "Another paper" in got["references"]
    assert "Introducing ExampleNet" not in got["references"]


def test_later_rejection_hides_an_old_hint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = data_at(tmp_path)
    monkeypatch.setattr("menagerie.pipeline.drive.event", lambda *a, **k: None)
    records = [{"id": "m123", "candidates": [cand()]}]
    for verdict in ("yes", "no", "yes"):
        resolve_confirm.apply([{ "id": "m123", "candidate": cand()["url"], "verdict": verdict}],
                              records, data=data, lane="trial-" + verdict + str(len(resolve_confirm._events(data))),
                              role="model-judgment")
        assert len(resolve_confirm.hints(["m123"], data=data)) == (verdict == "yes")


def test_stale_variant_is_refused(tmp_path: Path) -> None:
    data = data_at(tmp_path, [row(variant="new")])
    path = candidates_file(data, tmp_path)
    record = json.loads(path.read_text())
    record["row"]["variant"] = "old"
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(RecordError, match="current roster"):
        resolve_confirm._rows(path, data)


def test_unknown_launch_receipt_never_relaunches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = data_at(tmp_path)
    folder = tmp_path / "batch"
    folder.mkdir()
    (folder / "state.json").write_text('{"status":"launching"}')
    monkeypatch.setattr("menagerie.pipeline.drive.lanes", lambda **k: [])
    got = resolve_confirm._resume(folder, [], data=data, live=set(), mode="confirm")
    assert got["status"] == "held" and "unknown" in got["reason"]


def test_response_write_repairs_truncated_evidence(tmp_path: Path) -> None:
    path = tmp_path / "response"
    path.write_bytes(b"truncated")
    resolve._save_response(path, b"complete response")
    assert path.read_bytes() == b"complete response"
    assert list(tmp_path.iterdir()) == [path]


def test_invalid_result_does_not_wedge_following_batches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = data_at(tmp_path)
    folder = tmp_path / "batch"
    folder.mkdir()
    (folder / "BRIEF.result.md").write_text('{"answers": [{"id": []}]}')
    got = resolve_confirm._collect(folder, {"run": "trial", "status": "started"},
        [{"id": "m123", "candidates": [cand()]}], data=data)
    assert got["status"] == "invalid"
    assert resolve_confirm._resume(folder, [], data=data, live=set(), mode="confirm")["status"] == "invalid"
    monkeypatch.setattr("menagerie.pipeline.drive.event", lambda *a, **k: None)
    (folder / "BRIEF.result.md").write_text(json.dumps({"answers": [{"id": "m123", "candidate": cand()["url"], "verdict": "no"}]}))
    got = resolve_confirm._resume(folder, [{"id": "m123", "candidates": [cand()]}],
        data=data, live=set(), mode="confirm")
    assert got["status"] == "collected"
