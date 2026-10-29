"""Ledger currency on real rows: the probe's resnet18 trace and render, Slice A's beit failure
and the June routing manifest."""

from __future__ import annotations

import csv
import dataclasses
from collections import Counter
from pathlib import Path

import pytest

from menagerie.pipeline import DataLayout, ledger, read_jsonl
from menagerie.pipeline.records import (
    LedgerRow,
    LegacyRow,
    RenderArtifact,
    RenderRow,
    RenderSkip,
    parse_ledger_line,
)

FIXTURES = Path(__file__).parent / "fixtures"
MANIFEST = Path(__file__).parents[2] / "menagerie" / "data" / "routing_manifest.tsv"
TL, RENDERER = "2.35.0", "graphviz 2.43.0+probe-defaults"


def _fixture_rows() -> list:
    return read_jsonl(parse_ledger_line, FIXTURES / "ledger.jsonl")


@pytest.fixture()
def rows() -> dict:
    """The probe's real resnet18 trace made a validated pass, its real render row with a
    thumbnail added, and Slice A's real beit failure."""

    beit, trace, render = _fixture_rows()
    with (FIXTURES / "routing_manifest_row.tsv").open(encoding="utf-8", newline="") as handle:
        june = LegacyRow.from_manifest(next(csv.DictReader(handle, delimiter="\t")))
    passed = dataclasses.replace(trace, validation_state="passed", graph_hash=june.graph_hash,
                                 msg=None)
    svg = render.renders[0]
    thumb = RenderArtifact(kind="thumb", sha256=svg.sha256[::-1], bytes=4_000,
                           variant="unrolled-none")
    rendered = dataclasses.replace(render, at="2026-10-04T06:00:00Z", outcome="pass",
                                   renders=(*render.renders, thumb), skipped=render.skipped[1:])
    return {"beit": beit, "trace": trace, "pass": passed, "probe_render": render,
            "render": rendered, "rev": trace.recipe_rev}


def _state(lines: list, rev: str, *, tl: str = TL, renderer: str = RENDERER) -> str:
    return ledger.state("m8839", lines, recipe_rev=rev, tl=tl, renderer_rev=renderer)


def test_a_validated_rendered_trace_is_publishable(rows: dict) -> None:
    lines = [rows["beit"], rows["pass"], rows["render"]]
    assert _state(lines, rows["rev"]) == "publishable"
    assert ledger.publishable("m8839", lines, recipe_rev=rows["rev"], tl=TL,
                              renderer_rev=RENDERER)


def test_the_probe_trace_is_not_validated(rows: dict) -> None:
    """At 2.35.0 the one-capture validation said `unverified`; that never publishes."""

    assert _state([rows["trace"], rows["render"]], rows["rev"]) == "not_publishable"


def test_a_recipe_or_torchlens_change_retires_a_pass(rows: dict) -> None:
    lines = [rows["pass"], rows["render"]]
    assert _state(lines, "0123456789ab") == "not_publishable"
    assert _state(lines, rows["rev"], tl="2.35.1") == "not_publishable"


def test_an_infra_row_never_overrides_a_pass(rows: dict) -> None:
    infra = dataclasses.replace(rows["pass"], at="2026-10-05T00:00:00Z", outcome="infra",
                                cls="infra:worker_died", validation_state=None, n_ops=None)
    assert _state([rows["pass"], infra, rows["render"]], rows["rev"]) == "publishable"


def test_a_newer_genuine_failure_blocks_until_a_newer_pass(rows: dict) -> None:
    fail = dataclasses.replace(rows["pass"], at="2026-10-05T00:00:00Z", outcome="fail",
                               cls="torchlens_gap:batchnorm_parent", validation_state="failed")
    again = dataclasses.replace(rows["pass"], at="2026-10-06T00:00:00Z")
    assert _state([rows["pass"], fail, rows["render"]], rows["rev"]) == "not_publishable"
    assert _state([rows["pass"], fail, again, rows["render"]], rows["rev"]) == "publishable"


def test_a_render_timeout_leaves_the_model_validated_pending_render(rows: dict) -> None:
    timeout = RenderRow(id="m8839", at="2026-10-04T05:58:00Z", trace=rows["pass"].trace,
                        renderer_rev=RENDERER, tl=TL, outcome="fail",
                        skipped=(RenderSkip(kind="svg", why="render:timeout",
                                            variant="unrolled-none"),))
    lines = [rows["pass"], timeout]
    assert _state(lines, rows["rev"]) == "publishable_pending_render"
    assert ledger.validated(ledger.deciding_trace("m8839", lines, recipe_rev=rows["rev"], tl=TL))
    # The probe's own render had an SVG but no thumbnail: still pending.
    assert _state([*lines, rows["probe_render"]], rows["rev"]) == "publishable_pending_render"
    # An artifact-only retry publishes it, with no new trace row.
    assert _state([*lines, rows["render"]], rows["rev"]) == "publishable"
    # A renderer upgrade re-arms rendering, never the verdict.
    assert _state([*lines, rows["render"]], rows["rev"], renderer="graphviz 12.2.1+x") == (
        "publishable_pending_render")


def test_a_fabrication_review_blocks(rows: dict) -> None:
    review = LedgerRow(id="m8839", stage="review", at="2026-10-05T00:00:00Z", host="agent",
                       env="review", recipe_rev=rows["rev"], outcome="pass",
                       verdict="fabrication")
    minor = dataclasses.replace(review, at="2026-10-06T00:00:00Z", verdict="minor")
    assert _state([rows["pass"], rows["render"], review], rows["rev"]) == "not_publishable"
    assert _state([rows["pass"], rows["render"], review, minor], rows["rev"]) == "publishable"


def test_structure_only_and_train_rows_never_decide_the_verdict(rows: dict) -> None:
    meta = dataclasses.replace(rows["pass"], at="2026-10-05T00:00:00Z", capture="structure_only",
                               validation_state="structure_only", params_source="geometry")
    train = dataclasses.replace(rows["pass"], at="2026-10-05T00:00:00Z", mode="train",
                                outcome="fail", cls="capture_error", validation_state="failed")
    assert _state([meta, rows["render"]], rows["rev"]) == "not_publishable"
    assert _state([rows["pass"], meta, train, rows["render"]], rows["rev"]) == "publishable"


def test_append_routes_rows_to_month_shards(tmp_path: Path, rows: dict) -> None:
    data = DataLayout(tmp_path)
    paths = ledger.append([rows["beit"], rows["pass"], rows["render"]], data=data)
    assert paths == [data.ledger("2026-10")]
    assert ledger.read_all(data=data) == [rows["beit"], rows["pass"], rows["render"]]


def test_the_routing_manifest_loads_as_the_legacy_shard(tmp_path: Path, rows: dict) -> None:
    data = DataLayout(tmp_path)
    assert ledger.import_legacy(MANIFEST, data=data) == 11_641
    assert ledger.import_legacy(MANIFEST, data=data) == 11_641  # idempotent rewrite
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        june = Counter(row["status"] for row in csv.DictReader(handle, delimiter="\t"))
    ledger.append([rows["beit"], rows["pass"], rows["render"]], data=data)
    counts = ledger.status_counts(data=data)
    assert {k[len("legacy:"):]: v for k, v in counts.items() if k.startswith("legacy:")} == june
    assert counts["legacy:validated"] == 8_359
    assert counts["trace:pass"] == 1 and counts["trace:fail:torchlens_gap:sdpa_depth"] == 1
    assert counts["render:pass"] == 1
    # The legacy rows sort first and never validate a model: June's revision is 64 hex.
    first = ledger.read_all(data=data)[0]
    assert not ledger.HOME_PREFIX.search(data.ledger("2026-06").read_text(encoding="utf-8"))
    assert isinstance(first, LegacyRow) and len(first.recipe_rev) == 64


def test_import_legacy_refuses_to_overwrite_new_rows(tmp_path: Path, rows: dict) -> None:
    data = DataLayout(tmp_path)
    june_row = dataclasses.replace(rows["beit"], at="2026-06-30T00:00:00Z")
    ledger.append([june_row], data=data)
    with pytest.raises(ValueError, match="not legacy"):
        ledger.import_legacy(MANIFEST, data=data)


def test_the_committed_legacy_shard_is_the_manifest_import(tmp_path: Path) -> None:
    data = DataLayout(tmp_path)
    ledger.import_legacy(MANIFEST, data=data)
    committed = MANIFEST.parent / "ledger" / "2026-06.jsonl"
    assert data.ledger("2026-06").read_bytes() == committed.read_bytes()


def test_read_all_parses_only_the_lines_appended_since_the_last_read(
        tmp_path: Path, rows: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import drive

    data = DataLayout(tmp_path)
    ledger.append([rows["beit"]], data=data)
    parsed: list[str] = []
    real = drive._ledger_line
    monkeypatch.setattr(drive, "_ledger_line", lambda raw: parsed.append(raw["id"]) or real(raw))
    first = ledger.read_all(data=data)
    first.append("caller's own list")
    assert ledger.read_all(data=data) == [rows["beit"]] and parsed == [rows["beit"].id]
    ledger.append([rows["pass"]], data=data)
    assert ledger.read_all(data=data) == [rows["beit"], rows["pass"]]
    assert parsed == [rows["beit"].id, rows["pass"].id]  # the grown shard: its new line alone


def test_read_containing_parses_only_matching_whole_lines(tmp_path: Path, rows: dict) -> None:
    data = DataLayout(tmp_path)
    typed = dataclasses.replace(rows["beit"], outcome="giveup", cls="standin:kwargs")
    (shard,) = ledger.append([rows["trace"], typed], data=data)
    with shard.open("a", encoding="utf-8") as handle:
        handle.write('{"unfinished": "standin')
    assert ledger.read_containing('"standin', data=data) == [typed]
    assert ledger.read_containing('"no such text', data=data) == []
