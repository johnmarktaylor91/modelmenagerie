"""The paper check: a saved paper must name the family before prose rests on it, and links paper
harvest refused are never fetched; with the store and the network stood in."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from menagerie.pipeline import (
    DataLayout, StoreLayout, describe_text, describe_write, fetch, paper_check, paper_texts,
    publish, read_json, research, write_json,
)
from menagerie.pipeline import store as store_mod
from menagerie.pipeline.records import (
    Amendment, Availability, Citation, DescribeRecord, Identity, Member, Packet, Search, Source,
)

AT = "2026-10-29T03:56:00Z"
DITN = ("Unfolding Once is Enough: A Deployment-Friendly Transformer Unit for Super-Resolution. "
        "Abstract. Recent years have witnessed a few attempts of vision transformers for single "
        "image super-resolution. We propose DITN, a deployment-friendly inception transformer.")
DRCT = ("DRCT: Saving Image Super-resolution away from Information Bottleneck. Abstract. We "
        "propose the Dense-residual-connected Transformer (DRCT) to stabilize information flow.")
TEXTS = {"a" * 64: DITN, "b" * 64: DRCT, "c" * 64: "Skip to main content. arXiv. [2308.02794] "
         "Unfolding Once is Enough. Abstract: a deployment-friendly unit."}
SHARED = Amendment(at="2026-07-01T00:00:00Z", by="paper-harvest", changed=("paper_harvest",),
                   evidence=(),
                   why="paper harvest: refused; inputs 0123456789abcdef; shared docstring: the "
                       "module names several papers")


def _materialize(sha, dest, *, store):
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(TEXTS[sha])
    return dest


@pytest.fixture()
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[DataLayout, StoreLayout]:
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setattr(store_mod, "materialize", _materialize)
    return DataLayout(tmp_path / "data"), StoreLayout(tmp_path / "store")


def _packet(family: str, *sources: Source, amendments: tuple[Amendment, ...] = (),
            canonical: str = "DRCT") -> Packet:
    return Packet(family=family, written=AT, by="harvester@menagerie", members=(Member(id="m1"),),
                  identity=Identity(canonical=canonical), sources=sources, amendments=amendments,
                  availability=Availability(paper="found", code="found", license="found",
                                            description="insufficient"),
                  search=Search(places=("arxiv",)))


def _paper(sid: str, sha: str | None, url: str = "https://arxiv.org/abs/2308.02794") -> Source:
    return Source(sid=sid, role="paper", url=url, blob=sha, text=sha)


@pytest.mark.parametrize(("names", "text", "title", "ok"), [
    (["drct"], DRCT, None, True),
    (["drct"], DITN, None, False),
    (["stgcn"], "Spatial Temporal Graph Convolutional Networks (ST-GCN) for skeletons.", None, True),
    (["wongkinyiu_yolov7_e6e"], "YOLOv7: Trainable bag-of-freebies.", None, True),
    (["dla"], "Deep Layer Aggregation. Abstract.", "Deep Layer Aggregation", True),
    (["dla"], "Deep Layer Aggregation. Abstract.", None, False),
    (["dtsh-deep-hashing"], "Feature Learning based Deep Supervised Hashing.", None, False),
])
def test_a_paper_must_name_the_family_in_its_opening(names, text, title, ok):
    assert paper_check.matches(paper_check.tokens(names), text, title) is ok


def test_a_mismatched_text_is_retired_and_its_pdf_sibling_judged_with_it(roots):
    data, store = roots
    write_json(_packet("f-drct", _paper("s2", "a" * 64), _paper("s2-pdf", "c" * 64),
                       _paper("s3", "b" * 64, "https://arxiv.org/abs/2305.11235")),
               data.packet("f-drct"))
    [row] = paper_check.sweep(None, data=data, store=store, write=True)
    assert row["retired"] == ["s2", "s2-pdf"] and row["verdicts"]["s3"] == "ok"
    packet = read_json(Packet, data.packet("f-drct"))
    assert paper_check.retired(packet) == {"s2", "s2-pdf"}
    assert set(describe_text.candidates(json.loads(data.packet("f-drct").read_text()))) == {"s3"}
    assert paper_check.sweep(None, data=data, store=store, write=True)[0]["retired"] == []


def test_refused_link_reads_the_last_paper_harvest(roots):
    later = Amendment(at="2026-08-01T00:00:00Z", by="paper-harvest", changed=("paper_harvest",),
                      evidence=(),
                      why="paper harvest: filled; inputs 0123456789abcdef")
    assert paper_check.refused_link(_packet("f-drct", amendments=(SHARED,)))
    assert not paper_check.refused_link(_packet("f-drct", amendments=(SHARED, later)))


def test_complete_never_fetches_a_link_harvest_refused(roots, monkeypatch):
    data, store = roots
    write_json(_packet("f-drct", _paper("s2", None), amendments=(SHARED,)), data.packet("f-drct"))
    monkeypatch.setattr(fetch, "fetch_paper", lambda *a, **k: pytest.fail("fetched a refused link"))
    report = research.complete("f-drct", store=store, data=data)
    assert report["paper"].startswith("refused") and report["changed"] == []


def test_paper_texts_skips_refused_links_and_retires_what_names_another_family(roots, monkeypatch):
    data, store = roots
    write_json(_packet("f-drct", _paper("s2", None), amendments=(SHARED,)), data.packet("f-drct"))
    assert paper_texts.texts(None, data=data, store=store, write=False) == []
    write_json(_packet("f-drct", _paper("s2", None)), data.packet("f-drct"))
    got = Source(sid="s2", role="paper", url="https://arxiv.org/abs/2308.02794", blob="a" * 64,
                 text="a" * 64)
    monkeypatch.setattr(paper_texts, "_paper", lambda paper, **k: ([got], "arXiv: 1 source"))
    [row] = paper_texts.texts(None, data=data, store=store)
    assert row["status"] == "none" and "never names the family" in row["why"]
    assert paper_check.retired(read_json(Packet, data.packet("f-drct"))) == {"s2"}
    assert paper_texts.texts(None, data=data, store=store, write=False) == []


def test_a_record_that_read_a_retired_text_reopens_and_stops_publishing(roots):
    data, _ = roots
    packet = _packet("f-drct", _paper("s2", "a" * 64))
    write_json(packet, data.packet("f-drct"))
    record = DescribeRecord(family="f-drct", by="describe", inputs=("s2",),
                            summary_markdown="A transformer unit.", basis="full")
    write_json(record, data.describe("f-drct"))
    assert not describe_write._undescribed("f-drct", data)
    write_json(paper_check.retire(packet, ["s2"], "test"), data.packet("f-drct"))
    assert describe_write._undescribed("f-drct", data)
    context = publish._Context.load(data, models={"m1": SimpleNamespace(family="f-drct")},
                                    roster={})
    assert context.describes == {}


def test_gather_retires_a_mismatched_paper_before_the_lane_reads_it(roots, monkeypatch):
    data, store = roots
    write_json(_packet("f-drct", _paper("s2", "a" * 64), _paper("s3", "b" * 64)),
               data.packet("f-drct"))
    monkeypatch.setattr(describe_text, "_members", lambda packet, rows, data: [])
    monkeypatch.setattr(describe_text, "institutions", lambda packet, folder, store: ([], None))
    monkeypatch.setattr(describe_text, "stage3_outputs", lambda *a, **k: {})
    monkeypatch.setattr(describe_text.ledger, "read_all", lambda data: [])
    monkeypatch.setattr(describe_text.roster, "read", lambda data: [])
    folder = data.root.parent / "work" / "describe" / "b1" / "families" / "f-drct"
    inputs = describe_text.gather("f-drct", data=data, store=store, folder=folder, tl="2.36.0")
    assert sorted(inputs.texts) == ["s3"] and inputs.facts["refused_papers"] == ["s2"]


def test_the_citation_is_kept_when_a_paper_is_retired(roots):
    data, _ = roots
    cited = replace(_packet("f-drct", _paper("s2", "a" * 64)),
                    citation=Citation(title="Unfolding Once is Enough"))
    retired = paper_check.retire(cited, ["s2"], "test")
    assert retired.citation == cited.citation and retired.amendments[-1].by == "paper-rejected"


def test_a_member_name_s_library_prefix_never_names_the_family(roots):
    data, _ = roots
    packet = _packet("f-cawn", canonical="CAWN")
    data.model("m1").parent.mkdir(parents=True, exist_ok=True)
    data.model("m1").write_text(json.dumps({"name": "DyGLib CAWN"}))
    names = paper_check.names(packet, data)
    assert "cawn" in names and "dyglib" not in names


def test_a_confirmed_paper_passes_and_clears_its_retirement(roots):
    data, store = roots
    packet = _packet("f-volsdf", _paper("s2", "a" * 64), canonical="VolSDF")
    write_json(paper_check.retire(packet, ["s2"], "test"), data.packet("f-volsdf"))
    assert paper_check.retired(read_json(Packet, data.packet("f-volsdf"))) == {"s2"}
    sure = Amendment(at=AT, by=paper_check.CONFIRMED, changed=("sources",), evidence=("s2",),
                     why="paper check: read against the family")
    packet = read_json(Packet, data.packet("f-volsdf"))
    write_json(replace(packet, amendments=(*packet.amendments, sure)), data.packet("f-volsdf"))
    assert paper_check.retired(read_json(Packet, data.packet("f-volsdf"))) == frozenset()
    [row] = paper_check.sweep(None, data=data, store=store, write=True)
    assert row["verdicts"] == {"s2": "ok"} and row["retired"] == []


def test_a_paper_find_whose_paper_names_another_family_is_undone(roots, monkeypatch):
    data, store = roots
    find = Amendment(at=AT, by="paper-find", changed=("citation", "sources"),
                     evidence=("bib-openalex-1",), why="paper find: exact title")
    bib = Source(sid="bib-openalex-1", role="bib", url="https://api.openalex.org/works/W1",
                 api="openalex")
    found = replace(_packet("f-drct", bib, _paper("p-cited", None), amendments=(find,)),
                    citation=Citation(title="Unfolding Once is Enough", arxiv="2308.02794"))
    write_json(found, data.packet("f-drct"))
    got = Source(sid="p-cited", role="paper", url="https://arxiv.org/abs/2308.02794", blob="a" * 64,
                 text="a" * 64)
    monkeypatch.setattr(paper_texts, "_paper", lambda paper, **k: ([got], "arXiv: 1 source"))
    [row] = paper_texts.texts(None, data=data, store=store)
    assert row["status"] == "none"
    packet = read_json(Packet, data.packet("f-drct"))
    assert packet.citation is None and packet.availability.paper == "not_found"
    assert paper_check.retired(packet) >= {"p-cited", "bib-openalex-1"}
    monkeypatch.setattr(paper_texts, "_textless", lambda family, data: True)
    assert paper_texts._unidentified(None, data, {}) == []
