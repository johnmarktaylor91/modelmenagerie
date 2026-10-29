"""Rung 2 metadata: scraped-data checks, the harvester citation fill, June notes, era and the
trace statistics export.

Everything runs on temporary data and store roots with faked bibliographic records; the fill over
the real packets and the audit of the real store are the lane's acceptance runs, not unit tests.
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from menagerie.identity import ensure_stable_ids
from menagerie.pipeline import (
    DataLayout, StoreLayout, bib_harvest, citecheck, drive, fetch, harvest, publish, read_json,
    trace_metrics, write_json,
)
from menagerie.pipeline import store as store_mod
from menagerie.pipeline.records import (
    Availability, Citation, Identity, LedgerRow, Member, ModelRecord, Packet, RosterRow, Source,
)


# --- Format, SPDX, title and duplicate checks -----------------------------------------------


@pytest.mark.parametrize(("value", "kind"), [
    ("2021", None), ("2021-03", None), ("2021-03-31", None), (None, None),
    ("c. 1998", "date-format"), ("2021-13", "date-format"), ("2021-02-30", "date-format"),
    ("1066", "date-range"), ("2999-01-01", "date-range"),
])
def test_date_notes_type_format_and_range(value: str | None, kind: str | None) -> None:
    note = citecheck.date_note("first_public", value)
    assert (note.split(":")[0] if note else None) == kind


def test_year_of_never_raises_on_a_non_numeric_prefix() -> None:
    assert citecheck.year_of("c. 1998") is None
    assert citecheck.year_of("1998-05") == 1998


def test_citation_notes_check_ids_and_basis_years() -> None:
    good = Citation(title="T", first_public="2020-01-02", arxiv="2001.00001", doi="10.1000/x",
                    openreview="nWlk9jwupZ", basis="arXiv v1")
    assert citecheck.citation_notes(good) == []
    bad = Citation(title="T", arxiv="arXiv:2001.1", doi="doi.org/10.1/x", openreview="a b",
                   basis="proceedings of 1066")
    kinds = [n.split(":")[0] for n in citecheck.citation_notes(bad)]
    assert kinds == ["date-range", "id-format", "id-format", "id-format"]


@pytest.mark.parametrize(("value", "kind"), [
    ("MIT", None), ("Apache-2.0 OR MIT", None), ("unknown", None), (None, None),
    ("GPL-2.0-or-later WITH Classpath-exception-2.0", None), ("LicenseRef-Own", None),
    ("GPL-3.0", "spdx-deprecated"), ("Apache 2", "spdx-invalid"), ("BSD", "spdx-invalid"),
])
def test_spdx_ids_are_checked_against_the_pinned_list(value: str | None, kind: str | None) -> None:
    note = citecheck.spdx_note(value)
    assert (note.split(":")[0] if note else None) == kind


def test_the_spdx_list_is_pinned_and_vendored() -> None:
    listed = json.loads(citecheck.SPDX_LIST.read_text())
    assert listed["licenseListVersion"] == "3.27.0" and "v3.27.0" in listed["source"]
    assert "MIT" in listed["licenses"] and "GPL-3.0" in listed["deprecated"]


def test_title_note_is_typed_and_only_on_a_mismatch() -> None:
    records = {"s2": {"title": "Deep Residual Learning for Image Recognition"},
               "crossref": {"message": {"title": ["Something else"]}}}
    assert citecheck.title_note("Deep residual learning for image recognition.", records) is None
    note = citecheck.title_note("Attention Is All You Need", records)
    assert note and note.startswith("title-mismatch:") and "s2" in note
    assert citecheck.title_note("Anything", {"s2": {"paperId": "p"}}) is None


def test_bibtex_title_reads_braced_titles() -> None:
    text = "@misc{x,\n  title={{CrossViT}: Cross-Attention\n   Multi-Scale},\n  year={2021}\n}\n"
    assert citecheck.bibtex_title(text) == "CrossViT: Cross-Attention Multi-Scale"


def test_duplicates_across_model_records_and_active_roster_rows() -> None:
    def model(i: str, name: str) -> ModelRecord:
        return ModelRecord(id=i, name=name, zoo="timm", variant="", family="f-x",
                           source_case="library")

    rows = [RosterRow(name="a", zoo="timm", stable_id="m1"),
            RosterRow(name="a", zoo="timm", stable_id="m2"),
            RosterRow(name="b", zoo="timm", stable_id="m2", state="duplicate_of:m1")]
    notes = citecheck.duplicate_notes([model("m1", "a"), model("m2", "a")], rows)
    assert notes == ["duplicate-key: ('a', 'timm', '') in 2 model records (m1, m2)",
                     "duplicate-key: ('a', 'timm', '') in 2 roster rows (m1, m2)"]
    assert citecheck.duplicate_notes([model("m1", "a")], rows[:1]) == []


# --- Paper downloads: a challenge page is never saved as the paper ---------------------------

PDF = b"%PDF-1.5\n1 0 obj<<>>endobj\nxref\n0 1\ntrailer<<>>\nstartxref\n9\n%%EOF\n"
WALL = (b"<html><head><title>Just a moment...</title></head><body>"
        b"<div class='cf-chl'>Checking your browser</div></body></html>")


@pytest.mark.parametrize(("body", "pdf", "problem"), [
    (PDF, True, None), (b"%PDF-1.5\ntruncated\n%%EOF\n", True, "a PDF that does not parse"),
    (WALL, False, "a bot wall"), (b"<html><title>Paper</title>" + b"x" * 3000, True, "not a PDF"),
    (b"<html><title>Paper</title>" + b"x" * 3000, False, None),
])
def test_paper_problem(body: bytes, pdf: bool, problem: str | None) -> None:
    got = fetch.paper_problem(body, url="https://arxiv.org/pdf/1", mime="text/html", pdf=pdf)
    assert (got or "").startswith(problem or "") and bool(got) == bool(problem)


@pytest.fixture()
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[StoreLayout, DataLayout]:
    monkeypatch.setattr(drive, "event", lambda *a, **f: None)
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    monkeypatch.delenv("MENAGERIE_STORE_REMOTE", raising=False)
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    data = DataLayout(tmp_path / "data")
    data.root.mkdir()
    return StoreLayout(tmp_path / "store"), data


def test_fetch_paper_refuses_a_200_challenge_page_and_saves_nothing(
        roots: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    st, data = roots

    def download(url: str, dest: Path, **kw: Any) -> tuple[str, str]:
        dest.write_bytes(WALL)
        return url, "text/html"

    monkeypatch.setattr(fetch, "_download", download)
    with pytest.raises(fetch.Inaccessible, match="page-inaccessible"):
        fetch.fetch_paper("2103.14899", store=st, data=data)
    assert not (data.root / "store-index").exists()


def test_the_store_audit_types_a_saved_wall_with_its_packet(roots: tuple) -> None:
    st, data = roots
    lines = [_put(st, data, body, url, mime) for body, url, mime in (
        (WALL, "https://doi.org/10.1/x", "text/html"),
        (PDF, "https://arxiv.org/pdf/2001.00001", "application/pdf"))]
    write_json(_packet("f-x", sources=(Source(sid="s1", role="paper", url="https://doi.org/10.1/x",
                                              blob=lines[0]),)), data.packet("f-x"))
    report = citecheck.audit_documents(store=st, data=data)
    assert (report["documents"], report["ok"], report["bad"]) == (2, 1, 1)
    assert report["found"][0]["families"] == ["f-x"]
    assert report["found"][0]["problem"].startswith("page-inaccessible: a bot wall")


def _put(st: StoreLayout, data: DataLayout, body: bytes, url: str, mime: str) -> str:
    path = st.root.parent / "blob.tmp"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return store_mod.put(path, store=st, data=data, mime=mime, license_class="third-party",
                         url=url).sha256


# --- The harvester citation fill -------------------------------------------------------------

S2 = {"paperId": "p1", "title": "CrossViT: Cross-Attention Multi-Scale Vision Transformer",
      "publicationDate": "2021-03-27", "venue": "IEEE International Conference on Computer Vision",
      "authors": [{"name": "Chun-Fu Chen"}, {"name": "Quanfu Fan"}], "citationCount": 900,
      "abstract": "We propose CrossViT."}
OPENALEX = {"id": "https://openalex.org/W1", "title": S2["title"], "publication_date": "2021-03-27",
            "authorships": [{"author": {"display_name": "C. Chen"}}],
            "primary_location": {"source": {"display_name": "arXiv (Cornell University)"}}}
BIBTEX = b"@misc{chen2021,\n  title={CrossViT: Cross-Attention Multi-Scale Vision Transformer},\n}\n"


def _packet(family: str, *, citation: Citation | None = None,
            sources: tuple[Source, ...] = ()) -> Packet:
    paper = (Source(sid="s9", role="paper", url="https://arxiv.org/abs/2103.14899"),)
    return Packet(family=family, written="2026-10-04T00:00:00Z", by="harvester@timm-1.0.30",
                  members=(Member(id="m1"),), identity=Identity(canonical=family[2:]),
                  sources=sources or paper, citation=citation,
                  availability=Availability("found", "found", "found", "sufficient"))


def _records(roots: tuple, monkeypatch: pytest.MonkeyPatch, s2: dict[str, Any]) -> None:
    st, data = roots
    shas = {api: _put(st, data, body, f"https://{api}/x", mime) for api, body, mime in (
        ("s2", json.dumps(s2).encode(), "application/json"),
        ("openalex", json.dumps(OPENALEX).encode(), "application/json"),
        ("bibtex", BIBTEX, "application/x-bibtex"))}
    sources = tuple(Source(sid=f"bib-{api}-{sha[:8]}", role="bib", api=api, blob=sha)
                    for api, sha in shas.items())
    monkeypatch.setattr(fetch, "bib_records", lambda ref, **kw: fetch.BibResult(
        sources, (), {"s2": s2, "openalex": OPENALEX}))
    monkeypatch.setattr(bib_harvest, "_title_search", lambda *a, **kw: ((), None))


def test_a_titleless_packet_takes_the_title_its_records_agree_on(
        roots: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    st, data = roots
    write_json(_packet("f-crossvit"), data.packet("f-crossvit"))
    _records(roots, monkeypatch, S2)
    got = bib_harvest.fill("f-crossvit", store=st, data=data)
    assert (got.status, got.matched, got.name_match, got.spot_check) == (
        "filled", ("s2", "openalex"), True, False)
    packet = read_json(Packet, data.packet("f-crossvit"))
    cite = packet.citation
    assert (cite.title, cite.arxiv, cite.first_public, cite.basis) == (
        S2["title"], "2103.14899", "2021-03-27", bib_harvest.TITLELESS_BASIS)
    assert cite.authors == ("Chun-Fu Chen", "Quanfu Fan") and cite.venue == S2["venue"]
    assert (cite.from_, cite.s2, cite.openalex) == ("s9", "p1", "W1")
    assert packet.amendments[-1].by == "bib-harvest" and len(packet.amendments[-1].evidence) == 3
    assert len(list((data.root / "cite").glob("*.jsonl"))) == 1


def test_a_titleless_fill_without_the_family_name_is_marked_for_a_spot_check(
        roots: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    st, data = roots
    write_json(_packet("f-deit"), data.packet("f-deit"))
    _records(roots, monkeypatch, S2)
    got = bib_harvest.fill("f-deit", store=st, data=data)
    assert (got.status, got.name_match, got.spot_check) == ("filled", False, True)
    assert "spot check" in read_json(Packet, data.packet("f-deit")).amendments[-1].why


def test_a_titled_packet_fills_only_from_matching_records_and_keeps_its_fields(
        roots: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    st, data = roots
    titled = Citation(title="CrossViT: cross-attention multi-scale vision transformer",
                      arxiv="2103.14899", venue="ICCV", from_="s9")
    write_json(_packet("f-crossvit", citation=titled), data.packet("f-crossvit"))
    _records(roots, monkeypatch, S2)
    got = bib_harvest.fill("f-crossvit", store=st, data=data)
    assert got.status == "filled" and got.fields == ("authors", "first_public", "published")
    cite = read_json(Packet, data.packet("f-crossvit")).citation
    assert (cite.title, cite.venue, cite.first_public) == (titled.title, "ICCV", "2021-03-27")
    assert cite.basis == "s2 and openalex matched on title"


def test_records_that_disagree_with_the_title_are_reported_never_filled(
        roots: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    st, data = roots
    titled = Citation(title="Something Else Entirely", arxiv="2103.14899", from_="s9")
    write_json(_packet("f-x", citation=titled), data.packet("f-x"))
    before = data.packet("f-x").read_text()
    _records(roots, monkeypatch, S2)
    got = bib_harvest.fill("f-x", store=st, data=data)
    assert got.status == "unmatched" and "titles:" in got.note
    assert data.packet("f-x").read_text() == before


@pytest.mark.parametrize(("location", "status"), [
    ("https://arxiv.org/abs/2103.14899v2", "filled"), ("https://example.org/other", "unmatched")])
def test_a_title_search_counts_for_a_titleless_packet_only_when_it_names_the_id(
        roots: tuple, monkeypatch: pytest.MonkeyPatch, location: str, status: str) -> None:
    st, data = roots
    write_json(_packet("f-crossvit"), data.packet("f-crossvit"))
    sha = _put(st, data, BIBTEX, "https://arxiv.org/bibtex/2103.14899", "application/x-bibtex")
    monkeypatch.setattr(fetch, "bib_records", lambda ref, **kw: fetch.BibResult(
        (Source(sid=f"bib-bibtex-{sha[:8]}", role="bib", api="bibtex", blob=sha),),
        ("bib 2103.14899: s2 http_429 (cooling down)",), {}))
    work = {**OPENALEX, "primary_location": {"source": {"display_name": "ICCV"}},
            "locations": [{"landing_page_url": location}]}
    monkeypatch.setattr(bib_harvest, "_title_search", lambda *a, **kw: ((), work))
    got = bib_harvest.fill("f-crossvit", store=st, data=data)
    assert got.status == status
    if status == "filled":
        assert got.matched == ("openalex_search",)
        assert read_json(Packet, data.packet("f-crossvit")).citation.venue == "ICCV"


def test_a_packet_with_no_identifier_is_refused(roots: tuple) -> None:
    st, data = roots
    write_json(_packet("f-x", sources=(Source(sid="s1", role="library", dist="timm",
                                              version="1", symbol="x"),)), data.packet("f-x"))
    assert bib_harvest.fill("f-x", store=st, data=data).status == "refused"


def test_the_s2_key_header_is_sent_only_when_a_key_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("S2_API_KEY", raising=False)
    assert fetch.s2_headers() == {}
    monkeypatch.setenv("S2_API_KEY", "k-test")
    assert fetch.s2_headers() == {"x-api-key": "k-test"}


# --- June notes and era ----------------------------------------------------------------------


def test_june_notes_group_members_by_note(tmp_path: Path) -> None:
    data = DataLayout(tmp_path)
    catalog = [{"name": n, "zoo": "timm", "variant": "", "notes": note}
               for n, note in (("a", "random init"), ("b", "random init"), ("c", "needs 224 px"))]
    (tmp_path / "master_catalog.jsonl").write_text("".join(json.dumps(r) + "\n" for r in catalog))
    ids = ensure_stable_ids({(r["name"], "timm", "") for r in catalog},
                            tmp_path / "stable_ids.jsonl")
    m1, m2, m3 = (ids[(n, "timm", "")] for n in "abc")
    assert harvest.june_notes([m1, m2], data) == "June catalog notes: random init"
    assert harvest.june_notes([m1, m3], data) == (
        f"June catalog notes: {m1}: random init; {m3}: needs 224 px")
    assert harvest.june_notes(["m9"], data) == ""


@pytest.mark.parametrize(("first_public", "era", "year", "source"), [
    ("2021-03-27", "1998", 2021, "citation"), (None, "1998", 1998, "june_era"),
    (None, "1990s", None, None), ("c. 1998", None, None, None),
])
def test_era_is_published_and_backs_year_only_as_a_four_digit_year(
        first_public: str | None, era: str | None, year: int | None, source: str | None) -> None:
    packet = _packet("f-x", citation=Citation(title="T", first_public=first_public))
    row = RosterRow(name="a", zoo="timm", era=era)
    assert publish._era_fields(packet, row) == {"era": era, "year": year, "year_source": source}


# --- Trace statistics ------------------------------------------------------------------------


def _june_flagship() -> dict[str, str | None]:
    """``csv_export._FLAGSHIP_TRACE_COLUMNS``, read from the source (the module needs pyarrow,
    which the test environment does not carry)."""

    import ast

    from menagerie import __file__ as package

    tree = ast.parse((Path(package).parent / "csv_export.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", "") == "_FLAGSHIP_TRACE_COLUMNS")
    return ast.literal_eval(node.value)


def test_the_column_lists_match_the_june_exporter_and_dictionary() -> None:
    from menagerie.csv_dictionary import TRACE_METRICS_COLUMNS
    from menagerie.trace_summary import TRACE_SUMMARY_COLUMNS

    assert trace_metrics.FLAGSHIP == tuple(k for k in _june_flagship().values() if k)
    assert trace_metrics.METRICS == TRACE_METRICS_COLUMNS[1:]
    assert set(trace_metrics.METRICS) <= set(TRACE_SUMMARY_COLUMNS)


def _trace(summary: str | None) -> LedgerRow:
    return LedgerRow(id="m1", stage="trace", outcome="pass", at="2026-10-05T00:00:00Z",
                     host="worker", env="torch-main", recipe_rev="0" * 12, capture="full",
                     tl="2.35.3", summary=summary)


def test_summaries_become_flagship_columns_and_a_metrics_table(roots: tuple,
                                                               tmp_path: Path) -> None:
    st, data = roots
    found = {"trace_summary_version": "1.1.0", "n_params": 11, "graph_depth": 4,
             "total_flops_forward": 900, "op_type_histogram": {"conv2d": 3}, "pct_conv": 50.0}
    sha = _put(st, data, json.dumps(found).encode(), "summary", "application/json")
    trace = _trace(sha)
    got = trace_metrics.summary(trace, st)
    flagship = trace_metrics.flagship(got)
    assert flagship["n_params"] == 11 and flagship["graph_depth"] == 4
    assert set(flagship) == set(trace_metrics.FLAGSHIP)
    assert trace_metrics.flagship(trace_metrics.summary(_trace(None), st)) is None
    assert trace_metrics.summary(trace, None) is None
    jsonl, table = trace_metrics.write_table(tmp_path / "out",
                                             [trace_metrics.metrics_row("m1", trace, got)])
    row = next(csv.DictReader(table.open()))
    assert (row["stable_id"], row["summary_sha256"], row["total_flops_forward"]) == ("m1", sha, "900")
    assert json.loads(jsonl.read_text())["capture"] == "full"


def test_csv_export_trusts_store_summaries_on_the_torchlens_major() -> None:
    pytest.importorskip("pyarrow")
    from menagerie import csv_export
    from menagerie.trace_summary import TRACE_SUMMARY_VERSION

    row = {"summary_source": "store", "trace_summary_version": TRACE_SUMMARY_VERSION,
           "torchlens_version": "2.35.3", "recipe_revision_sha256": "abcdef012345"}
    catalog = type("Row", (), {"recipe_revision_sha256": "0" * 64})()
    assert csv_export._trace_summary_is_current(catalog, row, "2.36.0")
    assert not csv_export._trace_summary_is_current(catalog, {**row, "summary_source": None},
                                                    "2.36.0")


def test_a_classic_takes_a_paper_link_only_from_docs_that_name_one_paper() -> None:
    # The hashing module's docstring names DPSH and DSDH for four builders: none gets a link.
    assert harvest.classic_paper("menagerie.classics.ri_ts_graph_hash", "build_dsh") == (
        None, None)
    link, _ = harvest.classic_paper("menagerie.classics.artrack", "build_artrack")
    assert link == "arxiv.org/abs/2303.10826"
    assert bib_harvest._ids(link) == bib_harvest._ids("https://arxiv.org/abs/2303.10826") == (
        "arXiv:2303.10826")


def test_publish_holds_a_spot_check_fill_until_it_is_confirmed() -> None:
    from menagerie.pipeline.records import Amendment

    held = _packet("f-x", citation=Citation(title="Wrong Paper", first_public="2020"))
    mark = Amendment(at="t", why="bib harvest: title; spot check", evidence=(),
                     changed=("citation",), by=bib_harvest.BY)
    held = replace(held, amendments=(mark,))
    assert bib_harvest.citation_held(held) and publish._citation(held) is None
    assert bib_harvest.held_sources(held) == {"s9"} and publish._paper_url(held) is None
    ok = Amendment(at="t2", why="read the paper", evidence=(), changed=("citation",),
                   by=bib_harvest.SPOT_CHECK)
    confirmed = replace(held, amendments=(mark, ok))
    assert publish._citation(confirmed).title == "Wrong Paper"


def test_a_right_verdict_releases_a_held_fill_and_others_stay_held(
        roots: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    st, data = roots
    for family in ("f-deit", "f-c3d"):
        write_json(_packet(family), data.packet(family))
        _records(roots, monkeypatch, S2)
        assert bib_harvest.fill(family, store=st, data=data).spot_check
    released = bib_harvest.confirm([
        {"family": "f-deit", "verdict": "right", "reason": "DeiT's paper"},
        {"family": "f-c3d", "verdict": "wrong", "reason": "PSMNet's paper"}], data=data)
    assert released == ["f-deit"]
    assert not bib_harvest.citation_held(read_json(Packet, data.packet("f-deit")))
    assert bib_harvest.citation_held(read_json(Packet, data.packet("f-c3d")))
