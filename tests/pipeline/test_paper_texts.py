"""Open-access paper texts for paper sources never fetched, with the network and pdftotext stood in."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from menagerie.pipeline import DataLayout, StoreLayout, describe_text, fetch, paper_check, paper_texts
from menagerie.pipeline import read_json, store as store_mod, tick_full, write_json
from menagerie.pipeline.records import (
    Availability, Citation, Identity, Member, Packet, Search, Source,
)

AT = "2026-10-08T20:00:00Z"
ABS = ('<html><head><title>[1512.03385] Deep Residual Learning</title></head><body>'
       '<blockquote>Deeper neural networks are more difficult to train. We present a residual '
       'learning framework.</blockquote><a title="Rights to this article" '
       'href="http://creativecommons.org/licenses/by-nc-sa/4.0/">license</a></body></html>')
PDF = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\nstartxref\n0\n%%EOF\n"
FULL = "Deep Residual Learning. 1. Introduction. Deep convolutional neural networks have led..."


class Web:
    """Stands in for ``fetch._download``: URL -> (body, mime); any other URL is a 404."""

    def __init__(self, pages: dict[str, tuple[bytes, str]]) -> None:
        self.pages, self.asked = pages, []

    def __call__(self, url: str, dest: Path, *, tries: int = 3) -> tuple[str, str]:
        self.asked.append(url)
        if url not in self.pages:
            raise fetch.FetchError(f"{url}: HTTP 404")
        body, mime = self.pages[url]
        dest.write_bytes(body)
        return url, mime


@pytest.fixture()
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[DataLayout, StoreLayout]:
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setattr(store_mod, "_pdf_text", lambda blob, out: out.write_text(FULL))
    monkeypatch.setattr(paper_check, "matches", lambda *a, **k: True)  # test_paper_check has it
    return DataLayout(tmp_path / "data"), StoreLayout(tmp_path / "store")


def _web(monkeypatch, pages) -> Web:
    web = Web(pages)
    monkeypatch.setattr(fetch, "_download", web)
    return web


def _packet(family: str, *sources: Source) -> Packet:
    return Packet(family=family, written=AT, by="research", members=(Member(id="m1"),),
                  identity=Identity(canonical="Example"), sources=sources,
                  availability=Availability(paper="found", code="found", license="found",
                                            description="insufficient"),
                  search=Search(places=("arxiv",)))


ARXIV = Source(sid="s1", role="paper", url="https://arxiv.org/abs/1512.03385", official=True)


def test_an_arxiv_paper_gains_its_abstract_page_pdf_text_and_license(roots, monkeypatch):
    data, store = roots
    write_json(_packet("f-resnet", ARXIV), data.packet("f-resnet"))
    web = _web(monkeypatch, {"https://arxiv.org/abs/1512.03385": (ABS.encode(), "text/html"),
                             "https://arxiv.org/pdf/1512.03385": (PDF, "application/pdf")})
    [row] = paper_texts.texts(None, data=data, store=store)
    assert row == {"family": "f-resnet", "status": "text", "sids": ["s1", "s1-pdf"]}
    packet = read_json(Packet, data.packet("f-resnet"))
    by = {s.sid: s for s in packet.sources}
    assert by["s1"].blob and by["s1"].text and by["s1-pdf"].url == "https://arxiv.org/pdf/1512.03385"
    assert store_mod.get(by["s1-pdf"].text, store=store).read_text() == FULL
    rights = by["rights-s1"]
    assert (rights.role, rights.of, rights.spdx) == ("license", "s1", "CC-BY-NC-SA-4.0")
    assert packet.amendments[-1].by == "paper-texts"
    kinds = describe_text.candidates(json.loads(data.packet("f-resnet").read_text()))
    assert kinds == {"s1": "full", "s1-pdf": "full"}
    assert paper_texts.texts(None, data=data, store=store) == [] and len(web.asked) == 2


def test_a_closed_doi_fetches_nothing_and_rests(roots, monkeypatch):
    data, store = roots
    doi = Source(sid="s1", role="paper", url="https://doi.org/10.1109/ACCESS.2023.3266068")
    write_json(_packet("f-closed", doi), data.packet("f-closed"))
    work = "https://api.openalex.org/works/doi:10.1109/ACCESS.2023.3266068"
    _web(monkeypatch, {work: (json.dumps({"best_oa_location": None, "locations": []}).encode(),
                              "application/json")})
    [row] = paper_texts.texts(None, data=data, store=store)
    assert row["status"] == "none" and "no open-access location" in row["why"]
    assert read_json(Packet, data.packet("f-closed")).sources == (doi,)
    assert paper_texts.texts(None, data=data, store=store) == []  # resting


def test_an_open_doi_gains_its_open_access_pdf_with_the_location_license(roots, monkeypatch):
    data, store = roots
    doi = Source(sid="s1", role="paper", url="https://doi.org/10.1234/abc")
    write_json(_packet("f-open", doi), data.packet("f-open"))
    work = {"best_oa_location": {"pdf_url": "https://example.org/a.pdf", "license": "cc-by",
                                 "is_oa": True}}
    _web(monkeypatch, {"https://api.openalex.org/works/doi:10.1234/abc": (json.dumps(work).encode(), "application/json"),
                       "https://example.org/a.pdf": (PDF, "application/pdf")})
    [row] = paper_texts.texts(None, data=data, store=store)
    assert row["sids"] == ["s1-pdf"]
    by = {s.sid: s for s in read_json(Packet, data.packet("f-open")).sources}
    assert by["s1"] == doi and by["rights-s1-pdf"].spdx == "CC-BY-4.0"
    assert paper_texts.texts(None, data=data, store=store) == []  # the pdf sibling settles it


def test_a_failed_pdf_keeps_the_abstract_page_and_a_dry_run_writes_nothing(roots, monkeypatch):
    data, store = roots
    write_json(_packet("f-resnet", ARXIV), data.packet("f-resnet"))
    write_json(_packet("f-has", Source(sid="p", role="paper", url="https://arxiv.org/abs/1",
                                       blob="a" * 64, text="b" * 64)), data.packet("f-has"))
    before = {p: p.read_bytes() for p in data.root.rglob("*") if p.is_file()}
    web = _web(monkeypatch, {"https://arxiv.org/abs/1512.03385": (ABS.encode(), "text/html")})
    assert paper_texts.texts(None, data=data, store=store, write=False) == [
        {"family": "f-resnet", "status": "due"}]
    assert web.asked == [] and before == {p: p.read_bytes() for p in data.root.rglob("*") if p.is_file()}
    [row] = paper_texts.texts(None, data=data, store=store)
    assert row["sids"] == ["s1"]
    assert "s1-pdf" not in {s.sid for s in read_json(Packet, data.packet("f-resnet")).sources}


@pytest.mark.parametrize(("url", "spdx"), [
    ("http://creativecommons.org/licenses/by/4.0/", "CC-BY-4.0"),
    ("http://creativecommons.org/publicdomain/zero/1.0/", "CC0-1.0"),
    ("http://arxiv.org/licenses/nonexclusive-distrib/1.0/", "LicenseRef-arXiv-nonexclusive-distrib-1.0"),
    ("https://example.org/terms", "unknown")])
def test_rights_links_become_spdx_ids(url, spdx):
    assert paper_texts.spdx_of(url) == spdx


def test_the_trickle_adds_paper_texts_only_when_configured(monkeypatch, tmp_path):
    from menagerie.pipeline import find_paper

    monkeypatch.setenv("MENAGERIE_STORE", str(tmp_path / "store"))
    calls = []
    monkeypatch.setattr(find_paper, "harvest", lambda *a, **k: [{"family": "f-p"}])
    monkeypatch.setattr(paper_texts, "texts", lambda *a, **k: calls.append(k) or [
        {"family": "f-t", "status": "text"}])
    data = DataLayout(tmp_path / "data")
    assert tick_full.trickle(data, {"trickle_limit": 5}, dry_run=False) == [{"family": "f-p"}]
    got = tick_full.trickle(data, {"trickle_limit": 5, "paper_texts": 4, "paper_texts_seconds": 20},
                            dry_run=False)
    assert [r["family"] for r in got] == ["f-p", "f-t"]
    assert (calls[0]["limit"], calls[0]["seconds"], calls[0]["write"]) == (4, 20.0, True)


def test_the_backlog_unit_commits_exactly_what_each_batch_wrote(roots, monkeypatch, tmp_path):
    import subprocess

    from menagerie.pipeline import tick_commit

    data, store = roots
    repo = data.root.parent
    data.root.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    monkeypatch.setenv("MENAGERIE_DATA", str(data.root))
    monkeypatch.setenv("MENAGERIE_STORE", str(tmp_path / "store"))
    batches = [[_packet("f-a", ARXIV)], [_packet("f-b", ARXIV)], []]

    def texts(families, *, data, store, limit):
        assert limit == 1
        packets = batches.pop(0)
        for packet in packets:
            write_json(packet, data.packet(packet.family))
        return [{"family": p.family, "status": "text"} for p in packets]

    commits = []
    monkeypatch.setattr(paper_texts, "texts", texts)
    monkeypatch.setattr(tick_commit, "commit_owned", lambda repo, owned, message, deadline: (
        commits.append((sorted(owned.files), message)) or tick_commit.Outcome(commit="c")))
    assert paper_texts.main(["--commit-every", "1"]) == 0
    assert [paths for paths, _ in commits] == [[data.packet("f-a").relative_to(repo).as_posix()],
                                               [data.packet("f-b").relative_to(repo).as_posix()]]
    assert "1 gained a text" in commits[0][1]
    runs = (tmp_path / "work" / "paper-texts" / "runs.jsonl").read_text().splitlines()
    assert len(runs) == 3 and json.loads(runs[-1])["rows"] == []


CODE = Source(sid="s1", role="code", url="menagerie/classics/example.py")


def _cited(family: str, **ids: str) -> Packet:
    return replace(_packet(family, CODE), citation=Citation(title="Deep Residual Learning", **ids))


def test_a_citation_no_source_names_gets_its_paper_as_p_cited(roots, monkeypatch):
    data, store = roots
    write_json(_cited("f-resnet", arxiv="1512.03385"), data.packet("f-resnet"))
    _web(monkeypatch, {"https://arxiv.org/abs/1512.03385": (ABS.encode(), "text/html"),
                       "https://arxiv.org/pdf/1512.03385": (PDF, "application/pdf")})
    [row] = paper_texts.texts(None, data=data, store=store)
    assert row == {"family": "f-resnet", "status": "text", "sids": ["p-cited", "p-cited-pdf"]}
    by = {s.sid: s for s in read_json(Packet, data.packet("f-resnet")).sources}
    assert by["p-cited"].url == "https://arxiv.org/abs/1512.03385" and by["p-cited"].text
    assert (by["rights-p-cited"].of, by["rights-p-cited"].spdx) == ("p-cited", "CC-BY-NC-SA-4.0")
    assert paper_texts.texts(None, data=data, store=store) == []


def test_a_citation_a_paper_source_already_names_is_not_fetched_twice(roots):
    data, _ = roots
    packet = replace(_packet("f-resnet", ARXIV), citation=Citation(title="x", arxiv="1512.03385"))
    assert [s.sid for s in paper_texts._wanting(packet)] == ["s1"]


def test_citation_only_packets_come_after_the_rest(roots):
    data, store = roots
    write_json(_cited("f-a", doi="10.1234/abc"), data.packet("f-a"))
    write_json(_packet("f-z", ARXIV), data.packet("f-z"))
    rows = paper_texts.texts(None, data=data, store=store, write=False)
    assert [r["family"] for r in rows] == ["f-z", "f-a"]


def _describe(data: DataLayout, family: str, **record) -> None:
    path = data.describe(family)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"family": family, **record}))


def test_identify_tries_textless_families_without_an_id_and_rests_misses(roots, monkeypatch):
    from menagerie.pipeline import find_paper

    data, store = roots
    for family in ("f-found", "f-miss", "f-old", "f-full", "f-known"):
        write_json(_packet(family, CODE), data.packet(family))
    write_json(_cited("f-known", arxiv="1512.03385"), data.packet("f-known"))
    _describe(data, "f-found", basis="none")
    _describe(data, "f-miss", basis="none")
    _describe(data, "f-old", **{"class": "describe:no_text"})
    _describe(data, "f-full", basis="full")
    _describe(data, "f-known", basis="none")
    asked = []

    def find(families, *, data, store):
        asked.extend(families)
        status = {"f-found": "found", "f-old": "held"}.get(families[0], "needs_2a")
        return [{"family": families[0], "status": status}]

    monkeypatch.setattr(find_paper, "find", find)
    assert [r["family"] for r in paper_texts.identify(None, data=data, store=store, write=False)] \
        == ["f-found", "f-miss", "f-old"]
    rows = paper_texts.identify(None, data=data, store=store, limit=5)
    assert [r["status"] for r in rows] == ["found", "needs_2a", "held"] and asked == [
        "f-found", "f-miss", "f-old"]
    assert [r["family"] for r in paper_texts.identify(None, data=data, store=store, write=False)] \
        == ["f-found"]


def test_the_trickle_identifies_after_fetching_texts(roots, monkeypatch, tmp_path):
    from menagerie.pipeline import find_paper

    data, _ = roots
    monkeypatch.setenv("MENAGERIE_STORE", str(tmp_path / "store"))
    monkeypatch.setattr(find_paper, "harvest", lambda *a, **k: [])
    calls = []
    monkeypatch.setattr(paper_texts, "texts", lambda *a, **k: calls.append("texts") or [])
    monkeypatch.setattr(paper_texts, "identify",
                        lambda *a, **k: calls.append(("identify", k["limit"], k["seconds"])) or [])
    tick_full.trickle(data, {"trickle_limit": 0, "paper_texts": 2, "paper_find": 4},
                      dry_run=True)
    assert calls == ["texts", ("identify", 4, 60.0)]


def test_find_reads_a_module_docstring_paper_line():
    from menagerie.pipeline import find_paper

    text = ('"""Mogrifier LSTM.\n\nPaper: "Mogrifier LSTM", Melis, Dyer, and Blunsom, ICLR 2020.\n'
            '"""\n')
    other = '"""Arrow LM.\n\nPaper: Tarau 2026, "Arrow Language Models."\n"""\n'
    assert find_paper._titles(None, [("s1", text)]) == ["Mogrifier LSTM"]
    assert find_paper._titles(None, [("s1", other)]) == ["Arrow Language Models"]
