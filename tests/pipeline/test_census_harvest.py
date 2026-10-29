"""Harvest provenance, independent title agreement and agent confirmation gates."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from menagerie.pipeline import (
    DataLayout,
    StoreLayout,
    append_jsonl,
    bib_harvest,
    dossier,
    fetch,
    find_paper,
    read_json,
    read_jsonl,
    research,
    write_json,
)
from menagerie.pipeline import census_harvest as harvest
from menagerie.pipeline.records import (
    Amendment,
    Availability,
    CensusRow,
    Dossier,
    Identity,
    Member,
    Packet,
    RosterRow,
    Search,
    Source,
    dumps,
)

AT = "2026-10-09T12:00:00Z"
COMMIT = "a" * 40
TITLE = "Example: an architecture for images"
REF = "arXiv:2103.14899"


@pytest.fixture
def roots(tmp_path, monkeypatch):
    data, store = DataLayout(tmp_path / "data"), StoreLayout(tmp_path / "store")
    root = data.root / "sweeps" / "census-2026-10"
    root.mkdir(parents=True)
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setattr(bib_harvest, "_gather", gather)
    return data, store, root


def row(id="m1", repo="owner/example", **fields):
    return CensusRow(id=id, repo=repo, checked=AT, alive=True, archived=False,
                     pytorch="imports", size_kb=10, eligible=True, family="f-example",
                     license="MIT", **fields)


def gather(*args, **kwargs):
    sources = (Source(sid="b1", role="bib", api="openalex", blob="a" * 64),
               Source(sid="b2", role="bib", api="bibtex", blob="b" * 64))
    records = {"openalex": {"title": TITLE, "id": "https://openalex.org/W1"},
               "bibtex": "@misc{x,\n title={" + TITLE + "},\n year={2021}\n}"}
    return fetch.BibResult(sources, (), records), records, ["openalex"]


def seed(roots, monkeypatch, texts, records=None):
    data, store, root = roots
    records = records or [row()]
    append_jsonl(records, root / "census.jsonl")
    append_jsonl([RosterRow(name="Example", zoo="github", stable_id=r.id) for r in records], data.roster)
    for r in records:
        checkpoint = {"repo": r.repo, "commit": COMMIT, "branch": "main",
                      "tree": [{"path": p, "size": len(t.encode())} for p, t in texts.items()]}
        path = harvest.checkpoint_path(root, r.repo)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(checkpoint))
    calls = []

    def http(url, **kwargs):
        calls.append(url)
        name = url.split(COMMIT + "/", 1)[1]
        return SimpleNamespace(status=200, body=texts[name].encode())

    monkeypatch.setattr(fetch, "_http", http)
    monkeypatch.setattr(fetch, "_put_bytes", putter(store))
    return calls


def putter(store):
    import hashlib

    def put(body, **kwargs):
        sha = hashlib.sha256(body).hexdigest()
        path = store.blob(sha)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return SimpleNamespace(sha256=sha)
    return put


def test_readme_bib_title_license_and_resume(roots, monkeypatch):
    data, store, root = roots
    texts = {"README.md": f"https://arxiv.org/abs/2103.14899\n@misc{{x,title={{{TITLE}}},}}",
             "LICENSE": "MIT license", "CITATION.bib": f"@misc{{x,title={{{TITLE}}},}}"}
    calls = seed(roots, monkeypatch, texts)
    before = (root / "census.jsonl").read_bytes()
    assert harvest.harvest(data=data, store=store)[0]["status"] == "harvested"
    value = read_jsonl(CensusRow, root / "harvest.jsonl")[0]
    assert value.paper_refs == (REF,) and value.paper_title == TITLE
    assert value.readme_sha and value.citation_file == "CITATION.bib" and value.license == "MIT"
    assert all(COMMIT in u and "raw.githubusercontent.com" in u for u in calls)
    assert dossier.build(None, data=data, store=store)[0]["paper_verdict"] == "agrees"
    proposal = read_jsonl(Dossier, dossier.path(data))[0]
    assert proposal.commit == COMMIT and proposal.paper == REF
    count = len(calls)
    assert harvest.harvest(data=data, store=store)[0]["status"] == "cached"
    assert len(calls) == count and (root / "census.jsonl").read_bytes() == before


def test_cff_preferred_citation_and_no_readme(roots, monkeypatch):
    data, store, root = roots
    text = ("cff-version: 1.2.0\ntitle: Example software\npreferred-citation:\n"
            f"  type: article\n  title: '{TITLE}'\n  doi: 10.1234/example\n")
    seed(roots, monkeypatch, {"CITATION.cff": text})
    harvest.harvest(data=data, store=store)
    value = read_jsonl(CensusRow, root / "harvest.jsonl")[0]
    assert value.readme_sha is None and value.paper_title == TITLE
    assert value.paper_refs == ("10.1234/example",)


def test_fork_citing_other_paper_stays_held(roots, monkeypatch):
    data, store, _root = roots
    seed(roots, monkeypatch, {"README.md": f"arXiv:2103.14899\n@misc{{x,title={{{TITLE}}},}}"},
         [row(repo="fork/different-model")])
    harvest.harvest(data=data, store=store)
    assert dossier.build(None, data=data, store=store)[0]["paper_verdict"] == "held"
    assert not dossier.name_agrees("NotExample: an architecture", "Example", "x/example")
    assert not dossier.name_agrees("Example", "Example", "x/example")


def test_cff_software_doi_is_held_even_when_apis_agree(roots, monkeypatch):
    data, store, _root = roots
    text = f"cff-version: 1.2.0\ntitle: '{TITLE}'\ndoi: 10.1234/example\n"
    seed(roots, monkeypatch, {"CITATION.cff": text})
    harvest.harvest(data=data, store=store)
    assert find_paper._titles(None, [("CITATION.cff", text)]) == []
    assert dossier.build(None, data=data, store=store)[0]["paper_verdict"] == "held"


def test_non_utf8_and_404_documents_do_not_lose_the_code_dossier(roots, monkeypatch):
    data, store, root = roots
    seed(roots, monkeypatch, {"README": "x", "LICENSE": "x"})

    def http(url, **kwargs):
        return SimpleNamespace(status=404 if url.endswith("LICENSE") else 200, body=b"\xff")

    monkeypatch.setattr(fetch, "_http", http)
    assert harvest.harvest(data=data, store=store)[0]["status"] == "harvested"
    evidence = json.loads((root / "harvest-files/m1.json").read_text())
    assert {d["status"] for d in evidence["warnings"]} == {"not_utf8", "not_found"}
    assert dossier.build(None, data=data, store=store)[0]["status"] == "built"


def test_missing_files_create_dossier_without_paper(roots, monkeypatch):
    data, store, _root = roots
    seed(roots, monkeypatch, {})
    harvest.harvest(data=data, store=store)
    assert dossier.build(None, data=data, store=store)[0]["status"] == "built"
    value = read_jsonl(Dossier, dossier.path(data))[0]
    assert value.paper is None and value.readme_sha is None


def test_bibliographic_failure_keeps_code_hypothesis_and_reports_warning(roots, monkeypatch):
    data, store, _root = roots
    seed(roots, monkeypatch, {"README": f"arXiv:2103.14899\n@misc{{x,title={{{TITLE}}},}}"})
    harvest.harvest(data=data, store=store)

    def failed(*args, **kwargs):
        raise fetch.FetchError("API unavailable")

    monkeypatch.setattr(bib_harvest, "_gather", failed)
    result = dossier.build(None, data=data, store=store)[0]
    assert result["status"] == "built" and result["warnings"]
    value = read_jsonl(Dossier, dossier.path(data))[0]
    assert value.paper_verdict == "held" and value.commit == COMMIT


def test_dry_run_has_no_fetch_or_output_and_dead_rows_are_not_typed(roots, monkeypatch):
    data, store, root = roots
    dead = CensusRow(id="m2", repo="owner/dead", checked=AT, alive=False, why="not_alive")
    calls = seed(roots, monkeypatch, {"README": "Example"}, [row(), dead])
    result = harvest.harvest(data=data, store=store, write=False)
    assert [r["id"] for r in result] == ["m1"]
    assert not calls and not (root / "harvest.jsonl").exists()
    assert not list(data.root.glob("models/**/*.json"))


def test_transient_fetch_failure_retries_instead_of_recording_missing(roots, monkeypatch):
    data, store, root = roots
    seed(roots, monkeypatch, {"README.md": "Example"})
    monkeypatch.setattr(fetch, "_http", lambda *a, **k: SimpleNamespace(status=503, body=b""))
    assert harvest.harvest(data=data, store=store)[0]["status"] == "retry"
    assert not (root / "harvest.jsonl").exists()


def test_unsafe_truncated_and_unpinned_checkpoints_are_refused(roots, monkeypatch):
    data, store, root = roots
    seed(roots, monkeypatch, {"README.md": "Example"})
    path = harvest.checkpoint_path(root, "owner/example")
    point = json.loads(path.read_text())
    for change in ({"commit": "main"}, {"truncated": True}):
        path.write_text(json.dumps({**point, **change}))
        assert harvest.harvest(data=data, store=store)[0]["status"] == "invalid_checkpoint"
    tree = {"tree": [{"path": "../README", "type": "blob", "size": 10},
                      {"path": "README", "type": "blob", "mode": "120000", "size": 1}]}
    assert harvest.candidate_paths(tree) == []


def test_hash_order_limits_repositories_without_losing_duplicate_rows(roots, monkeypatch):
    _data, _store, root = roots
    seed(roots, monkeypatch, {}, [row(), row(id="m2"), row(id="m3", repo="z/other")])
    rows = harvest.selected(root, 1)
    assert len({r.repo for r in rows}) == 1
    assert {r.id for r in rows} in ({"m1", "m2"}, {"m3"})
    assert harvest.selected(root, 0) == []
    with pytest.raises(ValueError):
        harvest.selected(root, -1)


def test_prepare_resolves_default_branch_and_keeps_census_unchanged(roots, monkeypatch):
    data, _store, root = roots
    append_jsonl([row()], root / "census.jsonl")
    calls = []

    def get(self, endpoint):
        calls.append(endpoint)
        if "/commits/" in endpoint:
            return {"sha": COMMIT}
        if "/git/trees/" in endpoint:
            return {"tree": [{"path": "README.md", "type": "blob", "size": 10}]}
        return {"default_branch": "branch/with-slash", "archived": False}

    monkeypatch.setattr(harvest.census.GitHub, "get", get)
    before = (root / "census.jsonl").read_bytes()
    assert harvest.prepare(data=data)[0]["status"] == "prepared"
    assert "branch%2Fwith-slash" in calls[1] and COMMIT in calls[2]
    assert harvest.prepare(data=data)[0]["status"] == "cached"
    assert len(calls) == 3 and before == (root / "census.jsonl").read_bytes()


def test_dossier_brief_is_opt_in_and_requires_line_confirmation(roots, monkeypatch):
    data, store, root = roots
    seed(roots, monkeypatch, {"README": f"arXiv:2103.14899\n@misc{{x,title={{{TITLE}}},}}"})
    harvest.harvest(data=data, store=store)
    dossier.build(None, data=data, store=store)
    assert dossier.brief(["m1"], data=data, enabled=False) == ""
    text = dossier.brief(["m1"], data=data, enabled=True)
    assert "confirm or replace every line" in text and "dossier-confirmed:m1:" in text
    assert dossier.brief(["m9"], data=data, enabled=True) == ""
    batch = {"label": "trial", "families": [{"family": "f-example", "ids": ["m1"]}]}
    monkeypatch.setattr(research, "seed_family", lambda *a, **k: {
        "family": "f-example", "rows": [], "candidates": [], "more": 0, "existing": None})
    monkeypatch.setattr(research, "_sc_notes", lambda *a: [])
    plain = research.brief(batch, data=data, lane_dir=root)
    assert "untrusted hypothesis" not in plain
    assert plain == research.brief({**batch, "dossiers": False}, data=data, lane_dir=root)
    assert research.brief({**batch, "dossiers": True}, data=data, lane_dir=root).startswith(
        "Scripted dossier (untrusted hypothesis)")
    original, _ = dossier.template(research.PROMPT.read_text())
    assert "<!-- DOSSIER -->" not in original and "{DOSSIER}" not in plain


def test_from_dossier_needs_confirmation_pin_and_agreeing_title(roots, monkeypatch):
    data, store, _root = roots
    seed(roots, monkeypatch, {"README": f"arXiv:2103.14899\n@misc{{x,title={{{TITLE}}},}}"})
    harvest.harvest(data=data, store=store)
    dossier.build(None, data=data, store=store)
    value = read_jsonl(Dossier, dossier.path(data))[0]
    packet = Packet(family="f-example", written=AT, by="research", members=(Member(id="m1"),),
                    identity=Identity(canonical="Example"),
                    sources=(Source(sid="s1", role="code", url="https://github.com/owner/example",
                                    rev=COMMIT, blob="c" * 64),),
                    availability=Availability(paper="not_found", code="found", license="found", description="sufficient"),
                    search=Search(places=("github",), queries=("Example paper",)))
    write_json(packet, data.packet(packet.family))
    assert find_paper.find([packet.family], data=data, store=store, from_dossier=True)[0]["status"] == "held"
    confirmed = replace(packet, search=replace(packet.search, note=dossier.marker(value)))
    write_json(replace(confirmed, sources=(replace(packet.sources[0], blob=None),)), data.packet(packet.family))
    assert find_paper.find([packet.family], data=data, store=store, from_dossier=True)[0]["status"] == "held"
    write_json(replace(confirmed, sources=(replace(packet.sources[0], rev="b" * 40),)), data.packet(packet.family))
    assert find_paper.find([packet.family], data=data, store=store, from_dossier=True)[0]["status"] == "held"
    write_json(confirmed, data.packet(packet.family))
    dossier.path(data).write_text(dumps(replace(value, paper_verdict="held")) + "\n")
    assert find_paper.find([packet.family], data=data, store=store, from_dossier=True)[0]["status"] == "held"
    dossier.path(data).write_text(dumps(value) + "\n")
    assert find_paper.find([packet.family], data=data, store=store, from_dossier=True, write=False)[0]["status"] == "found"
    assert read_json(Packet, data.packet(packet.family)).citation is None
    assert find_paper.find([packet.family], data=data, store=store, from_dossier=True)[0]["status"] == "found"
    accepted = read_json(Packet, data.packet(packet.family))
    assert accepted.citation.arxiv == "2103.14899" and accepted.citation.from_ == "s1"
    assert dossier.marker(value) in accepted.amendments[-1].why
    find_paper.apply(packet.family, {"no_paper": "unpublished", "evidence": ["s1"], "reason": "corrected"},
                     data=data, store=store)
    rejected = read_json(Packet, data.packet(packet.family))
    assert "dossier-confirmed:" not in rejected.search.note
    write_json(replace(rejected, search=replace(rejected.search, note=dossier.marker(value))), data.packet(packet.family))
    assert find_paper.find([packet.family], data=data, store=store, from_dossier=True)[0]["status"] == "no_paper"
    held = replace(confirmed, amendments=(Amendment(at=AT, why="titleless: spot check",
                   evidence=("s1",), changed=("citation",), by=bib_harvest.BY),))
    write_json(held, data.packet(packet.family))
    assert find_paper.find([packet.family], data=data, store=store, from_dossier=True)[0]["status"] == "held"
