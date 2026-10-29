"""Rung 1 research (2a): family seeds, packet write, lane briefs and the second code search.

Everything runs on temporary data and store roots built here: a small roster with the shapes the
rung 1 rows test (checkpoint names, acronyms, parentheticals, shared repositories, S-C
candidates, paper ids, a name collision), saved blobs, and faked bibliographic records and lane
launches. The real-data runs are the lane's acceptance evidence, not unit tests.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from menagerie.identity import ensure_stable_ids
from menagerie.pipeline import DataLayout, StoreLayout, drive, fetch, read_json, read_jsonl, research
from menagerie.pipeline import store as store_mod
from menagerie.pipeline.records import (
    CitationLine, ModelRecord, Packet, RecordError, RosterRow, Source, dumps,
)

FIXTURES = Path(__file__).parent / "fixtures"
SNR_REPO = "https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance"
# (name, zoo, gets an id, extra fields)
ROWS: list[tuple[str, str, bool, dict[str, Any]]] = [
    ("dino_r50_4scale_official", "IDEA-Research/DINO", True, {}),
    ("dino_r50_4scale", "base-pytorch-compact", True, {}),
    ("dino_swin_l_4scale", "base-pytorch-compact", True, {}),
    ("DINO-DETR", "discovered-pytorch", False, {}),
    ("DINOProjectionHead", "lightly", True, {}),
    ("NetworkInNetwork-CIFAR", "base-pytorch-local", True, {}),
    ("NIN", "historical-classics", True, {}),
    ("NetworkInNetwork-CIFAR (Lin 2013: mlpconv blocks)", "unregistered-classics-pytorch", True, {}),
    ("nerfstudio_instant_ngp", "unregistered-classics-pytorch", True, {}),
    ("SNR-Net", "discovered-pytorch", True, {"hint_urls": (SNR_REPO,)}),
    ("SNR-Aware", "discovered-pytorch", False, {"hint_urls": (SNR_REPO,)}),
    ("Kalman Variational Autoencoder (KVAE)", "discovered-pytorch", True, {}),
    ("AutoencoderKLKVAE", "diffusers", True, {}),
    ("ScheduleNet", "discovered-pytorch", True, {"found_by": ("S-C:cand_05531",)}),
    ("Multi-agent scheduler", "discovered-pytorch", False, {"found_by": ("S-C:cand_05531",)}),
    ("EdgeNet (Encoder-Decoder Neural Auction)", "discovered-pytorch", True,
     {"hint_paper": "arxiv:2305.06158"}),
    ("Neural auction encoder", "discovered-pytorch", False,
     {"hint_urls": ("https://arxiv.org/abs/2305.06158",)}),
    ("Groove MIDI Dataset", "discovered-pytorch", True, {}),
    ("ResNet-50", "torchvision", True, {}),
]
PAPER_TEXT = ("Learning to See in the Dark\nWe propose a signal-to-noise-ratio-aware ﬁlter "
              "for low-light enhancement.\nThe Groove MIDI Dataset is a dataset of drumming.\n")


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    data, st = DataLayout(tmp_path / "data"), StoreLayout(tmp_path / "store")
    data.root.mkdir()
    keys = {(name, zoo, "") for name, zoo, has_id, _ in ROWS if has_id}
    ids = ensure_stable_ids(keys, data.root / "stable_ids.jsonl")
    rows = [RosterRow(name=n, zoo=z, stable_id=ids.get((n, z, "")) if has else None, **extra)
            for n, z, has, extra in ROWS]
    data.roster.write_text("".join(dumps(r) + "\n" for r in rows), encoding="utf-8")
    plan = data.root / "sweeps" / "S-C" / "IMPL_PLAN.tsv"
    plan.parent.mkdir(parents=True)
    plan.write_text("cand_05531\tScheduleNet\tSOURCE_AVAILABLE\thttps://github.com/z/s\n"
                    "cand_00001\tAGIF\tSOURCE_AVAILABLE\n", encoding="utf-8")
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(drive, "event", lambda lane, family, verb, **f: events.append(
        {"family": family, "verb": verb, **f}))
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.delenv("MENAGERIE_STORE_REMOTE", raising=False)
    by_name = {r.name: r for r in rows}
    return {"data": data, "store": st, "ids": {n: r.stable_id for n, r in by_name.items()},
            "events": events, "tmp": tmp_path}


def _put(env: dict[str, Any], body: bytes, *, url: str | None = None, mime: str = "text/plain",
         derived_from: str | None = None) -> str:
    path = env["tmp"] / "blob.tmp"
    path.write_bytes(body)
    return store_mod.put(path, store=env["store"], data=env["data"], mime=mime,
                         license_class="third-party", url=url, derived_from=derived_from).sha256


# --- family seeds ---------------------------------------------------------------------------


def _why(seed: dict[str, Any], name: str) -> list[str]:
    return next(c["why"] for c in seed["candidates"] if c["name"] == name)


def test_seed_family_lists_duplicates_stand_ins_and_collisions_with_reasons(env: dict) -> None:
    data, ids = env["data"], env["ids"]
    dino = research.seed_family([ids["dino_r50_4scale_official"]], data=data)
    assert dino["family"] == "f-dino" and dino["members"] == [ids["dino_r50_4scale_official"]]
    names = {c["name"] for c in dino["candidates"]}
    assert {"dino_r50_4scale", "dino_swin_l_4scale", "DINO-DETR"} <= names
    assert "DINOProjectionHead" not in names  # one token, not a stem
    assert next(c for c in dino["candidates"] if c["name"] == "DINO-DETR")["id"] is None
    nin = research.seed_family([ids["NetworkInNetwork-CIFAR"]], data=data)
    assert "alias" in _why(nin, "NIN")
    assert "same normalized name" in _why(nin, "NetworkInNetwork-CIFAR (Lin 2013: mlpconv blocks)")
    assert "nerfstudio_instant_ngp" not in {c["name"] for c in nin["candidates"]}
    assert _why(research.seed_family([ids["SNR-Net"]], data=data), "SNR-Aware") == [
        "same repository"]
    kvae = research.seed_family([ids["Kalman Variational Autoencoder (KVAE)"]], data=data)
    assert kvae["members"] == [ids["Kalman Variational Autoencoder (KVAE)"]]
    assert _why(kvae, "AutoencoderKLKVAE") == ["alias 'kvae' inside the name"]
    assert "same S-C candidate" in _why(research.seed_family([ids["ScheduleNet"]], data=data),
                                        "Multi-agent scheduler")
    edge = research.seed_family([ids["EdgeNet (Encoder-Decoder Neural Auction)"]], data=data)
    assert _why(edge, "Neural auction encoder") == ["same paper id"]
    assert all("ResNet-50" != c["name"] for s in (dino, nin, kvae, edge) for c in s["candidates"])


def test_seed_family_keys_the_roster_once_while_it_is_unchanged(
        env: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    data, ids = env["data"], env["ids"]
    keyed: list[str] = []
    real = research._keys
    monkeypatch.setattr(research, "_KEYED", {})
    monkeypatch.setattr(research, "_keys", lambda row: keyed.append(row.name) or real(row))
    first = research.seed_family([ids["SNR-Net"]], data=data)
    rows = len(keyed)
    assert research.seed_family([ids["SNR-Net"]], data=data) == first
    assert len(keyed) == rows + 1  # the second seed keys only its input row
    with data.roster.open("a", encoding="utf-8") as handle:
        handle.write(dumps(RosterRow(name="SNR-Aware v2", zoo="discovered-pytorch")) + "\n")
    research.seed_family([ids["SNR-Net"]], data=data)
    assert keyed[rows + 1:].count("SNR-Aware v2") == 1 and len(keyed) == 2 * rows + 2  # re-keyed


def test_packet_lookup_shares_the_memo_with_next_steps(env: dict) -> None:
    import os
    import shutil

    data = env["data"]
    fixture = Path(__file__).parent / "fixtures" / "f-resnet.packet.json"
    packet = read_json(Packet, fixture)
    path = data.packet(packet.family)
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(fixture, path)
    os.utime(path, (1, 1))  # an old stamp: the memo serves it from now on
    drive._load(data).packet(packet.family)  # next_steps memoises the packet record first
    assert research._packet_of(data) == {m.id: packet.family for m in packet.members}


def test_seed_family_refuses_an_unknown_id(env: dict) -> None:
    with pytest.raises(RecordError, match="m99999"):
        research.seed_family(["m99999"], data=env["data"])


# --- packet write ---------------------------------------------------------------------------


@pytest.fixture()
def draft(env: dict, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    page = _put(env, b"<html>Learning to See in the Dark</html>", url="https://arxiv.org/abs/x",
                mime="text/html")
    text = _put(env, PAPER_TEXT.encode(), derived_from=page)
    snapshot = _put(env, store_mod.pack_snapshot(
        {"README.md": b"# SNR-Aware\nCode for SNR-aware low-light enhancement.\n",
         "LICENSE": b"MIT License\n"}, {"url": SNR_REPO}), url=SNR_REPO, mime=store_mod.SNAPSHOT_MIME)
    _put(env, b"MIT License\n", url=f"{SNR_REPO}/LICENSE", derived_from=snapshot)

    def bib(ref: str, *, family: str, store: StoreLayout, data: DataLayout) -> fetch.BibResult:
        sha = _put(env, json.dumps({"paperId": "p1"}).encode(), mime="application/json")
        return fetch.BibResult(
            (Source(sid=f"bib-s2-{sha[:8]}", role="bib", api="s2", blob=sha),),
            (f"bib {ref}: openalex refused (insufficient budget)",),
            {"s2": {"paperId": "p1", "citationCount": 12},
             "crossref": {"message": {"published-print": {"date-parts": [[2022, 6, 19]]}}}})

    monkeypatch.setattr(fetch, "bib_records", bib)
    snr = env["ids"]["SNR-Net"]
    return {
        "members": [{"id": snr, "delta": ""}],
        "identity": {"canonical": "SNR-Aware Low-Light Enhancement", "aliases": ["SNR-Net"]},
        "sources": [{"sid": "s1", "role": "paper", "url": "https://arxiv.org/abs/x", "blob": page,
                     "text": text, "official": True},
                    {"sid": "s2", "role": "code", "url": SNR_REPO, "rev": "c" * 40,
                     "blob": snapshot, "framework": "pytorch", "spdx": "MIT", "official": True,
                     "weights": "found"}],
        "citation": {"title": "Learning to See in the Dark", "arxiv": "2206.00001", "from": "s1"},
        "availability": {"paper": "found", "code": "found", "license": "found",
                         "description": "sufficient"},
        "code_order": ["s2"],
        "claims": [
            {"field": "identity", "sid": "s1",
             "quote": "We propose a signal-to-noise-ratio-aware filter for low-light enhancement."},
            {"field": "roster:SNR-Aware@discovered-pytorch=duplicate_of:" + snr, "sid": "s2",
             "quote": "Code for SNR-aware low-light enhancement."}],
        "search": {"queries": ["SNR-aware github"], "places": ["arxiv", "github"]},
    }


def _write(env: dict, family: str, body: dict[str, Any]) -> Packet:
    path = env["tmp"] / f"{family}.draft.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return research.packet_write(family, path, store=env["store"], data=env["data"])


def test_packet_write_refuses_a_quote_edited_by_one_word_and_accepts_the_original(
        env: dict, draft: dict) -> None:
    edited = json.loads(json.dumps(draft))
    edited["claims"][0]["quote"] = edited["claims"][0]["quote"].replace("filter", "prior")
    with pytest.raises(RecordError, match="quote not found verbatim in s1"):
        _write(env, "f-snr", edited)
    assert not env["data"].packet("f-snr").exists()
    packet = _write(env, "f-snr", draft)  # the ligature in the saved text normalizes to "fi"
    assert read_json(Packet, env["data"].packet("f-snr")) == packet


def test_packet_write_writes_bib_license_record_roster_state_and_citation_seed(
        env: dict, draft: dict) -> None:
    data, snr = env["data"], env["ids"]["SNR-Net"]
    before = data.roster.read_text(encoding="utf-8").splitlines()
    existing = replace(read_json(ModelRecord, FIXTURES / "m3631.model.json"), id=snr, delta={})
    data.model(snr).parent.mkdir(parents=True)
    data.model(snr).write_text(dumps(existing) + "\n", encoding="utf-8")
    packet = _write(env, "f-snr", draft)
    roles = [s.role for s in packet.sources]
    assert roles == ["paper", "code", "bib", "license"]
    license_source = packet.sources[-1]
    assert (license_source.of, license_source.spdx) == ("s2", "MIT")
    assert packet.citation.bib == (packet.sources[2].sid,)
    assert (packet.citation.s2, packet.citation.published) == ("p1", "2022-06-19")
    assert "openalex refused (insufficient budget)" in packet.search.note
    record = read_json(ModelRecord, data.model(snr))
    assert (record.name, record.family, record.source_case) == ("SNR-Net", "f-snr", "vendor")
    assert record.recipe == existing.recipe  # 2b's half is kept
    after = data.roster.read_text(encoding="utf-8").splitlines()
    changed = [(a, b) for a, b in zip(before, after) if a != b]
    assert len(after) == len(before) and len(changed) == 1
    assert json.loads(changed[0][1])["state"] == f"duplicate_of:{snr}"
    [seed] = read_jsonl(CitationLine, next((data.root / "cite").glob("*.jsonl")))
    assert (seed.family, seed.s2, seed.from_) == ("f-snr", 12, "s2")
    assert env["events"][-1]["verb"] == "packet_write"


def test_packet_write_lists_every_error_at_once(env: dict, draft: dict) -> None:
    bad = json.loads(json.dumps(draft))
    bad["availability"]["code"] = "not_found"
    bad["sources"][0]["text"] = "f" * 64
    bad["claims"][1]["field"] = "roster:Nobody@nowhere=not_architecture"
    bad["citation"].pop("arxiv")
    with pytest.raises(RecordError) as caught:
        _write(env, "f-snr", bad)
    message = str(caught.value)
    for part in ("not in the store index", "lacks Hugging Face papers, an open-web search",
                 "no roster row 'Nobody@nowhere'", "no_identifier", "second-search"):
        assert part in message, part


def test_packet_write_mints_a_member_that_has_no_stable_id(env: dict, draft: dict) -> None:
    body = json.loads(json.dumps(draft))
    body["members"].append({"name": "DINO-DETR", "zoo": "discovered-pytorch", "delta": "x"})
    packet = _write(env, "f-snr", body)
    minted = packet.members[-1].id
    row = next(json.loads(line) for line in env["data"].roster.read_text().splitlines()
               if json.loads(line)["name"] == "DINO-DETR")
    assert row["stable_id"] == minted and env["data"].model(minted).exists()


def test_a_hub_member_needs_its_card_and_config(env: dict, draft: dict) -> None:
    body = json.loads(json.dumps(draft))
    body["members"][0]["hf"] = "org/snr"
    with pytest.raises(RecordError, match="hub model org/snr"):
        _write(env, "f-snr", body)
    rev = "a" * 40
    card = _put(env, b"# SNR card\n", url=f"https://huggingface.co/org/snr/resolve/{rev}/README.md")
    config = _put(env, b'{"architectures": ["X"]}',
                  url=f"https://huggingface.co/org/snr/resolve/{rev}/config.json")
    body["sources"].append({"sid": "s3", "role": "page", "url": "https://huggingface.co/org/snr",
                            "blob": card, "text": card})
    _write(env, "f-snr", body)
    record = read_json(ModelRecord, env["data"].model(env["ids"]["SNR-Net"]))
    assert (record.hf.id, record.hf.card, record.hf.config) == ("org/snr", card, config)


def test_two_hub_repos_may_ship_the_same_config_bytes(env: dict, draft: dict) -> None:
    # Kimi-K2-Base and Kimi-K2-Instruct (m7490): one config.json blob, two repo addresses.
    body = json.loads(json.dumps(draft))
    body["members"][0]["hf"] = "org/snr-base"
    body["members"].append({"name": "DINO-DETR", "zoo": "discovered-pytorch", "delta": "x",
                            "hf": "org/snr-instruct"})
    config = b'{"architectures": ["X"]}'
    for repo in ("org/snr-base", "org/snr-instruct"):
        rev = "b" * 40
        card = _put(env, f"# {repo} card\n".encode(),
                    url=f"https://huggingface.co/{repo}/resolve/{rev}/README.md")
        shared = _put(env, config, url=f"https://huggingface.co/{repo}/resolve/{rev}/config.json")
        body["sources"].append({"sid": f"s-{repo[4:]}", "role": "page", "blob": card,
                                "url": f"https://huggingface.co/{repo}", "text": card})
    packet = _write(env, "f-snr", body)
    refs = {read_json(ModelRecord, env["data"].model(m.id)).hf for m in packet.members}
    assert {(r.id, r.config) for r in refs if r} == {("org/snr-base", shared),
                                                     ("org/snr-instruct", shared)}


def test_a_written_packet_changes_only_by_a_new_amendment(env: dict, draft: dict) -> None:
    _write(env, "f-snr", draft)
    with pytest.raises(RecordError, match="add one saying what changed"):
        _write(env, "f-snr", draft)
    draft["amendments"] = [{"at": "2026-10-04T00:00:00Z", "why": "second search found a fork",
                            "evidence": ["s2"], "changed": ["sources"], "by": "2a@prompt-v1"}]
    assert len(_write(env, "f-snr", draft).amendments) == 1


def test_an_amended_packet_keeps_what_2b_recorded_on_its_members(env: dict, draft: dict) -> None:
    # m12019 (2026-10-24): a hand packet write erased the 2b lane's delta and build notes.
    _write(env, "f-snr", draft)
    path = env["data"].model(env["ids"]["SNR-Net"])
    built = replace(read_json(ModelRecord, path), delta={"built_as": "the authors' code"})
    path.write_text(dumps(built))
    draft["amendments"] = [{"at": "2026-10-04T00:00:00Z", "why": "a roster finding",
                            "evidence": ["s2"], "changed": ["claims"], "by": "2a@prompt-v1"}]
    _write(env, "f-snr", draft)
    kept = read_json(ModelRecord, path)
    assert kept.delta == {"built_as": "the authors' code"}
    # A draft that gives the member new values still sets them.
    draft["members"][0]["delta_fields"] = {"built_as": "a port"}
    draft["amendments"].append({"at": "2026-10-05T00:00:00Z", "why": "rebuilt", "evidence": ["s2"],
                                "changed": ["members"], "by": "2a@prompt-v1"})
    _write(env, "f-snr", draft)
    assert read_json(ModelRecord, path).delta == {"built_as": "a port"}


def test_not_architecture_finding_and_derived_source_cases(env: dict, draft: dict) -> None:
    body = json.loads(json.dumps(draft))
    groove = env["ids"]["Groove MIDI Dataset"]
    body["claims"][1] = {"field": f"roster:{groove}=not_architecture", "sid": "s1",
                         "quote": "The Groove MIDI Dataset is a dataset of drumming."}
    body["code_order"], body["availability"]["code"] = [], "not_found"
    body["search"]["places"] += ["hf-papers", "web:google", "second-search"]
    _write(env, "f-snr", body)
    row = next(json.loads(line) for line in env["data"].roster.read_text().splitlines()
               if json.loads(line).get("stable_id") == groove)
    assert row["state"] == "not_architecture"
    record = read_json(ModelRecord, env["data"].model(env["ids"]["SNR-Net"]))
    assert record.source_case == "reimplement"



@pytest.mark.parametrize(("framework", "case"), [
    ("pytorch", "vendor"), ("PyTorch 1.0", "vendor"), ("pytorch (torch_geometric)", "vendor"),
    ("torch", "vendor"), ("jax (flax)", "native"), ("paddle (dygraph)", "native"),
    ("tensorflow", "native"), ("tensorflow 2 (tf.keras)", "native"), ("tensorflow 2.15", "native"),
    ("tensorflow 1.x", "port"), ("tensorflow 1.15", "port"), ("tensorflow 1", "port"),
    ("tensorflow 0.9 (prettytensor)", "port"), ("paddle 1.6.1 (fluid dygraph)", "port"),
    ("paddle 2.1", "native"), ("caffe", "port"), ("lua-torch", "port"), ("", "port"),
])
def test_framework_case_reads_the_first_word(framework: str, case: str) -> None:
    from menagerie.pipeline import packet as packets

    assert packets.framework_case(framework) == case

# --- lanes ----------------------------------------------------------------------------------


def test_launch_writes_one_brief_per_batch_and_leases_its_families(
        env: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    launched: list[dict[str, Any]] = []

    def fake(kind: str, role: str, brief: Path, **kw: Any) -> str:
        launched.append({"kind": kind, "role": role, "brief": brief, **kw})
        return f"run-{len(launched)}"

    monkeypatch.setattr(drive, "launch", fake)
    spec = env["tmp"] / "batch.json"
    spec.write_text(json.dumps({"parent": "p-1", "batches": [
        {"label": "r1-2a-test", "web_search_usd": 1.5, "tier": "github", "families": [
            {"ids": [env["ids"]["ScheduleNet"]], "hints": "repo zcaicaros/ScheduleNet"},
            {"ids": [env["ids"]["SNR-Net"]], "family": "f-snr", "tier": "paper-only"}]}]}))
    assert research.launch(spec, data=env["data"]) == ["run-1"]
    assert [(e["family"], e["verb"], e["tier"]) for e in env["events"]] == [
        ("f-schedulenet", "tier", "github"), ("f-snr", "tier", "paper-only")]
    [lane] = launched
    assert (lane["kind"], lane["role"], lane["keys"], lane["parent"], lane["web"]) == (
        "2a", "model-default", ["f-schedulenet", "f-snr"], "p-1", True)
    text = lane["brief"].read_text()
    assert "{PIPE}" not in text and "python3 -m menagerie.pipeline packet write" in text
    assert "cand_05531 | ScheduleNet | SOURCE_AVAILABLE" in text and "AGIF" not in text
    assert "12 fetches, 25 tool calls, 15 minutes" in text and "$1.50" in text
    assert "BRIEF.result.md" in text and "## Family `f-snr`" in text
    assert "web-search --caller menagerie-2a --purpose <family>" in text
    assert research.launch(spec, data=env["data"], role="model-implementation") == ["run-2"]
    assert launched[-1]["role"] == "model-implementation"


def test_the_brief_names_each_open_research_gap(env: dict,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    """``drive._gap_step`` counts only an amendment naming the give-up's class as the answer."""

    from menagerie.pipeline.records import GiveUp  # noqa: PLC0415
    from menagerie.pipeline import write_json  # noqa: PLC0415

    briefs: list[str] = []
    monkeypatch.setattr(drive, "launch", lambda kind, role, brief, **kw: (
        briefs.append(brief.read_text()), "run-1")[1])
    gap_id, plain_id = env["ids"]["SNR-Net"], env["ids"]["ScheduleNet"]
    gap = GiveUp(cls="research_gap:code", reason="command: src; missing: the code tree",
                 at="2026-10-14T01:21:32Z", missing="the saved code tree", by="agent")
    write_json(ModelRecord(id=gap_id, name="SNR-Net", zoo="discovered-pytorch", family="f-snr",
                           source_case="reimplement", give_up=gap),
               env["data"].model(gap_id))
    spec = env["tmp"] / "batch.json"
    spec.write_text(json.dumps({"parent": None, "batches": [{"label": "r3-2a-gap", "families": [
        {"ids": [gap_id], "family": "f-snr"}, {"ids": [plain_id]}]}]}))
    assert research.launch(spec, data=env["data"]) == ["run-1"]
    [text] = briefs
    assert f"- {gap_id} `research_gap:code`: the saved code tree" in text
    assert text.count("Open 2b give-ups to answer") == 1


def test_second_search_returns_the_other_labs_answer(env: dict, draft: dict,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    drafts = env["tmp"] / "drafts"
    drafts.mkdir()
    (drafts / "f-snr.json").write_text(json.dumps(draft))
    monkeypatch.setenv("MENAGERIE_DRAFTS", str(drafts))
    roles = []

    def fake(kind: str, role: str, brief: Path, **kw: Any) -> str:
        roles.append(role)
        assert "Learning to See in the Dark" in brief.read_text()
        brief.with_name("BRIEF.result.md").write_text(
            'Done.\n```json\n{"found": ["https://github.com/a/b"], "queries": ["q"]}\n```\n')
        return "run-ss"

    monkeypatch.setattr(drive, "launch", fake)
    got = research.second_search("f-snr", data=env["data"])
    assert (got["found"], got["status"], roles) == (["https://github.com/a/b"], "done",
                                                    ["model-implementation"])
    monkeypatch.setattr(drive, "launch", lambda *a, **k: "run-slow")
    monkeypatch.setattr(research, "SECOND_SEARCH_WAIT_S", 0)
    assert research.second_search("f-snr", data=env["data"])["status"] == "timeout"
    with pytest.raises(FileNotFoundError, match="f-none"):
        research.second_search("f-none", data=env["data"])


def test_the_2a_prompt_is_short_and_states_the_rules() -> None:
    text = research.PROMPT.read_text(encoding="utf-8")
    assert len(text.splitlines()) < 120
    for rule in ("A URL without retained readable content does not count",
                 "Mark `not_found` with what you searched", "`code_order`, first to last",
                 "`weights`", "web-search", "{PIPE}", "Prompt version 2", "`bib: pending`",
                 "the one its saved code imports"):
        assert rule in text, rule


def test_two_trees_sharing_one_license_text_each_keep_their_own(env: dict, draft: dict) -> None:
    from menagerie.pipeline import packet as packets

    # Stock Apache-2.0 (blob c71d239d) is saved by OpenHGNN, PennyLane and PointMamba alike.
    other = "https://github.com/org/other"
    tree = _put(env, store_mod.pack_snapshot({"LICENSE": b"MIT License\n"}, {"url": other}),
                url=other, mime=store_mod.SNAPSHOT_MIME)
    _put(env, b"MIT License\n", url=f"{other}/LICENSE", derived_from=tree)
    body = json.loads(json.dumps(draft))
    body["sources"].append({"sid": "s4", "role": "code", "url": other, "rev": "d" * 40,
                            "blob": tree, "framework": "pytorch", "spdx": "MIT"})
    packet = _write(env, "f-snr", body)
    licenses = {s.of: (s.sid, s.url) for s in packet.sources if s.role == "license"}
    assert {of: url for of, (_, url) in licenses.items()} == {
        "s2": f"{SNR_REPO}/LICENSE", "s4": f"{other}/LICENSE"}
    assert len({sid for sid, _ in licenses.values()}) == 2
    assert packets.relicense(["f-snr"], store=env["store"], data=env["data"]) == []
    # A packet written before the fix names the wrong tree's copy; the verb corrects it.
    path = env["data"].packet("f-snr")
    raw = json.loads(path.read_text())
    wrong = next(s for s in raw["sources"] if s["role"] == "license" and s["of"] == "s2")
    wrong["url"] = f"{other}/LICENSE"
    path.write_text(json.dumps(raw))
    assert packets.relicense(None, store=env["store"], data=env["data"], write=False) == [
        f"f-snr: {wrong['sid']}"]
    fixed = packets.relicense(["f-snr"], store=env["store"], data=env["data"])
    packet = read_json(Packet, path)
    assert fixed == [f"f-snr: {wrong['sid']}"] and packet.amendments[-1].by == "packet-licenses"
    assert next(s.url for s in packet.sources if s.sid == wrong["sid"]) == f"{SNR_REPO}/LICENSE"
    assert packets.relicense(["f-snr"], store=env["store"], data=env["data"]) == []


def test_bib_fill_completes_a_pending_packet_and_by_names_what_ran(
        env: dict, draft: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import packet as packets

    log = env["tmp"] / "work" / "lanes" / "run-2a" / "events.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text(json.dumps({"verb": "launch", "role": "model-implementation",
                               "harness": "codex", "model": "m-codex"}) + "\n")
    monkeypatch.setenv("FLEET_RUN_ID", "run-2a")
    monkeypatch.setattr(fetch, "bib_records", lambda ref, **kw: fetch.BibResult(
        (), (f"bib {ref}: s2 http_429",), {}))
    pending = {**draft, "search": {**draft["search"], "note": "bib: pending"}}
    packet = _write(env, "f-snr", pending)
    assert packet.by == "2a@prompt-v2 model-implementation (codex m-codex)"
    assert packet.search.note == "bib: pending; bib arXiv:2206.00001: s2 http_429"
    sha = _put(env, b'{"paperId": "p1"}', mime="application/json")
    sid = f"bib-s2-{sha[:8]}"
    monkeypatch.setattr(fetch, "bib_records", lambda ref, **kw: fetch.BibResult(
        (Source(sid=sid, role="bib", api="s2", blob=sha),), (),
        {"s2": {"paperId": "p1", "citationCount": 3}}))
    filled = packets.bib_fill("f-snr", store=env["store"], data=env["data"])
    assert (filled.citation.bib, filled.citation.s2) == ((sid,), "p1")
    assert filled.search.note == "bib arXiv:2206.00001: s2 http_429"
    assert (filled.amendments[-1].by, filled.amendments[-1].evidence) == ("bib-fill", (sid,))
    assert read_json(Packet, env["data"].packet("f-snr")) == filled
    with pytest.raises(RecordError, match="has no 'bib: pending'"):
        packets.bib_fill("f-snr", store=env["store"], data=env["data"])
