"""The prose audit on the real FPN entry and packet: the deterministic pass, pending entries, and
the launch, wait and write cycle (the lane stood in for)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from menagerie.pipeline import DataLayout, describe, review
from menagerie.pipeline import store as store_mod

DATA = Path(__file__).parents[2] / "menagerie" / "data"
IDS = ("m9430", "m1573")


def _entry(sid: str) -> dict:
    return json.loads((DATA / "atlas_prose" / f"{sid}.json").read_text())["entries"][0]


def _packet() -> dict:
    return json.loads((DATA / "research" / "fp" / "f-fpn-segmentation.json").read_text())


def test_mechanical_settles_citation_version_and_quotes():
    entry = _entry("m9430")
    row = {"name": "smp_FPN_resnext101_32x16d", "family": "fpn-segmentation"}
    result = describe.mechanical(entry, _packet(), row, ["merge policy: add or cat"])
    hold = set(result["facts"]["hold"])
    assert {"citation.authors.1", "dates.first_public_date",
            "implementation.library_recipe.version"} <= hold  # "Dollar" holds for "Dollár"
    assert "citation.title" in result["facts"]["unsettled"]  # the prose never names the title
    assert result["quotes"] == {"hold": ["add"], "unsettled": []}
    assert result["evidence_ids"]["unresolved"] == entry["evidence_ids"]


def test_a_changed_fact_is_left_to_the_lane():
    packet = _packet()
    packet["citation"]["first_public"] = "2017-04-19"
    result = describe.mechanical(_entry("m9430"), packet, {}, [])
    assert "dates.first_public_date" in result["facts"]["unsettled"]


def _data(tmp_path: Path) -> DataLayout:
    root = tmp_path / "data"
    for rel in [*(f"atlas_prose/{i}.json" for i in IDS), "research/fp/f-fpn-segmentation.json"]:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DATA / rel, root / rel)
    rows = [line for line in (DATA / "crawl_roster.jsonl").read_text().splitlines()
            if any(f'"stable_id":"{i}"' in line for i in IDS)]
    (root / "crawl_roster.jsonl").write_text("\n".join(rows) + "\n")
    return DataLayout(root)


def test_audit_launches_waits_then_writes(tmp_path, monkeypatch):
    data = _data(tmp_path)
    monkeypatch.setenv("MENAGERIE_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setattr(store_mod, "materialize",
                        lambda sha, dest, *, store: dest.write_text("Feature pyramid. add") and dest)
    briefs, state = [], {"state": "RUNNING"}
    monkeypatch.setattr(review.drive, "launch",
                        lambda kind, role, brief, **kw: briefs.append((role, brief)) or "audit-run")
    monkeypatch.setattr(review, "_fleet_run", lambda run_id: state)

    first = {r["id"]: r for r in describe.audit_prose(data=data)}
    assert first["m1573"]["verdict"] == "pending"  # LeNet-5's family has no packet yet
    assert first["m9430"]["verdict"] == "launched" and briefs[0][0] == "model-judgment"
    folder = briefs[0][1].parent
    entry = folder / "entries" / "m9430"
    assert (entry / "source" / "s3-paper.txt").read_text() == "Feature pyramid. add"
    assert json.loads((entry / "mechanical.json").read_text())["quotes"]["hold"] == ["add"]
    assert describe.audit_prose(["m9430"], data=data)[0]["verdict"] == "running"

    state["state"] = "TERMINAL"
    lost = [{"claim": "took state-of-the-art on COCO", "why": "unsupported", "evidence": "s3",
             "sid": "s3"}]
    (folder / "out" / "m9430.json").write_text(json.dumps(
        {"schema": "prose-audit/1", "id": "m9430", "verdict": "revise", "no_longer_holds": lost}))
    [row] = describe.audit_prose(["m9430"], data=data)
    assert (row["verdict"], row["findings"], row["family"]) == ("revise", lost, "f-fpn-segmentation")
    assert describe.audit_prose(["m9430"], data=data) == [row]  # written once per packet revision
    assert len((data.root / "review" / "prose-audit.jsonl").read_text().splitlines()) == 1


def test_jev_drive_runs_jev_once_labels_by_lane_and_scores(tmp_path, monkeypatch):
    data = DataLayout(tmp_path / "data")
    for family in ("f-fpn-segmentation", "f-gcnconv-net"):
        src = DATA / "research" / family[2:4] / f"{family}.json"
        dest = data.packet(family)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
    (data.root / "review").mkdir(parents=True)
    shutil.copyfile(DATA / "review" / "jev-profile.json", data.root / "review" / "jev-profile.json")
    monkeypatch.setenv("MENAGERIE_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setattr(describe.roster, "read", lambda data: [])
    def materialize(sha, dest, *, store):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("Abstract. We present the model.")
        return dest

    runs = {}

    def start(role, brief, ids, label, keys=None):
        (brief.parent / "lane.json").write_text(json.dumps({"run": "lab"}))
        runs["lab"] = {"state": "RUNNING"}
        return "lab"

    monkeypatch.setattr(store_mod, "materialize", materialize)
    monkeypatch.setattr(review, "_start", start)
    monkeypatch.setattr(review, "_fleet_run", lambda run: runs[run])
    calls = []

    def triage(argv, **kw):
        calls.append(argv)
        states = [json.loads(line) for line in Path(argv[argv.index("--in") + 1]).open()]
        with open(argv[argv.index("--out") + 1], "w") as out:
            for row in states:
                out.write(json.dumps({"id": row["id"], "status": "ok", "answers": {
                    "identity": {"choice": "same"}, "quote": {"noul": 0.9},
                    "members": {"noul": 0.2 if row["id"] == "f-gcnconv-net" else 0.8}}}) + "\n")
        return describe.subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(describe.subprocess, "run", triage)
    first = describe.jev_drive(2, data=data)
    assert first["state"] == "launched" and first["jev"]["state"] == "done"
    folder = review.work_root() / "jev-drive"
    folder = next(folder.iterdir())
    state = json.loads((folder / "states.jsonl").read_text().splitlines()[0])["state"]
    assert set(state) == {"name", "members", "title", "abstract", "identity_quote"}
    assert describe.jev_drive(2, data=data)["state"] == "running" and len(calls) == 1
    runs["lab"]["state"] = "TERMINAL"
    (folder / "labels.jsonl").write_text("".join(json.dumps(
        {"id": f, "identity": "same", "quote": True, "members": True}) + "\n"
        for f in ("f-fpn-segmentation", "f-gcnconv-net")))
    result = describe.jev_drive(2, data=data)
    assert (result["packets"], result["agree"], result["passed"]) == (2, 1, False)
    assert result["disagreements"][0]["questions"] == ["members"]
    describe.jev_drive(2, data=data)
    assert len((data.root / "review" / "jev-drive.jsonl").read_text().splitlines()) == 1


def test_a_blocked_jev_run_is_kept_and_retried_into_a_new_journal(tmp_path, monkeypatch):
    (tmp_path / "states.jsonl").write_text('{"id": "f-x", "state": {}}\n')
    codes = [69, 0]

    def triage(argv, **kw):
        Path(argv[argv.index("--out") + 1]).write_text("")
        return describe.subprocess.CompletedProcess(argv, codes.pop(0), "", "jev: no key")

    monkeypatch.setattr(describe.subprocess, "run", triage)
    assert describe._run_jev(tmp_path)["state"] == "unavailable"
    assert describe._run_jev(tmp_path) == {"state": "done", "out": "jev-results-2.jsonl"}
    assert describe._run_jev(tmp_path)["out"] == "jev-results-2.jsonl" and not codes
    assert (tmp_path / "jev-results-1.jsonl").is_file()


def _retry_data(tmp_path: Path, monkeypatch) -> tuple[DataLayout, dict]:
    """Packets for the jev retry: harvester ones with a bib abstract or none, 2a ones with saved
    text, one from the rung 2 drive; a scored rung 2 drive in the work root; a stand-in store."""

    data = DataLayout(tmp_path / "data")
    blobs = {"oa-hit": json.dumps({"results": [
                 {"title": "Other Work", "abstract_inverted_index": {"no": [0]}},
                 {"title": "The H1 Model", "abstract_inverted_index": {"We": [0], "H1.": [2],
                                                                       "present": [1]}}]}),
             "oa-miss": json.dumps({"title": "The H2 Model", "abstract_inverted_index": None}),
             "s2-hit": json.dumps({"title": "the h3 model", "abstract": "H3 is new."}),
             "txt": "Abstract. We present the model."}
    packets = [("f-h1", "harvester@menagerie", [("bib", "openalex", "oa-hit")], "The H1 Model"),
               ("f-h2", "harvester@menagerie", [("bib", "openalex", "oa-miss")], "The H2 Model"),
               ("f-h3", "harvester@menagerie", [("bib", "s2", "s2-hit")], "The H3 Model"),
               ("f-a1", "2a@run-1", [("paper", None, "txt")], "A1"),
               ("f-a2", "2a@run-2", [("page", None, "txt")], "A2"),
               ("f-old", "2a@run-0", [("paper", None, "txt")], "Old")]
    for family, by, sources, title in packets:
        srcs = [{"sid": f"s{i}", "role": role, **({"api": api, "blob": sha} if role == "bib"
                                                  else {"text": sha})}
                for i, (role, api, sha) in enumerate(sources)]
        dest = data.packet(family)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({"family": family, "by": by, "sources": srcs,
                                    "citation": {"title": title}, "members": []}))
    monkeypatch.setenv("MENAGERIE_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))

    def materialize(sha, dest, *, store):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if sha in blobs:
            dest.write_text(blobs[sha])
        else:  # a blob the code put, such as the frozen states
            shutil.copyfile(store.blob(sha), dest)
        return dest

    monkeypatch.setattr(store_mod, "materialize", materialize)
    monkeypatch.setattr(describe.roster, "read", lambda data: [])
    (data.root / "review").mkdir(parents=True)
    shutil.copyfile(DATA / "review" / "jev-profile.json", data.root / "review" / "jev-profile.json")
    rung2 = review.work_root() / "jev-drive" / describe.RUNG2_BATCH
    rung2.mkdir(parents=True)
    scores = {"f-old": (0.6, False), "f-x": (0.7, False), "f-y": (0.9, True), "f-z": (0.2, False)}
    (rung2 / "ids.json").write_text(json.dumps(list(scores)))
    (rung2 / "jev.json").write_text(json.dumps({"state": "done", "out": "jev-results-1.jsonl"}))
    (rung2 / "jev-results-1.jsonl").write_text("".join(json.dumps(
        {"id": pid, "status": "ok", "answers": {"identity": {"choice": "same"},
                                               "quote": {"noul": s}, "members": {"noul": 0.9}}})
        + "\n" for pid, (s, _) in scores.items()))
    (rung2 / "labels.jsonl").write_text("".join(json.dumps(
        {"id": pid, "identity": "same", "quote": q, "members": True}) + "\n"
        for pid, (_, q) in scores.items()))
    return data, blobs


def test_jev_freeze_draws_half_and_half_and_fixes_the_cutoff_first(tmp_path, monkeypatch):
    data, _ = _retry_data(tmp_path, monkeypatch)
    frozen = describe.jev_freeze(4, data=data)
    rows = {r["id"]: r for r in frozen["packets"]}
    assert set(rows) == {"f-h1", "f-h3", "f-a1", "f-a2"}  # f-h2 holds no abstract, f-old was drawn
    assert (rows["f-h1"]["how"], rows["f-h3"]["how"], rows["f-a2"]["how"]) == (
        "openalex", "s2", "text")
    assert [r["hash"] for r in frozen["packets"]] == sorted(r["hash"] for r in frozen["packets"])
    cutoff = frozen["cutoff"]
    assert cutoff["accept_above"] == 0.7 and set(cutoff["over_accepted"]) == {"f-old", "f-x"}
    assert cutoff["rung2_rescored"]["agree"] == 4 and frozen["bar"] == 4
    assert frozen["states"]["count"] == 4 and frozen["profile_rev"] == "2026-10-05.1"
    path = data.root / "review" / describe.JEV_RETRY
    before = path.read_text()
    assert describe.jev_freeze(4, data=data) == frozen and path.read_text() == before


def test_jev_retry_drives_the_frozen_packets_at_the_frozen_cutoff(tmp_path, monkeypatch):
    data, _ = _retry_data(tmp_path, monkeypatch)
    frozen = describe.jev_freeze(4, data=data)
    live = json.loads(data.packet("f-a1").read_text())  # the crawl moves a packet after the freeze
    data.packet("f-a1").write_text(json.dumps({**live, "citation": {"title": "Moved On"}}))
    runs = {}

    def start(role, brief, ids, label, keys=None):
        (brief.parent / "lane.json").write_text(json.dumps({"run": "lab"}))
        runs["lab"] = {"state": "TERMINAL"}
        return "lab"

    def triage(argv, **kw):
        states = [json.loads(line) for line in Path(argv[argv.index("--in") + 1]).open()]
        with open(argv[argv.index("--out") + 1], "w") as out:
            for row in states:
                out.write(json.dumps({"id": row["id"], "status": "ok", "answers": {
                    "identity": {"choice": "same"}, "members": {"noul": 0.9},
                    "quote": {"noul": 0.75 if row["id"] == "f-h1" else 0.7}}}) + "\n")
        return describe.subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(review, "_start", start)
    monkeypatch.setattr(review, "_fleet_run", lambda run: runs[run])
    monkeypatch.setattr(describe.subprocess, "run", triage)
    assert describe.jev_drive(data=data, retry=True)["state"] == "launched"
    folder = review.work_root() / "jev-drive" / f"retry-{frozen['states']['sha256'][:8]}"
    states = {json.loads(line)["id"]: json.loads(line)["state"]
              for line in (folder / "states.jsonl").read_text().splitlines()}
    assert states["f-h1"]["abstract"] == "We present H1."
    assert states["f-h3"]["abstract"] == "H3 is new."
    assert states["f-a1"]["title"] == "A1"  # the frozen state, not the moved packet
    (folder / "labels.jsonl").write_text("".join(json.dumps(
        {"id": f, "identity": "same", "quote": f == "f-h1", "members": True}) + "\n"
        for f in states))
    result = describe.jev_drive(data=data, retry=True)
    assert (result["retry"], result["quote_above"], result["agree"]) == (True, 0.7, 4)
    profile = json.loads((data.root / "review" / "jev-profile.json").read_text())
    (data.root / "review" / "jev-profile.json").write_text(json.dumps({**profile, "rev": "x"}))
    with pytest.raises(review.ReviewError, match="no retune"):
        describe.jev_drive(data=data, retry=True)


def test_jev_retry_refuses_without_a_frozen_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    with pytest.raises(review.ReviewError, match="jev freeze"):
        describe.jev_drive(data=DataLayout(tmp_path / "data"), retry=True)
