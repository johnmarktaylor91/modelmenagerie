"""Record contracts: real fixtures round-trip, real data files parse, bad records are refused."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from menagerie.pipeline import (
    DataLayout,
    StoreLayout,
    append_jsonl,
    cli,
    read_json,
    read_jsonl,
    recipe_revision,
    tree_sha256,
    write_json,
)
from menagerie.pipeline.records import (
    CLASS_PATTERN,
    EVENT_VERBS,
    CensusRow,
    CitationLine,
    DescribeRecord,
    Dossier,
    GiveUp,
    LedgerRow,
    LegacyRow,
    ModelRecord,
    Packet,
    RecordError,
    ResolverHint,
    RosterRow,
    RunRequest,
    RunResult,
    StoreIndexLine,
    SweepLine,
    dumps,
    from_dict,
    parse_ledger_line,
    to_dict,
)
from menagerie.pipeline.stubs import NOT_BUILT

FIXTURES = Path(__file__).parent / "fixtures"
DATA = Path(__file__).parents[2] / "menagerie" / "data"
JSON_FIXTURES = {
    "f-resnet.packet.json": Packet, "f-beit.packet.json": Packet,
    "m8839.model.json": ModelRecord, "m3631.model.json": ModelRecord,
    "f-resnet.describe.json": DescribeRecord, "run_request.json": RunRequest,
    "run_result.json": RunResult,
}
JSONL_FIXTURES = {
    "roster.jsonl": RosterRow, "sweeps.jsonl": SweepLine, "cite.jsonl": CitationLine,
    "store_index.jsonl": StoreIndexLine, "ledger.jsonl": parse_ledger_line,
    "census.jsonl": CensusRow, "dossier.jsonl": Dossier, "hint.jsonl": ResolverHint,
}


def _manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


@pytest.mark.parametrize("name", sorted(JSON_FIXTURES))
def test_json_fixture_round_trips(name: str, tmp_path: Path) -> None:
    text = (FIXTURES / name).read_text(encoding="utf-8")
    record = read_json(JSON_FIXTURES[name], FIXTURES / name)
    assert to_dict(record) == json.loads(text)
    assert dumps(record) + "\n" == text
    write_json(record, tmp_path / name)
    assert (tmp_path / name).read_text(encoding="utf-8") == text


@pytest.mark.parametrize("name", sorted(JSONL_FIXTURES))
def test_jsonl_fixture_round_trips(name: str, tmp_path: Path) -> None:
    text = (FIXTURES / name).read_text(encoding="utf-8")
    records = read_jsonl(JSONL_FIXTURES[name], FIXTURES / name)
    assert [to_dict(r) for r in records] == [json.loads(line) for line in text.splitlines()]
    append_jsonl(records, tmp_path / name)
    assert (tmp_path / name).read_text(encoding="utf-8") == text


def test_fixture_sizes_meet_design_targets() -> None:
    assert all(len(p.read_bytes()) < 4096 for p in FIXTURES.glob("*.packet.json"))
    assert all(len(p.read_bytes()) < 2048 for p in FIXTURES.glob("*.model.json"))


def test_routing_manifest_row_is_the_legacy_shard_row() -> None:
    (row,) = _manifest_rows(FIXTURES / "routing_manifest_row.tsv")
    legacy = LegacyRow.from_manifest(row)
    assert (legacy.id, legacy.status, legacy.n_ops) == ("m8839", "validated", 85)
    assert legacy.metadata_ok is True
    assert legacy.graph_hash == row["graph_shape_hash"]
    assert parse_ledger_line(json.loads(dumps(legacy))) == legacy


def test_every_routing_manifest_row_converts() -> None:
    rows = _manifest_rows(DATA / "routing_manifest.tsv")
    legacy = [LegacyRow.from_manifest(row) for row in rows]
    assert len(legacy) == 11_641
    assert sum(r.status == "validated" for r in legacy) == 8_359


def test_every_roster_row_keeps_its_fields() -> None:
    lines = (DATA / "crawl_roster.jsonl").read_text(encoding="utf-8").splitlines()
    schema = {"schema": RosterRow.SCHEMA}
    rows = [from_dict(RosterRow, {**schema, **json.loads(line)}) for line in lines]
    assert len(rows) == 28_482
    # 2a packet findings move rows out of `active` (rung 1); nothing else changes a state.
    assert all(r.state in ("active", "not_architecture") or re.fullmatch(r"duplicate_of:m\d+", r.state)
               for r in rows)


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("cls", "name", "change", "message"),
    [
        (ModelRecord, "m8839.model.json", {"surprise": 1}, "unknown fields"),
        (ModelRecord, "m8839.model.json", {"schema": "model/2"}, "schema"),
        (ModelRecord, "m8839.model.json", {"id": "8839"}, "a match for"),
        (ModelRecord, "m8839.model.json", {"source_case": "scraped"}, "one of"),
        (Packet, "f-resnet.packet.json", {"code_order": ["s9"]}, "unknown sources"),
        (Packet, "f-resnet.packet.json", {"code_order": ["s1"]}, "non-code source"),
        (Packet, "f-resnet.packet.json",
         {"availability": {"paper": "found", "code": "not_found", "license": "found",
                           "description": "sufficient"}}, "no places searched"),
        (StoreIndexLine, None, None, "missing field"),
    ],
)
def test_bad_records_are_refused(cls, name, change, message) -> None:
    raw = {**_fixture(name), **change} if name else {"schema": "store-index/1", "bytes": 1}
    with pytest.raises(RecordError, match=message):
        from_dict(cls, raw)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"class": None}, "needs a class"),
        ({"class": "replay_mismatch"}, "match"),
        ({"outcome": "pass"}, "has no class"),
        ({"capture": "structure_only"}, "never validated"),
        ({"graph_hash": "not-a-sha"}, "graph_hash"),
    ],
)
def test_bad_ledger_rows_are_refused(change: dict, message: str) -> None:
    raw = {k: v for k, v in {**_fixture_line("ledger.jsonl", 0), **change}.items() if v is not None}
    with pytest.raises(RecordError, match=message):
        from_dict(LedgerRow, raw)


def _fixture_line(name: str, index: int) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8").splitlines()[index])


def test_run_request_mode_and_trace_agree() -> None:
    raw = _fixture("run_request.json")
    with pytest.raises(RecordError, match="render mode"):
        from_dict(RunRequest, {**raw, "mode": "render"})
    trace_sha = "0" * 64
    assert from_dict(RunRequest, {**raw, "mode": "render", "trace": trace_sha}).trace == trace_sha


LOCK = b"""version = 1
[[package]]
name = "env"
version = "0"
source = { virtual = "." }
dependencies = [{ name = "torch" }]

[[package]]
name = "torch"
version = "2.6.0+cpu"
source = { registry = "https://download.pytorch.org/whl/cpu" }
"""


def test_recipe_revision_tracks_what_runs() -> None:
    recipe = _fixture("m8839.model.json")["recipe"]
    lock = LOCK
    base = recipe_revision(recipe, None, lock)
    assert recipe_revision({**recipe, "modes": {"eval": "fail:forward_error"}}, None, lock) == base
    assert recipe_revision({**recipe, "sources": []}, None, lock) == base
    assert recipe_revision({**recipe, "kwargs": dict(reversed(list(recipe["kwargs"].items())))},
                           None, lock) == base
    assert recipe_revision({**recipe, "args": ["resnet34"]}, None, lock) != base
    assert recipe_revision(recipe, b"adapter", lock) != base
    assert recipe_revision(recipe, None, lock.replace(b"2.6.0+cpu", b"2.6.1+cpu")) != base
    assert recipe_revision({**recipe, "extras": ["torch_geometric==2.6.1"]}, None, lock) != base
    assert recipe_revision({**recipe, "vendor": "a" * 64}, None, lock) != base


def test_rung1_fields_round_trip() -> None:
    model = _fixture("m8839.model.json")
    recipe = {**model["recipe"], "kind": "adapter", "adapter": "f-resnet.py",
              "extras": ["torch_geometric==2.6.1"], "vendor": "b" * 64}
    record = from_dict(ModelRecord, {**model, "recipe": recipe})
    assert (record.recipe.extras, record.recipe.vendor) == (("torch_geometric==2.6.1",), "b" * 64)
    assert from_dict(ModelRecord, json.loads(dumps(record))) == record
    adapter_src = "VARIANTS = {}\n"
    request = from_dict(RunRequest, {**_fixture("run_request.json"), "adapter_src": adapter_src})
    assert json.loads(dumps(request))["adapter_src"] == adapter_src
    assert from_dict(RunRequest, json.loads(dumps(request))) == request
    packet = _fixture("f-beit.packet.json")
    page = {"sid": "s9", "role": "page", "url": "https://github.com/microsoft/unilm",
            "blob": "c" * 64}
    parsed = from_dict(Packet, {**packet, "sources": [*packet["sources"], page]})
    assert to_dict(parsed)["sources"][-1] == page
    with pytest.raises(RecordError, match="needs"):
        bare = {"sid": "s9", "role": "page"}
        from_dict(Packet, {**packet, "sources": [*packet["sources"], bare]})
    with pytest.raises(RecordError):
        from_dict(ModelRecord, {**model, "recipe": {**recipe, "vendor": "not-a-sha"}})


def test_rung2_contracts_round_trip_and_refuse() -> None:
    row = _fixture_line("ledger.jsonl", 1)
    review = {k: v for k, v in row.items() if k not in ("mode", "capture", "class")} | {
        "stage": "review", "outcome": "pass", "verdict": "minor", "reviewer": "model-design-review"}
    parsed = from_dict(LedgerRow, review)
    assert parsed.reviewer == "model-design-review"
    assert from_dict(LedgerRow, json.loads(dumps(parsed))) == parsed
    with pytest.raises(RecordError, match="only a review row"):
        from_dict(LedgerRow, {**row, "reviewer": "model-default"})
    for cls in ("standin:exec", "standin:compact", "standin:kwargs", "needs_adapter",
                "oversized:beyond_allocation", "env_unresolved", "unpublishable:fidelity"):
        assert from_dict(LedgerRow, {**_fixture_line("ledger.jsonl", 0), "class": cls}).cls == cls
    for cls in ("standin", "standin:other", "unpublishable:style"):
        with pytest.raises(RecordError, match="needs one of"):
            from_dict(LedgerRow, {**_fixture_line("ledger.jsonl", 0), "class": cls})
    model = _fixture("m8839.model.json")
    for placement in ("worker", "single-worker", "axon-cpu", "axon-gpu", "axon-reviewed",
                      "gpu-window", "structure-only"):
        recipe = {**model["recipe"], "placement": placement}
        assert from_dict(ModelRecord, {**model, "recipe": recipe}).recipe.placement == placement
    with pytest.raises(RecordError, match="placement: 'cluster' is not"):
        from_dict(ModelRecord, {**model, "recipe": {**model["recipe"], "placement": "cluster"}})
    with pytest.raises(RecordError, match="env 'torch-nightly'"):
        from_dict(ModelRecord, {**model, "recipe": {**model["recipe"], "env": "torch-nightly"}})
    request = from_dict(RunRequest, {**_fixture("run_request.json"), "mode": "gpu"})
    assert request.mode == "gpu" and request.timeout_s == request.caps.timeout_s
    assert "timeout_s" not in to_dict(request)


def test_rung3_contracts_round_trip_and_refuse() -> None:
    line = _fixture_line("ledger.jsonl", 0)
    for cls in ("describe:ungrounded", "describe:no_text", "no_paper:no_text_names_one",
                "standin:no_published_config", "june_code:flagged", "needs_env:mmlab"):
        assert from_dict(LedgerRow, {**line, "class": cls}).cls == cls
    for cls in ("describe", "describe:vague", "june_code", "june_code:ok"):
        with pytest.raises(RecordError, match="needs one of"):
            from_dict(LedgerRow, {**line, "class": cls})
    for cls in ("no_paper", "needs_env"):
        with pytest.raises(RecordError, match="needs a detail"):
            from_dict(LedgerRow, {**line, "class": cls})
    describe = _fixture("f-resnet.describe.json")
    assert from_dict(DescribeRecord, describe).grounding_state == "grounded"
    with pytest.raises(RecordError, match="did not read"):
        from_dict(DescribeRecord, {**describe, "inputs": ["s2"]})
    with pytest.raises(RecordError, match="grounding_state"):
        from_dict(DescribeRecord, {**describe, "grounding_state": "maybe"})
    bare = {k: v for k, v in describe.items()
            if k not in ("inputs", "prompt", "grounding_state", "grounding")}
    assert from_dict(DescribeRecord, bare).grounding_state == "unchecked"
    top, archived = (json.loads(s) for s in (FIXTURES / "census.jsonl").read_text().splitlines()[:2])
    for change, message in (({"eligible": False, "why": "x"}, "four tests"),
                            ({"why": "archived"}, "exactly when"),
                            ({"size_kb": 300 * 1024}, "four tests")):
        with pytest.raises(RecordError, match=message):
            from_dict(CensusRow, {**top, **change})
    with pytest.raises(RecordError, match="only an eligible"):
        from_dict(CensusRow, {**archived, "draw": "control"})


def test_vocabularies_load_closed_and_distinct() -> None:
    from menagerie.catalog import macro_domain
    from menagerie.pipeline.records import VOCAB, vocab

    assert {p.stem for p in VOCAB.glob("*.json")} >= {"domain", "output_type"}
    for path in VOCAB.glob("*.json"):
        assert vocab(path.stem)
    roster = (DATA / "crawl_roster.jsonl").read_text(encoding="utf-8").splitlines()
    live = {macro_domain(r.get("domain") or "") for r in map(json.loads, roster[:2000])}
    assert live <= vocab("domain") and len(vocab("domain")) == 23
    with pytest.raises(RecordError, match="no vocabulary"):
        vocab("not_a_field")


def test_committed_recipes_name_known_environments() -> None:
    from menagerie.pipeline.records import known_envs

    assert "torch-main" in known_envs()
    records = [read_json(ModelRecord, p) for p in (DATA / "models").rglob("*.json")]
    assert records and all(r.recipe is None or r.recipe.env in known_envs() for r in records)


def test_rung2_rows_are_exact_roster_rows() -> None:
    """``data/catalog/r2-rows.tsv``: rung 2's frozen rows (100 counted), each a roster row by id."""

    with (DATA / "catalog" / "r2-rows.tsv").open(encoding="utf-8", newline="") as handle:
        frozen = list(csv.DictReader(handle, delimiter="\t"))
    lines = (DATA / "crawl_roster.jsonl").read_text(encoding="utf-8").splitlines()
    roster = {r["stable_id"]: r for r in map(json.loads, lines) if r.get("stable_id")}
    assert sum(r["counted"] == "1" for r in frozen) == 100
    assert len({r["id"] for r in frozen}) == len(frozen)
    assert all((roster[r["id"]]["name"], roster[r["id"]]["zoo"], roster[r["id"]]["variant"])
               == (r["name"], r["zoo"], r["variant"]) for r in frozen)


APPENDER = """
import sys
from pathlib import Path
from menagerie.pipeline import append_jsonl
from menagerie.pipeline.records import CitationLine
worker = int(sys.argv[2])
for i in range(200):
    append_jsonl([CitationLine(family="f-resnet", at=f"2026-10-04T00:00:{worker:02d}Z", s2=i,
                               openalex=i * 1000)], Path(sys.argv[1]))
"""


def test_append_jsonl_keeps_lines_whole_across_processes(tmp_path: Path) -> None:
    path = tmp_path / "cite.jsonl"
    procs = [subprocess.Popen([sys.executable, "-c", APPENDER, str(path), str(w)],
                              cwd=Path(__file__).parents[2]) for w in range(4)]
    assert [proc.wait(timeout=120) for proc in procs] == [0, 0, 0, 0]
    lines = read_jsonl(CitationLine, path)
    assert sorted((line.at, line.s2) for line in lines) == sorted(
        (f"2026-10-04T00:00:{w:02d}Z", i) for w in range(4) for i in range(200))


def test_layouts_read_their_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MENAGERIE_DATA", str(tmp_path))
    data = DataLayout.from_env()
    assert data.model("m8839") == tmp_path / "models" / "88" / "m8839.json"
    assert data.packet("f-resnet") == tmp_path / "research" / "re" / "f-resnet.json"
    assert data.ledger("2026-10") == tmp_path / "ledger" / "2026-10.jsonl"
    monkeypatch.delenv("MENAGERIE_DATA")
    assert DataLayout.from_env().routing_manifest.is_file()
    monkeypatch.delenv("MENAGERIE_STORE", raising=False)
    with pytest.raises(RecordError, match="MENAGERIE_STORE"):
        StoreLayout.from_env()
    monkeypatch.setenv("MENAGERIE_STORE", str(tmp_path))
    sha = "ab" + "0" * 62
    assert StoreLayout.from_env().blob(sha) == tmp_path / "sha256" / "ab" / sha


def test_tree_sha256_is_content_only(tmp_path: Path) -> None:
    first, second = tmp_path / "a.tlspec", tmp_path / "b.tlspec"
    for root, order in ((first, ("x", "sub/y")), (second, ("sub/y", "x"))):
        for rel in order:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes(rel.encode())
    assert tree_sha256(first) == tree_sha256(second)
    assert tree_sha256(first / "x") == hashlib.sha256(b"x").hexdigest()
    (second / "x").write_bytes(b"changed")
    assert tree_sha256(first) != tree_sha256(second)


OWNED = {
    "store": ["put", "get", "extract_text", "mirror", "restore"],
    "fetch": ["fetch_paper", "fetch_license"],
    "ledger": ["append", "read_all", "state", "publishable", "import_legacy", "status_counts"],
    "roster": ["read", "repair"],
    "envs": ["lock_id", "build", "canary", "inventory"],
    "runner": ["default_caps", "request_for", "attempt", "run", "serve"],
    "worker": ["run_request", "main"],
    "render": ["renderer_rev", "variants_for", "default_variant", "render_set"],
    "inputs": ["presets", "asset_path", "make_input"],
    "harvest": ["harvest", "construct"],
    "publish": ["export_site"],
    # Rung 1 stubs (BUILD-PLAN r1 section 1): their lanes fill them in.
    "research": ["seed_family", "packet_write", "launch", "second_search"],
    "implement": ["src", "try_", "giveup", "done", "launch", "bridge"],
    "review": ["pick_reviewer", "calibrate", "launch", "record"],
    "describe": ["audit_prose"],
    "drive": ["launch", "event", "next_steps", "lanes", "measures", "collect"],
}
OWNED["fetch"] += ["fetch_code", "fetch_hf", "fetch_page", "fetch_bib"]
OWNED["store"] += ["drain", "materialize", "visible_text"]
# Rung 2 stubs (BUILD-PLAN r2 section 1, item 3): their lanes fill them in.
OWNED["stage3"] = ["queue", "serve", "collect"]
OWNED["harvest_tiers"] = ["catalog_plan", "harvest_tier"]
OWNED["cluster"] = ["axon_batch", "submit", "run", "collect"]
OWNED["hash_harness"] = ["harness"]
OWNED["describe"] += ["jev_drive"]
OWNED["drive"] += ["checkin", "type_render"]
OWNED["standin"] = ["sweep"]
# Rung 3 stubs (BUILD-PLAN r3 section 1, item 3): their lanes fill them in.
OWNED |= {"tick": ["tick"], "runner_job": ["ensure"], "find_paper": ["find", "harvest", "apply"],
          "describe_jev": ["drive"], "describe_write": ["write", "check"],
          "census": ["run", "draw"], "residual": ["plan"], "repin": ["check", "flip"],
          "dataset": ["build", "site_count"]}
OWNED["review"] += ["calibrate_all"]
# Speed-up stubs (speed-up BUILD-PLAN section 1, item 1): their lanes fill them in.
OWNED["implement"] += ["launch_batch"]
OWNED["tick_full"] = ["tripwires"]
OWNED["describe_write"] += ["seats"]
OWNED["runner_job"] += ["big"]
OWNED |= {"census_harvest": ["harvest"], "dossier": ["build"], "worklist_build": ["build"],
          "counts": ["build"], "replay_table": ["table"], "resolve": ["run"],
          "resolve_confirm": ["confirm"]}


def test_every_owned_function_exists() -> None:
    for module, names in OWNED.items():
        loaded = importlib.import_module(f"menagerie.pipeline.{module}")
        assert [n for n in names if not callable(getattr(loaded, n, None))] == [], module


def test_every_verb_parses_and_dispatches(monkeypatch: pytest.MonkeyPatch,
                                        tmp_path: Path) -> None:
    parser = cli.build_parser()
    argv = {
        "roster repair": ["roster", "repair"], "ledger import-legacy": ["ledger", "import-legacy"],
        "status": ["status"],
        "store put": ["store", "put", "f", "--mime", "m", "--license-class", "c"],
        "store get": ["store", "get", "s"], "store extract-text": ["store", "extract-text", "s"],
        "store mirror": ["store", "mirror", "--dest", "d"],
        "store restore": ["store", "restore", "s", "--source", "d"],
        "fetch paper": ["fetch", "paper", "1512.03385"], "fetch license": ["fetch", "license", "u"],
        "envs build": ["envs", "build", "torch-main"],
        "envs canary": ["envs", "canary", "torch-main"],
        "envs list": ["envs", "list"], "envs missing": ["envs", "missing"],
        "envs prune-overlays": ["envs", "prune-overlays"],
        "harvest zoo": ["harvest", "zoo", "timm", "resnet18"],
        "runner serve": ["runner", "serve", "--spool", "s"], "run": ["run", "r.json"],
        "try": ["try", "m8839"], "trace": ["trace", "m8839", "--structure-only"],
        "render": ["render", "m8839", "--trace", "t"], "worker": ["worker", "a"],
        "publish export": ["publish", "export", "out"],
        **RUNG1_VERBS, **{k: v for k, v in {**RUNG2_VERBS, **RUNG3_VERBS}.items()
                          if " --" not in k}, **SPEEDUP_VERBS,
    }
    decision = tmp_path / "paper-decision.json"
    decision.write_text(json.dumps({"no_paper": "heterogeneous_members", "evidence": ["s1"],
                                    "reason": "distinct papers per member"}), encoding="utf-8")
    argv["paper apply"] = ["paper", "apply", "f-example", str(decision)]
    assert set(argv) == set(cli.HANDLERS)
    for verb, args in argv.items():
        assert parser.parse_args(args).verb == verb
    monkeypatch.setenv("MENAGERIE_DATA", str(DATA))
    assert cli.main(["status"]) == 0  # dispatches to the data lane's ledger.status_counts


RUNG1_VERBS = {
    "fetch code": ["fetch", "code", "https://github.com/a/b", "--family", "f-b", "--rev", "v1"],
    "fetch hf": ["fetch", "hf", "zhihan1996/DNABERT-S", "--family", "f-dnabert"],
    "fetch page": ["fetch", "page", "https://example.org", "--family", "f-b"],
    "fetch bib": ["fetch", "bib", "1512.03385", "--family", "f-resnet"],
    "store drain": ["store", "drain", "--sha", "a" * 64],
    "store materialize": ["store", "materialize", "a" * 64, "dest"],
    "family seed": ["family", "seed", "m1873", "m1835"],
    "packet write": ["packet", "write", "f-nin", "draft.json"],
    "research launch": ["research", "launch", "batch.json"],
    "research second-search": ["research", "second-search", "f-nin"],
    "src": ["src", "f-nin"], "giveup": ["giveup", "m1873", "budget", "ten tries"],
    "done": ["done", "recipe.json"],
    "implement launch": ["implement", "launch", "f-nin", "--role", "model-default"],
    "bridge serve": ["bridge", "serve"],
    "review pick": ["review", "pick", "run-1"],
    "review calibrate": ["review", "calibrate", "--role", "model-default"],
    "review launch": ["review", "launch", "m1873"],
    "review record": ["review", "record", "m1873", "findings.json"],
    "prose audit": ["prose", "audit"], "next": ["next", "m8839"], "lanes": ["lanes"],
    "measures": ["measures", "run-1"], "collect": ["collect"],
}
RUNG2_VERBS = {
    "stage3 queue": ["stage3", "queue", "run-2b-1", "m1873"],
    "stage3 serve": ["stage3", "serve", "--spool", "s", "--once"],
    "stage3 collect": ["stage3", "collect"],
    "catalog plan": ["catalog", "plan", "--skip", "rows.tsv", "--dry-run"],
    "harvest tier": ["harvest", "tier", "timm", "resnet18", "--dry-run"],
    "jev drive": ["jev", "drive", "--packets", "50"],
    "axon batch": ["axon", "batch", "--kind", "cpu", "m3486", "m3813"],
    "axon submit": ["axon", "submit", "batch-dir", "--dry-run"],
    "axon run": ["axon", "run", "batch-dir"],
    "axon collect": ["axon", "collect", "batch-dir"],
    "hash harness": ["hash", "harness", "m8839"],
    "checkin": ["checkin", "run-1", "--out", "r2-checkin.md"],
    "standin sweep": ["standin", "sweep", "--apply"],
    "review calibrate --runs": ["review", "calibrate", "--role", "model-default", "--runs", "3"],
    "queue": ["queue"], "requeue": ["requeue"], "packet bib-fill": ["packet", "bib-fill", "f-nin"],
    "done --rewrite-family": ["done", "recipe.json", "--rewrite-family"],
    "packet bib-harvest": ["packet", "bib-harvest", "f-nin", "--dry-run"],
    "packet licenses": ["packet", "licenses", "f-nin", "--dry-run"],
    "packet june-notes": ["packet", "june-notes", "--dry-run"],
    "packet domains": ["packet", "domains", "domains.tsv"],
    "packet check": ["packet", "check"], "store audit-documents": ["store", "audit-documents"],
    "packet spot-check": ["packet", "spot-check", "verdicts.tsv"],
}
RUNG3_VERBS = {
    "tick": ["tick", "--mode", "collect", "--dry-run"],
    "runner ensure": ["runner", "ensure", "--dry-run"],
    "paper find": ["paper", "find", "f-nin", "--dry-run"],
    "paper harvest": ["paper", "harvest", "--limit", "20", "--dry-run"],
    "describe drive": ["describe", "drive", "--families", "50"],
    "describe write": ["describe", "write", "f-nin", "--dry-run"],
    "describe check": ["describe", "check", "f-nin"],
    "census run": ["census", "run", "--limit", "20"],
    "census draw": ["census", "draw", "--n", "500", "--control", "50"],
    "residual plan": ["residual", "plan", "--rows", "rows.tsv"],
    "implement library-adapter": ["implement", "library-adapter", "timm", "m1"],
    "implement sdist-wheel": ["implement", "sdist-wheel", "source.tar.gz", "--root", "envs-root",
                              "--python", "python", "--distribution", "example", "--version", "1"],
    "repin check": ["repin", "check", "2.35.4"],
    "repin flip": ["repin", "flip", "2.35.4", "--dry-run"],
    "release build": ["release", "build", "--dry-run"],
    "site count": ["site", "count", "dist"],
    "review calibrate --all": ["review", "calibrate", "--all", "--runs", "3"],
    "jev freeze": ["jev", "freeze", "--packets", "50"],
    "jev drive --retry": ["jev", "drive", "--retry"],
}
# The pack lane's verbs (store gate); they act on the worker's live store.
RUNG3_VERBS |= {"store offload": ["store", "offload", "tiles", "--catalog", "c", "--host", "h"],
                "store pack": ["store", "pack", "--catalog", "c", "--host", "h", "--limit", "1"],
                "store pack-views": ["store", "pack-views", "--catalog", "c", "--host", "h"]}
RUNG3_VERBS |= {"store ceiling": ["store", "ceiling", "--catalog", "c", "--host", "h", "--target-gib", "20"],
                "store fetch": ["store", "fetch", "a" * 64, "--remote", "w:store"]}
HUB_MODULES = ("cli", "fetch", "store", "research", "implement", "review", "describe", "drive",
               "stubs", "census_harvest", "dossier", "worklist_build", "counts", "replay_table",
               "resolve", "resolve_confirm", "stage3", "harvest_tiers", "cluster", "admission",
               "checkin", "packet", "fetch_repos", "bridge", "tick", "runner_job", "find_paper", "describe_jev",
               "describe_write", "census", "residual", "disk_guard", "tick_full")
# Verbs whose lane has merged: they no longer reach a stub (each lane adds its own on merge).
FILLED = {"next", "lanes", "measures", "collect",
          "fetch code", "fetch hf", "fetch page", "fetch bib", "store drain", "store materialize",
          "family seed", "packet write", "research launch", "research second-search",
          "src", "giveup", "done", "implement launch", "bridge serve",
          "review pick", "review calibrate", "review launch", "review record", "prose audit"}
# Filled verbs that would reach the network with these arguments; their lane's tests cover them.
NETWORK = {"fetch code", "fetch hf", "fetch page", "fetch bib"}
# Rung 3 verbs that read the worker, run-on or fleet-run from the hub.
NETWORK |= {"tick", "runner ensure"}
# Rung 2 verbs that run on the worker (they trace or serve attempts), never on the hub.
WORKER_VERBS = {"stage3 serve", "hash harness", "axon run"}
# Rung 3 verbs that run on the worker (they trace, build or count the site).
WORKER_VERBS |= {"repin check", "release build", "site count"}
WORKER_VERBS |= {"store offload", "store pack", "store pack-views", "store ceiling"}
NETWORK |= {"store fetch"}  # restores from the box through the hub's broker
WORKER_VERBS |= {"implement library-adapter", "implement sdist-wheel"}  # r3-residual


@pytest.mark.parametrize("verb", sorted(set(RUNG1_VERBS) - FILLED))
def test_rung1_verbs_reach_their_stubs(verb: str, monkeypatch: pytest.MonkeyPatch,
                                       tmp_path: Path) -> None:
    monkeypatch.setenv("MENAGERIE_DATA", str(tmp_path))
    monkeypatch.setenv("MENAGERIE_STORE", str(tmp_path))
    with pytest.raises(NotImplementedError, match="rung 1"):
        cli.main(RUNG1_VERBS[verb])


# Rung 2 verbs whose lane has merged (each lane adds its own on merge).
FILLED2: set[str] = {"checkin", "queue", "requeue", "packet bib-fill", "done --rewrite-family",
                     "standin sweep"}
FILLED2 |= {"stage3 queue", "stage3 serve", "stage3 collect"}  # the stage3 lane
FILLED2 |= {"catalog plan", "harvest tier"}  # the harvest lane
FILLED2 |= {"jev drive", "review calibrate --runs"}  # the review lane
FILLED2 |= {"axon batch", "axon submit", "axon run", "axon collect"}  # the giants lane
FILLED2 |= {"hash harness"}  # the hash lane
FILLED2 |= {"packet licenses"}  # the license-index fix
FILLED2 |= {"packet bib-harvest", "packet june-notes", "packet domains", "packet check",
            "store audit-documents", "packet spot-check"}  # the metadata lane


@pytest.mark.parametrize("verb", sorted(set(RUNG2_VERBS) - FILLED2))
def test_rung2_verbs_reach_their_stubs(verb: str, monkeypatch: pytest.MonkeyPatch,
                                       tmp_path: Path) -> None:
    monkeypatch.setenv("MENAGERIE_DATA", str(tmp_path))
    monkeypatch.setenv("MENAGERIE_STORE", str(tmp_path))
    with pytest.raises(NotImplementedError, match="rung 2"):
        cli.main(RUNG2_VERBS[verb])


# Rung 3 verbs whose lane has merged (each lane adds its own on merge).
FILLED3: set[str] = {"tick", "runner ensure", "review calibrate --all", "jev freeze", "jev drive --retry"}  # tick, r3-review
FILLED3 |= {"store offload", "store pack", "store pack-views"}  # the pack lane, view files
FILLED3 |= {"store ceiling", "store fetch"}  # the store ceiling
FILLED3 |= {"paper find", "paper harvest"}  # r3-papers
FILLED3 |= {"describe drive", "describe write", "describe check"}  # r3-describe
FILLED3 |= {"census run", "census draw"}  # r3-census
FILLED3 |= {"release build", "site count"}  # r3-release
FILLED3 |= {"residual plan", "implement library-adapter", "implement sdist-wheel"}  # r3-residual
FILLED3 |= {"repin check", "repin flip"}  # r3-repin


@pytest.mark.parametrize("verb", sorted(set(RUNG3_VERBS) - FILLED3))
def test_rung3_verbs_reach_their_stubs(verb: str, monkeypatch: pytest.MonkeyPatch,
                                       tmp_path: Path) -> None:
    monkeypatch.setenv("MENAGERIE_DATA", str(tmp_path))
    monkeypatch.setenv("MENAGERIE_STORE", str(tmp_path))
    with pytest.raises(NotImplementedError, match="rung 3"):
        cli.main(RUNG3_VERBS[verb])


# Speed-up verbs (speed-up BUILD-PLAN section 1, item 1), each with the lane that fills it.
SPEEDUP_VERBS = {
    "implement launch-batch": ["implement", "launch-batch", "f-a", "f-b", "--role", "r"],
    "census harvest": ["census", "harvest", "--limit", "20", "--dry-run"],
    "research dossier": ["research", "dossier", "m20601", "--dry-run"],
    "tick tripwires": ["tick", "tripwires", "--resume", "r4-github", "--dry-run"],
    "worklist build": ["worklist", "build", "--rows", "rows.tsv", "--draw", "draw.jsonl"],
    "describe seats": ["describe", "seats"],
    "counts build": ["counts", "build", "--out", "outputs/counts"],
    "replay table": ["replay", "table", "--pin", "2.35.3"],
    "runner big": ["runner", "big", "--dry-run"],
    "resolve run": ["resolve", "run", "m1873", "--limit", "3"],
    "resolve confirm": ["resolve", "confirm", "--candidates", "c.jsonl"],
}
SPEEDUP_LANES = {
    "implement launch-batch": "sp-batch", "census harvest": "sp-census-harvest",
    "research dossier": "sp-census-harvest", "tick tripwires": "sp-overlap",
    "worklist build": "sp-overlap", "describe seats": "sp-describe", "counts build": "sp-hubload",
    "replay table": "sp-replay", "runner big": "sp-bigrow", "resolve run": "sp-router",
    "resolve confirm": "sp-router",
}
# Speed-up verbs whose lane has merged (each lane adds its own on merge).
FILLED_SP: set[str] = {"census harvest", "counts build", "describe seats", "implement launch-batch",
                        "replay table", "research dossier", "resolve confirm", "resolve run", "runner big",
                        "tick tripwires", "worklist build"}


@pytest.mark.parametrize("verb", sorted(set(SPEEDUP_VERBS) - FILLED_SP))
def test_speedup_verbs_reach_their_stubs(verb: str, monkeypatch: pytest.MonkeyPatch,
                                         tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for root in ("MENAGERIE_DATA", "MENAGERIE_STORE", "MENAGERIE_WORK_ROOT"):
        monkeypatch.setenv(root, str(tmp_path))
    assert cli.main(SPEEDUP_VERBS[verb]) == NOT_BUILT
    assert capsys.readouterr().err.strip() == (
        f"menagerie-pipeline: {verb}: not built yet (speed-up lane {SPEEDUP_LANES[verb]})")


def test_a_stub_verb_exits_non_zero_from_the_shell(tmp_path: Path) -> None:
    unbuilt = sorted(set(SPEEDUP_VERBS) - FILLED_SP)
    if not unbuilt:
        pytest.skip("every speed-up verb is built")
    env = {"MENAGERIE_DATA": str(tmp_path), "MENAGERIE_STORE": str(tmp_path),
           "MENAGERIE_WORK_ROOT": str(tmp_path), "PATH": "/usr/bin"}
    done = subprocess.run([sys.executable, "-m", "menagerie.pipeline", *SPEEDUP_VERBS[unbuilt[-1]]],
                          capture_output=True, text=True, cwd=Path(__file__).parents[2], env=env,
                          timeout=120, check=False)
    assert (done.returncode, done.stdout) == (NOT_BUILT, "")
    assert f"{unbuilt[-1]}: not built yet" in done.stderr and "Traceback" not in done.stderr


def test_tick_stays_a_verb_beside_its_tripwires_action() -> None:
    parser = cli.build_parser()
    for argv in (["tick"], ["tick", "--mode", "full", "--dry-run"], ["tick", "--mode", "collect"]):
        assert parser.parse_args(argv).verb == "tick"
    args = parser.parse_args(["tick", "tripwires"])
    assert (args.verb, args.resume, args.dry_run, args.mode) == ("tick tripwires", None, False, None)


def _without_none(raw: dict, change: dict) -> dict:
    return {k: v for k, v in {**raw, **change}.items() if v is not None}


def test_speedup_records_round_trip_and_refuse() -> None:
    rows = [json.loads(s) for s in (FIXTURES / "census.jsonl").read_text().splitlines()]
    assert [from_dict(CensusRow, x).harvested is not None for x in rows] == [False, False, True]
    harvested = rows[2]
    assert from_dict(CensusRow, {**harvested, "paper_refs": []}).paper_refs == ()
    for change, message in (({"harvested": None, "paper_refs": None}, "need `harvested`"),
                            ({"paper_refs": None}, "lists its `paper_refs`"),
                            ({"readme_sha": "abc"}, "readme_sha")):
        with pytest.raises(RecordError, match=message):
            from_dict(CensusRow, _without_none(harvested, change))
    agrees, held = (json.loads(s) for s in (FIXTURES / "dossier.jsonl").read_text().splitlines())
    assert from_dict(Dossier, held).paper_verdict == "held"
    for change, message in (({"paper_verdict": None}, "title verdict"),
                            ({"paper": None, "paper_title": None}, "title verdict"),
                            ({"paper": None, "paper_verdict": None}, "paper title"),
                            ({"by": "agent"}, "by"), ({"commit": "main"}, "commit")):
        with pytest.raises(RecordError, match=message):
            from_dict(Dossier, _without_none(agrees, change))
    yes = json.loads((FIXTURES / "hint.jsonl").read_text().splitlines()[0])
    for change in ({"verdict": "maybe"}, {"kind": "repo"}, {"id": "x1"}):
        with pytest.raises(RecordError):
            from_dict(ResolverHint, {**yes, **change})


def test_speedup_event_verbs_are_events_never_settled_outcomes() -> None:
    """``2b:deferred``, ``cohort:stopped``, ``hint:confirmed`` and ``hint:rejected`` type nothing."""

    from menagerie.pipeline import drive

    assert EVENT_VERBS == {"2b:deferred", "cohort:stopped", "hint:confirmed", "hint:rejected"}
    line = _fixture_line("ledger.jsonl", 0)
    for verb in sorted(EVENT_VERBS):
        assert re.fullmatch(CLASS_PATTERN, verb) is None  # not a failure class
        with pytest.raises(RecordError, match="cls"):
            from_dict(LedgerRow, {**line, "outcome": "giveup", "class": verb})
        with pytest.raises(RecordError, match="cls"):  # a model record cannot give up with one
            from_dict(GiveUp, {"class": verb, "reason": "r", "at": "2026-10-09T13:00:00Z"})
        event = {"at": "2026-10-09T13:00:00Z", "lane": "run-1", "family": "f-a", "verb": verb,
                 "class": "budget"}
        assert not drive._verb(event).startswith(("giveup", "try", "fetch", "done"))
        assert drive._give_up([], [event]) == ""


def test_speedup_constants_keep_todays_behaviour(monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import admission, tick

    keys = ("families_per_2b_lane", "cohort_share", "big_runner")
    assert {k: tick.DEFAULTS[k] for k in keys} == {
        "families_per_2b_lane": 1, "cohort_share": {}, "big_runner": False}
    assert admission.LANE_MINUTES["2b"] == 180
    monkeypatch.delenv("MENAGERIE_LANES_DESCRIBE", raising=False)
    assert admission.lane_cap("describe") == 4 == admission.LANE_CAP_MAX["describe"]
    for given, cap in (("1", 1), ("9", 4)):
        monkeypatch.setenv("MENAGERIE_LANES_DESCRIBE", given)
        assert admission.lane_cap("describe") == cap


def test_review_calibrate_takes_a_role_or_all() -> None:
    parser = cli.build_parser()
    assert parser.parse_args(["review", "calibrate", "--all"]).all
    for argv in (["review", "calibrate"], ["review", "calibrate", "--all", "--role", "r"]):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)


def test_research_and_implement_launch_take_role_and_tier() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["research", "launch", "b.json", "--role", "model-implementation"])
    assert (args.verb, args.role) == ("research launch", "model-implementation")
    args = parser.parse_args(["implement", "launch", "f-nin", "--role", "r", "--tier", "native"])
    assert (args.verb, args.tier) == ("implement launch", "native")
    assert parser.parse_args(["review", "calibrate", "--role", "r"]).runs == 3


def test_implement_launch_places_on_a_configured_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import implement

    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path))
    monkeypatch.setenv("MENAGERIE_DATA", str(DATA))
    (tmp_path / "tick").mkdir()
    (tmp_path / "tick" / "config.json").write_text(json.dumps({"mini_workers": {"linux-a": {"cap": 1}}}))
    seen = []
    monkeypatch.setattr(implement, "launch", lambda family, **kw: seen.append((family, kw)) or "run-1")
    monkeypatch.setattr(implement, "launch_batch", lambda families, **kw: seen.append((families, kw)) or "run-2")
    cli.main(["implement", "launch", "f-nin", "--role", "r", "--host", "linux-a"])
    cli.main(["implement", "launch-batch", "f-a", "f-b", "--role", "r", "--host", "linux-a"])
    cli.main(["implement", "launch", "f-nin", "--role", "r"])
    assert [(kw.get("host"), kw.get("worker")) for _, kw in seen] == [
        ("linux-a", {"cap": 1}), ("linux-a", {"cap": 1}), (None, None)]
    with pytest.raises(SystemExit, match="not in the tick config"):
        cli.main(["implement", "launch", "f-nin", "--role", "r", "--host", "other"])


def test_try_takes_a_draft_recipe_and_a_member(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MENAGERIE_DATA", str(DATA))
    # The live data moves: m8839 validating must not turn this into a leave-alone refusal.
    monkeypatch.setattr("menagerie.pipeline.implement.settled", lambda members, data: {})
    args = cli.build_parser().parse_args(["try", "m8839", "--recipe", "r.json", "--member", "m2"])
    assert (args.verb, args.recipe, args.member) == ("try", Path("r.json"), "m2")
    with pytest.raises(FileNotFoundError, match="r.json"):
        cli.main(["try", "m8839", "--recipe", "r.json"])


def test_hub_verbs_load_neither_torch_nor_psutil(tmp_path: Path) -> None:
    """Import the CLI and the hub-side modules, and run every rung 1 verb, in a fresh process."""

    script = f"""
import importlib, os, sys
os.environ.update(MENAGERIE_DATA={str(tmp_path)!r}, MENAGERIE_STORE={str(tmp_path)!r},
                  MENAGERIE_WORK_ROOT={str(tmp_path)!r})
for name in {HUB_MODULES!r}:
    importlib.import_module("menagerie.pipeline." + name)
from menagerie.pipeline import cli
for argv in {[v for k, v in {**RUNG1_VERBS, **RUNG2_VERBS, **RUNG3_VERBS, **SPEEDUP_VERBS}.items()
              if k not in NETWORK | WORKER_VERBS]!r}:
    try:
        cli.main(argv)
    except Exception:  # a stub, or a filled verb with nothing to work on here
        pass
print(sorted(m for m in ("torch", "psutil") if m in sys.modules))
"""
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          check=True, cwd=Path(__file__).parents[2], timeout=120)
    assert done.stdout.strip().splitlines()[-1] == "[]"


def test_envs_missing_prints_one_name_per_line(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import envs

    monkeypatch.setattr(envs, "missing", lambda *, root: ["jax", "tf"])
    assert cli.main(["envs", "missing", "--root", str(tmp_path)]) in (None, 0)
    assert capsys.readouterr().out == "jax\ntf\n"  # runner_job.sh reads it line by line
