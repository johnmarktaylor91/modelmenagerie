"""scripts/data_status.py: the generated standing-counts block and its staleness check."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]
SCRIPT = REPO / "scripts" / "data_status.py"


def _load():
    spec = importlib.util.spec_from_file_location("data_status", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    data = tmp_path / "menagerie" / "data"
    _jsonl(data / "crawl_roster.jsonl", [
        {"name": "a", "zoo": "timm", "state": "active", "found_by": ["legacy:c1-mech"]},
        {"name": "b", "zoo": "discovered-pytorch", "state": "active",
         "found_by": ["legacy:c2-disco", "S001:axis=era"]},
        {"name": "c", "zoo": "timm", "state": "duplicate_of:a", "found_by": ["legacy:c1-mech"]},
    ])
    _jsonl(data / "sweeps" / "index.jsonl", [
        {"sweep": "S-A", "axes": {"era": {"state": "drained", "frontier": "2026-06-18"}}},
        {"sweep": "S001", "axes": {"venue": {"state": "drained", "frontier": "2026-07-04"},
                                   "recent": {"state": "open", "frontier": "2026-07-04"}}},
    ])
    _jsonl(data / "master_catalog.jsonl", [{"name": "a"}, {"name": "b"}])
    (data / "catalog").mkdir()
    (data / "catalog" / "plan.tsv").write_text("id\twave\nm1\tC1\nm2\tC2\n", encoding="utf-8")
    (data / "models" / "00").mkdir(parents=True)
    recipe = {"rev": "0123456789ab", "kind": "library", "env": "torch-main", "backend": "torch",
              "input": {"preset": "randn", "kind": "random-fallback", "shape": [1, 3]},
              "call": "torch.nn.Linear", "args": [3, 2]}
    model = {"schema": "model/1", "zoo": "timm", "family": "f-x", "source_case": "library",
             "recipe": recipe}
    give_up = {"class": "budget", "reason": "ran out", "at": "2026-10-08T00:00:00Z"}
    for mid, extra in (("m1", {"give_up": give_up}), ("m2", {}), ("m3", {"give_up": give_up})):
        (data / "models" / "00" / f"{mid}.json").write_text(
            json.dumps({**model, "id": mid, "name": mid, **extra}))
    row = {"schema": "ledger/1", "stage": "trace", "host": "w", "env": "torch-main",
           "recipe_rev": "0123456789ab",
           "outcome": "pass", "mode": "eval", "capture": "full", "tl": "2.36.0", "n_ops": 3,
           "graph_hash": "a" * 64, "validation_state": "passed"}
    _jsonl(data / "ledger" / "2026-10.jsonl", [
        {**row, "id": "m2", "at": "2026-10-05T00:00:00Z", "outcome": "fail",
         "class": "torchlens_gap:replay", "validation_state": "failed"},
        {**row, "id": "m2", "at": "2026-10-06T00:00:00Z"},
        {**row, "id": "m3", "at": "2026-10-06T00:00:00Z"},  # a pass beside a give-up
        {**row, "id": "m1", "stage": "forward", "at": "2026-10-07T00:00:00Z"},
    ])
    site = tmp_path / "site" / "site-data"
    site.mkdir(parents=True)
    (site / "export_manifest.json").write_text(json.dumps(
        {"counts": {"models": 7, "validated": 6, "by_torchlens_version": {"2.36.0": 7}},
         "export_version": "phase0-preview"}))
    (tmp_path / "docs").mkdir()
    return tmp_path


def test_render_names_each_count_with_its_file(root: Path) -> None:
    text = _load().render(root)
    assert "`menagerie/data/crawl_roster.jsonl`: 3 rows, 2 active (duplicate_of 1)" in text
    assert "c1-mech 2, c2-disco 1" in text
    assert "latest frontier 2026-07-04; open axes: S001/recent" in text
    assert "`menagerie/data/master_catalog.jsonl`: 2 recipe rows" in text
    assert "2 rows; by wave C1 1, C2 1" in text
    assert ("7 models, 6 of them validated, export `phase0-preview`; traces by TorchLens "
            "version 2.36.0 7." in text)
    assert "As of ledger entry 2026-10-07T00:00:00Z" in text
    assert "3, of which 2 carry a typed give-up reason" in text
    assert "Models with a trace row: 2; validated: 1 (" in text  # m3's give-up wins


def test_check_passes_after_write_and_fails_when_the_roster_changes(root: Path) -> None:
    mod = _load()
    assert mod.main(["--root", str(root)]) == 0
    assert mod.main(["--check", "--root", str(root)]) == 0
    roster = root / "menagerie" / "data" / "crawl_roster.jsonl"
    roster.write_text(roster.read_text() + json.dumps({"name": "d", "zoo": "timm"}) + "\n")
    assert mod.main(["--check", "--root", str(root)]) == 1


def test_committed_status_matches_the_data_files() -> None:
    if not (REPO / "menagerie" / "data" / "crawl_roster.jsonl").exists():
        pytest.skip("no data files in this checkout")
    mod = _load()
    assert mod.main(["--check", "--root", str(REPO)]) == 0, \
        "docs/status.md is stale: run python3 scripts/data_status.py and commit docs/status.md"
    assert "crawl_roster.jsonl`: 28," in mod.render(REPO)
