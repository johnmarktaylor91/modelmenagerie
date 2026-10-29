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
    (data / "models" / "00" / "m1.json").write_text(json.dumps({"id": "m1", "give_up": {"class": "x"}}))
    (data / "models" / "00" / "m2.json").write_text(json.dumps({"id": "m2"}))
    _jsonl(data / "ledger" / "2026-10.jsonl", [
        {"id": "m2", "stage": "trace", "at": "2026-10-05T00:00:00Z", "validation_state": "failed"},
        {"id": "m2", "stage": "trace", "at": "2026-10-06T00:00:00Z", "validation_state": "passed"},
        {"id": "m1", "stage": "forward", "at": "2026-10-07T00:00:00Z"},
    ])
    site = tmp_path / "site" / "site-data"
    site.mkdir(parents=True)
    (site / "export_manifest.json").write_text(json.dumps(
        {"counts": {"models": 7}, "export_version": "phase0-preview"}))
    (tmp_path / "docs").mkdir()
    return tmp_path


def test_render_names_each_count_with_its_file(root: Path) -> None:
    text = _load().render(root)
    assert "`menagerie/data/crawl_roster.jsonl`: 3 rows, 2 active (duplicate_of 1)" in text
    assert "c1-mech 2, c2-disco 1" in text
    assert "latest frontier 2026-07-04; open axes: S001/recent" in text
    assert "`menagerie/data/master_catalog.jsonl`: 2 recipe rows" in text
    assert "2 rows; by wave C1 1, C2 1" in text
    assert "7 render-only rows, export `phase0-preview`" in text
    assert "As of ledger entry 2026-10-07T00:00:00Z" in text
    assert "2, of which 1 carry a typed give-up reason" in text
    assert "Models with a trace row: 1; whose latest trace passed validation: 1" in text


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
