"""Roster provenance repair on the real roster, stable-id table, July list and S-C plan."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from menagerie.identity import load_stable_ids
from menagerie.pipeline import DataLayout, read_jsonl, roster
from menagerie.pipeline.records import SweepLine

DATA = Path(__file__).parents[2] / "menagerie" / "data"


@pytest.fixture(scope="module")
def repaired(tmp_path_factory: pytest.TempPathFactory) -> tuple[DataLayout, dict, dict]:
    root = tmp_path_factory.mktemp("data")
    shutil.copy(DATA / "crawl_roster.jsonl", root)
    shutil.copy(DATA / "stable_ids.jsonl", root)
    shutil.copytree(DATA / "sweeps", root / "sweeps")
    data = DataLayout(root)
    before = {r.name: r for r in roster.read(data=data)}
    counts = roster.repair(data=data, write=True)
    return data, before, counts


def test_every_row_is_kept_with_its_fields(repaired: tuple) -> None:
    data, before, counts = repaired
    after = roster.read(data=data)
    assert counts["rows"] == len(after) == len(before) == 28_482
    for row in after:
        old = before[row.name]
        assert (row.zoo, row.variant, row.family, row.source_url, row.domain, row.era,
                row.triage_class, row.aliases) == (old.zoo, old.variant, old.family,
                                                   old.source_url, old.domain, old.era,
                                                   old.triage_class, old.aliases)
        assert row.found_by and row.state == old.state  # 2a findings set states; repair keeps them
        if row.source_url:
            assert row.hint_urls[0] == row.source_url


def test_stable_ids_come_unchanged_from_the_table(repaired: tuple) -> None:
    data, _, counts = repaired
    table = load_stable_ids(data.root / "stable_ids.jsonl")
    by_id = {stable_id: key for key, stable_id in table.items()}
    rows = roster.read(data=data)
    stamped = [r for r in rows if r.stable_id]
    assert counts["stable_id"] == len(stamped) > 12_000
    for r in stamped:
        name, zoo, variant = by_id[r.stable_id]
        assert (name, variant) == (r.name, r.variant)
        assert zoo == r.zoo or (r.zoo, zoo in roster.CATALOG_ZOOS) == (roster.INTAKE_ZOO, True)
    assert all((r.name, r.zoo, r.variant) not in table for r in rows if not r.stable_id)
    assert len({r.stable_id for r in stamped}) == len(stamped)


def test_every_active_row_has_an_id(repaired: tuple) -> None:
    data, _, _ = repaired
    assert all(r.stable_id for r in roster.read(data=data) if r.state == "active")


def test_real_table_ids_remain_append_only(repaired: tuple) -> None:
    data, _, _ = repaired
    original = DATA / "stable_ids.jsonl"
    assert (data.root / original.name).read_bytes().startswith(original.read_bytes())


def test_the_s_c_join_reproduces_the_design_measurement(repaired: tuple) -> None:
    data, _, counts = repaired
    # DESIGN section 2, measured 2026-10-03: 5,342 candidates match a roster row and about
    # 1,217 roster rows with no source_url gain the plan's repository URL.
    assert abs(counts["sc_candidates_matched"] - 5_342) <= 5
    assert abs(counts["url_gained"] - 1_217) <= 5
    rows = roster.read(data=data)
    joined = [r for r in rows if any(f.startswith("S-C:cand_") for f in r.found_by)]
    assert len(joined) == counts["s_c"]
    gained = [r for r in rows if not r.source_url and r.hint_urls]
    assert len(gained) >= counts["url_gained"]


def test_every_row_has_a_legacy_partition_and_july_rows_their_axis(repaired: tuple) -> None:
    data, _, counts = repaired
    rows = roster.read(data=data)
    legacy = {f for r in rows for f in r.found_by if f.startswith("legacy:")}
    assert legacy <= {"legacy:c1-mech", "legacy:c2-disco", "legacy:c3-classics",
                      "legacy:c4-native", "legacy:unknown"}
    assert sum(counts.get(k, 0) for k in legacy) == 28_482
    s001 = [r for r in rows if any(f.startswith("S001") for f in r.found_by)]
    assert len(s001) == counts["s001"] > 12_000
    assert sum(any(f.startswith("S001:axis=") for f in r.found_by) for r in s001) > 10_000


def test_repair_is_idempotent(repaired: tuple) -> None:
    data, _, _ = repaired
    first = data.roster.read_bytes()
    roster.repair(data=data, write=True)
    assert data.roster.read_bytes() == first


def test_the_sweep_index_seeds_parse() -> None:
    lines = read_jsonl(SweepLine, DATA / "sweeps" / "index.jsonl")
    assert {line.sweep for line in lines} >= {"S-A", "S-B", "S-C", "S001", "S000"}
    s001 = next(line for line in lines if line.sweep == "S001")
    assert s001.axes["mechanism"].state == "frozen"


def test_norm_is_punctuation_insensitive() -> None:
    assert roster.norm("Wave-MLP") == roster.norm("wave mlp") == "wavemlp"
    assert json.dumps(roster.norm("Neural RRT*")) == '"neuralrrt"'


@pytest.fixture
def small_roster(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DataLayout:
    from menagerie.identity import ensure_stable_ids
    from menagerie.pipeline.records import RosterRow, dumps

    data = DataLayout(tmp_path)
    ensure_stable_ids({("existing", "test", ""), ("off-roster", "test", ""),
                       ("rezoned", "classics-pytorch", "")}, tmp_path / "stable_ids.jsonl")
    table = load_stable_ids(tmp_path / "stable_ids.jsonl")
    rows = [RosterRow(name="existing", zoo="test", stable_id=table[("existing", "test", "")]),
            RosterRow(name="new", zoo="new-zoo"),
            RosterRow(name="rezoned", zoo=roster.INTAKE_ZOO),
            RosterRow(name="inactive", zoo="test", state="not_architecture")]
    data.roster.write_text("".join(dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(roster, "_july", lambda _: {})
    monkeypatch.setattr(roster, "_plan", lambda _: {})
    monkeypatch.setattr(roster, "_partitions", lambda _: {})
    return data


def test_writing_repair_mints_new_active_rows(small_roster: DataLayout) -> None:
    counts = roster.repair(data=small_roster, write=True)
    rows = {r.name: r for r in roster.read(data=small_roster)}
    table = load_stable_ids(small_roster.root / "stable_ids.jsonl")
    assert rows["new"].stable_id == table[("new", "new-zoo", "")]
    assert rows["rezoned"].stable_id == table[("rezoned", "classics-pytorch", "")]
    assert rows["inactive"].stable_id is None
    assert counts["ids_minted"] == 1


def test_mint_preserves_table_and_is_globally_unique(small_roster: DataLayout) -> None:
    path = small_roster.root / "stable_ids.jsonl"
    before_bytes, before = path.read_bytes(), load_stable_ids(path)
    roster.repair(data=small_roster, write=True)
    after = load_stable_ids(path)
    assert path.read_bytes().startswith(before_bytes)
    assert all(after[key] == stable_id for key, stable_id in before.items())
    assert len(after.values()) == len(set(after.values()))
    assert next(r for r in roster.read(data=small_roster) if r.name == "existing").stable_id \
        == before[("existing", "test", "")]
    first = path.read_bytes(), small_roster.roster.read_bytes()
    assert roster.repair(data=small_roster, write=True)["ids_minted"] == 0
    assert first == (path.read_bytes(), small_roster.roster.read_bytes())


def test_dry_repair_never_mints(small_roster: DataLayout) -> None:
    path = small_roster.root / "stable_ids.jsonl"
    before = path.read_bytes(), small_roster.roster.read_bytes()
    counts = roster.repair(data=small_roster)
    assert counts["ids_missing"] == 1
    assert counts["ids_minted"] == 0
    assert before == (path.read_bytes(), small_roster.roster.read_bytes())


def test_repair_mints_with_packet_roster_lock(small_roster: DataLayout,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    import fcntl

    from menagerie import identity

    ensure = identity.ensure_stable_ids
    checked = []

    def under_lock(keys: set, path: Path) -> dict:
        with small_roster.roster.open("r+") as handle, pytest.raises(BlockingIOError):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checked.append(True)
        return ensure(keys, path)

    monkeypatch.setattr(identity, "ensure_stable_ids", under_lock)
    inode = small_roster.roster.stat().st_ino
    roster.repair(data=small_roster, write=True)
    assert checked == [True]
    assert small_roster.roster.stat().st_ino == inode


def test_ambiguous_catalog_alias_keeps_its_own_natural_key(small_roster: DataLayout) -> None:
    from menagerie.identity import ensure_stable_ids

    path = small_roster.root / "stable_ids.jsonl"
    before = ensure_stable_ids({("rezoned", "vendored-pytorch", "")}, path)
    roster.repair(data=small_roster, write=True)
    after = load_stable_ids(path)
    row = next(r for r in roster.read(data=small_roster) if r.name == "rezoned")
    assert row.stable_id == after[("rezoned", roster.INTAKE_ZOO, "")]
    assert row.stable_id not in before.values()
    assert all(after[key] == stable_id for key, stable_id in before.items())


def test_read_parses_an_unchanged_roster_once_and_rereads_a_rewritten_one(
        small_roster: DataLayout, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    os.utime(small_roster.roster, ns=(10**18, 10**18))  # past the memo's racy-write window
    parsed: list[Path] = []
    real = roster._parse
    monkeypatch.setattr(roster, "_parse", lambda p: parsed.append(p) or real(p))
    first = roster.read(data=small_roster)
    first.clear()  # the caller's own list
    assert len(roster.read(data=small_roster)) == 4 and len(parsed) == 1
    small_roster.roster.write_text(small_roster.roster.read_text().splitlines()[0] + "\n")
    assert [r.name for r in roster.read(data=small_roster)] == ["existing"] and len(parsed) == 2


def test_read_of_a_missing_roster_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        roster.read(data=DataLayout(tmp_path))
