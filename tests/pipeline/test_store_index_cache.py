"""The per-process store index cache: lookups read only appended bytes and rebuild on rewrites."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from menagerie.pipeline import DataLayout, StoreLayout, append_jsonl, store
from menagerie.pipeline.records import StoreIndexLine, dumps

SHARD = "2026-10"


@pytest.fixture()
def data(tmp_path: Path) -> DataLayout:
    store._SHARDS.clear()
    return DataLayout(tmp_path / "data")


@pytest.fixture()
def scans(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every line text the shard reader parses, in order."""

    seen: list[str] = []
    real = store._scan_line

    def counting(text: str) -> str:
        seen.append(text)
        return real(text)

    monkeypatch.setattr(store, "_scan_line", counting)
    return seen


def _line(n: int, **extra: object) -> StoreIndexLine:
    return StoreIndexLine(sha256=f"{n:064x}", bytes=n, mime="text/plain",
                          fetched="2026-10-29T00:00:00Z", license_class="own",
                          url=f"https://example.org/{n}", **extra)  # type: ignore[arg-type]


def _text(*lines: StoreIndexLine) -> str:
    return "".join(dumps(line) + "\n" for line in lines)


def _write(data: DataLayout, *lines: StoreIndexLine, shard: str = SHARD) -> Path:
    path = data.store_index(shard)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_text(*lines), encoding="utf-8")
    return path


def test_appended_lines_are_seen_by_parsing_only_the_new_bytes(data: DataLayout,
                                                               scans: list[str]) -> None:
    _write(data, _line(1), _line(2), _line(3))
    assert store.index_line(_line(2).sha256, data=data) == _line(2)
    assert len(scans) == 3
    assert store.index_line(_line(3).sha256, data=data) == _line(3)
    assert len(scans) == 3  # unchanged shard: a stat, no parse
    append_jsonl([_line(4)], data.store_index(SHARD))  # another writer's append
    assert store.index_line(_line(4).sha256, data=data) == _line(4)
    assert len(scans) == 4
    assert store.index_line("f" * 64, data=data) is None


def test_a_replaced_shard_is_read_again(data: DataLayout, scans: list[str]) -> None:
    path = _write(data, _line(1), _line(2))
    assert store.index_line(_line(1).sha256, data=data) == _line(1)
    tmp = path.with_name("tmp.part")
    tmp.write_text(_text(_line(2), _line(5), _line(6)), encoding="utf-8")
    os.replace(tmp, path)  # store_trim and store_ceiling rewrite this way
    assert store.index_line(_line(1).sha256, data=data) is None
    assert store.index_line(_line(5).sha256, data=data) == _line(5)
    assert len(scans) == 2 + 3


def test_a_truncated_shard_is_read_again(data: DataLayout) -> None:
    path = _write(data, _line(1), _line(2), _line(3))
    assert store.index_line(_line(3).sha256, data=data) == _line(3)
    with path.open("r+", encoding="utf-8") as handle:  # rewritten in place, shorter
        handle.truncate(0)
        handle.write(_text(_line(1)))
    assert store.index_line(_line(3).sha256, data=data) is None
    assert store.index_line(_line(1).sha256, data=data) == _line(1)


def test_a_shard_rewritten_in_place_and_longer_is_read_again(data: DataLayout) -> None:
    path = _write(data, _line(1), _line(2))
    assert store.index_line(_line(1).sha256, data=data) == _line(1)
    with path.open("r+", encoding="utf-8") as handle:  # truncate-and-append, larger than before
        handle.truncate(0)
        handle.write(_text(_line(7), _line(8), _line(9)))
    assert store.index_line(_line(1).sha256, data=data) is None
    assert [store.index_line(_line(n).sha256, data=data) for n in (7, 8, 9)] == \
        [_line(7), _line(8), _line(9)]


def test_a_same_size_rewrite_is_read_again(data: DataLayout) -> None:
    path = _write(data, _line(1))
    assert store.index_line(_line(1).sha256, data=data) == _line(1)
    before = path.stat()
    path.write_text(_text(_line(2)), encoding="utf-8")  # same length: only mtime tells
    assert path.stat().st_size == before.st_size
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
    assert store.index_line(_line(1).sha256, data=data) is None
    assert store.index_line(_line(2).sha256, data=data) == _line(2)


def test_a_new_shard_is_picked_up_in_name_order(data: DataLayout) -> None:
    _write(data, _line(1, encoding="gzip", location="local"), shard="2026-11")
    assert store.index_line(_line(1).sha256, data=data).encoding == "gzip"
    _write(data, _line(1), shard="2026-09")  # sorts first, so its line is now the first match
    assert store.index_line(_line(1).sha256, data=data) == _line(1)
    assert store.placement(_line(1).sha256, data=data).encoding == "gzip"
    data.store_index("2026-09").unlink()
    assert store.index_line(_line(1).sha256, data=data).encoding == "gzip"


def test_a_partial_last_line_waits_for_its_newline(data: DataLayout, scans: list[str]) -> None:
    path = _write(data, _line(1))
    whole = dumps(_line(2)) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(whole[:40])  # a writer mid-append
    assert store.index_line(_line(1).sha256, data=data) == _line(1)
    assert store.index_line(_line(2).sha256, data=data) is None
    assert len(scans) == 1
    with path.open("a", encoding="utf-8") as handle:
        handle.write(whole[40:])
    assert store.index_line(_line(2).sha256, data=data) == _line(2)
    assert scans == [dumps(_line(1)), dumps(_line(2))]


def test_duplicate_shas_keep_first_match_and_latest_placement(data: DataLayout) -> None:
    sha = _line(1).sha256
    packed = _line(1, encoding="tar.gz", location="local")
    _write(data, _line(1), packed)
    assert store.index_line(sha, data=data) == _line(1)
    assert store.placement(sha, data=data) == packed
    moved = _line(1, encoding="tar.gz", location="box:" + "c" * 64)
    append_jsonl([moved], data.store_index(SHARD))
    assert store.index_line(sha, data=data) == _line(1)
    assert store.placement(sha, data=data) == moved


def test_a_put_sees_its_own_known_append(tmp_path: Path, data: DataLayout) -> None:
    st = StoreLayout(tmp_path / "store")
    blob = tmp_path / "blob.txt"
    blob.write_text("hello\n", encoding="utf-8")
    first = store.put(blob, store=st, data=data, mime="text/plain", license_class="own")
    assert store.index_line(first.sha256, data=data) == first  # warms the cache
    again = store.put(blob, store=st, data=data, mime="text/plain", license_class="own",
                      url="https://example.org/blob.txt")  # ``_known`` appends a line
    assert again.url == "https://example.org/blob.txt"
    assert [line.url for line in store._index_lines(first.sha256, data)] == [None, again.url]
    # The append is visible: the same address finds that line and appends nothing more.
    assert store.put(blob, store=st, data=data, mime="text/plain", license_class="own",
                     url="https://example.org/blob.txt") == again
    assert len(store._index_lines(first.sha256, data)) == 2
    assert store.index_line(first.sha256, data=data) == first
