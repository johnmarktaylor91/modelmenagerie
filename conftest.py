"""Skip, in the public code repo, the tests that need data or recorded runs it does not ship.

The project's model roster, records and run archives stay private, so the tests that read
them cannot pass here. They are listed by node id prefix in ``public-skips.txt`` and skipped
with that reason; a line starting ``known-failure`` names a test that also fails in the working
repository and is skipped with that reason instead. Every other test runs as usual.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_SKIPS = Path(__file__).with_name("public-skips.txt")


KNOWN = "known-failure "
NEEDS_DATA = pytest.mark.skip(reason="needs project data not included in the public code repo")
FAILS_UPSTREAM = pytest.mark.skip(reason="known failure, also failing in the working repository")


def _lines() -> list[str]:
    if not _SKIPS.is_file():
        return []
    lines = (line.strip() for line in _SKIPS.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def _prefixes() -> tuple[str, ...]:
    return tuple(line.removeprefix(KNOWN) for line in _lines())


def _covers(nodeid: str, entries: tuple[str, ...]) -> bool:
    """An entry names a file, a test, or one parametrized case of a test."""
    return any(nodeid == e or nodeid.startswith((e + "::", e + "[")) for e in entries)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    lines = _lines()
    data = tuple(line for line in lines if not line.startswith(KNOWN))
    known = tuple(line.removeprefix(KNOWN) for line in lines if line.startswith(KNOWN))
    for item in items:
        if _covers(item.nodeid, data):
            item.add_marker(NEEDS_DATA)
        elif _covers(item.nodeid, known):
            item.add_marker(FAILS_UPSTREAM)


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """Whole files listed without a ``::`` part are not even imported (they may need data to import)."""
    root = Path(__file__).parent
    try:
        rel = collection_path.relative_to(root).as_posix()
    except ValueError:
        return None
    return True if rel in {p for p in _prefixes() if "::" not in p} else None
