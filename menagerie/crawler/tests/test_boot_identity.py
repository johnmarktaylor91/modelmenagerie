"""Boot identity comes from a true boot generation, never the wall clock.

D-56 (OC-1): an NTP clock step must never mint a new "boot". The Darwin boot
identity is ``kern.bootsessionuuid`` (a UUID minted once per boot session);
the clock-derived ``kern.boottime`` fallback is deleted, and the whole crawler
derives boot identity through exactly ONE chokepoint,
:func:`menagerie.crawler.worker_supervisor.current_boot_id`.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from menagerie.crawler import worker_supervisor

_CRAWLER_ROOT = Path(worker_supervisor.__file__).resolve().parent
_LINUX_BOOT_ID_PATH = "/proc/sys/kernel/random/boot_id"


class _UnreadablePath:
    """Stand-in for ``Path`` whose reads always fail (no /proc on Darwin)."""

    def __init__(self, *_args: object) -> None:
        pass

    def read_text(self, *_args: object, **_kwargs: object) -> str:
        raise OSError("no linux boot id on this host")


def _fake_sysctl_module(responses: dict[str, list[str]]) -> SimpleNamespace:
    """Serve queued per-key sysctl outputs through a fake subprocess module.

    Parameters
    ----------
    responses:
        Mapping of sysctl key to the queue of values successive calls return.
    """

    def run(args, **_kwargs):  # noqa: ANN001, ANN202
        key = args[-1]
        if key not in responses or not responses[key]:
            raise subprocess.SubprocessError(f"unexpected sysctl key: {key}")
        queue = responses[key]
        value = queue[0] if len(queue) == 1 else queue.pop(0)
        return SimpleNamespace(stdout=value + "\n", returncode=0)

    return SimpleNamespace(run=run, SubprocessError=subprocess.SubprocessError)


@pytest.mark.smoke
def test_clock_step_leaves_boot_id_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """An NTP step that moves kern.boottime never changes the boot identity."""

    monkeypatch.setattr(worker_supervisor, "Path", _UnreadablePath)
    fake = _fake_sysctl_module(
        {
            "kern.bootsessionuuid": ["A0A0A0A0-1111-2222-3333-444444444444"],
            # A clock step re-derives kern.boottime: two different values.
            "kern.boottime": [
                "{ sec = 1756000000, usec = 1 } Sun Aug 24 01:00:00 2026",
                "{ sec = 1756003600, usec = 1 } Sun Aug 24 02:00:00 2026",
            ],
        }
    )
    monkeypatch.setattr(worker_supervisor, "subprocess", fake)
    before = worker_supervisor.current_boot_id()
    after = worker_supervisor.current_boot_id()
    assert before == after
    assert before == "A0A0A0A0-1111-2222-3333-444444444444"


@pytest.mark.smoke
def test_missing_every_boot_source_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """No verifiable boot source refuses typed instead of inventing one."""

    monkeypatch.setattr(worker_supervisor, "Path", _UnreadablePath)
    fake = _fake_sysctl_module({})
    monkeypatch.setattr(worker_supervisor, "subprocess", fake)
    with pytest.raises(RuntimeError, match="boot identity"):
        worker_supervisor.current_boot_id()


@pytest.mark.smoke
def test_empty_boot_session_uuid_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blank sysctl answer is refused, never returned as an identity."""

    monkeypatch.setattr(worker_supervisor, "Path", _UnreadablePath)
    fake = _fake_sysctl_module({"kern.bootsessionuuid": [""]})
    monkeypatch.setattr(worker_supervisor, "subprocess", fake)
    with pytest.raises(RuntimeError, match="boot identity"):
        worker_supervisor.current_boot_id()


def _production_sources() -> list[tuple[Path, str]]:
    """Return every non-test crawler production source with its text."""

    sources: list[tuple[Path, str]] = []
    for path in sorted(_CRAWLER_ROOT.rglob("*.py")):
        if "tests" in path.parts:
            continue
        sources.append((path, path.read_text(encoding="utf-8")))
    return sources


@pytest.mark.smoke
def test_no_clock_derived_boot_identity_anywhere() -> None:
    """kern.boottime is deleted from the whole crawler production tree."""

    offenders = [
        str(path)
        for path, source in _production_sources()
        if "kern.boottime" in source
    ]
    assert offenders == []


@pytest.mark.smoke
def test_exactly_one_boot_identity_derivation() -> None:
    """Boot identity is derived at ONE chokepoint: current_boot_id.

    Every other module must consume ``current_boot_id`` (or a value plumbed
    from it); a second reader of the kernel boot sources is a new derivation
    and fails this census.
    """

    derivations: list[str] = []
    for path, source in _production_sources():
        for marker in (_LINUX_BOOT_ID_PATH, "kern.bootsessionuuid"):
            if marker in source:
                derivations.append(f"{path.name}:{marker}")
    assert sorted(derivations) == [
        f"worker_supervisor.py:{_LINUX_BOOT_ID_PATH}",
        "worker_supervisor.py:kern.bootsessionuuid",
    ]
    tree = ast.parse((_CRAWLER_ROOT / "worker_supervisor.py").read_text(encoding="utf-8"))
    holders = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            isinstance(literal, ast.Constant)
            and isinstance(literal.value, str)
            and literal.value in (_LINUX_BOOT_ID_PATH, "kern.bootsessionuuid")
            for literal in ast.walk(node)
        )
    ]
    assert holders == ["current_boot_id"]
