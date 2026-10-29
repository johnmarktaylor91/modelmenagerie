"""Every durable crawler write goes through ONE power-loss barrier (R2 BL-7 gap).

On macOS -- the only host the round21 release registry supports -- ``os.fsync``
does NOT guarantee power-loss durability: it pushes the write out of the kernel
page cache and returns, leaving the bytes in the drive's volatile cache. Only
``fcntl(fd, F_FULLFSYNC)`` asks the device to flush that cache. BL-7 introduced
``durable_fsync`` for exactly that reason -- and then wired it into the core
``recordio`` ledgers ONLY. Every ledger the wave added afterwards (the three
close meta-ledgers, the eight machine-deferral valves, the park ledgers, the
supervisor/author receipt writers) kept calling bare ``os.fsync``, so a power
loss could erase a row the campaign had already ACKNOWLEDGED -- resurrecting the
award-eating class from a new direction: an acknowledged close, park, or award
row that later does not exist.

This is an AST census, not a spot check, and its scope is EVERY production module
under ``menagerie/crawler/`` (recursively, tests excluded). The rule:

    ``os.fsync`` may be called from exactly ONE place in the tree -- inside
    ``menagerie.crawler.identity.durable_fsync``, as the non-Darwin branch of
    the barrier itself.

That makes durability true BY CONSTRUCTION rather than by review: a ledger added
next quarter fails this census the moment it syncs its own descriptor, and the
only way to pass is to route through the shared barrier. The census is paired
with a behavioural proof that the barrier really issues ``F_FULLFSYNC`` on
Darwin, so it can never pass vacuously against a helper that has quietly
degraded into a plain ``fsync`` wrapper.
"""

from __future__ import annotations

import ast
import fcntl
import os
import sys
from pathlib import Path
from typing import Iterator

import pytest

from menagerie.crawler import identity
from menagerie.crawler.identity import DARWIN_F_FULLFSYNC, durable_fsync
from menagerie.crawler.leg_generation import CLOSE_SOURCE_NAMES

pytestmark = pytest.mark.smoke

_CRAWLER_ROOT = Path(__file__).resolve().parents[1]

#: The ONE sanctioned ``os.fsync`` call site: the barrier's own portable
#: fallback, in the module that owns it. Nothing else in the tree may call it.
_BARRIER_OWNER = ("identity.py", "durable_fsync")

#: THE writer FUNCTIONS for every ledger the close census reads, plus the three
#: close meta-ledgers and the operational event ledger. Keyed by the census
#: scope names so a NEW census slot cannot be added without naming its writer
#: here -- the coverage assertion below is set equality against
#: ``CLOSE_SOURCE_NAMES``, never a subset check.
#:
#: PSLRE5 P5-D-3: this used to name the writer MODULE, and the second arm of the
#: census ("this module calls ``durable_fsync`` at least once") is therefore
#: module-granular. One module writes two slots -- the money ROW and its anchor --
#: so the anchor's barrier satisfied the row's duty too, and DELETING the money
#: row's power-loss barrier entirely (one line, the single most plausible
#: regression there is) left the census fully green. Naming the FUNCTION makes the
#: duty per writer: every named writer must issue the barrier in its own body, and
#: after its last write.
_CENSUS_LEDGER_WRITERS: dict[str, tuple[str, tuple[str, ...]]] = {
    # The four fact ledgers, in both layouts, are written by the one canonical
    # JsonlLedger writer.
    "models.shard": ("recordio.py", ("append",)),
    "models.flat": ("recordio.py", ("append",)),
    "attempts.local": ("recordio.py", ("append",)),
    "attempts.flat": ("recordio.py", ("append",)),
    "gates.shard": ("recordio.py", ("append",)),
    "gates.flat": ("recordio.py", ("append",)),
    "artifacts.shard": ("recordio.py", ("append",)),
    "artifacts.flat": ("recordio.py", ("append",)),
    # The paid-session money ledger and its external extent anchor share one
    # MODULE -- the durability concern the accounting and custody concerns are
    # separated from -- but they are two distinct writers with two distinct duties.
    "paid.sessions": ("paid_session_durability.py", ("append",)),
    "paid.sessions.anchor": ("paid_session_durability.py", ("_write_anchor",)),
    # The eight machine-deferral valve / park ledgers, each hand-rolled, each with
    # its row writer and (where it has one) its void writer.
    "valve.capacity": ("host_capacity.py", ("append_capacity_deferral_row",)),
    "valve.coverage": ("environment_coverage.py", ("append_coverage_deferral_row",)),
    "valve.checker-lane": ("checker_lane.py", ("append_checker_lane_deferral_row",)),
    "valve.rebuild": ("environment_rebuild.py", ("append_rebuild_deferral_row",)),
    "valve.disk-floor": ("disk_floor.py", ("append_disk_floor_deferral_row",)),
    "valve.scheduler-quarantine": (
        "scheduler_parks.py",
        ("append_scheduler_quarantine_row", "append_scheduler_quarantine_void"),
    ),
    "park.grant": ("grant_parks.py", ("append_grant_park_row", "append_grant_park_void")),
    "park.refusal": (
        "refusal_parks.py",
        ("append_refusal_park_row", "append_refusal_park_void"),
    ),
}

#: The close/licensing spine's own ledgers plus the operational event ledger.
#: These are not census SOURCES (a close does not attest itself), but they are
#: read by every close and staging decision and carry the same durability duty.
_SPINE_LEDGER_WRITERS: dict[str, tuple[str, tuple[str, ...]]] = {
    "meta.leg-events": ("leg_generation.py", ("_append_leg_event", "open_leg_generation")),
    "meta.closes": ("close_object.py", ("append_close_object",)),
    "meta.registry": ("rung_staging.py", ("append_rung_registration",)),
    "operational.events": (
        "checkpoint.py",
        ("append_canonical_requeue_grant", "append_canonical_requeue_void"),
    ),
}


def _production_modules() -> Iterator[Path]:
    """Yield every production crawler module, tests and caches excluded."""

    for path in sorted(_CRAWLER_ROOT.rglob("*.py")):
        relative = path.relative_to(_CRAWLER_ROOT)
        if relative.parts[0] in {"tests", "__pycache__"}:
            continue
        yield path


def _enclosing_functions(tree: ast.AST) -> dict[ast.AST, str]:
    """Map every node to the name of the function that lexically encloses it."""

    owner: dict[ast.AST, str] = {}

    def walk(node: ast.AST, current: str) -> None:
        for child in ast.iter_child_nodes(node):
            name = (
                child.name
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                else current
            )
            owner[child] = name
            walk(child, name)

    walk(tree, "<module>")
    return owner


def bare_fsync_sites(source: str) -> list[tuple[int, str]]:
    """Return every bare-``fsync`` site as ``(line, enclosing function)``.

    THE scanner both this census and its own negative control run, so the census
    can never disagree with the check that proves the census works.

    Two spellings, because the census is a structural guarantee and an
    ``ast.Attribute`` match is only a guarantee about ONE way to write the call
    (PSLRE4 D-4). ``from os import fsync as _f; _f(fd)`` reaches the identical
    page-cache-only syscall while matching nothing an attribute scan looks at, so
    the IMPORT itself is a site (a production module has no other reason to bind
    the name), and so is any call through a name that import bound.
    """

    tree = ast.parse(source)
    owner = _enclosing_functions(tree)
    sites: list[tuple[int, str]] = []
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module != "os":
            continue
        bound = [alias for alias in node.names if alias.name == "fsync"]
        if not bound:
            continue
        aliases.update(alias.asname or alias.name for alias in bound)
        sites.append((node.lineno, owner.get(node, "<module>")))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if (
            isinstance(function, ast.Attribute)
            and function.attr == "fsync"
            and isinstance(function.value, ast.Name)
            and function.value.id == "os"
        ):
            sites.append((node.lineno, owner.get(node, "<module>")))
        elif isinstance(function, ast.Name) and function.id in aliases:
            sites.append((node.lineno, owner.get(node, "<module>")))
    return sorted(sites)


def _calls_durable_fsync(source: str) -> bool:
    """Return whether a module calls the shared barrier at least once."""

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name) and function.id == "durable_fsync":
            return True
        if isinstance(function, ast.Attribute) and function.attr == "durable_fsync":
            return True
    return False


def _called_names(tree: ast.AST, wanted: frozenset[str]) -> list[tuple[int, str]]:
    """Return every call of a wanted name in one subtree as ``(line, name)``."""

    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = (
            function.attr
            if isinstance(function, ast.Attribute)
            else function.id
            if isinstance(function, ast.Name)
            else ""
        )
        if name in wanted:
            found.append((node.lineno, name))
    return sorted(found)


#: The write spellings a durable writer must have flushed before it returns.
_WRITE_CALLS = frozenset({"write", "writelines"})

#: Barriers that flush the FILE's OWN BYTES: ``durable_fsync`` on the data
#: descriptor, and ``atomic_replace_bytes`` which owns that call for a replaced
#: object. Only these discharge a data-row writer's duty.
_DATA_BARRIER_CALLS = frozenset({"durable_fsync", "atomic_replace_bytes"})

#: Barriers that flush a DIRECTORY ENTRY. A different syscall making a different
#: guarantee: it persists the NAME of a newly created file, and says nothing about
#: the bytes inside it. PSLRE6 D6-1: this used to sit in one undifferentiated
#: barrier vocabulary with the two above, so 17 of the 24 named writers -- every
#: writer that fsyncs its parent directory on creation, including the MONEY row --
#: could have their data barrier deleted outright and still pass, because the
#: trailing ``fsync_directory`` answered for it. A directory barrier can no longer
#: satisfy a data-row duty.
_DIRECTORY_BARRIER_CALLS = frozenset({"fsync_directory"})

_BARRIER_CALLS = _DATA_BARRIER_CALLS | _DIRECTORY_BARRIER_CALLS


def durable_writer_findings(source: str, *, function: str) -> list[str]:
    """Return why one NAMED writer function is not power-loss durable, if it is not.

    THE per-writer scanner both this census and its own negative control run, so the
    census can never disagree with the check that proves it works.

    Two structural conditions, both inside the named function's own AST subtree:

    * it issues a DATA barrier itself -- a sibling writer in the same module can no
      longer discharge its duty (P5-D-3), and neither can its own trailing DIRECTORY
      barrier, which persists the file's name and not its bytes (PSLRE6 D6-1);
    * that data barrier comes AFTER its last write, so a barrier that has drifted
      above the write it is supposed to make durable is a finding rather than a green
      tick. A directory barrier after the last write does not satisfy the ordering
      either, for the same reason it does not satisfy the duty.
    """

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != function:
            continue
        writes = _called_names(node, _WRITE_CALLS)
        barriers = _called_names(node, _DATA_BARRIER_CALLS)
        if not barriers:
            directory = _called_names(node, _DIRECTORY_BARRIER_CALLS)
            return [
                f"{function} writes a durable ledger row but never issues the "
                "power-loss barrier for its own bytes: its acknowledged rows can "
                "vanish"
                + (
                    f" (its only barrier is a DIRECTORY sync at line {directory[0][0]}, "
                    "which persists the file's name, not the row inside it)"
                    if directory
                    else ""
                )
            ]
        if writes and max(line for line, _ in barriers) < max(
            line for line, _ in writes
        ):
            return [
                f"{function} writes after its last power-loss barrier at line "
                f"{max(line for line, _ in barriers)}: the trailing write is not durable"
            ]
        return []
    return [f"{function} is missing: a named durable writer may not simply disappear"]


def test_no_production_module_syncs_outside_the_one_barrier() -> None:
    """Bare ``os.fsync`` exists in exactly one function in the whole tree.

    A durable append that calls ``os.fsync`` on macOS acknowledges a row the
    device may still lose. Routing every site through ``durable_fsync`` is what
    makes the acknowledgement honest, and confining the raw call to the barrier
    itself is what keeps a NEW ledger from silently opting out.

    Every production module is swept in ONE test that reports every offender at
    once, so a durability regression is never a single-module surprise that
    hides the rest of the sweep behind an early exit.
    """

    modules = sorted(_production_modules())
    assert len(modules) > 100, "the census must sweep the whole production tree"

    offenders: list[str] = []
    barrier_sites: list[tuple[int, str]] = []
    for module in modules:
        sites = bare_fsync_sites(module.read_text(encoding="utf-8"))
        if not sites:
            continue
        if module.name == _BARRIER_OWNER[0]:
            barrier_sites = sites
            continue
        offenders.append(f"{module.relative_to(_CRAWLER_ROOT)}:{[line for line, _ in sites]}")

    assert offenders == [], (
        "these modules call os.fsync instead of menagerie.crawler.identity."
        "durable_fsync; on macOS os.fsync is NOT a power-loss barrier, so an "
        "acknowledged row can vanish: " + ", ".join(offenders)
    )
    assert [name for _, name in barrier_sites] == [_BARRIER_OWNER[1]], (
        f"{_BARRIER_OWNER[0]} may only call os.fsync inside {_BARRIER_OWNER[1]} -- the "
        f"barrier's own portable fallback; found {barrier_sites}"
    )


def test_the_census_scanner_flags_a_reintroduced_bare_fsync() -> None:
    """The scanner is not vacuous: it sees both spellings it must forbid."""

    offender = "import os\ndef append(handle):\n    os.fsync(handle.fileno())\n"
    assert bare_fsync_sites(offender) == [(3, "append")]

    nested = "import os\ndef outer():\n    def inner(fd):\n        os.fsync(fd)\n    return inner\n"
    assert bare_fsync_sites(nested) == [(4, "inner")]

    # PSLRE4 D-4: the aliased import reaches the same page-cache-only syscall
    # while matching nothing an ``os.fsync`` attribute scan looks at. Both the
    # import that binds the name and the call through it are sites.
    aliased = (
        "from os import fsync as _flush\n"
        "def append(handle):\n"
        "    _flush(handle.fileno())\n"
    )
    assert bare_fsync_sites(aliased) == [(1, "<module>"), (3, "append")]

    unaliased = "from os import fsync\ndef append(fd):\n    fsync(fd)\n"
    assert bare_fsync_sites(unaliased) == [(1, "<module>"), (3, "append")]

    # Importing a sibling ``os`` name is not a durability escape.
    innocent = "from os import replace\ndef swap(a, b):\n    replace(a, b)\n"
    assert bare_fsync_sites(innocent) == []

    # A same-named local that no ``os`` import bound is not a site either.
    shadowed = "def fsync(fd):\n    return fd\ndef append(fd):\n    fsync(fd)\n"
    assert bare_fsync_sites(shadowed) == []

    compliant = (
        "from menagerie.crawler.identity import durable_fsync\n"
        "def append(handle):\n"
        "    durable_fsync(handle.fileno())\n"
    )
    assert bare_fsync_sites(compliant) == []
    assert _calls_durable_fsync(compliant)
    assert not _calls_durable_fsync(offender)


def test_durable_fsync_issues_the_platform_power_loss_barrier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The barrier really flushes the device cache, per platform.

    Without this the census above could pass against a ``durable_fsync`` that
    had quietly become an ``os.fsync`` alias -- every call site "compliant" and
    every acknowledgement still a lie.
    """

    issued: list[tuple[int, int]] = []
    plain: list[int] = []
    monkeypatch.setattr(
        identity.fcntl, "fcntl", lambda fd, command, *rest: issued.append((fd, command))
    )
    monkeypatch.setattr(identity.os, "fsync", plain.append)

    path = tmp_path / "ledger.jsonl"
    path.write_bytes(b'{"row":1}\n')
    with path.open("ab") as handle:
        durable_fsync(handle.fileno())
        descriptor = handle.fileno()

    if sys.platform == "darwin":
        assert issued == [(descriptor, DARWIN_F_FULLFSYNC)]
        assert plain == [], "Darwin must not settle for a page-cache-only fsync"
    else:
        assert plain == [descriptor]
        assert issued == []


def test_the_darwin_barrier_command_is_the_real_kernel_constant(tmp_path: Path) -> None:
    """``DARWIN_F_FULLFSYNC`` is the live kernel command, not a stale literal."""

    assert DARWIN_F_FULLFSYNC == 51
    if sys.platform != "darwin":
        pytest.skip("F_FULLFSYNC is a Darwin command")
    path = tmp_path / "probe.jsonl"
    path.write_bytes(b'{"row":1}\n')
    with path.open("ab") as handle:
        handle.write(b'{"row":2}\n')
        handle.flush()
        # Raises OSError if the command is wrong for a regular file.
        fcntl.fcntl(handle.fileno(), DARWIN_F_FULLFSYNC)
        os.fsync(handle.fileno())


def test_every_census_read_ledger_writer_is_named_and_durable() -> None:
    """Every ledger a close reads has ONE named writer, and it uses the barrier.

    The keys are checked for exact set equality against ``CLOSE_SOURCE_NAMES``
    plus the paid-session accounting ledger. The latter is deliberately not a
    close input: its writer is durability-gated here without perturbing
    unrelated close/licensing content addresses.
    """

    assert set(_CENSUS_LEDGER_WRITERS) == {
        *CLOSE_SOURCE_NAMES,
        "paid.sessions",
        "paid.sessions.anchor",
    }, (
        "the census universe changed: name the writer module for every slot in "
        "_CENSUS_LEDGER_WRITERS so its durability is enforced here"
    )

    slots = {**_CENSUS_LEDGER_WRITERS, **_SPINE_LEDGER_WRITERS}
    for scope, (module_name, functions) in sorted(slots.items()):
        module = _CRAWLER_ROOT / module_name
        assert module.is_file(), f"{scope}: writer module {module_name} is missing"
        source = module.read_text(encoding="utf-8")
        assert _calls_durable_fsync(source), (
            f"{scope}: {module_name} writes a close-relevant ledger but never calls "
            "durable_fsync -- its acknowledged rows are not power-loss durable"
        )
        assert bare_fsync_sites(source) == [], (
            f"{scope}: {module_name} still has a bare os.fsync at "
            f"{[line for line, _ in bare_fsync_sites(source)]}"
        )
        assert functions, f"{scope}: name the function that writes this ledger"
        for function in functions:
            findings = durable_writer_findings(source, function=function)
            assert findings == [], f"{scope} ({module_name}): " + "; ".join(findings)


def test_the_per_writer_scanner_flags_an_omitted_barrier() -> None:
    """The per-writer scanner is not vacuous: it sees the DELETED barrier.

    PSLRE5 P5-D-3's exact regression, which the module-granular census could not
    see: one line removed from the money row's append while a SIBLING writer in the
    same module (the anchor) kept calling the barrier. The scanner is per function,
    so the sibling's barrier no longer answers for this writer's duty.
    """

    durable = (
        "from menagerie.crawler.identity import durable_fsync\n"
        "def append(handle, line):\n"
        "    handle.write(line)\n"
        "    handle.flush()\n"
        "    durable_fsync(handle.fileno())\n"
        "def _write_anchor(handle, row):\n"
        "    handle.write(row)\n"
        "    durable_fsync(handle.fileno())\n"
    )
    assert durable_writer_findings(durable, function="append") == []
    assert durable_writer_findings(durable, function="_write_anchor") == []

    omitted = durable.replace("    durable_fsync(handle.fileno())\n", "", 1)
    findings = durable_writer_findings(omitted, function="append")
    assert findings and "never issues the power-loss barrier" in findings[0], findings
    # ...while the sibling writer in the SAME module is still durable, which is
    # precisely why a module-granular census stayed green.
    assert durable_writer_findings(omitted, function="_write_anchor") == []
    assert _calls_durable_fsync(omitted), (
        "the module-level arm cannot see the omission: that is the finding this "
        "per-writer arm exists to catch"
    )

    reordered = (
        "from menagerie.crawler.identity import durable_fsync\n"
        "def append(handle, line):\n"
        "    durable_fsync(handle.fileno())\n"
        "    handle.write(line)\n"
    )
    findings = durable_writer_findings(reordered, function="append")
    assert findings and "after its last power-loss barrier" in findings[0], findings

    renamed = durable.replace("def append(", "def append_renamed(")
    findings = durable_writer_findings(renamed, function="append")
    assert findings and "is missing" in findings[0], findings

    # An atomically replaced object is durable through the shared primitive that
    # owns the barrier, so naming it is not an escape from the duty.
    atomic = (
        "from menagerie.crawler.identity import atomic_replace_bytes\n"
        "def append_close_object(path, payload):\n"
        "    atomic_replace_bytes(path, payload)\n"
    )
    assert durable_writer_findings(atomic, function="append_close_object") == []


def test_a_directory_barrier_cannot_discharge_a_data_row_duty() -> None:
    """PSLRE6 D6-1: ``fsync_directory`` persists the NAME, never the row's bytes.

    17 of the 24 named writers -- every one that fsyncs its parent directory when it
    creates the ledger, including the MONEY row's append -- sat in a census whose
    barrier vocabulary made no distinction, so deleting the data barrier outright
    (one line, the single most plausible durability regression there is) left the
    trailing directory sync answering for it and the census fully green.

    The two vocabularies are now disjoint and only the data half discharges the duty,
    which is asserted here rather than assumed.
    """

    assert not (_DATA_BARRIER_CALLS & _DIRECTORY_BARRIER_CALLS)
    assert _BARRIER_CALLS == _DATA_BARRIER_CALLS | _DIRECTORY_BARRIER_CALLS

    shadowed = (
        "from menagerie.crawler.identity import durable_fsync, fsync_directory\n"
        "def append(path, handle, line):\n"
        "    handle.write(line)\n"
        "    handle.flush()\n"
        "    durable_fsync(handle.fileno())\n"
        "    fsync_directory(path.parent)\n"
    )
    assert durable_writer_findings(shadowed, function="append") == []

    # The exact regression: the data barrier deleted, the directory sync kept.
    deleted = shadowed.replace("    durable_fsync(handle.fileno())\n", "", 1)
    findings = durable_writer_findings(deleted, function="append")
    assert findings and "never issues the power-loss barrier for its own bytes" in (
        findings[0]
    ), findings
    assert "DIRECTORY sync" in findings[0], findings

    # ...and the ordering arm is data-only too: a directory sync trailing the last
    # write does not make the write durable, so it cannot repair the ordering.
    reordered = (
        "from menagerie.crawler.identity import durable_fsync, fsync_directory\n"
        "def append(path, handle, line):\n"
        "    durable_fsync(handle.fileno())\n"
        "    handle.write(line)\n"
        "    fsync_directory(path.parent)\n"
    )
    findings = durable_writer_findings(reordered, function="append")
    assert findings and "after its last power-loss barrier" in findings[0], findings
