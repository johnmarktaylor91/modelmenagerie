"""The RUNTIME arm of the durability census: measured barriers, not parsed spellings.

PSLRE6 D6-4's direction, adopted literally: STOP chasing AST spellings. The census in
``test_durable_fsync_census.py`` has been patched twice for spellings it structurally
could not see, and the next one -- a money-row append reaching the page-cache-only
syscall through a COMPUTED attribute, ``getattr(os, "fs" + "ync")(fd)`` -- is invisible
to every attribute and import arm by construction. A third and fourth pattern would
close two more spellings out of unboundedly many.

This module changes the evidence instead. It patches the barrier PRIMITIVES on the
``os`` and ``fcntl`` module objects, drives each named writer's REAL append, and
asserts what reached the syscall boundary: a device barrier on the ledger's own data
inode, after its last write, issued from that writer's own call stack. Any spelling
that resolves ``os.fsync`` at runtime lands in the spy; on Darwin -- the only host the
round21 release registry supports -- a page-cache-only flush of a ledger inode is now
a FAILURE however it is written.

The two arms are complementary and the complementarity is asserted, not assumed:
an IMPORT-time alias (``from os import fsync as _flush``) captured the primitive
before the spy existed and is therefore invisible HERE, and is exactly what the AST
census's import arm forbids structurally.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from menagerie.crawler.tests.durable_runtime_support import (
    DEVICE_BARRIER,
    PAGE_CACHE_ONLY,
    RUNTIME_DRIVERS,
    BarrierRecorder,
    durable_runtime_findings,
)
from menagerie.crawler.tests.test_durable_fsync_census import (
    _CENSUS_LEDGER_WRITERS,
    _SPINE_LEDGER_WRITERS,
)

pytestmark = pytest.mark.smoke


def census_writer_duties() -> frozenset[tuple[str, str]]:
    """Return every ``(module, function)`` durability duty the AST census names."""

    duties: set[tuple[str, str]] = set()
    for module_name, functions in {
        **_CENSUS_LEDGER_WRITERS,
        **_SPINE_LEDGER_WRITERS,
    }.values():
        duties.update((module_name, function) for function in functions)
    return frozenset(duties)


def test_every_named_writer_duty_has_a_runtime_driver() -> None:
    """The runtime arm's coverage is SET EQUALITY against the census, not a sample.

    A subset check would let the next ledger land with an AST-only duty -- the same
    coverage-hole shape as PSLRE5 acct F3, where the check existed and worked and the
    newest slot simply was not in its list. Equality means a new writer must be driven
    for real before it can pass.
    """

    assert set(RUNTIME_DRIVERS) == set(census_writer_duties()), (
        "the runtime arm and the AST census disagree about the writer universe; "
        f"AST-only: {sorted(census_writer_duties() - set(RUNTIME_DRIVERS))}; "
        f"runtime-only: {sorted(set(RUNTIME_DRIVERS) - census_writer_duties())}"
    )


@pytest.mark.parametrize(
    ("module_name", "function"), sorted(RUNTIME_DRIVERS), ids=lambda value: str(value)
)
def test_each_named_writer_issues_a_data_barrier_at_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module_name: str, function: str
) -> None:
    """Each writer's REAL append flushes its own bytes to the device, in order."""

    recorder = BarrierRecorder()
    recorder.install(monkeypatch)
    ledger = RUNTIME_DRIVERS[(module_name, function)](tmp_path)
    findings = durable_runtime_findings(
        recorder, ledger=ledger, module_name=module_name, function=function
    )
    assert findings == [], f"{module_name}:{function}: " + "; ".join(findings)


def test_the_recorder_sees_a_computed_attribute_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The D6-4 spelling: a runtime attribute lookup lands in the spy.

    Without this the whole runtime arm could pass vacuously against a recorder that
    only ever observed the sanctioned helper. The escape is reproduced exactly as the
    lens found it -- a computed name, so no attribute or import scan matches -- and it
    must be VISIBLE here and classified as page-cache-only.
    """

    recorder = BarrierRecorder()
    recorder.install(monkeypatch)

    path = tmp_path / "ledger.jsonl"
    path.write_bytes(b'{"row":1}\n')
    with path.open("ab") as handle:
        handle.write(b'{"row":2}\n')
        handle.flush()
        getattr(os, "fs" + "ync")(handle.fileno())

    observed = recorder.data_barriers(path.stat())
    assert [event.kind for event in observed] == [PAGE_CACHE_ONLY]
    assert observed[0].size == path.stat().st_size
    assert observed[0].issued_by(
        "test_durable_fsync_runtime.py", "test_the_recorder_sees_a_computed_attribute_escape"
    )

    if sys.platform == "darwin":
        findings = durable_runtime_findings(
            recorder,
            ledger=path,
            module_name="test_durable_fsync_runtime.py",
            function="test_the_recorder_sees_a_computed_attribute_escape",
        )
        assert findings and "rather than F_FULLFSYNC" in findings[0], findings


def test_the_shared_barrier_is_a_device_barrier_on_darwin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``durable_fsync`` is what the runtime arm accepts, and the escape is not."""

    from menagerie.crawler.identity import durable_fsync

    recorder = BarrierRecorder()
    recorder.install(monkeypatch)

    path = tmp_path / "ledger.jsonl"
    path.write_bytes(b'{"row":1}\n')
    with path.open("ab") as handle:
        handle.write(b'{"row":2}\n')
        handle.flush()
        durable_fsync(handle.fileno())

    observed = recorder.data_barriers(path.stat())
    expected = DEVICE_BARRIER if sys.platform == "darwin" else PAGE_CACHE_ONLY
    assert [event.kind for event in observed] == [expected]
    assert (
        durable_runtime_findings(
            recorder,
            ledger=path,
            module_name="test_durable_fsync_runtime.py",
            function="test_the_shared_barrier_is_a_device_barrier_on_darwin",
        )
        == []
    )


def test_the_runtime_arm_flags_a_directory_only_and_a_premature_barrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D6-1 and the ordering arm, measured: dir-only and barrier-before-write both fail."""

    from menagerie.crawler.identity import durable_fsync, fsync_directory

    path = tmp_path / "records" / "ledger.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)

    directory_only = BarrierRecorder()
    directory_only.install(monkeypatch)
    with path.open("ab") as handle:
        handle.write(b'{"row":1}\n')
        handle.flush()
    fsync_directory(path.parent)
    assert directory_only.directory_barriers(), "the probe issued no directory barrier"
    findings = durable_runtime_findings(
        directory_only,
        ledger=path,
        module_name="test_durable_fsync_runtime.py",
        function="test_the_runtime_arm_flags_a_directory_only_and_a_premature_barrier",
    )
    assert findings and "issued NO data barrier" in findings[0], findings

    premature = BarrierRecorder()
    premature.install(monkeypatch)
    with path.open("ab") as handle:
        durable_fsync(handle.fileno())
        handle.write(b'{"row":2}\n')
        handle.flush()
    findings = durable_runtime_findings(
        premature,
        ledger=path,
        module_name="test_durable_fsync_runtime.py",
        function="test_the_runtime_arm_flags_a_directory_only_and_a_premature_barrier",
    )
    assert findings and "the trailing write was never flushed" in findings[0], findings


def test_an_import_time_alias_is_the_ast_arm_s_job_not_this_one() -> None:
    """The two arms' division of labour is asserted, not assumed.

    ``from os import fsync as _flush`` binds the primitive at IMPORT, before any spy
    exists, so no runtime patch can observe a call through it. That is not a hole:
    it is the one spelling the AST census forbids STRUCTURALLY -- the import itself is
    a site, so a production module cannot even bind the name. Recording the boundary
    here keeps a future reader from "closing" it with a fourth AST pattern or from
    believing the runtime arm covers it.
    """

    from menagerie.crawler.tests.test_durable_fsync_census import bare_fsync_sites

    aliased = (
        "from os import fsync as _flush\n"
        "def append(handle):\n"
        "    _flush(handle.fileno())\n"
    )
    assert bare_fsync_sites(aliased) == [(1, "<module>"), (3, "append")]
