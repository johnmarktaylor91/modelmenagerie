"""RC-1 acceptance: no door module executes on the ``crawl`` path.

Two assertions, in the existing interdiction idiom (``sys.modules``
membership) plus an execution tracer:

* the door modules the crawl path never needs are NOT imported by a crawl
  invocation run in a fresh interpreter (import interdiction);
* no function defined in any door module -- including the ones other kernel
  modules still import at load -- EXECUTES during the invocation (execution
  interdiction, ``sys.setprofile`` on the writer thread).

The door set is the ruling's B.2 delete/collapse column as it stands in this
tree. Modules the kernel still reads at module load for non-door reasons are
listed with the reason and are NOT in the set: ``rung_staging`` (the frozen
lineage registry the B.4 currency carve-out reads), ``rotation_registry`` and
``checker_calibration`` (the reducer projection and the leg-config attestation
consume them; RC-2 owns the former, the checker replacement the latter), and
``close_reads`` (shared ledger-prefix plumbing used by leg_generation and the
paid-session durability reader, not a door).
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

#: Door modules the crawl path must neither import lazily nor execute.
DOOR_MODULES: tuple[str, ...] = (
    "menagerie.crawler.leg_licensing_guard",
    "menagerie.crawler.partition_proof",
    "menagerie.crawler.close_object",
    "menagerie.crawler.close_drift",
    "menagerie.crawler.census_ack",
    "menagerie.crawler.leg_purpose",
    "menagerie.crawler.leg_class",
    "menagerie.crawler.checker_calibration_mint",
    "menagerie.crawler.tools.launch_certificate",
    "menagerie.crawler.tools.backfill_close",
    "menagerie.crawler.tools.close_drift_probe",
    "menagerie.crawler.tools.shadow_close_probe",
)

#: Door modules that a kernel module still imports at LOAD (so they may sit
#: in ``sys.modules``) but whose functions must never run on the crawl path.
LOAD_TIME_ONLY: frozenset[str] = frozenset(
    {
        "menagerie.crawler.close_object",  # driver.py imports the close writer
        "menagerie.crawler.census_ack",  # driver.py imports the ack digests
        "menagerie.crawler.leg_purpose",  # driver.py + cli.py import the vocabulary
        "menagerie.crawler.leg_class",  # cli.py imports required_certificate_kind
        # tools/launch_preflight.py (whose ARM FUNCTIONS the crawl preflight
        # calls) imports the certificate module at load; RC-3 retires both.
        "menagerie.crawler.tools.launch_certificate",
    }
)

#: Named helper functions from door modules the kept kernel path may call:
#: pure path constants with no verdict content. Widening this set is a
#: reviewed change, exactly like the door set itself.
TOLERATED_DOOR_HELPERS: frozenset[str] = frozenset(
    {
        # leg_generation reads the frozen closes ledger to know which
        # generations are closed (the leg_open row is the invocation record);
        # the path helper carries no decision.
        "menagerie.crawler.close_object:close_object_path",
    }
)

_PROBE = r'''
import json, sys, tempfile
from pathlib import Path
from menagerie.crawler.driver import CrawlerDriver
from menagerie.crawler.tests.crawl_support import build_crawl_repo, crawl_driver, plenty_of_disk

doors = json.loads(sys.argv[1])
tmp = Path(tempfile.mkdtemp())
repo_root = build_crawl_repo(tmp, members=4)
CrawlerDriver._crawl_disk_free = staticmethod(plenty_of_disk)
driver = crawl_driver(tmp, repo_root, limit=3)
import os
# Door module SOURCE paths, as the import system spells them in code objects
# (``co_filename``): both the loaded modules' ``__file__`` and the on-disk
# module paths under the package, so a door imported lazily mid-run is
# still recognised without touching the filesystem per call.
package_root = Path(__import__("menagerie.crawler", fromlist=["x"]).__file__).resolve().parent
door_files = {}
for name in doors:
    relative = name.removeprefix("menagerie.crawler.").replace(".", os.sep) + ".py"
    for candidate in (package_root / relative, package_root.parent.parent / "menagerie" / "crawler" / relative):
        door_files[str(candidate)] = name
    module = sys.modules.get(name)
    if module is not None and getattr(module, "__file__", None):
        door_files[str(module.__file__)] = name
        door_files[str(Path(module.__file__).resolve())] = name
executed = set()
def tracer(frame, event, arg):
    if event == "call":
        hit = door_files.get(frame.f_code.co_filename)
        if hit is not None:
            executed.add((hit, frame.f_code.co_name))
    return None
sys.setprofile(tracer)
try:
    result = driver.run()
finally:
    sys.setprofile(None)
# Module bodies and class bodies are IMPORT-time frames (a load-time import
# pulled in by a kernel module), not door decisions; only functions count.
def is_function_frame(module_name, code_name):
    if code_name == "<module>":
        return False
    module = sys.modules.get(module_name)
    attribute = getattr(module, code_name, None) if module is not None else None
    return not isinstance(attribute, type)
print(json.dumps({
    "status": result.status,
    "imported": sorted(name for name in doors if name in sys.modules),
    "executed": sorted(f"{m}:{n}" for m, n in executed if is_function_frame(m, n)),
}))
'''


@pytest.mark.smoke
def test_no_door_module_is_imported_or_executed_on_the_crawl_path() -> None:
    """Fresh interpreter: the crawl runs to work_exhausted touching no door."""

    completed = subprocess.run(
        [sys.executable, "-c", _PROBE, json.dumps(list(DOOR_MODULES))],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["status"] == "work_exhausted"
    lazily_imported = sorted(set(report["imported"]) - LOAD_TIME_ONLY)
    assert lazily_imported == [], f"door modules imported on the crawl path: {lazily_imported}"
    executed = sorted(set(report["executed"]) - TOLERATED_DOOR_HELPERS)
    assert executed == [], f"door functions executed on the crawl path: {executed}"
