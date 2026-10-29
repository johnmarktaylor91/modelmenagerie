"""OB-M1 census: ONE machine-deferral valve registry, every consumer delegating.

At main 289445e6 four hand-maintained valve enumerations disagreed with the
partition authority (``status.MACHINE_DEFERRAL_DISPOSITIONS``) in four distinct
ways: ``launch_monitor.PARK_LEDGER_RELATIVE_PATHS`` omitted
``environment-rebuild-deferrals.jsonl`` (m10502 -- a live rebuild-valve claim --
was invisible to both completion detectors, so the monitor could never read the
campaign complete and never counted the member blocked);
``rung_stratification._valve_paths`` omitted ``scheduler-quarantines.jsonl``
(rung_report under-counted valve claims at a close boundary);
``scheduler_shadow_audit`` loaded five of the eight ledgers. This census makes
that drift class unrepresentable: the registry's dispositions must equal the
partition authority, its paths must equal the owning modules' constants, every
consumer must derive from it, and no production module may re-spell a valve
ledger filename as a literal.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from menagerie.crawler import status
from menagerie.crawler.quota_deferral import QUOTA_DEFERRAL_RELATIVE_PATH
from menagerie.crawler.checker_lane import CHECKER_LANE_DEFERRAL_RELATIVE_PATH
from menagerie.crawler.disk_floor import DISK_FLOOR_DEFERRAL_RELATIVE_PATH
from menagerie.crawler.environment_coverage import COVERAGE_DEFERRAL_RELATIVE_PATH
from menagerie.crawler.environment_rebuild import REBUILD_DEFERRAL_RELATIVE_PATH
from menagerie.crawler.grant_parks import GRANT_PARK_RELATIVE_PATH
from menagerie.crawler.host_capacity import CAPACITY_DEFERRAL_RELATIVE_PATH
from menagerie.crawler.refusal_parks import REFUSAL_PARK_RELATIVE_PATH
from menagerie.crawler.scheduler_parks import SCHEDULER_QUARANTINE_RELATIVE_PATH

pytestmark = pytest.mark.smoke

CRAWLER_ROOT = Path(__file__).resolve().parents[1]

#: Every intake-extensions ledger constant that is deliberately NOT a
#: machine-deferral valve. A new intake-extensions ledger must either join the
#: registry or be named here in the same reviewed change -- never drift silently.
NON_VALVE_INTAKE_EXTENSION_LEDGERS = frozenset(
    {
        "c3-classics.jsonl",
        "membership-acknowledgments.jsonl",
        # W14 seam records: typed prerequisite naming for the release engine,
        # never a partition-accounting valve row (no member withholding).
        "blocked-prerequisites.jsonl",
        # AG-4 disclosure ledger (checker_conflicts.py): verdict conflicts,
        # identity-attributed flips, and duplicate-accept confirmations. The
        # PARK a conflict fires rides the existing checker-lane valve as
        # failure class ``checker-verdict-conflict``, so this ledger is
        # evidence, never a deferral valve with its own disposition.
        "checker-verdict-conflicts.jsonl",
    }
)

#: The owning module per valve ledger filename: the ONLY production sources
#: allowed to spell the filename as a literal (their ``*_RELATIVE_PATH``
#: definition). Everything else must import the registry or the constant.
VALVE_LEDGER_OWNERS = {
    "environment-coverage-deferrals.jsonl": "environment_coverage.py",
    "host-capacity-deferrals.jsonl": "host_capacity.py",
    "environment-rebuild-deferrals.jsonl": "environment_rebuild.py",
    "disk-floor-deferrals.jsonl": "disk_floor.py",
    "checker-lane-deferrals.jsonl": "checker_lane.py",
    "grant-pending-supersessions.jsonl": "grant_parks.py",
    "terminal-refusal-parks.jsonl": "refusal_parks.py",
    "scheduler-quarantines.jsonl": "scheduler_parks.py",
    "quota-deferrals.jsonl": "quota_deferral.py",
}


def _production_sources() -> list[Path]:
    return [
        path
        for path in sorted(CRAWLER_ROOT.rglob("*.py"))
        if "tests" not in path.relative_to(CRAWLER_ROOT).parts
    ]


def test_registry_dispositions_equal_the_partition_authority() -> None:
    union: frozenset[str] = frozenset()
    for valve in status.MACHINE_DEFERRAL_VALVES:
        assert valve.dispositions, f"{valve.short_name} declares no disposition"
        union |= valve.dispositions
    assert union == status.MACHINE_DEFERRAL_DISPOSITIONS
    names = [valve.short_name for valve in status.MACHINE_DEFERRAL_VALVES]
    assert len(names) == len(set(names)), "duplicate valve short names"


def test_registry_paths_are_the_owning_modules_constants() -> None:
    expected = {
        "environment-coverage": COVERAGE_DEFERRAL_RELATIVE_PATH.as_posix(),
        "host-capacity": CAPACITY_DEFERRAL_RELATIVE_PATH.as_posix(),
        "environment-rebuild": REBUILD_DEFERRAL_RELATIVE_PATH.as_posix(),
        "disk-floor": DISK_FLOOR_DEFERRAL_RELATIVE_PATH.as_posix(),
        "checker-lane": CHECKER_LANE_DEFERRAL_RELATIVE_PATH.as_posix(),
        "grant-parks": GRANT_PARK_RELATIVE_PATH.as_posix(),
        "refusal-parks": REFUSAL_PARK_RELATIVE_PATH.as_posix(),
        "scheduler-quarantines": SCHEDULER_QUARANTINE_RELATIVE_PATH.as_posix(),
        "quota-deferrals": QUOTA_DEFERRAL_RELATIVE_PATH.as_posix(),
    }
    observed = {
        valve.short_name: valve.relative_path
        for valve in status.MACHINE_DEFERRAL_VALVES
    }
    assert observed == expected


def test_every_intake_extension_ledger_is_registered_or_named_non_valve() -> None:
    constant = re.compile(
        r'^[A-Z0-9_]+_RELATIVE_PATH\s*=\s*Path\("intake-extensions"\)\s*/\s*"([^"]+)"',
        re.MULTILINE,
    )
    found: set[str] = set()
    for path in _production_sources():
        found |= set(constant.findall(path.read_text(encoding="utf-8")))
    registered = {
        Path(valve.relative_path).name for valve in status.MACHINE_DEFERRAL_VALVES
    }
    unaccounted = found - registered - NON_VALVE_INTAKE_EXTENSION_LEDGERS
    assert not unaccounted, (
        "intake-extensions ledger(s) neither registered as machine-deferral"
        f" valves nor named non-valve: {sorted(unaccounted)}"
    )
    assert registered <= found, (
        "registry names a ledger no production module defines:"
        f" {sorted(registered - found)}"
    )


def test_launch_monitor_derives_from_the_registry() -> None:
    from menagerie.crawler.tools import launch_monitor

    assert launch_monitor.PARK_LEDGER_RELATIVE_PATHS == tuple(
        valve.relative_path for valve in status.MACHINE_DEFERRAL_VALVES
    )
    # The m10502 pin: the rebuild valve is what both completion detectors missed.
    assert (
        REBUILD_DEFERRAL_RELATIVE_PATH.as_posix()
        in launch_monitor.PARK_LEDGER_RELATIVE_PATHS
    )


def test_rung_stratification_derives_from_the_registry(tmp_path: Path) -> None:
    from menagerie.crawler.rung_stratification import _valve_paths

    assert _valve_paths(tmp_path) == {
        valve.short_name: tmp_path / valve.relative_path
        for valve in status.MACHINE_DEFERRAL_VALVES
    }
    assert "scheduler-quarantines" in _valve_paths(tmp_path)


def test_scheduler_shadow_audit_derives_from_the_registry() -> None:
    source = (CRAWLER_ROOT / "tools" / "scheduler_shadow_audit.py").read_text(
        encoding="utf-8"
    )
    assert "MACHINE_DEFERRAL_VALVES" in source
    for accessor in (
        "rebuild_deferral_path",
        "disk_floor_deferral_path",
        "checker_lane_deferral_path",
        "grant_park_path",
        "refusal_park_path",
        "capacity_deferral_path",
        "coverage_deferral_path",
        "scheduler_quarantine_path",
    ):
        assert accessor not in source, (
            f"scheduler_shadow_audit hand-picks valve ledgers via {accessor};"
            " it must iterate status.MACHINE_DEFERRAL_VALVES"
        )


def test_registry_loaders_are_the_owning_modules_validated_readers() -> None:
    from menagerie.crawler.checker_lane import load_checker_lane_deferral_rows
    from menagerie.crawler.disk_floor import load_disk_floor_deferral_rows
    from menagerie.crawler.environment_coverage import load_coverage_deferral_rows
    from menagerie.crawler.environment_rebuild import load_rebuild_deferral_rows
    from menagerie.crawler.grant_parks import load_grant_park_rows
    from menagerie.crawler.host_capacity import load_capacity_deferral_rows
    from menagerie.crawler.refusal_parks import load_refusal_park_rows
    from menagerie.crawler.scheduler_parks import load_scheduler_quarantine_rows

    expected = {
        "environment-coverage": load_coverage_deferral_rows,
        "host-capacity": load_capacity_deferral_rows,
        "environment-rebuild": load_rebuild_deferral_rows,
        "disk-floor": load_disk_floor_deferral_rows,
        "checker-lane": load_checker_lane_deferral_rows,
        "grant-parks": load_grant_park_rows,
        "refusal-parks": load_refusal_park_rows,
        "scheduler-quarantines": load_scheduler_quarantine_rows,
        "quota-deferrals": status._load_quota_deferral_valve_rows,
    }
    observed = {
        valve.short_name: valve.load_rows for valve in status.MACHINE_DEFERRAL_VALVES
    }
    assert observed == expected


def test_valve_family_split_follows_disposition_kind() -> None:
    families = {
        valve.short_name: valve.family for valve in status.MACHINE_DEFERRAL_VALVES
    }
    assert families == {
        "environment-coverage": "deferral",
        "host-capacity": "deferral",
        "environment-rebuild": "deferral",
        "disk-floor": "deferral",
        "checker-lane": "deferral",
        "grant-parks": "park",
        "refusal-parks": "park",
        "scheduler-quarantines": "deferral",
        # Mixed by design: the storm valve is a deferral; the PD-3 recidivist
        # row is review-held, so the FAMILY stays deferral (not all-park).
        "quota-deferrals": "deferral",
    }


def test_no_valve_ledger_filename_literal_outside_owning_modules() -> None:
    offenders: list[str] = []
    for path in _production_sources():
        text = path.read_text(encoding="utf-8")
        for filename, owner in VALVE_LEDGER_OWNERS.items():
            if filename in text and path.name != owner:
                offenders.append(f"{path.relative_to(CRAWLER_ROOT)}: {filename}")
    assert not offenders, (
        "valve ledger filename literal(s) outside their owning modules"
        f" (import status.MACHINE_DEFERRAL_VALVES instead): {offenders}"
    )
