"""Shadow-close verifier (AB-2): per-leg alarm on NEW close-verdict classes.

The close verifier exists (``status --verify-partition`` shares the driver
close's ``assert_partition`` implementation); what 200a lacked was anyone
RUNNING it between legs — stale axes and unaccounted heads surfaced at rung
end, the worst possible latency. This tool consumes one ``status
--verify-partition`` JSON output per leg, extracts the verdict-shaped
classes, diffs them against the durable baseline from the previous leg, and
ALARMS (typed, nonzero) on any NEW class member: a new unaccounted stale id,
a new missing id, a partition flipping invalid, or a new deferral
disposition. Pre-existing disclosed classes never re-alarm (the baseline is
the disclosure); shrinkage updates the baseline silently.

Wiring: the leg runner calls this after every leg close and before staging::

    python -m menagerie.crawler status ... --verify-partition > status.json
    python -m menagerie.crawler.tools.shadow_close_verify \
        --status-json status.json --baseline .crawl-local/shadow-close-baseline.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

BASELINE_SCHEMA_VERSION = "menagerie.crawler.shadow-close-baseline.v1"


class ShadowCloseError(RuntimeError):
    """Raised when a status payload or baseline cannot be consumed."""


def extract_classes(status_payload: Mapping[str, Any]) -> dict[str, list[str]]:
    """Extract the close-verdict class sets from one status JSON payload."""

    if "partition_valid" not in status_payload:
        raise ShadowCloseError(
            "status payload carries no partition verdict; run status with "
            "--verify-partition (a verifier fed a non-verifying status would "
            "silently alarm on nothing)"
        )
    stale = status_payload.get("stale")
    stale_block = stale if isinstance(stale, Mapping) else {}
    deferred = stale_block.get("deferred_for_redrive")
    deferred_map = deferred if isinstance(deferred, Mapping) else {}
    return {
        "partition_invalid": (
            [] if bool(status_payload.get("partition_valid")) else ["partition-invalid"]
        ),
        "missing": sorted(str(item) for item in status_payload.get("missing", [])),
        "unaccounted_stale": sorted(str(item) for item in stale_block.get("unaccounted", [])),
        "deferral_dispositions": sorted(str(key) for key in deferred_map),
    }


def diff_new_members(
    baseline: Mapping[str, Sequence[str]], observed: Mapping[str, Sequence[str]]
) -> dict[str, list[str]]:
    """Return every class member present now that the baseline lacked."""

    alarms: dict[str, list[str]] = {}
    for class_name, members in observed.items():
        known = set(baseline.get(class_name, ()))
        fresh = sorted(set(members) - known)
        if fresh:
            alarms[class_name] = fresh
    return alarms


def verify(status_path: Path, baseline_path: Path) -> tuple[dict[str, list[str]], bool]:
    """Diff one status payload against the baseline; persist the new baseline.

    Returns ``(alarms, had_baseline)``. The first run seeds the baseline and
    DISCLOSES every pre-existing class member as the starting debt without
    alarming (the alarm contract is "NEW after every leg").
    """

    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ShadowCloseError(f"unreadable status payload {status_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ShadowCloseError(f"{status_path} is not a JSON object")
    observed = extract_classes(payload)
    had_baseline = baseline_path.is_file()
    baseline: Mapping[str, Any] = {}
    if had_baseline:
        stored = json.loads(baseline_path.read_text(encoding="utf-8"))
        if (
            not isinstance(stored, Mapping)
            or stored.get("schema_version") != BASELINE_SCHEMA_VERSION
        ):
            raise ShadowCloseError(
                f"{baseline_path} is not a shadow-close baseline artifact; refusing "
                "to diff against foreign bytes"
            )
        baseline = stored.get("classes", {})
    alarms = diff_new_members(baseline, observed) if had_baseline else {}
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = baseline_path.with_name(f".{baseline_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            {"schema_version": BASELINE_SCHEMA_VERSION, "classes": observed},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, baseline_path)
    return alarms, had_baseline


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run one shadow-close verification pass."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args(argv)
    alarms, had_baseline = verify(args.status_json, args.baseline)
    if not had_baseline:
        print("shadow-close: baseline seeded; pre-existing classes disclosed, not alarmed")
        return 0
    if alarms:
        for class_name, members in sorted(alarms.items()):
            print(
                f"shadow-close ALARM: new {class_name} members since last leg: {members}",
                file=sys.stderr,
            )
        return 2
    print("shadow-close: no new close-verdict classes since last leg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
