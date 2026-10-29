"""LADDER_PROTOCOL rung-binding gate checker.

The ladder protocol binds merged capabilities to rung sizes as staging
preconditions. This tool makes the map machine-checkable:

    python -m menagerie.crawler.tools.rung_gates --next-rung-size 500

Every gate row carries an import-capability probe. Probes fail CLOSED: a
gate whose probe target is absent is UNMET -- there is no way to skip a gate
silently. When a gate's worker lands, its probe target is updated in the same
reviewed change (the probe existing IS the merge evidence).

Size-bound gates (``binds_at_size``) block any next rung of at least that
size; event-bound gates (two-host leg, environment rotation) are always
reported but block only their event, which other tooling owns
(``merge_host_shards`` / ``rotate_setup_artifacts`` refuse on their own).

Exit codes: 0 = every gate binding at the requested size is satisfied;
2 = at least one binding gate is unmet.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class GateSpec:
    """One rung-binding gate and its merge-evidence probe.

    Parameters
    ----------
    gate:
        Stable gate key (matches the LADDER_PROTOCOL row).
    binds_at_size:
        Smallest next-rung size the gate blocks, or ``None`` for event-bound.
    probes:
        Import targets proving the capability is merged: ``module`` or
        ``module:attribute``.
    detail:
        The protocol row, verbatim enough to find the authority.
    """

    gate: str
    binds_at_size: Optional[int]
    probes: tuple[str, ...]
    detail: str


#: The LADDER_PROTOCOL rung-binding map (2026-08-11 JMT direction). A new
#: worker landing updates its row's probes in the same change.
RUNG_GATES: tuple[GateSpec, ...] = (
    GateSpec(
        "stratified-tail",
        200,
        ("menagerie.crawler.rung_stratification:stratify_rung_members",),
        "every rung from 200 onward is stratified (STRAT1)",
    ),
    GateSpec(
        "disk-gc-report-only",
        500,
        ("menagerie.crawler.reaper", "menagerie.crawler.tools.reap_disk"),
        "before 500: disk-GC report-only reaper (TW-B)",
    ),
    GateSpec(
        "ledger-open-perf",
        500,
        ("menagerie.crawler.recordio:LEDGER_OPEN_STATE_SCHEMA_VERSION",),
        "before 500-1000: single-scan ledger open + O(1) append (TW-A)",
    ),
    GateSpec(
        "provider-governance",
        500,
        ("menagerie.crawler.provider_governance",),
        "before 500-1000: jittered ladders, fleet RPM/TPM/spend caps, durable "
        "spend ledger (TW-D; unbuilt -- probe updates when it lands)",
    ),
    GateSpec(
        "observability",
        1000,
        ("menagerie.crawler.observability",),
        "before 1000: typed events, heartbeat, status --live, rung dimension "
        "(TW-C; unbuilt -- probe updates when it lands)",
    ),
    GateSpec(
        "catalog-epoch",
        1000,
        ("menagerie.crawler.catalog_epoch",),
        "before 1000: declared-subset hash, one id authority, schema version "
        "pin (TW-F; JMT ping before landing; probe updates when it lands)",
    ),
    GateSpec(
        "ledger-segmentation",
        2000,
        ("menagerie.crawler.ledger_segmentation",),
        "before 2000-3000: shard segmentation before any ~200MB shard "
        "(own worker; probe updates when it lands)",
    ),
    GateSpec(
        "shard-merge",
        None,
        (
            "menagerie.crawler.tools.merge_host_shards",
            "menagerie.crawler.tools.handoff_bundle",
        ),
        "event gate: before the FIRST two-host leg (TW-G, JMT gate)",
    ),
    GateSpec(
        "rotation-transaction",
        None,
        ("menagerie.crawler.artifact_transactions",),
        "event gate: before ANY next environment rotation (TW-E)",
    ),
)


def _probe(target: str) -> bool:
    """Return whether one ``module`` or ``module:attribute`` target exists."""

    module_name, _, attribute = target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return False
    return not attribute or hasattr(module, attribute)


def evaluate_rung_gates(*, next_rung_size: int) -> list[dict[str, object]]:
    """Evaluate every gate against one planned next rung size.

    Parameters
    ----------
    next_rung_size:
        The rung about to be staged.

    Returns
    -------
    list[dict[str, object]]
        One row per gate: whether it binds at this size, whether its probes
        are satisfied, and whether it blocks this staging.
    """

    rows: list[dict[str, object]] = []
    for spec in RUNG_GATES:
        binds = spec.binds_at_size is not None and next_rung_size >= spec.binds_at_size
        satisfied = all(_probe(target) for target in spec.probes)
        rows.append(
            {
                "gate": spec.gate,
                "binds_at_size": spec.binds_at_size,
                "binds": binds,
                "satisfied": satisfied,
                "blocking": binds and not satisfied,
                "probes": list(spec.probes),
                "detail": spec.detail,
            }
        )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Check the rung-binding gates for one planned rung size.

    Returns
    -------
    int
        Zero when every binding gate is satisfied, 2 otherwise.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--next-rung-size",
        type=int,
        required=True,
        help="the rung size about to be staged",
    )
    args = parser.parse_args(argv)
    rows = evaluate_rung_gates(next_rung_size=args.next_rung_size)
    blocking = [row for row in rows if row["blocking"]]
    print(
        json.dumps(
            {
                "next_rung_size": args.next_rung_size,
                "blocking": [row["gate"] for row in blocking],
                "gates": rows,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if blocking:
        names = ", ".join(str(row["gate"]) for row in blocking)
        print(
            f"rung-gates refused: unmet gates bind at size {args.next_rung_size}: {names}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
