"""PM-6/W20 (+AB-1): the machine-readable audit register + regression pins.

The failure-pattern catalog's 42 rows become a committed, machine-checked
register. Vocabulary (closed, PLAN §4.2's register bullet): FINDING rows
carry exactly one of ``FIXED`` (sha + >=1 tripwire, every pytest tripwire
resolving in the committed smoke inventory lock), ``DEFERRED`` (+ firing
trigger), or ``REJECTED`` (+ reason); ITEM-RESOLUTION rows carry
``BLOCKED-ON-JMT-SIGNATURE`` (bound receipt) or ``DEAD-WITH-EVIDENCE``
(bound census entry) and are disclosed operationally-resolved, never counted
as fixed. A **GAP** is a row lacking a lawful closed-vocabulary disposition;
the launch preflight's ``audit-register`` arm refuses launch while any
exists (PM-Q10: zero catalog GAPs joins the launch bar).

Rows whose only tripwires are ``process`` artifacts (off-tree, private by
the internal-notes policy) are lawful but SEGREGATED as process-enforced —
disclosed, never counted as machine-tripwired. ``external-suite`` tripwires
live in the tools/ test suite outside the crawler corpus lock; their files
are existence-checked.

The regression-pin registry (PM-Q7: machine-readable; O: covers every
leg-death class) maps every code failure class C-01..C-30 to collected
tripwire nodes. Never-lower is a web, not a counter: every register FIXED
code row's pytest tripwires must stay a subset of its pin entry, every pin
node must resolve in the inventory lock (deleting the test breaks it), and
every class must stay covered.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

AUDIT_REGISTER_SCHEMA_VERSION = "menagerie.crawler.audit-register.v1"
REGRESSION_PINS_SCHEMA_VERSION = "menagerie.crawler.regression-pins.v1"

AUDIT_REGISTER_PATH = Path(__file__).with_name("audit_register.json")
REGRESSION_PINS_PATH = Path(__file__).resolve().parents[1] / "tests" / "regression_pins.json"

FINDING_DISPOSITIONS = frozenset({"FIXED", "DEFERRED", "REJECTED"})
ITEM_RESOLUTION_DISPOSITIONS = frozenset({"BLOCKED-ON-JMT-SIGNATURE", "DEAD-WITH-EVIDENCE"})
TRIPWIRE_KINDS = frozenset({"pytest-node", "tool", "process", "external-suite"})


class AuditRegisterError(RuntimeError):
    """Raised when the register or pin registry cannot be parsed."""


def load_register(path: Path) -> dict[str, Any]:
    """Load one audit register, refusing prior-era or malformed artifacts."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditRegisterError(f"audit register unreadable: {path}: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != AUDIT_REGISTER_SCHEMA_VERSION
        or not isinstance(payload.get("rows"), list)
        or not isinstance(payload.get("staleness"), dict)
    ):
        raise AuditRegisterError(
            f"audit register {path} does not carry schema "
            f"{AUDIT_REGISTER_SCHEMA_VERSION} with rows + staleness header"
        )
    return payload


def resolve_pytest_ref(ref: str, universe: Iterable[str]) -> list[str]:
    """Return every collected node one register/pin ref resolves to.

    Accepts exact node ids, parametrized variants, whole-module refs
    (``module.py``), and class-hosted functions (``module.py::test_x``
    resolving ``module.py::Class::test_x``).
    """

    nodes = list(universe)
    if ref in nodes:
        return [ref]
    hits = [node for node in nodes if node.startswith(ref + "[")]
    if hits:
        return hits
    if ref.endswith(".py"):
        return [node for node in nodes if node.startswith(ref + "::")]
    if "::" in ref:
        module, function = ref.rsplit("::", 1)
        return [
            node
            for node in nodes
            if node.startswith(module + "::")
            and node.split("[", 1)[0].rsplit("::", 1)[-1] == function
        ]
    return []


def _evaluate_fixed(
    row_id: str,
    disposition: Mapping[str, Any],
    *,
    collected_nodes: frozenset[str],
    repo_root: Path,
    refusals: list[str],
) -> str:
    """Validate one FIXED disposition; return its enforcement class."""

    tripwires = disposition.get("tripwires")
    if not str(disposition.get("sha") or "").strip():
        refusals.append(f"{row_id}: FIXED requires its verification sha (FIXED@sha)")
    if not isinstance(tripwires, list) or not tripwires:
        refusals.append(f"{row_id}: FIXED requires at least one landed tripwire")
        return "invalid"
    machine_backed = False
    for tripwire in tripwires:
        kind = tripwire.get("kind") if isinstance(tripwire, Mapping) else None
        ref = str(tripwire.get("ref", "")) if isinstance(tripwire, Mapping) else ""
        if kind not in TRIPWIRE_KINDS or not ref:
            refusals.append(f"{row_id}: tripwire kind {kind!r} outside {sorted(TRIPWIRE_KINDS)}")
            continue
        if kind == "pytest-node":
            if resolve_pytest_ref(ref, collected_nodes):
                machine_backed = True
            else:
                refusals.append(
                    f"{row_id}: pytest tripwire {ref!r} does not resolve in the "
                    "committed smoke inventory lock (a falsely-FIXED row is worse "
                    "than an honest GAP)"
                )
        elif kind in {"tool", "external-suite"}:
            path = ref.split("::", 1)[0]
            if (repo_root / path).is_file():
                machine_backed = True
            else:
                refusals.append(f"{row_id}: {kind} tripwire {path!r} is absent from the tree")
        # kind == "process": lawful, disclosed, never machine-backed.
    return "machine" if machine_backed else "process"


def evaluate_register(
    register: Mapping[str, Any],
    *,
    collected_nodes: frozenset[str],
    repo_root: Path,
) -> dict[str, Any]:
    """Evaluate GAP semantics + tripwire machine checks over one register."""

    refusals: list[str] = []
    gaps: list[str] = []
    deferred: list[str] = []
    rejected: list[str] = []
    process_enforced: list[str] = []
    operationally_resolved: list[str] = []
    fixed: list[str] = []
    seen: set[str] = set()

    for row in register["rows"]:
        row_id = str(row.get("row_id", ""))
        if not row_id or row_id in seen:
            refusals.append(f"register row without a unique row_id: {row_id!r}")
            continue
        seen.add(row_id)
        kind = row.get("kind")
        disposition = row.get("disposition")
        if not isinstance(disposition, Mapping) or not disposition.get("kind"):
            gaps.append(row_id)
            continue
        d_kind = str(disposition["kind"])
        if kind == "item-resolution":
            if d_kind not in ITEM_RESOLUTION_DISPOSITIONS:
                refusals.append(
                    f"{row_id}: item-resolution disposition {d_kind!r} outside the "
                    f"closed set {sorted(ITEM_RESOLUTION_DISPOSITIONS)}"
                )
                continue
            binding = "receipt" if d_kind == "BLOCKED-ON-JMT-SIGNATURE" else "census_entry"
            if not str(disposition.get(binding) or "").strip():
                refusals.append(f"{row_id}: {d_kind} requires its bound {binding}")
                continue
            operationally_resolved.append(row_id)
            continue
        if d_kind not in FINDING_DISPOSITIONS:
            refusals.append(
                f"{row_id}: finding disposition {d_kind!r} outside the closed set "
                f"{sorted(FINDING_DISPOSITIONS)} (lanes may write only these three)"
            )
            continue
        if d_kind == "FIXED":
            enforcement = _evaluate_fixed(
                row_id,
                disposition,
                collected_nodes=collected_nodes,
                repo_root=repo_root,
                refusals=refusals,
            )
            if enforcement == "machine":
                fixed.append(row_id)
            elif enforcement == "process":
                process_enforced.append(row_id)
        elif d_kind == "DEFERRED":
            if not str(disposition.get("trigger") or "").strip():
                refusals.append(f"{row_id}: DEFERRED requires its firing trigger (P-12)")
            else:
                deferred.append(row_id)
        else:  # REJECTED
            if not str(disposition.get("reason") or "").strip():
                refusals.append(f"{row_id}: REJECTED requires its reason")
            else:
                rejected.append(row_id)

    status = "passed" if not refusals and not gaps else "failed"
    return {
        "schema_version": AUDIT_REGISTER_SCHEMA_VERSION,
        "status": status,
        "rows_total": len(register["rows"]),
        "staleness": dict(register["staleness"]),
        "fixed_machine_checked": sorted(fixed),
        "deferred": sorted(deferred),
        "rejected": sorted(rejected),
        "process_enforced": sorted(process_enforced),
        "operationally_resolved": sorted(operationally_resolved),
        "gaps": sorted(gaps),
        "refusals": refusals,
    }


def load_regression_pins(path: Path) -> dict[str, Any]:
    """Load the regression-pin registry, refusing malformed artifacts."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditRegisterError(f"regression-pin registry unreadable: {path}: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != REGRESSION_PINS_SCHEMA_VERSION
        or not isinstance(payload.get("pins"), dict)
    ):
        raise AuditRegisterError(
            f"regression-pin registry {path} does not carry schema "
            f"{REGRESSION_PINS_SCHEMA_VERSION} with a pins mapping"
        )
    return payload


def evaluate_regression_pins(
    pins: Mapping[str, Any],
    register: Mapping[str, Any],
    *,
    collected_nodes: frozenset[str],
    smoke_nodes: frozenset[str],
) -> dict[str, Any]:
    """Evaluate the covers-every-leg-death + never-lower pin web."""

    refusals: list[str] = []
    unresolvable: list[str] = []
    pin_rows: Mapping[str, Any] = pins["pins"]

    code_rows = [
        row
        for row in register["rows"]
        if str(row.get("row_id", "")).startswith("C-") and row.get("kind") == "finding"
    ]
    uncovered = sorted(
        str(row["row_id"])
        for row in code_rows
        if str(row["row_id"]) not in pin_rows
        or not (pin_rows[str(row["row_id"])].get("nodes") or [])
    )
    for incident_id, entry in sorted(pin_rows.items()):
        nodes = entry.get("nodes") or []
        for node in nodes:
            if not resolve_pytest_ref(str(node), collected_nodes):
                unresolvable.append(str(node))
    # Never-lower: every FIXED code row's pytest tripwires stay pinned; every
    # DEFERRED code row's partial tripwires too (they are the standing floor).
    for row in code_rows:
        disposition = row.get("disposition") or {}
        tripwires = (
            disposition.get("tripwires")
            if disposition.get("kind") == "FIXED"
            else disposition.get("partial_tripwires")
        ) or []
        register_nodes = {
            str(t["ref"]) for t in tripwires if isinstance(t, Mapping) and t.get("kind") == "pytest-node"
        }
        pinned = set(map(str, (pin_rows.get(str(row["row_id"]), {}).get("nodes") or [])))
        missing = sorted(register_nodes - pinned)
        if missing:
            refusals.append(
                f"never-lower: {row['row_id']} register tripwires missing from the "
                f"pin registry: {missing}"
            )
    non_blocking = sorted(
        incident_id
        for incident_id, entry in pin_rows.items()
        if not any(resolve_pytest_ref(str(node), smoke_nodes) for node in entry.get("nodes") or [])
    )
    status = "passed" if not refusals and not uncovered and not unresolvable else "failed"
    return {
        "schema_version": REGRESSION_PINS_SCHEMA_VERSION,
        "status": status,
        "covered_incidents": sorted(set(pin_rows) & {str(r["row_id"]) for r in code_rows}),
        "uncovered_incidents": uncovered,
        "unresolvable_nodes": sorted(set(unresolvable)),
        "outside_blocking_gate": non_blocking,
        "refusals": refusals,
    }


def all_pin_nodes(pins: Mapping[str, Any]) -> list[str]:
    """Return the deduplicated union of every pinned node ref."""

    seen: set[str] = set()
    ordered: list[str] = []
    for entry in pins["pins"].values():
        for node in entry.get("nodes") or []:
            if str(node) not in seen:
                seen.add(str(node))
                ordered.append(str(node))
    return ordered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--register", type=Path, default=AUDIT_REGISTER_PATH)
    parser.add_argument("--pins", type=Path, default=REGRESSION_PINS_PATH)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    from menagerie.crawler.tools import collection_gate

    repo_root = args.repo_root.resolve()
    try:
        # RC-3: the node universe is a LIVE collection (the exact-inventory
        # lock retired); an uncollectable corpus refuses the census outright.
        collected = collection_gate.collect(repo_root, python=sys.executable)
        if not collected.ok:
            raise AuditRegisterError("corpus does not collect: " + "; ".join(collected.failures))
        smoke = collection_gate.collect(
            repo_root,
            python=sys.executable,
            tracked_modules=collected.tracked_modules,
            marker="smoke",
        )
        register = load_register(args.register)
        pins = load_regression_pins(args.pins)
    except (AuditRegisterError, Exception) as exc:  # noqa: BLE001 - CLI boundary
        print(f"audit-register failed: {exc}", file=sys.stderr)
        return 1
    register_census = evaluate_register(
        register, collected_nodes=frozenset(collected.node_ids), repo_root=repo_root
    )
    pin_census = evaluate_regression_pins(
        pins,
        register,
        collected_nodes=frozenset(collected.node_ids),
        smoke_nodes=frozenset(smoke.node_ids),
    )
    print(json.dumps({"register": register_census, "pins": pin_census}, sort_keys=True))
    return 0 if register_census["status"] == "passed" and pin_census["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
