"""Blocking-item resolution schema + BLOCKING_TIERS completeness census.

ONE closed, machine-readable resolution vocabulary over RESOLUTION UNITS,
consumed identically by every consumer of the blocking-item-resolution fact
(merge-queue completeness ledger, candidate freeze, this census as a
certificate input, and the P5 entry condition).

The unit inventory is REPLAY-DERIVED on every evaluation from the tagged
blocking-item inventory joined to the closed signature-arm table — never from
a producer-supplied unit list. A unit not derivable from those two sources
refuses as producer-created. No persisted status field steers anything.

Closed tier vocabulary: exactly ``{"200B-BLOCKING", "200B-BLOCKING-EFFICIENCY"}``.
An item tagged with any other blocking-tier spelling refuses: a newly invented
tier cannot escape wave placement or census coverage.

LEGAL RESOLUTIONS (each unit has EXACTLY ONE):
- ORDINARY / BASE unit: ``merged`` (sha + test node; a composed item's merge
  receipt BINDS its exact included-arm set) or — efficiency tier only —
  ``dead_with_evidence``; or — ONLY when the item's entire substance is one
  signature arm — ``c8_unsigned_negative``.
- ARM unit: membership in its item's merge receipt's included-arm set (lawful
  only if the arm was SIGNED at the seal) XOR its own ``c8_unsigned_negative``
  receipt. Both at once refuses as a duplicate resolution.
- A ``c8_unsigned_negative`` receipt is valid ONLY for an arm that was
  unsigned at the deadline, carries its bound deadline event, audit-register
  row, and every specified dependent exclusion, and — the NON-VACUITY rule —
  whose arm-table row states an explicit negative disposition. A row with no
  stated disposition can never be discharged by this receipt.

Units resolved by ``dead_with_evidence`` or ``c8_unsigned_negative`` are
OPERATIONALLY RESOLVED: disclosed, segregated, never counted as merged.

FIRST-GENERATION surface: this vocabulary and its canonical resolution
envelope are created in the same reviewed change as their first consumer.
Every pre-schema resolution-shaped artifact refuses typed AS an envelope
input rather than parses (no downgrade ladder); legacy merge or
dead-with-evidence bookkeeping is admitted only as authenticated INPUT rows
inside a newly minted canonical envelope, never AS an envelope.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ENVELOPE_SCHEMA_VERSION = "menagerie.crawler.blocking-resolution-envelope.v1"
CENSUS_SCHEMA_VERSION = "menagerie.crawler.blocking-tiers-census.v1"

#: The closed, machine-readable launch-gating tier set. This ONE set is the
#: whole vocabulary; any other blocking-tier spelling refuses.
BLOCKING_TIERS = frozenset({"200B-BLOCKING", "200B-BLOCKING-EFFICIENCY"})

#: Closed enumeration of FIRST-GENERATION surfaces (membership closure is
#: machine-checked by the census RED test): a later change self-declaring
#: first-generation status fails the census and goes to the STOP-and-file path.
FIRST_GENERATION_SURFACES = frozenset(
    {
        "certificate-kind-vocabulary",
        "per-kind-required-input-manifest",
        "blocking-tiers-census-membership",
        "blocking-item-resolution-vocabulary",
    }
)

_EVIDENCE_KINDS = frozenset({"merged", "dead_with_evidence", "c8_unsigned_negative"})


class ResolutionSchemaError(RuntimeError):
    """Raised when an envelope, inventory, arm table, or evidence row refuses."""


def load_envelope(path: Path) -> dict[str, Any]:
    """Load one canonical resolution envelope, refusing prior-era artifacts.

    Raises
    ------
    ResolutionSchemaError
        For any artifact without the exact envelope schema version: a
        pre-schema resolution-shaped artifact refuses typed AS an envelope
        input rather than parses. No downgrade ladder exists.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResolutionSchemaError(f"resolution-envelope-invalid: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResolutionSchemaError(f"resolution-envelope-invalid: {path} is not an object")
    if payload.get("schema_version") != ENVELOPE_SCHEMA_VERSION:
        raise ResolutionSchemaError(
            "resolution-envelope-legacy: "
            f"{path} carries schema {payload.get('schema_version')!r}; every "
            "pre-schema resolution-shaped artifact refuses typed as an envelope "
            "input — mint a canonical envelope from authenticated inputs instead"
        )
    for key in ("items", "arm_table", "evidence"):
        if not isinstance(payload.get(key), list):
            raise ResolutionSchemaError(
                f"resolution-envelope-invalid: {path} lacks list field {key!r}"
            )
    return payload


def derive_units(
    items: Sequence[Mapping[str, Any]], arm_table: Sequence[Mapping[str, Any]]
) -> dict[tuple[str, str], dict[str, Any]]:
    """Replay-derive the closed unit inventory from items joined to arms.

    Returns
    -------
    dict
        ``(item_id, unit_key) -> unit`` where ``unit_key`` is ``"base"`` for
        ordinary/base units and the ``arm_id`` for arm units.
    """

    arms_by_item: dict[str, list[dict[str, Any]]] = {}
    for row in arm_table:
        if not isinstance(row, Mapping) or not row.get("item_id") or not row.get("arm_id"):
            raise ResolutionSchemaError(
                "resolution-inventory-invalid: arm-table row lacks item_id/arm_id"
            )
        if row.get("produces_unit") is False:
            # A C.8 row gating a disposition or non-blocking work produces NO
            # resolution unit; its unsigned fallback is a dossier act.
            continue
        arms_by_item.setdefault(str(row["item_id"]), []).append(dict(row))

    units: dict[tuple[str, str], dict[str, Any]] = {}
    item_ids: set[str] = set()
    for item in items:
        if not isinstance(item, Mapping) or not item.get("item_id"):
            raise ResolutionSchemaError("resolution-inventory-invalid: item lacks item_id")
        item_id = str(item["item_id"])
        if item_id in item_ids:
            raise ResolutionSchemaError(f"resolution-inventory-invalid: duplicate item {item_id!r}")
        item_ids.add(item_id)
        tier = item.get("tier")
        if tier not in BLOCKING_TIERS:
            raise ResolutionSchemaError(
                f"tier-vocabulary-refused: item {item_id!r} carries blocking-tier "
                f"spelling {tier!r} outside the closed set {sorted(BLOCKING_TIERS)}"
            )
        arms = arms_by_item.get(item_id, [])
        entire_substance = bool(item.get("entire_substance_is_arm"))
        if not arms or (len(arms) == 1 and entire_substance):
            units[(item_id, "base")] = {
                "item_id": item_id,
                "unit": "base",
                "tier": tier,
                "ordinary": True,
                "entire_substance_arm": arms[0] if entire_substance and arms else None,
            }
        else:
            units[(item_id, "base")] = {
                "item_id": item_id,
                "unit": "base",
                "tier": tier,
                "ordinary": False,
                "entire_substance_arm": None,
            }
            for arm in arms:
                units[(item_id, str(arm["arm_id"]))] = {
                    "item_id": item_id,
                    "unit": str(arm["arm_id"]),
                    "tier": tier,
                    "ordinary": False,
                    "arm": arm,
                }
    return units


def _valid_c8_receipt(row: Mapping[str, Any], arm: Mapping[str, Any] | None) -> str | None:
    """Return a refusal reason for a c8_unsigned_negative receipt, or None."""

    if arm is None:
        return "c8 receipt for a unit with no signature arm"
    if arm.get("signed_at_seal"):
        return "c8 receipt for an arm whose signature was standing at the seal"
    if not str(arm.get("stated_negative_disposition") or "").strip():
        return (
            "non-vacuity refusal: the arm's deadline block states no explicit "
            "negative disposition, so this receipt can never discharge it"
        )
    if not row.get("deadline_event"):
        return "c8 receipt lacks its bound deadline event"
    if not row.get("audit_register_row"):
        return "c8 receipt lacks its BLOCKED-ON-JMT-SIGNATURE audit-register row"
    required = [str(value) for value in arm.get("dependent_exclusions", [])]
    checked = {str(value) for value in row.get("dependent_exclusions_present", [])}
    missing = [value for value in required if value not in checked]
    if missing:
        return f"c8 receipt missing specified dependent exclusions: {missing}"
    return None


def evaluate_census(envelope: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the BLOCKING_TIERS completeness census over one envelope.

    The census is green iff every replay-derived resolution unit of every
    blocking-tier item has exactly one legal resolution. Operationally
    resolved rows are disclosed and segregated, never counted as merged.
    """

    units = derive_units(envelope["items"], envelope["arm_table"])
    arms_index = {
        (str(row["item_id"]), str(row["arm_id"])): dict(row)
        for row in envelope["arm_table"]
        if isinstance(row, Mapping) and row.get("item_id") and row.get("arm_id")
    }

    resolutions: dict[tuple[str, str], list[dict[str, Any]]] = {key: [] for key in units}
    refusals: list[str] = []

    included_arm_keys: set[tuple[str, str]] = set()
    for row in envelope["evidence"]:
        if isinstance(row, Mapping) and row.get("kind") == "merged":
            for arm_id in row.get("included_arms", []):
                included_arm_keys.add((str(row.get("item_id", "")), str(arm_id)))

    for row in envelope["evidence"]:
        if not isinstance(row, Mapping) or row.get("kind") not in _EVIDENCE_KINDS:
            refusals.append(
                f"evidence-refused: row kind {row.get('kind')!r} outside the closed "
                f"vocabulary {sorted(_EVIDENCE_KINDS)} (a producer-supplied status "
                "cannot steer the census)"
            )
            continue
        item_id = str(row.get("item_id", ""))
        kind = row["kind"]
        if kind == "merged":
            key = (item_id, "base")
            if key not in units:
                refusals.append(f"unit-refused: merge receipt names producer-created unit {key}")
                continue
            if not row.get("sha") or not row.get("test_node"):
                refusals.append(
                    f"evidence-refused: merge receipt for {item_id!r} lacks sha/test_node"
                )
                continue
            included = [str(value) for value in row.get("included_arms", [])]
            arm_refused = False
            for arm_id in included:
                arm_key = (item_id, arm_id)
                if arm_key not in units:
                    refusals.append(
                        f"unit-refused: merge receipt binds producer-created arm {arm_key}"
                    )
                    arm_refused = True
                    continue
                arm = arms_index.get(arm_key)
                if arm is None or not arm.get("signed_at_seal"):
                    refusals.append(
                        f"evidence-refused: merge receipt for {item_id!r} claims arm "
                        f"{arm_id!r} whose signature was absent at the seal"
                    )
                    arm_refused = True
                    continue
                resolutions[arm_key].append({"resolution": "merged-arm", "row": dict(row)})
            if arm_refused:
                continue
            resolutions[key].append({"resolution": "merged", "row": dict(row)})
        elif kind == "dead_with_evidence":
            key = (item_id, "base")
            unit = units.get(key)
            if unit is None:
                refusals.append(
                    f"unit-refused: dead-with-evidence names producer-created unit {key}"
                )
                continue
            if unit["tier"] != "200B-BLOCKING-EFFICIENCY":
                refusals.append(
                    f"evidence-refused: dead-with-evidence exit is efficiency-tier "
                    f"only; {item_id!r} is {unit['tier']}"
                )
                continue
            if not row.get("census_entry"):
                refusals.append(
                    f"evidence-refused: dead-with-evidence for {item_id!r} lacks its "
                    "disclosed census entry"
                )
                continue
            resolutions[key].append({"resolution": "dead_with_evidence", "row": dict(row)})
        else:  # c8_unsigned_negative
            arm_id = str(row.get("arm_id", ""))
            key = (item_id, arm_id or "base")
            unit = units.get(key)
            if unit is None:
                refusals.append(f"unit-refused: c8 receipt names producer-created unit {key}")
                continue
            if key in included_arm_keys:
                refusals.append(
                    f"unit-refused: duplicate resolution for {item_id!r}/{arm_id!r}: "
                    "the arm is in a merge receipt's included-arm set AND carries a "
                    "c8_unsigned_negative receipt"
                )
                continue
            if unit["unit"] == "base":
                arm = unit.get("entire_substance_arm")
                if arm is None:
                    refusals.append(
                        f"evidence-refused: c8 receipt for {item_id!r} base unit, but "
                        "the item's substance is not a single signature arm"
                    )
                    continue
            else:
                arm = unit["arm"]
            reason = _valid_c8_receipt(row, arm)
            if reason is not None:
                refusals.append(f"evidence-refused: {item_id!r}/{arm_id or 'base'!r}: {reason}")
                continue
            resolutions[key].append({"resolution": "c8_unsigned_negative", "row": dict(row)})

    unit_rows: list[dict[str, Any]] = []
    operationally_resolved: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for key in sorted(units):
        resolved = resolutions[key]
        item_id, unit_key = key
        if len(resolved) > 1:
            refusals.append(
                f"unit-refused: {item_id!r}/{unit_key!r} carries "
                f"{len(resolved)} resolutions; exactly one is legal (an arm both in "
                "an included-arm set and carrying a receipt is a duplicate)"
            )
            continue
        if not resolved:
            unresolved.append(f"{item_id}/{unit_key}")
            continue
        entry = resolved[0]
        row = {
            "item_id": item_id,
            "unit": unit_key,
            "tier": units[key]["tier"],
            "resolution": entry["resolution"],
        }
        unit_rows.append(row)
        if entry["resolution"] in {"dead_with_evidence", "c8_unsigned_negative"}:
            operationally_resolved.append(row)

    status = "passed" if not refusals and not unresolved else "failed"
    return {
        "schema_version": CENSUS_SCHEMA_VERSION,
        "status": status,
        "units_total": len(units),
        "units": unit_rows,
        "operationally_resolved": operationally_resolved,
        "unresolved": unresolved,
        "refusals": refusals,
    }


def write_census(envelope_path: Path, census_path: Path) -> dict[str, Any]:
    """Evaluate one envelope file and write the census artifact."""

    census = evaluate_census(load_envelope(envelope_path))
    census_path.parent.mkdir(parents=True, exist_ok=True)
    census_path.write_text(json.dumps(census, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return census
