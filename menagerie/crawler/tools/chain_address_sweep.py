"""READ-ONLY pre-cutover sweep: does any ledger's chain head MOVE under derivation?

The Track A chain address used to be SELECTED from the first self-declared digest
field a row carried; it is now DERIVED from the row body
(:func:`menagerie.crawler.ledger_chain.row_content_digest`).  Algebra and a
whole-corpus census both say the two answers agree on every honest row of every
family -- each writer already stamped its digest as the hash of its own body, under
one canonicalization, and no honest row carries a second digest field.  But that is
a statement about the code reachable from HEAD, not about arbitrary bytes already on
disk: a row stamped by some older, different rule would diverge, and the first
refusal on old data would read as a false positive and invite a tolerance.

So this tool converts the residual into a checked fact.  For every chain-bearing
ledger below a records root it recomputes BOTH heads -- the deleted ladder's and the
body derivation's -- and reports any divergence, plus every row that the fused
re-certification or the closed envelope vocabulary would now refuse.  A clean report
is a RELAUNCH PRECONDITION; it is not a substitute for the verifier.

It opens nothing for writing, takes no lock, and never repairs: run it against a
copy or a live root freely.

Usage::

    PYTHONPATH=. python -m menagerie.crawler.tools.chain_address_sweep RECORDS_ROOT
    PYTHONPATH=. python -m menagerie.crawler.tools.chain_address_sweep ROOT --json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from menagerie.crawler import ledger_chain
from menagerie.crawler.identity import (
    ROW_CONTENT_ADDRESS_EXCLUDED,
    row_content_address,
    stable_hash,
)

#: The order the deleted ladder consulted, preserved HERE and nowhere else: this is
#: the only place in the tree that still needs to know what the old rule was.
_LADDER_FIELDS = ("close_sha256", "row_sha256", "record_revision", "payload_sha256")

#: The money scope, the only one whose rows the settlement-schema gate governs.
_MONEY_SCOPE = "paid.sessions"


def legacy_settlement_rows(scope: str, rows: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    """Return the money rows a PRE-EXISTING legacy settlement schema would brick.

    PSLRE6 acct F4. ``_StateBuilder._admit_settlement_schema`` refuses a legacy
    settlement schema on this ledger unconditionally, deriving the decision from the
    ledger's STRUCTURE rather than from anything a producer declares -- which is the
    right shape, and is what made the PSLRE5 forgery worthless. Its cost is that a row
    already on disk under ``paid-session-settlement.v1``/``v2``/``v3`` refuses the
    WHOLE money authority, permanently, on its first read.

    That is fail-closed, never a forgery: no charge is lost or invented, and the
    refusal is typed. But it is a BRICK an operator would meet for the first time at
    relaunch, with a message about a cutover rather than about the row. The relaunch
    sweep already exists to convert "no on-disk row was stamped by an older rule" from
    an assumption into a checked fact, so the same sweep names this row too.

    Parameters
    ----------
    scope:
        Canonical chain scope of the ledger being swept.
    rows:
        Its decoded rows, in ledger order.

    Returns
    -------
    tuple[int, ...]
        Zero-based positions of legacy settlement rows, empty for every other scope.
    """

    if scope != _MONEY_SCOPE:
        return ()
    from menagerie.crawler.paid_session_accounting import LEGACY_SETTLEMENT_SCHEMAS

    return tuple(
        index
        for index, row in enumerate(rows)
        if row.get("event") == "session_settled"
        and row.get("schema_version") in LEGACY_SETTLEMENT_SCHEMAS
    )


def ladder_content_digest(row: Mapping[str, Any]) -> str:
    """Return the content address the DELETED ladder would have used for a row."""

    logical = {str(key): value for key, value in row.items() if str(key) != "chain"}
    for field_name in _LADDER_FIELDS:
        value = logical.get(field_name)
        if isinstance(value, str) and value:
            return value
    return stable_hash(logical)


@dataclass(frozen=True)
class LedgerSweep:
    """One ledger's both-ways head recomputation and refusal census."""

    scope: str
    path: str
    rows: int
    ladder_head: str
    derived_head: str
    address_divergences: tuple[Mapping[str, Any], ...] = ()
    fused_refusals: tuple[Mapping[str, Any], ...] = ()
    envelope_refusals: tuple[Mapping[str, Any], ...] = ()
    legacy_settlements: tuple[Mapping[str, Any], ...] = ()
    unreadable: tuple[str, ...] = ()

    @property
    def clean(self) -> bool:
        """Whether nothing about this ledger changes under the derived address."""

        return (
            self.ladder_head == self.derived_head
            and not self.address_divergences
            and not self.fused_refusals
            and not self.envelope_refusals
            and not self.legacy_settlements
            and not self.unreadable
        )

    def as_json(self) -> dict[str, Any]:
        """Return the report row for machine consumption."""

        return {
            "scope": self.scope,
            "path": self.path,
            "rows": self.rows,
            "ladder_head": self.ladder_head,
            "derived_head": self.derived_head,
            "heads_agree": self.ladder_head == self.derived_head,
            "address_divergences": list(self.address_divergences),
            "fused_refusals": list(self.fused_refusals),
            "envelope_refusals": list(self.envelope_refusals),
            "legacy_settlements": list(self.legacy_settlements),
            "unreadable": list(self.unreadable),
            "clean": self.clean,
        }


@dataclass
class _Accumulator:
    """Mutable per-ledger scratch, so the sweep itself stays a pure report."""

    address_divergences: list[Mapping[str, Any]] = field(default_factory=list)
    fused_refusals: list[Mapping[str, Any]] = field(default_factory=list)
    envelope_refusals: list[Mapping[str, Any]] = field(default_factory=list)
    legacy_settlements: list[Mapping[str, Any]] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)


def _decode(path: Path, found: _Accumulator) -> tuple[dict[str, Any], ...]:
    """Decode one ledger's rows, recording rather than raising on bad lines."""

    if not path.is_file():
        return ()
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_bytes().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except ValueError as exc:
            found.unreadable.append(f"{path}:{number} is not valid JSON: {exc}")
            continue
        if not isinstance(value, Mapping):
            found.unreadable.append(f"{path}:{number} is not a JSON object")
            continue
        rows.append(dict(value))
    return tuple(rows)


def sweep_ledger(scope: str, path: Path) -> LedgerSweep:
    """Recompute one ledger's head BOTH ways and census what would now refuse."""

    found = _Accumulator()
    rows = _decode(path, found)
    ladder_previous = ledger_chain._genesis(scope)  # noqa: SLF001 -- read-only probe
    derived_previous = ladder_previous
    for index, row in enumerate(rows):
        ladder = ladder_content_digest(row)
        derived = row_content_address(row)
        if ladder != derived:
            found.address_divergences.append(
                {
                    "row": index,
                    "ladder_address": ladder,
                    "derived_address": derived,
                    "declared_digest_fields": sorted(
                        name
                        for name in ROW_CONTENT_ADDRESS_EXCLUDED - {"chain"}
                        if row.get(name) is not None
                    ),
                }
            )
        for name in sorted(ROW_CONTENT_ADDRESS_EXCLUDED - {"chain"}):
            declared = row.get(name)
            if declared is not None and declared != derived:
                found.fused_refusals.append(
                    {"row": index, "field": name, "declared": declared, "derived": derived}
                )
        envelope = row.get("chain")
        if isinstance(envelope, Mapping):
            extra = sorted(set(map(str, envelope)) - set(ledger_chain.CHAIN_ENVELOPE_FIELDS))
            if extra:
                found.envelope_refusals.append({"row": index, "unexpected_keys": extra})
            # The envelope's VALUES are machinery too (PSLRE6 D6-5), so a row the
            # verifier would refuse for a non-conforming value is surfaced here as
            # well -- the sweep exists so a relaunch meets no refusal it has not
            # already named.
            value_findings = ledger_chain.chain_envelope_value_findings(envelope)
            if value_findings:
                found.envelope_refusals.append(
                    {"row": index, "nonconforming_values": list(value_findings)}
                )
        ladder_previous = ledger_chain._extend(  # noqa: SLF001
            scope, ladder_previous, index, ladder
        )
        derived_previous = ledger_chain._extend(  # noqa: SLF001
            scope, derived_previous, index, derived
        )
    for position in legacy_settlement_rows(scope, rows):
        found.legacy_settlements.append(
            {
                "row": position,
                "schema_version": rows[position].get("schema_version"),
                "spend_id": rows[position].get("spend_id"),
                "consequence": (
                    "the settlement-schema gate refuses every row of this ledger while "
                    "this row is resident: the money authority is BRICKED (fail-closed, "
                    "never a forgery), and the prefix must be pinned as the separate "
                    "legacy operational object it describes"
                ),
            }
        )
    return LedgerSweep(
        scope=scope,
        path=str(path),
        rows=len(rows),
        ladder_head=ladder_previous,
        derived_head=derived_previous,
        address_divergences=tuple(found.address_divergences),
        fused_refusals=tuple(found.fused_refusals),
        envelope_refusals=tuple(found.envelope_refusals),
        legacy_settlements=tuple(found.legacy_settlements),
        unreadable=tuple(found.unreadable),
    )


def chain_bearing_ledgers(records_root: Path) -> tuple[tuple[str, Path], ...]:
    """Return every ``(scope, path)`` whose rows flow through the chain.

    The same universe the Track A verifier walks: the three close-spine meta
    ledgers, the paid-session money ledger, and every census source a close pins a
    ``chain_head`` for.
    """

    from menagerie.crawler.close_object import close_object_path
    from menagerie.crawler.leg_generation import close_ledger_sources, leg_events_path
    from menagerie.crawler.rung_staging import rung_registry_path

    ledgers: list[tuple[str, Path]] = [
        ("meta.leg-events", leg_events_path(records_root)),
        ("meta.closes", close_object_path(records_root)),
        ("meta.registry", rung_registry_path(records_root)),
        ("paid.sessions", records_root / "operational" / "spend.jsonl"),
    ]
    for scope, path in sorted(close_ledger_sources(records_root).items()):
        ledgers.append((scope, path))
    return tuple(ledgers)


def sweep_records_root(records_root: Path) -> tuple[LedgerSweep, ...]:
    """Recompute every chain-bearing ledger's head both ways below one root."""

    return tuple(
        sweep_ledger(scope, path) for scope, path in chain_bearing_ledgers(records_root)
    )


def render(sweeps: Iterable[LedgerSweep]) -> str:
    """Return the human-readable sweep report."""

    lines: list[str] = []
    divergent = 0
    for sweep in sweeps:
        mark = "OK  " if sweep.clean else "MOVE"
        lines.append(
            f"{mark} {sweep.scope:28s} rows={sweep.rows:<7d} "
            f"{'heads agree' if sweep.ladder_head == sweep.derived_head else 'HEADS DIFFER'}"
        )
        if not sweep.clean:
            divergent += 1
            for item in sweep.address_divergences[:5]:
                lines.append(f"       row {item['row']}: address moves {item}")
            for item in sweep.fused_refusals[:5]:
                lines.append(f"       row {item['row']}: {item['field']} would refuse")
            for item in sweep.envelope_refusals[:5]:
                lines.append(f"       row {item['row']}: envelope refuses {dict(item)}")
            for item in sweep.legacy_settlements[:5]:
                lines.append(
                    f"       row {item['row']}: LEGACY settlement schema "
                    f"{item['schema_version']} would BRICK this money authority"
                )
            for unreadable_entry in sweep.unreadable[:5]:
                lines.append(f"       {unreadable_entry}")
    lines.append("")
    lines.append(
        "CHAIN_ADDRESS_SWEEP_CLEAN"
        if divergent == 0
        else f"CHAIN_ADDRESS_SWEEP_DIVERGENT: {divergent} ledger(s)"
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the read-only both-ways head sweep from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records_root", type=Path)
    parser.add_argument(
        "--json", action="store_true", help="emit the machine-readable report instead"
    )
    args = parser.parse_args(argv)
    sweeps = sweep_records_root(args.records_root)
    if args.json:
        print(
            json.dumps(
                {
                    "schema_version": "menagerie.crawler.chain-address-sweep.v2",
                    "records_root": str(args.records_root),
                    "clean": all(sweep.clean for sweep in sweeps),
                    "ledgers": [sweep.as_json() for sweep in sweeps],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(render(sweeps))
    return 0 if all(sweep.clean for sweep in sweeps) else 1


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
