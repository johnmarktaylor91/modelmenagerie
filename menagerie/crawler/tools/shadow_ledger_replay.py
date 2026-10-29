"""Shadow-ledger replay over a snapshotted records root (AB-15 core).

The 200a campaign left ~1,547 settled paid sessions, 1,382 recorded attempts,
and a canonical model ledger whose projections were only ever derived by the
LIVE driver. This tool is the R0 instrument that rederives everything from
raw bytes, in shadow, against a snapshot that provably belongs to ONE census
epoch -- so every later merge can prove (R1's standing production-records
replay gate, B3-XV-4's owner replay, B2-X-V37's process replay) that new code
still reads the real campaign's history the same way, without ever touching
the live root.

Subcommands::

    snapshot        Copy a records root into a shadow root with a one-epoch
                    proof: every source file is hashed before and after the
                    copy, and ANY drift refuses the snapshot (a census over a
                    moving root proves nothing -- sol 3.13).
    replay          Verify the shadow root's tamper-evident chains, rebuild
                    the canonical current projection through the REAL reducer
                    and authority machinery, fold the paid-session ledger
                    through the REAL accounting state builder, and emit one
                    machine-readable report artifact.
    attempts-owners Replay every recorded attempt failure through the closed
                    owner authorities (B3-XV-4's instrument): the per-attempt
                    stamp and WORKER_FAILURE_REASON_OWNERS first, then the
                    closed replay derivation in
                    ``attempt_owner_derivation`` for the driver-lane reason
                    families the worker map deliberately never covered.
                    Derived owners and typed unowned dispositions are counted
                    in their own segregated buckets; a row outside every
                    closed authority is silent residue and keeps the census
                    red. ``--derivation-packet`` additionally writes the
                    per-row packet the final-sweep operator step consumes
                    (this tool never writes records).

Replays REFUSE to run against a root without a snapshot manifest: the live
root is never a replay substrate. Exit codes: 0 green, 2 typed refusal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.identity import canonical_json_bytes, hash_bytes

SCHEMA_VERSION = "menagerie.crawler.shadow-ledger-replay.v1"
#: The owner census gained segregated derived/disclosed buckets (B3-XV-4);
#: its report carries its own kind-scoped version so v1 consumers can tell.
#: Snapshot manifests and replay reports keep SCHEMA_VERSION unchanged.
CENSUS_SCHEMA_VERSION = "menagerie.crawler.attempts-owner-census.v2"
PACKET_SCHEMA_VERSION = "menagerie.crawler.attempts-owner-derivation-packet.v1"
MANIFEST_NAME = "shadow-snapshot-manifest.json"

#: Runtime lock artifacts are writer state, not campaign history.
_EXCLUDED_SUFFIXES = (".lock",)


def _campaign_files(records_root: Path) -> list[Path]:
    """Return every campaign history file below one records root, sorted."""

    return sorted(
        path
        for path in records_root.rglob("*")
        if path.is_file()
        and path.name != MANIFEST_NAME
        and not any(path.name.endswith(suffix) for suffix in _EXCLUDED_SUFFIXES)
    )


def _hash_tree(records_root: Path) -> dict[str, str]:
    """Hash every campaign file below one records root."""

    return {
        str(path.relative_to(records_root)): hash_bytes(path.read_bytes())
        for path in _campaign_files(records_root)
    }


def snapshot(records_root: Path, shadow_root: Path) -> dict[str, Any]:
    """Copy one records root into a shadow root with a one-epoch proof.

    Raises
    ------
    ValueError
        When the source tree changed between the pre- and post-copy hash
        passes (the snapshot spans more than one census epoch), or the shadow
        root already carries a snapshot.
    """

    records_root = records_root.resolve(strict=True)
    shadow_root = shadow_root.resolve()
    if shadow_root == records_root:
        raise ValueError("shadow root must not be the live records root")
    if (shadow_root / MANIFEST_NAME).exists():
        raise ValueError(f"{shadow_root} already carries a snapshot manifest")
    before = _hash_tree(records_root)
    for relative, _digest in before.items():
        source = records_root / relative
        target = shadow_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    after = _hash_tree(records_root)
    if after != before:
        changed = sorted(
            set(before.items()) ^ set(after.items()),
        )
        raise ValueError(
            "records root changed during the snapshot; the copy spans more "
            f"than one census epoch and is refused: {changed[:6]}"
        )
    copied = _hash_tree(shadow_root)
    if copied != before:
        raise ValueError("shadow copy does not byte-match its source")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_root": str(records_root),
        "files": before,
    }
    (shadow_root / MANIFEST_NAME).write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def _verified_shadow(shadow_root: Path) -> Mapping[str, Any]:
    """Load one shadow root's manifest and refuse any post-snapshot tamper."""

    manifest_path = shadow_root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(
            f"{shadow_root} carries no {MANIFEST_NAME}: replays never run "
            "against a live records root -- snapshot it first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unknown snapshot schema: {manifest.get('schema_version')}")
    current = _hash_tree(shadow_root)
    if current != manifest["files"]:
        drifted = sorted(
            set(map(tuple, manifest["files"].items())) ^ set(current.items())
        )
        raise ValueError(f"shadow root drifted from its snapshot manifest: {drifted[:6]}")
    return manifest


def replay(
    shadow_root: Path,
    intake_root: Path,
    *,
    author_model: Optional[str],
    checker_model: Optional[str],
) -> dict[str, Any]:
    """Rederive projections, chains, and money folds from shadow bytes."""

    from menagerie.crawler.driver_admission import restored_environment_generations
    from menagerie.crawler.driver_contracts import DriverConfig
    from menagerie.crawler.authority import build_authority_context
    from menagerie.crawler.checkpoint import canonical_operational_ledger_path
    from menagerie.crawler.intake import load_intake_snapshot
    from menagerie.crawler.ledger_chain import ledger_chain_refusals
    from menagerie.crawler.paid_session_accounting import AccountingState
    from menagerie.crawler.paid_session_durability import preserved_lineage_verifier
    from menagerie.crawler.paid_session_lifecycle import load_paid_session_rows
    from menagerie.crawler.recordio import scan_jsonl
    from menagerie.crawler.reducer import materialize_current
    from menagerie.crawler.rung_staging import (
        registered_prior_intake_members,
        registered_prior_intake_snapshots,
    )
    from menagerie.crawler.status import funnel_counts
    from menagerie.crawler.tools.scheduler_shadow_audit import resolve_ledger_paths
    from dataclasses import replace as dataclass_replace

    manifest = _verified_shadow(shadow_root)
    chain_refusals = tuple(ledger_chain_refusals(shadow_root))
    ledgers = resolve_ledger_paths(shadow_root)
    intake_snapshot = load_intake_snapshot(intake_root)
    config = DriverConfig()
    operational_path = canonical_operational_ledger_path(ledgers.models)
    operational_records = (
        tuple(scan_jsonl(operational_path)) if operational_path.exists() else ()
    )
    base_context = build_authority_context(
        active_intake_snapshot_id=intake_snapshot.snapshot_id,
        active_intake_snapshot_sha256=intake_snapshot.snapshot_sha256,
        intake_rows=(item.to_dict() for item in intake_snapshot.items),
        author_model=author_model or config.author_model,
        author_version=config.author_version,
        checker_model=checker_model or config.checker_model,
        checker_version=config.checker_version,
        prior_intake_snapshots=registered_prior_intake_snapshots(
            intake_snapshot.root, intake_snapshot.snapshot_id
        ),
        prior_intake_member_ids=registered_prior_intake_members(
            intake_snapshot.root, intake_snapshot.snapshot_id
        ),
    )
    context = dataclass_replace(
        base_context,
        environment_generations=restored_environment_generations(
            base_context.environment_generations,
            ledgers.attempts,
            operational_records,
        ),
    )
    current = materialize_current(ledgers, context=context)
    status_counts: dict[str, int] = {}
    for record in current.values():
        code = str(record["status"]["code"])
        status_counts[code] = status_counts.get(code, 0) + 1
    # Raw ledger heads, independent of authority currency: the difference
    # between heads and the projected current set is the campaign's live
    # identity-rotation debt (a head recorded under rotated axes projects to
    # nothing until re-driven), measured rather than estimated.
    head_ids: set[str] = set()
    if ledgers.models.exists():
        for record in scan_jsonl(ledgers.models):
            head_ids.add(str(record["stable_id"]))

    spend_report: dict[str, Any] = {"present": False}
    spend_path = shadow_root / "operational" / "spend.jsonl"
    if spend_path.is_file():
        rows = load_paid_session_rows(spend_path)
        # The lineage-equipped fold, exactly as the live lifecycle seeds it: a
        # rollover row is admissible only while its preserved ``.poisoned-*``
        # predecessor and anchor authenticate beside the (snapshotted) ledger.
        state = AccountingState(
            lineage_verifier=preserved_lineage_verifier(
                spend_path.parent,
                ledger_name=spend_path.name,
                subject="shadow-replay paid-session fold",
            )
        ).extend(rows)
        events: dict[str, int] = {}
        for row in rows:
            event = str(row.get("event"))
            events[event] = events.get(event, 0) + 1
        spend_report = {
            "present": True,
            "rows": len(rows),
            "events": events,
            "reservations": len(state.reservations),
            "settlements": len(state.settlements),
            "unsettled_reservations": sorted(
                set(state.reservations) - set(state.settlements)
            ),
            "attempt_groups": len(state.attempt_groups),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "shadow-replay-report",
        "source_root": manifest["source_root"],
        "snapshot_files": len(manifest["files"]),
        "chain_refusals": list(chain_refusals),
        "intake_members": len(intake_snapshot.items),
        "ledger_heads": len(head_ids),
        "projected_records": len(current),
        "stale_heads": len(head_ids - set(current)),
        "status_counts": dict(sorted(status_counts.items())),
        "funnel": funnel_counts(tuple(current.values())),
        "spend": spend_report,
        "green": not chain_refusals,
    }


def attempts_owners(
    shadow_root: Path, *, derivation_packet: Optional[Path] = None
) -> dict[str, Any]:
    """Replay every recorded attempt failure through the closed owner
    authorities: per-attempt stamp / worker vocabulary first, then the closed
    replay derivation for the driver-lane reason families, then the typed
    unowned dispositions. Owned, derived, and disclosed rows live in
    segregated buckets; anything left is silent residue and reds the census.
    """

    from menagerie.crawler.attempt_owner_derivation import (
        ATTEMPT_TRIPLE_OWNERS,
        UNOWNED_ATTEMPT_DISPOSITIONS,
        derive_attempt_failure_owner,
    )
    from menagerie.crawler.driver_receipts import (
        NEVER_MODEL_TERMINAL_OWNERS,
        WORKER_FAILURE_REASON_OWNERS,
        _failure_owner,
    )
    from menagerie.crawler.paid_session_lifecycle import load_paid_session_rows
    from menagerie.crawler.recordio import scan_jsonl
    from menagerie.crawler.tools.scheduler_shadow_audit import resolve_ledger_paths

    _verified_shadow(shadow_root)
    ledgers = resolve_ledger_paths(shadow_root)
    attempts = tuple(scan_jsonl(ledgers.attempts))
    spend_path = shadow_root / "operational" / "spend.jsonl"
    spend_events: tuple[Mapping[str, Any], ...] = (
        tuple(load_paid_session_rows(spend_path)) if spend_path.is_file() else ()
    )
    owners: dict[str, int] = {}
    derived_owners: dict[str, int] = {}
    derivation_rules: dict[str, int] = {}
    reasons: dict[str, int] = {}
    unowned_disclosed: list[dict[str, Any]] = []
    ownerless: list[dict[str, Any]] = []
    packet_rows: list[dict[str, Any]] = []
    failures = 0
    for row in attempts:
        error = row.get("error")
        if not isinstance(error, Mapping):
            continue
        failures += 1
        reason = str(error.get("reason_code"))
        reasons[reason] = reasons.get(reason, 0) + 1
        details = error.get("details")
        stamped = isinstance(details, Mapping) and isinstance(
            details.get("owner"), str
        )
        owner = _failure_owner(error)
        source: str
        rule: Optional[str] = None
        disposition: Optional[str] = None
        if owner is not None:
            source = "stamped" if stamped else "worker-vocabulary"
            owners[owner] = owners.get(owner, 0) + 1
        else:
            derivation = derive_attempt_failure_owner(
                error,
                stable_id=str(row.get("stable_id")),
                work_id=str(row.get("work_id")),
                spend_events=spend_events,
            )
            if derivation is None:
                source = "ownerless"
                ownerless.append(
                    {"stable_id": row.get("stable_id"), "reason_code": reason}
                )
            elif derivation.owner is not None:
                source = "derived-owner"
                owner = derivation.owner
                rule = derivation.rule
                derived_owners[owner] = derived_owners.get(owner, 0) + 1
                derivation_rules[rule] = derivation_rules.get(rule, 0) + 1
            else:
                source = "unowned-disclosed"
                rule = derivation.rule
                disposition = derivation.disposition
                derivation_rules[rule] = derivation_rules.get(rule, 0) + 1
                unowned_disclosed.append(
                    {
                        "stable_id": row.get("stable_id"),
                        "reason_code": reason,
                        "disposition": disposition,
                    }
                )
        if derivation_packet is not None:
            packet_rows.append(
                {
                    "attempt_id": row.get("attempt_id"),
                    "stable_id": row.get("stable_id"),
                    "work_id": row.get("work_id"),
                    "stage": error.get("stage"),
                    "reason_code": reason,
                    "exception_type": error.get("exception_type"),
                    "source": source,
                    "owner": owner,
                    "rule": rule,
                    "disposition": disposition,
                    "never_model_terminal": owner in NEVER_MODEL_TERMINAL_OWNERS,
                }
            )
    if ownerless:
        verdict = "red"
    elif unowned_disclosed:
        verdict = "green-with-disclosed-unowned"
    else:
        verdict = "green"
    report = {
        "schema_version": CENSUS_SCHEMA_VERSION,
        "kind": "attempts-owner-census",
        "attempts": len(attempts),
        "failed_attempts": failures,
        "owners": dict(sorted(owners.items())),
        "derived_owners": dict(sorted(derived_owners.items())),
        "derivation_rules": dict(sorted(derivation_rules.items())),
        "reasons": dict(sorted(reasons.items())),
        "closed_vocabulary_size": len(WORKER_FAILURE_REASON_OWNERS),
        "closed_derivation_size": len(ATTEMPT_TRIPLE_OWNERS)
        + len(UNOWNED_ATTEMPT_DISPOSITIONS)
        + 1,
        "unowned_disclosed": unowned_disclosed,
        "unowned_disclosed_count": len(unowned_disclosed),
        "ownerless": ownerless,
        "verdict": verdict,
        "green": not ownerless,
    }
    if derivation_packet is not None:
        packet = {
            "schema_version": PACKET_SCHEMA_VERSION,
            "kind": "attempts-owner-derivation-packet",
            "source_root": str(shadow_root),
            "failed_attempts": failures,
            "verdict": verdict,
            "rows": packet_rows,
        }
        derivation_packet.parent.mkdir(parents=True, exist_ok=True)
        derivation_packet.write_bytes(canonical_json_bytes(packet) + b"\n")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run one subcommand and write its machine artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    snap = commands.add_parser("snapshot")
    snap.add_argument("--records-root", type=Path, required=True)
    snap.add_argument("--shadow-root", type=Path, required=True)
    rep = commands.add_parser("replay")
    rep.add_argument("--shadow-root", type=Path, required=True)
    rep.add_argument("--intake-root", type=Path, required=True)
    rep.add_argument("--author-model", default=None)
    rep.add_argument("--checker-model", default=None)
    rep.add_argument("--out", type=Path, default=None)
    own = commands.add_parser("attempts-owners")
    own.add_argument("--shadow-root", type=Path, required=True)
    own.add_argument("--out", type=Path, default=None)
    own.add_argument(
        "--derivation-packet",
        type=Path,
        default=None,
        help=(
            "Also write the per-failed-attempt owner-derivation packet "
            "(the final-sweep operator-step input); records are never written."
        ),
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    try:
        if args.command == "snapshot":
            manifest = snapshot(args.records_root, args.shadow_root)
            print(
                f"shadow snapshot: {len(manifest['files'])} files under "
                f"{args.shadow_root}"
            )
            return 0
        if args.command == "replay":
            report = replay(
                args.shadow_root,
                args.intake_root,
                author_model=args.author_model,
                checker_model=args.checker_model,
            )
        else:
            report = attempts_owners(
                args.shadow_root, derivation_packet=args.derivation_packet
            )
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"shadow replay refused: {exc}", file=sys.stderr)
        return 2
    payload = canonical_json_bytes(report) + b"\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(payload)
    print(payload.decode("utf-8"), end="")
    return 0 if report["green"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
