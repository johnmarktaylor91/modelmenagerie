"""Read-only census of the author/checker identity epochs a records root recorded.

BL-3's second half asked for "a reviewed records migration or typed carve-out for
the 10 live models recorded under ``gpt-5.6-terra``, so pre-flip rows stay
adjudicable". The carve-out is what landed, and deliberately so: the model ledger
is append-only, and a migration that rewrote ``provenance.checker_model`` would
destroy the one durable fact the rows still carry -- which model actually
adjudicated them. The declared alternative is
:data:`menagerie.crawler.constants.TIER_CAMPAIGN_CHECKER_MODEL_HISTORY`, a closed
per-campaign history that lets a prior binding be READ.

This tool is the disclosure half of that carve-out. It reads a records root and
answers, per campaign, exactly three questions:

* which ``(author_model, author_version, checker_model, checker_version)`` epochs
  the recorded rows carry, and how many rows sit under each;
* whether every recorded model label is inside its campaign's CLOSED declared
  history -- an undeclared label exits 2, because a rotation nobody declared is
  indistinguishable from corruption;
* which recorded identity components are UNPINNED, i.e. equal to a non-versioned
  literal from :data:`menagerie.crawler.constants.NON_VERSIONED_IDENTITY_LITERALS`.
  These are disclosed, never counted as pinned: a sole authority identified by an
  alias plus the word ``current`` cannot be distinguished across a multi-week run.

It writes nothing. It is safe to point at a live campaign.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.constants import (
    NON_VERSIONED_IDENTITY_LITERALS,
    TIER_CAMPAIGN_AUTHOR_MODEL_HISTORY,
    TIER_CAMPAIGN_CHECKER_MODEL_HISTORY,
)
from menagerie.crawler.partitioner import (
    CAMPAIGN_SPECS,
    CampaignPartitionError,
    load_campaign_bindings,
)
from menagerie.crawler.recordio import scan_jsonl

#: Recorded provenance components this census adjudicates, paired with the closed
#: declared history that governs each model label. Version components have no
#: enumerable history -- they are adjudicated only for pinnedness.
_MODEL_AXES: tuple[tuple[str, Mapping[str, tuple[str, ...]]], ...] = (
    ("author_model", TIER_CAMPAIGN_AUTHOR_MODEL_HISTORY),
    ("checker_model", TIER_CAMPAIGN_CHECKER_MODEL_HISTORY),
)
_VERSION_AXES: tuple[str, ...] = ("author_version", "checker_version")

_EPOCH_KEYS: tuple[str, ...] = (
    "author_model",
    "author_version",
    "checker_model",
    "checker_version",
)


class CheckerIdentityAuditError(RuntimeError):
    """A records root cannot be censused for recorded identity epochs."""


def _model_ledger_paths(records_root: Path) -> tuple[Path, ...]:
    """Return every model-ledger shard below one records root, in byte order.

    Parameters
    ----------
    records_root:
        Canonical crawler records root.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Sorted model JSONL shard paths; empty when the root holds no ledger.
    """

    models_dir = records_root / "models"
    if not models_dir.is_dir():
        return ()
    return tuple(sorted(path for path in models_dir.rglob("*.jsonl") if path.is_file()))


def _declared_history(campaign_id: str, axis: str) -> tuple[str, ...]:
    """Return the closed declared history for one campaign and model axis.

    Parameters
    ----------
    campaign_id:
        Tier campaign identifier, or ``"unattributed"`` for rows with no campaign.
    axis:
        ``author_model`` or ``checker_model``.

    Returns
    -------
    tuple[str, ...]
        Oldest-first declared labels. Empty when the campaign is undeclared, so
        every observed label reads as undeclared rather than as vacuously fine.
    """

    table = dict(_MODEL_AXES)[axis]
    return table.get(campaign_id, ())


def _campaign_of(record: Mapping[str, Any], default: Optional[str]) -> str:
    """Return the campaign a model record belongs to.

    Parameters
    ----------
    record:
        Model revision row.
    default:
        Campaign asserted by the caller, used when the row does not name one.

    Returns
    -------
    str
        Campaign identifier, or ``"unattributed"`` when neither source names one.
    """

    provenance = record.get("provenance")
    if isinstance(provenance, Mapping):
        recorded = provenance.get("campaign_id")
        if isinstance(recorded, str) and recorded:
            return recorded
    return default or "unattributed"


def audit_records_root(
    records_root: Path, *, campaign_id: Optional[str] = None
) -> dict[str, Any]:
    """Census the recorded identity epochs below one records root.

    Parameters
    ----------
    records_root:
        Canonical crawler records root to read.
    campaign_id:
        Campaign to attribute rows that do not name one. When the records root is
        one campaign's clone -- the normal shape -- this is that campaign.

    Returns
    -------
    dict[str, Any]
        Census document: per-campaign epoch tables, undeclared-label reasons,
        unpinned-component disclosure, and the manifest projection echo.

    Raises
    ------
    CheckerIdentityAuditError
        If the records root does not exist or its model ledger cannot be read.
    """

    if not records_root.is_dir():
        raise CheckerIdentityAuditError(f"records root is not a directory: {records_root}")

    epochs: dict[str, Counter[tuple[Any, ...]]] = {}
    for path in _model_ledger_paths(records_root):
        try:
            records = scan_jsonl(path, validate=False)
        except Exception as exc:  # noqa: BLE001 -- a census must name the file
            raise CheckerIdentityAuditError(f"cannot read model ledger {path}: {exc}") from exc
        for record in records:
            owner = _campaign_of(record, campaign_id)
            provenance = record.get("provenance")
            values = (
                tuple(provenance.get(key) for key in _EPOCH_KEYS)
                if isinstance(provenance, Mapping)
                else tuple(None for _ in _EPOCH_KEYS)
            )
            epochs.setdefault(owner, Counter())[values] += 1

    campaigns: list[dict[str, Any]] = []
    undeclared: list[dict[str, Any]] = []
    unpinned: list[dict[str, Any]] = []
    for owner in sorted(epochs):
        rows: list[dict[str, Any]] = []
        for values, count in sorted(epochs[owner].items(), key=lambda item: str(item[0])):
            epoch = dict(zip(_EPOCH_KEYS, values))
            classifications: dict[str, str] = {}
            for axis, _table in _MODEL_AXES:
                history = _declared_history(owner, axis)
                observed = epoch[axis]
                if observed in history:
                    classification = (
                        "declared-current" if observed == history[-1] else "declared-prior"
                    )
                else:
                    classification = "undeclared"
                    undeclared.append(
                        {
                            "campaign_id": owner,
                            "axis": axis,
                            "observed": observed,
                            "declared_history": list(history),
                            "row_count": count,
                            "reason": (
                                "recorded label is outside the closed declared history; "
                                "register the rotation or the rows are unadjudicable"
                            ),
                        }
                    )
                classifications[axis] = classification
            for axis in _VERSION_AXES:
                observed = epoch[axis]
                if isinstance(observed, str) and observed in NON_VERSIONED_IDENTITY_LITERALS:
                    classifications[axis] = "unpinned"
                    unpinned.append(
                        {
                            "campaign_id": owner,
                            "axis": axis,
                            "observed": observed,
                            "row_count": count,
                            "reason": (
                                "non-versioned identity literal: this component cannot "
                                "distinguish two runs of the same alias"
                            ),
                        }
                    )
                else:
                    classifications[axis] = "pinned"
            rows.append(
                {"epoch": epoch, "row_count": count, "classification": classifications}
            )
        campaigns.append(
            {
                "campaign_id": owner,
                "row_count": sum(epochs[owner].values()),
                "epochs": rows,
            }
        )

    manifest_path = records_root / "campaigns.json"
    manifest: dict[str, Any]
    if not manifest_path.is_file():
        manifest = {"status": "absent", "path": manifest_path.as_posix()}
    else:
        try:
            bindings = load_campaign_bindings(manifest_path)
        except CampaignPartitionError as exc:
            manifest = {
                "status": "unreadable",
                "path": manifest_path.as_posix(),
                "refusal": str(exc),
            }
        else:
            manifest = {
                "status": "readable",
                "path": manifest_path.as_posix(),
                "campaigns": {
                    binding.spec.campaign_id: {
                        "recorded_projection": dict(binding.recorded_projection),
                        "live_projection": {
                            "workload_class": binding.spec.workload_class,
                            "phase": binding.spec.phase,
                            "author_model": binding.spec.author_model,
                            "checker_model": binding.spec.checker_model,
                            "review_checkpoint_at": binding.spec.review_checkpoint_at,
                        },
                        "projection_drift": list(binding.projection_drift),
                    }
                    for binding in bindings
                },
            }

    return {
        "records_root": records_root.as_posix(),
        "declared_history": {
            "author_model": {key: list(value) for key, value in sorted(
                TIER_CAMPAIGN_AUTHOR_MODEL_HISTORY.items()
            )},
            "checker_model": {key: list(value) for key, value in sorted(
                TIER_CAMPAIGN_CHECKER_MODEL_HISTORY.items()
            )},
        },
        "campaigns": campaigns,
        "member_reasons": {
            "undeclared_identity": undeclared,
            "unpinned_identity_component": unpinned,
        },
        "manifest": manifest,
        "verdict": "undeclared-identity" if undeclared else "all-identities-declared",
    }


def build_parser() -> argparse.ArgumentParser:
    """Return the census argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """

    parser = argparse.ArgumentParser(
        prog="python -m menagerie.crawler.tools.audit_checker_identity_epochs",
        description=(
            "Read-only census of the author/checker identity epochs a records "
            "root recorded. Exits 2 when a recorded label is outside its "
            "campaign's closed declared history."
        ),
    )
    parser.add_argument(
        "--records-root",
        required=True,
        type=Path,
        help="canonical crawler records root to read (never written)",
    )
    parser.add_argument(
        "--campaign",
        choices=[spec.campaign_id for spec in CAMPAIGN_SPECS],
        default=None,
        help="campaign to attribute rows that do not name one",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the full census document instead of the operator table",
    )
    return parser


def _render_table(census: Mapping[str, Any]) -> str:
    """Render the census as an operator-readable table.

    Parameters
    ----------
    census:
        Census document from :func:`audit_records_root`.

    Returns
    -------
    str
        Plain-text table.
    """

    lines = [f"records-root: {census['records_root']}", f"verdict: {census['verdict']}"]
    for campaign in census["campaigns"]:
        lines.append(f"\n{campaign['campaign_id']}  rows={campaign['row_count']}")
        for row in campaign["epochs"]:
            epoch = row["epoch"]
            marks = row["classification"]
            lines.append(
                f"  {row['row_count']:>6}  "
                f"author={epoch['author_model']}/{epoch['author_version']} "
                f"[{marks['author_model']}/{marks['author_version']}]  "
                f"checker={epoch['checker_model']}/{epoch['checker_version']} "
                f"[{marks['checker_model']}/{marks['checker_version']}]"
            )
    for kind, reasons in sorted(census["member_reasons"].items()):
        if not reasons:
            continue
        lines.append(f"\n{kind}:")
        for reason in reasons:
            lines.append(
                f"  {reason['campaign_id']} {reason['axis']}={reason['observed']!r} "
                f"rows={reason['row_count']}: {reason['reason']}"
            )
    manifest = census["manifest"]
    lines.append(f"\nmanifest: {manifest['status']} ({manifest['path']})")
    if manifest["status"] == "unreadable":
        lines.append(f"  refusal: {manifest['refusal']}")
    for campaign_id, entry in sorted(manifest.get("campaigns", {}).items()):
        drift = ",".join(entry["projection_drift"]) or "none"
        lines.append(f"  {campaign_id} projection_drift={drift}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the census and print it.

    Parameters
    ----------
    argv:
        Optional command arguments, excluding the executable name.

    Returns
    -------
    int
        Zero when every recorded label is declared, 2 after a typed refusal or an
        undeclared label. An operator tool that tracebacks instead of refusing is
        indistinguishable from a crash.
    """

    args = build_parser().parse_args(argv)
    try:
        census = audit_records_root(args.records_root, campaign_id=args.campaign)
    except CheckerIdentityAuditError as exc:
        print(f"identity-epoch census refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(census, indent=2, sort_keys=True) if args.json else _render_table(census))
    return 0 if census["verdict"] == "all-identities-declared" else 2


if __name__ == "__main__":  # pragma: no cover -- operator entry point
    raise SystemExit(main())
