"""Measured-rotation tooling for ``records/control/rotations.jsonl`` (RC-2).

The operator-declared registry this tool used to write
(``intake/registered-authority-rotations.jsonl``) is frozen history. Rotation
classes are MEASURED by the one instrument that exercises each axis
(:data:`menagerie.crawler.rotations.MEASUREMENT_AUTHORITY`); there is no
operator write path to a positive class.

``seed``
    One-time import of the frozen registry into the measured ledger (one row
    per registered row, recorded class, ``measured_by = declared``). The only
    lawful source of a ``declared`` row.
``probe``
    The predicate replay probe over one records root: every head replays at
    its recorded policy basis under the running code and the derived
    disposition is compared with the recorded one field by field; any
    difference, or a full-replay refusal that is not a changed input, is a
    FLIP. ``--write-row`` appends the MEASURED row for every recorded
    identity of a predicate-replay axis that the ledger does not yet connect
    to the live identity (class = the probe's class on THIS root); the
    ruling's "the merge writes the row" act, run at deploy against the root
    it governs.
``acknowledge-closure``
    Record the live identity per predicate-replay axis (plus the probe's
    measurement when ``--probe-report`` is given) into the committed
    ``kernel_closure_declaration.json``. The smoke corpus refuses a tree whose
    live code-axis identity the file does not name: a kernel edit cannot ship
    without acknowledging the move. The declaration is acknowledgement and
    disclosure; it is never a row source.
``measure-golden``
    Classify a ``checker_prompt_identity`` move from a checker blind-replay
    report's rotation arm (a recorded ACCEPT that flips either way ->
    ``soundness``) and append the measured row. Refuses any other axis: the
    instrument never runs the author.
``list`` / ``print-live``
    Disclosure.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.identity import utc_now
from menagerie.crawler.models import JsonObject
from menagerie.crawler.rotations import (
    CODE_AXES,
    MEASUREMENT_METHODS,
    ROTATION_CLASSES,
    RotationLedgerError,
    append_rotation,
    build_rotation,
    classify_golden_replay,
    edge_classes,
    load_rotations,
    move_class,
    rotation_history,
    rotations_path,
    seed_from_registry,
)

KERNEL_CLOSURE_DECLARATION_SCHEMA_VERSION = "menagerie.crawler.kernel-closure-declaration.v1"
KERNEL_CLOSURE_DECLARATION_PATH = Path(__file__).with_name("kernel_closure_declaration.json")
ACKNOWLEDGE_COMMAND = (
    "python -m menagerie.crawler.tools.register_authority_rotation acknowledge-closure "
    "[--probe-report <probe.json>]"
)


def build_parser() -> argparse.ArgumentParser:
    """Build the rotation tooling argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--records-root", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("print-live", help="print the live identity per registrable axis")
    commands.add_parser("list", help="print the measured-rotations ledger and edge classes")

    seed = commands.add_parser("seed", help="import the frozen registry once")
    seed.add_argument("--registry", type=Path, help="registry path (default: below the records root)")
    seed.add_argument("--tree-sha", help="tree the seed ran on (default: git HEAD)")

    probe = commands.add_parser("probe", help="predicate replay probe over one records root")
    probe.add_argument("--intake", type=Path, required=True, help="intake snapshot directory")
    probe.add_argument("--tree-sha")
    probe.add_argument("--reason", default="predicate replay probe")
    probe.add_argument(
        "--write-row",
        action="store_true",
        help="append the measured rows for every unconnected predicate-replay axis move",
    )
    probe.add_argument("--report", type=Path, help="also write the probe report here")
    probe.add_argument(
        "--all-heads",
        action="store_true",
        help="replay every ledger head, not only the intake's members (the D.2 acceptance "
        "over all awards on the root)",
    )

    acknowledge = commands.add_parser(
        "acknowledge-closure",
        help="record the live predicate-replay axis identities into the committed declaration",
    )
    acknowledge.add_argument("--probe-report", type=Path, help="a probe report to disclose")
    acknowledge.add_argument("--reason", default="kernel closure moved")
    acknowledge.add_argument("--tree-sha")
    acknowledge.add_argument("--declaration", type=Path, default=KERNEL_CLOSURE_DECLARATION_PATH)

    golden = commands.add_parser("measure-golden", help="classify a checker-prompt move")
    golden.add_argument("--report", type=Path, required=True)
    golden.add_argument("--axis", default="checker_prompt_identity")
    golden.add_argument("--from-identity", dest="from_identity", required=True)
    golden.add_argument("--to-identity", dest="to_identity", help="default: the live identity")
    golden.add_argument("--tree-sha")
    golden.add_argument("--write-row", action="store_true")
    return parser


def _records_root(args: argparse.Namespace) -> Path:
    if args.records_root is not None:
        return args.records_root.resolve()
    return args.repo_root.resolve() / "menagerie" / "crawler" / "records"


def _tree_sha(repo_root: Path, declared: Optional[str]) -> str:
    if declared:
        return declared
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown-tree"


def live_authority_identities(campaign_id: Optional[str] = None) -> dict[str, str]:
    """Return the live shipped identity per registrable axis.

    THE authority for "what identity does the running code ship" (item 11):
    ``--print-live`` prints it, every registration's live-target guard
    compares against it, and the licensing-debt census (SD-2) and the W43
    stale-head census derive ``award_closure_identity`` from it. One
    derivation, one space.

    Derived from the same context builder every reducer consumer uses, over a
    minimal synthetic intake -- the shipped-policy identities are intake-
    independent byte hashes. ``toolchain_identity`` derives from the reviewed
    pin table and ``author_steering_surface_identity`` from the live prompt
    surfaces (pass ``campaign_id`` when the campaign standards fragment is part
    of the rotated surface), so no axis is outside the live view.
    """

    from menagerie.crawler.author_executor import author_steering_surface_digest
    from menagerie.crawler.tests.conftest import make_authority_context
    from menagerie.crawler.toolchain_pins import toolchain_identity

    context = make_authority_context(["m_probe"])
    return {
        "author_result_schema_identity": context.author_schema_identity,
        "author_dispatcher_identity": context.author_dispatcher_identity,
        "author_prompt_identity": context.author_prompt_identity,
        "checker_prompt_identity": context.checker_prompt_identity,
        "terminal_rule_identity": context.terminal_policy_identity,
        "publication_policy_identity": context.publication_policy_identity,
        "runner_identity": context.runner_policy_identity,
        "award_closure_identity": context.reducer_policy_identity,
        "toolchain_identity": toolchain_identity(),
        "author_steering_surface_identity": author_steering_surface_digest(
            campaign_id=campaign_id
        ),
    }


#: Historical spelling kept for existing callers/tests.
_live_identities = live_authority_identities
def load_kernel_closure_declaration(path: Path = KERNEL_CLOSURE_DECLARATION_PATH) -> JsonObject:
    """Load and validate the committed kernel-closure declaration."""

    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != KERNEL_CLOSURE_DECLARATION_SCHEMA_VERSION:
        raise RotationLedgerError(
            f"kernel closure declaration {path} declares unknown schema "
            f"{document.get('schema_version')!r}"
        )
    axes = document.get("axes")
    if not isinstance(axes, Mapping):
        raise RotationLedgerError(f"kernel closure declaration {path} names no axes")
    for axis, entry in axes.items():
        _validate_declaration_entry(str(axis), entry)
    return dict(document)


def _validate_declaration_entry(axis: str, entry: Any) -> None:
    if axis not in CODE_AXES:
        raise RotationLedgerError(f"declaration names a non-predicate-replay axis: {axis!r}")
    if not isinstance(entry, Mapping) or not isinstance(entry.get("identity"), str):
        raise RotationLedgerError(f"declaration entry for {axis!r} lacks an identity")
    measurement = entry.get("measurement")
    if measurement is None:
        return
    if not isinstance(measurement, Mapping):
        raise RotationLedgerError(f"declaration entry for {axis!r} has a malformed measurement")
    if measurement.get("class") not in ROTATION_CLASSES | {None}:
        raise RotationLedgerError(f"declaration entry for {axis!r} has an unknown class")
    if measurement.get("measured_by") not in MEASUREMENT_METHODS:
        raise RotationLedgerError(
            f"declaration entry for {axis!r} has an unknown measurement method"
        )


def undeclared_code_axes(
    declaration: Mapping[str, Any], live: Mapping[str, str]
) -> dict[str, tuple[Optional[str], str]]:
    """Return predicate-replay axes whose live identity the declaration does not name.

    ``{axis: (declared_identity_or_None, live_identity)}`` -- the CI red set.
    """

    axes = declaration.get("axes") or {}
    moved: dict[str, tuple[Optional[str], str]] = {}
    for axis in sorted(CODE_AXES):
        declared = axes.get(axis) if isinstance(axes, Mapping) else None
        declared_identity = declared.get("identity") if isinstance(declared, Mapping) else None
        if declared_identity != live[axis]:
            moved[axis] = (declared_identity, live[axis])
    return moved


def _probe_context(repo_root: Path, records_root: Path, intake: Path) -> Any:
    """Build the same authority context ``status`` projects under."""

    from dataclasses import replace

    from menagerie.crawler.checkpoint import canonical_operational_ledger_path
    from menagerie.crawler.cli import _snapshot_authority_context, _snapshot_driver_config
    from menagerie.crawler.driver_admission import restored_environment_generations
    from menagerie.crawler.intake import load_intake_snapshot
    from menagerie.crawler.recordio import scan_jsonl
    from menagerie.crawler.reducer import default_ledger_paths

    snapshot = load_intake_snapshot(intake)
    paths = default_ledger_paths(records_root)
    context = _snapshot_authority_context(snapshot, _snapshot_driver_config(repo_root, snapshot))
    operational_path = canonical_operational_ledger_path(paths.models)
    context = replace(
        context,
        environment_generations=restored_environment_generations(
            context.environment_generations,
            paths.attempts,
            scan_jsonl(operational_path) if operational_path.exists() else (),
        ),
    )
    return snapshot, paths, context


def unconnected_code_axis_moves(
    rows: Sequence[Mapping[str, Any]], live: Mapping[str, str], paths: Any
) -> dict[str, list[str]]:
    """Return ``{axis: [recorded identities]}`` the ledger does not connect to live.

    Set-based over the ledger heads' recorded values (never over a "newest"
    ledger row): every distinct recorded identity on a predicate-replay axis
    whose move to the live identity no row chain reaches.
    """

    from menagerie.crawler.recordio import scan_jsonl
    from menagerie.crawler.reducer import _select_current

    heads = _select_current(scan_jsonl(paths.models))
    moved: dict[str, list[str]] = {}
    for axis in sorted(CODE_AXES):
        recorded_values = sorted(
            {
                str(head["dependency_vector"][axis])
                for head in heads.values()
                if isinstance(head.get("dependency_vector"), Mapping)
                and isinstance(head["dependency_vector"].get(axis), str)
                and head["dependency_vector"][axis].startswith("sha256:")
            }
        )
        unconnected = [
            value
            for value in recorded_values
            if move_class(rows, axis=axis, recorded=value, live=live[axis]) is None
        ]
        if unconnected:
            moved[axis] = unconnected
    return moved


def measured_rows_from_probe(
    report: Mapping[str, Any],
    *,
    moved: Mapping[str, Sequence[str]],
    live: Mapping[str, str],
    tree_sha: str,
    reason: str,
    at: str,
) -> list[JsonObject]:
    """Build the measured rows a probe report licenses for the unconnected moves.

    Raises
    ------
    RotationLedgerError
        If the probe measured zero heads (a class over nothing licenses no row).
    """

    measured_class = report.get("class")
    if measured_class not in ROTATION_CLASSES:
        raise RotationLedgerError(
            f"probe over {report.get('heads')} heads measured no class; a row needs a non-empty "
            "head set on the root it governs"
        )
    rows: list[JsonObject] = []
    for axis, sources in sorted(moved.items()):
        for source in sources:
            rows.append(
                build_rotation(
                    axis=axis,
                    from_identity=source,
                    to_identity=live[axis],
                    rotation_class=str(measured_class),
                    reason=(
                        f"{reason}: {report['heads']} heads ({report['awards']} awards) "
                        f"replayed under tree {tree_sha}; {len(report['flips'])} flips"
                    ),
                    measured_by="predicate_replay",
                    tree_sha=tree_sha,
                    at=at,
                )
            )
    return rows


def _run_probe(args: argparse.Namespace) -> int:
    from menagerie.crawler.recheck_census import predicate_replay_probe

    repo_root = args.repo_root.resolve()
    records_root = _records_root(args)
    snapshot, paths, context = _probe_context(repo_root, records_root, args.intake.resolve())
    report = predicate_replay_probe(
        paths,
        context,
        scope=None if args.all_heads else (item.stable_id for item in snapshot.items),
    )
    live = _live_identities()
    rows = load_rotations(records_root)
    tree_sha = _tree_sha(repo_root, args.tree_sha)
    moved = unconnected_code_axis_moves(rows, live, paths)
    report["tree_sha"] = tree_sha
    report["records_root"] = str(records_root)
    report["intake_snapshot_id"] = snapshot.snapshot_id
    report["scope"] = "all-heads" if args.all_heads else "intake"
    report["unconnected_code_axis_moves"] = moved
    report["live_code_identities"] = {axis: live[axis] for axis in sorted(CODE_AXES)}
    written: list[JsonObject] = []
    if args.write_row and moved:
        for row in measured_rows_from_probe(
            report, moved=moved, live=live, tree_sha=tree_sha, reason=args.reason, at=utc_now()
        ):
            if append_rotation(records_root, row):
                written.append(row)
    report["written_rows"] = written
    if args.report is not None:
        args.report.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if not report["flips"] or args.write_row else 2


def _probe_measurement(probe_report: Mapping[str, Any]) -> JsonObject:
    """Render the disclosure block of one probe report for the declaration."""

    flips = probe_report.get("flips") or []
    return {
        "heads": probe_report.get("heads"),
        "awards": probe_report.get("awards"),
        "flips": len(flips),
        "flipped_members": sorted(str(entry["stable_id"]) for entry in flips),
        "changed_input_stale": len(probe_report.get("changed_input_stale") or []),
        "class": probe_report.get("class"),
        "measured_by": "predicate_replay",
        "records_root": probe_report.get("records_root"),
        "tree_sha": probe_report.get("tree_sha"),
    }


def write_kernel_closure_declaration(
    path: Path,
    *,
    live: Mapping[str, str],
    tree_sha: str,
    reason: str,
    probe_report: Optional[Mapping[str, Any]] = None,
    at: Optional[str] = None,
) -> JsonObject:
    """Rewrite the committed declaration for the live predicate-replay axes.

    Only axes whose identity moved get a new entry; an entry's measurement is
    the probe report given (or ``None``: acknowledged, unmeasured -- rows for
    a real root are always measured on that root by ``probe --write-row``).
    """

    try:
        existing = load_kernel_closure_declaration(path) if path.is_file() else {"axes": {}}
    except RotationLedgerError:
        existing = {"axes": {}}
    axes: dict[str, Any] = dict(existing.get("axes") or {})
    measurement = None if probe_report is None else _probe_measurement(probe_report)
    for axis in sorted(CODE_AXES):
        prior = axes.get(axis) if isinstance(axes.get(axis), Mapping) else {}
        if prior.get("identity") == live[axis]:
            continue
        axes[axis] = {
            "identity": live[axis],
            "tree_sha": tree_sha,
            "at": at or utc_now(),
            "reason": reason,
            "measurement": measurement,
        }
    document = {
        "schema_version": KERNEL_CLOSURE_DECLARATION_SCHEMA_VERSION,
        "comment": (
            "RC-2: the live identity per predicate-replay axis (terminal_rule, "
            "publication_policy, award_closure), acknowledged per kernel edit. The smoke "
            "corpus refuses a tree whose live identity this file does not name. Regenerate "
            f"with `{ACKNOWLEDGE_COMMAND}`. This file is acknowledgement and disclosure; "
            "rows for a records root are measured on that root by `probe --write-row`."
        ),
        "axes": {axis: axes[axis] for axis in sorted(axes)},
    }
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return document


def _run_acknowledge(args: argparse.Namespace) -> int:
    live = _live_identities()
    probe_report = (
        json.loads(args.probe_report.read_text(encoding="utf-8"))
        if args.probe_report is not None
        else None
    )
    document = write_kernel_closure_declaration(
        args.declaration,
        live=live,
        tree_sha=_tree_sha(args.repo_root.resolve(), args.tree_sha),
        reason=args.reason,
        probe_report=probe_report,
    )
    print(json.dumps({"declaration": str(args.declaration), "axes": document["axes"]}, sort_keys=True))
    return 0


def _run_measure_golden(args: argparse.Namespace) -> int:
    records_root = _records_root(args)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    classified = classify_golden_replay(report, axis=args.axis)
    live = _live_identities()
    to_identity = args.to_identity or live[args.axis]
    row = build_rotation(
        axis=args.axis,
        from_identity=args.from_identity,
        to_identity=to_identity,
        rotation_class=str(classified["class"]),
        reason=(
            f"golden replay over {classified['goldens']} goldens: "
            f"{len(classified['flipped'])} recorded accepts flipped"
        ),
        measured_by="golden_replay",
        tree_sha=_tree_sha(args.repo_root.resolve(), args.tree_sha),
        at=utc_now(),
    )
    appended = append_rotation(records_root, row) if args.write_row else False
    print(json.dumps({**classified, "row": row, "appended": appended}, sort_keys=True))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run one rotation-tooling command."""

    args = build_parser().parse_args(argv)
    try:
        if args.command == "print-live":
            print(json.dumps(_live_identities(), sort_keys=True))
            return 0
        if args.command == "acknowledge-closure":
            return _run_acknowledge(args)
        records_root = _records_root(args)
        if args.command == "list":
            rows = load_rotations(records_root)
            print(
                json.dumps(
                    {
                        "ledger": str(rotations_path(records_root)),
                        "rotations": rotation_history(rows),
                        "edge_classes": [
                            {"axis": axis, "from": source, "to": target, "class": resolved}
                            for (axis, source, target), resolved in sorted(edge_classes(rows).items())
                        ],
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "seed":
            result = seed_from_registry(
                records_root,
                tree_sha=_tree_sha(args.repo_root.resolve(), args.tree_sha),
                registry_path=args.registry,
            )
            print(json.dumps(result, sort_keys=True))
            return 0
        if args.command == "probe":
            return _run_probe(args)
        if args.command == "measure-golden":
            return _run_measure_golden(args)
    except (OSError, RotationLedgerError, ValueError) as exc:
        print(f"register-authority-rotation failed: {exc}", file=sys.stderr)
        return 1
    return 1  # pragma: no cover - argparse enforces the closed command set


if __name__ == "__main__":
    raise SystemExit(main())
