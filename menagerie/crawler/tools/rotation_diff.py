"""Rotation-control toolchain: TW-CLOSURE-DIFF, stale-head census, batch seal.

Identity-axis rotation is CERTAIN if unmanaged: the award closure hashes the
transitive semantic closure of the parent/reducer code, so any merge touching
a closure-reachable symbol rotates ``award_closure_identity`` — and an
unregistered rotation strands every carried head at rung end (the leg-29
precedent: 139 stale heads, refusal at the worst possible latency).

This tool makes rotation a MANAGED event:

- ``diff``: pre/post ``_award_closure_from_bytes`` comparison between two git
  revisions (or a revision and the working tree). Nonzero exit on rotation so
  the merge queue can refuse a rotation-bearing merge outside the rotation
  batch. The identity authority is the EXECUTING tree's closure function
  applied to both byte snapshots — a pure function of bytes.
- ``self-test``: plants a rotation into the head snapshot and asserts the
  diff DETECTS it, then asserts determinism on identical snapshots. A tool
  that cannot detect a planted rotation is vacuous; the self-test refuses it.
- ``census``: stale-head census — for every closed rotation axis, compare
  the newest registered head in the append-only rotation registry against
  the supplied current identities (the award-closure axis is derived through
  the register tool's live identity, the space the heads were registered
  in — item 11). Unregistered drift is a stale head; the census fails closed
  on any.
- ``seal`` / ``verify-seal``: the Wave-2 rotation-batch BATCH SEAL — one
  artifact binding tree sha, closure identity, registry digest, and the
  enumerated head set, digest-bound. ``verify-seal`` recomputes everything
  and refuses ANY post-seal mutation (tree, closure, registry, or artifact
  bytes).

Registration-freshness as a launch-preflight arm registers through the
preflight registry API when it lands (WS10/W4); until then ``census`` is the
callable authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

SEAL_SCHEMA_VERSION = "menagerie.crawler.rotation-batch-seal.v1"
CENSUS_SCHEMA_VERSION = "menagerie.crawler.stale-head-census.v1"

#: Files whose change can rotate the award closure: every top-level crawler
#: module plus the closure schema set. The merge queue treats a diff touching
#: any of these as rotation-surface and runs ``diff``.
ROTATION_SURFACE_PREFIX = "menagerie/crawler/"


class RotationDiffError(RuntimeError):
    """Raised when a snapshot, census, or seal cannot be produced or verified."""


def _git(repo_root: Path, *arguments: str) -> str:
    """Run one read-only git query and return stripped stdout."""

    try:
        result = subprocess.run(
            ("git", *arguments),
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RotationDiffError(f"git {' '.join(arguments)} failed: {exc}") from exc
    return result.stdout.strip()


def _git_bytes(repo_root: Path, *arguments: str) -> bytes:
    """Run one read-only git query and return raw stdout bytes."""

    try:
        result = subprocess.run(("git", *arguments), cwd=repo_root, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RotationDiffError(f"git {' '.join(arguments)} failed: {exc}") from exc
    return result.stdout


def _closure_authorities() -> tuple[Any, tuple[str, ...]]:
    """Import the executing tree's closure function and schema list."""

    from menagerie.crawler.driver_admission import (
        _AWARD_CLOSURE_SCHEMAS,
        _award_closure_from_bytes,
    )

    return _award_closure_from_bytes, tuple(_AWARD_CLOSURE_SCHEMAS)


def closure_snapshot(
    repo_root: Path, rev: Optional[str]
) -> tuple[tuple[tuple[str, bytes], ...], tuple[tuple[str, bytes], ...]]:
    """Extract the closure input bytes at one revision (None = working tree).

    Returns
    -------
    tuple
        ``(source_items, schema_items)`` in the exact shape
        ``_award_closure_from_bytes`` consumes: module-relative names and
        exact bytes for every top-level crawler module and closure schema.
    """

    _, schemas = _closure_authorities()
    crawler = repo_root / "menagerie" / "crawler"
    if rev is None:
        source_items = tuple(
            (path.name, path.read_bytes()) for path in sorted(crawler.glob("*.py"))
        )
        schema_items = tuple((relative, (crawler / relative).read_bytes()) for relative in schemas)
        return source_items, schema_items
    listing = _git(repo_root, "ls-tree", "--full-tree", "-r", rev, "--", "menagerie/crawler")
    sources: list[tuple[str, bytes]] = []
    for line in listing.splitlines():
        _, _, path = line.partition("\t")
        parts = Path(path).parts
        if len(parts) == 3 and path.endswith(".py"):
            sources.append((Path(path).name, _git_bytes(repo_root, "show", f"{rev}:{path}")))
    schema_items = tuple(
        (
            relative,
            _git_bytes(repo_root, "show", f"{rev}:menagerie/crawler/{relative}"),
        )
        for relative in schemas
    )
    return tuple(sorted(sources)), schema_items


def closure_identity_at(repo_root: Path, rev: Optional[str]) -> str:
    """Compute the award-closure identity of one revision's byte snapshot."""

    closure_from_bytes, _ = _closure_authorities()
    source_items, schema_items = closure_snapshot(repo_root, rev)
    return str(closure_from_bytes(source_items, schema_items))


def tw_closure_diff(repo_root: Path, base_rev: str, head_rev: Optional[str]) -> dict[str, Any]:
    """Diff the award-closure identity between two trees.

    Returns a report carrying both identities, whether the closure ROTATED,
    and the changed rotation-surface files (file-level moved-component view).
    """

    base_identity = closure_identity_at(repo_root, base_rev)
    head_identity = closure_identity_at(repo_root, head_rev)
    diff_args = ["diff", "--name-only", base_rev]
    if head_rev is not None:
        diff_args.append(head_rev)
    diff_args += ["--", "menagerie/crawler"]
    changed = [
        line
        for line in _git(repo_root, *diff_args).splitlines()
        if line.startswith(ROTATION_SURFACE_PREFIX)
    ]
    return {
        "base_rev": base_rev,
        "head_rev": head_rev if head_rev is not None else "<working-tree>",
        "base_closure_identity": base_identity,
        "head_closure_identity": head_identity,
        "rotated": base_identity != head_identity,
        "changed_rotation_surface_files": changed,
    }


def run_self_test(repo_root: Path) -> list[str]:
    """Prove the diff can DETECT a planted rotation (a blind tool refuses).

    Returns the list of failures (empty = self-test passed).
    """

    closure_from_bytes, _ = _closure_authorities()
    source_items, schema_items = closure_snapshot(repo_root, None)
    baseline = closure_from_bytes(source_items, schema_items)
    failures: list[str] = []
    if closure_from_bytes(source_items, schema_items) != baseline:
        failures.append("closure identity is nondeterministic on identical snapshots")
    # Plant a semantic rotation into a KNOWN closure schema input: schema
    # bytes are hashed unconditionally, so this MUST rotate the identity.
    planted_schemas = tuple((name, data + b"\n// planted-rotation") for name, data in schema_items)
    if closure_from_bytes(source_items, planted_schemas) == baseline:
        failures.append("planted schema rotation was NOT detected — the closure diff is vacuous")
    # Plant a SEMANTIC source rotation into one closure ROOT symbol: insert a
    # statement at the top of the root function's body — detected iff the
    # closure genuinely walks source semantics.
    import ast

    from menagerie.crawler.driver_admission import _AWARD_CLOSURE_SYMBOLS

    root_relative, root_symbol = next(
        (relative, symbols[0])
        for relative, symbols in sorted(_AWARD_CLOSURE_SYMBOLS.items())
        if symbols
    )
    root_name = Path(root_relative).name
    sources = dict(source_items)
    tree = ast.parse(sources[root_name].decode("utf-8"))
    outer_name, _, method_name = root_symbol.partition(".")
    target = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name == outer_name
    )
    if method_name:
        target = next(
            node
            for node in target.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == method_name
        )
    first_statement = target.body[0]
    lines = sources[root_name].decode("utf-8").splitlines(keepends=True)
    indent = " " * first_statement.col_offset
    lines.insert(first_statement.lineno - 1, f"{indent}_planted_rotation_sentinel = 1\n")
    sources[root_name] = "".join(lines).encode("utf-8")
    planted_sources = tuple(sorted(sources.items()))
    if closure_from_bytes(planted_sources, schema_items) == baseline:
        failures.append(
            f"planted source rotation in closure root {root_relative}:{root_symbol} "
            "was NOT detected"
        )
    # Negative control: a comment-only change to the same root file must NOT
    # rotate (the closure hashes semantics, not bytes).
    commented = dict(source_items)
    commented[root_name] = commented[root_name] + b"\n# planted non-semantic comment\n"
    if closure_from_bytes(tuple(sorted(commented.items())), schema_items) != baseline:
        failures.append("a comment-only change rotated the closure (over-sensitive)")
    return failures


def _registry_rows(records_root: Path) -> tuple[list[dict[str, Any]], str]:
    """Load rotation registrations plus the registry content digest."""

    from menagerie.crawler.rotation_registry import (
        load_rotation_registrations,
        rotation_registry_path,
    )

    path = rotation_registry_path(records_root)
    if not path.is_file():
        return [], hashlib.sha256(b"").hexdigest()
    rows = [dict(row) for row in load_rotation_registrations(path)]
    return rows, hashlib.sha256(path.read_bytes()).hexdigest()


def stale_head_census(
    repo_root: Optional[Path],
    records_root: Path,
    current_identities: Mapping[str, str],
) -> dict[str, Any]:
    """Compare every axis's newest registered head against its current identity.

    Every axis's current identity comes from the caller (the driver close
    context is their computation authority); ``award_closure_identity`` is
    derived here when the caller does not supply it, through the register
    tool's live-identity function -- the SAME derivation ``--print-live``
    prints and every registry head was registered from (item 11, D-17 one-
    derivation class). The former native tree hash
    (:func:`closure_identity_at`) is a different function of different bytes
    and made this census read the axis stale forever; it survives only for
    the TW-CLOSURE-DIFF and seal artifacts, never for the registry census.
    ``repo_root`` is accepted for signature compatibility and no longer
    steers the axis. An axis whose current identity differs from its newest
    registered head — or that has a current identity and NO registration —
    is a STALE HEAD and fails the census.
    """

    from menagerie.crawler.rotation_registry import ROTATION_AXES, rotation_edge_census

    del repo_root  # item 11: the registry census never hashes a tree natively
    current = dict(current_identities)
    if "award_closure_identity" not in current:
        from menagerie.crawler.tools.register_authority_rotation import (
            live_authority_identities,
        )

        current["award_closure_identity"] = str(
            live_authority_identities()["award_closure_identity"]
        )
    unknown = sorted(set(current) - set(ROTATION_AXES))
    if unknown:
        raise RotationDiffError(f"census axes outside the closed vocabulary: {unknown}")
    rows, registry_digest = _registry_rows(records_root)
    newest_registered: dict[str, str] = {}
    for row in rows:
        newest_registered[str(row["axis"])] = str(row["to_identity"])
    axis_rows: list[dict[str, Any]] = []
    stale: list[str] = []
    for axis in sorted(ROTATION_AXES):
        current_identity = current.get(axis)
        registered = newest_registered.get(axis)
        if current_identity is None:
            status = "not-computed"
        elif registered is None:
            status = "stale-head-unregistered"
        elif registered != current_identity:
            status = "stale-head-drifted"
        else:
            status = "fresh"
        if status.startswith("stale-head"):
            stale.append(axis)
        axis_rows.append(
            {
                "axis": axis,
                "current_identity": current_identity,
                "newest_registered_head": registered,
                "status": status,
            }
        )
    return {
        "schema_version": CENSUS_SCHEMA_VERSION,
        "status": "passed" if not stale else "failed",
        "stale_axes": stale,
        "axes": axis_rows,
        "registry_sha256": registry_digest,
        # Same-edge class disclosure (fable-F3 / sol-04): every edge whose
        # rows were reclassified (superseded) or still conflict is named
        # here, so a class downgrade is never invisible to a census reader.
        # Pure disclosure -- freshness status and the licensing debt digest
        # deliberately do not consume it (this census keys on to_identity).
        "reclassified_edges": [
            edge
            for edge in rotation_edge_census(rows)
            if edge["conflicted"] or edge["superseded_registration_ids"]
        ],
    }


def mint_seal(
    repo_root: Path, records_root: Path, current_identities: Mapping[str, str]
) -> dict[str, Any]:
    """Mint the rotation-batch seal: tree + closure + registry + heads, bound.

    The seal is mintable only over a CLEAN census (zero stale heads): sealing
    a batch with unregistered drift would notarize exactly the defect the
    seal exists to prevent.
    """

    census = stale_head_census(repo_root, records_root, current_identities)
    if census["status"] != "passed":
        raise RotationDiffError(f"seal-refused: stale heads present: {census['stale_axes']}")
    tree_sha = _git(repo_root, "rev-parse", "HEAD^{tree}")
    payload: dict[str, Any] = {
        "schema_version": SEAL_SCHEMA_VERSION,
        "tree_sha": tree_sha,
        "closure_identity": closure_identity_at(repo_root, None),
        "registry_sha256": census["registry_sha256"],
        "heads": {
            row["axis"]: row["current_identity"]
            for row in census["axes"]
            if row["current_identity"] is not None
        },
        "sealed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    canonical = json.dumps(
        {key: value for key, value in payload.items() if key != "seal_sha256"},
        sort_keys=True,
        separators=(",", ":"),
    )
    payload["seal_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def verify_seal(repo_root: Path, records_root: Path, seal_path: Path) -> tuple[bool, str]:
    """Recompute every seal binding; ANY post-seal mutation refuses typed."""

    try:
        payload = json.loads(seal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"seal-invalid: {seal_path} unreadable: {exc}"
    if not isinstance(payload, dict) or payload.get("schema_version") != SEAL_SCHEMA_VERSION:
        return False, "seal-invalid: not a rotation-batch seal artifact"
    canonical = json.dumps(
        {key: value for key, value in payload.items() if key != "seal_sha256"},
        sort_keys=True,
        separators=(",", ":"),
    )
    if payload.get("seal_sha256") != hashlib.sha256(canonical.encode("utf-8")).hexdigest():
        return False, "seal-mutated: artifact bytes do not match their seal digest"
    tree_sha = _git(repo_root, "rev-parse", "HEAD^{tree}")
    if payload.get("tree_sha") != tree_sha:
        return False, (f"seal-mutated: sealed tree {payload.get('tree_sha')} != current {tree_sha}")
    closure = closure_identity_at(repo_root, None)
    if payload.get("closure_identity") != closure:
        return False, "seal-mutated: award closure rotated after the seal"
    _, registry_digest = _registry_rows(records_root)
    if payload.get("registry_sha256") != registry_digest:
        return False, "seal-mutated: rotation registry changed after the seal"
    return True, f"seal-verified: tree {tree_sha} closure {closure[:16]}…"


def _load_identities(path: Optional[Path]) -> dict[str, str]:
    """Load the caller-supplied current-identity map (may be empty)."""

    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RotationDiffError(f"{path} is not a JSON object of axis identities")
    return {str(key): str(item) for key, item in value.items()}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run one rotation-control command."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    diff = subparsers.add_parser("diff", help="TW-CLOSURE-DIFF between two trees")
    diff.add_argument("--repo-root", type=Path, default=Path.cwd())
    diff.add_argument("--base", required=True)
    diff.add_argument("--head", default=None, help="revision; omit for working tree")
    self_test = subparsers.add_parser("self-test", help="planted-rotation detection proof")
    self_test.add_argument("--repo-root", type=Path, default=Path.cwd())
    census = subparsers.add_parser("census", help="stale-head census vs the registry")
    census.add_argument("--repo-root", type=Path, default=Path.cwd())
    census.add_argument("--records-root", type=Path, required=True)
    census.add_argument("--identities", type=Path, default=None)
    seal = subparsers.add_parser("seal", help="mint the rotation-batch seal")
    seal.add_argument("--repo-root", type=Path, default=Path.cwd())
    seal.add_argument("--records-root", type=Path, required=True)
    seal.add_argument("--identities", type=Path, default=None)
    seal.add_argument("--out", type=Path, required=True)
    verify = subparsers.add_parser("verify-seal", help="refuse any post-seal mutation")
    verify.add_argument("--repo-root", type=Path, default=Path.cwd())
    verify.add_argument("--records-root", type=Path, required=True)
    verify.add_argument("--seal", type=Path, required=True)
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    if args.command == "diff":
        report = tw_closure_diff(repo_root, args.base, args.head)
        print(json.dumps(report, indent=2, sort_keys=True))
        if report["rotated"]:
            print(
                "rotation-detected: award closure rotated; this merge belongs in "
                "the sealed rotation batch, not an ordinary slot",
                file=sys.stderr,
            )
            return 2
        return 0
    if args.command == "self-test":
        failures = run_self_test(repo_root)
        for failure in failures:
            print(f"self-test failure: {failure}", file=sys.stderr)
        print("self-test: " + ("FAILED" if failures else "passed"))
        return 2 if failures else 0
    if args.command == "census":
        report = stale_head_census(repo_root, args.records_root, _load_identities(args.identities))
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "passed" else 2
    if args.command == "seal":
        payload = mint_seal(repo_root, args.records_root, _load_identities(args.identities))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")
        print(f"sealed: {args.out}")
        return 0
    ok, message = verify_seal(repo_root, args.records_root, args.seal)
    print(message)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
