"""Scripted environment-operator CLI for rotation end-to-end tests.

Implements exactly the operator wrapper protocol ``CommandEnvironmentBackend``
speaks -- ``solve`` / ``create`` / ``probe`` / ``remove`` / ``owned-paths``
actions with JSON stdout -- while replaying one deterministic fake solve
(``rotation_transaction_support.solve_fixture``) selected by the
``MENAGERIE_SCRIPTED_OPERATOR_TAG`` environment variable. This lets
``tools/rotate_setup_artifacts.main`` run its production wiring (argv
construction, ``--state-root`` threading, per-intent transaction, campaign
republication) against a real subprocess boundary without any solver.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from menagerie.crawler.identity import canonical_json_bytes
from menagerie.crawler.tests.rotation_transaction_support import (
    cas_artifact_path,
    populate_solve_commit,
    solve_fixture,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("action")
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args(argv)

    fixture = solve_fixture(os.environ.get("MENAGERIE_SCRIPTED_OPERATOR_TAG", "new"))

    if args.action == "solve":
        environment_file = Path(args.arguments[0])
        intent = environment_file.parent.name
        populate_solve_commit(args.state_root, intent, fixture)
        solve_root = args.state_root / "solves" / intent / "osx-arm64"
        print(
            json.dumps(
                {
                    "lock_path": str(solve_root / "explicit.lock"),
                    "resolved_export_path": str(solve_root / "resolved.json"),
                    "artifacts": [
                        {
                            "url": fixture.url,
                            "path": str(cas_artifact_path(args.state_root, fixture)),
                            "sha256": fixture.artifact_sha256,
                        }
                    ],
                    "artifact_bytes": len(fixture.artifact_bytes),
                    "elapsed_seconds": 0.1,
                }
            )
        )
        return 0
    if args.action == "create":
        lock_file, prefix = Path(args.arguments[0]), Path(args.arguments[1])
        if lock_file.read_bytes() != fixture.lock_bytes:
            print("staged lock does not match the scripted solve", file=sys.stderr)
            return 64
        meta = prefix / "conda-meta"
        meta.mkdir(parents=True, exist_ok=True)
        (meta / "python-3.11.json").write_bytes(
            canonical_json_bytes(
                {
                    "name": "python",
                    "version": "3.11",
                    "build": f"h{fixture.tag}_0",
                    "url": fixture.url,
                    "sha256": fixture.artifact_sha256,
                }
            )
        )
        print(json.dumps({"created": str(prefix)}))
        return 0
    if args.action == "probe":
        request = json.loads(args.arguments[1])
        results = [
            {"name": f"import:{name}", "passed": True, "detail": "ok"}
            for name in request.get("imports", [])
        ]
        results.extend(
            {
                "name": f"export:{check['module']}:{check['attribute']}",
                "passed": True,
                "detail": "ok",
            }
            for check in request.get("export_checks", [])
        )
        results.extend(
            {"name": f"source-build:{build['name']}", "passed": True, "detail": "ok"}
            for build in request.get("source_build", [])
        )
        print(json.dumps({"results": results, "elapsed_seconds": 0.1}))
        return 0
    if args.action == "remove":
        shutil.rmtree(Path(args.arguments[0]), ignore_errors=True)
        print(json.dumps({"removed": args.arguments[0]}))
        return 0
    if args.action == "owned-paths":
        print(json.dumps({"paths": [args.arguments[0]]}))
        return 0
    print(f"unknown scripted operator action: {args.action}", file=sys.stderr)
    return 64


if __name__ == "__main__":
    raise SystemExit(main())
