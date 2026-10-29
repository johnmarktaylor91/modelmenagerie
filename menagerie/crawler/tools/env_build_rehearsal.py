"""EN-4 -- build-only rehearsal of committed environment intents, with receipts.

Six of eleven locked intents were NEVER built by the 200a driver (oddballs staged 29
members; detect-misc carries the only compiled-extension ``source_build`` probe;
graph/speech's first driver turn would be the first production pip-overlay install),
so by the campaign's own first-real-run law they are queued first-fire breaks. This
tool converts those paper claims into evidence BEFORE a paid leg depends on them: for
each locked intent it runs the PRODUCTION lifecycle transaction --
materialize -> create (inventory==export byte-validated) -> probe (receipt
byte-compared against the committed family) -> teardown -> residue check -- with a
no-op use callback, and mints a durable rehearsal receipt bound to
(intent, generation, target, machine).

The receipt is written ONLY after the whole transaction is green, so a crash (kill -9
mid-create included) can never leave a receipt for an unproven build; the operator's
create-registration healing makes the re-run converge. ``evaluate_rehearsal_receipts``
is the preflight-arm shape: it refuses any routed locked intent lacking a current
receipt for this (generation, machine), and is intended to be registered through the
launch-preflight registry API (W4) as the ``env-build-rehearsal`` arm.

Usage (real build, launch host, ~1-1.5h for 11 intents)::

    PYTHONPATH=. python -m menagerie.crawler.tools.env_build_rehearsal \
        --state-root /abs/operator-state --runtime-root /abs/runtime \
        [--intent NAME ...] [--target osx-arm64]

    # receipt audit only (no builds):
    PYTHONPATH=. python -m menagerie.crawler.tools.env_build_rehearsal --evaluate ...
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import socket
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from menagerie.crawler.effort import EffortTracker, StageCap
from menagerie.crawler.env_lifecycle import (
    EnvironmentLifecycleError,
    LifecycleResult,
    SequentialEnvironmentLifecycle,
)
from menagerie.crawler.envs import EnvironmentIntent, load_environment_registry
from menagerie.crawler.identity import atomic_replace_bytes, canonical_json_bytes, hash_bytes

EXIT_OK = 0
EXIT_REFUSED = 64

REHEARSAL_RECEIPT_VERSION = "menagerie.crawler.environment-build-rehearsal.v1"

#: One rehearsal build per intent; the generous ceilings exist so the rehearsal
#: measures real cost instead of masking it, while a hung tool still terminates.
_REHEARSAL_STAGE_CAP = StageCap(attempts=1, seconds=45 * 60, bytes=200 * 1024**3)

_SLUG_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")

LifecycleFactory = Callable[[], SequentialEnvironmentLifecycle]


def machine_identity() -> str:
    """Return the machine binding for rehearsal receipts (host + architecture)."""

    return f"{socket.gethostname()}:{platform.machine()}"


def rehearsal_receipts_root(state_root: Path) -> Path:
    """Return the durable receipt directory below the operator state root."""

    return state_root / "rehearsal-receipts"


def _receipt_path(receipts_root: Path, intent: str, target: str, machine: str) -> Path:
    slug = _SLUG_PATTERN.sub("_", f"{intent}--{target}--{machine}")
    return receipts_root / f"{slug}.json"


def _sealed_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    return {**body, "receipt_sha256": hash_bytes(canonical_json_bytes(body))}


def rehearse_intents(
    intents: Sequence[EnvironmentIntent],
    *,
    lifecycle_factory: LifecycleFactory,
    receipts_root: Path,
    machine: str,
    keep_going: bool = False,
) -> dict[str, Any]:
    """Run the build-only rehearsal for each locked intent and mint receipts.

    Parameters
    ----------
    intents:
        Loaded intents to rehearse; every one must be locked with a proven
        generation (an unlocked intent is a typed refusal, never a skip).
    lifecycle_factory:
        Builds ONE fresh production lifecycle per intent, so effort accounting
        and the one-active-environment invariant reset between intents.
    receipts_root:
        Durable receipt directory (created if absent).
    machine:
        Machine binding recorded into each receipt.
    keep_going:
        Continue to remaining intents after a failure instead of stopping.

    Returns
    -------
    dict[str, Any]
        Per-intent outcomes: minted receipt paths and typed failures.
    """

    receipts_root.mkdir(parents=True, exist_ok=True)
    outcomes: list[dict[str, Any]] = []
    for intent in intents:
        if intent.lock.status != "locked" or intent.generation is None:
            outcomes.append(
                {
                    "intent": intent.name,
                    "status": "refused",
                    "detail": f"intent is not a proven locked generation ({intent.lock.status})",
                }
            )
            if keep_going:
                continue
            break
        started = time.monotonic()
        try:
            result: LifecycleResult = lifecycle_factory().run(intent, use=lambda _p, _r: None)
        except EnvironmentLifecycleError as exc:
            outcomes.append(
                {
                    "intent": intent.name,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "detail": str(exc)[:1_500],
                }
            )
            if keep_going:
                continue
            break
        receipt = _sealed_receipt(
            {
                "schema_version": REHEARSAL_RECEIPT_VERSION,
                "intent": intent.name,
                "generation": intent.generation,
                "target": intent.lock.target,
                "machine": machine,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "created_at_unix": time.time(),
                "lifecycle": asdict(result),
            }
        )
        path = _receipt_path(receipts_root, intent.name, intent.lock.target, machine)
        atomic_replace_bytes(path, canonical_json_bytes(receipt) + b"\n")
        outcomes.append({"intent": intent.name, "status": "rehearsed", "receipt": str(path)})
    return {
        "machine": machine,
        "receipts_root": str(receipts_root),
        "outcomes": outcomes,
        "all_rehearsed": bool(outcomes)
        and all(outcome["status"] == "rehearsed" for outcome in outcomes),
    }


def _load_receipt(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if _sealed_receipt(payload).get("receipt_sha256") != payload.get("receipt_sha256"):
        return None
    return payload


def evaluate_rehearsal_receipts(
    intents: Sequence[EnvironmentIntent],
    *,
    receipts_root: Path,
    machine: str,
) -> dict[str, Any]:
    """Adjudicate rehearsal coverage for the routed intents on this machine.

    This is the preflight-arm predicate (W4 registry name:
    ``env-build-rehearsal``): a routed LOCKED intent whose current
    (generation, target, machine) carries no green rehearsal receipt is a
    refusal. Unlocked intents are refused here too -- an intent with nothing
    provable to rehearse must be refused by the lockless-admission arm (EN-2),
    never silently passed by this one.

    Returns
    -------
    dict[str, Any]
        ``armed`` verdict plus per-intent refusal evidence rows.
    """

    refusals: list[dict[str, str]] = []
    receipts: dict[str, str] = {}
    for intent in intents:
        if intent.lock.status != "locked" or intent.generation is None:
            refusals.append(
                {
                    "intent": intent.name,
                    "reason": "not-locked",
                    "detail": f"lock status {intent.lock.status!r}",
                }
            )
            continue
        path = _receipt_path(receipts_root, intent.name, intent.lock.target, machine)
        payload = _load_receipt(path)
        if payload is None:
            refusals.append(
                {
                    "intent": intent.name,
                    "reason": "missing-or-invalid-receipt",
                    "detail": str(path),
                }
            )
            continue
        expected = {
            "schema_version": REHEARSAL_RECEIPT_VERSION,
            "intent": intent.name,
            "generation": intent.generation,
            "target": intent.lock.target,
            "machine": machine,
        }
        mismatched = {
            key: (payload.get(key), value)
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if mismatched:
            refusals.append(
                {
                    "intent": intent.name,
                    "reason": "stale-receipt",
                    "detail": json.dumps(mismatched, sort_keys=True, default=str),
                }
            )
            continue
        receipts[intent.name] = str(path)
    return {
        "arm": "env-build-rehearsal",
        "machine": machine,
        "armed": not refusals,
        "refusals": refusals,
        "receipts": receipts,
    }


def _production_lifecycle_factory(
    *,
    command: Sequence[str],
    runtime_root: Path,
    tmp_root: Path | None,
) -> LifecycleFactory:
    """Wire the production backend with rehearsal-scoped in-memory effort.

    Rehearsals never write the campaign's environment-turn effort ledger: they
    are pre-launch evidence gathering, not campaign turns, and charging them
    would corrupt the turn accounting D-60 audits.
    """

    from menagerie.crawler.driver_admission import CommandEnvironmentBackend

    def build() -> SequentialEnvironmentLifecycle:
        return SequentialEnvironmentLifecycle(
            CommandEnvironmentBackend(tuple(command), tmp_root=tmp_root),
            EffortTracker({"environment": _REHEARSAL_STAGE_CAP}),
            env_root=runtime_root / "envs",
            tmp_root=tmp_root,
        )

    return build


def main(argv: Sequence[str] | None = None) -> int:
    """Rehearse (or audit) every routed locked intent and print the outcome."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs-root", type=Path, default=None)
    parser.add_argument("--target", default="osx-arm64")
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--tmp-root", type=Path, default=None)
    parser.add_argument("--intent", action="append", default=None)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "--command",
        nargs="+",
        default=(sys.executable, "-m", "menagerie.crawler.operator_environment"),
        help="environment-operator wrapper argv (production default)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="audit existing receipts only; build nothing",
    )
    args = parser.parse_args(argv)
    if not args.state_root.is_absolute():
        print("--state-root must be absolute", file=sys.stderr)
        return EXIT_REFUSED
    registry = (
        load_environment_registry(args.envs_root, target=args.target)
        if args.envs_root is not None
        else load_environment_registry(target=args.target)
    )
    if args.intent:
        unknown = sorted(set(args.intent) - set(registry.intents))
        if unknown:
            print(f"unknown intents: {unknown}", file=sys.stderr)
            return EXIT_REFUSED
        selected = [registry.intents[name] for name in args.intent]
    else:
        selected = [
            intent
            for intent in registry.intents.values()
            if intent.lock.status == "locked" and intent.generation is not None
        ]
    machine = machine_identity()
    receipts_root = rehearsal_receipts_root(args.state_root)
    if args.evaluate:
        verdict = evaluate_rehearsal_receipts(
            selected, receipts_root=receipts_root, machine=machine
        )
        print(json.dumps(verdict, indent=2, sort_keys=True))
        return EXIT_OK if verdict["armed"] else EXIT_REFUSED
    if args.runtime_root is None or not args.runtime_root.is_absolute():
        print("--runtime-root (absolute) is required to build", file=sys.stderr)
        return EXIT_REFUSED
    outcome = rehearse_intents(
        selected,
        lifecycle_factory=_production_lifecycle_factory(
            command=tuple(args.command),
            runtime_root=args.runtime_root,
            tmp_root=args.tmp_root,
        ),
        receipts_root=receipts_root,
        machine=machine,
        keep_going=args.keep_going,
    )
    print(json.dumps(outcome, indent=2, sort_keys=True))
    return EXIT_OK if outcome["all_rehearsed"] else EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
