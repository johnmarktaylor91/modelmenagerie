"""Prove root placement from a REAL launchd-spawned process (PD-8's validator).

macOS TCC denies launchd-spawned processes access to removable volumes in BOTH
directions (``EPERM``) unless the exact binary holds a per-binary TCC grant —
while the SAME command works from a Terminal-context shell (rehearsal evidence
``r2-tcc-launchd-probe.json``, 2026-08-25: the campaign records root read fine,
the SSD state root refused reads AND writes). A Terminal-context write sentinel
therefore proves nothing about what the wake/resume path can touch. This tool
is the ground-truth instrument: it bootstraps a one-shot user LaunchAgent that
re-runs this module in ``--payload`` mode INSIDE the launchd context, runs the
create+fsync+readback+unlink sentinel over every named root there, and judges
the receipt.

The instrument checks itself two ways, fail-closed:

* the payload records ``getppid()``; a receipt whose writer was not
  launchd-spawned (``ppid != 1``) is refused — a Terminal descendant
  masquerading as the probe proves nothing;
* ``--expect-denied`` roots (e.g. the retired SSD state root) must actually
  refuse with a permission error — a probe context where the negative control
  unexpectedly SUCCEEDS is not the TCC seam this tool exists to measure, so
  the run refuses rather than blessing the placement.

Typed refusal first token: ``tcc-placement-unproven`` (exit 2). Green prints
the verdict JSON and exits 0. The agent is always booted out and its plist
swept, success or failure.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from menagerie.crawler.identity import durable_fsync, utc_now

TCC_PLACEMENT_UNPROVEN = "tcc-placement-unproven"

TCC_PROBE_RECEIPT_SCHEMA_VERSION = "menagerie.crawler.tcc-launchd-probe.v2"

#: A probe syscall that never returns. Observed LIVE (2026-08-25, the Mac crawl host):
#: from a launchd-spawned venv python, the SSD touch does not fast-fail with
#: ``EPERM`` the way the r2 rehearsal probe's ``/bin/zsh``-wrapped system
#: python did — the TCC consent evaluation BLOCKS the syscall indefinitely.
#: For the campaign that is strictly worse than a typed error: a wake act
#: touching the removable volume goes DARK, not red. The payload therefore
#: caps every probe and records this typed disposition on expiry.
PROBE_TIMEOUT_ERROR = "ProbeTimeout"

#: The CLOSED set of write-probe errors that count as launchd-context denial
#: evidence for a negative-control root: the fast typed refusal, and the
#: consent-evaluation hang above. Anything else (missing path, I/O error)
#: is NOT denial evidence and refuses the run.
DENIAL_ERRORS = frozenset({"PermissionError", PROBE_TIMEOUT_ERROR})

#: Per-probe syscall cap; see :data:`PROBE_TIMEOUT_ERROR`.
DEFAULT_PROBE_TIMEOUT_SECONDS = 15.0

#: Label prefix for the one-shot probe agent; the unique token keeps repeated
#: runs from colliding and makes leftover litter attributable.
PROBE_LABEL_PREFIX = "org.menagerie.tcc-probe"


def _xml_escape(value: str) -> str:
    """Escape one string for property-list XML text content."""

    return (
        value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def parse_root_specs(specs: Sequence[str]) -> dict[str, Path]:
    """Parse repeated ``NAME=PATH`` root specifications.

    Parameters
    ----------
    specs:
        Raw ``NAME=PATH`` strings from the command line.

    Returns
    -------
    dict[str, pathlib.Path]
        Root name to absolute path.

    Raises
    ------
    ValueError
        On a missing ``=``, a duplicate name, or a relative path — a relative
        root is meaningless to a launchd-spawned process whose working
        directory is launchd's.
    """

    roots: dict[str, Path] = {}
    for spec in specs:
        name, separator, raw_path = spec.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"root spec {spec!r} is not NAME=PATH")
        if name in roots:
            raise ValueError(f"root spec name {name!r} is repeated")
        path = Path(raw_path)
        if not path.is_absolute():
            raise ValueError(f"root spec {spec!r} must use an absolute path")
        roots[name] = path
    return roots


def _bounded_probe(
    probe: Callable[[], dict[str, Any]],
    *,
    denied_key: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Run one probe with a hard cap, converting a hang into a typed row.

    The probe runs in a daemon thread; on expiry the row records
    :data:`PROBE_TIMEOUT_ERROR` and the payload moves on (the blocked syscall
    thread dies with the process). A hung write probe that later unblocks
    still unlinks its own sentinel through its ``finally``.
    """

    outcome: list[dict[str, Any]] = []
    thread = threading.Thread(target=lambda: outcome.append(probe()), daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if outcome:
        return outcome[0]
    return {
        denied_key: False,
        "error": PROBE_TIMEOUT_ERROR,
        "detail": (
            f"no syscall response within {timeout_seconds:.0f}s; a "
            "launchd-context TCC consent evaluation on a removable volume "
            "BLOCKS indefinitely instead of returning EPERM"
        ),
    }


def _probe_read(path: Path, *, timeout_seconds: float) -> dict[str, Any]:
    """Record whether the acting context can list one directory."""

    def probe() -> dict[str, Any]:
        try:
            entries = os.listdir(path)
        except OSError as exc:
            return {
                "readable": False,
                "error": type(exc).__name__,
                "detail": str(exc)[:200],
            }
        return {"readable": True, "n_entries": len(entries)}

    return _bounded_probe(probe, denied_key="readable", timeout_seconds=timeout_seconds)


def _probe_write(path: Path, *, timeout_seconds: float) -> dict[str, Any]:
    """Run the acting-context create+fsync+readback+unlink sentinel."""

    def probe() -> dict[str, Any]:
        sentinel = path / f".tcc-probe-{os.getpid()}-{secrets.token_hex(8)}"
        payload = secrets.token_bytes(64)
        try:
            with sentinel.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                durable_fsync(handle.fileno())
            matched = sentinel.read_bytes() == payload
        except OSError as exc:
            return {
                "writable": False,
                "error": type(exc).__name__,
                "detail": str(exc)[:200],
            }
        finally:
            try:
                sentinel.unlink(missing_ok=True)
            except OSError:
                pass
        return {"writable": bool(matched)}

    return _bounded_probe(probe, denied_key="writable", timeout_seconds=timeout_seconds)


def run_payload(
    *,
    emit: Path,
    roots: Mapping[str, Path],
    expect_denied: Mapping[str, Path],
    probe_timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run the in-context probes and atomically write the receipt.

    Parameters
    ----------
    emit:
        Receipt destination; written via a same-directory temporary and
        ``os.replace`` so the polling orchestrator never reads a torn JSON.
    roots:
        Roots that must prove readable and writable in this context.
    expect_denied:
        Negative-control roots expected to refuse in this context.
    probe_timeout_seconds:
        Per-probe syscall cap; see :data:`PROBE_TIMEOUT_ERROR`.

    Returns
    -------
    dict[str, Any]
        The written receipt.
    """

    def _rows(candidates: Mapping[str, Path]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for name in sorted(candidates):
            path = candidates[name]
            rows.append(
                {
                    "name": name,
                    "path": str(path),
                    "read": _probe_read(path, timeout_seconds=probe_timeout_seconds),
                    "write": _probe_write(path, timeout_seconds=probe_timeout_seconds),
                }
            )
        return rows

    receipt: dict[str, Any] = {
        "schema_version": TCC_PROBE_RECEIPT_SCHEMA_VERSION,
        "created_at": utc_now(),
        "uid": os.getuid(),
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "launchd_spawned": os.getppid() == 1,
        "xpc_service_name": os.environ.get("XPC_SERVICE_NAME"),
        "fd_soft_limit": resource.getrlimit(resource.RLIMIT_NOFILE)[0],
        "fd_hard_limit": resource.getrlimit(resource.RLIMIT_NOFILE)[1],
        "env_var_count": len(os.environ),
        "env_names": sorted(os.environ),
        "roots": _rows(roots),
        "expect_denied": _rows(expect_denied),
    }
    emit.parent.mkdir(parents=True, exist_ok=True)
    temporary = emit.with_name(f".{emit.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(json.dumps(receipt, indent=1, sort_keys=True).encode("utf-8") + b"\n")
        handle.flush()
        durable_fsync(handle.fileno())
    os.replace(temporary, emit)
    return receipt


def judge_receipt(
    receipt: Mapping[str, Any],
    *,
    roots: Mapping[str, Path],
    expect_denied: Mapping[str, Path],
) -> dict[str, Any]:
    """Judge one payload receipt against the placement contract, fail-closed.

    Parameters
    ----------
    receipt:
        Parsed payload receipt.
    roots, expect_denied:
        The same specifications the payload ran, so a receipt that silently
        dropped a root refuses instead of passing by omission.

    Returns
    -------
    dict[str, Any]
        ``{"ok": bool, "refusals": [...]}``.
    """

    refusals: list[str] = []
    if receipt.get("schema_version") != TCC_PROBE_RECEIPT_SCHEMA_VERSION:
        refusals.append(
            f"{TCC_PLACEMENT_UNPROVEN}: receipt declares schema "
            f"{receipt.get('schema_version')!r}, not {TCC_PROBE_RECEIPT_SCHEMA_VERSION!r}"
        )
        return {"ok": False, "refusals": refusals}
    if receipt.get("launchd_spawned") is not True:
        refusals.append(
            f"{TCC_PLACEMENT_UNPROVEN}: the payload was not launchd-spawned "
            f"(ppid {receipt.get('ppid')!r}); a Terminal-context sentinel proves "
            "nothing about the wake path"
        )
    observed_roots = {str(row.get("name")): row for row in receipt.get("roots", ())}
    for name in sorted(roots):
        row = observed_roots.get(name)
        if row is None or str(row.get("path")) != str(roots[name]):
            refusals.append(
                f"{TCC_PLACEMENT_UNPROVEN}: the receipt carries no probe of root "
                f"{name} ({roots[name]})"
            )
            continue
        if not row.get("read", {}).get("readable"):
            refusals.append(
                f"{TCC_PLACEMENT_UNPROVEN}: root {name} ({roots[name]}) is not "
                f"readable from the launchd context: {row.get('read')}"
            )
        if not row.get("write", {}).get("writable"):
            refusals.append(
                f"{TCC_PLACEMENT_UNPROVEN}: root {name} ({roots[name]}) failed the "
                f"launchd-context write sentinel: {row.get('write')}"
            )
    observed_denied = {str(row.get("name")): row for row in receipt.get("expect_denied", ())}
    for name in sorted(expect_denied):
        row = observed_denied.get(name)
        if row is None or str(row.get("path")) != str(expect_denied[name]):
            refusals.append(
                f"{TCC_PLACEMENT_UNPROVEN}: the receipt carries no probe of the "
                f"negative-control root {name} ({expect_denied[name]})"
            )
            continue
        write = row.get("write", {})
        if write.get("writable") or write.get("error") not in DENIAL_ERRORS:
            refusals.append(
                f"{TCC_PLACEMENT_UNPROVEN}: negative-control root {name} "
                f"({expect_denied[name]}) did not refuse with launchd-context "
                f"denial evidence ({sorted(DENIAL_ERRORS)}): {write}; this "
                "context is not measuring the TCC seam, so the positive "
                "results prove nothing"
            )
    return {"ok": not refusals, "refusals": refusals}


def render_probe_plist(
    *,
    label: str,
    python_executable: str,
    repo_root: Path,
    receipt_path: Path,
    log_root: Path,
    roots: Mapping[str, Path],
    expect_denied: Mapping[str, Path],
    probe_timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> str:
    """Render the one-shot RunAtLoad agent that re-runs this module in payload mode."""

    argv = [
        python_executable,
        "-m",
        "menagerie.crawler.tools.tcc_launchd_probe",
        "--payload",
        "--emit",
        str(receipt_path),
        "--probe-timeout-seconds",
        str(probe_timeout_seconds),
    ]
    for name in sorted(roots):
        argv.extend(["--root", f"{name}={roots[name]}"])
    for name in sorted(expect_denied):
        argv.extend(["--expect-denied", f"{name}={expect_denied[name]}"])
    args = "".join(f"<string>{_xml_escape(arg)}</string>" for arg in argv)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0"><dict>'
        f"<key>Label</key><string>{_xml_escape(label)}</string>"
        f"<key>ProgramArguments</key><array>{args}</array>"
        f"<key>WorkingDirectory</key><string>{_xml_escape(str(repo_root))}</string>"
        "<key>EnvironmentVariables</key><dict>"
        f"<key>PYTHONPATH</key><string>{_xml_escape(str(repo_root))}</string>"
        "</dict>"
        "<key>StandardOutPath</key>"
        f"<string>{_xml_escape(str(log_root / f'{label}.out'))}</string>"
        "<key>StandardErrorPath</key>"
        f"<string>{_xml_escape(str(log_root / f'{label}.err'))}</string>"
        "<key>RunAtLoad</key><true/></dict></plist>\n"
    )


def _launchctl(*args: str) -> subprocess.CompletedProcess[str]:
    """Run one launchctl invocation, capturing output, never raising."""

    return subprocess.run(
        ("/bin/launchctl", *args), capture_output=True, text=True, timeout=60
    )


def run_orchestrator(
    *,
    roots: Mapping[str, Path],
    expect_denied: Mapping[str, Path],
    agent_root: Path,
    scratch_root: Path,
    repo_root: Path,
    python_executable: str,
    timeout_seconds: float,
    probe_timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Bootstrap the probe agent, wait for its receipt, judge it, sweep.

    Parameters
    ----------
    roots, expect_denied:
        The placement contract to prove and its negative controls.
    agent_root:
        Directory launchd loads the one-shot plist from.
    scratch_root:
        Receipt and log destination; must be launchd-writable itself (it is
        part of what the probe proves).
    repo_root:
        Working directory and ``PYTHONPATH`` for the payload.
    python_executable:
        Interpreter the payload runs under.
    timeout_seconds:
        How long to wait for the receipt before refusing.

    Returns
    -------
    dict[str, Any]
        ``{"ok", "refusals", "receipt_path", "receipt", "label"}``.
    """

    token = secrets.token_hex(6)
    label = f"{PROBE_LABEL_PREFIX}.{token}"
    scratch_root.mkdir(parents=True, exist_ok=True)
    agent_root.mkdir(parents=True, exist_ok=True)
    receipt_path = scratch_root / f"{label}.receipt.json"
    plist_path = agent_root / f"{label}.plist"
    plist_path.write_text(
        render_probe_plist(
            label=label,
            python_executable=python_executable,
            repo_root=repo_root,
            receipt_path=receipt_path,
            log_root=scratch_root,
            roots=roots,
            expect_denied=expect_denied,
            probe_timeout_seconds=probe_timeout_seconds,
        ),
        encoding="utf-8",
    )
    refusals: list[str] = []
    receipt: Optional[dict[str, Any]] = None
    domain = f"gui/{os.getuid()}"
    try:
        bootstrap = _launchctl("bootstrap", domain, str(plist_path))
        if bootstrap.returncode != 0:
            refusals.append(
                f"{TCC_PLACEMENT_UNPROVEN}: launchctl bootstrap failed "
                f"(rc {bootstrap.returncode}): {bootstrap.stderr.strip()[:300]}"
            )
        else:
            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                if receipt_path.is_file():
                    try:
                        receipt = json.loads(receipt_path.read_bytes())
                    except (json.JSONDecodeError, OSError):
                        receipt = None
                    if receipt is not None:
                        break
                time.sleep(0.5)
            if receipt is None:
                stderr_log = scratch_root / f"{label}.err"
                detail = ""
                if stderr_log.is_file():
                    detail = stderr_log.read_text(encoding="utf-8", errors="replace")[-300:]
                refusals.append(
                    f"{TCC_PLACEMENT_UNPROVEN}: the launchd-spawned payload wrote no "
                    f"receipt within {timeout_seconds:.0f}s; payload stderr tail: "
                    f"{detail!r}"
                )
    finally:
        _launchctl("bootout", f"{domain}/{label}")
        try:
            plist_path.unlink(missing_ok=True)
        except OSError:
            pass
        residue = _launchctl("print", f"{domain}/{label}")
        if residue.returncode == 0:
            refusals.append(
                f"{TCC_PLACEMENT_UNPROVEN}: probe agent {label} survived bootout; "
                "sweep it before trusting this host's launchd domain"
            )
    if receipt is not None:
        verdict = judge_receipt(receipt, roots=roots, expect_denied=expect_denied)
        refusals.extend(verdict["refusals"])
    return {
        "ok": not refusals,
        "refusals": refusals,
        "label": label,
        "receipt_path": str(receipt_path),
        "receipt": receipt,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the payload (inside launchd) or the orchestrator (anywhere)."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--emit", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="root that must prove readable+writable in the launchd context; repeatable",
    )
    parser.add_argument(
        "--expect-denied",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "negative-control root that must refuse with a permission error in the "
            "launchd context (proves the probe is measuring the TCC seam); repeatable"
        ),
    )
    parser.add_argument(
        "--agent-root",
        type=Path,
        default=Path.home() / "Library" / "LaunchAgents",
        help="where launchd loads the one-shot probe plist from",
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=Path.cwd() / ".tcc-probe",
        help="receipt/log destination (itself part of what the probe proves)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="repository root the payload imports menagerie from",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="interpreter the launchd payload runs under",
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--probe-timeout-seconds",
        type=float,
        default=DEFAULT_PROBE_TIMEOUT_SECONDS,
        help=(
            "per-probe syscall cap inside the payload; a TCC consent evaluation "
            "on a removable volume blocks INDEFINITELY (never EPERM) from this "
            "context, so an uncapped probe would hang the whole run. Size "
            "--timeout-seconds above (hanging probes) x this cap"
        ),
    )
    args = parser.parse_args(argv)
    roots = parse_root_specs(args.root)
    expect_denied = parse_root_specs(args.expect_denied)
    if args.payload:
        if args.emit is None:
            parser.error("--payload requires --emit")
        run_payload(
            emit=args.emit,
            roots=roots,
            expect_denied=expect_denied,
            probe_timeout_seconds=args.probe_timeout_seconds,
        )
        return 0
    if not roots:
        parser.error("at least one --root NAME=PATH is required")
    result = run_orchestrator(
        roots=roots,
        expect_denied=expect_denied,
        agent_root=args.agent_root,
        scratch_root=args.scratch_root,
        repo_root=args.repo_root,
        python_executable=args.python,
        timeout_seconds=args.timeout_seconds,
        probe_timeout_seconds=args.probe_timeout_seconds,
    )
    print(json.dumps(result, indent=1, sort_keys=True))
    if not result["ok"]:
        for refusal in result["refusals"]:
            print(refusal, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
