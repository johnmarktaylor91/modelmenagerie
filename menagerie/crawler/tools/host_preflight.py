"""DX-62 (UU-8): the leg-open host probe pack + armed-guard census.

200a ran legs whose HOST preconditions were never asserted: the daemon context
carried 256 file descriptors (the interactive shell carried 10240), sleep
immunity was a stray process, the records filesystem type was assumed, and no
durable row said which guards were armed when the leg opened. This pack is one
composed leg-open step: every probe registers exactly one
``{name, armed, evidence, refusal_shape}`` row through the W4 registry API
(:class:`menagerie.crawler.tools.launch_preflight.PreflightArmRegistry`), so a
guard cannot exist outside the census and an unarmed guard is disclosed, never
silent.

Probes that belong to OTHER lanes' seams (token validity — WS4's DX-63;
model receipts — WS2; adapter identity — D-47's run_leg) register UNARMED
disclosure rows here; arming them extends :data:`HOST_PREFLIGHT_ARMS` in the
same reviewed change that wires the real probe, exactly per the registry rule.

The pack's JSON output is written for the durable ``leg_open`` row; binding it
into :mod:`menagerie.crawler.leg_generation`'s row schema is the driver-side
seam and lands with that file's owning change, not here.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.code_identity import origins_outside_tree, runner_code_identity
from menagerie.crawler.tools.launch_preflight import (
    PreflightArmRegistry,
    evaluate_roots_contract,
)
from menagerie.crawler.worker_supervisor import current_boot_id

RLIMIT_NOFILE_TOO_LOW = "rlimit-nofile-too-low"
RUNNER_IDENTITY_OUTSIDE_TREE = "runner-identity-outside-tree"
BOOT_IDENTITY_UNAVAILABLE = "boot-identity-unavailable"
RECORDS_FS_TYPE_FORBIDDEN = "records-fs-type-forbidden"
VOLUME_ANCHORS_UNMET = "roots-contract-unmet"
CLI_PIN_UNAVAILABLE = "cli-pin-unavailable"
POWER_ASSERTION_UNHELD = "power-assertion-unheld"

#: The daemon context ships 256 descriptors TODAY; a leg refuses below this.
MINIMUM_NOFILE = 4096

#: Filesystems the records root may live on (UU-9's statfs allowlist: local,
#: journaled, rename-atomic). Network and userspace filesystems break the
#: durability contract every ledger append assumes.
RECORDS_FS_ALLOWLIST = ("apfs", "hfs", "ext4", "xfs", "btrfs")

#: The CLOSED census of leg-open host guards. Every row below registers, armed
#: or explicitly unarmed; extending this vocabulary happens in the same
#: reviewed change that wires the new probe.
HOST_PREFLIGHT_ARMS = (
    "adapter-identity",
    "boot-id",
    "cli-versions",
    "libomp-census",
    "model-receipts",
    "ntp-offset",
    "power-assertion",
    "records-fs-type",
    "rlimit-nofile",
    "runner-identity",
    "token-validity",
    "volume-anchors",
)


def filesystem_type(path: Path) -> Optional[str]:
    """Return the filesystem type name backing one existing path.

    Parameters
    ----------
    path:
        Existing path whose mount's filesystem type is wanted.

    Returns
    -------
    str | None
        Lower-case type name, or ``None`` when it cannot be determined.
    """

    resolved = path.resolve()
    entries: list[tuple[str, str]] = []
    if sys.platform == "darwin":
        try:
            completed = subprocess.run(
                ("/sbin/mount",), check=True, capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        for line in completed.stdout.splitlines():
            match = re.match(r"^.+ on (.+) \(([^,)]+)", line)
            if match is not None:
                entries.append((match.group(1), match.group(2).strip().lower()))
    else:
        try:
            mounts = Path("/proc/mounts").read_text(encoding="utf-8")
        except OSError:
            return None
        for line in mounts.splitlines():
            fields = line.split()
            if len(fields) >= 3:
                entries.append((fields[1], fields[2].lower()))
    best: Optional[tuple[str, str]] = None
    for mount_point, fs_type in entries:
        anchored = Path(mount_point)
        if resolved == anchored or anchored in resolved.parents:
            if best is None or len(mount_point) > len(best[0]):
                best = (mount_point, fs_type)
    return best[1] if best is not None else None


def _probe_rlimit_nofile(
    registry: PreflightArmRegistry, refusals: list[str], *, minimum_nofile: int
) -> None:
    """Raise the descriptor limit toward the hard cap, refuse below floor."""

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    initial = (soft, hard)
    if soft < minimum_nofile:
        target = min(hard, max(minimum_nofile, soft)) if hard > 0 else minimum_nofile
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        except (OSError, ValueError):
            pass
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    registry.register_preflight_arm(
        "rlimit-nofile",
        armed=True,
        evidence={"initial": list(initial), "effective": [soft, hard], "floor": minimum_nofile},
        refusal_shape=RLIMIT_NOFILE_TOO_LOW,
    )
    if soft < minimum_nofile:
        refusals.append(
            f"{RLIMIT_NOFILE_TOO_LOW}: the acting context allows {soft} open "
            f"descriptors (hard cap {hard}); a leg requires >= {minimum_nofile}. "
            "The daemon context ships 256 today — raise the launchd/service "
            "limit, do not launch from this context."
        )


def _probe_boot_id(registry: PreflightArmRegistry, refusals: list[str]) -> None:
    """Record the true boot generation; refuse when none is derivable."""

    try:
        boot_id = current_boot_id()
    except RuntimeError as exc:
        registry.register_preflight_arm(
            "boot-id",
            armed=True,
            evidence={"error": str(exc)},
            refusal_shape=BOOT_IDENTITY_UNAVAILABLE,
        )
        refusals.append(f"{BOOT_IDENTITY_UNAVAILABLE}: {exc}")
        return
    registry.register_preflight_arm(
        "boot-id",
        armed=True,
        evidence={"boot_id": boot_id},
        refusal_shape=BOOT_IDENTITY_UNAVAILABLE,
    )


def _probe_records_fs_type(
    registry: PreflightArmRegistry,
    refusals: list[str],
    *,
    records_root: Path,
    fs_allowlist: Sequence[str],
) -> None:
    """Refuse a records root on a filesystem outside the durability allowlist."""

    fs_type = filesystem_type(records_root) if records_root.exists() else None
    registry.register_preflight_arm(
        "records-fs-type",
        armed=True,
        evidence={"fs_type": fs_type, "allowlist": list(fs_allowlist)},
        refusal_shape=RECORDS_FS_TYPE_FORBIDDEN,
    )
    if fs_type is None or fs_type not in tuple(fs_allowlist):
        refusals.append(
            f"{RECORDS_FS_TYPE_FORBIDDEN}: records root {records_root} is on "
            f"filesystem {fs_type!r}, outside the durability allowlist "
            f"{tuple(fs_allowlist)}"
        )


def _probe_volume_anchors(
    registry: PreflightArmRegistry,
    refusals: list[str],
    *,
    roots: Mapping[str, Optional[Path]],
) -> None:
    """Re-prove every declared root in THIS (acting) process context."""

    contract = evaluate_roots_contract(
        roots, minimum_free_bytes=0, allocation_probe_bytes=0
    )
    registry.register_preflight_arm(
        "volume-anchors",
        armed=True,
        evidence={
            "declared": [row["root"] for row in contract["rows"] if row.get("declared")],
            "refused": len(contract["refusals"]),
        },
        refusal_shape=VOLUME_ANCHORS_UNMET,
    )
    refusals.extend(contract["refusals"])


def _probe_cli_versions(
    registry: PreflightArmRegistry,
    refusals: list[str],
    *,
    cli_pins: Mapping[str, Path],
) -> None:
    """Capture pinned-CLI identities (rotation axes, W32); refuse absent pins."""

    if not cli_pins:
        registry.register_preflight_arm(
            "cli-versions",
            armed=False,
            evidence={"reason": "no CLI pins declared for this leg"},
            refusal_shape=CLI_PIN_UNAVAILABLE,
        )
        return
    versions: dict[str, Any] = {}
    for name in sorted(cli_pins):
        pin = cli_pins[name]
        if not pin.is_file():
            versions[name] = {"pin": str(pin), "present": False}
            refusals.append(
                f"{CLI_PIN_UNAVAILABLE}: pinned CLI {name} is absent at {pin}"
            )
            continue
        try:
            completed = subprocess.run(
                (str(pin), "--version"),
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            reported = (completed.stdout or completed.stderr).strip().splitlines()
            versions[name] = {
                "pin": str(pin),
                "present": True,
                "version": reported[0] if reported else "",
            }
        except (OSError, subprocess.SubprocessError) as exc:
            versions[name] = {"pin": str(pin), "present": True, "error": str(exc)}
            refusals.append(
                f"{CLI_PIN_UNAVAILABLE}: pinned CLI {name} at {pin} failed its "
                f"version probe: {exc}"
            )
    registry.register_preflight_arm(
        "cli-versions",
        armed=True,
        evidence={"pins": versions},
        refusal_shape=CLI_PIN_UNAVAILABLE,
    )


def _probe_power_assertion(
    registry: PreflightArmRegistry,
    refusals: list[str],
    *,
    power_assertion_state: Optional[Mapping[str, Any]],
) -> None:
    """Refuse a Darwin leg whose driver-held sleep immunity is not held."""

    if power_assertion_state is None:
        registry.register_preflight_arm(
            "power-assertion",
            armed=False,
            evidence={"reason": "no assertion state supplied at leg open"},
            refusal_shape=POWER_ASSERTION_UNHELD,
        )
        return
    state = dict(power_assertion_state)
    registry.register_preflight_arm(
        "power-assertion",
        armed=True,
        evidence=state,
        refusal_shape=POWER_ASSERTION_UNHELD,
    )
    if state.get("mechanism") == "caffeinate" and not state.get("held"):
        refusals.append(
            f"{POWER_ASSERTION_UNHELD}: the driver's own sleep immunity is not "
            f"held ({state.get('detail')!r}); 200a's stray-caffeinate immunity "
            "is not a substitute"
        )


def _probe_libomp_census(registry: PreflightArmRegistry) -> None:
    """Disclose OMP/KMP litter (W11's class); strikes are owned elsewhere."""

    kmp_litter = sorted(
        str(path) for path in Path("/private/tmp").glob("__KMP_REGISTERED_LIB_*")
    ) if Path("/private/tmp").is_dir() else []
    omp_environment = sorted(
        name for name in os.environ if name.startswith(("KMP_", "OMP_"))
    )
    registry.register_preflight_arm(
        "libomp-census",
        armed=True,
        evidence={"kmp_litter": kmp_litter, "omp_environment": omp_environment},
        refusal_shape="disclosure-only",
    )


def _probe_runner_identity(
    registry: PreflightArmRegistry,
    refusals: list[str],
    *,
    code_tree_root: Optional[Path],
) -> None:
    """Record the runner's code identity; refuse origins outside the tree."""

    identity = runner_code_identity()
    registry.register_preflight_arm(
        "runner-identity",
        armed=True,
        evidence=identity,
        refusal_shape=RUNNER_IDENTITY_OUTSIDE_TREE,
    )
    if code_tree_root is None:
        return
    outside = origins_outside_tree(identity, code_tree_root)
    if outside:
        refusals.append(
            f"{RUNNER_IDENTITY_OUTSIDE_TREE}: the editable finder resolves "
            f"{', '.join(outside)} outside the declared tree {code_tree_root} "
            "(mid-leg dev edits would silently adjudicate this leg)"
        )


def _register_unarmed_seams(registry: PreflightArmRegistry) -> None:
    """Disclose the probes other lanes own until their seams arm them."""

    for name, reason in (
        ("ntp-offset", "no sanctioned offset measurement wired; UU-12/W33 clock arms"),
        ("token-validity", "credential probe is WS4's DX-63 seam (source_broker)"),
        ("model-receipts", "model-receipt probe is the money lane's receipts seam"),
        ("adapter-identity", "supplied by the single launch path (D-47 run_leg)"),
    ):
        registry.register_preflight_arm(
            name,
            armed=False,
            evidence={"reason": reason},
            refusal_shape="disclosure-only",
        )


def evaluate_host_preflight(
    *,
    records_root: Path,
    roots: Mapping[str, Optional[Path]],
    cli_pins: Optional[Mapping[str, Path]] = None,
    power_assertion_state: Optional[Mapping[str, Any]] = None,
    minimum_nofile: int = MINIMUM_NOFILE,
    fs_allowlist: Sequence[str] = RECORDS_FS_ALLOWLIST,
    code_tree_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Run the composed leg-open host probe pack.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    roots:
        Declared roots to anchor (records, env store, state, tmp).
    cli_pins:
        Pinned absolute CLI paths (rotation axes) to identity-probe.
    power_assertion_state:
        The driver's :class:`~menagerie.crawler.power_assertion.PowerAssertionState`
        mapping, when the caller holds one.
    minimum_nofile:
        Descriptor floor the acting context must clear (after best-effort raise).
    fs_allowlist:
        Filesystem types the records root may live on.

    Returns
    -------
    dict[str, Any]
        ``{"ok": bool, "armed_guard_census": [...], "refusals": [...]}`` —
        the pack the durable ``leg_open`` row embeds.
    """

    registry = PreflightArmRegistry()
    refusals: list[str] = []
    _probe_rlimit_nofile(registry, refusals, minimum_nofile=minimum_nofile)
    _probe_boot_id(registry, refusals)
    _probe_records_fs_type(
        registry, refusals, records_root=records_root, fs_allowlist=fs_allowlist
    )
    _probe_volume_anchors(registry, refusals, roots=roots)
    _probe_cli_versions(registry, refusals, cli_pins=dict(cli_pins or {}))
    _probe_runner_identity(registry, refusals, code_tree_root=code_tree_root)
    _probe_power_assertion(
        registry, refusals, power_assertion_state=power_assertion_state
    )
    _probe_libomp_census(registry)
    _register_unarmed_seams(registry)
    registry.assert_complete(HOST_PREFLIGHT_ARMS)
    return {
        "ok": not refusals,
        "armed_guard_census": registry.census(),
        "refusals": refusals,
    }


def _parse_cli_pin(value: str) -> tuple[str, Path]:
    """Parse one ``name=/abs/path`` CLI pin argument."""

    name, separator, path = value.partition("=")
    if not separator or not name or not path.startswith("/"):
        raise argparse.ArgumentTypeError(
            f"--cli-pin takes name=/absolute/path, got {value!r}"
        )
    return name, Path(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the host probe pack; refuse (exit 2) on any failed probe."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--env-root", type=Path, default=None)
    parser.add_argument("--state-root", type=Path, default=None)
    parser.add_argument("--tmp-root", type=Path, default=None)
    parser.add_argument(
        "--cli-pin",
        action="append",
        type=_parse_cli_pin,
        default=[],
        metavar="NAME=/ABS/PATH",
        help="pinned CLI to identity-probe; repeatable",
    )
    parser.add_argument("--minimum-nofile", type=int, default=MINIMUM_NOFILE)
    parser.add_argument(
        "--code-tree-root",
        type=Path,
        default=None,
        help="tree the launch declares it runs; editable origins outside refuse",
    )
    args = parser.parse_args(argv)
    pack = evaluate_host_preflight(
        records_root=args.records_root,
        roots={
            "records-root": args.records_root,
            "env-root": args.env_root,
            "state-root": args.state_root,
            "tmp-root": args.tmp_root,
        },
        cli_pins=dict(args.cli_pin),
        minimum_nofile=int(args.minimum_nofile),
        code_tree_root=args.code_tree_root,
    )
    print(json.dumps(pack, indent=2))
    if not pack["ok"]:
        for refusal in pack["refusals"]:
            print(refusal, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
