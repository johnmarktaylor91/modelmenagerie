"""Launch preflight: prove a promised environment rotation actually rotated.

Rung 12's zero-runs mystery: two long legs were launched on the belief that the
wave-9 ``environment.yml`` edits had rotated the quarantined core environment,
but the park/quarantine identity is computed over the FOUR COMMITTED SETUP
ARTIFACTS (``envs/<intent>/locks/<target>.lock`` / ``.resolved.json`` /
``.resolved.sha256`` / ``.probes.json``), none of which changed -- so every
rebuild park row remained provably live and the deferral valve re-claimed each
re-drive before env/worker/reducer ever ran. Nothing in the launch path checked
the promise.

This preflight is that check. Given a campaign records root, it loads every
current ``deferred:environment-rebuild-required`` row and evaluates the EXACT
production claimant -- ``driver_admission._deferral_row_fact_still_holds`` --
against the environment registry the launch would load. A row whose fact still
holds would be re-claimed again; a row whose fact no longer holds (changed
setup-artifact identity, or an explicit quarantine clearance event) releases.

Riding C-ENV-LOCKS, the preflight also asserts the cold-start preconditions of
the committed-locks-as-inputs env lane:

* **store mount** (``--env-root``): the environment store resolves to a real
  mounted directory -- a dangling symlink otherwise dies deep inside the lane
  as an untyped ``FileExistsError``;
* **CAS completeness** (``--state-root``): every artifact of every committed
  lock family (and every pinned overlay wheel) is present and digest-verified
  in the campaign CAS -- the offline build-from-committed path can materialize
  nothing at launch time;
* **disk authority** (``--tmp-root`` when temporary staging is explicit): the
  shared authority measures the environment store, operator state, and actual
  launch temp root, then applies the floor to the lowest-free filesystem;
* **lockless-family refusal**: every registry intent carries a proven
  ``locked`` family, because the lane refuses to solve in place; intents
  deliberately routed away from model work must be named with
  ``--allow-lockless`` so the exemption is explicit in the launch log.

Riding C-ROTATION-TXN, the preflight also validates every locked family
against its committed rotation provenance (T3-T7): per-member digest binding
(a mixed family refuses loudly instead of releasing parks), ``spec_sha256``
currency (a post-rotation ``environment.yml`` edit reads stale), per-intent
probe-receipt currency (never the structurally always-stale whole-file
``probes.yml`` hash), and completed flat materialization for manifest-backed
publications.

Default contract: REFUSE (exit 2) with a typed first token
(``rotation-did-not-rotate``, ``store-mount-unavailable``, ``cas-incomplete``,
``lockless-family``, ``mixed-family``, ``stale-spec``, ``stale-probe-receipt``,
``family-materialization-pending``, ``state-placement-unsafe``) for each
failed assert. ``--report-only`` downgrades every refusal to a report. Omitting
``--env-root``/``--state-root`` skips those asserts LOUDLY (a warning naming
the unchecked precondition).
"""

from __future__ import annotations

import argparse
import json
import os
import pwd
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Optional, Sequence

from menagerie.crawler.driver_admission import (
    _deferral_row_fact_still_holds,
    _intent_setup_artifact_identity,
    _intent_setup_artifact_state,
)
from menagerie.crawler.environment_rebuild import (
    REBUILD_DEFERRAL_RELATIVE_PATH,
    load_rebuild_deferral_rows,
)
from menagerie.crawler.env_lifecycle import (
    DiskSpaceAuthority,
    EnvironmentProbeError,
    disk_free_bytes,
    parse_exact_lock,
    parse_probe_receipt_bytes,
)
from menagerie.crawler.envs import (
    DEFAULT_ENVIRONMENT_DISK_FLOOR_BYTES,
    load_environment_registry,
    overlay_package_rows,
    probe_declaration_digest,
)
from menagerie.crawler.identity import (
    durable_fsync,
    hash_bytes,
)
from menagerie.crawler.operator_environment import (
    PermanentEnvironmentOperatorError,
    _cas_path,
    _stream_sha256,
    _wheel_cas_file,
    resolve_environment_state_root,
)
from menagerie.crawler.tools.rotate_setup_artifacts import (
    ROTATION_PROVENANCE_SCHEMA_VERSION,
)

ROTATION_DID_NOT_ROTATE = "rotation-did-not-rotate"
STORE_MOUNT_UNAVAILABLE = "store-mount-unavailable"
CAS_INCOMPLETE = "cas-incomplete"
LOCKLESS_FAMILY = "lockless-family"
MIXED_FAMILY = "mixed-family"
STALE_SPEC = "stale-spec"
STALE_PROBE_RECEIPT = "stale-probe-receipt"
MATERIALIZATION_PENDING = "family-materialization-pending"
STATE_ROOT_UNBOUND = "state-root-unbound"
DISK_FLOOR_UNMET = "disk-floor-unmet"
UNSTABLE_PROBE_RECEIPT = "unstable-probe-receipt"
ROOTS_CONTRACT_UNMET = "roots-contract-unmet"
STATE_PLACEMENT_UNSAFE = "state-placement-unsafe"

#: The CLOSED armed-guard census (W4 registry rule, O-M1). Every preflight
#: assertion registers exactly one arm through ``register_preflight_arm``;
#: main() refuses to report a census that is not exactly this set, so a new
#: guard cannot exist outside the census and a deleted guard cannot vanish
#: silently. Extend this tuple in the same reviewed change that adds the arm.
EXPECTED_PREFLIGHT_ARMS = (
    "cas-completeness",
    "checker-calibration-mode",
    "disk-floor",
    "family-integrity",
    "family-readiness",
    "intake-partial-residues",
    "operator-state-root",
    "probe-receipt-stability",
    "roots-contract",
    "rotation-parks",
    "state-placement",
    "store-mount",
)


@dataclass(frozen=True)
class PreflightArm:
    """One evaluated preflight guard, as the census records it.

    Parameters
    ----------
    name:
        Census key, a member of :data:`EXPECTED_PREFLIGHT_ARMS`.
    armed:
        Whether the guard actually evaluated (an omitted precondition input
        leaves the guard UNARMED and disclosed, never silently absent).
    evidence:
        JSON-serializable facts the evaluation produced or the reason the
        guard is unarmed.
    refusal_shape:
        The typed first token(s) this arm refuses with.
    """

    name: str
    armed: bool
    evidence: Mapping[str, Any]
    refusal_shape: str

    def to_dict(self) -> dict[str, Any]:
        """Return the census row for the report and the verdict artifact."""

        return {
            "name": self.name,
            "armed": self.armed,
            "evidence": dict(self.evidence),
            "refusal_shape": self.refusal_shape,
        }


class PreflightArmRegistry:
    """The ONE registry every preflight assertion lands through (O-M1).

    Other lanes REGISTER arms; they never edit the evaluation body. The
    census this registry derives is embedded in the report and the minted
    verdict, giving the leg-open armed-guard census its machine-readable
    input.
    """

    def __init__(self) -> None:
        self._arms: dict[str, PreflightArm] = {}

    def register_preflight_arm(
        self,
        name: str,
        *,
        armed: bool,
        evidence: Mapping[str, Any],
        refusal_shape: str,
    ) -> PreflightArm:
        """Record one evaluated (or explicitly unarmed) guard.

        Parameters
        ----------
        name, armed, evidence, refusal_shape:
            Census facts; see :class:`PreflightArm`.

        Raises
        ------
        ValueError
            On a duplicate registration — two writers of one census row means
            the census no longer names its evaluator.
        """

        if name in self._arms:
            raise ValueError(f"preflight arm {name!r} is already registered")
        arm = PreflightArm(
            name=name, armed=bool(armed), evidence=dict(evidence), refusal_shape=refusal_shape
        )
        self._arms[name] = arm
        return arm

    def census(self) -> list[dict[str, Any]]:
        """Return the sorted armed-guard census rows."""

        return [self._arms[name].to_dict() for name in sorted(self._arms)]

    def assert_complete(self, expected: Sequence[str] = EXPECTED_PREFLIGHT_ARMS) -> None:
        """Refuse a census that is not exactly the closed expected arm set.

        Parameters
        ----------
        expected:
            The closed arm-name vocabulary this registry must cover exactly;
            defaults to the launch preflight's own census.
        """

        observed = tuple(sorted(self._arms))
        if observed != tuple(sorted(expected)):
            raise AssertionError(
                "preflight armed-guard census mismatch: observed "
                f"{observed} != expected {tuple(sorted(expected))}"
            )


def evaluate_roots_contract(
    roots: Mapping[str, Optional[Path]],
    *,
    minimum_free_bytes: int,
    allocation_probe_bytes: int,
) -> dict[str, Any]:
    """Prove every DECLARED root is a mounted, writable directory above floor.

    W4's roots contract. The declared path itself must exist as a directory —
    a missing component NEVER degrades to the nearest existing ancestor (the
    ``_nearest_existing`` walk-up refusal): an unmounted volume must read as
    unmounted, not as its healthy mount point. Writability is proven by a
    create+fsync+readback+unlink sentinel IN THE ACTING PROCESS CONTEXT, and
    the floor check ends with a real allocation probe because statvfs lies.

    Parameters
    ----------
    roots:
        Declared-root name to path; ``None`` values are undeclared (recorded,
        never asserted).
    minimum_free_bytes:
        Free-bytes floor every declared root's filesystem must clear; 0
        disables the measurement (fixtures only).
    allocation_probe_bytes:
        Size of the real allocation probe, run once per distinct filesystem;
        0 disables it (fixtures and post-lock revalidation).

    Returns
    -------
    dict[str, Any]
        ``{"ok": bool, "rows": [...], "refusals": [...]}``.
    """

    rows: list[dict[str, Any]] = []
    refusals: list[str] = []
    probed_devices: set[int] = set()
    for name in sorted(roots):
        root = roots[name]
        if root is None:
            rows.append({"root": name, "declared": False})
            continue
        row: dict[str, Any] = {"root": name, "declared": True, "path": str(root)}
        rows.append(row)
        if not root.is_dir():
            row["exists"] = False
            refusals.append(
                f"{ROOTS_CONTRACT_UNMET}: declared root {name} ({root}) does not "
                "exist as a directory; the preflight never walks up to the "
                "nearest existing ancestor — mount or create the exact root"
            )
            continue
        row["exists"] = True
        try:
            stat_result = os.stat(root)
            vfs = os.statvfs(root)
        except OSError as exc:
            refusals.append(
                f"{ROOTS_CONTRACT_UNMET}: declared root {name} ({root}) cannot be stat'ed: {exc}"
            )
            continue
        read_only = bool(vfs.f_flag & os.ST_RDONLY)
        free_bytes = vfs.f_bavail * vfs.f_frsize
        row["st_dev"] = stat_result.st_dev
        row["f_fsid"] = getattr(vfs, "f_fsid", None)
        row["read_only"] = read_only
        row["free_bytes"] = free_bytes
        if read_only:
            refusals.append(
                f"{ROOTS_CONTRACT_UNMET}: declared root {name} ({root}) is on a "
                "READ-ONLY filesystem"
            )
            continue
        if minimum_free_bytes and free_bytes < minimum_free_bytes:
            refusals.append(
                f"{ROOTS_CONTRACT_UNMET}: declared root {name} ({root}) is below "
                f"the free-bytes floor ({free_bytes} < {minimum_free_bytes})"
            )
            continue
        sentinel = root / f".preflight-sentinel-{os.getpid()}-{secrets.token_hex(8)}"
        payload = secrets.token_bytes(64)
        try:
            with sentinel.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                durable_fsync(handle.fileno())
            row["sentinel_ok"] = sentinel.read_bytes() == payload
        except OSError as exc:
            row["sentinel_ok"] = False
            refusals.append(
                f"{ROOTS_CONTRACT_UNMET}: declared root {name} ({root}) failed "
                f"the acting-process write sentinel: {exc}"
            )
            continue
        finally:
            try:
                sentinel.unlink(missing_ok=True)
            except OSError:
                pass
        if not row["sentinel_ok"]:
            refusals.append(
                f"{ROOTS_CONTRACT_UNMET}: declared root {name} ({root}) sentinel "
                "readback disagreed with the written bytes"
            )
            continue
        if allocation_probe_bytes <= 0:
            row["allocation_probe"] = "disabled"
        elif stat_result.st_dev in probed_devices:
            row["allocation_probe"] = "same-device-already-probed"
        else:
            probe = root / f".preflight-alloc-{os.getpid()}-{secrets.token_hex(8)}"
            try:
                with probe.open("xb") as handle:
                    handle.write(b"\0" * allocation_probe_bytes)
                    handle.flush()
                    durable_fsync(handle.fileno())
                row["allocation_probe"] = "probed"
                probed_devices.add(stat_result.st_dev)
            except OSError as exc:
                row["allocation_probe"] = "failed"
                refusals.append(
                    f"{ROOTS_CONTRACT_UNMET}: declared root {name} ({root}) failed "
                    f"the real {allocation_probe_bytes}-byte allocation probe "
                    f"(statvfs lied about usable space): {exc}"
                )
            finally:
                try:
                    probe.unlink(missing_ok=True)
                except OSError:
                    pass
    return {"ok": not refusals, "rows": rows, "refusals": refusals}


def _account_home() -> Path:
    """Return the kernel-derived account home (the launchd-proven anchor).

    Derived from the kernel user identity rather than ``HOME`` for the same
    reason :func:`menagerie.crawler.execution_lock.global_execution_flock_path`
    is: separate launchd environments and clones cannot accidentally select
    different anchors.
    """

    return Path(pwd.getpwuid(os.getuid()).pw_dir)


def _wake_infrastructure_roots(account_home: Path) -> dict[str, Path]:
    """Return the account-derived roots every wake/resume act also touches.

    These are not launch flags — they are derived the same way the acting
    code derives them, so the placement census names EVERY wake-touched state
    family, not just the declared ones.
    """

    return {
        "launchd-agent-root": account_home / "Library" / "LaunchAgents",
        "execution-flock-home": account_home / ".cache" / "menagerie" / "crawler",
    }


def evaluate_state_placement(
    roots: Mapping[str, Optional[Path]],
    *,
    account_home: Optional[Path] = None,
    platform_name: Optional[str] = None,
    device_of: Callable[[Path], int] = lambda path: os.stat(path).st_dev,
) -> dict[str, Any]:
    """Prove every wake-touched root lives on the launchd-proven anchor volume.

    PD-8's TCC placement contract. On macOS, a launchd-spawned process (every
    wake/resume-driven act: the usage-limit wake callback, the resumed driver,
    its author/checker/environment operators and workers) gets ``EPERM`` on
    removable volumes in BOTH directions unless the exact binary holds a TCC
    grant — while the SAME command works from a Terminal-context shell
    (rehearsal evidence ``r2-tcc-launchd-probe.json``, 2026-08-25). The W4
    roots-contract sentinel therefore proves NOTHING about wake-context
    writability when the preflight runs from a Terminal: it would pass at
    launch and the campaign would die hours later, unattended, at the first
    wake. This arm closes that gap with a placement predicate that IS provable
    at launch time from any context: every wake-touched root must resolve to
    the same filesystem device as the account home, the one volume a REAL
    launchd-spawned probe has proven writable on this host class.

    Declared roots must EXIST — placement of a missing path is unprovable and
    refuses (fail-closed; the roots contract refuses the same path for its own
    reason). Derived wake-infrastructure roots (the launchd agent root and the
    execution-flock home) may not exist yet on a fresh account; they are
    evaluated on their nearest existing ancestor, which must itself resolve to
    at-or-under the account home — creation then inherits the anchor device by
    ``mkdir`` semantics, and a symlink escaping the anchor volume still
    refuses.

    Parameters
    ----------
    roots:
        Declared-root name to path; ``None`` values are undeclared (recorded,
        never asserted).
    account_home:
        Anchor override for tests; production derives it from the kernel user
        identity.
    platform_name:
        Platform override for tests; production uses ``sys.platform``. The
        guarded seam (TCC denial of launchd-spawned processes on removable
        volumes) exists only on ``darwin``; elsewhere the evaluation records
        placement evidence without refusing, disclosed via ``enforced``.
    device_of:
        Filesystem-device measurement, injectable for deterministic
        cross-device tests.

    Returns
    -------
    dict[str, Any]
        ``{"ok": bool, "enforced": bool, "anchor": {...}, "rows": [...],
        "refusals": [...]}``.
    """

    platform_value = sys.platform if platform_name is None else platform_name
    enforced = platform_value == "darwin"
    home = _account_home() if account_home is None else account_home
    rows: list[dict[str, Any]] = []
    refusals: list[str] = []
    remedy = (
        "move the root onto the account-home volume (PD-8: internal "
        "relocation, removable volumes off the hot path), or prove a new "
        "placement with a REAL launchd-spawned probe "
        "(tools/tcc_launchd_probe.py) and land it as a reviewed change"
    )
    anchor: dict[str, Any] = {"path": str(home), "platform": platform_value}
    try:
        anchor_device: Optional[int] = device_of(home)
    except OSError as exc:
        anchor_device = None
        anchor["error"] = str(exc)
        if enforced:
            refusals.append(
                f"{STATE_PLACEMENT_UNSAFE}: the account home {home} cannot be "
                f"stat'ed ({exc}), so NO wake-touched root's placement is "
                "provable against the launchd-proven anchor volume"
            )
    anchor["st_dev"] = anchor_device

    def _evaluate(name: str, path: Path, *, derived: bool) -> None:
        row: dict[str, Any] = {
            "root": name,
            "declared": True,
            "derived": derived,
            "path": str(path),
        }
        rows.append(row)
        if anchor_device is None:
            row["placement"] = "unprovable"
            return
        probe = path
        if not path.exists():
            if not derived:
                row["exists"] = False
                row["placement"] = "unprovable"
                if enforced:
                    refusals.append(
                        f"{STATE_PLACEMENT_UNSAFE}: declared root {name} ({path}) "
                        "does not exist, so its volume placement cannot be "
                        "proven; an unmounted volume must never read as its "
                        f"healthy mount point — {remedy}"
                    )
                return
            ancestor = path.parent
            while not ancestor.exists() and ancestor != ancestor.parent:
                ancestor = ancestor.parent
            row["evaluated_at"] = str(ancestor)
            try:
                resolved = ancestor.resolve()
                resolved_home = home.resolve()
            except OSError as exc:
                row["placement"] = "unprovable"
                if enforced:
                    refusals.append(
                        f"{STATE_PLACEMENT_UNSAFE}: wake-infrastructure root "
                        f"{name} ({path}) has no resolvable existing ancestor "
                        f"({exc}) — {remedy}"
                    )
                return
            if resolved != resolved_home and resolved_home not in resolved.parents:
                row["placement"] = "escapes-anchor"
                if enforced:
                    refusals.append(
                        f"{STATE_PLACEMENT_UNSAFE}: wake-infrastructure root "
                        f"{name} ({path}) resolves outside the account home "
                        f"({resolved}); a wake act would create it off the "
                        f"launchd-proven anchor volume — {remedy}"
                    )
                return
            probe = ancestor
        row["exists"] = path.exists()
        try:
            root_device = device_of(probe)
        except OSError as exc:
            row["placement"] = "unprovable"
            if enforced:
                refusals.append(
                    f"{STATE_PLACEMENT_UNSAFE}: root {name} ({probe}) cannot be "
                    f"stat'ed ({exc}), so its placement is unprovable — {remedy}"
                )
            return
        row["st_dev"] = root_device
        if root_device == anchor_device:
            row["placement"] = "anchor-volume"
            return
        row["placement"] = "foreign-volume"
        if enforced:
            refusals.append(
                f"{STATE_PLACEMENT_UNSAFE}: wake-touched root {name} ({path}) "
                f"resolves to filesystem device {root_device}, not the "
                f"launchd-proven anchor device {anchor_device} ({home}): a "
                "launchd-spawned wake/resume act gets EPERM on removable "
                "volumes (TCC) even though this Terminal-context preflight "
                f"can write there — {remedy}"
            )

    for name in sorted(roots):
        declared = roots[name]
        if declared is None:
            rows.append({"root": name, "declared": False, "derived": False})
            continue
        _evaluate(name, declared, derived=False)
    for name, derived_path in sorted(_wake_infrastructure_roots(home).items()):
        _evaluate(name, derived_path, derived=True)
    return {
        "ok": not refusals,
        "enforced": enforced,
        "anchor": anchor,
        "rows": rows,
        "refusals": refusals,
    }


#: The one detail a PASSED probe receipt may carry (``operator_environment.
#: _probe_result``). Anything else is captured host chatter, which the lane's
#: post-probe re-attestation will rewrite -- moving a provenance-bound member.
CANONICAL_PASSED_PROBE_DETAIL = "ok"

#: The floor the production lane enforces (``build_command_environment_lane``
#: takes ``SequentialEnvironmentLifecycle``'s ``minimum_free_bytes`` default).
#: Kept as a named constant so the preflight and the lane cannot disagree about
#: what "enough disk to build" means.
PRODUCTION_ENVIRONMENT_DISK_FLOOR_BYTES = DEFAULT_ENVIRONMENT_DISK_FLOOR_BYTES

#: Repository-relative fallback the environment operator resolves when
#: ``MENAGERIE_ENVIRONMENT_STATE_ROOT`` is unset (mirrors
#: ``EnvironmentOperator._state_root_for`` and
#: ``CommandEnvironmentBackend.write_destinations``).
OPERATOR_STATE_ROOT_ENVIRONMENT_VARIABLE = "MENAGERIE_ENVIRONMENT_STATE_ROOT"


def evaluate_rebuild_parks(
    *,
    records_root: Path,
    envs_root: Path,
    target: str,
) -> dict[str, Any]:
    """Evaluate every current rebuild park against the launch-time registry.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory (the parent of ``models/`` and
        ``intake-extensions/``).
    envs_root:
        Environment registry root the launch would load.
    target:
        Exact conda-lock platform.

    Returns
    -------
    dict[str, Any]
        Typed report: per-row dispositions, per-intent identity pairs, and the
        release/held counts.
    """

    registry = load_environment_registry(envs_root, target=target)
    ledger = records_root / REBUILD_DEFERRAL_RELATIVE_PATH
    rows = load_rebuild_deferral_rows([ledger])
    driver = SimpleNamespace(
        registry=registry,
        paths=SimpleNamespace(
            ledgers=SimpleNamespace(models=records_root / "models" / "current-shard.jsonl")
        ),
    )
    held: list[dict[str, Any]] = []
    released: list[dict[str, Any]] = []
    intents: dict[str, dict[str, Any]] = {}
    for row in rows:
        quarantine = row.get("quarantine") or {}
        intent_name = str(quarantine.get("intent"))
        item = SimpleNamespace(route=SimpleNamespace(intent=intent_name))
        still_holds = _deferral_row_fact_still_holds(
            driver,
            item,  # type: ignore[arg-type]  # duck-typed exactly like the claimant tests
            dict(row),
        )
        entry = {
            "stable_id": row.get("stable_id"),
            "work_id": row.get("work_id"),
            "intent": intent_name,
            "quarantined_identity": quarantine.get("artifact_identity"),
        }
        (held if still_holds else released).append(entry)
        intent = registry.intents.get(intent_name)
        if intent_name not in intents:
            intents[intent_name] = {
                "intent": intent_name,
                "current_identity": (
                    _intent_setup_artifact_identity(intent) if intent is not None else None
                ),
                "quarantined_identities": [],
            }
        recorded = quarantine.get("artifact_identity")
        if recorded not in intents[intent_name]["quarantined_identities"]:
            intents[intent_name]["quarantined_identities"].append(recorded)
    return {
        "records_root": str(records_root),
        "envs_root": str(envs_root),
        "target": target,
        "rebuild_rows": len(rows),
        "released": len(released),
        "still_held": len(held),
        "intents": sorted(intents.values(), key=lambda value: str(value["intent"])),
        "held_rows": held,
        "released_rows": released,
    }


def evaluate_store_mount(env_root: Path) -> dict[str, Any]:
    """Prove the environment store root resolves to a mounted real directory.

    T3 cold-start precondition 6: the pilot's ``env_root`` is a symlink
    convention; when the target volume is unmounted the lane dies deep inside
    ``mkdir(exist_ok=True)`` with an untyped ``FileExistsError`` (reproduced
    independently by two audit labs) and the failure fans out over models.

    Parameters
    ----------
    env_root:
        The environment store root a launch would build prefixes under.

    Returns
    -------
    dict[str, Any]
        ``{"ok": bool, ...}`` typed report; failures carry
        ``store-mount-unavailable`` and the reason.
    """

    report: dict[str, Any] = {"env_root": str(env_root), "ok": False}
    try:
        resolved = env_root.resolve(strict=True)
    except OSError as exc:
        report["error"] = (
            f"{STORE_MOUNT_UNAVAILABLE}: env_root does not resolve to a real "
            f"path ({exc}); a dangling store symlink means the backing volume "
            "is not mounted"
        )
        return report
    if not resolved.is_dir():
        report["error"] = (
            f"{STORE_MOUNT_UNAVAILABLE}: env_root resolves to {resolved}, which is not a directory"
        )
        return report
    try:
        os.statvfs(resolved)
    except OSError as exc:
        report["error"] = (
            f"{STORE_MOUNT_UNAVAILABLE}: env_root filesystem is not measurable ({exc})"
        )
        return report
    report["ok"] = True
    report["resolved"] = str(resolved)
    return report


def evaluate_operator_state_root(*, state_root: Path, launch_cwd: Path) -> dict[str, Any]:
    """Bind the verified CAS root to the root the operator will actually open.

    The CAS receipt below is only evidence for the state root the launch's
    environment operator resolves, and it resolves that root itself: the
    absolute ``MENAGERIE_ENVIRONMENT_STATE_ROOT`` if exported, else
    ``<cwd>/.crawl-local/environment-operator`` (``EnvironmentOperator.
    _state_root_for``; ``CommandEnvironmentBackend.write_destinations`` mirrors
    the same rule for the disk floor). The driver spawns the operator without
    an explicit ``--state-root`` and without an ``env=``, so the variable in the
    launching shell is the whole binding. A launch that verifies a seeded
    external CAS and then forgets to export the variable silently builds
    offline against the repository-local root instead -- a green receipt for a
    CAS nothing will read.

    Parameters
    ----------
    state_root:
        The state root whose CAS completeness this preflight asserts.
    launch_cwd:
        Working directory the launch will run the driver (and therefore the
        operator) from.

    Returns
    -------
    dict[str, Any]
        ``{"ok": bool, ...}`` typed report; failures carry
        ``state-root-unbound`` and both roots.
    """

    configured = os.environ.get(OPERATOR_STATE_ROOT_ENVIRONMENT_VARIABLE)
    report: dict[str, Any] = {
        "asserted": str(state_root),
        "launch_cwd": str(launch_cwd),
        "environment_variable": OPERATOR_STATE_ROOT_ENVIRONMENT_VARIABLE,
        "environment_variable_set": bool(configured),
        "ok": False,
    }
    try:
        resolved = resolve_environment_state_root(launch_cwd)
    except PermanentEnvironmentOperatorError as exc:
        report["resolved"] = None
        report["error"] = f"{STATE_ROOT_UNBOUND}: {exc}"
        return report
    report["resolved"] = str(resolved)
    if os.path.normpath(resolved) != os.path.normpath(state_root):
        exported = (
            f"{OPERATOR_STATE_ROOT_ENVIRONMENT_VARIABLE}={configured!r}"
            if configured
            else f"{OPERATOR_STATE_ROOT_ENVIRONMENT_VARIABLE} is NOT exported, so the "
            f"operator falls back to {resolved}"
        )
        report["error"] = (
            f"{STATE_ROOT_UNBOUND}: the CAS completeness assert covers "
            f"{state_root}, but the launch's environment operator will open "
            f"{resolved} ({exported}). The offline build reads the CAS below the "
            "root the operator resolves, not the one this preflight verified: "
            f"export {OPERATOR_STATE_ROOT_ENVIRONMENT_VARIABLE}={state_root} for the "
            "launch, or pass --state-root for the root the launch will really use."
        )
        return report
    report["ok"] = True
    return report


def evaluate_disk_floor(
    *,
    env_root: Path,
    state_root: Path | None,
    tmp_root: Path | None = None,
    minimum_free_bytes: int = PRODUCTION_ENVIRONMENT_DISK_FLOOR_BYTES,
    disk_free: Any = disk_free_bytes,
) -> dict[str, Any]:
    """Measure every lane write destination against the production disk floor.

    ``evaluate_store_mount`` proves the store is mounted and its filesystem is
    measurable, but never compares the measurement to the floor the lane
    enforces. On a host that is ALREADY below the floor, every intent parks
    behind the disk-floor valve one build at a time (typed and recoverable, but
    a whole leg's worth of admissions spent to learn a fact ``statvfs`` knows
    before launch). The destinations and the measurement here are the lane's own
    (``SequentialEnvironmentLifecycle._disk_space_authority`` /
    ``_assert_disk_floor``): the env root, the operator state root, and the
    temporary directory, each measured at its nearest existing ancestor and
    deduplicated by the shared :class:`DiskSpaceAuthority`. The lowest-free
    actual destination is the sole authoritative floor value.

    Parameters
    ----------
    env_root:
        Environment store root the launch will build prefixes under.
    state_root:
        Operator state root (CAS, package caches, wheelhouses), or ``None`` when
        the launch did not name one.
    tmp_root:
        Temporary staging root the launch will export. ``None`` uses this
        preflight process's active temporary directory.
    minimum_free_bytes:
        Floor to enforce; defaults to the production lane's.
    disk_free:
        Free-bytes measurement, injected only by tests.

    Returns
    -------
    dict[str, Any]
        Per-destination measurements plus ``{"ok": bool}``; failures carry
        ``disk-floor-unmet`` and name the starved destination and its role.
    """

    destinations: list[tuple[str, Path]] = [("env-root", env_root)]
    if state_root is not None:
        destinations.append(("backend-state", state_root))
    if tmp_root is None:
        import tempfile

        tmp_root = Path(tempfile.gettempdir())
    destinations.append(("tmp", tmp_root))
    report: dict[str, Any] = {
        "minimum_free_bytes": minimum_free_bytes,
        "destinations": [],
        "ok": False,
    }
    if minimum_free_bytes <= 0:
        report["ok"] = True
        return report
    snapshot = DiskSpaceAuthority(destinations, disk_free=disk_free).snapshot()
    starved: list[dict[str, Any]] = []
    for observation in snapshot.destinations:
        row: dict[str, Any] = {
            "role": observation.role,
            "destination": str(observation.destination),
            "probe": str(observation.probe),
            "free_bytes": observation.free_bytes,
            "ok": (
                observation.error is None
                and observation.free_bytes is not None
                and observation.free_bytes >= minimum_free_bytes
            ),
        }
        if observation.error is not None:
            row["error"] = observation.error
        report["destinations"].append(row)
        if not row["ok"]:
            starved.append(row)
    if snapshot.authority is not None:
        report["authority"] = {
            "role": snapshot.authority.role,
            "destination": str(snapshot.authority.destination),
            "probe": str(snapshot.authority.probe),
            "free_bytes": snapshot.authority.free_bytes,
        }
    if starved:
        listed = "; ".join(
            f"{row['role']} at {row['probe']} "
            f"({row['free_bytes'] if row['free_bytes'] is not None else row.get('error')})"
            for row in starved
        )
        report["error"] = (
            f"{DISK_FLOOR_UNMET}: the environment lane refuses any build while a "
            f"write destination holds less than {minimum_free_bytes} free bytes, and "
            f"the following already do: {listed}. Every routed intent would park "
            "behind the disk-floor valve; free space (tools/reap_disk.py) or move "
            "the starved destination before launching."
        )
        return report
    report["ok"] = True
    return report


def evaluate_cas_completeness(*, envs_root: Path, state_root: Path, target: str) -> dict[str, Any]:
    """Verify every committed-family artifact is present in the campaign CAS.

    T3-T5: the build-from-committed lane is strictly offline at create time,
    so a launch is only cold-start-safe when every conda artifact named by
    every committed lock -- and every pinned overlay wheel -- already sits in
    the campaign CAS with its exact digest. This receipt is the preflight's
    proof; a miss is a typed refusal, not a mid-leg network surprise.

    Parameters
    ----------
    envs_root, state_root, target:
        Registry root, campaign environment-operator state root, and platform.

    Returns
    -------
    dict[str, Any]
        Receipt with verified counts and the exact missing/corrupt entries.
    """

    registry = load_environment_registry(envs_root, target=target)
    missing: list[dict[str, str]] = []
    verified_artifacts = 0
    verified_wheels = 0
    for intent in registry.intents.values():
        if intent.lock.status != "locked":
            continue
        for receipt in parse_exact_lock(intent.lock.lock_bytes or b""):
            path = _cas_path(state_root, receipt.sha256)
            if not path.is_file() or _stream_sha256(path) != receipt.sha256:
                missing.append(
                    {"intent": intent.name, "url": receipt.url, "sha256": receipt.sha256}
                )
            else:
                verified_artifacts += 1
        for row in overlay_package_rows(intent):
            digest = str(row["sha256"])
            if _wheel_cas_file(state_root / "wheel-cas", digest) is None:
                missing.append({"intent": intent.name, "wheel": str(row["name"]), "sha256": digest})
            else:
                verified_wheels += 1
    return {
        "state_root": str(state_root),
        "target": target,
        "ok": not missing,
        "verified_artifacts": verified_artifacts,
        "verified_wheels": verified_wheels,
        "missing": missing,
    }


def evaluate_family_readiness(
    *, envs_root: Path, target: str, allow_lockless: Sequence[str]
) -> dict[str, Any]:
    """Require a proven committed family for every non-exempt registry intent.

    The env lane builds FROM the committed family and refuses to solve in
    place, so an intent without a ``locked`` family (unlocked, incomplete,
    hash-mismatch, superseded, or mid-publication ``publishing``) can only
    park or fail every model routed to it. Intents deliberately kept lockless
    (routed away from model work per the route census) must be named
    explicitly so the exemption is visible in the launch log.

    Parameters
    ----------
    envs_root, target:
        Registry root and platform the launch will load.
    allow_lockless:
        Intent names exempted from the locked-family requirement.

    Returns
    -------
    dict[str, Any]
        Per-intent statuses and the refused set.
    """

    registry = load_environment_registry(envs_root, target=target)
    allowed = set(allow_lockless)
    unknown = sorted(allowed - set(registry.intents))
    statuses: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    for intent in registry.intents.values():
        status = intent.lock.status
        complete_family = all(
            member.get("state") == "present"
            for member in _intent_setup_artifact_state(intent).values()
        )
        row = {
            "intent": intent.name,
            "status": status,
            "family_complete": complete_family,
            "generation": intent.generation,
            "allowed_lockless": intent.name in allowed,
        }
        statuses.append(row)
        if intent.name in allowed:
            continue
        if status != "locked" or not complete_family or intent.generation is None:
            refused.append(row)
    return {
        "target": target,
        "ok": not refused and not unknown,
        "unknown_allowances": unknown,
        "intents": statuses,
        "refused": refused,
    }


def evaluate_family_integrity(
    *, envs_root: Path, target: str, allow_lockless: Sequence[str] = ()
) -> dict[str, Any]:
    """Validate every locked family against its committed provenance binding.

    T3-T7: the park-release claimant reacts to ANY member-byte change, so a
    mixed family (a torn pre-manifest publication, a hand-edited member, a
    lost family manifest over drifted flat members) carries an accidental new
    identity that would release parks against an environment no clean create
    ever proved. This is the independent detector: a family only counts as a
    rotation product when its committed provenance binds every member by
    exact digest, its spec is the one the solve consumed (``spec_sha256`` --
    the rotation-lock-provenance.v1 lesson: the field existed but was compared
    by NOTHING at runtime), and its probe receipt still satisfies the intent's
    OWN declared probe contract (per-intent, never the structurally
    always-stale whole-file ``probes.yml`` hash).

    Parameters
    ----------
    envs_root, target:
        Registry root and platform the launch will load.
    allow_lockless:
        Intent names exempted from the committed-family requirement.

    Returns
    -------
    dict[str, Any]
        Per-intent rows and typed refusals (``reason`` in ``mixed-family``,
        ``stale-spec``, ``stale-probe-receipt``,
        ``family-materialization-pending``).
    """

    registry = load_environment_registry(envs_root, target=target)
    allowed = set(allow_lockless)
    rows: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []

    def refuse(intent_name: str, reason: str, detail: str) -> None:
        refused.append({"intent": intent_name, "reason": reason, "detail": detail})

    for intent in registry.intents.values():
        if intent.name in allowed:
            rows.append({"intent": intent.name, "checked": False, "allowed_lockless": True})
            continue
        lock = intent.lock
        if lock.family_fault is not None:
            refuse(
                intent.name,
                MIXED_FAMILY,
                f"family manifest fails validation: {lock.family_fault}",
            )
            continue
        if lock.status != "locked":
            # Readiness owns the non-locked refusals; nothing to bind here.
            rows.append({"intent": intent.name, "checked": False, "status": lock.status})
            continue
        rows.append({"intent": intent.name, "checked": True, "status": lock.status})
        if lock.materialization_pending:
            refuse(
                intent.name,
                MATERIALIZATION_PENDING,
                "the committed publication's flat materialization is incomplete; "
                "run tools/rotate_setup_artifacts.py --complete-publication "
                f"--intent {intent.name} before launch",
            )
            continue
        member_dir = lock.lock_path.parent
        provenance_path = member_dir / f"{target}.provenance.json"
        receipt_path = member_dir / f"{target}.probes.json"
        if not provenance_path.is_file():
            refuse(
                intent.name,
                MIXED_FAMILY,
                f"no committed rotation provenance at {provenance_path}; an "
                "unattested family proves nothing about its members",
            )
            continue
        try:
            provenance = json.loads(provenance_path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            refuse(intent.name, MIXED_FAMILY, f"provenance is unreadable: {exc}")
            continue
        if (
            not isinstance(provenance, dict)
            or provenance.get("schema_version") != ROTATION_PROVENANCE_SCHEMA_VERSION
        ):
            refuse(
                intent.name,
                MIXED_FAMILY,
                "provenance is not a rotation-lock-provenance record",
            )
            continue
        if provenance.get("intent") != intent.name or provenance.get("target") != target:
            refuse(intent.name, MIXED_FAMILY, "provenance names a foreign intent/target")
            continue
        try:
            receipt_bytes = receipt_path.read_bytes()
        except OSError as exc:
            refuse(intent.name, MIXED_FAMILY, f"probe receipt is unreadable: {exc}")
            continue
        bindings = (
            ("lock_sha256", hash_bytes(lock.lock_bytes or b""), "committed lock"),
            (
                "resolved_export_sha256",
                hash_bytes(lock.export_bytes or b""),
                "resolved export",
            ),
            ("probe_receipt_sha256", hash_bytes(receipt_bytes), "probe receipt"),
        )
        mismatched = [
            (key, label) for key, observed, label in bindings if provenance.get(key) != observed
        ]
        overlay = provenance.get("pip_overlay")
        if intent.pip_overlay is not None:
            declared_manifest = hash_bytes(intent.pip_overlay.manifest_bytes or b"")
            if not isinstance(overlay, dict) or overlay.get("manifest_sha256") != declared_manifest:
                mismatched.append(("pip_overlay.manifest_sha256", "pip-overlay manifest"))
        elif overlay is not None:
            mismatched.append(("pip_overlay", "undeclared pip overlay"))
        if mismatched:
            listed = ", ".join(label for _key, label in mismatched)
            refuse(
                intent.name,
                MIXED_FAMILY,
                f"provenance does not bind the on-disk member(s): {listed}; a "
                "member-byte change without its rotation is a mixed family, "
                "not a rotation",
            )
            continue
        spec_path = envs_root / intent.name / "environment.yml"
        if provenance.get("spec_sha256") != hash_bytes(spec_path.read_bytes()):
            refuse(
                intent.name,
                STALE_SPEC,
                f"{spec_path.name} changed after the last rotation; an "
                "environment.yml edit is not a rotation -- rotate the intent "
                "or revert the edit",
            )
            continue
        try:
            parse_probe_receipt_bytes(intent.probes, receipt_bytes)
        except EnvironmentProbeError as exc:
            refuse(
                intent.name,
                STALE_PROBE_RECEIPT,
                f"the committed probe receipt no longer satisfies the intent's "
                f"declared probe contract ({exc}); rotate the intent",
            )
            continue
        declared_probe_digest = provenance.get("probe_declaration_sha256")
        if declared_probe_digest is not None and declared_probe_digest != (
            probe_declaration_digest(intent.name, intent.probes)
        ):
            refuse(
                intent.name,
                STALE_PROBE_RECEIPT,
                "the intent's probe declaration changed after the last rotation; rotate the intent",
            )
            continue
    return {
        "target": target,
        "ok": not refused,
        "intents": rows,
        "refused": refused,
    }


def evaluate_probe_receipt_stability(
    *,
    envs_root: Path,
    target: str,
    allow_unstable: Sequence[str] = (),
    allow_lockless: Sequence[str] = (),
) -> dict[str, Any]:
    """Refuse a committed receipt the next launch is guaranteed to re-attest.

    The env lane writes ``<target>.probes.json`` once, strictly after every
    declared probe passed against the created prefix (``env_lifecycle.run``), and
    that member is bound by the committed rotation provenance. So whenever a
    committed receipt's PASSED rows carry text the launch host will not
    reproduce, the launch necessarily rewrites a provenance-bound member and the
    NEXT preflight refuses with ``mixed-family`` -- a whole leg spent to learn a
    fact that is statically visible in the committed artifact right now. The
    shipped audio/speech/detect-misc/paddle receipts embed the rotation host's
    absolute prefix paths, which no launch prefix can ever equal.

    The predicate is closed and provable, never a heuristic: a passed row's
    detail must be exactly :data:`CANONICAL_PASSED_PROBE_DETAIL`, the one value
    the operator records for a probe that produced no output. Failed rows cannot
    appear here at all (``validate_probe_receipts`` refuses them), so no evidence
    is being ignored -- a passed row's detail is diagnostic-only by construction.

    Parameters
    ----------
    envs_root, target:
        Registry root and platform the launch will load.
    allow_unstable:
        Intent names whose re-attestation the launch deliberately accepts; the
        allowance is named in the launch log rather than inferred.
    allow_lockless:
        Intent names exempted from the committed-family requirement.

    Returns
    -------
    dict[str, Any]
        Per-intent census with the exact unstable rows, plus ``{"ok": bool}``.
    """

    registry = load_environment_registry(envs_root, target=target)
    lockless = set(allow_lockless)
    allowed = set(allow_unstable)
    rows: list[dict[str, Any]] = []
    unstable: list[dict[str, Any]] = []
    for intent in registry.intents.values():
        if intent.name in lockless or intent.lock.status != "locked":
            rows.append({"intent": intent.name, "checked": False, "status": intent.lock.status})
            continue
        receipt_path = intent.lock.lock_path.with_name(f"{target}.probes.json")
        try:
            receipts = parse_probe_receipt_bytes(intent.probes, receipt_path.read_bytes())
        except (OSError, EnvironmentProbeError) as exc:
            # Integrity owns the unreadable/contract-violating receipt refusals;
            # an unprovable receipt is never silently counted as stable.
            rows.append({"intent": intent.name, "checked": False, "unreadable_receipt": str(exc)})
            continue
        offending = [
            {"name": receipt.name, "detail": receipt.detail}
            for receipt in receipts
            if receipt.passed and receipt.detail != CANONICAL_PASSED_PROBE_DETAIL
        ]
        rows.append(
            {
                "intent": intent.name,
                "checked": True,
                "probes": len(receipts),
                "unstable_rows": len(offending),
                "allowed_unstable": intent.name in allowed,
            }
        )
        if offending and intent.name not in allowed:
            unstable.append({"intent": intent.name, "rows": offending})
    return {
        "target": target,
        "ok": not unstable,
        "allowed_unstable": sorted(allowed),
        "intents": rows,
        "unstable": unstable,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the launch preflight; refuse on any failed cold-start assert."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records-root",
        type=Path,
        required=True,
        help="campaign canonical records directory (contains models/, intake-extensions/)",
    )
    parser.add_argument(
        "--envs-root",
        type=Path,
        required=True,
        help="environment registry root the launch will load",
    )
    parser.add_argument("--target", default="osx-arm64")
    parser.add_argument(
        "--env-root",
        type=Path,
        help="environment store root the launch will build prefixes under",
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        help="campaign environment-operator state root (.crawl-local/environment-operator)",
    )
    parser.add_argument(
        "--tmp-root",
        type=Path,
        help=(
            "temporary staging root the launch will export; defaults to this "
            "preflight process's active temporary directory"
        ),
    )
    parser.add_argument(
        "--allow-lockless",
        action="append",
        default=[],
        help="intent explicitly exempt from the locked-family requirement; repeatable",
    )
    parser.add_argument(
        "--allow-unstable-probe-receipt",
        action="append",
        default=[],
        help=(
            "intent whose committed probe receipt the launch deliberately accepts "
            "re-attesting (de-binding its rotation provenance); repeatable"
        ),
    )
    parser.add_argument(
        "--disk-floor-bytes",
        type=int,
        default=PRODUCTION_ENVIRONMENT_DISK_FLOOR_BYTES,
        help=(
            "free-bytes floor every lane write destination must clear; defaults to "
            "the production lane's. 0 disables the measurement, exactly as a zero "
            "minimum_free_bytes disables the lane's own floor (fixtures only)"
        ),
    )
    parser.add_argument(
        "--launch-cwd",
        type=Path,
        default=None,
        help=(
            "working directory the launch will run the driver from; the environment "
            "operator resolves its state root relative to it when "
            f"{OPERATOR_STATE_ROOT_ENVIRONMENT_VARIABLE} is unset (default: this process's cwd)"
        ),
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="print the evaluation without refusing (for launches promising no rotation)",
    )
    parser.add_argument(
        "--renders-no-gate-verdicts",
        action="store_true",
        help=(
            "declare that this launch's legs render NO gate verdicts (the closing "
            "licensing leg, final-sweep step 3): the W2 receipt-XOR calibration arm "
            "then passes vacuously with the declaration disclosed (F-O19). The "
            "default assumes verdict-rendering, fail-closed"
        ),
    )
    parser.add_argument(
        "--allocation-probe-bytes",
        type=int,
        default=64 * 1024**2,
        help=(
            "size of the real allocation probe the roots contract ends with "
            "(statvfs lies); run once per distinct filesystem. 0 disables it "
            "(fixtures only)"
        ),
    )
    parser.add_argument(
        "--placement-anchor",
        type=Path,
        default=None,
        help=(
            "anchor path whose filesystem device every wake-touched root must "
            "share (fixture parameterization, host-independent); production "
            "omits it and the anchor is the kernel-derived account home, the "
            "one volume the launchd-spawned TCC probe has proven writable"
        ),
    )
    args = parser.parse_args(argv)
    refusals: list[str] = []
    registry = PreflightArmRegistry()

    parks = evaluate_rebuild_parks(
        records_root=args.records_root,
        envs_root=args.envs_root,
        target=args.target,
    )
    registry.register_preflight_arm(
        "rotation-parks",
        armed=True,
        evidence={"rebuild_rows": parks["rebuild_rows"], "still_held": parks["still_held"]},
        refusal_shape=ROTATION_DID_NOT_ROTATE,
    )
    if parks["still_held"]:
        held_intents = sorted({str(row["intent"]) for row in parks["held_rows"]})
        refusals.append(
            f"{ROTATION_DID_NOT_ROTATE}: {parks['still_held']} of "
            f"{parks['rebuild_rows']} environment-rebuild park rows would be "
            "re-claimed at launch because the committed setup-artifact identity "
            f"for intent(s) {', '.join(held_intents)} still equals the quarantined "
            "identity those rows name. An environment.yml edit is not a rotation: "
            "run tools/rotate_setup_artifacts.py (or record an explicit "
            "environment-quarantine-cleared event) and re-run this preflight."
        )

    families = evaluate_family_readiness(
        envs_root=args.envs_root,
        target=args.target,
        allow_lockless=tuple(args.allow_lockless),
    )
    registry.register_preflight_arm(
        "family-readiness",
        armed=True,
        evidence={
            "refused": len(families["refused"]),
            "unknown_allowances": list(families["unknown_allowances"]),
        },
        refusal_shape=LOCKLESS_FAMILY,
    )
    if not families["ok"]:
        details = ", ".join(f"{row['intent']}({row['status']})" for row in families["refused"])
        unknown = families["unknown_allowances"]
        if unknown:
            details = f"{details + '; ' if details else ''}unknown allowances: {', '.join(unknown)}"
        refusals.append(
            f"{LOCKLESS_FAMILY}: the env lane builds FROM committed families and "
            f"never solves in place; intent(s) without a proven locked family: "
            f"{details}. Rotate them (tools/rotate_setup_artifacts.py) or exempt "
            "deliberately routed-away intents with --allow-lockless."
        )

    integrity = evaluate_family_integrity(
        envs_root=args.envs_root,
        target=args.target,
        allow_lockless=tuple(args.allow_lockless),
    )
    registry.register_preflight_arm(
        "family-integrity",
        armed=True,
        evidence={"refused": len(integrity["refused"])},
        refusal_shape=",".join(
            (MIXED_FAMILY, STALE_SPEC, STALE_PROBE_RECEIPT, MATERIALIZATION_PENDING)
        ),
    )
    if not integrity["ok"]:
        for row in integrity["refused"]:
            refusals.append(f"{row['reason']}: intent {row['intent']}: {row['detail']}")

    receipts = evaluate_probe_receipt_stability(
        envs_root=args.envs_root,
        target=args.target,
        allow_unstable=tuple(args.allow_unstable_probe_receipt),
        allow_lockless=tuple(args.allow_lockless),
    )
    registry.register_preflight_arm(
        "probe-receipt-stability",
        armed=True,
        evidence={"unstable": len(receipts["unstable"])},
        refusal_shape=UNSTABLE_PROBE_RECEIPT,
    )
    if not receipts["ok"]:
        listed = ", ".join(
            f"{row['intent']}({len(row['rows'])} row(s))" for row in receipts["unstable"]
        )
        refusals.append(
            f"{UNSTABLE_PROBE_RECEIPT}: the env lane re-attests "
            f"<target>.probes.json after every build, and these committed "
            f"receipts carry host-dependent detail on PASSED probes, so the "
            f"re-attestation will rewrite a provenance-bound member and the next "
            f"preflight will refuse mixed-family: {listed}. Rotate them "
            "(tools/rotate_setup_artifacts.py) or accept the de-binding "
            "explicitly with --allow-unstable-probe-receipt."
        )

    report: dict[str, Any] = {
        "rebuild_parks": parks,
        "family_readiness": families,
        "family_integrity": integrity,
        "probe_receipt_stability": receipts,
    }

    # W2 (WS1): the receipt-XOR calibration-mode arm, registered per the W4
    # registry rule -- the EVALUATOR lives in checker_calibration.py (the
    # owning lane's module, called through it so the census-exhaustiveness
    # AST test keeps binding this file's OWN evaluators); this file only
    # registers the arm.
    from menagerie.crawler import checker_calibration
    from menagerie.crawler.checker_calibration import CHECKER_CALIBRATION_MODE_UNPROVEN
    from menagerie.crawler.constants import GateKind
    from menagerie.crawler.operator_checker import required_checker_model

    calibration_mode = checker_calibration.evaluate_checker_calibration_mode(
        renders_gate_verdicts=not args.renders_no_gate_verdicts,
        pinned_models={kind.value: required_checker_model(kind.value) for kind in GateKind},
    )
    report["checker_calibration_mode"] = calibration_mode
    registry.register_preflight_arm(
        "checker-calibration-mode",
        armed=True,
        evidence={
            "renders_gate_verdicts": calibration_mode["renders_gate_verdicts"],
            "vacuous": calibration_mode["vacuous"],
            "calibration_run_active": calibration_mode["calibration_run_active"],
            "shadow_rehearsal_active": calibration_mode["shadow_rehearsal_active"],
            "lanes": calibration_mode["lanes"],
        },
        refusal_shape=CHECKER_CALIBRATION_MODE_UNPROVEN,
    )
    refusals.extend(calibration_mode["refusals"])

    from menagerie.crawler.checkpoint import partial_intake_snapshot_residues

    residues = partial_intake_snapshot_residues(args.records_root / "intake")
    report["intake_partial_residues"] = [str(residue) for residue in residues]
    registry.register_preflight_arm(
        "intake-partial-residues",
        armed=True,
        evidence={"residues": len(residues)},
        refusal_shape="disclosure-only",
    )
    roots_contract = evaluate_roots_contract(
        {
            "records-root": args.records_root,
            "env-root": args.env_root,
            "state-root": args.state_root,
            "tmp-root": args.tmp_root,
        },
        minimum_free_bytes=int(args.disk_floor_bytes),
        allocation_probe_bytes=int(args.allocation_probe_bytes),
    )
    report["roots_contract"] = roots_contract
    registry.register_preflight_arm(
        "roots-contract",
        armed=True,
        evidence={
            "declared": [row["root"] for row in roots_contract["rows"] if row.get("declared")],
            "refused": len(roots_contract["refusals"]),
        },
        refusal_shape=ROOTS_CONTRACT_UNMET,
    )
    refusals.extend(roots_contract["refusals"])
    # PD-8: the TCC placement contract over every wake-touched root. The
    # mirror roots ride the ambient environment (they are not launch flags and
    # not members of the frozen launch-environment vocabulary); unset mirrors
    # are recorded undeclared, never asserted.
    placement = evaluate_state_placement(
        {
            "records-root": args.records_root,
            "envs-root": args.envs_root,
            "env-root": args.env_root,
            "state-root": args.state_root,
            "tmp-root": args.tmp_root,
            "public-mirror": (
                Path(os.environ["MENAGERIE_PUBLIC_MIRROR"])
                if os.environ.get("MENAGERIE_PUBLIC_MIRROR")
                else None
            ),
            "private-mirror": (
                Path(os.environ["MENAGERIE_PRIVATE_MIRROR"])
                if os.environ.get("MENAGERIE_PRIVATE_MIRROR")
                else None
            ),
        },
        account_home=args.placement_anchor,
    )
    report["state_placement"] = placement
    registry.register_preflight_arm(
        "state-placement",
        armed=placement["enforced"],
        evidence={
            "enforced": placement["enforced"],
            "anchor": placement["anchor"],
            "declared": [row["root"] for row in placement["rows"] if row.get("declared")],
            "refused": len(placement["refusals"]),
        },
        refusal_shape=STATE_PLACEMENT_UNSAFE,
    )
    refusals.extend(placement["refusals"])
    if residues:
        # Disclosure, not refusal: a manifest-less partial dir never claims to
        # be a snapshot (H1 F2), so nothing bricks on it -- but the operator
        # should quarantine it or re-run the register to complete the copy.
        print(
            f"launch-preflight warning: {len(residues)} manifest-less partial intake "
            "snapshot dir(s) present (crashed register copy); quarantine or re-run "
            "register_existing_snapshot: " + ", ".join(str(residue) for residue in residues),
            file=sys.stderr,
        )
    if args.env_root is not None:
        mount = evaluate_store_mount(args.env_root)
        report["store_mount"] = mount
        registry.register_preflight_arm(
            "store-mount",
            armed=True,
            evidence={"env_root": str(args.env_root)},
            refusal_shape=STORE_MOUNT_UNAVAILABLE,
        )
        if not mount["ok"]:
            refusals.append(str(mount.get("error")))
    else:
        registry.register_preflight_arm(
            "store-mount",
            armed=False,
            evidence={"reason": "--env-root not given"},
            refusal_shape=STORE_MOUNT_UNAVAILABLE,
        )
        print(
            "launch-preflight warning: --env-root not given; the store-mount "
            "precondition is NOT asserted",
            file=sys.stderr,
        )
    if args.env_root is not None:
        floor = evaluate_disk_floor(
            env_root=args.env_root,
            state_root=args.state_root,
            tmp_root=args.tmp_root,
            minimum_free_bytes=int(args.disk_floor_bytes),
        )
        report["disk_floor"] = floor
        registry.register_preflight_arm(
            "disk-floor",
            armed=True,
            evidence={"minimum_free_bytes": int(args.disk_floor_bytes)},
            refusal_shape=DISK_FLOOR_UNMET,
        )
        if not floor["ok"]:
            refusals.append(str(floor.get("error")))
    else:
        registry.register_preflight_arm(
            "disk-floor",
            armed=False,
            evidence={"reason": "--env-root not given"},
            refusal_shape=DISK_FLOOR_UNMET,
        )
    if args.state_root is not None:
        binding = evaluate_operator_state_root(
            state_root=args.state_root,
            launch_cwd=(args.launch_cwd or Path.cwd()).resolve(),
        )
        report["operator_state_root"] = binding
        registry.register_preflight_arm(
            "operator-state-root",
            armed=True,
            evidence={"state_root": str(args.state_root)},
            refusal_shape=STATE_ROOT_UNBOUND,
        )
        if not binding["ok"]:
            refusals.append(str(binding.get("error")))
        cas = evaluate_cas_completeness(
            envs_root=args.envs_root, state_root=args.state_root, target=args.target
        )
        report["cas_completeness"] = cas
        registry.register_preflight_arm(
            "cas-completeness",
            armed=True,
            evidence={"missing": len(cas["missing"])},
            refusal_shape=CAS_INCOMPLETE,
        )
        if not cas["ok"]:
            listed = "; ".join(
                f"{row['intent']}:{row.get('url') or row.get('wheel')}"
                for row in cas["missing"][:10]
            )
            refusals.append(
                f"{CAS_INCOMPLETE}: {len(cas['missing'])} committed-family "
                f"artifact(s) are absent or corrupt in the campaign CAS at "
                f"{args.state_root} (first: {listed}). Pre-stage the CAS before "
                "an offline launch."
            )
    else:
        registry.register_preflight_arm(
            "operator-state-root",
            armed=False,
            evidence={"reason": "--state-root not given"},
            refusal_shape=STATE_ROOT_UNBOUND,
        )
        registry.register_preflight_arm(
            "cas-completeness",
            armed=False,
            evidence={"reason": "--state-root not given"},
            refusal_shape=CAS_INCOMPLETE,
        )
        print(
            "launch-preflight warning: --state-root not given; the CAS "
            "completeness precondition is NOT asserted",
            file=sys.stderr,
        )

    registry.assert_complete()
    report["armed_guard_census"] = registry.census()
    print(json.dumps(report, indent=2))
    if refusals and not args.report_only:
        for refusal in refusals:
            print(refusal, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
