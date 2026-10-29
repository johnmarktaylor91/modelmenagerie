"""The in-process ``crawl`` invocation preflight (overhaul RC-1, D.1 behavior 3).

One composed step, evaluated under the driver lock before any spend, in this
order: campaign binding, required secrets, disk floor (production floor plus a
5 GiB margin), pause sentinel, then the host-fact arms of the retired launch
preflight that answer REAL run failures -- environment rotation actually
rotated, CAS completeness for offline builds, lockless-intent refusal, the
CLI pins -- called as functions, never through a verdict file, and finally
closure-green: a consumable green verdict for every invocation-required gate,
keyed by that gate's input closure on the running tree (RC-3 certification by
closure: a commit that leaves a gate's declared inputs byte-identical keeps
its verdict consumable; a commit that moves them has none until a fresh run
appends one).

Dropped on purpose (RULING B.2 / D.1): certificate kind, audit-register
zero-GAP, verdict nonce and age. A refusal is typed (:class:`InvocationRefused`)
and the driver records it as an ``invocation-refused`` operational event that
counts toward NO fuse -- nothing was bought.

Arms that adjudicate facts only a real paid lane can violate (secrets, CAS,
CLI pins) register UNARMED with their reason when the invocation runs
in-process fake lanes; that disclosure follows the retired preflight's own
armed-guard census rule (a guard cannot exist outside the census, an unarmed
guard is disclosed, never silent).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from menagerie.crawler.driver_contracts import DriverIntegrationError

__all__ = [
    "DISK_FLOOR_MARGIN_BYTES",
    "PREFLIGHT_ARMS",
    "InvocationRefused",
    "PreflightArm",
    "PreflightReport",
    "run_invocation_preflight",
]

#: Margin above the production environment disk floor (D.1 behavior 3).
DISK_FLOOR_MARGIN_BYTES = 5 * 1024**3


class InvocationRefused(DriverIntegrationError):
    """A typed invocation refusal: the arm that refused and why."""

    def __init__(self, arm: str, detail: str) -> None:
        super().__init__(f"invocation refused at {arm}: {detail}")
        self.arm = arm
        self.detail = detail


@dataclass(frozen=True)
class PreflightArm:
    """One evaluated (or explicitly unarmed) preflight guard."""

    name: str
    armed: bool
    evidence: Mapping[str, Any] = field(default_factory=dict)
    refusal: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Return the census row."""

        return {
            "name": self.name,
            "armed": self.armed,
            "evidence": dict(self.evidence),
            "refusal": self.refusal,
        }


@dataclass(frozen=True)
class PreflightReport:
    """The ordered arm census and the first refusal, if any."""

    arms: tuple[PreflightArm, ...]

    @property
    def refusal(self) -> Optional[PreflightArm]:
        """Return the refusing arm, if one refused."""

        for arm in self.arms:
            if arm.refusal is not None:
                return arm
        return None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON census."""

        refusing = self.refusal
        return {
            "arms": [arm.to_dict() for arm in self.arms],
            "refused": refusing is not None,
            "refusing_arm": None if refusing is None else refusing.name,
        }

    def raise_for_refusal(self) -> None:
        """Raise :class:`InvocationRefused` for the first refusing arm."""

        refusing = self.refusal
        if refusing is not None:
            raise InvocationRefused(refusing.name, str(refusing.refusal))


def _disk_free(path: Path) -> int:
    """Return free bytes at the nearest existing ancestor of ``path``."""

    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return int(shutil.disk_usage(probe).free)


@dataclass(frozen=True)
class _ArmContext:
    """Everything one arm may read; arms never mutate it."""

    records_root: Path
    repo_root: Path
    runtime_root: Path
    campaign_id: str
    target: str
    real_lanes: bool
    envs_root: Path
    state_root: Optional[Path]
    tmp_root: Optional[Path]
    now: str
    disk_free: Callable[[Path], int]
    allow_lockless: tuple[str, ...]


def _refused(name: str, detail: str, **evidence: Any) -> PreflightArm:
    return PreflightArm(name, armed=True, evidence=evidence, refusal=detail)


def _unarmed(name: str, reason: str) -> PreflightArm:
    return PreflightArm(name, armed=False, evidence={"reason": reason})


def _arm_campaign_binding(ctx: _ArmContext) -> PreflightArm:
    """The campaign root binding must verify against the bytes."""

    from menagerie.crawler.rung_staging import (  # noqa: PLC0415 -- verified-root resolver (frozen manifest read)
        RungStagingError,
        resolve_verified_root,
    )

    try:
        binding, snapshot = resolve_verified_root(ctx.records_root, ctx.campaign_id)
    except RungStagingError as exc:
        return _refused("campaign-binding", str(exc), campaign_id=ctx.campaign_id)
    return PreflightArm(
        "campaign-binding",
        armed=True,
        evidence={
            "campaign_id": ctx.campaign_id,
            "intake_snapshot_id": snapshot.snapshot_id,
            "row_count": binding.row_count,
        },
    )


def _arm_secrets(ctx: _ArmContext) -> PreflightArm:
    """Required secrets present (real lanes only)."""

    if not ctx.real_lanes:
        return _unarmed("secrets", "in-process fake lanes")
    from menagerie.crawler.env_census import missing_wake_secrets

    missing = missing_wake_secrets()
    if missing:
        return _refused(
            "secrets", f"required secrets absent: {', '.join(missing)}", missing=list(missing)
        )
    return PreflightArm("secrets", armed=True, evidence={"missing": []})


def _arm_disk_floor(ctx: _ArmContext) -> PreflightArm:
    """Every write destination above the production floor plus the margin."""

    from menagerie.crawler.envs import DEFAULT_ENVIRONMENT_DISK_FLOOR_BYTES

    floor = int(DEFAULT_ENVIRONMENT_DISK_FLOOR_BYTES) + DISK_FLOOR_MARGIN_BYTES
    destinations = {"records-root": ctx.records_root, "driver-runtime": ctx.runtime_root}
    if ctx.state_root is not None:
        destinations["state-root"] = ctx.state_root
    if ctx.tmp_root is not None:
        destinations["tmp-root"] = ctx.tmp_root
    measured = {name: int(ctx.disk_free(path)) for name, path in destinations.items()}
    lowest = min(measured, key=lambda name: measured[name])
    if measured[lowest] < floor:
        return _refused(
            "disk-floor",
            f"{lowest} has {measured[lowest]} bytes free, below the floor {floor} "
            f"(production floor {DEFAULT_ENVIRONMENT_DISK_FLOOR_BYTES} + "
            f"{DISK_FLOOR_MARGIN_BYTES} margin)",
            measured=measured,
            floor_bytes=floor,
        )
    return PreflightArm(
        "disk-floor", armed=True, evidence={"measured": measured, "floor_bytes": floor}
    )


def _arm_pause_sentinel(ctx: _ArmContext) -> PreflightArm:
    """The durable sprint-pause sentinel (fail-closed on ambiguity)."""

    from menagerie.crawler.pause_authority import evaluate_sprint_pause

    hold = evaluate_sprint_pause(ctx.records_root, now=ctx.now)
    if hold.held:
        return _refused(
            "pause-sentinel",
            f"sprint-pause hold ({hold.decision}: {hold.detail or 'operator hold'})",
            decision=hold.decision,
        )
    return PreflightArm("pause-sentinel", armed=True, evidence={"decision": hold.decision})


def _arm_environment_rotation(ctx: _ArmContext) -> PreflightArm:
    """Environment rotation actually rotated (no re-claimable rebuild park)."""

    from menagerie.crawler.tools.launch_preflight import evaluate_rebuild_parks

    parks = evaluate_rebuild_parks(
        records_root=ctx.records_root, envs_root=ctx.envs_root, target=ctx.target
    )
    if parks["still_held"]:
        return _refused(
            "environment-rotation",
            f"{parks['still_held']} of {parks['rebuild_rows']} environment-rebuild park rows "
            "would be re-claimed: the committed setup-artifact identity still equals the "
            "quarantined one (an environment.yml edit is not a rotation)",
            rebuild_rows=parks["rebuild_rows"],
            still_held=parks["still_held"],
        )
    return PreflightArm(
        "environment-rotation",
        armed=True,
        evidence={"rebuild_rows": parks["rebuild_rows"], "still_held": parks["still_held"]},
    )


def _arm_cas_completeness(ctx: _ArmContext) -> PreflightArm:
    """Every committed-family artifact present in the campaign CAS (offline builds)."""

    if ctx.state_root is None:
        return _unarmed("cas-completeness", "no environment state root declared")
    from menagerie.crawler.tools.launch_preflight import evaluate_cas_completeness

    cas = evaluate_cas_completeness(
        envs_root=ctx.envs_root, state_root=ctx.state_root, target=ctx.target
    )
    if not cas.get("ok", False):
        return _refused(
            "cas-completeness",
            "campaign CAS is missing committed-family artifacts; offline builds would "
            f"fail mid-leg ({len(cas.get('missing', ()))} missing)",
            missing=len(cas.get("missing", ())),
        )
    return PreflightArm(
        "cas-completeness",
        armed=True,
        evidence={
            "verified_artifacts": cas.get("verified_artifacts"),
            "verified_wheels": cas.get("verified_wheels"),
        },
    )


def _arm_lockless_intent(ctx: _ArmContext) -> PreflightArm:
    """Every non-exempt intent has a proven locked family."""

    from menagerie.crawler.tools.launch_preflight import evaluate_family_readiness

    families = evaluate_family_readiness(
        envs_root=ctx.envs_root, target=ctx.target, allow_lockless=ctx.allow_lockless
    )
    if not families["ok"]:
        details = ", ".join(f"{row['intent']}({row['status']})" for row in families["refused"])
        return _refused(
            "lockless-intent",
            f"intent(s) without a proven locked family: {details}; unknown allowances: "
            f"{list(families['unknown_allowances'])}",
            refused=len(families["refused"]),
        )
    return PreflightArm(
        "lockless-intent", armed=True, evidence={"allow_lockless": list(ctx.allow_lockless)}
    )


def _arm_cli_pin(ctx: _ArmContext) -> PreflightArm:
    """Both paid-lane CLIs pinned and on their reviewed versions (real lanes only)."""

    if not ctx.real_lanes:
        return _unarmed("cli-pin", "in-process fake lanes")
    from menagerie.crawler.toolchain_pins import ToolchainPinError, require_verified_toolchain

    pins: dict[str, Any] = {}
    for lane in ("author", "checker"):
        try:
            report = require_verified_toolchain(lane, cache_key=("crawl", ctx.campaign_id))
        except ToolchainPinError as exc:
            return _refused("cli-pin", f"{lane} lane: {exc}", lane=lane)
        if not report.get("pinned"):
            return _refused(
                "cli-pin",
                f"{lane} lane CLI is not pinned (an unpinned production launch resolves "
                "through $PATH; declare the absolute pin)",
                lane=lane,
            )
        pins[lane] = report.get("observed_version")
    return PreflightArm("cli-pin", armed=True, evidence={"observed": pins})


def _arm_closure_green(ctx: _ArmContext) -> PreflightArm:
    """A consumable green verdict for every invocation-required gate, by closure.

    RC-3: each required gate's ``closure_sha`` is recomputed from the running
    tree's git objects and consumed through
    :func:`gate_verdicts.consume_required_gates_green`; the manifest's
    ``required_for_invocation`` list is the only authority over which gates
    must be green. A malformed manifest, a missing, stale, red, drifted, or
    evidence-truncated verdict all refuse typed. No flag and no environment
    variable skips this arm.
    """

    from menagerie.crawler.code_identity import observed_repo_tree
    from menagerie.crawler.tools import gate_closure, gate_verdicts

    tree = observed_repo_tree(ctx.repo_root)
    tree_sha = tree.get("tree_sha")
    if not isinstance(tree_sha, str):
        return _refused(
            "closure-green",
            f"the running tree at {ctx.repo_root} is unobservable (git answered nothing)",
        )
    try:
        rows = gate_verdicts.consume_required_gates_green(ctx.repo_root)
    except (gate_verdicts.GateVerdictError, gate_closure.GateClosureError) as exc:
        return _refused("closure-green", f"on tree {tree_sha}: {exc}", tree_sha=tree_sha)
    return PreflightArm(
        "closure-green",
        armed=True,
        evidence={
            "tree_sha": tree_sha,
            "dirty": tree.get("dirty"),
            "closures": {gate: str(row["closure_sha"]) for gate, row in sorted(rows.items())},
            "consumed": {gate: str(row["log_sha256"]) for gate, row in sorted(rows.items())},
        },
    )


#: The CLOSED arm sequence, in the order RULING D.1 behavior 3 names. The
#: driver lock is held by the caller and recorded first; evaluation stops at
#: the first refusal.
PREFLIGHT_ARMS: tuple[Callable[[_ArmContext], PreflightArm], ...] = (
    _arm_campaign_binding,
    _arm_secrets,
    _arm_disk_floor,
    _arm_pause_sentinel,
    _arm_environment_rotation,
    _arm_cas_completeness,
    _arm_lockless_intent,
    _arm_cli_pin,
    _arm_closure_green,
)


def run_invocation_preflight(
    *,
    records_root: Path,
    repo_root: Path,
    runtime_root: Path,
    campaign_id: str,
    target: str,
    real_lanes: bool,
    envs_root: Path,
    state_root: Optional[Path],
    tmp_root: Optional[Path],
    now: str,
    lock_path: Optional[Path] = None,
    disk_free: Callable[[Path], int] = _disk_free,
    allow_lockless: Sequence[str] = (),
) -> PreflightReport:
    """Evaluate every arm of :data:`PREFLIGHT_ARMS` in order; stop at the first refusal.

    Parameters
    ----------
    records_root, repo_root, runtime_root:
        Campaign records, the running tree, and the driver runtime root.
    campaign_id, target:
        Bound campaign and conda-lock platform.
    real_lanes:
        Whether the invocation buys real provider sessions (command or queue
        lanes). Fake in-process lanes leave the secrets/CAS/CLI-pin arms
        unarmed and disclosed.
    envs_root, state_root, tmp_root:
        Environment registry root, the operator environment state root
        (``MENAGERIE_ENVIRONMENT_STATE_ROOT``), and the env lane's staging root.
    now:
        Clock for the pause sentinel.
    lock_path:
        The held driver lock, recorded as evidence.
    disk_free:
        Measurement seam (bytes free at a path).
    allow_lockless:
        Lockless-intent exemptions (the EN-2 vocabulary).
    """

    ctx = _ArmContext(
        records_root=records_root,
        repo_root=repo_root,
        runtime_root=runtime_root,
        campaign_id=campaign_id,
        target=target,
        real_lanes=real_lanes,
        envs_root=envs_root,
        state_root=state_root,
        tmp_root=tmp_root,
        now=now,
        disk_free=disk_free,
        allow_lockless=tuple(allow_lockless),
    )
    arms: list[PreflightArm] = [
        PreflightArm(
            "driver-lock",
            armed=True,
            evidence={"lock_path": None if lock_path is None else str(lock_path)},
        )
    ]
    for arm in PREFLIGHT_ARMS:
        result = arm(ctx)
        arms.append(result)
        if result.refusal is not None:
            break
    return PreflightReport(tuple(arms))
