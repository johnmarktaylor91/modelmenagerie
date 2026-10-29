"""R6 checker-calibration runner: the live adjudicator binding (W2 rider).

The sanctioned ``MENAGERIE_CHECKER_CALIBRATION_RUN=1`` capture run, composed
ENTIRELY from production seams (never a reimplementation of any of them):

- frozen step-100 pack ITEMS re-sealed under the CURRENT pinned checker
  identity by the production envelope builders (``checker_dispatch``);
- the production operator wrapper subprocess through
  :class:`~menagerie.crawler.driver_admission.CommandCheckerLane` — which
  itself rides ``_run_operator_command``'s paid-custody guard, so every
  session holds a durable reservation, custody binding, and telemetry row
  exactly as a driver leg's sessions do;
- paid-session reserve/settle through :class:`PaidSessionAuthority` with the
  production checker :class:`BudgetPolicy` fuses and the receipt-backed
  provider reconciler;
- gate rows appended through :meth:`CanonicalReducer.append_gate` (all of its
  write-time doors armed: current prompt binding, derived component
  identities, intake membership).

SUBSTRATE: a DEDICATED calibration campaign root (fresh records + work
trees over the frozen step-100 intake snapshot), never the production
pilot's ledgers. The receipt mints and verifies against THIS root; the
production campaign's ledgers gain zero rows from the rung, so no
calibration verdict can ever be reused, cited, or awarded by a later leg
(W3/PD-9 by construction on top of the structural withholding). The root is
a durable artifact: keep it for receipt verification and the P5 re-audit.

W3 attestation: the runner refuses to start unless the calibration flag is
set, records :func:`calibration_environment_snapshot` into
``records/control/calibration-leg-config.json`` on every phase, and the
``census`` command derives the PD-9 bypass census over the root's own gate
rows (every verdict rendered here is bypass-classified forever — a
retroactively minted receipt can never bless them).

Commands
--------
``plan``       select frozen packs (no spend), write the plan manifest.
``preflight``  open the root, verify the toolchain pin, build one envelope,
               open the reducer — everything short of a paid dispatch.
``measure``    run measurement sessions to the per-lane floors (PAID).
``matrix``     run the B1-XV2 blind-replay matrix over the golden corpus
               (PAID) and write the report artifact.
``census``     derive the withholding/bypass census artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.checker_calibration import (
    CHECKER_CALIBRATION_RUN_ENV,
    calibration_bypass_census,
    calibration_environment_snapshot,
    calibration_run_active,
)
from menagerie.crawler.constants import (
    METADATA_BATCH_MAX,
    METADATA_BATCH_MIN,
    CHECKER_LEG_SESSION_BUDGET,
    CHECKER_LEG_WALL_BUDGET_SECONDS,
    CHECKER_MODEL_WALL_BUDGET_SECONDS,
    GateKind,
)
from menagerie.crawler.identity import stable_hash, utc_now
from menagerie.crawler.models import JsonObject

#: Per-lane measurement targets: floor + 1 margin, and the buy cap that
#: bounds dead sessions (a lane that cannot render its floor inside its cap
#: is broken weather, not calibration — the runner halts typed, never loops).
MEASURE_TARGETS: Mapping[str, int] = {
    GateKind.METADATA_BATCH.value: 21,
    GateKind.FIDELITY.value: 6,
    GateKind.TERMINAL_DISPOSITION.value: 6,
}
MEASURE_BUY_CAPS: Mapping[str, int] = {
    GateKind.METADATA_BATCH.value: 26,
    GateKind.FIDELITY.value: 9,
    GateKind.TERMINAL_DISPOSITION.value: 9,
}
#: Consecutive same-lane dispatch failures that halt the phase (anti-flail).
CONSECUTIVE_FAILURE_FUSE = 3

#: The matrix's spend-ledger disclosure kind — deliberately OUTSIDE the
#: closed GateKind vocabulary so replay sessions can never inflate any
#: measured lane's bought-session denominator.
MATRIX_DISPATCH_KIND = "blind-replay-matrix"


class CalibrationRunnerError(RuntimeError):
    """Typed halt: the runner refuses to continue (fuse, contract, weather)."""


def _require_calibration_mode() -> None:
    if not calibration_run_active():
        raise CalibrationRunnerError(
            f"the calibration runner requires {CHECKER_CALIBRATION_RUN_ENV}=1 — "
            "it is the sanctioned capture run, never a production lane"
        )


def _records_root(calibration_root: Path) -> Path:
    return calibration_root / "menagerie" / "crawler" / "records"


def _work_root(calibration_root: Path) -> Path:
    return calibration_root / ".crawl-local" / "work"


def _matrix_work_root(calibration_root: Path) -> Path:
    # Deliberately OUTSIDE the measurement work root: matrix attempt
    # telemetry must never pool into the receipt's measured attempt tail.
    return calibration_root / ".crawl-local" / "matrix-work"


def _write_leg_config_attestation(calibration_root: Path, *, phase: str) -> Path:
    """W3/TE-3: durably attest the verdict-steering env inventory per phase."""

    control = _records_root(calibration_root) / "control"
    control.mkdir(parents=True, exist_ok=True)
    path = control / "calibration-leg-config.json"
    rows = []
    if path.is_file():
        rows = json.loads(path.read_text(encoding="utf-8"))
    rows.append(
        {
            "phase": phase,
            "at": utc_now(),
            "checker_calibration": calibration_environment_snapshot(),
            "pid": os.getpid(),
        }
    )
    path.write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Frozen-pack selection (pure; unit-tested)
# ---------------------------------------------------------------------------


def eligible_frozen_packs(
    work_root: Path,
    *,
    intake_ids: frozenset[str],
    gate_kind: str,
    width: Optional[int] = None,
) -> list[JsonObject]:
    """Enumerate frozen request envelopes whose items all sit in the intake.

    Returns one entry per frozen ``request.json``:
    ``{"path", "items", "stable_ids", "member_set", "had_result"}`` in
    deterministic path order. ``width`` filters metadata packs to the
    production cap; single-item lanes pass ``None``.
    """

    packs: list[JsonObject] = []
    for request_path in sorted(work_root.rglob("request.json")):
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if request.get("gate_kind") != gate_kind:
            continue
        items = request.get("items")
        if not isinstance(items, list) or not items:
            continue
        if width is not None and len(items) != width:
            continue
        stable_ids = [str(item.get("stable_id")) for item in items]
        if any(stable_id not in intake_ids for stable_id in stable_ids):
            continue
        packs.append(
            {
                "path": str(request_path),
                "items": items,
                "stable_ids": stable_ids,
                "member_set": tuple(sorted(stable_ids)),
                "had_result": request_path.with_name("result.json").is_file(),
            }
        )
    return packs


def derive_width_packs(
    packs: Sequence[Mapping[str, Any]],
    *,
    width: int,
) -> list[JsonObject]:
    """Re-chunk frozen packs recorded at another width into production-width packs.

    A metadata batch is a GROUPING choice, not a property of its items: every
    frozen item re-seals at the current production width through the same
    envelope builder production uses (exactly how the blind-replay matrix's
    width variants are built). Packs already at ``width`` pass through
    unchanged; wider packs contribute every complete ``width``-sized chunk,
    each under a synthetic ``<path>#w<width>o<offset>`` key so the runner's
    resume accounting stays per-derived-pack. Short tails are dropped -- the
    measurement wants exact production-width sessions only (fable U-F2/C8).
    """

    derived: list[JsonObject] = []
    for pack in packs:
        items = list(pack["items"])
        if len(items) == width:
            derived.append(dict(pack))
            continue
        for offset in range(0, len(items) - width + 1, width):
            chunk = items[offset : offset + width]
            stable_ids = [str(item.get("stable_id")) for item in chunk]
            derived.append(
                {
                    "path": f"{pack['path']}#w{width}o{offset}",
                    "items": chunk,
                    "stable_ids": stable_ids,
                    "member_set": tuple(sorted(stable_ids)),
                    "had_result": pack["had_result"],
                }
            )
    return derived


def select_measurement_plan(
    packs: Sequence[Mapping[str, Any]],
    *,
    count: int,
) -> list[JsonObject]:
    """Pick ``count`` packs, distinct member-sets first, stable order.

    Distinct questions maximize what the sample measures; only when the
    distinct pool is exhausted are re-asks of already-selected member sets
    admitted (a re-ask is still a real production-shaped session).
    """

    chosen: list[JsonObject] = []
    taken_paths: set[str] = set()
    seen: set[tuple[str, ...]] = set()
    for pack in packs:
        if len(chosen) >= count:
            return chosen
        if pack["member_set"] in seen:
            continue
        seen.add(pack["member_set"])
        taken_paths.add(str(pack["path"]))
        chosen.append(dict(pack))
    for pack in packs:
        if len(chosen) >= count:
            break
        if str(pack["path"]) in taken_paths:
            continue
        taken_paths.add(str(pack["path"]))
        chosen.append(dict(pack))
    return chosen


# ---------------------------------------------------------------------------
# Verdict projection (pure; unit-tested) — the matrix's comparison surface
# ---------------------------------------------------------------------------


def verdict_projection(gate_kind: str, result: Mapping[str, Any]) -> dict[str, Any]:
    """Project one validated checker result onto its verdict-relevant facts.

    The projection is exactly what a rotation or replay may not silently
    flip (the m4976 shape): per-item verdicts, integrity, and rung checks —
    never prose, latency, or machine-owned scaffold fields.
    """

    projection: dict[str, Any] = {}
    for item in result.get("items", []):
        stable_id = str(item.get("stable_id"))
        if gate_kind == GateKind.FIDELITY.value:
            projection[stable_id] = {
                "fidelity": (item.get("fidelity") or {}).get("verdict"),
                "rung_check": (item.get("rung_check") or {}).get("verdict"),
                "integrity": (item.get("integrity") or {}).get("verdict"),
            }
        elif gate_kind == GateKind.TERMINAL_DISPOSITION.value:
            projection[stable_id] = {
                "terminal_disposition": (item.get("terminal_disposition") or {}).get("verdict"),
                "integrity": (item.get("integrity") or {}).get("verdict"),
            }
        else:
            projection[stable_id] = {
                "verdict": item.get("verdict"),
                "rung_check": (item.get("rung_check") or {}).get("verdict"),
                "integrity": (item.get("integrity") or {}).get("verdict"),
            }
    return projection


# ---------------------------------------------------------------------------
# The live session engine (production seams only)
# ---------------------------------------------------------------------------


class CalibrationEngine:
    """One opened calibration root: snapshot, generation, money, lane, reducer."""

    def __init__(self, calibration_root: Path, *, intake_dir: Path, checker_command: Sequence[str]):
        _require_calibration_mode()
        from menagerie.crawler.cli import _snapshot_authority_context
        from menagerie.crawler.driver_admission import CommandCheckerLane
        from menagerie.crawler.driver_contracts import DriverConfig
        from menagerie.crawler.intake import load_intake_snapshot
        from menagerie.crawler.leg_generation import open_leg_generation
        from menagerie.crawler.paid_session_lifecycle import (
            CommandProviderReconciler,
            PaidSessionAuthority,
        )

        self.calibration_root = calibration_root
        self.records_root = _records_root(calibration_root)
        self.work_root = _work_root(calibration_root)
        self.records_root.mkdir(parents=True, exist_ok=True)
        self.work_root.mkdir(parents=True, exist_ok=True)
        (self.records_root / "operational").mkdir(parents=True, exist_ok=True)

        staged_intake = calibration_root / "intake-snapshot"
        if not staged_intake.exists():
            shutil.copytree(intake_dir, staged_intake)
        self.snapshot = load_intake_snapshot(staged_intake)
        self.intake_ids = frozenset(item.stable_id for item in self.snapshot.items)

        # Campaign provenance: the checker serves the c1-mech tier; the run id
        # names the rung so every money row and gate row is self-identifying.
        from menagerie.crawler.operator_checker import METADATA_CHECKER_MODEL

        self.config = DriverConfig(
            review_checkpoint_at=10**9,
            author_model="claude-fable-5-1",
            # The seated PRIMARY (seat swap 2026-09-01): the command lane
            # refuses a campaign binding that disagrees with the seat it seals.
            checker_model=METADATA_CHECKER_MODEL,
            run_id=f"r6-checker-calibration-{stable_hash(str(calibration_root))[7:19]}",
            campaign_id="c1-mech",
        )
        self.context = _snapshot_authority_context(self.snapshot, self.config)
        self.generation = open_leg_generation(
            self.records_root,
            campaign_id="c1-mech",
            intake_snapshot_id=self.snapshot.snapshot_id,
            intake_snapshot_sha256=self.snapshot.snapshot_sha256,
            phase="r6-checker-calibration",
            run_id=self.config.run_id,
            opened_at=utc_now(),
        )
        self.authority = PaidSessionAuthority(
            self.records_root / "operational" / "spend.jsonl",
            runtime_root=calibration_root / ".crawl-local",
            clock=utc_now,
        )
        interpreter = str(checker_command[0])
        self.reconciler = CommandProviderReconciler(
            (
                interpreter,
                "-m",
                "menagerie.crawler.tools.provider_reconcile_evidence",
                "--records-root",
                str(self.records_root),
            )
        )
        self.lane = CommandCheckerLane(tuple(checker_command))
        self.checker_command = tuple(checker_command)

    # -- money ---------------------------------------------------------------

    def _verify_toolchain(self) -> None:
        from menagerie.crawler.toolchain_pins import require_verified_toolchain

        from menagerie.crawler.toolchain_pins import checker_toolchain_lane

        # The pin of the seat this engine's identity runs under (the Claude CLI
        # for the Fable primary, codex for the openai seat).
        identity = self.lane.adjudication_identity(GateKind.METADATA_BATCH.value, self.config)
        require_verified_toolchain(
            checker_toolchain_lane(identity.provider),
            cache_key=("c1-mech", self.generation.generation_id),
        )

    def reserve(self, *, dispatch_kind: str, gate_kind: Optional[str], items: Sequence[Mapping[str, Any]]):
        """W32-verified production reserve under the checker BudgetPolicy fuses."""

        from menagerie.crawler.driver_admission import (
            CHECKER_LANE_BACKOFF_SLACK_SECONDS,
            CHECKER_LANE_WALL_GRACE_SECONDS,
        )
        from menagerie.crawler.operator_protocol import checker_session_kill_seconds
        from menagerie.crawler.paid_session_lifecycle import BudgetPolicy

        self._verify_toolchain()
        identity = self.lane.adjudication_identity(
            gate_kind or GateKind.METADATA_BATCH.value, self.config
        )
        # The per-work-generation QUOTA instrument
        # (CHECKER_MODEL_WALL_BUDGET_SECONDS cumulative per member) bounds how
        # much checker wall one model's work generation may consume — a
        # production anti-runaway semantic. The blind-replay matrix DELIBERATELY
        # re-asks one frozen pack >= 4 times (B1-XV2's replays + rotation), so
        # for matrix reservations that instrument double-charges planned
        # replays and structurally refuses the leg mid-matrix (observed live
        # 2026-08-26: m10949 refused the metadata_accept golden's first
        # replay). Matrix reservations therefore carry the LEG-level wall as
        # their per-member quota; every runaway guard stays fully armed for
        # them (runaway session budget, generation wall budget, reservation
        # deadline, custody binding, chain). Measurement lanes keep the
        # production per-member cap unchanged.
        quota_wall_seconds = (
            CHECKER_LEG_WALL_BUDGET_SECONDS
            if dispatch_kind == MATRIX_DISPATCH_KIND
            else CHECKER_MODEL_WALL_BUDGET_SECONDS
        )
        return self.authority.reserve(
            scope={
                "campaign_id": "c1-mech",
                "generation_id": self.generation.generation_id,
                "lane": "checker",
                "dispatch_kind": dispatch_kind,
                "gate_kind": gate_kind if gate_kind is not None else dispatch_kind,
                "work_generations": [
                    {
                        "stable_id": str(item.get("stable_id")),
                        "work_id": str(item.get("work_id") or f"work-{item.get('stable_id')}"),
                    }
                    for item in items
                ],
                "checker_identity": {
                    "provider": identity.provider,
                    "model": identity.model,
                    "version": identity.version,
                    "prompt_sha256": identity.prompt_sha256,
                },
                "run_id": self.config.run_id,
                "machine_id": self.config.machine_id,
            },
            policy=BudgetPolicy(
                quota_sessions=2**31 - 1,
                quota_wall_seconds=quota_wall_seconds,
                runaway_sessions=CHECKER_LEG_SESSION_BUDGET,
                runaway_wall_seconds=CHECKER_LEG_WALL_BUDGET_SECONDS,
                reservation_wall_seconds=(
                    checker_session_kill_seconds()
                    + CHECKER_LANE_BACKOFF_SLACK_SECONDS
                    + CHECKER_LANE_WALL_GRACE_SECONDS
                ),
            ),
        )

    def settle(self, reservation, *, elapsed_seconds: float, completed: bool) -> None:
        self.authority.settle(
            reservation,
            provider_query=self.reconciler,
            custody_outcome="exited" if completed else "exited-error",
            machine_wall_seconds=max(0.0, float(elapsed_seconds)),
            local_outcome="completed" if completed else "failed",
        )

    # -- envelopes -----------------------------------------------------------

    def build_envelope(
        self,
        gate_kind: str,
        items: Sequence[Mapping[str, Any]],
        *,
        nonce: str,
        session_root: Path,
        checker_model: Optional[str] = None,
    ) -> JsonObject:
        from menagerie.crawler.checker_dispatch import (
            build_fidelity_envelope,
            build_metadata_vet_envelope,
            build_terminal_disposition_envelope,
        )
        from menagerie.crawler.operator_checker import (
            FIDELITY_CHECKER_MODEL,
            METADATA_CHECKER_MODEL,
            TERMINAL_CHECKER_MODEL,
        )

        output_path = session_root / "result.json"
        if gate_kind == GateKind.METADATA_BATCH.value:
            return build_metadata_vet_envelope(
                list(items),
                gate_round=1,
                output_path=output_path,
                checker_model=checker_model or METADATA_CHECKER_MODEL,
                checker_version=self.config.checker_version,
                request_nonce=nonce,
                final_tail=len(items) < METADATA_BATCH_MIN,
            )
        if gate_kind == GateKind.FIDELITY.value:
            return build_fidelity_envelope(
                dict(items[0]),
                gate_round=1,
                output_path=output_path,
                checker_model=checker_model or FIDELITY_CHECKER_MODEL,
                checker_version=self.config.checker_version,
                request_nonce=nonce,
            )
        return build_terminal_disposition_envelope(
            dict(items[0]),
            gate_round=1,
            output_path=output_path,
            checker_model=checker_model or TERMINAL_CHECKER_MODEL,
            checker_version=self.config.checker_version,
            request_nonce=nonce,
        )

    def dispatch(self, envelope: JsonObject, session_root: Path, reservation):
        """One paid wrapper session under the production custody guard."""

        from menagerie.crawler.paid_session_lifecycle import authorized_paid_dispatch

        started = time.monotonic()
        try:
            with authorized_paid_dispatch(reservation):
                # Production seam reuse over reimplementation: _run is the
                # exact write/execute/classify/validate path a driver leg's
                # sessions ride (including telemetry + custody FDs).
                outcome = self.lane._run(envelope, session_root)  # noqa: SLF001
        except BaseException:
            self.settle(reservation, elapsed_seconds=time.monotonic() - started, completed=False)
            raise
        self.settle(reservation, elapsed_seconds=time.monotonic() - started, completed=True)
        return outcome


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


def _flush_ledger(path: Path, ledger: Mapping[str, Any]) -> None:
    """Durably persist the runner's session log after every completion.

    The spend and gates ledgers are the authorities; this log only keeps the
    runner's own resume accounting (buy caps, used packs) crash-honest.
    """

    path.write_text(json.dumps(ledger, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def run_measure(
    engine: CalibrationEngine,
    plan_path: Path,
    *,
    concurrency: int,
    max_sessions: Optional[int] = None,
    metadata_extra: int = 0,
) -> JsonObject:
    """Buy sessions to the per-lane floors; append every rendered gate row.

    ``metadata_extra`` raises the metadata rendered target AND buy cap by that
    many sessions for THIS invocation only -- the corpus-coverage top-up lever
    (e.g. the width-1 rotation needed a real accept-class exemplar beyond the
    measurement floor). It can only buy MORE production-shaped evidence, never
    less: the receipt floors (``CALIBRATION_MIN_SESSIONS``) and the money
    fuses (``BudgetPolicy``) are enforced elsewhere and are untouched. The
    override is disclosed in the returned ledger.
    """

    from menagerie.crawler.driver_models import _normalize_gate_generation, _require_gate
    from menagerie.crawler.reducer import CanonicalReducer
    from menagerie.crawler.tools.restore_superseded_award import _ledger_paths

    _write_leg_config_attestation(engine.calibration_root, phase="measure")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    ledger_path = engine.calibration_root / "measurement-ledger.json"
    ledger: JsonObject = {"sessions": [], "halted": None}
    if ledger_path.is_file():
        prior = json.loads(ledger_path.read_text(encoding="utf-8"))
        ledger["sessions"] = list(prior.get("sessions", []))

    with CanonicalReducer(_ledger_paths(engine.records_root), engine.context) as reducer:

        def rendered_counts() -> dict[str, int]:
            counts = {kind.value: 0 for kind in GateKind}
            for row in reducer.gate_records:
                kind = row.get("gate_kind")
                if kind not in counts or row.get("verdict_source") == "reused":
                    continue
                if (
                    kind == GateKind.METADATA_BATCH.value
                    and row.get("batch_size") != METADATA_BATCH_MAX
                ):
                    # Rendered-session floors count the PRODUCTION width only,
                    # mirroring gather_lane_measurements: a verdict rendered at
                    # another width measured a different lane (fable U-F2/C8),
                    # so it can never satisfy this lane's floor.
                    continue
                counts[kind] += 1
            return counts

        # Resume-aware accounting: buys are durable facts (spend rows), so a
        # re-invocation seeds its caps from the prior sessions and never
        # re-asks a pack it already spent on.
        bought = {kind.value: 0 for kind in GateKind}
        used_packs: set[str] = set()
        for session in ledger["sessions"]:
            lane = session.get("lane")
            if lane in bought:
                bought[lane] += 1
            used_packs.add(str(session.get("pack")))
        failures = {kind.value: 0 for kind in GateKind}

        queue: list[tuple[str, JsonObject]] = []
        for lane_kind in (
            GateKind.TERMINAL_DISPOSITION.value,
            GateKind.FIDELITY.value,
            GateKind.METADATA_BATCH.value,
        ):
            for pack in plan.get(lane_kind, []):
                if str(pack["path"]) not in used_packs:
                    queue.append((lane_kind, pack))

        def run_one(lane_kind: str, pack: JsonObject) -> tuple[str, JsonObject, Any]:
            nonce = stable_hash([lane_kind, pack["path"], utc_now()])[7:23]
            session_root = engine.work_root / lane_kind / f"cal-{nonce}"
            session_root.mkdir(parents=True, exist_ok=True)
            reservation = engine.reserve(
                dispatch_kind=lane_kind, gate_kind=lane_kind, items=pack["items"]
            )
            envelope = engine.build_envelope(
                lane_kind, pack["items"], nonce=nonce, session_root=session_root
            )
            started = time.monotonic()
            outcome = engine.dispatch(envelope, session_root, reservation)
            return lane_kind, pack, (outcome, time.monotonic() - started, str(session_root))

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            pending: dict[Any, tuple[str, JsonObject]] = {}

            def refill() -> None:
                counts = rendered_counts()
                inflight = {kind.value: 0 for kind in GateKind}
                for lane_kind, _pack in pending.values():
                    inflight[lane_kind] += 1
                while queue:
                    if max_sessions is not None and sum(bought.values()) >= max_sessions:
                        return
                    lane_kind, pack = queue[0]
                    extra = (
                        metadata_extra
                        if lane_kind == GateKind.METADATA_BATCH.value
                        else 0
                    )
                    target = MEASURE_TARGETS[lane_kind] + extra
                    cap = MEASURE_BUY_CAPS[lane_kind] + extra
                    if counts[lane_kind] + inflight[lane_kind] >= target:
                        queue.pop(0)
                        continue
                    if bought[lane_kind] + inflight[lane_kind] >= cap:
                        queue.pop(0)
                        continue
                    if len(pending) >= max(1, concurrency):
                        return
                    queue.pop(0)
                    bought[lane_kind] += 1
                    future = pool.submit(run_one, lane_kind, pack)
                    pending[future] = (lane_kind, pack)
                    inflight[lane_kind] += 1

            refill()
            while pending:
                done = next(as_completed(list(pending)))
                lane_kind, pack = pending.pop(done)
                try:
                    _kind, _pack, (outcome, elapsed, session_root) = done.result()
                    if outcome.backoff is not None:
                        raise CalibrationRunnerError(
                            f"checker provider backoff during calibration: {outcome.backoff}"
                        )
                    raw_gate = _require_gate(outcome)
                    gate = _normalize_gate_generation(
                        raw_gate, reducer.gate_records, pack["stable_ids"]
                    )
                    reducer.append_gate(gate)
                    failures[lane_kind] = 0
                    ledger["sessions"].append(
                        {
                            "lane": lane_kind,
                            "pack": pack["path"],
                            "session_root": session_root,
                            "wall_seconds": round(elapsed, 3),
                            "rendered": True,
                            "gate_id": gate.get("gate_id"),
                        }
                    )
                    _flush_ledger(ledger_path, ledger)
                except CalibrationRunnerError:
                    raise
                except Exception as exc:  # noqa: BLE001 — recorded, bounded by the fuse
                    failures[lane_kind] += 1
                    ledger["sessions"].append(
                        {
                            "lane": lane_kind,
                            "pack": pack["path"],
                            "rendered": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    _flush_ledger(ledger_path, ledger)
                    if failures[lane_kind] >= CONSECUTIVE_FAILURE_FUSE:
                        ledger["halted"] = (
                            f"{lane_kind}: {CONSECUTIVE_FAILURE_FUSE} consecutive dispatch "
                            "failures — typed halt, root-cause before more spend"
                        )
                        for future in pending:
                            future.cancel()
                        raise CalibrationRunnerError(ledger["halted"])
                refill()

        ledger["rendered"] = rendered_counts()
        ledger["bought"] = bought
        if metadata_extra:
            ledger["metadata_extra_override"] = int(metadata_extra)
    return ledger


def adjudicate_one(engine: CalibrationEngine, golden: Mapping[str, Any]) -> JsonObject:
    """Answer ONE golden question with a paid production-wrapper session.

    The in-process half of the rotation contract: a rotated identity epoch is
    a LEG-BOUNDARY act (``leg_checker_prompt_sha256`` pins per process and
    refuses drift, BL-10 e), so the rotation adjudicator runs THIS function in
    a fresh subprocess whose ``PYTHONPATH`` names the deliberately rotated
    tree — that process re-pins the rotated prompt, builds envelopes under it,
    and its wrapper child validates against the same rotated identity.
    """

    from menagerie.crawler.checker_dispatch import validate_checker_result_mapping

    kind = str(golden.get("kind"))
    request = golden.get("request") or {}
    items = (request.get("items") or []) if isinstance(request, Mapping) else []
    if not items:
        raise CalibrationRunnerError(f"golden {golden.get('name')!r} carries no items")
    nonce = stable_hash([golden.get("name"), utc_now()])[7:23]
    session_root = _matrix_work_root(engine.calibration_root) / f"{golden.get('name')}-{nonce}"
    session_root.mkdir(parents=True, exist_ok=True)
    reservation = engine.reserve(dispatch_kind=MATRIX_DISPATCH_KIND, gate_kind=None, items=items)
    envelope = engine.build_envelope(kind, items, nonce=nonce, session_root=session_root)
    started = time.monotonic()
    outcome = engine.dispatch(envelope, session_root, reservation)
    elapsed = time.monotonic() - started
    if outcome.backoff is not None:
        raise CalibrationRunnerError(
            f"checker provider backoff during matrix replay: {outcome.backoff}"
        )
    result = outcome.gate
    validate_checker_result_mapping(result, envelope)
    disputes = 0
    for item in result.get("items", []):
        integrity = item.get("integrity")
        integrity = integrity if isinstance(integrity, Mapping) else {}
        disputes += len(integrity.get("locator_failures") or []) + len(
            integrity.get("excerpt_discrepancies") or []
        )
    result_path = session_root / "result.json"
    result_sha256 = (
        "sha256:" + hashlib.sha256(result_path.read_bytes()).hexdigest()
        if result_path.is_file()
        else stable_hash(result)
    )
    return {
        "verdicts": verdict_projection(kind, result),
        "latency_seconds": outcome.wall_seconds or elapsed,
        "result_sha256": result_sha256,
        "session_root": str(session_root),
        "integrity_dispute_entries": disputes,
    }


def run_matrix(
    engine: CalibrationEngine,
    corpus_dir: Path,
    *,
    report_path: Path,
    rotated_tree: Path,
    rotation_args: Sequence[str],
    width_variant_goldens: int = 2,
) -> JsonObject:
    """B1-XV2: replay every golden 3x same-epoch + rotation + width variants."""

    import subprocess
    import tempfile

    from menagerie.crawler.tools.blind_replay_matrix import (
        load_golden_corpus_entries,
        run_blind_replay_matrix,
        write_report,
    )

    _write_leg_config_attestation(engine.calibration_root, phase="matrix")
    entries = load_golden_corpus_entries(corpus_dir)
    matrix_root = _matrix_work_root(engine.calibration_root)
    matrix_root.mkdir(parents=True, exist_ok=True)

    # Width-invariance variants: the first N cap-width metadata goldens gain a
    # width-1 variant built from their own first item (the shared-item
    # comparison is the harness's contract).
    varied = 0
    for entry in entries:
        request = entry.get("request") or {}
        items = request.get("items") or []
        if (
            varied < width_variant_goldens
            and entry.get("kind") == GateKind.METADATA_BATCH.value
            and len(items) == METADATA_BATCH_MAX
        ):
            entry["width_variants"] = {1: {"items": [items[0]]}}
            varied += 1

    def same_epoch_adjudicate(golden: Mapping[str, Any]) -> Mapping[str, Any]:
        return adjudicate_one(engine, golden)

    def rotation_adjudicate(golden: Mapping[str, Any]) -> Mapping[str, Any]:
        # A DIFFERENT identity epoch: the rotated tree carries a deliberately
        # edited checker prompt, and prompt identity pins per PROCESS
        # (BL-10 e), so the rotated ask runs adjudicate-one in a fresh
        # subprocess whose PYTHONPATH names that tree. Its wrapper child
        # inherits the same environment and validates against the rotated
        # prompt bytes — a real epoch change end to end, same pinned model.
        with tempfile.TemporaryDirectory(prefix="rotation-golden-") as scratch:
            golden_path = Path(scratch) / "golden.json"
            out_path = Path(scratch) / "answer.json"
            golden_path.write_text(
                json.dumps(
                    {
                        "name": golden.get("name"),
                        "kind": golden.get("kind"),
                        "width": golden.get("width"),
                        "request": golden.get("request"),
                    }
                ),
                encoding="utf-8",
            )
            child_env = dict(os.environ)
            child_env["PYTHONPATH"] = str(rotated_tree)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "menagerie.crawler.tools.checker_calibration_runner",
                    *rotation_args,
                    "adjudicate-one",
                    "--golden-file",
                    str(golden_path),
                    "--out",
                    str(out_path),
                ],
                env=child_env,
                cwd=str(rotated_tree),
                capture_output=True,
                text=True,
                timeout=3600,
            )
            if completed.returncode != 0 or not out_path.is_file():
                raise CalibrationRunnerError(
                    "rotation adjudicator subprocess failed "
                    f"(rc={completed.returncode}): {completed.stderr[-800:]}"
                )
            return json.loads(out_path.read_text(encoding="utf-8"))

    report = run_blind_replay_matrix(
        entries,
        same_epoch_adjudicate,
        rotation_adjudicate=rotation_adjudicate,
        replays_per_golden=3,
        checker_model="gpt-5.6-sol",
        captured_at=utc_now(),
    )
    write_report(report, report_path)
    return report


def run_census(calibration_root: Path, *, receipt_path: Optional[Path]) -> JsonObject:
    """Derive the PD-9 bypass / W3 withholding census over the root's rows."""

    from menagerie.crawler.checker_calibration import (
        load_checker_calibration_receipt,
        shadow_rehearsal_census,
    )
    from menagerie.crawler.recordio import scan_jsonl

    gates_path = _records_root(calibration_root) / "gates" / "current-shard.jsonl"
    if not gates_path.is_file():
        gates_path = _records_root(calibration_root) / "gates.jsonl"
    rows = list(scan_jsonl(gates_path)) if gates_path.is_file() else []
    receipt = None
    if receipt_path is not None and receipt_path.is_file():
        receipt = load_checker_calibration_receipt(receipt_path)
    census = calibration_bypass_census(rows, receipt)
    census["calibration_gate_rows"] = len(rows)
    # W3 third-mode disclosure arm, beside PD-9 and derived the same way:
    # from the ledger rows' own markers, never from the env flag. Segregated
    # -- shadow-rendered verdicts are attested measurements, distinct from
    # bypass-adjudicated production verdicts, and never counted as either
    # production-adjudicated or receipt-attested.
    census["shadow_rehearsal"] = shadow_rehearsal_census(rows)
    census["withholding"] = {
        "award_heads_issued": 0,
        "note": (
            "the calibration root's models ledger gains no runs award heads; the "
            "reducer's calibration-mode withhold (W3) refuses issuance structurally "
            "and this rung never routes an award candidate at all"
        ),
    }
    return census


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--pilot-work-root", type=Path, default=None)
    parser.add_argument("--intake", type=Path, default=None)
    parser.add_argument(
        "--checker-command",
        default=None,
        help="argv prefix for the production checker wrapper (space-split)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    sub.add_parser("preflight")
    measure = sub.add_parser("measure")
    measure.add_argument("--concurrency", type=int, default=4)
    measure.add_argument("--max-sessions", type=int, default=None)
    measure.add_argument(
        "--extra-metadata-sessions",
        type=int,
        default=0,
        help=(
            "raise the metadata rendered target and buy cap by N for this "
            "invocation (corpus-coverage top-up; disclosed in the ledger)"
        ),
    )
    matrix = sub.add_parser("matrix")
    matrix.add_argument("--corpus", type=Path, required=True)
    matrix.add_argument("--report", type=Path, required=True)
    matrix.add_argument(
        "--rotated-tree",
        type=Path,
        required=True,
        help="checkout carrying the deliberately rotated checker prompt (B1-XV2's 4th replay)",
    )
    adjudicate = sub.add_parser("adjudicate-one")
    adjudicate.add_argument("--golden-file", type=Path, required=True)
    adjudicate.add_argument("--out", type=Path, required=True)
    census = sub.add_parser("census")
    census.add_argument("--receipt", type=Path, default=None)
    census.add_argument("--out", type=Path, required=True)
    return parser


def _engine(args: argparse.Namespace) -> CalibrationEngine:
    if args.intake is None or args.checker_command is None:
        raise CalibrationRunnerError("--intake and --checker-command are required")
    return CalibrationEngine(
        args.calibration_root.resolve(),
        intake_dir=args.intake.resolve(),
        checker_command=tuple(str(args.checker_command).split()),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    plan_path = args.calibration_root / "measurement-plan.json"
    if args.command == "plan":
        if args.pilot_work_root is None or args.intake is None:
            raise CalibrationRunnerError("plan requires --pilot-work-root and --intake")
        intake_ids = frozenset(
            json.loads(line)["stable_id"]
            for line in (args.intake / "items.jsonl").read_text().splitlines()
            if line.strip()
        )
        plan: JsonObject = {}
        for lane_kind, width, count in (
            (GateKind.METADATA_BATCH.value, METADATA_BATCH_MAX, MEASURE_BUY_CAPS["metadata_batch"]),
            (GateKind.FIDELITY.value, None, MEASURE_BUY_CAPS["fidelity"]),
            (GateKind.TERMINAL_DISPOSITION.value, None, MEASURE_BUY_CAPS["terminal_disposition"]),
        ):
            packs = eligible_frozen_packs(
                args.pilot_work_root, intake_ids=intake_ids, gate_kind=lane_kind, width=width
            )
            if width is not None and len({pack["member_set"] for pack in packs}) < count:
                # Not enough frozen packs recorded AT the production width
                # (e.g. after a width rotation): re-chunk the lane's wider
                # frozen packs into production-width packs -- same items, same
                # envelope builder, the grouping production would use.
                wider = eligible_frozen_packs(
                    args.pilot_work_root, intake_ids=intake_ids, gate_kind=lane_kind, width=None
                )
                packs = list(packs) + [
                    pack
                    for pack in derive_width_packs(wider, width=width)
                    if "#" in str(pack["path"])  # derived chunks only; exact-width
                    # passthroughs are already in `packs`
                ]
            plan[lane_kind] = select_measurement_plan(packs, count=count)
        args.calibration_root.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {lane: len(entries) for lane, entries in plan.items()}
                | {"plan": str(plan_path)},
                indent=1,
            )
        )
        return 0
    if args.command == "preflight":
        engine = _engine(args)
        pack = json.loads(plan_path.read_text(encoding="utf-8"))[GateKind.METADATA_BATCH.value][0]
        envelope = engine.build_envelope(
            GateKind.METADATA_BATCH.value,
            pack["items"],
            nonce="preflight",
            session_root=engine.work_root / "preflight",
        )
        engine._verify_toolchain()  # noqa: SLF001 — the W32 gate, pre-spend
        from menagerie.crawler.reducer import CanonicalReducer
        from menagerie.crawler.tools.restore_superseded_award import _ledger_paths

        with CanonicalReducer(_ledger_paths(engine.records_root), engine.context) as reducer:
            persisted = len(reducer.gate_records)
        print(
            json.dumps(
                {
                    "generation_id": engine.generation.generation_id,
                    "envelope_gate_kind": envelope.get("gate_kind"),
                    "envelope_model": (envelope.get("checker") or {}).get("model"),
                    "envelope_prompt_sha256": (envelope.get("checker") or {}).get("prompt_sha256"),
                    "persisted_gate_rows": persisted,
                    "intake_members": len(engine.intake_ids),
                    "calibration_mode": calibration_run_active(),
                },
                indent=1,
            )
        )
        return 0
    if args.command == "measure":
        engine = _engine(args)
        ledger = run_measure(
            engine,
            plan_path,
            concurrency=args.concurrency,
            max_sessions=args.max_sessions,
            metadata_extra=max(0, int(args.extra_metadata_sessions)),
        )
        out = args.calibration_root / "measurement-ledger.json"
        out.write_text(json.dumps(ledger, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"rendered": ledger["rendered"], "bought": ledger["bought"]}, indent=1))
        return 0
    if args.command == "adjudicate-one":
        engine = _engine(args)
        golden = json.loads(args.golden_file.read_text(encoding="utf-8"))
        answer = adjudicate_one(engine, golden)
        args.out.write_text(json.dumps(answer, sort_keys=True) + "\n", encoding="utf-8")
        return 0
    if args.command == "matrix":
        engine = _engine(args)
        report = run_matrix(
            engine,
            args.corpus.resolve(),
            report_path=args.report.resolve(),
            rotated_tree=args.rotated_tree.resolve(),
            rotation_args=(
                "--calibration-root",
                str(args.calibration_root.resolve()),
                "--intake",
                str(args.intake.resolve()),
                "--checker-command",
                str(args.checker_command),
            ),
        )
        print(
            json.dumps(
                {
                    "goldens": len(report.get("goldens", [])),
                    "production_width_latencies": len(
                        report.get("latency_seconds_at_production_width", [])
                    ),
                    "report": str(args.report),
                },
                indent=1,
            )
        )
        return 0
    if args.command == "census":
        census = run_census(args.calibration_root.resolve(), receipt_path=args.receipt)
        args.out.write_text(json.dumps(census, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(census, indent=1, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
