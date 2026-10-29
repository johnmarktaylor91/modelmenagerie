"""Progress reporting, notification, and durable driver state I/O."""

from __future__ import annotations
import json
import logging
import os
import shlex
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional
from menagerie.crawler.author_dispatch import (
    AuthorBackoffSignal,
)
from menagerie.crawler.checker_dispatch import (
    CheckerBackoffSignal,
)
from menagerie.crawler.checkpoint import (
    FunnelSnapshot,
)
from menagerie.crawler.constants import (
    AUTHOR_BILLING_RECHECK_SECONDS,
    DEFAULT_NOTIFY_TIMEOUT_SECONDS,
    AuthorPauseReason,
    framework_from_markers,
)
from menagerie.crawler.env_lifecycle import (
    DiskFloorError,
    DiskRecoveryError,
    EnvironmentProbeError,
    EnvironmentSolveError,
)
from menagerie.crawler.identity import (
    canonical_json_bytes,
    durable_fsync,
    stable_hash,
    utc_now,
)
from menagerie.crawler.notification_outbox import (
    default_outbox_path,
    record_send_intent,
    record_send_settled,
)
from menagerie.crawler.notification_policy import (
    ESSENTIAL_NOTIFICATION_CLASSES,
    NotificationClass,
    should_deliver,
)
from menagerie.crawler.operator_notify import EXIT_TRANSPORT_DISCARDING, read_receipt
from menagerie.crawler.intake import (
    IntakeItem,
    legacy_requires_fidelity_audit,
)
from menagerie.crawler.models import JsonObject
from menagerie.crawler.status import (
    funnel_counts,
)
from menagerie.crawler.driver_contracts import (
    DriverIntegrationError,
)

LOGGER = logging.getLogger("menagerie.crawler.driver")


@dataclass(frozen=True)
class EnvironmentFailureRoute:
    """One deliberate routing decision for one lifecycle exception type.

    ``kind`` is the closed decision vocabulary:

    * ``"park-intent-setup"`` -- a fact about the intent's SETUP (committed
      family, materialized bytes, overlay, probes): park every pending work
      identity; the park holds while the setup-artifact identity is unchanged
      and releases on rotation or an explicit operator clearance.
    * ``"park-host-condition"`` -- a transient HOST/infrastructure condition
      (busy lifecycle, store unmounted, teardown residue, wrapper outage):
      park every pending work identity; the park never re-claims on a later
      run -- the live valve re-derives it if the condition persists.
    * ``"park-disk-floor"`` -- a measured disk refusal (floor or observed
      ENOSPC): park in the durable disk-floor machine-deferral ledger.
    * ``"terminal"`` -- a deliberate, typed model disposition; ``stage`` and
      ``reason_code`` are required. The ONLY current member is the
      supervisor's fail-closed sandbox signal.
    * ``"propagate"`` -- a harness/integration defect: re-raise loudly. A
      broken program is not a fact about any model and will recur for every
      model after it.
    """

    exception_type: type[BaseException]
    kind: str
    failure_class: str
    stage: Optional[str] = None
    reason_code: Optional[str] = None


def _environment_failure_routes() -> tuple[EnvironmentFailureRoute, ...]:
    """Build the closed, ordered pre-use environment failure routing registry.

    Most-specific rows first (``isinstance`` walks the table in order). There
    is deliberately NO ``Exception`` row: the old catch-all ``(Exception,
    "environment", "build-failed")`` minted intent-wide MODEL terminals for
    pure host conditions (T5-2 mutation-verified, T3-T2 3-lab, 19/63 live
    pilot heads), and an exception nobody adjudicated must be a loud typed
    refusal, never a silent world-fact.

    Imported lazily to keep this module's import graph light; the registry is
    built once at module import below.
    """

    from menagerie.crawler.authority import AuthorityDerivationError
    from menagerie.crawler.effort import EffortCapExceeded
    from menagerie.crawler.env_lifecycle import (
        CommittedLockFamilyError,
        EnvironmentBackendContractError,
        EnvironmentBackendUnavailableError,
        EnvironmentBusyError,
        EnvironmentCleanupError,
        EnvironmentExactnessError,
        EnvironmentOverlayError,
    )

    return (
        # Measured disk refusals park behind the disk-floor valve.
        EnvironmentFailureRoute(DiskFloorError, "park-disk-floor", "disk-floor"),
        # NON-floor disk recovery (teardown residue) is host state needing
        # cleanup: it defers, and never re-mints the abolished
        # ``failed:resource/disk-floor`` terminal (T3-T2c).
        EnvironmentFailureRoute(DiskRecoveryError, "park-host-condition", "host-cleanup"),
        EnvironmentFailureRoute(EnvironmentBusyError, "park-host-condition", "lifecycle-busy"),
        EnvironmentFailureRoute(EnvironmentCleanupError, "park-host-condition", "cleanup-failed"),
        EnvironmentFailureRoute(
            EnvironmentBackendUnavailableError,
            "park-host-condition",
            "backend-unavailable",
        ),
        EnvironmentFailureRoute(
            EnvironmentBackendContractError, "park-intent-setup", "backend-rejected"
        ),
        EnvironmentFailureRoute(EnvironmentExactnessError, "park-intent-setup", "exactness"),
        # CommittedLockFamilyError subclasses the solve arm; the explicit row
        # keeps its failure_class distinct in park rows and events.
        EnvironmentFailureRoute(
            CommittedLockFamilyError, "park-intent-setup", "no-committed-family"
        ),
        EnvironmentFailureRoute(EnvironmentSolveError, "park-intent-setup", "solve"),
        EnvironmentFailureRoute(EnvironmentProbeError, "park-intent-setup", "probe"),
        EnvironmentFailureRoute(EnvironmentOverlayError, "park-intent-setup", "overlay"),
        EnvironmentFailureRoute(
            AuthorityDerivationError, "park-intent-setup", "environment-authority"
        ),
        EnvironmentFailureRoute(
            EffortCapExceeded, "park-intent-setup", "environment-effort-exhausted"
        ),
        # Bare host I/O (store SSD unmounted, dangling env_root symlink). A
        # raw ENOSPC is converted upstream by the lifecycle into the typed
        # floor condition; anything that still arrives here is host state.
        EnvironmentFailureRoute(OSError, "park-host-condition", "host-io"),
        # A harness defect is campaign-level and loud, never a model fact.
        EnvironmentFailureRoute(DriverIntegrationError, "propagate", "integration"),
    )


_ENVIRONMENT_FAILURE_ROUTES: tuple[EnvironmentFailureRoute, ...] = _environment_failure_routes()
if any(route.exception_type in (Exception, BaseException) for route in _ENVIRONMENT_FAILURE_ROUTES):
    raise RuntimeError("the environment failure routing registry may never carry a catch-all row")


def write_driver_status_marker(
    runtime_root: Path,
    *,
    leg_ordinal: Optional[int],
    arm: str,
    detail: str = "",
) -> None:
    """Write the driver-owned "what am I doing right now" marker atomically.

    ONE writer (the driver, on BOTH run paths -- supervised and daemonized;
    the 200a supervisor heartbeat rotted because it was bound to only one),
    consumed by ``crawl_doctor`` with staleness-as-death semantics: a stale
    marker's claims are disclosed as dead, never trusted. Write failures are
    swallowed -- the marker is advisory; the driver.lock kernel flock stays
    the liveness authority.
    """

    from menagerie.crawler.tools.crawl_doctor import (
        DRIVER_STATUS_MARKER_SCHEMA_VERSION,
        driver_status_marker_path,
    )

    path = driver_status_marker_path(runtime_root)
    payload = {
        "schema_version": DRIVER_STATUS_MARKER_SCHEMA_VERSION,
        "written_at": utc_now(),
        "pid": os.getpid(),
        "leg_ordinal": leg_ordinal,
        "arm": arm,
        "detail": detail[:500],
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp, path)
    except OSError:  # pragma: no cover -- advisory surface, never driver-fatal
        LOGGER.warning("driver status marker write failed at %s", path)


def _is_critical_class(notification_class: Optional[str]) -> bool:
    """Return whether one class value names an essential (critical) class.

    Unknown and unclassified values are NOT critical: they already fail open
    through :func:`should_deliver`, and the outbox obligation is earned by
    explicit classification exactly like suppression is.
    """

    if notification_class is None:
        return False
    try:
        resolved = NotificationClass(notification_class)
    except ValueError:
        return False
    return resolved in ESSENTIAL_NOTIFICATION_CLASSES


class CommandNotifier:
    """Best-effort argv-only notifier with log-only fallback."""

    def __init__(
        self,
        command: Optional[str],
        *,
        timeout_seconds: float = DEFAULT_NOTIFY_TIMEOUT_SECONDS,
        receipt_root: Optional[Path] = None,
        outbox_path: Optional[Path] = None,
    ) -> None:
        """Resolve a notifier and bind its nonce-receipt evidence root."""

        if timeout_seconds <= 0:
            raise ValueError("notifier timeout must be positive")
        self._argv = _resolve_notify_command(command)
        self._timeout_seconds = timeout_seconds
        self._receipt_root = receipt_root
        self._outbox_path = outbox_path

    @property
    def command(self) -> Optional[tuple[str, ...]]:
        """Return the resolved notifier command.

        Returns
        -------
        tuple[str, ...] | None
            Resolved argv, or ``None`` for log-only notification.
        """

        return self._argv

    def notify(
        self,
        summary: str,
        *,
        idempotency_key: str,
        notification_class: Optional[str] = None,
    ) -> bool:
        """Invoke the notifier and require its matching nonce receipt.

        A classified, non-essential notification is settled by the class
        policy WITHOUT reaching the transport: it is logged, and the call
        reports settled (True) so retry ladders and dedup state treat the
        deliberate silence as handled rather than as a delivery failure to
        escalate. Unclassified calls always attempt delivery (fail-open).

        A CRITICAL-class send (essential per the closed policy set) is
        additionally receipt-backed through the durable outbox (D-38): the
        intent is recorded before the transport runs, and only a
        transport-ECHOED receipt settles it as sent. An absent transport is a
        typed REFUSAL for critical classes -- never a shrugged log line --
        and an exit-0 send whose transport did not echo the key stays
        pending, so the retry ladder re-drives it and the monitor alarms.
        """

        ascii_summary = _ascii_line(summary)
        if not should_deliver(notification_class):
            LOGGER.info(
                "crawler notification suppressed by class policy (%s): %s",
                notification_class,
                ascii_summary,
            )
            return True
        receipt_root = self._receipt_root or Path.cwd() / ".crawl-local" / "notification-receipts"
        critical = _is_critical_class(notification_class)
        outbox_path = self._outbox_path or default_outbox_path(receipt_root)
        if critical:
            record_send_intent(
                outbox_path,
                nonce=idempotency_key,
                notification_class=str(notification_class),
                message=ascii_summary,
            )
        if self._argv is None:
            if critical:
                LOGGER.error(
                    "crawler CRITICAL notification REFUSED: no transport is"
                    " configured for class %s: %s",
                    notification_class,
                    ascii_summary,
                )
                record_send_settled(
                    outbox_path,
                    nonce=idempotency_key,
                    outcome="refused-transport-absent",
                )
                return False
            LOGGER.warning("crawler notification (log-only): %s", ascii_summary)
            return False
        receipt_root.mkdir(parents=True, exist_ok=True)
        receipt_name = stable_hash({"notification_idempotency_key": idempotency_key}).removeprefix(
            "sha256:"
        )
        receipt_path = receipt_root / f"{receipt_name}.json"
        receipt_path.unlink(missing_ok=True)
        try:
            completed = subprocess.run(
                [*self._argv, ascii_summary],
                check=False,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
                env={
                    **os.environ,
                    "MENAGERIE_NOTIFICATION_IDEMPOTENCY_KEY": idempotency_key,
                    "MENAGERIE_NOTIFICATION_RECEIPT_PATH": str(receipt_path),
                },
            )
        except subprocess.TimeoutExpired:
            LOGGER.warning(
                "crawler notifier timed out after %.1fs: %s",
                self._timeout_seconds,
                ascii_summary,
            )
            if critical:
                record_send_settled(
                    outbox_path, nonce=idempotency_key, outcome="transport-failed",
                    detail=f"timeout after {self._timeout_seconds:.1f}s",
                )
            return False
        except Exception as exc:  # noqa: BLE001 -- notifications are strictly best-effort
            LOGGER.warning("crawler notifier failed (%s): %s", exc, ascii_summary)
            if critical:
                record_send_settled(
                    outbox_path, nonce=idempotency_key, outcome="transport-failed",
                    detail=str(exc),
                )
            return False
        if completed.returncode != 0:
            LOGGER.warning(
                "crawler notifier exited %s (%s): %s",
                completed.returncode,
                completed.stderr.strip(),
                ascii_summary,
            )
            if critical:
                outcome = (
                    "refused-transport-discarding"
                    if completed.returncode == EXIT_TRANSPORT_DISCARDING
                    else "transport-failed"
                )
                record_send_settled(
                    outbox_path, nonce=idempotency_key, outcome=outcome,
                    detail=f"exit {completed.returncode}: {completed.stderr.strip()[:200]}",
                )
            return False
        receipt = read_receipt(receipt_path)
        if receipt is None or receipt.get("nonce") != idempotency_key:
            LOGGER.warning("crawler notifier returned success without a matching nonce receipt")
            if critical:
                record_send_settled(
                    outbox_path, nonce=idempotency_key, outcome="transport-failed",
                    detail="exit 0 without a matching nonce receipt",
                )
            return False
        if critical:
            if receipt.get("transport_echoed_key") is True:
                record_send_settled(
                    outbox_path, nonce=idempotency_key, outcome="sent-echoed",
                )
                return True
            LOGGER.warning(
                "crawler CRITICAL notification is UNPROVEN: transport exited 0"
                " without echoing the idempotency key; the send stays pending"
                " for re-drive: %s",
                ascii_summary,
            )
            record_send_settled(
                outbox_path, nonce=idempotency_key, outcome="sent-unechoed",
            )
            return False
        return True


def _funnel_snapshot(current: Mapping[str, Mapping[str, Any]]) -> FunnelSnapshot:
    """Collapse terminal status codes into the four review buckets."""

    counts: Counter[str] = Counter()
    for record in current.values():
        counts[str(record["status"]["kind"])] += 1
    return FunnelSnapshot(
        runs=counts["runs"],
        deferred=counts["deferred"],
        skipped=counts["skipped"],
        failed=counts["failed"],
    )


def _completion_workflows(current: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    """Return pending workflow gates not expressible by terminal partition shape."""

    workflows: list[str] = []
    for record in current.values():
        flags = set(record.get("flags", [])) | set(
            record.get("intake", {}).get("preserved_legacy_flags", [])
        )
        fidelity = record.get("fidelity", {})
        if legacy_requires_fidelity_audit(flags) and (
            not fidelity.get("gate_id")
            or fidelity.get("verdict")
            not in {"match", "minor-drift", "major-drift", "slop", "cannot-verify"}
        ):
            workflows.append("fidelity-pending")
    return tuple(workflows)


def _review_report(current: Mapping[str, JsonObject], threshold: int) -> JsonObject:
    """Build the blocking checkpoint report required for human review."""

    snapshot = _funnel_snapshot(current)
    fidelity = Counter(
        str(record.get("fidelity", {}).get("verdict") or "not-required")
        for record in current.values()
    )
    accepted = [
        record
        for _stable_id, record in sorted(current.items())
        if record.get("status", {}).get("kind") == "runs"
    ][:5]
    total = max(1, len(current))
    skip_rate = snapshot.skipped / total
    concerns: list[str] = []
    if skip_rate >= 0.20:
        concerns.append(f"skip-rate spike: {skip_rate:.1%}")
    if snapshot.failed / total >= 0.20:
        concerns.append(f"failure-rate spike: {snapshot.failed / total:.1%}")
    return {
        "report_kind": "menagerie-crawler-review-checkpoint-v1",
        "review_checkpoint_at": threshold,
        "models_completed": len(current),
        "funnel_snapshot": snapshot.to_dict(),
        "funnel_counts": dict(funnel_counts(current)),
        "fidelity_verdict_distribution": dict(sorted(fidelity.items())),
        "accepted_sample": accepted,
        "concerning_patterns": concerns,
    }


def _progress_summary(completed: int, milestone: int, snapshot: FunnelSnapshot) -> str:
    """Return one concise ASCII progress notification."""

    return _ascii_line(
        f"Menagerie crawler milestone {milestone}: completed={completed} "
        f"runs={snapshot.runs} deferred={snapshot.deferred} skipped={snapshot.skipped} "
        f"failed={snapshot.failed}"
    )


def _review_summary(completed: int, snapshot: FunnelSnapshot, report_path: Path) -> str:
    """Return one concise ASCII blocking-review notification."""

    return _ascii_line(
        f"Menagerie crawler review checkpoint: completed={completed} runs={snapshot.runs} "
        f"deferred={snapshot.deferred} skipped={snapshot.skipped} failed={snapshot.failed} "
        f"report={report_path}"
    )


def _ascii_line(value: str) -> str:
    """Normalize arbitrary text into one plain-ASCII line."""

    return " ".join(value.encode("ascii", errors="replace").decode("ascii").split())


def _resolve_notify_command(command: Optional[str]) -> Optional[tuple[str, ...]]:
    """Resolve an explicit notifier, or the delivery script behind the receipt shim.

    The transport search order is unchanged. What changed is that the resolved default is
    wrapped in ``menagerie.crawler.operator_notify``, which delivers through that same
    script and then writes the nonce receipt the strict doctor requires. Without the
    wrapper there is no way to distinguish "notification delivered" from "notifier script
    happened to exist", which is precisely the unattended-operation failure the doctor
    check exists to catch. An explicit ``command`` is still honored verbatim; an operator
    who overrides it owns providing the receipt.
    """

    from menagerie.crawler.operator_notify import resolve_transport

    if command:
        parsed = tuple(shlex.split(command))
        return parsed or None
    transport = resolve_transport()
    if transport is None:
        return None
    return (
        sys.executable,
        "-m",
        "menagerie.crawler.operator_notify",
        "--transport",
        shlex.join(transport),
    )


def _framework_from_intake(item: IntakeItem) -> str:
    """Infer only the closed native-tail markers; otherwise route to PyTorch.

    Delegates to :func:`menagerie.crawler.constants.framework_from_markers` so
    the read-only rung staging tools can ask the same question without importing
    the driver.
    """

    return framework_from_markers(item.zoo, item.name)


def _framework_phase(item: IntakeItem) -> str:
    """Return the configured scheduler phase for one intake row."""

    return "pytorch" if _framework_from_intake(item) == "pytorch" else "native-tail"


def _environment_failure(exc: Exception) -> EnvironmentFailureRoute:
    """Resolve one pre-use environment failure to its deliberate route.

    Parameters
    ----------
    exc:
        Exception observed by the driver's environment admission arm.

    Returns
    -------
    EnvironmentFailureRoute
        The registry row governing this exception.

    Raises
    ------
    DriverIntegrationError
        For an exception with NO deliberate routing. This is the removed
        catch-all's replacement: an unadjudicated exception refuses the run
        loudly with its real type named, instead of silently minting
        intent-wide ``failed:environment / build-failed`` model terminals
        (T5-2/T3-T2; 19/63 live pilot heads were that fan-out's product).
    """

    # Sandbox admission is identified by both type name and defining module. It stays
    # explicit because reducing that precondition to an exception-class row would weaken it.
    if _is_sandbox_unavailable(exc):
        return EnvironmentFailureRoute(
            type(exc),
            "terminal",
            "sandbox-unavailable",
            stage="policy",
            reason_code="sandbox-unavailable-v1",
        )
    for route in _ENVIRONMENT_FAILURE_ROUTES:
        if isinstance(exc, route.exception_type):
            return route
    raise DriverIntegrationError(
        "environment failure has no deliberate routing and refuses loudly instead "
        f"of minting model terminals: {type(exc).__module__}.{type(exc).__qualname__}: {exc}"
    )


def _is_sandbox_unavailable(exc: Exception) -> bool:
    """Recognize the supervisor's typed fail-closed sandbox signal."""

    return (
        type(exc).__name__ == "SandboxUnavailableError"
        and type(exc).__module__ == "menagerie.crawler.policy"
    )


def _future_reset(now: str, signal: CheckerBackoffSignal | AuthorBackoffSignal) -> str:
    """Compute an exact reset timestamp when the provider supplied only a delay.

    The one-hour default is a usage-window guess. A billing failure has no window
    at all -- money does not refill on a schedule -- so it gets the sparse
    operator-recheck interval instead: an hourly wake into a dead billing account
    buys one failed session per hour, forever, and clears nothing.
    """

    instant = datetime.fromisoformat(now.removesuffix("Z") + "+00:00")
    if signal.retry_after_seconds is not None:
        seconds: float = signal.retry_after_seconds
    elif signal.reason.value == AuthorPauseReason.BILLING_FAILURE.value:
        seconds = AUTHOR_BILLING_RECHECK_SECONDS
    else:
        seconds = 3600
    return (instant + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def _normalize_wake_reset(reset_at: str) -> str:
    """Render one validated reset in the RFC 3339 ``Z`` form the wake layer demands.

    ``plausible_author_reset_at`` deliberately returns its candidate verbatim, so a
    provider-declared reset can arrive in ``+00:00`` offset form. ``wakeup._parse_utc``
    rejects anything not ending in ``Z``, which would turn a correctly detected usage
    pause into a ``WakeupConfigurationError`` at the moment the campaign most needs to
    park. Guessed resets already end in ``Z`` (see :func:`_future_reset`), so only the
    observed path was exposed -- i.e. exactly the path that a real limit takes.

    This normalizes representation only. An unparseable value is returned untouched so
    the wake layer still refuses it rather than having a malformed reset laundered here.
    """

    try:
        parsed = datetime.fromisoformat(reset_at.strip().replace("Z", "+00:00"))
    except ValueError:
        return reset_at
    if parsed.tzinfo is None:
        return reset_at
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_driver_state(path: Path) -> JsonObject:
    """Load the disposable cursor, treating absence as a fresh campaign."""

    if not path.is_file():
        return {"status": "new"}
    return _read_json(path)


def _write_driver_state(path: Path, state: Mapping[str, Any]) -> None:
    """Atomically persist the disposable scheduler cursor."""

    _write_json_atomic(path, state)


def _read_json(path: Path) -> JsonObject:
    """Read one JSON object or raise a typed integration error."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DriverIntegrationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DriverIntegrationError(f"expected a JSON object at {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically fsync one deterministic JSON object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = canonical_json_bytes(value) + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            durable_fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
