"""One durable paid-session authority, composed from three separated concerns.

This module is the seam and nothing else.  It owns no budget arithmetic (that is
:mod:`menagerie.crawler.paid_session_accounting`, pure over durable rows), no
byte durability (that is :mod:`menagerie.crawler.paid_session_durability`, the
anchored append-only substrate), and no kernel liveness (that is
:mod:`menagerie.crawler.paid_session_custody`).  The three concerns share ZERO
mutable state, so a fix in one cannot regress another -- the anti-regression rule
that six incremental rounds of this surface violated.

What the seam guarantees:

* Authorization is INSIDE :meth:`PaidSessionAuthority.reserve`.  A lane cannot
  dispatch without a :class:`Reservation` it cannot fabricate, and a REFUSED
  reservation mints no row at all -- the budget predicate runs before any byte
  is written, inside the same exclusive lock that would have written it.
* :meth:`PaidSessionAuthority.settle` finalizes BOTH axes from evidence.  The
  session count is provider-attested in both directions and cannot be minted
  from a local flag by any caller, writer or subprocess.  The wall charge is
  ``min(reserved, max(machine_floor, provider_attested))`` -- a machine-side
  floor a remote answer can raise but never delete, recorded beside the
  provider's own raw number so a replay re-derives the charge and sees the clamp.
* There is no subtractive money operation and no mutable spend counter.  Money
  truth is the row sequence; every number is a pure projection of it.
* Crash ambiguity between reserve and settle is resolved by
  :meth:`PaidSessionAuthority.reconcile` -- a provider idempotency key plus a
  provider query -- wired at leg start, terminal, and never a permanent block.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

from menagerie.crawler.identity import (
    stable_hash,
    stamp_row_digest,
    stamp_shadow_rehearsal,
)
from menagerie.crawler.paid_session_accounting import (
    ACCOUNTING_AXIS_PROVENANCE,
    ACCOUNTING_STATS,
    CUSTODY_RELEASE_OUTCOMES,
    INSTRUMENTS,
    LANES,
    LEGACY_SETTLEMENT_SCHEMAS,
    LIFECYCLE_EVENTS,
    LOCAL_OUTCOMES,
    MACHINE_WALL_SOURCES,
    MONEY_EVENTS,
    PAID_SESSION_CHAIN_SCOPE,
    PAID_SESSION_CUSTODY_RELEASE_SCHEMA,
    PAID_SESSION_CUSTODY_SCHEMA,
    PAID_SESSION_GENESIS_SCHEMA,
    PAID_SESSION_RESERVATION_SCHEMA,
    PAID_SESSION_ROLLOVER_SCHEMA,
    PAID_SESSION_SETTLEMENT_SCHEMA,
    REQUIRED_AXIS_FIELDS,
    SETTLEMENT_EVIDENCE,
    AccountingState,
    BudgetPolicy,
    PaidSessionBudgetExceeded,
    PaidSessionError,
    Weight,
    authorize,
    campaign_spend,
    charge,
    charged_wall_seconds,
    finite_nonnegative,
    index_rows,
    quota_attempt_groups,
    quota_spend,
    reservation_attempt_group,
    runaway_spend,
    settled_session_weight,
    work_generations,
)
from menagerie.crawler.ledger_chain import extend_chain_state
from menagerie.crawler.paid_session_custody import PaidCustodyLease
from menagerie.crawler.paid_session_durability import (
    DURABILITY_STATS,
    AnchoredLedger,
    LedgerSnapshot,
    preserved_lineage_verifier,
)
from menagerie.crawler.worker_supervisor import current_boot_id, process_start_token

__all__ = [
    "ACCOUNTING_AXIS_PROVENANCE",
    "ACCOUNTING_STATS",
    "CUSTODY_RELEASE_OUTCOMES",
    "DURABILITY_STATS",
    "INSTRUMENTS",
    "LIFECYCLE_EVENTS",
    "LOCAL_OUTCOMES",
    "MACHINE_WALL_SOURCES",
    "MONEY_EVENTS",
    "PAID_SESSION_CHAIN_SCOPE",
    "PROVIDER_STATUS_FIELDS",
    "REQUIRED_AXIS_FIELDS",
    "SETTLEMENT_EVIDENCE",
    "AccountingState",
    "AnchoredLedger",
    "BudgetPolicy",
    "CommandProviderReconciler",
    "NEGATIVE_CONTROL_CONTRACT",
    "PaidSessionAuthority",
    "PaidSessionBudgetExceeded",
    "PaidSessionError",
    "ProviderReconcileRefusal",
    "ProviderReconciliation",
    "ReconcilerNegativeControlError",
    "Reservation",
    "Weight",
    "authorized_paid_dispatch",
    "verify_reconciler_negative_control",
    "campaign_spend",
    "charge",
    "charged_wall_seconds",
    "current_paid_reservation",
    "index_rows",
    "load_paid_session_rows",
    "quota_attempt_groups",
    "quota_spend",
    "reservation_attempt_group",
    "runaway_spend",
    "settled_session_weight",
]

#: Named derivation for the incremental verified read of the paid ledger.
_PROJECTION = "paid.sessions.accounting"
_SUBJECT = "paid-session"

#: The exact field contract a provider status adapter must answer with.  The
#: echoed key is what makes the answer an attestation ABOUT a reservation rather
#: than a self-describing string: the same nonce-echo the notification lane
#: already requires of a third party.  ``adapter_identity`` (D-1) is the
#: adapter's own name/version stamp: 200a's two deployed adapters (the retired
#: constant-accept stub and the evidence adapter) were observationally
#: identical on every ledger row, so no receipt could show WHICH reconciler
#: answered.  The identity rides inside the status receipt, so
#: ``receipt_sha256`` digest-covers it on every settlement row prospectively;
#: a legacy three-field adapter now refuses typed at the contract check.
PROVIDER_STATUS_FIELDS = frozenset(
    {"idempotency_key", "accepted", "observed_wall_seconds", "adapter_identity"}
)

#: Preimage contract for the canary nonce the negative control queries with.
NEGATIVE_CONTROL_CONTRACT = "paid-session-reconciler-negative-control-v1"


class ProviderReconcileRefusal(PaidSessionError):
    """The adapter REFUSED to answer one key (nonzero exit), typed.

    Distinct from the base class so the negative control can tell an honest
    unknown-key refusal (the control HOLDING) from an outage or a malformed
    answer without ever reading message text.
    """


class ReconcilerNegativeControlError(PaidSessionError):
    """The wired reconciler ANSWERED a canary nonce no reservation names.

    The constant-accept class: an adapter that answers about keys outside its
    ledger is not evidence-derived, whatever it answered.  Raised typed at the
    admission boundary so the leg refuses loudly until the wiring is fixed.
    """


@dataclass(frozen=True)
class ProviderReconciliation:
    """Authoritative provider answer for one idempotency key.

    Parameters
    ----------
    accepted:
        Whether the provider consummated (and therefore bills) the session.
    observed_wall_seconds:
        The provider's own attested billing wall, or ``None`` when the provider
        does not meter wall clock.  A DECLINED session may still report metered
        wall: that is a truthful answer, and refusing to represent it is what
        previously made such a reservation permanently unsettleable.
    receipt_sha256:
        Digest of the exact status receipt.
    adapter_identity:
        The answering adapter's name/version stamp, digest-covered through the
        receipt for command adapters.  ``None`` only for in-process rig
        reconcilers; ``CommandProviderReconciler`` always populates it.
    """

    accepted: bool
    observed_wall_seconds: Optional[float]
    receipt_sha256: str
    adapter_identity: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate the provider receipt without accepting ambiguous answers."""

        if not isinstance(self.accepted, bool):
            raise ValueError("provider reconciliation acceptance must be boolean")
        if not self.receipt_sha256:
            raise ValueError("provider reconciliation requires a receipt digest")
        if self.observed_wall_seconds is not None and (
            not math.isfinite(self.observed_wall_seconds) or self.observed_wall_seconds < 0.0
        ):
            raise ValueError("provider reconciliation wall must be finite and non-negative")
        if self.adapter_identity is not None and (
            not isinstance(self.adapter_identity, str) or not self.adapter_identity.strip()
        ):
            raise ValueError("provider reconciliation adapter identity must be a nonempty string")


@dataclass(frozen=True)
class CommandProviderReconciler:
    """Bounded provider-status query command used by production crash recovery.

    The command receives one paid-session idempotency key as its only argument
    and must print exactly ``{"idempotency_key", "accepted",
    "observed_wall_seconds", "adapter_identity"}`` as JSON.  The echoed key is
    REQUIRED and is checked against the queried one, so an adapter cannot
    answer about a different session and the receipt digest cannot merely
    attest itself; the identity stamp names WHICH adapter answered, inside the
    digested receipt.
    """

    command: tuple[str, ...]
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        """Reject an absent command, an invalid timeout, or the retired stub."""

        if not self.command or any(not isinstance(part, str) or not part for part in self.command):
            raise ValueError("provider reconciliation command must be a nonempty argv")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0.0:
            raise ValueError("provider reconciliation timeout must be finite and positive")
        # D-1: the launch-era constant-accept stub is RETIRED. A persisted
        # legacy campaign config (the pilot's v5) or a stale launchd export can
        # resurrect its command line on a bare-environment wake; refusing the
        # exact retired spelling at construction turns that resurrection into a
        # typed launch-time refusal instead of a silent ambiguity-as-spent
        # reconciler.  Impostors under other names are caught behaviorally by
        # :func:`verify_reconciler_negative_control`.
        if any(part.endswith("provider_reconcile_accept.py") for part in self.command):
            raise ValueError(
                "provider reconciliation command names the retired constant-accept "
                "adapter (provider_reconcile_accept.py); re-point the campaign "
                "config at an evidence-custody adapter"
            )

    def __call__(self, provider_idempotency_key: str) -> ProviderReconciliation:
        """Query one key and validate the command's exact JSON status receipt."""

        if not provider_idempotency_key:
            raise PaidSessionError("provider reconciliation requires an idempotency key")
        try:
            process = subprocess.run(
                [*self.command, provider_idempotency_key],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PaidSessionError("provider reconciliation query did not complete") from exc
        if process.returncode != 0:
            raise ProviderReconcileRefusal(
                "provider reconciliation query refused the idempotency key: "
                + process.stderr[-500:]
            )
        try:
            value = json.loads(process.stdout)
        except ValueError as exc:
            raise PaidSessionError("provider reconciliation query returned invalid JSON") from exc
        if not isinstance(value, Mapping) or set(value) != PROVIDER_STATUS_FIELDS:
            raise PaidSessionError(
                "provider reconciliation query returned an invalid field contract"
            )
        if value["idempotency_key"] != provider_idempotency_key:
            raise PaidSessionError(
                "provider reconciliation answered about a different idempotency key"
            )
        accepted = value["accepted"]
        observed = value["observed_wall_seconds"]
        adapter_identity = value["adapter_identity"]
        if not isinstance(accepted, bool):
            raise PaidSessionError("provider reconciliation accepted must be boolean")
        if not isinstance(adapter_identity, str) or not adapter_identity.strip():
            raise PaidSessionError(
                "provider reconciliation adapter identity must be a nonempty string"
            )
        if observed is not None:
            try:
                observed = finite_nonnegative(observed, field="observed_wall_seconds")
            except ValueError as exc:
                raise PaidSessionError(str(exc)) from exc
        receipt_sha256 = stable_hash(
            {
                "provider_idempotency_key": provider_idempotency_key,
                "status_receipt": dict(value),
            }
        )
        return ProviderReconciliation(
            accepted, observed, receipt_sha256, adapter_identity=adapter_identity
        )


def verify_reconciler_negative_control(
    query: Callable[[str], "ProviderReconciliation"],
) -> str:
    """Prove the wired reconciler refuses a canary key no reservation names.

    The STANDING negative control (D-1/DX-75/B2-X-V12): 200a's constant-accept
    stub and the evidence adapter were observationally identical on every
    ledger row (1547/1547 ``accepted: true``), so no ledger evidence could
    falsify which reconciler was wired.  This probe can: a well-formed key is
    synthesized under :data:`NEGATIVE_CONTROL_CONTRACT` with fresh OS entropy,
    so no ledger can hold a reservation naming it, and an evidence-derived
    adapter must REFUSE it (:class:`ProviderReconcileRefusal`).  ANY answer --
    accepted or declined -- proves the adapter speaks about keys outside its
    ledger and refuses admission typed.

    An outage (the query did not complete, malformed JSON, contract mismatch)
    propagates as its own typed :class:`PaidSessionError`: it is not a pass --
    the admission boundary surfaces it loudly and retries at the next leg
    start, exactly like a failed real reconciliation.

    Parameters
    ----------
    query:
        The wired provider status adapter (production: the command adapter).

    Returns
    -------
    str
        The canary nonce key, for the caller's disclosure trail.

    Raises
    ------
    ReconcilerNegativeControlError
        When the adapter answers the canary instead of refusing it.
    PaidSessionError
        When the probe query itself fails (outage, malformed answer).
    """

    nonce_key = stable_hash(
        {"contract": NEGATIVE_CONTROL_CONTRACT, "nonce": os.urandom(32).hex()}
    )
    try:
        answer = query(nonce_key)
    except ProviderReconcileRefusal:
        return nonce_key
    raise ReconcilerNegativeControlError(
        "provider reconciler answered accepted="
        f"{getattr(answer, 'accepted', answer)!r} for a canary nonce no "
        "reservation names: an adapter that answers about keys outside its "
        "ledger is not evidence-derived (the constant-accept class) and is "
        "refused at admission"
    )


_RESERVATION_TOKEN = object()


@dataclass(frozen=True, init=False)
class Reservation:
    """Unforgeable authorization value required before a paid dispatch."""

    spend_id: str
    provider_idempotency_key: str
    ledger_path: Path
    lease_id: str
    lock_path: Path
    host_id: str
    reserved_wall_seconds: float

    def __init__(
        self,
        *,
        spend_id: str,
        provider_idempotency_key: str,
        ledger_path: Path,
        lease_id: str,
        lock_path: Path,
        host_id: str,
        reserved_wall_seconds: float,
        _token: object,
    ) -> None:
        """Refuse construction outside :meth:`PaidSessionAuthority.reserve`."""

        if _token is not _RESERVATION_TOKEN:
            raise ValueError("Reservation values are minted only by reserve()")
        object.__setattr__(self, "spend_id", spend_id)
        object.__setattr__(self, "provider_idempotency_key", provider_idempotency_key)
        object.__setattr__(self, "ledger_path", ledger_path)
        object.__setattr__(self, "lease_id", lease_id)
        object.__setattr__(self, "lock_path", lock_path)
        object.__setattr__(self, "host_id", host_id)
        object.__setattr__(self, "reserved_wall_seconds", reserved_wall_seconds)


_CURRENT_RESERVATION: ContextVar[Optional[Reservation]] = ContextVar(
    "paid_session_reservation", default=None
)


@contextmanager
def authorized_paid_dispatch(reservation: Reservation) -> Iterator[None]:
    """Bind the unforgeable reservation across exactly one paid dispatch."""

    if not isinstance(reservation, Reservation):
        raise TypeError("paid dispatch requires a Reservation minted by reserve()")
    token = _CURRENT_RESERVATION.set(reservation)
    try:
        yield
    finally:
        _CURRENT_RESERVATION.reset(token)


def current_paid_reservation() -> Reservation:
    """Return the reservation bound to the current paid dispatch or refuse."""

    reservation = _CURRENT_RESERVATION.get()
    if reservation is None:
        raise PaidSessionError("paid provider dispatch has no durable reservation")
    return reservation


def _paid_ledger(path: Path) -> AnchoredLedger:
    """Return the anchored durability substrate bound to the accounting fold.

    The fold's seed carries the ledger directory's own lineage authority, so a
    rollover row is admissible -- at mint AND on every later replay -- only
    while its preserved ``.poisoned-*`` predecessor and anchor authenticate on
    disk beside this exact ledger.  The accounting concern stays pure: it
    defines the obligation and calls the injected verifier.
    """

    lineage = preserved_lineage_verifier(
        path.parent, ledger_name=path.name, subject=_SUBJECT
    )
    return AnchoredLedger(
        path,
        chain_scope=PAID_SESSION_CHAIN_SCOPE,
        subject=_SUBJECT,
        seed=lambda: AccountingState(lineage_verifier=lineage),
        fold=lambda state, rows: state.extend(rows),
        projection=_PROJECTION,
    )


def load_paid_session_rows(path: Path) -> tuple[dict[str, Any], ...]:
    """Load and fail-closed validate one paid-session ledger.

    Parameters
    ----------
    path:
        Canonical paid-session ledger path.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Chain-verified, anchor-verified lifecycle rows.
    """

    return _paid_ledger(path).read_locked().rows


def _rfc3339_deadline(created_at: str, seconds: float) -> str:
    """Return the machine-owned reservation deadline in RFC 3339 spelling."""

    try:
        instant = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("paid-session clock must emit RFC 3339 timestamps") from exc
    return (
        (instant + timedelta(seconds=seconds))
        .astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _reservation_age_seconds(row: Mapping[str, Any], now: str) -> Optional[float]:
    """Return how long a reservation has been durably open, if computable.

    A pre-settlement crash leaves no in-process measurement, but the durable
    reservation timestamp is still MACHINE evidence that the session was open
    for at least this long.  It becomes the wall floor for crash reconciliation,
    so a provider that meters nothing cannot make a crashed session free.
    """

    try:
        opened = datetime.fromisoformat(str(row.get("created_at")).replace("Z", "+00:00"))
        observed = datetime.fromisoformat(now.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return max(0.0, (observed - opened).total_seconds())


class PaidSessionAuthority:
    """Serialized reserve/settle writer and sole paid-session money authority."""

    def __init__(
        self,
        path: Path,
        *,
        runtime_root: Path,
        clock: Callable[[], str],
    ) -> None:
        """Bind the canonical spend ledger, runtime custody root, and machine clock."""

        self.path = path
        self.runtime_root = runtime_root
        self.clock = clock
        self._ledger = _paid_ledger(path)
        self._thread_lock = threading.RLock()

    # -- reads ------------------------------------------------------------

    def snapshot(self) -> LedgerSnapshot:
        """Return one verified snapshot, decoding only rows appended since the last."""

        with self._thread_lock:
            return self._ledger.read_locked()

    def state(self) -> AccountingState:
        """Return the folded accounting state over the durable rows."""

        return self.snapshot().value

    def rows(self) -> tuple[dict[str, Any], ...]:
        """Return the verified durable rows."""

        return self.snapshot().rows

    def quota_sessions(self, campaign_id: str, stable_id: str, work_id: str, lane: str) -> int:
        """Return consummated QUOTA sessions for one exact work coordinate."""

        return (
            self.state()
            .quota_by_lane.get((campaign_id, stable_id, work_id, lane), Weight())
            .sessions
        )

    # -- writes -----------------------------------------------------------

    def _append(self, build: Callable[[LedgerSnapshot], Mapping[str, Any]]) -> dict[str, Any]:
        """Append one writer-minted transition under the durability protocol."""

        with self._thread_lock:
            linked, _snapshot = self._ledger.append(build)
        return linked

    def _stamp(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """Return one row body with its content-addressed identity.

        W3 write-side producer (spend-row mint): the shadow-rehearsal marker
        is stamped BEFORE ``row_sha256`` is derived, so the marker is inside
        every spend row's own content address and the chain link over it --
        stamping any later would mint a row whose declared digest the chain
        chokepoint refuses.
        """

        return stamp_row_digest(stamp_shadow_rehearsal(dict(body)))

    def initialize_legacy_prefix(
        self,
        legacy_operational_path: Path,
        *,
        legacy_settlement_schemas: Sequence[str] = (),
    ) -> None:
        """Seed an empty spend ledger with the content-addressed legacy prefix.

        Parameters
        ----------
        legacy_operational_path:
            The pre-cutover operational ledger whose bytes this row pins.
        legacy_settlement_schemas:
            The pre-machine-floor settlement schemas this campaign's operational
            prefix actually contains.  This is DISCLOSURE about that separately
            pinned prefix and licenses nothing on this ledger: every row appended
            here follows the cutover row, so a legacy settlement schema refuses
            whatever is declared (PSLRE5 acct F2 -- the declaration was mintable by
            the same producer as the rows it would have licensed).  Production
            declares nothing.
        """

        declared = sorted(set(legacy_settlement_schemas))
        unknown = [schema for schema in declared if schema not in LEGACY_SETTLEMENT_SCHEMAS]
        if unknown:
            raise ValueError(
                "an accounting genesis row may only declare legacy settlement schemas: "
                + ", ".join(unknown)
            )
        payload = (
            legacy_operational_path.read_bytes() if legacy_operational_path.is_file() else b""
        )

        def build(snapshot: LedgerSnapshot) -> Mapping[str, Any]:
            """Mint the cutover genesis row, or refuse a non-empty ledger."""

            if snapshot.rows:
                raise _AlreadyInitialized()
            return self._stamp(
                {
                    "schema_version": PAID_SESSION_GENESIS_SCHEMA,
                    "event": "accounting_genesis",
                    "legacy_operational_path": str(legacy_operational_path),
                    "legacy_operational_sha256": stable_hash({"bytes_hex": payload.hex()}),
                    "legacy_operational_bytes": len(payload),
                    "legacy_settlement_schemas": declared,
                    "created_at": self.clock(),
                }
            )

        try:
            self._append(build)
        except _AlreadyInitialized:
            return

    def initialize_rollover(
        self,
        *,
        carried_rows: Sequence[Mapping[str, Any]],
        poisoned_ledger_name: str,
        poisoned_ledger_sha256: str,
        poisoned_ledger_bytes: int,
    ) -> dict[str, Any]:
        """Mint the rollover genesis for a fold-refused predecessor ledger.

        The append is fold-first, so ELIGIBILITY is decided by the accounting
        fold itself (:meth:`_StateBuilder._apply_rollover`), not by this
        writer: carried rows that chain-verify AND still refuse the fold are
        the only admissible input, a healthy predecessor refuses typed, and a
        refused rollover mints nothing.  See the four-part waiver doctrine.

        Parameters
        ----------
        carried_rows:
            The poisoned ledger's chain-verified rows, verbatim, in order.
        poisoned_ledger_name, poisoned_ledger_sha256, poisoned_ledger_bytes:
            Identity of the immutably preserved poisoned sibling file.

        Returns
        -------
        dict[str, Any]
            The linked durable rollover row.
        """

        materialized = [dict(member) for member in carried_rows]
        if not materialized:
            raise ValueError("paid-session rollover requires the carried poisoned rows")
        chain = extend_chain_state(
            materialized, scope=PAID_SESSION_CHAIN_SCOPE, require_links=True
        )
        row = self._stamp(
            {
                "schema_version": PAID_SESSION_ROLLOVER_SCHEMA,
                "event": "accounting_rollover",
                "poisoned_ledger_name": str(poisoned_ledger_name),
                "poisoned_ledger_sha256": str(poisoned_ledger_sha256),
                "poisoned_ledger_bytes": int(poisoned_ledger_bytes),
                "poisoned_chain_head": chain.head_sha256,
                "poisoned_row_count": len(materialized),
                # The anchor pin the fold re-derives from the embedded chain
                # and the on-disk preserved anchor must both confirm.
                "poisoned_anchor": {
                    "ledger_identity": chain.heads[0],
                    "rows": len(materialized),
                    "bytes": int(poisoned_ledger_bytes),
                    "chain_head": chain.head_sha256,
                },
                "carried_rows": materialized,
                "created_at": self.clock(),
            }
        )

        def build(snapshot: LedgerSnapshot) -> Mapping[str, Any]:
            """Refuse to reauthor a ledger that already holds any row."""

            if snapshot.rows:
                raise ValueError(
                    "paid-session rollover requires an empty successor ledger: "
                    f"{self.path} already holds {len(snapshot.rows)} rows"
                )
            return row

        return self._append(build)

    def reserve(self, *, scope: Mapping[str, Any], policy: BudgetPolicy) -> Reservation:
        """Authorize and durably reserve one paid session under both instruments.

        Parameters
        ----------
        scope:
            The reservation's own accounting coordinates and disclosure fields.
        policy:
            Machine-owned QUOTA and RUNAWAY_BREAKER limits.

        Returns
        -------
        Reservation
            The unforgeable authorization value one dispatch requires.

        Raises
        ------
        PaidSessionBudgetExceeded
            Before any byte is written, so a refused buy mints no money row.
        """

        campaign_id = scope.get("campaign_id")
        generation_id = scope.get("generation_id")
        lane = scope.get("lane")
        dispatch_kind = scope.get("dispatch_kind")
        if not all(
            isinstance(value, str) and value
            for value in (campaign_id, generation_id, lane, dispatch_kind)
        ):
            raise ValueError("paid-session scope requires machine-owned string coordinates")
        if lane not in LANES:
            raise ValueError("paid-session lane is outside the closed set")
        # The durable attempt identity: None heads a new attempt; a spend id
        # joins that attempt's group as a later guard session.  The fold owns
        # the join rules (same lane, same axis, settled members, real head).
        attempt_group = scope.get("attempt_group")
        if attempt_group is not None and (
            not isinstance(attempt_group, str) or not attempt_group
        ):
            raise ValueError("paid-session attempt_group must be None or a spend id")
        members = work_generations(scope.get("work_generations"))
        host_id = str(scope.get("machine_id") or "unknown-host")
        boot_id = current_boot_id()
        driver_token = process_start_token(_current_pid())
        if driver_token is None:
            raise PaidSessionError("cannot bind paid-session reservation to driver start token")
        minted: dict[str, Any] = {}

        def build(snapshot: LedgerSnapshot) -> Mapping[str, Any]:
            """Authorize inside the lock, then mint the self-keyed money row."""

            state: AccountingState = snapshot.value
            authorize(
                state,
                campaign_id=str(campaign_id),
                generation_id=str(generation_id),
                lane=str(lane),
                members=members,
                policy=policy,
            )
            axis = {
                "campaign_id": str(campaign_id),
                "generation_id": str(generation_id),
                "work_generations": [
                    {"stable_id": stable_id, "work_id": work_id}
                    for stable_id, work_id in members
                ],
            }
            created_at = self.clock()
            identity = {
                "contract": "paid-session-spend-identity-v2",
                "axis": axis,
                "lane": str(lane),
                "dispatch_kind": str(dispatch_kind),
                "occurrence": 1 + len(state.reservations),
                # The ledger's own verified head, the host, and the boot make the
                # key unique across independent ledgers at the same extent: two
                # hosts can no longer mint the identical provider key.
                "ledger_head": snapshot.chain_state.head_sha256,
                "host_id": host_id,
                "boot_id": boot_id,
            }
            spend_id = f"paid-spend-{stable_hash(identity).removeprefix('sha256:')[:24]}"
            provider_key = stable_hash(
                {"contract": "paid-session-provider-idempotency-v2", "identity": identity}
            )
            lease_id = f"paid-lease-{spend_id.removeprefix('paid-spend-')}"
            lock_path = self.runtime_root / "locks" / f"{lease_id}.lock"
            minted.update(
                spend_id=spend_id,
                provider_key=provider_key,
                lease_id=lease_id,
                lock_path=lock_path,
            )
            return self._stamp(
                {
                    "schema_version": PAID_SESSION_RESERVATION_SCHEMA,
                    "event": "session_reserved",
                    "spend_id": spend_id,
                    "provider_idempotency_key": provider_key,
                    "lane": str(lane),
                    "dispatch_kind": str(dispatch_kind),
                    "attempt_group": attempt_group,
                    "axis": axis,
                    "reserved": {
                        "wall_seconds": policy.reservation_wall_seconds,
                        "sessions": 1,
                    },
                    "reservation_evidence": "writer-fsync",
                    "gate_kind": scope.get("gate_kind"),
                    "checker_identity": scope.get("checker_identity"),
                    "run_id": scope.get("run_id"),
                    "machine_id": scope.get("machine_id"),
                    "created_at": created_at,
                    "custody": {
                        "lease_id": lease_id,
                        "boot_id": boot_id,
                        "host_id": host_id,
                        "lock_path": str(lock_path),
                        "driver_pid": _current_pid(),
                        "driver_start_token": driver_token,
                        "leader_pid": None,
                        "leader_start_token": None,
                        "leader_pgid": None,
                        "deadline_at": _rfc3339_deadline(
                            created_at, policy.reservation_wall_seconds
                        ),
                    },
                }
            )

        self._append(build)
        return Reservation(
            spend_id=str(minted["spend_id"]),
            provider_idempotency_key=str(minted["provider_key"]),
            ledger_path=self.path,
            lease_id=str(minted["lease_id"]),
            lock_path=Path(minted["lock_path"]),
            host_id=host_id,
            reserved_wall_seconds=policy.reservation_wall_seconds,
            _token=_RESERVATION_TOKEN,
        )

    def settle(
        self,
        reservation: Reservation,
        *,
        provider_query: Callable[[str], ProviderReconciliation],
        custody_outcome: str,
        machine_wall_seconds: Optional[float] = None,
        machine_wall_source: str = "writer-monotonic",
        local_outcome: str = "unknown",
    ) -> None:
        """Query the provider and append its sole authoritative finalization.

        Parameters
        ----------
        reservation:
            The unforgeable value :meth:`reserve` minted.
        provider_query:
            Driver-owned adapter answering for one idempotency key.
        custody_outcome:
            Machine-observed custody disposition, disclosed on the row.
        machine_wall_seconds:
            Writer-measured elapsed seconds, the floor a provider answer can
            raise but never delete.  ``None`` declares no local observation.
        local_outcome:
            Closed-set disclosure of what this host observed locally.  It is
            accounting-inert: no projection reads it, so a local flag can
            neither mint nor dodge a charge.
        """

        if reservation.ledger_path != self.path:
            raise ValueError("reservation belongs to a different paid-session authority")
        receipt = provider_query(reservation.provider_idempotency_key)
        self._settle_provider_receipt(
            reservation,
            receipt=receipt,
            custody_outcome=custody_outcome,
            machine_wall_seconds=machine_wall_seconds,
            machine_wall_source=machine_wall_source,
            local_outcome=local_outcome,
        )

    def _settle_provider_receipt(
        self,
        reservation: Reservation,
        *,
        receipt: ProviderReconciliation,
        custody_outcome: str,
        machine_wall_seconds: Optional[float],
        machine_wall_source: str,
        local_outcome: str,
    ) -> None:
        """Append one already-queried provider receipt under the writer lock."""

        if not isinstance(receipt, ProviderReconciliation):
            raise TypeError("provider query must return ProviderReconciliation")
        if local_outcome not in LOCAL_OUTCOMES:
            raise ValueError("paid-session local outcome is outside the closed set")
        if machine_wall_source not in MACHINE_WALL_SOURCES:
            raise ValueError("paid-session machine wall source is outside the closed set")
        floor = (
            None
            if machine_wall_seconds is None
            else finite_nonnegative(machine_wall_seconds, field="machine.wall_seconds")
        )
        observed = charged_wall_seconds(
            reserved_wall_seconds=reservation.reserved_wall_seconds,
            machine_wall_seconds=floor,
            provider_wall_seconds=receipt.observed_wall_seconds,
        )
        outcome = custody_outcome if receipt.accepted else "reconciled-provider-not-accepted"

        def build(snapshot: LedgerSnapshot) -> Mapping[str, Any]:
            """Mint the one settlement this reservation is allowed."""

            state: AccountingState = snapshot.value
            if reservation.spend_id not in state.reservations:
                raise ValueError("reservation is absent from its paid-session authority")
            if reservation.spend_id in state.settlements:
                raise ValueError(
                    f"paid-session reservation {reservation.spend_id} is already settled"
                )
            return self._stamp(
                {
                    "schema_version": PAID_SESSION_SETTLEMENT_SCHEMA,
                    "event": "session_settled",
                    "spend_id": reservation.spend_id,
                    "observed": {"wall_seconds": observed},
                    "evidence": "provider-query",
                    "provider": {
                        "accepted": receipt.accepted,
                        "attested_wall_seconds": receipt.observed_wall_seconds,
                        "receipt_sha256": receipt.receipt_sha256,
                        # D-1 (v5): the row itself names WHICH reconciler
                        # answered; None is the disclosed in-process spelling.
                        "adapter_identity": receipt.adapter_identity,
                    },
                    "machine": {
                        "wall_seconds": floor,
                        "source": (
                            "writer-unobserved" if floor is None else machine_wall_source
                        ),
                    },
                    "custody_outcome": outcome,
                    "local_outcome": local_outcome,
                    "created_at": self.clock(),
                }
            )

        self._append(build)

    def bind_custody(
        self,
        reservation: Reservation,
        *,
        leader_pid: int,
        leader_pgid: int,
        leader_start_token: str,
        boot_id: str,
        host_id: str,
        lock_device: int,
        lock_inode: int,
    ) -> None:
        """Durably bind the spawned paid-subtree identity before provider exposure.

        Parameters
        ----------
        reservation:
            The dispatch's own reservation.
        leader_pid, leader_pgid, leader_start_token:
            Kernel identity of the paid subtree's session leader.
        boot_id, host_id:
            Machine identity, so a reboot is never mistaken for a live process.
        lock_device, lock_inode:
            Identity of the inode the subtree's ``flock`` is held on, so an
            unlinked-and-recreated lock pathname cannot make the liveness oracle
            report a live subtree dead.
        """

        if reservation.ledger_path != self.path:
            raise ValueError("reservation belongs to a different paid-session authority")
        row = self._stamp(
            {
                "schema_version": PAID_SESSION_CUSTODY_SCHEMA,
                "event": "session_custody_bound",
                "spend_id": reservation.spend_id,
                "lease_id": reservation.lease_id,
                "boot_id": boot_id,
                "host_id": host_id,
                "lock_path": str(reservation.lock_path),
                "lock_device": int(lock_device),
                "lock_inode": int(lock_inode),
                "leader_pid": leader_pid,
                "leader_start_token": leader_start_token,
                "leader_pgid": leader_pgid,
                "created_at": self.clock(),
            }
        )

        def build(snapshot: LedgerSnapshot) -> Mapping[str, Any]:
            """Refuse a binding whose reservation is absent, spent, or already bound.

            These are the same refusals the accounting fold makes on every
            replay (``_apply_binding``), asserted BEFORE the row is minted: a
            reservation is one paid provider session with exactly ONE custody
            epoch, so a second guard dispatch must buy a fresh reservation
            rather than rebind this one.  The 2026-08-20 step-200 leg rebound a
            released reservation for its second guard session; the durable
            duplicate row then made every later money read refuse.
            """

            state: AccountingState = snapshot.value
            if reservation.spend_id not in state.reservations:
                raise ValueError("custody binding has no reservation")
            if reservation.spend_id in state.custody_bindings:
                raise ValueError(
                    f"paid-session reservation {reservation.spend_id} has duplicate custody"
                )
            if reservation.spend_id in state.settlements:
                raise ValueError(
                    f"custody binding follows settlement for {reservation.spend_id}"
                )
            return row

        self._append(build)

    def release_custody(self, reservation: Reservation, *, outcome: str) -> None:
        """Durably close custody only after a machine-verified subtree exit."""

        if reservation.ledger_path != self.path:
            raise ValueError("reservation belongs to a different paid-session authority")
        self._release_custody(
            spend_id=reservation.spend_id,
            lease_id=reservation.lease_id,
            outcome=outcome,
            operator=None,
            reason=None,
        )

    def release_reconciled_custody(self, lease: PaidCustodyLease, *, outcome: str) -> None:
        """Durably close one restart-reconciled bound custody lease."""

        self._release_custody(
            spend_id=lease.spend_id,
            lease_id=lease.lease_id,
            outcome=outcome,
            operator=None,
            reason=None,
        )

    def drain_custody(self, lease: PaidCustodyLease, *, operator: str, reason: str) -> None:
        """Append an explicit operator recovery after an orphan is resolved."""

        if not operator.strip() or not reason.strip():
            raise ValueError("operator custody drain requires operator and reason")
        self._release_custody(
            spend_id=lease.spend_id,
            lease_id=lease.lease_id,
            outcome="operator-drain",
            operator=operator.strip(),
            reason=reason.strip(),
        )

    def _release_custody(
        self,
        *,
        spend_id: str,
        lease_id: str,
        outcome: str,
        operator: Optional[str],
        reason: Optional[str],
    ) -> None:
        """Append the common machine-verified or operator-drained release."""

        if outcome not in CUSTODY_RELEASE_OUTCOMES:
            raise ValueError("custody release outcome is outside the closed set")
        row = self._stamp(
            {
                "schema_version": PAID_SESSION_CUSTODY_RELEASE_SCHEMA,
                "event": "session_custody_released",
                "spend_id": spend_id,
                "lease_id": lease_id,
                "outcome": outcome,
                "operator": operator,
                "reason": reason,
                "created_at": self.clock(),
            }
        )
        self._append(lambda _snapshot: row)

    # -- crash reconciliation ---------------------------------------------

    def reconcile(self, query: Callable[[str], ProviderReconciliation]) -> tuple[str, ...]:
        """Settle every outstanding reservation from a provider-key query.

        Parameters
        ----------
        query:
            Driver-owned provider status adapter.

        Returns
        -------
        tuple[str, ...]
            The spend ids this call settled terminally.

        Raises
        ------
        PaidSessionError
            Typed, always: an adapter that answers unrepresentably must not
            escape as an untyped error that no leg-start caller can classify.
        """

        reconciled: list[str] = []
        state = self.state()
        now = self.clock()
        for spend_id, row in state.reservations.items():
            if spend_id in state.settlements:
                continue
            try:
                receipt = query(str(row["provider_idempotency_key"]))
            except PaidSessionError:
                raise
            except Exception as exc:  # noqa: BLE001 -- one typed vocabulary at the boundary
                raise PaidSessionError(
                    f"provider reconciliation for {spend_id} did not answer: {exc}"
                ) from exc
            if not isinstance(receipt, ProviderReconciliation):
                raise PaidSessionError("provider query must return ProviderReconciliation")
            reserved_wall = float(row["reserved"]["wall_seconds"])
            reservation = Reservation(
                spend_id=spend_id,
                provider_idempotency_key=str(row["provider_idempotency_key"]),
                ledger_path=self.path,
                lease_id=str(row["custody"]["lease_id"]),
                lock_path=Path(str(row["custody"]["lock_path"])),
                host_id=str(row["custody"]["host_id"]),
                reserved_wall_seconds=reserved_wall,
                _token=_RESERVATION_TOKEN,
            )
            self._settle_provider_receipt(
                reservation,
                receipt=receipt,
                custody_outcome=(
                    "reconciled-provider-accepted"
                    if receipt.accepted
                    else "reconciled-provider-not-accepted"
                ),
                machine_wall_seconds=_reservation_age_seconds(row, now),
                machine_wall_source="writer-reservation-age",
                local_outcome="unknown",
            )
            reconciled.append(spend_id)
        return tuple(reconciled)

    def settle_reaped(
        self,
        spend_id: str,
        *,
        query: Callable[[str], ProviderReconciliation],
        local_outcome: str,
    ) -> None:
        """Force-settle ONE reservation whose custodian the driver proved dead.

        The custody deadline reaper's settlement arm: everything except the
        dead-process evidence is re-derived from durable rows exactly as
        :meth:`reconcile` does -- the provider answers for the idempotency key
        in both directions, the machine wall floor is the durable reservation
        age, and the raw ``local_outcome`` is re-weighed by the fold on every
        replay.  The caller supplies only WHICH reservation to settle and the
        closed machine classification; it cannot alter the charge.

        Parameters
        ----------
        spend_id:
            The bound-dead reservation's durable spend id.
        query:
            Driver-owned provider status adapter.
        local_outcome:
            Closed-set machine disposition disclosed on the settlement row.

        Raises
        ------
        PaidSessionError
            Typed, when the spend id is unknown, already settled, or the
            provider adapter cannot answer -- the caller leaves the lease
            disclosed for the next boundary instead of wedging.
        """

        state = self.state()
        row = state.reservations.get(spend_id)
        if row is None:
            raise PaidSessionError(f"reaped settlement has no reservation: {spend_id}")
        if spend_id in state.settlements:
            raise PaidSessionError(
                f"paid-session reservation {spend_id} is already settled"
            )
        now = self.clock()
        try:
            receipt = query(str(row["provider_idempotency_key"]))
        except PaidSessionError:
            raise
        except Exception as exc:  # noqa: BLE001 -- one typed vocabulary at the boundary
            raise PaidSessionError(
                f"provider reconciliation for {spend_id} did not answer: {exc}"
            ) from exc
        if not isinstance(receipt, ProviderReconciliation):
            raise PaidSessionError("provider query must return ProviderReconciliation")
        reservation = Reservation(
            spend_id=spend_id,
            provider_idempotency_key=str(row["provider_idempotency_key"]),
            ledger_path=self.path,
            lease_id=str(row["custody"]["lease_id"]),
            lock_path=Path(str(row["custody"]["lock_path"])),
            host_id=str(row["custody"]["host_id"]),
            reserved_wall_seconds=float(row["reserved"]["wall_seconds"]),
            _token=_RESERVATION_TOKEN,
        )
        self._settle_provider_receipt(
            reservation,
            receipt=receipt,
            custody_outcome=(
                "reconciled-provider-accepted"
                if receipt.accepted
                else "reconciled-provider-not-accepted"
            ),
            machine_wall_seconds=_reservation_age_seconds(row, now),
            machine_wall_source="writer-reservation-age",
            local_outcome=local_outcome,
        )

    def outstanding_spend_ids(self) -> tuple[str, ...]:
        """Return durable reservations that still require provider reconciliation."""

        state = self.state()
        return tuple(
            spend_id for spend_id in state.reservations if spend_id not in state.settlements
        )

    def outstanding_custody_leases(self) -> tuple[PaidCustodyLease, ...]:
        """Return every unreleased binding and every unsettled pre-bind reservation."""

        state = self.state()
        leases: list[PaidCustodyLease] = []
        for spend_id, row in state.reservations.items():
            if spend_id in state.custody_releases:
                continue
            binding = state.custody_bindings.get(spend_id)
            if binding is None and spend_id in state.settlements:
                continue
            custody = row["custody"]
            leases.append(
                PaidCustodyLease(
                    lease_id=str(custody["lease_id"]),
                    spend_id=spend_id,
                    boot_id=str(custody["boot_id"]),
                    host_id=str(custody["host_id"]),
                    lock_path=Path(str(custody["lock_path"])),
                    leader_pid=(int(binding["leader_pid"]) if binding is not None else None),
                    leader_start_token=(
                        str(binding["leader_start_token"]) if binding is not None else None
                    ),
                    leader_pgid=(int(binding["leader_pgid"]) if binding is not None else None),
                    deadline_at=str(custody["deadline_at"]),
                    reserved_wall_seconds=float(row["reserved"]["wall_seconds"]),
                    lock_device=(
                        int(binding["lock_device"])
                        if binding is not None and "lock_device" in binding
                        else None
                    ),
                    lock_inode=(
                        int(binding["lock_inode"])
                        if binding is not None and "lock_inode" in binding
                        else None
                    ),
                )
            )
        return tuple(leases)


class _AlreadyInitialized(Exception):
    """Internal signal that the cutover genesis row already exists."""


def _current_pid() -> int:
    """Return this writer's pid."""

    return os.getpid()
