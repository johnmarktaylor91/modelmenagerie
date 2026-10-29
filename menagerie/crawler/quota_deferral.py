"""QUOTA-storm deferral valve: transient rows instead of permanent exhaustion terminals (W1).

A sustained provider storm burns a work generation's author-session allowance on
infrastructure noise. 200a terminalized that as permanent
``failed:author`` / ``effort-exhausted:author-sessions`` -- pooling restorable
storm damage with genuine effort exhaustion (R7: 28 members in one leg; 21 of the
63 exhaustion heads carried the ``used < max`` arithmetic contradiction). This
module is the durable valve for the storm class, mirroring the sibling
pre-admission valves (:mod:`menagerie.crawler.checker_lane`,
:mod:`menagerie.crawler.host_capacity`):

* **A storm-classified exhaustion never mints a terminal.** The member is
  WITHHELD: recorded here as ``machine-deferred:quota-storm``, stashed for the
  completion partition exactly like the sibling valve rows, and released for a
  fresh generation through the W14 machine-class release engine
  (:func:`menagerie.crawler.infra_release.derive_machine_class_release`,
  class ``quota-deferral``) or a human grant.
* **The valve is PD-3 bounded.** At most :data:`QUOTA_DEFERRAL_MAX_GRANTS`
  deferral rows per ``(stable_id, work_id)``; the next storm mints ONE
  ``review-held:storm-recidivist`` park row instead -- a hold only a human can
  release (the class engine refuses recidivists), so a member storming forever
  cannot draw unbounded machine retries.
* **Storm-ness is settlement-derived, never claimed.** The ONLY path here is the
  driver's own exhaustion arithmetic whose dimension the closed
  :func:`menagerie.crawler.author_spend.author_sessions_exhaustion_dimension`
  classifier read as ``provider-storm`` from settlement evidence
  (attempt-refunding machine outcomes); producer text participates nowhere.

``effort-exhausted:author-sessions`` is thereby RESERVED for the true attempt
allowance (``used == max`` of genuinely consumed sessions); the census invariant
(:func:`quota_exhaustion_census`) trips in both directions when either side of
that reservation is violated.
"""

from __future__ import annotations

import fcntl
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping, Optional, Sequence

from menagerie.crawler.identity import (
    canonical_json_bytes,
    durable_fsync,
    foreign_content_address_fields,
    fsync_directory,
    row_content_address,
    stamp_row_digest,
)
from menagerie.crawler.models import JsonObject
from menagerie.crawler.recordio import read_with_tail_recovery

__all__ = [
    "QUOTA_DEFERRAL_DISPOSITION",
    "QUOTA_DEFERRAL_MAX_GRANTS",
    "QUOTA_DEFERRAL_RELATIVE_PATH",
    "QUOTA_DEFERRAL_SCHEMA_VERSION",
    "QuotaDeferralError",
    "QuotaExhaustionCensus",
    "STORM_RECIDIVIST_DISPOSITION",
    "load_quota_deferral_rows",
    "mint_quota_deferral",
    "quota_deferral_path",
    "quota_exhaustion_census",
    "storm_recidivist_hold",
    "validate_quota_deferral_row",
]


class QuotaDeferralError(ValueError):
    """A quota-deferral row violates its closed contract."""


QUOTA_DEFERRAL_SCHEMA_VERSION = "menagerie.crawler.quota-deferral.v1"
QUOTA_DEFERRAL_RELATIVE_PATH = Path("intake-extensions") / "quota-deferrals.jsonl"

#: The transient valve disposition: withheld this run, machine-releasable for a
#: fresh generation under the W14 ``quota-deferral`` class.
QUOTA_DEFERRAL_DISPOSITION = "machine-deferred:quota-storm"

#: The PD-3 recidivist park: the bound is spent, only a human releases. The class
#: engine refuses any generation carrying one of these.
STORM_RECIDIVIST_DISPOSITION = "review-held:storm-recidivist"

#: PD-3's K: machine deferral grants per (member, generation). Chosen with the
#: tri-lab PD-3 ruling (K=3): three storms across distinct legs is already a
#: recidivism pattern no healthy 200a member exhibited, and the 4th trip must
#: surface to a human rather than draw a 4th machine retry.
QUOTA_DEFERRAL_MAX_GRANTS = 3

_DISPOSITIONS = frozenset({QUOTA_DEFERRAL_DISPOSITION, STORM_RECIDIVIST_DISPOSITION})


def quota_deferral_path(records_root: Path) -> Path:
    """Return the canonical quota-deferral ledger path under one records root."""

    return records_root / QUOTA_DEFERRAL_RELATIVE_PATH


def _build_row(
    *,
    stable_id: str,
    work_id: str,
    campaign_id: str,
    run_id: str,
    machine_id: str,
    created_at: str,
    disposition: str,
    deferral_ordinal: int,
    dimension: str,
    evidence: Mapping[str, Any],
) -> JsonObject:
    """Build one valve row; the mint chokepoint validates before appending."""

    row: JsonObject = {
        "schema_version": QUOTA_DEFERRAL_SCHEMA_VERSION,
        "stable_id": stable_id,
        "work_id": work_id,
        "campaign_id": campaign_id,
        "run_id": run_id,
        "machine_id": machine_id,
        "created_at": created_at,
        "disposition": disposition,
        "deferral_ordinal": deferral_ordinal,
        "quota_deferral": {
            "dimension": dimension,
            "evidence": dict(evidence),
        },
        "recheck_hint": (
            "quota storm: the generation's session allowance was consumed by "
            "machine-classified infrastructure outcomes; the member is withheld, "
            "not failed -- releasable for a fresh generation via the W14 "
            "quota-deferral class within the PD-3 bound"
            if disposition == QUOTA_DEFERRAL_DISPOSITION
            else "storm recidivist: the PD-3 deferral bound is spent; only a "
            "human review releases this member"
        ),
    }
    return stamp_row_digest(row)


def validate_quota_deferral_row(row: Mapping[str, Any]) -> JsonObject:
    """Validate one quota-deferral row and return its canonical copy.

    Raises
    ------
    QuotaDeferralError
        If a required field is missing, mistyped, outside a closed vocabulary,
        or the content address disagrees.
    """

    required = (
        "schema_version",
        "stable_id",
        "work_id",
        "campaign_id",
        "run_id",
        "machine_id",
        "created_at",
        "disposition",
        "deferral_ordinal",
        "quota_deferral",
        "recheck_hint",
        "row_sha256",
    )
    missing = [field for field in required if field not in row]
    if missing:
        raise QuotaDeferralError(f"quota-deferral row is missing {', '.join(sorted(missing))}")
    if row["schema_version"] != QUOTA_DEFERRAL_SCHEMA_VERSION:
        raise QuotaDeferralError(
            f"quota-deferral row declares unknown schema {row['schema_version']!r}"
        )
    if row["disposition"] not in _DISPOSITIONS:
        raise QuotaDeferralError(
            f"quota-deferral row declares unknown disposition {row['disposition']!r}"
        )
    for field in ("stable_id", "work_id", "campaign_id", "run_id", "machine_id", "created_at"):
        if not isinstance(row[field], str) or not row[field] or row[field] == str(None):
            raise QuotaDeferralError(f"quota-deferral row has an empty {field}")
    ordinal = row["deferral_ordinal"]
    if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 1:
        raise QuotaDeferralError(f"quota-deferral ordinal must be a positive int, got {ordinal!r}")
    if row["disposition"] == QUOTA_DEFERRAL_DISPOSITION and ordinal > QUOTA_DEFERRAL_MAX_GRANTS:
        raise QuotaDeferralError(
            f"quota-deferral ordinal {ordinal} exceeds the PD-3 bound "
            f"{QUOTA_DEFERRAL_MAX_GRANTS}; the recidivist park is the only row past it"
        )
    if (
        row["disposition"] == STORM_RECIDIVIST_DISPOSITION
        and ordinal != QUOTA_DEFERRAL_MAX_GRANTS + 1
    ):
        raise QuotaDeferralError(
            "a storm-recidivist park is exactly the row past the PD-3 bound "
            f"(ordinal {QUOTA_DEFERRAL_MAX_GRANTS + 1}), got {ordinal}"
        )
    payload = row["quota_deferral"]
    if not isinstance(payload, Mapping) or payload.get("dimension") != "provider-storm":
        raise QuotaDeferralError(
            "quota-deferral row must carry the settlement-derived provider-storm dimension"
        )
    if not isinstance(payload.get("evidence"), Mapping):
        raise QuotaDeferralError("quota-deferral row must carry its evidence mapping")
    validated = dict(row)
    foreign = foreign_content_address_fields(validated, own="row_sha256")
    if foreign:
        raise QuotaDeferralError(
            f"quota-deferral row carries a foreign content-address field {', '.join(foreign)}"
        )
    digest = validated.pop("row_sha256")
    if row_content_address(validated) != digest:
        raise QuotaDeferralError("quota-deferral row digest does not match its payload")
    validated["row_sha256"] = digest
    return validated


def _decode_lines(payload: bytes, path: Path) -> list[JsonObject]:
    """Decode and validate every complete ledger line."""

    rows: list[JsonObject] = []
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise QuotaDeferralError(f"{path} is not a UTF-8 quota-deferral ledger: {exc}") from exc
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError as exc:
            raise QuotaDeferralError(
                f"{path}:{index} is not a decodable quota-deferral row: {exc}"
            ) from exc
        rows.append(validate_quota_deferral_row(candidate))
    return rows


def _ledger_bytes(path: Path, handle: Optional[BinaryIO] = None) -> bytes:
    """Read the whole ledger, recovering only a torn final line with evidence."""

    return read_with_tail_recovery(
        path,
        prefix_check=lambda retained: _decode_lines(retained.read_bytes(), path),
        locked_handle=handle,
    )


def load_quota_deferral_rows(path: Path) -> tuple[JsonObject, ...]:
    """Return every validated valve row in ledger order (missing file = empty)."""

    if not path.exists():
        return ()
    return tuple(_decode_lines(_ledger_bytes(path), path))


def mint_quota_deferral(
    path: Path,
    *,
    stable_id: str,
    work_id: str,
    campaign_id: str,
    run_id: str,
    machine_id: str,
    created_at: str,
    evidence: Mapping[str, Any],
) -> JsonObject:
    """Mint one storm deferral (or the PD-3 recidivist park) durably.

    The ONE valve chokepoint, called only from the driver's storm-classified
    exhaustion arm. Under the ledger lock it counts the pair's existing rows:

    * the SAME run re-deriving the same pair returns the persisted row (resume
      idempotency, exactly the sibling valves' rule);
    * an existing recidivist park is returned unchanged (the hold stands);
    * rows 1..K append as ``machine-deferred:quota-storm``;
    * row K+1 appends as ``review-held:storm-recidivist`` -- LOUD, durable, and
      the last row this pair can ever mint.

    Parameters
    ----------
    path:
        Destination valve ledger.
    evidence:
        Settlement-derived storm evidence (counts only, no text): the caller
        passes what the classifier saw.

    Returns
    -------
    dict[str, Any]
        The persisted, validated row (existing or fresh); callers branch on its
        ``disposition``.
    """

    if not campaign_id or campaign_id == str(None):
        raise QuotaDeferralError(
            "campaign_id must name a real campaign, never a stringified absence"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        existing = _decode_lines(_ledger_bytes(path, handle), path)
        pair = [row for row in existing if (row["stable_id"], row["work_id"]) == (stable_id, work_id)]
        parks = [row for row in pair if row["disposition"] == STORM_RECIDIVIST_DISPOSITION]
        if parks:
            return parks[0]
        same_run = [row for row in pair if row["run_id"] == run_id]
        if same_run:
            return same_run[-1]
        ordinal = len(pair) + 1
        disposition = (
            QUOTA_DEFERRAL_DISPOSITION
            if ordinal <= QUOTA_DEFERRAL_MAX_GRANTS
            else STORM_RECIDIVIST_DISPOSITION
        )
        row = _build_row(
            stable_id=stable_id,
            work_id=work_id,
            campaign_id=campaign_id,
            run_id=run_id,
            machine_id=machine_id,
            created_at=created_at,
            disposition=disposition,
            deferral_ordinal=ordinal,
            dimension="provider-storm",
            evidence=evidence,
        )
        validated = validate_quota_deferral_row(row)
        handle.seek(0, os.SEEK_END)
        handle.write(canonical_json_bytes(validated) + b"\n")
        handle.flush()
        durable_fsync(handle.fileno())
    fsync_directory(path.parent)
    return validated


def storm_recidivist_hold(
    rows: Iterable[Mapping[str, Any]], stable_id: str, work_id: str
) -> bool:
    """Return whether a recidivist park holds this exact generation.

    Admission consults this before buying the pair another session: a held
    member is withheld, never re-admitted, until a human releases it.
    """

    for row in rows:
        if (
            row.get("stable_id") == stable_id
            and row.get("work_id") == work_id
            and row.get("disposition") == STORM_RECIDIVIST_DISPOSITION
        ):
            return True
    return False


@dataclass(frozen=True)
class QuotaExhaustionCensus:
    """Both directions of the W1 reservation invariant, replay-derived.

    Attributes
    ----------
    exhausted_heads:
        Every ``effort-exhausted:author-sessions`` head as ``(stable_id,
        work_id, used, allowance)``.
    understated_heads:
        Direction 1 CONTRADICTIONS: exhaustion heads whose replay-derived
        ``used < allowance`` (the m697 class) -- the reason asserts an
        allowance that arithmetic refutes.
    storm_valves:
        Every ``machine-deferred:quota-storm`` valve pair.
    unsupported_valves:
        Direction 2 CONTRADICTIONS: valve rows whose generation's settlement
        evidence does NOT classify ``provider-storm`` under the closed
        machine classifier -- a valve row no storm supports.
    """

    exhausted_heads: tuple[tuple[str, str, int, int], ...]
    understated_heads: tuple[tuple[str, str, int, int], ...]
    storm_valves: tuple[tuple[str, str], ...]
    unsupported_valves: tuple[tuple[str, str], ...]

    @property
    def clean(self) -> bool:
        """Return whether both directions of the invariant hold."""

        return not self.understated_heads and not self.unsupported_valves


def quota_exhaustion_census(
    model_rows: Sequence[Mapping[str, Any]],
    valve_rows: Sequence[Mapping[str, Any]],
    artifact_events: Sequence[Mapping[str, Any]],
    operational_events: Sequence[Mapping[str, Any]],
    *,
    granted_attempts: Optional[Mapping[tuple[str, str], int]] = None,
) -> QuotaExhaustionCensus:
    """Recompute the two-direction exhaustion reservation invariant.

    Direction 1: no ``effort-exhausted:author-sessions`` head with
    ``used < allowance`` -- the reason is RESERVED for the spent attempt
    allowance, so an understated head is a recorded false statement (the 200a
    m697 class: 21 of 63 heads).

    Direction 2: no ``machine-deferred:quota-storm`` valve row whose
    generation's settlement evidence fails the closed ``provider-storm``
    majority -- the valve is reserved for machine-witnessed storms.

    Everything is replay-derived: ``used`` through the same distinct-chargeable
    counting the admission valve enforces, storm-ness through the same closed
    settlement classifier the mint path ran.

    Parameters
    ----------
    model_rows:
        Model ledger rows (heads or full history; the newest row per
        ``(stable_id, work_id)`` governs).
    valve_rows:
        Quota-deferral valve rows (pre-validated or raw; invalid rows refuse).
    artifact_events, operational_events:
        The evidence streams the allowance and storm derivations replay.
    granted_attempts:
        Optional ``(stable_id, work_id) -> granted attempts extension``
        mapping; absent pairs derive the base allowance.
    """

    # Lazy: author_spend transitively imports checkpoint, which imports status,
    # which imports this module's row vocabulary at module scope.
    from menagerie.crawler.author_spend import (  # noqa: PLC0415
        author_session_allowance,
        author_sessions_exhaustion_dimension,
        chargeable_author_sessions,
    )

    heads: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in model_rows:
        status = row.get("status")
        if not isinstance(status, Mapping):
            continue
        stable_id = row.get("stable_id")
        work_id = status.get("work_id") or row.get("work_id")
        if not isinstance(stable_id, str) or not isinstance(work_id, str):
            continue
        heads[(stable_id, work_id)] = row

    merged = (*artifact_events, *operational_events)
    exhausted: list[tuple[str, str, int, int]] = []
    understated: list[tuple[str, str, int, int]] = []
    for (stable_id, work_id), row in sorted(heads.items()):
        status = row.get("status")
        if not isinstance(status, Mapping):
            continue
        if status.get("reason_code") != "effort-exhausted:author-sessions":
            continue
        used = chargeable_author_sessions(merged, stable_id, work_id)
        extension = (granted_attempts or {}).get((stable_id, work_id), 0)
        allowance = author_session_allowance(extension or None)
        exhausted.append((stable_id, work_id, used, allowance))
        if used < allowance:
            understated.append((stable_id, work_id, used, allowance))

    storm_valves: list[tuple[str, str]] = []
    unsupported: list[tuple[str, str]] = []
    for row in valve_rows:
        validated = validate_quota_deferral_row(row)
        if validated["disposition"] != QUOTA_DEFERRAL_DISPOSITION:
            continue
        pair = (str(validated["stable_id"]), str(validated["work_id"]))
        if pair in storm_valves:
            continue
        storm_valves.append(pair)
        dimension = author_sessions_exhaustion_dimension(merged, pair[0], pair[1])
        if dimension != "provider-storm":
            unsupported.append(pair)

    return QuotaExhaustionCensus(
        exhausted_heads=tuple(exhausted),
        understated_heads=tuple(understated),
        storm_valves=tuple(storm_valves),
        unsupported_valves=tuple(unsupported),
    )
