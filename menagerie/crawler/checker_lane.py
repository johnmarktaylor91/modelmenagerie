"""Checker-lane contract-failure parking for models the gate lane never judged.

A metadata gate batch is checked by an external agentic checker. When that lane
cannot deliver a schema-valid verdict -- a permanent contract rejection after the
wrapper's bounded re-asks, a partial result that omits batch members, or a gate
the routing validators refuse -- there is NO verdict about any member model. The
2026-08-06 head50 rung (rung 10) terminalized 41 innocent models
``failed:runner`` / ``protocol-violation`` from exactly three such batch
failures, including a model that had already proven it runs.

This module is the durable valve for that outcome class, mirroring the two
pre-admission valves (:mod:`menagerie.crawler.environment_coverage`,
:mod:`menagerie.crawler.host_capacity`):

* **It refuses to mint canonical terminals.** A model whose gate lane failed is
  PARKED: withheld from this run, recorded here, and requeued on the next run,
  where a healthy checker lane can still gate it. No model record asserts a
  failure the model never earned.
* **It is accounted, never silent.** The terminal partition dereferences these
  rows exactly like the admission-valve rows: every in-scope model must be
  terminal XOR machine-deferral-claimed, so a parked model can never simply
  vanish (:data:`menagerie.crawler.status.MACHINE_DEFERRAL_DISPOSITIONS`).
* **It is a last resort, not a first response.** The driver first downshifts the
  failed batch along a halving ladder (floor one); only a model whose gate
  cannot be delivered even as a single-item batch parks here.

Campaign-scoped checker weather -- provider outages, transport failures,
usage-limit backoff -- never reaches this valve: those signals stay loud and
campaign-level in the driver.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Optional, Sequence

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

CHECKER_LANE_DEFERRAL_SCHEMA_VERSION_V1 = "menagerie.crawler.checker-lane-deferral.v1"
"""Historical schema identity: no checker adjudication identity on the row."""

CHECKER_LANE_DEFERRAL_SCHEMA_VERSION = "menagerie.crawler.checker-lane-deferral.v2"
"""Frozen schema identity for one checker-lane deferral row.

v2 (BL-10 f) additionally records the checker ADJUDICATION IDENTITY the lane
failed under (``checker_lane.checker_identity``: provider/model/version/
prompt_sha256), which is what gives the park a causal release predicate: the
row holds while that identity is unchanged and the model has exhausted its
bounded recidivism budget, and releases the moment the lane's identity rotates.
v1 rows carry no identity and keep their historical requeue-every-run behavior.
"""

CHECKER_LANE_DEFERRAL_SCHEMA_VERSIONS = frozenset(
    {CHECKER_LANE_DEFERRAL_SCHEMA_VERSION_V1, CHECKER_LANE_DEFERRAL_SCHEMA_VERSION}
)
"""Every readable checker-lane deferral schema (append-only history stays valid)."""

CHECKER_LANE_DEFERRAL_DISPOSITION = "deferred:checker-lane-contract-failure"
"""Greppable disposition for a model parked because the checker lane failed it."""

CHECKER_LANE_DEFERRAL_REMEDY = "checker-lane-contract-failure"
"""Closed remedy token naming the recovery: requeue once the lane honors its contract."""

CHECKER_LANE_DEFERRAL_RELATIVE_PATH = Path("intake-extensions") / "checker-lane-deferrals.jsonl"
"""Ledger location below the canonical records root."""

CHECKER_LANE_FAILURE_CLASSES = frozenset(
    {
        "permanent-contract-rejection",
        "partial-result",
        "invalid-gate-result",
        "lane-error",
        "author-evidence-defect",
        "checker-disclosure-debt",
        "checker-wall-exhausted",
        "checker-verdict-conflict",
        "checker-accept-unconfirmed",
    }
)
"""Closed vocabulary for what the checker lane did instead of rendering a verdict.

``permanent-contract-rejection`` is the operator wrapper's own exit-64
classification (the checker ran; its result failed the gate contract after the
bounded re-asks). ``partial-result`` is the typed partial/extra-items refusal.
``invalid-gate-result`` is a delivered gate the driver's routing validators
refused. ``lane-error`` is any other lane-owned failure that produced no
verdict.

The three BL-10 classes: ``author-evidence-defect`` is the typed ITEM-scoped
write-contract refusal (the item's own frozen author artifact binds evidence
that does not support its claim -- a fact about that model, truthfully
attributed to the author lane instead of the checker). ``checker-disclosure-debt``
is the absolute metadata round ceiling (the checker kept disclosing fresh
complaints past ``METADATA_GATE_ROUND_CEILING`` total rounds -- checker debt,
never author incorrigibility, so the model parks for human review instead of
minting a ``failed:*`` terminal). ``checker-wall-exhausted`` is the per-model
checker wall budget breaker (the lane spent its bounded wall on this model
without rendering a verdict).

``checker-verdict-conflict`` (AG-4) is the conflict-authority park: the lane
rendered verdicts that DISAGREE for byte-identical content under the SAME
full adjudication identity -- either a fresh verdict contradicting a
persisted one (the m4976 same-prompt flip shape) or a blind duplicate-accept
confirmation disagreeing with its accept. The instability is checker
property routed to human review; the durable disagreement evidence lives on
``intake-extensions/checker-verdict-conflicts.jsonl``
(:mod:`menagerie.crawler.checker_conflicts`).

``checker-accept-unconfirmed`` (H-5) is the duplicate-accept BUDGET FUSE: a
metadata/fidelity acceptance whose bound K=2 blind confirmation could not be
FUNDED (the per-model checker wall budget refused the confirmation session
pre-dispatch). The item parks unconfirmed rather than publishing unconfirmed
-- disposition-invariant, exactly the ``unfunded_confirmation_outcome`` const
the calibration receipt binds (:mod:`menagerie.crawler.checker_calibration`).
"""

CHECKER_LANE_NON_LADDER_FAILURE_CLASSES = frozenset(
    {
        "checker-disclosure-debt",
        "checker-wall-exhausted",
        "checker-verdict-conflict",
        "checker-accept-unconfirmed",
    }
)
"""Classes parked by a BUDGET or CEILING decision rather than a spent halving
ladder: their ``attempted_batch_sizes`` trail records what was actually spent
(possibly ending above one) instead of proving a ladder ran to the floor."""

CHECKER_LANE_FAILURE_EVIDENCE_MAX_CHARS = 1_500
"""Head-bounded failure evidence, matching the operator-lane STDIO tail bound."""


class CheckerLaneDeferralError(ValueError):
    """A checker-lane deferral row is malformed or does not record a lane failure."""


def checker_lane_deferral_path(records_root: Path) -> Path:
    """Return the canonical checker-lane deferral ledger below one records root.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.

    Returns
    -------
    pathlib.Path
        Append-only checker-lane deferral JSONL path.
    """

    return records_root / CHECKER_LANE_DEFERRAL_RELATIVE_PATH


def build_checker_lane_deferral_row(
    *,
    stable_id: str,
    work_id: str,
    name: str,
    campaign_id: str,
    run_id: str,
    machine_id: str,
    created_at: str,
    gate_kind: str,
    failure_class: str,
    failure_evidence: str,
    attempted_batch_sizes: Sequence[int],
    checker_identity: Optional[Mapping[str, Any]] = None,
) -> JsonObject:
    """Assemble one complete, self-describing checker-lane deferral row.

    The row states WHAT the lane did instead of rendering a verdict, the exact
    downshift ladder the driver already spent, and the head-bounded failure
    evidence, so a later reviewer can re-derive the decision without the code
    that made it.

    Parameters
    ----------
    stable_id, work_id, name:
        Durable model identity and the active scheduled work generation.
    campaign_id, run_id, machine_id:
        Campaign and host provenance for the parking decision.
    created_at:
        UTC timestamp of the decision.
    gate_kind:
        Gate kind whose lane failed (``metadata_batch`` today).
    failure_class:
        One of :data:`CHECKER_LANE_FAILURE_CLASSES`.
    failure_evidence:
        Diagnosis text from the final floor-one failure; head-truncated.
    attempted_batch_sizes:
        Failed batch sizes this model rode down the halving ladder, largest
        first, ending at one -- or, for the budget/ceiling classes in
        :data:`CHECKER_LANE_NON_LADDER_FAILURE_CLASSES`, the sizes actually
        spent before the breaker fired.
    checker_identity:
        The checker adjudication identity the lane failed under
        (``provider``/``model``/``version``/``prompt_sha256``), when the lane
        can name one. This is what arms the causal release predicate: the park
        holds under an unchanged identity once the recidivism budget is spent,
        and releases when the identity rotates.

    Returns
    -------
    dict[str, Any]
        Content-addressed deferral row.

    Raises
    ------
    CheckerLaneDeferralError
        If the failure class is unknown or the ladder does not end at one.
    """

    if not campaign_id or campaign_id == str(None):
        # T2-12: the pilot froze campaign_id="None" (a stringified absence)
        # into 62 row digests. The absence must refuse at BUILD time; read-side
        # validators stay tolerant so those historical rows remain readable.
        raise ValueError("campaign_id must name a real campaign, never a stringified absence")
    if failure_class not in CHECKER_LANE_FAILURE_CLASSES:
        raise CheckerLaneDeferralError(
            f"checker-lane deferral names unknown failure class {failure_class!r}"
        )
    sizes = [int(size) for size in attempted_batch_sizes]
    if not sizes or any(size < 1 for size in sizes):
        raise CheckerLaneDeferralError(
            f"a checker-lane deferral records the batch sizes it spent (got {sizes!r})"
        )
    if failure_class not in CHECKER_LANE_NON_LADDER_FAILURE_CLASSES and sizes[-1] != 1:
        raise CheckerLaneDeferralError(
            "a checker-lane deferral records a spent halving ladder, so its attempted "
            f"batch sizes must end at the floor of one (got {sizes!r})"
        )
    evidence = failure_evidence.strip()
    if len(evidence) > CHECKER_LANE_FAILURE_EVIDENCE_MAX_CHARS:
        evidence = evidence[:CHECKER_LANE_FAILURE_EVIDENCE_MAX_CHARS] + "...<truncated>"
    lane: JsonObject = {
        "gate_kind": gate_kind,
        "failure_class": failure_class,
        "failure_evidence": evidence,
        "attempted_batch_sizes": sizes,
        "remedy": CHECKER_LANE_DEFERRAL_REMEDY,
    }
    if checker_identity is not None:
        identity = {
            field: checker_identity.get(field)
            for field in ("provider", "model", "version", "prompt_sha256")
        }
        if not all(isinstance(value, str) and value for value in identity.values()):
            raise CheckerLaneDeferralError(
                "checker-lane deferral checker_identity must carry non-empty "
                "provider/model/version/prompt_sha256"
            )
        lane["checker_identity"] = identity
    row: JsonObject = {
        "schema_version": CHECKER_LANE_DEFERRAL_SCHEMA_VERSION,
        "stable_id": stable_id,
        "work_id": work_id,
        "name": name,
        "campaign_id": campaign_id,
        "run_id": run_id,
        "machine_id": machine_id,
        "created_at": created_at,
        "disposition": CHECKER_LANE_DEFERRAL_DISPOSITION,
        "checker_lane": lane,
        "recheck_hint": (
            f"{CHECKER_LANE_DEFERRAL_REMEDY}: the checker lane could not deliver a "
            "schema-valid verdict for this model even as a single-item gate; no verdict "
            "was rendered on it and its authored proposal is intact -- requeue it on the "
            "next run"
        ),
    }
    row = stamp_row_digest(row)
    return row


def validate_checker_lane_deferral_row(row: Mapping[str, Any]) -> JsonObject:
    """Validate one checker-lane deferral row and return its canonical copy.

    Parameters
    ----------
    row:
        Candidate deferral row.

    Returns
    -------
    dict[str, Any]
        Validated row.

    Raises
    ------
    CheckerLaneDeferralError
        If a required field is missing, mistyped, or the digest disagrees.
    """

    required = (
        "schema_version",
        "stable_id",
        "work_id",
        "name",
        "campaign_id",
        "run_id",
        "machine_id",
        "created_at",
        "disposition",
        "checker_lane",
        "recheck_hint",
        "row_sha256",
    )
    missing = [field for field in required if field not in row]
    if missing:
        raise CheckerLaneDeferralError(
            f"checker-lane deferral row is missing {', '.join(sorted(missing))}"
        )
    if row["schema_version"] not in CHECKER_LANE_DEFERRAL_SCHEMA_VERSIONS:
        raise CheckerLaneDeferralError(
            f"checker-lane deferral row declares unknown schema {row['schema_version']!r}"
        )
    if row["disposition"] != CHECKER_LANE_DEFERRAL_DISPOSITION:
        raise CheckerLaneDeferralError(
            f"checker-lane deferral row declares unknown disposition {row['disposition']!r}"
        )
    for field in ("stable_id", "work_id", "name", "campaign_id", "run_id", "machine_id"):
        if not isinstance(row[field], str) or not row[field]:
            raise CheckerLaneDeferralError(f"checker-lane deferral row has an empty {field}")
    lane = row["checker_lane"]
    if not isinstance(lane, Mapping):
        raise CheckerLaneDeferralError("checker-lane deferral row must describe the lane failure")
    if lane.get("failure_class") not in CHECKER_LANE_FAILURE_CLASSES:
        raise CheckerLaneDeferralError(
            f"checker-lane deferral row names unknown failure class {lane.get('failure_class')!r}"
        )
    if lane.get("remedy") != CHECKER_LANE_DEFERRAL_REMEDY:
        raise CheckerLaneDeferralError(
            f"checker-lane deferral row names unknown remedy {lane.get('remedy')!r}"
        )
    sizes = lane.get("attempted_batch_sizes")
    if (
        not isinstance(sizes, Sequence)
        or isinstance(sizes, (str, bytes))
        or not sizes
        or not all(isinstance(size, int) and size >= 1 for size in sizes)
        or (
            lane.get("failure_class") not in CHECKER_LANE_NON_LADDER_FAILURE_CLASSES
            and sizes[-1] != 1
        )
    ):
        raise CheckerLaneDeferralError(
            "checker-lane deferral row must record the spent halving ladder ending at one"
        )
    identity = lane.get("checker_identity")
    if identity is not None and (
        not isinstance(identity, Mapping)
        or set(identity) != {"provider", "model", "version", "prompt_sha256"}
        or not all(isinstance(value, str) and value for value in identity.values())
    ):
        raise CheckerLaneDeferralError(
            "checker-lane deferral row checker_identity must carry exactly "
            "provider/model/version/prompt_sha256"
        )
    validated = dict(json.loads(canonical_json_bytes(row).decode("utf-8")))
    foreign = foreign_content_address_fields(validated, own="row_sha256")
    if foreign:
        raise CheckerLaneDeferralError(
            f"checker-lane deferral row carries a foreign content-address field "
            f"{', '.join(foreign)}: a row of this family owns exactly one content address"
        )
    digest = validated.pop("row_sha256")
    if row_content_address(validated) != digest:
        raise CheckerLaneDeferralError(
            "checker-lane deferral row digest does not match its payload"
        )
    validated["row_sha256"] = digest
    return validated


def append_checker_lane_deferral_row(path: Path, row: Mapping[str, Any]) -> JsonObject:
    """Append one checker-lane deferral durably and idempotently.

    Parameters
    ----------
    path:
        Destination checker-lane deferral JSONL.
    row:
        Complete typed deferral row.

    Returns
    -------
    dict[str, Any]
        Validated persisted row.

    Raises
    ------
    CheckerLaneDeferralError
        If the row is malformed.

    Notes
    -----
    The FIRST parking recorded for a work generation is the record, and a later
    run that re-parks the same generation is a no-op: ``created_at``, ``run_id``,
    and the ladder trail necessarily differ on every resume, so treating a
    re-derived parking as a conflict would abort the driver on the second storm
    over the same model. A model the checker lane eventually gates is not parked
    at all and writes nothing here, so a stale row can only ever be superseded by
    the model completing normally.
    """

    validated = validate_checker_lane_deferral_row(row)
    key = (validated["stable_id"], validated["work_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        existing = _decode_checker_lane_lines(_ledger_bytes(path, handle), path)
        matching = [item for item in existing if (item["stable_id"], item["work_id"]) == key]
        if matching:
            return matching[0]
        handle.seek(0, os.SEEK_END)
        handle.write(canonical_json_bytes(validated) + b"\n")
        handle.flush()
        durable_fsync(handle.fileno())
    fsync_directory(path.parent)
    return validated


def _ledger_bytes(path: Path, handle: Optional[BinaryIO] = None) -> bytes:
    """Read the whole ledger, recovering only a torn final line with evidence.

    A complete malformed line stays this valve's typed hard refusal: the
    recovery's prefix check is this module's own strict decoder.
    """

    return read_with_tail_recovery(
        path,
        prefix_check=lambda retained: _decode_checker_lane_lines(retained.read_bytes(), path),
        locked_handle=handle,
    )


def load_checker_lane_deferral_rows(paths: Sequence[Path]) -> tuple[JsonObject, ...]:
    """Load validated checker-lane deferrals in ledger order.

    The one validated, tail-recovering read path for status and preflight
    consumers, exactly as the sibling valves expose. Rows keep LEDGER order
    (no per-work-generation reduction): the first parking recorded for a work
    generation is the record, and downstream per-model merges rely on
    chronological order, not on the lexical order a keyed reduction imposes.

    Parameters
    ----------
    paths:
        Candidate ledger paths. Missing files are empty ledgers.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Validated rows in ledger order.
    """

    rows: list[JsonObject] = []
    for path in paths:
        if not path.is_file():
            continue
        rows.extend(_decode_checker_lane_lines(_ledger_bytes(path), path))
    return tuple(rows)


def _decode_checker_lane_lines(payload: bytes, path: Path) -> tuple[JsonObject, ...]:
    """Decode and validate every row in one deferral ledger's bytes."""

    rows: list[JsonObject] = []
    for index, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CheckerLaneDeferralError(
                f"checker-lane deferral ledger {path} has an unreadable row at line {index}"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise CheckerLaneDeferralError(
                f"checker-lane deferral ledger {path} has a non-object row at line {index}"
            )
        rows.append(validate_checker_lane_deferral_row(decoded))
    return tuple(rows)
