"""Checker verdict conflict authority + the terminal duplicate-accept ledger.

AG-4 (PLAN_v6 §2 WS1, Wave-2 rotation batch). 200a evidence: the only proven
checker nondeterminism is same-prompt disagreement on byte-identical content
(m4976 flipped accept->reject under the IDENTICAL prompt digest during the
08-23 sweep), and the measured bad-pass floor on the bless direction is 6.7%
(sol U-S1). Two mechanisms answer it, both durable and fail-closed:

* **Conflict authority.** A fresh terminal-disposition verdict that DISAGREES
  with any persisted verdict for the same ``(stable_id, author_result_sha256)``
  under the SAME full adjudication identity -- the
  :func:`~menagerie.crawler.checker_dispatch.gate_adjudication_identity`
  4-tuple ``(provider, model, version, prompt_sha256)`` -- is a typed
  ``verdict-conflict`` row here plus a ``checker-verdict-conflict`` park on
  the checker-lane valve: the instability is operator property, never a
  silent re-roll. Cross-identity disagreement (the m3222/m8215 rotation
  shape) is LAWFUL and recorded as an ``identity-attributed-flip``
  disclosure row, never parked -- rotation is a cause the census watches,
  not a defect the lane adjudicates.
* **Duplicate-accept.** Terminal ACCEPTANCE requires two blind consistent
  accepts (B1-XQ5's ~2x TD spend): after a fresh accept the driver buys one
  more LIVE session over the same envelope and records the confirmation as a
  ``duplicate-accept-confirmation`` row bound to the confirmed row's
  ``gate_id``. The confirmation is deliberately NEVER a second gates-ledger
  row: a duplicate accepted item row would violate the exact-membership
  anti-substitution rule at publication
  (``resolve_exact_gate_item_membership``), so the second verdict's identity,
  envelope binding, and result-envelope digest live here. ``status`` is
  DERIVED from the confirmation verdict (``confirmed`` iff ``accepted``,
  else ``conflict``) and a hand-flipped status refuses at validation.

The ledger is an ``intake-extensions`` DISCLOSURE ledger (like the promotion
extension), not a machine-deferral valve: the PARK a conflict fires rides the
existing checker-lane valve as failure class ``checker-verdict-conflict``
(ladder-exempt -- a decision-minted park, not a halving-ladder outcome), so
release, drain, and census machinery need no new registry row.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping, Optional, Sequence

from menagerie.crawler.checker_dispatch import gate_adjudication_identity
from menagerie.crawler.constants import (
    checker_confirmation_partner,
    fidelity_item_accepted,
    metadata_item_accepted,
    terminal_confirmation_sampled,
)
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

CHECKER_VERDICT_CONFLICT_SCHEMA_VERSION = "menagerie.crawler.checker-verdict-conflict.v1"

#: H-5 (rule-2 trilab consensus, 2026-08-29): the v2 row family generalizes
#: ``duplicate-accept-confirmation`` beyond the terminal lane to the metadata
#: and fidelity ACCEPTANCE write doors, binding the exact semantic QUESTION
#: (``question_sha256``) and both sessions' FULL verdict projections (Sol §6.5)
#: -- the same three-field-per-item projection the calibration matrix replays,
#: so the receipt's ``acceptance_witness`` K=2 bar and the production door
#: enforce one fact. The reader is mixed-era: v1 rows (terminal) validate under
#: v1 rules forever; only the two accuracy gate kinds may mint v2 rows.
CHECKER_VERDICT_CONFLICT_SCHEMA_VERSION_V2 = "menagerie.crawler.checker-verdict-conflict.v2"

#: Seat swap (2026-09-01, SEAT_RULING.md / item 32): the v3 row family records
#: a CROSS-MODEL duplicate-accept confirmation -- the primary seat's identity
#: (``first_identity``) AND the confirmation seat's, on every gate kind. The
#: two identities lawfully differ (Fable primary, sol confirmation at the same
#: frozen prompt epoch), so the v1/v2 rule "one adjudication identity per
#: question" is replaced by the seat-table pairing rule
#: (:func:`confirmation_pairing`), re-derived on every read from code, never
#: from a producer field. Same-identity pairs keep minting v1 (terminal) and
#: v2 (accuracy) rows byte-for-byte as before; v3 exists only for cross-model.
CHECKER_VERDICT_CONFLICT_SCHEMA_VERSION_V3 = "menagerie.crawler.checker-verdict-conflict.v3"

#: Closed K=2 pairing classes (derived, never declared).
CONFIRMATION_PAIRING_SAME_IDENTITY = "same-identity"
CONFIRMATION_PAIRING_CROSS_MODEL = "cross-model"
CONFIRMATION_PAIRINGS = frozenset(
    {CONFIRMATION_PAIRING_SAME_IDENTITY, CONFIRMATION_PAIRING_CROSS_MODEL}
)

CHECKER_VERDICT_CONFLICT_RELATIVE_PATH = (
    Path("intake-extensions") / "checker-verdict-conflicts.jsonl"
)

#: Closed row-kind vocabulary. Extending it is a Wave-2-by-law vocabulary
#: change; the validator refuses anything outside it.
CONFLICT_ROW_KINDS = frozenset(
    {"verdict-conflict", "duplicate-accept-confirmation", "identity-attributed-flip"}
)

#: The two gate kinds whose acceptances settle through v2 duplicate-accept
#: rows (terminal keeps its v1 rows and verdict vocabulary untouched).
ACCURACY_DUPLICATE_ACCEPT_GATE_KINDS = frozenset({"metadata_batch", "fidelity"})

#: Closed duplicate-accept settlement vocabulary, DERIVED from the
#: confirmation verdict -- never producer-declared.
DUPLICATE_ACCEPT_STATUSES = frozenset({"confirmed", "conflict"})

#: The terminal-disposition verdict vocabulary this module compares
#: (gate-v3 ``terminal_disposition.verdict``).
TERMINAL_VERDICTS = frozenset({"accepted", "rejected", "cannot-verify"})

_IDENTITY_FIELDS = ("provider", "model", "version", "prompt_sha256")


class CheckerVerdictConflictError(ValueError):
    """Raised when a conflict-ledger row or comparison input is malformed."""


def checker_verdict_conflict_path(records_root: Path) -> Path:
    """Return the canonical conflict ledger below one records root.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.

    Returns
    -------
    pathlib.Path
        Append-only checker-verdict-conflict JSONL path.
    """

    return records_root / CHECKER_VERDICT_CONFLICT_RELATIVE_PATH


def _require_identity(identity: Any, *, field: str) -> dict[str, str]:
    """Validate one adjudication-identity mapping, fail-closed."""

    if (
        not isinstance(identity, Mapping)
        or set(identity) != set(_IDENTITY_FIELDS)
        or not all(isinstance(identity[key], str) and identity[key] for key in _IDENTITY_FIELDS)
    ):
        raise CheckerVerdictConflictError(
            f"checker-verdict-conflict row {field} must carry exactly non-empty "
            "provider/model/version/prompt_sha256"
        )
    return {key: str(identity[key]) for key in _IDENTITY_FIELDS}


def _require_verdict(verdict: Any, *, field: str) -> str:
    if verdict not in TERMINAL_VERDICTS:
        raise CheckerVerdictConflictError(
            f"checker-verdict-conflict row {field} must be one of "
            f"{sorted(TERMINAL_VERDICTS)}, got {verdict!r}"
        )
    return str(verdict)


def _require_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise CheckerVerdictConflictError(f"checker-verdict-conflict row has an empty {field}")
    return value


def _common_row(
    *,
    row_kind: str,
    stable_id: str,
    work_id: str,
    campaign_id: str,
    run_id: str,
    machine_id: str,
    created_at: str,
    gate_kind: str,
    author_result_sha256: str,
) -> JsonObject:
    for field, value in (
        ("stable_id", stable_id),
        ("work_id", work_id),
        ("campaign_id", campaign_id),
        ("run_id", run_id),
        ("machine_id", machine_id),
        ("created_at", created_at),
        ("gate_kind", gate_kind),
        ("author_result_sha256", author_result_sha256),
    ):
        _require_text(value, field=field)
    return {
        "schema_version": CHECKER_VERDICT_CONFLICT_SCHEMA_VERSION,
        "row_kind": row_kind,
        "stable_id": stable_id,
        "work_id": work_id,
        "campaign_id": campaign_id,
        "run_id": run_id,
        "machine_id": machine_id,
        "created_at": created_at,
        "gate_kind": gate_kind,
        "author_result_sha256": author_result_sha256,
    }


def _pair_row(
    *,
    row_kind: str,
    new_gate_id: str,
    prior_gate_id: str,
    new_verdict: str,
    prior_verdict: str,
    new_identity: Mapping[str, Any],
    prior_identity: Mapping[str, Any],
    **common: str,
) -> JsonObject:
    """Assemble one verdict-conflict or identity-attributed-flip row."""

    row = _common_row(row_kind=row_kind, **common)
    new_bound = _require_identity(new_identity, field="new_identity")
    prior_bound = _require_identity(prior_identity, field="prior_identity")
    verdict_new = _require_verdict(new_verdict, field="new_verdict")
    verdict_prior = _require_verdict(prior_verdict, field="prior_verdict")
    if verdict_new == verdict_prior:
        raise CheckerVerdictConflictError(
            f"a {row_kind} row records a DISAGREEMENT; both verdicts are {verdict_new!r}"
        )
    identities_equal = new_bound == prior_bound
    if row_kind == "verdict-conflict" and not identities_equal:
        raise CheckerVerdictConflictError(
            "a verdict-conflict row binds the SAME full adjudication identity on both "
            "sides; a cross-identity disagreement is an identity-attributed-flip"
        )
    if row_kind == "identity-attributed-flip" and identities_equal:
        raise CheckerVerdictConflictError(
            "an identity-attributed-flip row binds two DIFFERENT adjudication "
            "identities; a same-identity disagreement is a verdict-conflict"
        )
    row.update(
        {
            "new_gate_id": _require_text(new_gate_id, field="new_gate_id"),
            "prior_gate_id": _require_text(prior_gate_id, field="prior_gate_id"),
            "new_verdict": verdict_new,
            "prior_verdict": verdict_prior,
            "new_identity": new_bound,
            "prior_identity": prior_bound,
        }
    )
    return stamp_row_digest(row)


def build_verdict_conflict_row(**kwargs: Any) -> JsonObject:
    """Assemble one same-identity verdict-conflict row (parks the model)."""

    return _pair_row(row_kind="verdict-conflict", **kwargs)


def build_identity_flip_row(**kwargs: Any) -> JsonObject:
    """Assemble one cross-identity flip disclosure row (never parks)."""

    return _pair_row(row_kind="identity-attributed-flip", **kwargs)


def confirmation_pairing(
    gate_kind: str,
    first_identity: Mapping[str, Any],
    confirmation_identity: Mapping[str, Any],
) -> str:
    """Return the lawful K=2 pairing class of two adjudication identities.

    The seat table (``constants.CHECKER_SEATS``) is the authority, re-derived
    on every call: a first identity whose model is NOT the seated primary for
    ``gate_kind`` (a test lane's label, a legacy sol-primary record) pairs only
    with ITSELF -- the historical same-identity K=2 -- while the seated
    primary pairs only with the seated CONFIRMATION model under that model's
    provider at the SAME frozen prompt epoch. Two Fable sessions are never a
    pair; a partner at another prompt epoch answered a different question.

    Raises
    ------
    CheckerVerdictConflictError
        If the pair is unlawful under the seat table.
    """

    first = _require_identity(first_identity, field="first_identity")
    second = _require_identity(confirmation_identity, field="confirmation_identity")
    partner = checker_confirmation_partner(gate_kind, first["model"])
    if partner is None:
        if first != second:
            raise CheckerVerdictConflictError(
                "duplicate-accept confirmation adjudicated a different question "
                f"(adjudication identity diverged: unseated primary {first['model']!r} pairs "
                f"only with itself, got {second['model']!r} at prompt {second['prompt_sha256']})"
            )
        return CONFIRMATION_PAIRING_SAME_IDENTITY
    if second["model"] != partner.model or second["provider"] != partner.provider:
        raise CheckerVerdictConflictError(
            f"duplicate-accept confirmation for seated primary {first['model']!r} on "
            f"{gate_kind} must come from the seated confirmation seat {partner.model!r} "
            f"({partner.provider}), got {second['model']!r} ({second['provider']}); two "
            "sessions of the primary model are never a cross-model K=2 pair"
        )
    if second["prompt_sha256"] != first["prompt_sha256"]:
        raise CheckerVerdictConflictError(
            "duplicate-accept confirmation adjudicated a different question (prompt epoch "
            f"diverged: the primary ran under {first['prompt_sha256']}, the confirmation "
            f"under {second['prompt_sha256']}); a cross-model pair shares one frozen epoch"
        )
    return CONFIRMATION_PAIRING_CROSS_MODEL


def _question_core(gate: Mapping[str, Any], stable_id: Optional[str]) -> dict[str, Any]:
    """Return the identity-free adjudication question one gate row answered."""

    if gate.get("gate_kind") == "terminal_disposition":
        core = terminal_question_binding(gate)
    else:
        if stable_id is None:
            raise CheckerVerdictConflictError("an accuracy question binding needs its stable_id")
        core = accuracy_question_binding(gate, stable_id)
    core.pop("adjudication_identity", None)
    return core


def require_confirmation_pairing(
    gate: Mapping[str, Any],
    confirmation: Mapping[str, Any],
    *,
    stable_id: Optional[str] = None,
) -> str:
    """Verify a K=2 pair answered the SAME question under a LAWFUL seat pairing.

    Returns the pairing class. Subsumes the pre-swap check (binding equality
    including identity): for an unseated primary the lawful pairing IS
    identity equality, so every historical pair verifies exactly as before.
    """

    if _question_core(gate, stable_id) != _question_core(confirmation, stable_id):
        raise CheckerVerdictConflictError(
            "duplicate-accept confirmation adjudicated a different question (item or "
            "content identity diverged); a blind confirmation replays the SAME question "
            "or records nothing"
        )
    return confirmation_pairing(
        str(gate.get("gate_kind")),
        adjudication_identity_fields(gate),
        adjudication_identity_fields(confirmation),
    )


def confirmation_sampled(gate: Mapping[str, Any], *, stable_id: str) -> bool:
    """Decide whether one accepted gate row buys its blind K=2 confirmation.

    A seated primary confirms at its partner seat's ``confirmation_sample_rate``
    (every metadata and fidelity accept; a 10 percent deterministic sample of
    terminal dispositions per the RULING checker row); an unseated primary
    confirms every accept, the pre-swap rule. Pure over the persisted row.
    """

    kind = str(gate.get("gate_kind"))
    identity = adjudication_identity_fields(gate)
    partner = checker_confirmation_partner(kind, identity["model"]) if identity["model"] else None
    if partner is None or partner.confirmation_sample_rate >= 1.0:
        return True
    if kind == "terminal_disposition":
        content = str(_terminal_item(gate)["terminal_disposition"].get("author_result_sha256"))
    else:
        content = str(_accuracy_gate_item(gate, stable_id).get("vet_identity"))
    return terminal_confirmation_sampled(
        stable_id, content, rate=partner.confirmation_sample_rate
    )


def build_duplicate_accept_row(
    *,
    gate_id: str,
    gate_identity: str,
    confirmation_gate_identity: str,
    confirmation_verdict: str,
    confirmation_identity: Mapping[str, Any],
    confirmation_result_envelope_sha256: str,
    first_identity: Optional[Mapping[str, Any]] = None,
    **common: str,
) -> JsonObject:
    """Assemble one duplicate-accept confirmation row.

    ``status`` is DERIVED: ``confirmed`` iff the blind confirmation session
    also accepted, else ``conflict``. Question equality is the CALLER's
    pre-condition, verified over :func:`require_confirmation_pairing` (the
    semantic adjudication question plus the seat pairing) BEFORE this builder
    runs -- envelope digests lawfully differ across sessions (each dispatch
    mints a fresh deadline and output path), so both are RECORDED here, never
    compared. With ``first_identity`` supplied, a CROSS-MODEL pair mints a v3
    row carrying both identities; a same-identity pair (or no first identity,
    the legacy call shape) mints the v1 row byte-for-byte as before.
    """

    pairing = CONFIRMATION_PAIRING_SAME_IDENTITY
    if first_identity is not None:
        pairing = confirmation_pairing(
            str(common.get("gate_kind")), first_identity, confirmation_identity
        )
    row = _common_row(row_kind="duplicate-accept-confirmation", **common)
    bound_gate_identity = _require_text(gate_identity, field="gate_identity")
    bound_confirmation_identity = _require_text(
        confirmation_gate_identity, field="confirmation_gate_identity"
    )
    if bound_confirmation_identity == bound_gate_identity:
        # P5 fable-F16: the K=2 door's independence was call-path-only. A
        # confirmation is a SECOND blind session; every dispatch mints a fresh
        # envelope, so equal gate identities mean the row self-witnesses.
        raise CheckerVerdictConflictError(
            "duplicate-accept confirmation must come from a distinct gate: "
            f"confirmation_gate_identity equals gate_identity ({bound_gate_identity})"
        )
    verdict = _require_verdict(confirmation_verdict, field="confirmation_verdict")
    row.update(
        {
            "gate_id": _require_text(gate_id, field="gate_id"),
            "gate_identity": bound_gate_identity,
            "confirmation_gate_identity": bound_confirmation_identity,
            "first_verdict": "accepted",
            "confirmation_verdict": verdict,
            "confirmation_identity": _require_identity(
                confirmation_identity, field="confirmation_identity"
            ),
            "confirmation_result_envelope_sha256": _require_text(
                confirmation_result_envelope_sha256,
                field="confirmation_result_envelope_sha256",
            ),
            "status": "confirmed" if verdict == "accepted" else "conflict",
        }
    )
    if pairing == CONFIRMATION_PAIRING_CROSS_MODEL and first_identity is not None:
        row["schema_version"] = CHECKER_VERDICT_CONFLICT_SCHEMA_VERSION_V3
        row["first_identity"] = _require_identity(first_identity, field="first_identity")
        row["confirmation_pairing"] = pairing
    return stamp_row_digest(row)


# ---------------------------------------------------------------------------
# H-5 accuracy-lane kernels: projection, acceptance, question binding.
# ---------------------------------------------------------------------------


def _accuracy_gate_item(gate: Mapping[str, Any], stable_id: str) -> Mapping[str, Any]:
    """Return one accuracy gate row's item for a member, fail-closed."""

    kind = gate.get("gate_kind")
    if kind not in ACCURACY_DUPLICATE_ACCEPT_GATE_KINDS:
        raise CheckerVerdictConflictError(
            f"accuracy duplicate-accept settles only {sorted(ACCURACY_DUPLICATE_ACCEPT_GATE_KINDS)} "
            f"gate rows, got {kind!r}"
        )
    items = gate.get("items")
    if not isinstance(items, Sequence):
        raise CheckerVerdictConflictError("accuracy gate row carries no items")
    for item in items:
        if isinstance(item, Mapping) and item.get("stable_id") == stable_id:
            return item
    raise CheckerVerdictConflictError(
        f"accuracy gate row {gate.get('gate_id')!r} carries no item for {stable_id!r}"
    )


def accuracy_item_projection(gate_kind: str, item: Mapping[str, Any]) -> dict[str, Any]:
    """Project one accuracy gate item onto its full verdict projection.

    Byte-for-byte the projection the blind-replay matrix compares
    (``tools.checker_calibration_runner.verdict_projection``): exactly the
    verdict-steering enums, never prose, latency, or machine-owned scaffold.
    The receipt's zero bars and this ledger's confirmation rows therefore
    adjudicate one comparison surface.
    """

    integrity = item.get("integrity")
    rung_check = item.get("rung_check")
    projection: dict[str, Any] = {
        "rung_check": rung_check.get("verdict") if isinstance(rung_check, Mapping) else None,
        "integrity": integrity.get("verdict") if isinstance(integrity, Mapping) else None,
    }
    if gate_kind == "fidelity":
        fidelity = item.get("fidelity")
        projection["fidelity"] = (
            fidelity.get("verdict") if isinstance(fidelity, Mapping) else None
        )
    else:
        projection["verdict"] = item.get("verdict")
    return projection


def accuracy_item_accepted(gate_kind: str, item: Mapping[str, Any]) -> bool:
    """Return one accuracy gate item's acceptance under its lane's ONE kernel."""

    if gate_kind == "metadata_batch":
        return metadata_item_accepted(item)
    if gate_kind == "fidelity":
        return fidelity_item_accepted(item)
    raise CheckerVerdictConflictError(f"unknown accuracy gate kind {gate_kind!r}")


def accuracy_projection_accepted(gate_kind: str, projection: Mapping[str, Any]) -> bool:
    """Re-derive acceptance from a persisted projection (validator side).

    The projection carries exactly the acceptance-steering fields, so this is
    the same fact :func:`accuracy_item_accepted` derives from the live item;
    spelled separately because a ledger row must re-derive without the item.
    """

    if gate_kind == "metadata_batch":
        return (
            projection.get("verdict") == "accurate"
            and projection.get("integrity") == "accurate"
            and projection.get("rung_check") == "accurate"
        )
    if gate_kind == "fidelity":
        return (
            projection.get("fidelity") in {"match", "minor-drift"}
            and projection.get("rung_check") == "accurate"
        )
    raise CheckerVerdictConflictError(f"unknown accuracy gate kind {gate_kind!r}")


def accuracy_question_binding(gate: Mapping[str, Any], stable_id: str) -> dict[str, Any]:
    """Return the semantic adjudication QUESTION one accuracy gate row answered.

    The accuracy twin of :func:`terminal_question_binding` over the identity
    vector every gate item already binds (the ``_gate_item_matches_proposal``
    fields, Sol §3.1): blindness is replication of the SAME question -- item,
    content identity, and full adjudication identity -- never of envelope
    bytes, which lawfully differ per session.
    """

    item = _accuracy_gate_item(gate, stable_id)
    return {
        "gate_kind": gate.get("gate_kind"),
        "stable_id": item.get("stable_id"),
        "work_id": item.get("work_id"),
        "campaign_root_work_id": item.get("campaign_root_work_id"),
        "family_representative_id": item.get("family_representative_id"),
        "vet_identity": item.get("vet_identity"),
        "fidelity_identity": item.get("fidelity_identity"),
        "verified_hashes": item.get("verified_hashes"),
        "adjudication_identity": gate_adjudication_identity(gate),
    }


def accuracy_question_sha256(gate: Mapping[str, Any], stable_id: str) -> str:
    """Return the canonical digest of one accuracy question binding."""

    binding = accuracy_question_binding(gate, stable_id)
    return row_content_address(
        {
            **binding,
            "adjudication_identity": list(binding["adjudication_identity"]),
        }
    )


def build_accuracy_duplicate_accept_row(
    *,
    gate: Mapping[str, Any],
    confirmation: Mapping[str, Any],
    stable_id: str,
    campaign_id: str,
    run_id: str,
    machine_id: str,
    created_at: str,
    author_result_sha256: str,
) -> JsonObject:
    """Assemble one v2 (metadata/fidelity) duplicate-accept confirmation row.

    Question equality is verified HERE (both sessions must have adjudicated
    the byte-identical semantic question under one adjudication identity);
    ``status`` is DERIVED: ``confirmed`` iff the two FULL verdict projections
    are exact-equal and the confirmation projection passes the lane's
    acceptance kernel, else ``conflict``. The first projection must be an
    acceptance -- rejections are never duplicated.
    """

    kind = str(gate.get("gate_kind"))
    # Kind membership first ("settles only ..."), then question + pairing.
    first_item = _accuracy_gate_item(gate, stable_id)
    pairing = require_confirmation_pairing(gate, confirmation, stable_id=stable_id)
    confirmation_item = _accuracy_gate_item(confirmation, stable_id)
    if not accuracy_item_accepted(kind, first_item):
        raise CheckerVerdictConflictError(
            "accuracy duplicate-accept rows exist only for acceptances; the first "
            "verdict is not an accept"
        )
    first_projection = accuracy_item_projection(kind, first_item)
    confirmation_projection = accuracy_item_projection(kind, confirmation_item)
    row = _common_row(
        row_kind="duplicate-accept-confirmation",
        stable_id=stable_id,
        work_id=str(first_item.get("work_id")),
        campaign_id=campaign_id,
        run_id=run_id,
        machine_id=machine_id,
        created_at=created_at,
        gate_kind=kind,
        author_result_sha256=author_result_sha256,
    )
    row["schema_version"] = CHECKER_VERDICT_CONFLICT_SCHEMA_VERSION_V2
    status = (
        "confirmed"
        if first_projection == confirmation_projection
        and accuracy_projection_accepted(kind, confirmation_projection)
        else "conflict"
    )
    row.update(
        {
            "gate_id": _require_text(gate.get("gate_id"), field="gate_id"),
            "gate_identity": _require_text(gate.get("gate_identity"), field="gate_identity"),
            "confirmation_gate_identity": _require_text(
                confirmation.get("gate_identity"), field="confirmation_gate_identity"
            ),
            "question_sha256": accuracy_question_sha256(gate, stable_id),
            "first_projection": first_projection,
            "first_projection_sha256": row_content_address(first_projection),
            "confirmation_projection": confirmation_projection,
            "confirmation_projection_sha256": row_content_address(confirmation_projection),
            "confirmation_identity": _require_identity(
                dict(
                    zip(
                        _IDENTITY_FIELDS,
                        gate_adjudication_identity(confirmation),
                    )
                ),
                field="confirmation_identity",
            ),
            "confirmation_result_envelope_sha256": _require_text(
                confirmation.get("result_envelope_sha256"),
                field="confirmation_result_envelope_sha256",
            ),
            "status": status,
        }
    )
    if pairing == CONFIRMATION_PAIRING_CROSS_MODEL:
        row["schema_version"] = CHECKER_VERDICT_CONFLICT_SCHEMA_VERSION_V3
        row["first_identity"] = adjudication_identity_fields(gate)
        row["confirmation_pairing"] = pairing
    return stamp_row_digest(row)


def _validate_accuracy_duplicate_accept_row(row: Mapping[str, Any]) -> None:
    """Re-derive every v2 consistency rule (validator half of the builder)."""

    if row.get("row_kind") != "duplicate-accept-confirmation":
        raise CheckerVerdictConflictError(
            f"checker-verdict-conflict v2 rows are duplicate-accept confirmations only, "
            f"got {row.get('row_kind')!r}"
        )
    kind = row.get("gate_kind")
    if kind not in ACCURACY_DUPLICATE_ACCEPT_GATE_KINDS:
        raise CheckerVerdictConflictError(
            f"a v2 duplicate-accept row settles {sorted(ACCURACY_DUPLICATE_ACCEPT_GATE_KINDS)} "
            f"acceptances only, got {kind!r} (terminal rows stay v1)"
        )
    for field in (
        "gate_id",
        "gate_identity",
        "confirmation_gate_identity",
        "question_sha256",
        "confirmation_result_envelope_sha256",
    ):
        _require_text(row.get(field), field=field)
    if row.get("confirmation_gate_identity") == row.get("gate_identity"):
        # P5 fable-F16 (validator half): a persisted v2 row whose confirmation
        # names the SAME gate identity self-witnesses and refuses at load.
        raise CheckerVerdictConflictError(
            "v2 duplicate-accept row self-witnesses: confirmation_gate_identity "
            f"equals gate_identity ({row.get('gate_identity')!r})"
        )
    _require_identity(row.get("confirmation_identity"), field="confirmation_identity")
    projections: dict[str, Mapping[str, Any]] = {}
    for field in ("first_projection", "confirmation_projection"):
        projection = row.get(field)
        expected_keys = (
            {"fidelity", "rung_check", "integrity"}
            if kind == "fidelity"
            else {"verdict", "rung_check", "integrity"}
        )
        if not isinstance(projection, Mapping) or set(projection) != expected_keys:
            raise CheckerVerdictConflictError(
                f"v2 duplicate-accept row {field} must carry exactly the {kind} full "
                f"verdict projection {sorted(expected_keys)}"
            )
        digest = row.get(f"{field}_sha256")
        if row_content_address(dict(projection)) != digest:
            raise CheckerVerdictConflictError(
                f"v2 duplicate-accept row {field}_sha256 does not bind its projection"
            )
        projections[field] = projection
    if not accuracy_projection_accepted(str(kind), projections["first_projection"]):
        raise CheckerVerdictConflictError(
            "v2 duplicate-accept rows exist only for acceptances; the first projection "
            "is not an accept"
        )
    derived = (
        "confirmed"
        if projections["first_projection"] == projections["confirmation_projection"]
        and accuracy_projection_accepted(str(kind), projections["confirmation_projection"])
        else "conflict"
    )
    if row.get("status") != derived:
        raise CheckerVerdictConflictError(
            f"v2 duplicate-accept status must re-derive to {derived!r} from the two "
            f"projections, got {row.get('status')!r}"
        )


_COMMON_ROW_FIELDS = (
    "stable_id",
    "work_id",
    "campaign_id",
    "run_id",
    "machine_id",
    "created_at",
    "gate_kind",
    "author_result_sha256",
)


def _validate_cross_model_duplicate_accept_row(row: Mapping[str, Any]) -> JsonObject:
    """Validate one v3 (cross-model) duplicate-accept row, fail-closed.

    The kind-specific half is the v1 (terminal) or v2 (accuracy) rule set
    re-applied verbatim; the v3 half re-derives the seat pairing from the two
    recorded identities against the LIVE seat table and requires it to be
    cross-model -- a v3 row whose identities are equal, or whose confirmation
    is not the seated partner, refuses on read exactly as at build.
    """

    if row.get("row_kind") != "duplicate-accept-confirmation":
        raise CheckerVerdictConflictError(
            "checker-verdict-conflict v3 rows are cross-model duplicate-accept "
            f"confirmations only, got {row.get('row_kind')!r}"
        )
    for field in _COMMON_ROW_FIELDS:
        _require_text(row.get(field), field=field)
    kind = str(row.get("gate_kind"))
    if kind in ACCURACY_DUPLICATE_ACCEPT_GATE_KINDS:
        _validate_accuracy_duplicate_accept_row(row)
    elif kind == "terminal_disposition":
        _validate_terminal_duplicate_accept_fields(row)
    else:
        raise CheckerVerdictConflictError(f"v3 duplicate-accept row names unknown gate kind {kind!r}")
    first = _require_identity(row.get("first_identity"), field="first_identity")
    second = _require_identity(row.get("confirmation_identity"), field="confirmation_identity")
    derived = confirmation_pairing(kind, first, second)
    if derived != CONFIRMATION_PAIRING_CROSS_MODEL or row.get("confirmation_pairing") != derived:
        raise CheckerVerdictConflictError(
            f"v3 duplicate-accept row confirmation_pairing must re-derive to "
            f"{CONFIRMATION_PAIRING_CROSS_MODEL!r} from its two identities, got "
            f"{row.get('confirmation_pairing')!r} (derived {derived!r})"
        )
    validated = dict(json.loads(canonical_json_bytes(row).decode("utf-8")))
    foreign = foreign_content_address_fields(validated, own="row_sha256")
    if foreign:
        raise CheckerVerdictConflictError(
            f"checker-verdict-conflict v3 row carries a foreign content-address field "
            f"{', '.join(foreign)}"
        )
    digest = validated.pop("row_sha256", None)
    if not isinstance(digest, str) or row_content_address(validated) != digest:
        raise CheckerVerdictConflictError(
            "checker-verdict-conflict row digest does not match its payload"
        )
    validated["row_sha256"] = digest
    return validated


def _validate_terminal_duplicate_accept_fields(row: Mapping[str, Any]) -> None:
    """Re-derive the v1 terminal duplicate-accept field rules (shared with v3)."""

    for field in (
        "gate_id",
        "gate_identity",
        "confirmation_gate_identity",
        "confirmation_result_envelope_sha256",
    ):
        _require_text(row.get(field), field=field)
    if row.get("confirmation_gate_identity") == row.get("gate_identity"):
        raise CheckerVerdictConflictError(
            "duplicate-accept row self-witnesses: confirmation_gate_identity equals "
            f"gate_identity ({row.get('gate_identity')!r})"
        )
    if row.get("first_verdict") != "accepted":
        raise CheckerVerdictConflictError(
            "duplicate-accept confirmation rows exist only for accepts; "
            f"first_verdict is {row.get('first_verdict')!r}"
        )
    verdict = _require_verdict(row.get("confirmation_verdict"), field="confirmation_verdict")
    _require_identity(row.get("confirmation_identity"), field="confirmation_identity")
    derived = "confirmed" if verdict == "accepted" else "conflict"
    if row.get("status") != derived:
        raise CheckerVerdictConflictError(
            f"duplicate-accept status must re-derive to {derived!r} from the "
            f"confirmation verdict {verdict!r}, got {row.get('status')!r}"
        )


def accuracy_duplicate_accept_state(
    gate_id: str, stable_id: str, rows: Iterable[Mapping[str, Any]]
) -> Optional[str]:
    """Return one accepted accuracy item's confirmation settlement, if any."""

    for row in rows:
        if (
            row.get("schema_version")
            in (CHECKER_VERDICT_CONFLICT_SCHEMA_VERSION_V2, CHECKER_VERDICT_CONFLICT_SCHEMA_VERSION_V3)
            and row.get("gate_kind") in ACCURACY_DUPLICATE_ACCEPT_GATE_KINDS
            and row.get("row_kind") == "duplicate-accept-confirmation"
            and row.get("gate_id") == gate_id
            and row.get("stable_id") == stable_id
        ):
            return str(row.get("status"))
    return None


def accuracy_acceptance_witnesses(
    gate: Mapping[str, Any],
    stable_id: str,
    persisted: Iterable[Mapping[str, Any]],
) -> list[JsonObject]:
    """Return distinct persisted accepts already witnessing one acceptance.

    The accuracy twin of :func:`terminal_acceptance_witnesses`: a persisted
    accepted row with a DIFFERENT ``gate_id`` answering the SAME question
    (:func:`accuracy_question_binding` equality, which binds the full
    adjudication identity) is already the second blind witness; buying a third
    session would be pure spend. Reused-marked rows are transparent, and the
    fresh row never witnesses itself.
    """

    kind = str(gate.get("gate_kind"))
    item = _accuracy_gate_item(gate, stable_id)
    if not accuracy_item_accepted(kind, item):
        return []
    question = accuracy_question_binding(gate, stable_id)
    gate_id = gate.get("gate_id")
    witnesses: list[JsonObject] = []
    for prior in persisted:
        if not isinstance(prior, Mapping) or prior.get("gate_kind") != kind:
            continue
        if prior.get("verdict_source") == "reused":
            continue
        if prior.get("gate_id") == gate_id:
            continue
        try:
            prior_item = _accuracy_gate_item(prior, stable_id)
        except CheckerVerdictConflictError:
            continue
        if not accuracy_item_accepted(kind, prior_item):
            continue
        if accuracy_question_binding(prior, stable_id) != question:
            continue
        witnesses.append(dict(prior))
    return witnesses


def validate_checker_verdict_conflict_row(row: Mapping[str, Any]) -> JsonObject:
    """Validate one conflict-ledger row and return its canonical copy.

    Re-derives every closed-vocabulary and consistency rule the builders
    enforce (kind membership, verdict vocabulary, identity-equality per kind,
    derived ``status``) plus the row digest, so a hand-edited or forged row
    refuses on read exactly as it would refuse at build. Mixed-era: v1 rows
    (terminal) keep v1 rules forever; v2 rows are the accuracy-lane
    duplicate-accept generalization.
    """

    if not isinstance(row, Mapping):
        raise CheckerVerdictConflictError("checker-verdict-conflict row must be a mapping")
    if row.get("schema_version") == CHECKER_VERDICT_CONFLICT_SCHEMA_VERSION_V3:
        return _validate_cross_model_duplicate_accept_row(row)
    if row.get("schema_version") == CHECKER_VERDICT_CONFLICT_SCHEMA_VERSION_V2:
        common_fields = (
            "stable_id",
            "work_id",
            "campaign_id",
            "run_id",
            "machine_id",
            "created_at",
            "gate_kind",
            "author_result_sha256",
        )
        for field in common_fields:
            _require_text(row.get(field), field=field)
        _validate_accuracy_duplicate_accept_row(row)
        validated = dict(json.loads(canonical_json_bytes(row).decode("utf-8")))
        foreign = foreign_content_address_fields(validated, own="row_sha256")
        if foreign:
            raise CheckerVerdictConflictError(
                f"checker-verdict-conflict v2 row carries a foreign content-address "
                f"field {', '.join(foreign)}"
            )
        digest = validated.pop("row_sha256", None)
        if not isinstance(digest, str) or row_content_address(validated) != digest:
            raise CheckerVerdictConflictError(
                "checker-verdict-conflict row digest does not match its payload"
            )
        validated["row_sha256"] = digest
        return validated
    if row.get("schema_version") != CHECKER_VERDICT_CONFLICT_SCHEMA_VERSION:
        raise CheckerVerdictConflictError(
            f"checker-verdict-conflict row declares unknown schema {row.get('schema_version')!r}"
        )
    row_kind = row.get("row_kind")
    if row_kind not in CONFLICT_ROW_KINDS:
        raise CheckerVerdictConflictError(
            f"checker-verdict-conflict row names unknown kind {row_kind!r}"
        )
    common_fields = (
        "stable_id",
        "work_id",
        "campaign_id",
        "run_id",
        "machine_id",
        "created_at",
        "gate_kind",
        "author_result_sha256",
    )
    for field in common_fields:
        _require_text(row.get(field), field=field)
    if row_kind == "duplicate-accept-confirmation":
        for field in (
            "gate_id",
            "gate_identity",
            "confirmation_gate_identity",
            "confirmation_result_envelope_sha256",
        ):
            _require_text(row.get(field), field=field)
        if row.get("first_verdict") != "accepted":
            raise CheckerVerdictConflictError(
                "duplicate-accept confirmation rows exist only for accepts; "
                f"first_verdict is {row.get('first_verdict')!r}"
            )
        verdict = _require_verdict(row.get("confirmation_verdict"), field="confirmation_verdict")
        _require_identity(row.get("confirmation_identity"), field="confirmation_identity")
        derived = "confirmed" if verdict == "accepted" else "conflict"
        if row.get("status") != derived:
            raise CheckerVerdictConflictError(
                f"duplicate-accept status must re-derive to {derived!r} from the "
                f"confirmation verdict {verdict!r}, got {row.get('status')!r}"
            )
    else:
        for field in ("new_gate_id", "prior_gate_id"):
            _require_text(row.get(field), field=field)
        verdict_new = _require_verdict(row.get("new_verdict"), field="new_verdict")
        verdict_prior = _require_verdict(row.get("prior_verdict"), field="prior_verdict")
        if verdict_new == verdict_prior:
            raise CheckerVerdictConflictError(
                f"a {row_kind} row records a disagreement; both verdicts are equal"
            )
        new_bound = _require_identity(row.get("new_identity"), field="new_identity")
        prior_bound = _require_identity(row.get("prior_identity"), field="prior_identity")
        if (new_bound == prior_bound) != (row_kind == "verdict-conflict"):
            raise CheckerVerdictConflictError(
                f"{row_kind} row identity topology is cross-wired: same-identity "
                "disagreement is verdict-conflict, cross-identity is "
                "identity-attributed-flip"
            )
    validated = dict(json.loads(canonical_json_bytes(row).decode("utf-8")))
    foreign = foreign_content_address_fields(validated, own="row_sha256")
    if foreign:
        raise CheckerVerdictConflictError(
            f"checker-verdict-conflict row carries a foreign content-address field "
            f"{', '.join(foreign)}: a row of this family owns exactly one content address"
        )
    digest = validated.pop("row_sha256", None)
    if not isinstance(digest, str) or row_content_address(validated) != digest:
        raise CheckerVerdictConflictError(
            "checker-verdict-conflict row digest does not match its payload"
        )
    validated["row_sha256"] = digest
    return validated


def _row_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the idempotency key: one settlement per adjudication fact."""

    if row["row_kind"] == "duplicate-accept-confirmation":
        if row.get("schema_version") == CHECKER_VERDICT_CONFLICT_SCHEMA_VERSION_V2 or (
            row.get("schema_version") == CHECKER_VERDICT_CONFLICT_SCHEMA_VERSION_V3
            and row.get("gate_kind") in ACCURACY_DUPLICATE_ACCEPT_GATE_KINDS
        ):
            # A metadata gate row can carry several members; the settled fact
            # is per accepted ITEM.
            return ("duplicate-accept-confirmation", str(row["gate_id"]), str(row["stable_id"]))
        return ("duplicate-accept-confirmation", str(row["gate_id"]))
    return (str(row["row_kind"]), str(row["new_gate_id"]), str(row["prior_gate_id"]))


def _decode_conflict_lines(payload: bytes, path: Path) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    for index, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CheckerVerdictConflictError(
                f"{path}:{index} is not a valid checker-verdict-conflict row: {exc}"
            ) from exc
        rows.append(validate_checker_verdict_conflict_row(decoded))
    return tuple(rows)


def _ledger_bytes(path: Path, handle: Optional[BinaryIO] = None) -> bytes:
    """Read the whole ledger, recovering only a torn final line with evidence."""

    return read_with_tail_recovery(
        path,
        prefix_check=lambda retained: _decode_conflict_lines(retained.read_bytes(), path),
        locked_handle=handle,
    )


def append_checker_verdict_conflict_row(path: Path, row: Mapping[str, Any]) -> JsonObject:
    """Append one conflict-ledger row durably and idempotently.

    The FIRST settlement recorded for an adjudication fact is the record: a
    resume that re-derives the same confirmation or the same conflict pair is
    a no-op returning the persisted row (``created_at``/``run_id`` may differ
    on every resume, so equality on the fact key -- not the bytes -- decides).
    """

    validated = validate_checker_verdict_conflict_row(row)
    key = _row_key(validated)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        existing = _decode_conflict_lines(_ledger_bytes(path, handle), path)
        matching = [item for item in existing if _row_key(item) == key]
        if matching:
            return matching[0]
        handle.seek(0, os.SEEK_END)
        handle.write(canonical_json_bytes(validated) + b"\n")
        handle.flush()
        durable_fsync(handle.fileno())
    fsync_directory(path.parent)
    return validated


def load_checker_verdict_conflict_rows(paths: Sequence[Path]) -> tuple[JsonObject, ...]:
    """Load validated conflict-ledger rows in ledger order.

    Parameters
    ----------
    paths:
        Candidate ledger paths. Missing files are empty ledgers.
    """

    rows: list[JsonObject] = []
    for path in paths:
        if not path.is_file():
            continue
        rows.extend(_decode_conflict_lines(_ledger_bytes(path), path))
    return tuple(rows)


def duplicate_accept_state(gate_id: str, rows: Iterable[Mapping[str, Any]]) -> Optional[str]:
    """Return one accepted gate row's confirmation settlement, if any.

    Parameters
    ----------
    gate_id:
        The accepted gates-ledger row under confirmation.
    rows:
        Conflict-ledger rows (:func:`load_checker_verdict_conflict_rows`).

    Returns
    -------
    str | None
        ``"confirmed"`` / ``"conflict"`` from the first settlement row, or
        ``None`` when the acceptance is still unconfirmed.
    """

    for row in rows:
        if (
            row.get("row_kind") == "duplicate-accept-confirmation"
            and row.get("gate_id") == gate_id
            and row.get("gate_kind") not in ACCURACY_DUPLICATE_ACCEPT_GATE_KINDS
        ):
            return str(row.get("status"))
    return None


# ---------------------------------------------------------------------------
# Comparison kernels over persisted gate rows.
# ---------------------------------------------------------------------------


def _terminal_item(gate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the single TD item of one gate row, fail-closed."""

    items = gate.get("items")
    if not isinstance(items, Sequence) or len(items) != 1 or not isinstance(items[0], Mapping):
        raise CheckerVerdictConflictError(
            "a terminal-disposition gate row carries exactly one item"
        )
    disposition = items[0].get("terminal_disposition")
    if not isinstance(disposition, Mapping):
        raise CheckerVerdictConflictError(
            "a terminal-disposition gate item carries a terminal_disposition block"
        )
    return items[0]


def terminal_question_binding(gate: Mapping[str, Any]) -> dict[str, Any]:
    """Return the semantic adjudication QUESTION one TD gate row answered.

    Blindness in the duplicate-accept policy is replication of the SAME
    question -- item, content, and full adjudication identity -- not of the
    envelope bytes: every dispatch mints a fresh deadline and output path, so
    ``gate_identity`` lawfully differs across the two sessions. Two gates with
    equal bindings adjudicated the same question.
    """

    item = _terminal_item(gate)
    disposition = item["terminal_disposition"]
    return {
        "stable_id": item.get("stable_id"),
        "work_id": item.get("work_id"),
        "author_result_sha256": disposition.get("author_result_sha256"),
        "source_manifest_identity": disposition.get("source_manifest_identity"),
        "evidence_identity": disposition.get("evidence_identity"),
        "license_identity": disposition.get("license_identity"),
        "adjudication_identity": gate_adjudication_identity(gate),
    }


def adjudication_identity_fields(gate: Mapping[str, Any]) -> dict[str, str]:
    """Return one gate row's adjudication identity as builder-ready fields.

    Components fail closed to ``""`` exactly as
    :func:`~menagerie.crawler.checker_dispatch.gate_adjudication_identity`
    does; the row builders then refuse empty components, so a malformed
    checker block can never mint a half-identified disclosure row.
    """

    provider, model, version, prompt_sha256 = gate_adjudication_identity(gate)
    return {
        "provider": provider,
        "model": model,
        "version": version,
        "prompt_sha256": prompt_sha256,
    }


def terminal_gate_verdict(gate: Mapping[str, Any]) -> str:
    """Return one TD gate row's verdict under the closed vocabulary."""

    disposition = _terminal_item(gate)["terminal_disposition"]
    return _require_verdict(disposition.get("verdict"), field="terminal_disposition.verdict")


def terminal_verdict_disagreements(
    new_gate: Mapping[str, Any],
    persisted: Iterable[Mapping[str, Any]],
) -> dict[str, list[JsonObject]]:
    """Split persisted disagreeing verdicts into conflicts and lawful flips.

    Compares the fresh row against every persisted terminal-disposition row
    adjudicating the same ``(stable_id, author_result_sha256)``. Reused-marked
    rows are transparent (they are replays, never verdict sources), malformed
    history rows are skipped (they cannot testify), and the fresh row itself
    (matched by ``gate_id``) is excluded.

    Returns
    -------
    dict[str, list[dict]]
        ``{"conflicts": [...], "flips": [...]}`` -- prior rows whose verdict
        disagrees under the same adjudication identity (conflicts) or under a
        different one (flips), in ledger order.
    """

    item = _terminal_item(new_gate)
    new_verdict = terminal_gate_verdict(new_gate)
    new_identity = gate_adjudication_identity(new_gate)
    stable_id = item.get("stable_id")
    content = item["terminal_disposition"].get("author_result_sha256")
    new_gate_id = new_gate.get("gate_id")
    conflicts: list[JsonObject] = []
    flips: list[JsonObject] = []
    for prior in persisted:
        if not isinstance(prior, Mapping):
            continue
        if prior.get("gate_kind") != "terminal_disposition":
            continue
        if prior.get("verdict_source") == "reused":
            continue
        if new_gate_id is not None and prior.get("gate_id") == new_gate_id:
            continue
        try:
            prior_item = _terminal_item(prior)
            prior_verdict = terminal_gate_verdict(prior)
        except CheckerVerdictConflictError:
            continue
        if prior_item.get("stable_id") != stable_id:
            continue
        if prior_item["terminal_disposition"].get("author_result_sha256") != content:
            continue
        if prior_verdict == new_verdict:
            continue
        if gate_adjudication_identity(prior) == new_identity:
            conflicts.append(dict(prior))
        else:
            flips.append(dict(prior))
    return {"conflicts": conflicts, "flips": flips}


def terminal_acceptance_witnesses(
    new_gate: Mapping[str, Any],
    persisted: Iterable[Mapping[str, Any]],
) -> list[JsonObject]:
    """Return distinct persisted accepts that already witness one acceptance.

    The duplicate-accept policy binds an acceptance FACT: ``(stable_id,
    author_result_sha256, gate_identity, adjudication identity)``. Two blind
    consistent accepts of that fact = two independent sessions over the
    byte-identical envelope under the same identity. A persisted accept row
    with a DIFFERENT ``gate_id`` (a separate session -- e.g. a review-sweep
    re-gate of a prior accept) is therefore already the second blind witness,
    and buying a third session would be pure spend. Reused-marked rows are
    transparent (a replay is not an independent session), and the fresh row
    itself never witnesses its own acceptance.
    """

    item = _terminal_item(new_gate)
    if terminal_gate_verdict(new_gate) != "accepted":
        return []
    identity = gate_adjudication_identity(new_gate)
    stable_id = item.get("stable_id")
    content = item["terminal_disposition"].get("author_result_sha256")
    envelope = new_gate.get("gate_identity")
    gate_id = new_gate.get("gate_id")
    witnesses: list[JsonObject] = []
    for prior in persisted:
        if not isinstance(prior, Mapping) or prior.get("gate_kind") != "terminal_disposition":
            continue
        if prior.get("verdict_source") == "reused":
            continue
        if prior.get("gate_id") == gate_id:
            continue
        try:
            prior_item = _terminal_item(prior)
            prior_verdict = terminal_gate_verdict(prior)
        except CheckerVerdictConflictError:
            continue
        if prior_verdict != "accepted":
            continue
        if prior_item.get("stable_id") != stable_id:
            continue
        if prior_item["terminal_disposition"].get("author_result_sha256") != content:
            continue
        if prior.get("gate_identity") != envelope:
            continue
        if gate_adjudication_identity(prior) != identity:
            continue
        witnesses.append(dict(prior))
    return witnesses


def unconfirmed_accepted_terminal_gate(
    gate_records: Iterable[Mapping[str, Any]],
    conflict_rows: Iterable[Mapping[str, Any]],
    *,
    stable_id: str,
    work_id: str,
    author_result_sha256: str,
    checker_identity: Optional[Any],
) -> Optional[JsonObject]:
    """Return the newest accepted row whose blind confirmation never settled.

    The duplicate-accept resume guard: the crash window between an accept's
    gate append and its confirmation settlement must resume the CONFIRMATION,
    never buy a duplicate first session (a second accepted row for the same
    item bytes would violate the exact-membership rule at publication). The
    probe binds identity, work generation, and content; ``None`` identity
    fails open to a live buy (same posture as retired reuse), and a row with
    any settlement -- confirmed, conflict, or a same-identity conflict row
    naming it -- never resumes.
    """

    if checker_identity is None:
        return None
    identity = (
        checker_identity.provider,
        checker_identity.model,
        checker_identity.version,
        checker_identity.prompt_sha256,
    )
    rows = tuple(conflict_rows)
    records = list(gate_records)
    for gate in reversed(records):
        if not isinstance(gate, Mapping) or gate.get("gate_kind") != "terminal_disposition":
            continue
        if gate.get("verdict_source") == "reused":
            continue
        try:
            item = _terminal_item(gate)
            verdict = terminal_gate_verdict(gate)
        except CheckerVerdictConflictError:
            continue
        if verdict != "accepted":
            continue
        if item.get("stable_id") != stable_id or item.get("work_id") != work_id:
            continue
        if item["terminal_disposition"].get("author_result_sha256") != author_result_sha256:
            continue
        if gate_adjudication_identity(gate) != identity:
            continue
        gate_id = str(gate.get("gate_id"))
        if duplicate_accept_state(gate_id, rows) is not None:
            continue
        if any(
            row.get("row_kind") == "verdict-conflict" and row.get("new_gate_id") == gate_id
            for row in rows
        ):
            continue
        if terminal_acceptance_witnesses(gate, records):
            # A distinct consistent accept session already witnesses this
            # acceptance: the policy is satisfied, nothing to resume.
            continue
        if not confirmation_sampled(gate, stable_id=stable_id):
            # RULING checker row: an accept the deterministic 10 percent
            # terminal sample did not select owes no confirmation, so a resume
            # never buys one for it (the predicate re-derives identically).
            continue
        return dict(gate)
    return None
