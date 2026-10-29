"""Tool-minted checker calibration receipts: measure, mint, recompute, refuse.

W2 (PLAN_v6 §2 WS1, Wave-2 rotation batch). The calibration receipt is NEVER
hand-written: this module derives every measured number from the campaign's
own ledgers -- the gates ledger (rendered-verdict sessions and their walls),
the paid-session spend ledger (the bought-session denominator, sol P2-12
denominator honesty), and the operator wrapper telemetry (completed-attempt
latency tails and output tokens; the 600s-lesson evidence) -- plus the
blind-replay matrix report (B1-XV2, W2's acceptance instrument). The VALIDATOR
recomputes every measured field from the same rows and refuses divergence
(fable C2): a receipt that does not re-derive is not evidence.

Ops entry point: ``python -m menagerie.crawler.tools.calibration_receipt``
(mint/verify). The structural loader and activation gate stay in
:mod:`menagerie.crawler.checker_calibration`; minting ends with that loader as
its own self-check, so an under-measured or timing-incompatible receipt can
never be written "for later".
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.checker_calibration import (
    ACCEPTANCE_DISAGREEMENT_OUTCOME,
    ACCEPTANCE_UNFUNDED_OUTCOME,
    ACCEPTANCE_WITNESS_MINIMUM,
    CALIBRATION_PRODUCTION_BATCH_WIDTH,
    CHECKER_CALIBRATION_RECEIPT_SCHEMA,
    CONTESTED_CENSUS_ARM,
    LOCATOR_ATTESTED_BY,
    POLE_CLASS_CONTRADICTION,
    POLE_CLASS_ORDERED_ABSTENTION,
    CheckerCalibrationInvalidError,
    CheckerGoldenCorpusError,
    live_checker_prompt_identity,
    load_checker_calibration_receipt,
)
from menagerie.crawler.constants import GateKind
from menagerie.crawler.evidence import (
    LOCATOR_ATTESTATION_INVALID,
    LOCATOR_ATTESTATION_UNATTESTABLE,
    TEXT_LINE_LOCATOR_BASIS,
)
from menagerie.crawler.identity import stable_hash
from menagerie.crawler.models import JsonObject
from menagerie.crawler.paid_session_lifecycle import load_paid_session_rows
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.tools.checker_latency import iter_attempt_records, quantile

#: The pre-rule-2 report schema: no per-draw rows, no attestation, no
#: mechanical-claim inventory. Zero v2 receipts can be derived from it, so it
#: refuses typed at load.
BLIND_REPLAY_REPORT_SCHEMA_V1 = "menagerie.crawler.blind-replay-report.v1"
BLIND_REPLAY_REPORT_SCHEMA = "menagerie.crawler.blind-replay-report.v2"

#: The committed pole-ordering table (Opus 6.2): a disagreement whose poles
#: span a positive AND a negative verdict is a CONTRADICTION (one draw asserts
#: a positive falsehood -- root-cause it, never average it); every other pole
#: pair differs only in blessing strength and is ORDERED-ABSTENTION.
_POSITIVE_POLES = frozenset({"accurate", "match", "minor-drift", "accepted"})
_NEGATIVE_POLES = frozenset({"inaccurate", "major-drift", "slop", "rejected"})


class CheckerCalibrationDivergenceError(CheckerCalibrationInvalidError):
    """A receipt's measured numbers do not re-derive from the same ledgers."""


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_rfc3339(value: Any) -> Optional[float]:
    """Return one RFC3339 ``Z`` timestamp as epoch seconds, or ``None``."""

    if not isinstance(value, str) or not value:
        return None
    from datetime import datetime, timezone

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _paid_ledger_genesis(spend_rows: Sequence[Mapping[str, Any]]) -> Optional[float]:
    """Return the paid ledger's genesis anchor as epoch seconds, or ``None``.

    A ledger whose first row is an ``accounting_rollover`` or
    ``accounting_genesis`` row begins its accounting AT that row's
    ``created_at``: every session rendered before that instant settled on a
    predecessor ledger (the rolled-over poisoned sibling or the legacy
    prefix) and can have no rows here by construction. The anchor is
    replay-derived from the ledger itself on every call -- never a
    caller-supplied window. A ledger with no anchor row covers all time.
    """

    if not spend_rows:
        return None
    first = spend_rows[0]
    if first.get("event") not in ("accounting_rollover", "accounting_genesis"):
        return None
    anchor = _parse_rfc3339(first.get("created_at"))
    if anchor is None:
        raise CheckerCalibrationInvalidError(
            "paid-session ledger genesis row carries no parseable created_at; an "
            "anchor row without a timestamp cannot define the measurement window"
        )
    return anchor


def _gates_ledger_path(records_root: Path) -> Path:
    """Resolve the gates ledger below one records root (shard, else flat)."""

    shard = records_root / "gates" / "current-shard.jsonl"
    if shard.is_file():
        return shard
    return records_root / "gates.jsonl"


def gather_lane_measurements(
    records_root: Path,
    *,
    work_roots: Sequence[Path],
    checker_model: str,
) -> dict[str, JsonObject]:
    """Derive every per-lane measured field from the campaign's own ledgers.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory (gates + spend ledgers).
    work_roots:
        Roots searched recursively for ``operator-telemetry.jsonl`` (the
        ``.crawl-local/work`` tree lives OUTSIDE the records root).
    checker_model:
        The pinned model; rows recorded under any other model are excluded
        (a receipt measures exactly one adjudication authority).

    The rendered/bought measurement window is epoch-scoped to the paid
    ledger's genesis (:func:`_paid_ledger_genesis`): gate rows rendered
    before a rollover/cutover anchor settled on a predecessor ledger, so
    they are disclosed per lane as ``pre_genesis_rendered`` and never
    counted against this ledger's bought denominator.

    Returns
    -------
    dict[str, dict]
        One measurement mapping per :class:`GateKind` value, in the exact
        per-lane receipt field vocabulary.

    Raises
    ------
    CheckerCalibrationInvalidError
        On a ledger inconsistency the receipt must never paper over (more
        rendered verdicts than bought sessions, or an empty token sample).
    """

    spend_path = records_root / "operational" / "spend.jsonl"
    spend_rows = load_paid_session_rows(spend_path) if spend_path.is_file() else ()
    genesis = _paid_ledger_genesis(spend_rows)

    gates_path = _gates_ledger_path(records_root)
    gate_rows = list(scan_jsonl(gates_path)) if gates_path.is_file() else []
    walls: dict[str, list[float]] = {kind.value: [] for kind in GateKind}
    rendered: dict[str, int] = {kind.value: 0 for kind in GateKind}
    pre_genesis: dict[str, int] = {kind.value: 0 for kind in GateKind}
    for row in gate_rows:
        kind = row.get("gate_kind")
        if kind not in rendered:
            continue
        if row.get("verdict_source") == "reused":
            # A replay is not a session; counting its wall would double-charge
            # the one session the campaign actually bought.
            continue
        checker = row.get("checker")
        checker = checker if isinstance(checker, Mapping) else {}
        if checker.get("model") != checker_model:
            continue
        if kind == GateKind.METADATA_BATCH.value:
            width = row.get("batch_size")
            if width != CALIBRATION_PRODUCTION_BATCH_WIDTH:
                # The metadata lane is calibrated AT production width only
                # (fable U-F2/C8); other widths measured a different lane.
                continue
        started = _parse_rfc3339(checker.get("started_at"))
        finished = _parse_rfc3339(checker.get("finished_at"))
        stamp = started if started is not None else finished
        if genesis is not None and stamp is not None and stamp < genesis:
            # The measurement window is epoch-scoped to the paid ledger's
            # genesis: a verdict rendered before the anchor settled on a
            # predecessor ledger and has no bought row HERE by construction.
            # It is disclosed, never counted. An undated row falls through
            # and counts in-window -- it can only trigger the inconsistency
            # refusal below, never escape it through the epoch.
            pre_genesis[kind] += 1
            continue
        rendered[kind] += 1
        if started is not None and finished is not None and finished >= started:
            walls[kind].append(finished - started)

    reserved_scope: dict[str, tuple[Optional[str], Optional[str], Optional[int]]] = {}
    for row in spend_rows:
        if row.get("event") != "session_reserved" or row.get("lane") != "checker":
            continue
        identity = row.get("checker_identity")
        model = identity.get("model") if isinstance(identity, Mapping) else None
        axis = row.get("axis")
        members = axis.get("work_generations") if isinstance(axis, Mapping) else None
        width = len(members) if isinstance(members, list) and members else None
        reserved_scope[str(row.get("spend_id"))] = (row.get("gate_kind"), model, width)
    bought: dict[str, int] = {kind.value: 0 for kind in GateKind}
    for row in spend_rows:
        if row.get("event") != "session_settled":
            continue
        gate_kind, model, width = reserved_scope.get(str(row.get("spend_id")), (None, None, None))
        if gate_kind not in bought:
            continue
        if model is not None and model != checker_model:
            continue
        if (
            gate_kind == GateKind.METADATA_BATCH.value
            and width is not None
            and width != CALIBRATION_PRODUCTION_BATCH_WIDTH
        ):
            # The bought-session denominator is scoped exactly like the rendered
            # numerator (fable U-F2/C8): a session reserved at another batch
            # width bought a different lane's measurement. A reservation whose
            # width cannot be derived still COUNTS -- stripping the member list
            # can never hide a dead session from the refusal rate.
            continue
        bought[gate_kind] += 1

    attempts: dict[str, dict[str, list[float]]] = {
        kind.value: {"completed": [], "censored": [], "tokens": []} for kind in GateKind
    }
    for record in iter_attempt_records(list(work_roots)):
        kind = record.get("gate_kind")
        if kind not in attempts:
            continue
        if kind == GateKind.METADATA_BATCH.value:
            batch_size = record.get("batch_size")
            if (
                isinstance(batch_size, int)
                and not isinstance(batch_size, bool)
                and batch_size != CALIBRATION_PRODUCTION_BATCH_WIDTH
            ):
                # Attempt-tail evidence is width-scoped like every other
                # metadata measurement (fable U-F2/C8). A record with no
                # readable batch_size still pools in: an unknown-width sample
                # can only lengthen the reported tail, never shorten it.
                continue
        duration = record.get("duration_seconds")
        if not isinstance(duration, (int, float)) or isinstance(duration, bool):
            continue
        bucket = attempts[kind]
        if bool(record.get("timed_out")):
            bucket["censored"].append(float(duration))
            continue
        bucket["completed"].append(float(duration))
        tokens = record.get("output_tokens")
        if isinstance(tokens, (int, float)) and not isinstance(tokens, bool):
            bucket["tokens"].append(float(tokens))

    measurements: dict[str, JsonObject] = {}
    for kind in GateKind:
        lane = kind.value
        sessions = rendered[lane]
        sessions_bought = bought[lane]
        if sessions > sessions_bought:
            raise CheckerCalibrationInvalidError(
                f"calibration measurement for lane {lane}: {sessions} rendered verdict "
                f"sessions but only {sessions_bought} settled bought sessions on the "
                "spend ledger; a denominator smaller than its numerator is a ledger "
                "inconsistency the receipt must never paper over"
            )
        completed = sorted(attempts[lane]["completed"])
        censored = attempts[lane]["censored"]
        timed = len(completed) + len(censored)
        tokens = sorted(attempts[lane]["tokens"])
        wall_samples = sorted(walls[lane])
        measurements[lane] = {
            "sessions": sessions,
            "sessions_bought": sessions_bought,
            "pre_genesis_rendered": pre_genesis[lane],
            "latency_p95_seconds": quantile(wall_samples, 0.95),
            "latency_p99_seconds": quantile(wall_samples, 0.99),
            "output_tokens_p99": quantile(tokens, 0.99),
            "refusal_rate": (
                (sessions_bought - sessions) / sessions_bought if sessions_bought else 0.0
            ),
            "attempt_latency_p99_seconds": quantile(completed, 0.99),
            "attempt_samples": len(completed),
            "attempt_censoring_rate": (len(censored) / timed) if timed else 0.0,
        }
    return measurements


def load_blind_replay_report(path: Path) -> JsonObject:
    """Load and structurally validate one blind-replay matrix report."""

    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckerCalibrationInvalidError(
            f"blind-replay report {path} is unreadable: {exc}"
        ) from exc
    if isinstance(report, dict) and report.get("schema_version") == BLIND_REPLAY_REPORT_SCHEMA_V1:
        raise CheckerCalibrationInvalidError(
            f"blind-replay report {path} declares the retired v1 report schema; a v1 "
            "report carries no per-draw rows, no machine locator attestation, and no "
            "mechanical-claim inventory, so no v2 receipt can be derived from it -- "
            "re-run the matrix under the current epoch"
        )
    if not isinstance(report, dict) or report.get("schema_version") != BLIND_REPLAY_REPORT_SCHEMA:
        raise CheckerCalibrationInvalidError(
            f"blind-replay report {path} does not declare {BLIND_REPLAY_REPORT_SCHEMA}"
        )
    goldens = report.get("goldens")
    if not isinstance(goldens, list) or not goldens:
        raise CheckerCalibrationInvalidError(
            f"blind-replay report {path} carries no per-golden rows; an empty matrix "
            "attests nothing"
        )
    return report


def _projection_fields(value: Any) -> Mapping[str, Any]:
    """Coerce one projection item to a field mapping.

    The harness contract keeps projections opaque JSON; the production
    ``verdict_projection`` emits per-field mappings, and a scripted flat
    value reads as its own single ``verdict`` field.
    """

    return value if isinstance(value, Mapping) else {"verdict": value}


def _projection_item_accepted(kind: str, fields: Mapping[str, Any]) -> bool:
    """Acceptance predicate over one verdict-projection item (per gate kind).

    Mirrors ``constants.metadata_item_accepted`` / ``fidelity_item_accepted``
    at the projection level (the projection carries the component verdict
    strings, not the full blocks). Fail-closed: anything unreadable is not an
    acceptance.
    """

    if kind == GateKind.FIDELITY.value:
        return fields.get("fidelity") in {"match", "minor-drift"} and (
            fields.get("rung_check") == "accurate"
        )
    if kind == GateKind.TERMINAL_DISPOSITION.value:
        return fields.get("terminal_disposition") == "accepted"
    return all(fields.get(key) == "accurate" for key in ("verdict", "integrity", "rung_check"))


def _pole_class(poles: Sequence[str]) -> str:
    """Classify one disagreement's poles against the committed ordering table."""

    values = set(poles)
    if values & _POSITIVE_POLES and values & _NEGATIVE_POLES:
        return POLE_CLASS_CONTRADICTION
    return POLE_CLASS_ORDERED_ABSTENTION


def _attestation_keyset_sha256(rows_by_golden: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    """Digest the sorted per-excerpt attestation keyset (fable A-3).

    The keyset binds WHICH excerpts were checked, not just how many, so a
    producer cannot satisfy the count by omitting or swapping members. The
    verifier recomputes this digest from the corpus bytes.
    """

    keyset = sorted(
        [
            str(golden),
            str(row.get("evidence_id")),
            str(row.get("source_content_sha256")),
            str(row.get("excerpt_text_sha256")),
            str(row.get("form")),
            str(row.get("start_line")),
            str(row.get("end_line")),
            str(row.get("status")),
        ]
        for golden, rows in rows_by_golden.items()
        for row in rows
    )
    return stable_hash(keyset)


def derive_locator_attestation_block(
    rows_by_golden: Mapping[str, Sequence[Mapping[str, Any]]],
) -> JsonObject:
    """Re-derive the receipt's ``locator_attestation`` zero-bar block."""

    all_rows = [row for rows in rows_by_golden.values() for row in rows]
    return {
        "checks": len(all_rows),
        "failures": sum(
            1 for row in all_rows if row.get("status") == LOCATOR_ATTESTATION_INVALID
        ),
        "unattestable": sum(
            1 for row in all_rows if row.get("status") == LOCATOR_ATTESTATION_UNATTESTABLE
        ),
        "basis": TEXT_LINE_LOCATOR_BASIS,
        "attested_by": LOCATOR_ATTESTED_BY,
        "keyset_sha256": _attestation_keyset_sha256(rows_by_golden),
    }


def derive_corpus_locator_attestation(corpus_dir: Path) -> dict[str, list[JsonObject]]:
    """Recompute every golden's machine locator attestation from corpus bytes.

    The verifier-side half of fable A-3: the receipt's attestation block must
    re-derive from the committed corpus itself, never only from the report's
    producer-written rows. Uses the SAME derivation the envelope builder
    freezes at dispatch, so the two spellings cannot drift.
    """

    from menagerie.crawler.checker_dispatch import _derive_locator_attestation
    from menagerie.crawler.tools.blind_replay_matrix import (
        golden_item_excerpts,
        load_golden_corpus_entries,
    )

    rows_by_golden: dict[str, list[JsonObject]] = {}
    for entry in load_golden_corpus_entries(corpus_dir):
        request = entry.get("request") or {}
        rows: list[JsonObject] = []
        for item in request.get("items") or []:
            attestation = _derive_locator_attestation(
                golden_item_excerpts(str(entry.get("kind")), item),
                item.get("source_manifest"),
                model_dir=item.get("model_dir"),
            )
            rows.extend(attestation["rows"])
        rows_by_golden[str(entry["name"])] = rows
    return rows_by_golden


def derive_blind_replay_block(report: Mapping[str, Any], *, report_path: Path) -> JsonObject:
    """Re-derive the receipt's ``blind_replay`` block from per-golden raw rows.

    The report's per-golden rows are the facts; every aggregate here --
    including the rule-2 v2 blocks (disagreement diagnosis, mechanical-claim
    inventory, corpus locator attestation, acceptance witnesses) -- is
    recomputed from them on every call (never copied from a producer-written
    summary), and ``report_sha256`` binds the exact artifact bytes.
    """

    goldens = report["goldens"]
    replays: list[int] = []
    same_epoch_disagreements = 0
    rotation_ran_everywhere = True
    rotation_disagreements = 0
    widths: set[int] = set()
    width_disagreements = 0
    production_width = int(report.get("production_batch_width") or 0)
    real_second_width_seen = False
    diagnosis: list[JsonObject] = []
    claims_required = 0
    claims_discharged = 0
    claims_gaps = 0
    attestation_rows: dict[str, list[Mapping[str, Any]]] = {}
    accept_witnesses = {kind.value: 0 for kind in GateKind}
    missing_confirmations = 0
    confirmation_disagreements = 0
    conflicts = report.get("authority_conflicts")
    conflicts = conflicts if isinstance(conflicts, list) else []
    refuted_by_golden: dict[str, int] = {}
    for conflict in conflicts:
        if isinstance(conflict, Mapping) and conflict.get("mode") != "rotation":
            name = str(conflict.get("golden"))
            refuted_by_golden[name] = refuted_by_golden.get(name, 0) + 1
    for row in goldens:
        if not isinstance(row, Mapping):
            raise CheckerCalibrationInvalidError("blind-replay report golden rows must be mappings")
        name = str(row.get("name"))
        kind = str(row.get("kind"))
        same_epoch = row.get("same_epoch")
        if not isinstance(same_epoch, Mapping):
            raise CheckerCalibrationInvalidError(
                f"blind-replay golden {name!r} carries no same_epoch block"
            )
        projections = same_epoch.get("projections")
        if not isinstance(projections, list) or not projections:
            raise CheckerCalibrationInvalidError(
                f"blind-replay golden {name!r} carries no same-epoch projections"
            )
        replays.append(int(same_epoch.get("replays", 0)))
        row_disagrees = any(
            projection != projections[0] for projection in projections[1:]
        )
        same_epoch_disagreements += int(row_disagrees)
        # Diagnosis rows derive from the projections, never from a producer
        # summary; usually empty because any row here already refuses the mint
        # through the unchanged same-epoch zero bar.
        for stable_id in sorted(
            {key for projection in projections for key in projection}
        ):
            fields_by_draw = [
                _projection_fields(projection.get(stable_id)) for projection in projections
            ]
            for field in sorted({key for fields in fields_by_draw for key in fields}):
                poles = sorted(
                    {str(fields.get(field)) for fields in fields_by_draw}
                )
                if len(poles) > 1:
                    diagnosis.append(
                        {
                            "golden": name,
                            "item": stable_id,
                            "field": field,
                            "poles": poles,
                            "pole_class": _pole_class(poles),
                            "mechanical_refutation": (
                                "refuted" if refuted_by_golden.get(name) else "unrefuted"
                            ),
                            "defect_class": None,
                        }
                    )
        rotation = row.get("rotation")
        if not isinstance(rotation, Mapping) or rotation.get("ran") is not True:
            rotation_ran_everywhere = False
        else:
            rotation_disagreements += int(bool(rotation.get("disagreement")))
        invariance = row.get("width_invariance")
        if isinstance(invariance, Mapping):
            row_widths = invariance.get("widths")
            row_width_set = (
                {int(value) for value in row_widths}
                if isinstance(row_widths, (list, tuple))
                else set()
            )
            widths.update(row_width_set)
            row_degenerate = len(row_width_set) <= 1
            if row_degenerate:
                # A single-width arm compared nothing: it must SAY so and its
                # extra draw must have folded into the same-epoch projections
                # (Opus 6.6) -- an undeclared vacuous comparison can never
                # present as a passed invariance proof.
                if (
                    invariance.get("degenerate_at_production_width") is not True
                    or invariance.get("folded_into_same_epoch") is not True
                ):
                    raise CheckerCalibrationInvalidError(
                        f"blind-replay golden {name!r}: a single-width invariance arm "
                        "is degenerate by construction and its extra draw must fold "
                        "into the same-epoch projections"
                    )
            else:
                real_second_width_seen = True
                width_disagreements += int(bool(invariance.get("disagreement")))
        # Mechanical-claim inventory over the same-epoch-scope draws (Sol 4.3).
        # Every published integrity dispute must be LICENSED by a non-valid
        # machine attestation row; the write door refuses unlicensed ones, so
        # a nonzero gap here means a draw escaped the authority entirely.
        draws = row.get("draws")
        if not isinstance(draws, list) or not draws:
            raise CheckerCalibrationInvalidError(
                f"blind-replay golden {name!r} carries no per-draw rows; a v2 report "
                "must retain every draw for the mechanical-claim inventory"
            )
        for draw in draws:
            if not isinstance(draw, Mapping) or draw.get("mode") == "rotation":
                continue
            entries = int(draw.get("integrity_dispute_entries") or 0)
            claims_required += entries
            if bool(draw.get("licensed")):
                claims_discharged += entries
            else:
                claims_gaps += entries
        attestation = row.get("locator_attestation")
        rows_list = attestation.get("rows") if isinstance(attestation, Mapping) else None
        if not isinstance(rows_list, list):
            raise CheckerCalibrationInvalidError(
                f"blind-replay golden {name!r} carries no machine locator attestation "
                "rows; a v2 report attests the whole corpus, no sampling"
            )
        attestation_rows[name] = [r for r in rows_list if isinstance(r, Mapping)]
        # Acceptance witnesses: an accept-side golden is one whose unanimous
        # same-epoch projection accepts >= 1 item; its witness count is the
        # number of agreeing same-epoch draws (>= 2 required, fable A-4).
        if not row_disagrees and any(
            _projection_item_accepted(kind, _projection_fields(projections[0].get(stable_id)))
            for stable_id in projections[0]
        ):
            if kind in accept_witnesses:
                accept_witnesses[kind] += len(projections)
            if len(projections) < ACCEPTANCE_WITNESS_MINIMUM:
                missing_confirmations += 1
        elif row_disagrees and any(
            _projection_item_accepted(kind, _projection_fields(fields.get(stable_id)))
            for fields in projections
            for stable_id in fields
        ):
            confirmation_disagreements += 1
    latencies = report.get("latency_seconds_at_production_width")
    samples = (
        sorted(float(value) for value in latencies)
        if isinstance(latencies, list) and latencies
        else []
    )
    block_degenerate = production_width == 1 and not real_second_width_seen
    return {
        "goldens": len(goldens),
        "replays_per_golden": min(replays) if replays else 0,
        "same_epoch_disagreements": same_epoch_disagreements,
        "rotation_replay_ran": rotation_ran_everywhere,
        "rotation_disagreements": rotation_disagreements,
        "width_invariance": {
            "widths": sorted(widths) if widths else ([1] if block_degenerate else []),
            "disagreements": width_disagreements,
            "degenerate_at_production_width": block_degenerate,
            "folded_into_same_epoch": block_degenerate,
        },
        "disagreement_diagnosis": diagnosis,
        "mechanical_claims": {
            "required": claims_required,
            "discharged": claims_discharged,
            "refuted": sum(refuted_by_golden.values()),
            "coverage_gaps": claims_gaps,
        },
        "locator_attestation": derive_locator_attestation_block(attestation_rows),
        "acceptance_witness": {
            "metadata_accept_witnesses": accept_witnesses[GateKind.METADATA_BATCH.value],
            "fidelity_accept_witnesses": accept_witnesses[GateKind.FIDELITY.value],
            "terminal_accept_witnesses": accept_witnesses[GateKind.TERMINAL_DISPOSITION.value],
            "required_witnesses_per_accept": ACCEPTANCE_WITNESS_MINIMUM,
            "missing_confirmations": missing_confirmations,
            "confirmation_disagreements": confirmation_disagreements,
            "disagreement_outcome": ACCEPTANCE_DISAGREEMENT_OUTCOME,
            "unfunded_confirmation_outcome": ACCEPTANCE_UNFUNDED_OUTCOME,
            "contested_census_arm": CONTESTED_CENSUS_ARM,
        },
        "p99_at_production_width_seconds": quantile(samples, 0.99),
        "report_sha256": _sha256_file(report_path),
    }


def _require_corpus_membership_floor(corpus_dir: Optional[Path] = None) -> None:
    """W-7: refuse typed unless the corpus byte-matches its committed manifest.

    ``corpus_dir`` defaults to the committed corpus; a staged calibration copy
    passes exactly when its bytes are the committed bytes.
    """

    from menagerie.crawler.tools.golden_corpus_floor import (
        GoldenCorpusFloorError,
        enforce_committed_corpus,
    )

    try:
        failures = enforce_committed_corpus(corpus_dir=corpus_dir)
    except GoldenCorpusFloorError as exc:
        raise CheckerGoldenCorpusError(
            f"golden corpus membership floor (W-7) cannot be measured: {exc}"
        ) from exc
    if failures:
        raise CheckerGoldenCorpusError(
            "golden corpus membership floor (W-7): the corpus does not match its "
            "committed manifest -- " + "; ".join(failures)
        )


def mint_calibration_receipt(
    records_root: Path,
    *,
    work_roots: Sequence[Path],
    checker_model: str,
    blind_replay_report_path: Path,
    captured_at: str,
    source: str,
    next_rung_models: float,
    projected_wall_hours: float,
    wall_budget_hours: float,
    projected_sessions: float,
    session_budget: float,
    projected_dollars: float,
    dollar_budget: float,
    projected_output_tokens: Optional[float] = None,
    token_budget: Optional[float] = None,
    out_path: Path,
) -> JsonObject:
    """Mint one calibration receipt from ledgers + the blind-replay report.

    Every measured field is tool-derived; the budgets and projections are the
    operator-declared capacity plan (the §6 C.9 dossier band). Minting ends by
    round-tripping the written artifact through
    :func:`load_checker_calibration_receipt` -- an under-measured,
    over-refusing, or timing-incompatible receipt refuses HERE and the file is
    removed, so a broken receipt can never sit on disk waiting to activate.

    The mint opens with the W-7 corpus membership floor: the committed golden
    corpus must byte-match its committed manifest, so replacing, removing, or
    re-recording a golden in the same change as a mint attempt refuses typed --
    "re-harvest a clearer exemplar" is a reviewed manifest change, never a
    silent swap that a fresh mint launders into a receipt.
    """

    _require_corpus_membership_floor()
    report = load_blind_replay_report(blind_replay_report_path)
    receipt: JsonObject = {
        "schema_version": CHECKER_CALIBRATION_RECEIPT_SCHEMA,
        "checker_model": checker_model,
        "captured_at": captured_at,
        "source": source,
        # R2-3: the mint runs on the tree it mints for, so the structured
        # epoch it stamps IS the shipping prompt's content hash; the
        # activation door refuses any receipt whose epoch is not live.
        "checker_prompt_identity": live_checker_prompt_identity(),
        "production_batch_width": CALIBRATION_PRODUCTION_BATCH_WIDTH,
        "lanes": gather_lane_measurements(
            records_root, work_roots=work_roots, checker_model=checker_model
        ),
        "blind_replay": derive_blind_replay_block(report, report_path=blind_replay_report_path),
        "capacity": {
            "next_rung_models": next_rung_models,
            "projected_wall_hours": projected_wall_hours,
            "wall_budget_hours": wall_budget_hours,
            "projected_sessions": projected_sessions,
            "session_budget": session_budget,
            "projected_dollars": projected_dollars,
            "dollar_budget": dollar_budget,
        },
    }
    if projected_output_tokens is not None or token_budget is not None:
        receipt["capacity"]["projected_output_tokens"] = projected_output_tokens
        receipt["capacity"]["token_budget"] = token_budget
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    try:
        return load_checker_calibration_receipt(out_path)
    except Exception:
        out_path.unlink(missing_ok=True)
        raise


def verify_calibration_receipt(
    receipt_path: Path,
    records_root: Path,
    *,
    work_roots: Sequence[Path],
    blind_replay_report_path: Optional[Path] = None,
    corpus_dir: Optional[Path] = None,
) -> JsonObject:
    """Recompute every measured field from the same ledgers; refuse divergence.

    The fable-C2 validator: structural validity is necessary but not
    sufficient -- the receipt's numbers must RE-DERIVE from the ledgers it
    claims to summarize. Any field that does not re-derive is named in the
    refusal; when the blind-replay report is supplied its whole block
    (including the byte digest) is re-derived too; and when ``corpus_dir`` is
    supplied the locator-attestation zero bar is recomputed FROM THE CORPUS
    BYTES themselves (fable A-3: the report's producer-written rows alone
    would be self-referential).
    """

    receipt = load_checker_calibration_receipt(receipt_path)
    recomputed = gather_lane_measurements(
        records_root, work_roots=work_roots, checker_model=str(receipt["checker_model"])
    )
    divergences: list[str] = []
    for lane, expected in recomputed.items():
        recorded = receipt["lanes"][lane]
        for field, value in expected.items():
            if _normalize(recorded.get(field)) != _normalize(value):
                divergences.append(
                    f"lanes.{lane}.{field}: receipt={recorded.get(field)!r} recomputed={value!r}"
                )
    if blind_replay_report_path is not None:
        report = load_blind_replay_report(blind_replay_report_path)
        expected_block = derive_blind_replay_block(report, report_path=blind_replay_report_path)
        for field, value in expected_block.items():
            if _normalize(receipt["blind_replay"].get(field)) != _normalize(value):
                divergences.append(
                    f"blind_replay.{field}: receipt={receipt['blind_replay'].get(field)!r} "
                    f"recomputed={value!r}"
                )
    if corpus_dir is not None:
        _require_corpus_membership_floor(corpus_dir)
        corpus_block = derive_locator_attestation_block(
            derive_corpus_locator_attestation(corpus_dir)
        )
        recorded_attestation = (receipt.get("blind_replay") or {}).get("locator_attestation")
        recorded_attestation = (
            recorded_attestation if isinstance(recorded_attestation, Mapping) else {}
        )
        for field, value in corpus_block.items():
            if _normalize(recorded_attestation.get(field)) != _normalize(value):
                divergences.append(
                    f"blind_replay.locator_attestation.{field} (corpus recompute): "
                    f"receipt={recorded_attestation.get(field)!r} recomputed={value!r}"
                )
    if divergences:
        raise CheckerCalibrationDivergenceError(
            f"checker calibration receipt {receipt_path} does not re-derive from its "
            "ledgers: " + "; ".join(divergences)
        )
    return receipt


def _normalize(value: Any) -> Any:
    """Compare ints and floats as numbers, everything else exactly."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _normalize(item) for key, item in value.items()}
    return value
