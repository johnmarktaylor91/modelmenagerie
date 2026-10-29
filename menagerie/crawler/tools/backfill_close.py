"""Mint an honest v2 close for a rung that closed before the leg machinery.

The pilot's mini-rung and step-100 legs really ran and really closed -- but
they predate leg generations, source censuses, and sealed partition proofs, so
no v2 close exists and ``leg-close.v1`` rows license nothing. This tool is the
ONLY mint path for that history (contract, migration clause): it DERIVES every
arm from the pinned bytes through the same
:func:`~menagerie.crawler.partition_proof.derive_partition_proof` the gate
recomputes, under the era-honest ``backfill`` predicates:

* a member whose current head is recorded-under EXACTLY this snapshot id is
  ADJUDICATED (the discriminator the register door already trusts);
* a headless member with a live valve/park row cites that row (the pilot's
  never-authored five and the host-capacity deferrals);
* a member an EARLIER backfilled close already resolved (its head names an
  ancestor rung) is CARRIED, delegated by digest -- process rungs in lineage
  order so the chain bottoms out at a close that adjudicated in its own era;
* a pre-machinery head with NO recorded-under resolves only through an
  explicit ``--generation-authority stable_id=revision`` operator authority,
  dead the moment the head moves; and
* a member none of these can prove REFUSES THE ENTIRE CLOSE, named. You
  cannot backfill a leg you cannot prove -- exhibit (a)'s shape (a rung
  claiming its predecessor's heads) fails per member here exactly as it fails
  at the live gate.

Before any of that, the ERA GATE (rt4 break 1) refuses the whole mint unless
the rung is provably pre-machinery history, derived from the records and never
a caller flag: a ``leg-close.v1`` row must attest this exact rung identity
terminally closed in its own era, NO leg machinery of its own -- no
machinery-era ``leg`` close, no ``leg``-kind generation live or closed -- may
name the rung, and the rung's registry row must not be licensed by a
machinery-era ``leg`` close. A live, in-flight, or never-run rung is a typed
:class:`~menagerie.crawler.partition_proof.BackfillEraError`, and the staging
gate recomputes the same predicate on every replay of a backfill close, so an
era violation that appears later (or a mint that bypassed this tool) stops
licensing.

The operator supplies no partition arm, ever. Minting is idempotent on
unchanged books and the idempotency is derived from CONTENT, never from a
hand-typed timestamp: the same census derives the same proof, and a re-mint
whose derived binding (generation, sealed proof, source census) equals an
existing backfill close returns that row unchanged whatever ``--minted-at``
says. Books that moved since mint make the old close stale (the staging
census re-observation refuses it) and a re-mint attests the new state beside
the old row.

This tool is ALSO the production writer of the ``leg-close.v1`` HISTORY
attestation the era gate demands (``--attest-v1-history``). The pilot's
pre-machinery legs really closed, but nothing production-side ever recorded
that fact -- the only v1 writer in the repo was a test fixture, so the era
gate was unsatisfiable on the real pilot and step-200 could not stage
(OPS-10 dead-end (a)). The mint derives the rung identity from the REGISTRY
row, requires an explicit operator attestation naming the exact era
(``terminally-closed-pre-machinery`` -- a closed vocabulary, C-08), refuses
any rung with leg machinery of its own or a machinery-era registration
licence, refuses a rung with zero recorded evidence (nothing to attest), and
is idempotent on the rung identity. The minted row chains through the durable
same-campaign v2 close tip, ledger-derived (A-1), so history stays attestable
after machinery-era closes exist. The v1 row licenses NOTHING by itself: it is
the historical existence proof behind which the backfill's per-member
derivation and every staging-gate replay still adjudicate from evidence alone.

Heads recorded-under a NAMED non-chain snapshot (the pilot's re-adjudicated
history: later smoke/head50 runs moved current heads onto foreign or
descendant snapshots) resolve only through ``--recorded-under-authority
STABLE_ID=SNAPSHOT_ID:REVISION`` -- an explicit operator authority binding
BOTH the named snapshot and the exact current revision, disclosed in the
close and dead the moment either moves (OPS-10 dead-ends (b)/(c)).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from menagerie.crawler.close_object import (
    CLOSE_LEDGER_NAMES,
    CLOSE_OBJECT_SCHEMA_VERSION,
    CLOSE_OBJECT_SCHEMA_VERSION_V1,
    append_close_object,
    build_close_object,
    close_object_path,
    ledger_identity,
    load_close_objects,
    predecessor_close_sha256,
    verifier_identity,
)
from menagerie.crawler.identity import stamp_row_digest
from menagerie.crawler.leg_generation import (
    census_identities,
    load_leg_open_rows,
    model_head_citations,
    open_leg_generation,
)
from menagerie.crawler.models import JsonObject
from menagerie.crawler.partition_proof import (
    CARRIED,
    WITHHELD,
    BackfillEraError,
    adjudicated_here,
    attributed_here,
    backfill_era_refusals,
    carried,
    derive_partition_proof,
    leg_machinery_refusals,
    machinery_licensed_registration_refusals,
    proved_here,
    withheld_here,
    write_partition_proof,
)

#: The one attested era this migration door serves (C-08: a migration path may
#: only mint for the era it exists to serve, named explicitly by the operator,
#: from a closed vocabulary -- never a free-text claim).
V1_HISTORY_ATTESTED_ERA = "terminally-closed-pre-machinery"

_PARK_KIND_ARMS = {
    "review-held:supersession-pending-grant": "grant_pending",
    "review-held:award-supersession-pending-grant": "grant_pending",
    "review-held:terminal-recording-refused": "refusal_parked",
}


def _backfill_generation(
    records_root: Path,
    *,
    campaign_id: str,
    intake_snapshot_id: str,
    intake_snapshot_sha256: str,
    phase: Optional[str],
    minted_by: str,
    minted_at: str,
):
    """Return the rung's backfill generation, reusing an existing epoch.

    A backfill epoch is reused across re-mints on purpose: re-deriving
    unchanged books under the same generation reproduces the identical
    content-addressed close, which is what makes minting idempotent.
    """

    for row in reversed(load_leg_open_rows(records_root)):
        if (
            row["close_kind"] == "backfill"
            and row["campaign_id"] == campaign_id
            and row["intake_snapshot_sha256"] == intake_snapshot_sha256
            and row.get("phase") == phase
        ):
            from menagerie.crawler.leg_generation import _generation_from_row  # noqa: PLC2701 -- tool reuses the one row typer

            return _generation_from_row(row)
    return open_leg_generation(
        records_root,
        campaign_id=campaign_id,
        intake_snapshot_id=intake_snapshot_id,
        intake_snapshot_sha256=intake_snapshot_sha256,
        phase=phase,
        run_id=f"backfill:{minted_by}",
        opened_at=minted_at,
        close_kind="backfill",
    )


def _disclosure_arms(proof: Mapping[str, Any]) -> dict[str, Any]:
    """Project one derived backfill proof into honest close disclosure arms."""

    members = dict(proof.get("members") or {})
    buckets: dict[str, list[str]] = {}
    machine_deferred: dict[str, list[str]] = {}
    grant_pending: list[str] = []
    refusal_parked: list[str] = []
    for stable_id in sorted(members):
        entry = members[stable_id]
        if entry.get("disposition") in (WITHHELD, "continuing"):
            kind = str(dict(entry.get("evidence") or {}).get("kind"))
            arm = _PARK_KIND_ARMS.get(kind)
            # DX-22: the park arms are DISCLOSURE, not accounting. A parked
            # member's accounting is its machine-deferral claim; the park arm
            # discloses the live operator decision on top of it. Counting the
            # park arms double-counted every parked member (the 212-of-210
            # overlap that made the truthful 200a close unstageable, R2).
            machine_deferred.setdefault(kind, []).append(stable_id)
            if arm == "grant_pending":
                grant_pending.append(stable_id)
            elif arm == "refusal_parked":
                refusal_parked.append(stable_id)
        else:
            buckets.setdefault("recorded", []).append(stable_id)
    dispositions = [str(entry.get("disposition")) for entry in members.values()]
    arms: dict[str, Any] = {
        "intake": len(members),
        "valid": True,
        "missing": [],
        "extra": [],
        "duplicates": [],
        "invalid": [],
        "buckets": buckets,
        "machine_deferred": machine_deferred,
        "superseded_deferral_rows": [],
        "grant_pending": grant_pending,
        "refusal_parked": refusal_parked,
        "adjudicated_here": sorted(adjudicated_here(proof)),
        "carried_count": dispositions.count(CARRIED),
        "withheld_here": sorted(withheld_here(proof)),
        "attributed_here": sorted(attributed_here(proof)),
        "carried": sorted(carried(proof)),
        "proved_here": sorted(proved_here(proof)),
        "introduced": sorted(members),
    }
    from menagerie.crawler.partition_proof import (
        PARTITION_PROOF_SCHEMA_VERSION_V3,
        continuing_here,
    )

    if proof.get("schema_version") != PARTITION_PROOF_SCHEMA_VERSION_V3:
        # RULING W7 clause 6: v4 closes disclose the first-class continuation
        # arm; the staging gate recomputes it from the sealed proof.
        arms["continuing"] = sorted(continuing_here(proof))
    return arms


def _registered_rung(records_root: Path, campaign_id: str, intake_snapshot_id: str):
    """Return the registry row for one rung -- the identity authority -- or refuse."""

    from menagerie.crawler.rung_staging import load_rung_registrations

    registration = next(
        (
            row
            for row in load_rung_registrations(records_root)
            if row.campaign_id == campaign_id and row.snapshot_id == intake_snapshot_id
        ),
        None,
    )
    if registration is None:
        raise ValueError(
            f"{intake_snapshot_id} is not a registered rung of {campaign_id}: the "
            "migration door mints only for registered history (register the "
            "snapshot first; the registry row is the identity authority)"
        )
    return registration


def mint_v1_history_attestation(
    records_root: Path,
    *,
    campaign_id: str,
    intake_snapshot_id: str,
    attested_era: str,
    attested_by: str,
    closed_at: str,
    phase: Optional[str] = None,
) -> JsonObject:
    """Mint the ``leg-close.v1`` HISTORY attestation for one pre-machinery rung.

    The production equivalent of the test fixture the pilot never had: the
    era gate demands a ``leg-close.v1`` row attesting that the rung really
    terminally closed in its own era, and until this mint existed the only
    writer of such rows was a test fixture -- the honest pilot backfill was
    unsatisfiable by construction (OPS-10 dead-end (a)).

    The row LICENSES NOTHING. It is readable history: the staging gate refuses
    every v1 close outright, the backfill's per-member derivation still proves
    every member from recorded evidence, and every gate replay recomputes the
    era predicates from the ledgers. What this mint adds is era-scoped,
    provenanced disclosure of the operator's historical claim.

    Era scope (C-08 -- all recomputed from the records, never caller flags):

    * identity derives from the REGISTRY row (the digest is never typed);
    * ``attested_era`` must be the exact closed-vocabulary literal
      :data:`V1_HISTORY_ATTESTED_ERA`;
    * a rung with leg machinery of its own (a machinery-era ``leg`` close, a
      ``leg``-kind generation live or closed) refuses typed -- the mint may
      never outrank a live-machinery close;
    * a rung REGISTERED inside the machinery era (its registry row licensed
      by a machinery-era ``leg`` close, the licence kind re-derived from the
      close ledger) refuses typed -- machinery-era pipeline entries can never
      be re-described as pre-machinery history; and
    * a rung with ZERO recorded evidence (no member holds a current head or a
      live withholding row) has no history to attest and refuses.

    The minted row chains: its ``predecessor_close_sha256`` is the durable
    same-campaign v2 tip, DERIVED from the close ledger at mint time (A-1) --
    ``None`` only at genesis -- so v1 history attestations stay mintable after
    the campaign's machinery-era closes exist, and the append tripwire keeps
    refusing any row that names a different predecessor.

    Minting is idempotent on the rung identity: an existing v1 row for the
    exact ``(campaign, snapshot id, digest, phase)`` is returned unchanged,
    whatever ``closed_at`` says.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    campaign_id, intake_snapshot_id:
        The registered rung whose pre-machinery close is being attested.
    attested_era:
        Must equal :data:`V1_HISTORY_ATTESTED_ERA` exactly.
    attested_by:
        Operator provenance.
    closed_at:
        The HISTORICAL close time being attested (operator-stated; this is a
        claim about the past, so it has no honest default).
    phase:
        Phase scope (``None`` for a whole-rung close).

    Returns
    -------
    dict[str, Any]
        The persisted (or already-persisted) v1 close row.

    Raises
    ------
    ValueError
        If the rung is not registered or the era literal is wrong.
    BackfillEraError
        If the rung has leg machinery of its own, or nothing to attest.
    """

    if attested_era != V1_HISTORY_ATTESTED_ERA:
        raise ValueError(
            f"attested era {attested_era!r} is not the one era this migration "
            f"door serves ({V1_HISTORY_ATTESTED_ERA!r}): the attestation names "
            "its era from a closed vocabulary, never a free-text claim"
        )
    registration = _registered_rung(records_root, campaign_id, intake_snapshot_id)
    era_findings = (
        *leg_machinery_refusals(
            records_root,
            campaign_id=campaign_id,
            intake_snapshot_id=intake_snapshot_id,
            intake_snapshot_sha256=registration.snapshot_sha256,
        ),
        *machinery_licensed_registration_refusals(
            records_root,
            campaign_id=campaign_id,
            intake_snapshot_id=intake_snapshot_id,
            intake_snapshot_sha256=registration.snapshot_sha256,
        ),
    )
    if era_findings:
        raise BackfillEraError(
            f"cannot attest {intake_snapshot_id} as pre-machinery history: "
            + "; ".join(era_findings)
        )
    for row in load_close_objects([close_object_path(records_root)]):
        if (
            row.get("schema_version") == CLOSE_OBJECT_SCHEMA_VERSION_V1
            and str(row.get("campaign_id")) == campaign_id
            and str(row.get("intake_snapshot_id")) == intake_snapshot_id
            and str(row.get("intake_snapshot_sha256")) == registration.snapshot_sha256
            and row.get("phase") == phase
        ):
            return dict(row)
    from menagerie.crawler.partition_proof import (
        _live_withholding_rows,  # noqa: PLC2701 -- the one live-withholding census
    )
    from menagerie.crawler.rung_staging import (
        _lineage_member_ids,  # noqa: PLC2701 -- the one registry-verified membership reader
    )

    members = _lineage_member_ids(records_root, registration)
    heads = model_head_citations(records_root, require_ledger=False)
    live = _live_withholding_rows(records_root, campaign_id, heads, tolerate_pilot_campaign=True)
    if not any(member in heads or member in live for member in members):
        raise BackfillEraError(
            f"nothing to attest for {intake_snapshot_id}: no member holds a "
            "current head or a live withholding row. A v1 history attestation "
            "states that a leg really ran and terminally closed; a rung with "
            "zero recorded evidence has no history to attest"
        )
    ledger_paths = {
        "models": records_root / "models" / "current-shard.jsonl",
        "attempts": records_root / "attempts" / "local.jsonl",
        "gates": records_root / "gates" / "current-shard.jsonl",
        "artifacts": records_root / "artifacts" / "current-shard.jsonl",
    }
    body: JsonObject = {
        "schema_version": CLOSE_OBJECT_SCHEMA_VERSION_V1,
        "campaign_id": campaign_id,
        "intake_snapshot_id": intake_snapshot_id,
        "intake_snapshot_sha256": registration.snapshot_sha256,
        "phase": phase,
        "attested_era": attested_era,
        "run_id": f"v1-history-attestation:{attested_by}",
        "machine_id": f"v1-history-attestation:{attested_by}",
        "closed_at": closed_at,
        "completion_status": "complete",
        # The chain invariant applies to EVERY appended row (A-1): the
        # predecessor is the durable same-campaign v2 tip, derived from the
        # ledger and never supplied by the caller, so a v1 history attestation
        # minted after machinery-era closes exist still appends -- and a row
        # naming anything else keeps refusing at the append tripwire.
        "predecessor_close_sha256": predecessor_close_sha256(records_root, campaign_id),
        "ledgers": {name: ledger_identity(ledger_paths[name]) for name in CLOSE_LEDGER_NAMES},
        "partition": {
            "intake": 0,
            "valid": True,
            "missing": [],
            "extra": [],
            "duplicates": [],
            "invalid": [],
            "buckets": {},
            "machine_deferred": {},
            "grant_pending": [],
            "refusal_parked": [],
        },
        "currency": {"stale_by_axis": {}},
        "verifier_identity": verifier_identity({}),
        "authority_identities": {},
    }
    row = stamp_row_digest(body, field="close_sha256")
    return append_close_object(close_object_path(records_root), row)


def backfill_close(
    records_root: Path,
    *,
    campaign_id: str,
    intake_snapshot_id: str,
    minted_by: str,
    minted_at: str,
    phase: Optional[str] = None,
    operator_attributions: Optional[Mapping[str, Any]] = None,
) -> JsonObject:
    """Derive and append one honest backfill close for a registered rung.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    campaign_id, intake_snapshot_id:
        The registered rung to backfill; its digest comes from the REGISTRY
        row, never from the caller.
    minted_by, minted_at:
        Mint provenance. The timestamp never steers idempotency: a re-mint
        whose derived content equals an existing close returns that row.
    phase:
        Phase scope (``None`` for a whole-rung close, the consumable kind).
    operator_attributions:
        Operator authorities, two closed shapes: ``stable_id -> exact current
        record_revision`` for a pre-machinery head lacking its recorded-under
        field, or ``stable_id -> {"recorded_under": snapshot_id,
        "record_revision": revision}`` for a head recorded-under a NAMED
        non-chain snapshot (disclosed in the close, dead when either moves).

    Returns
    -------
    dict[str, Any]
        The persisted close object.

    Raises
    ------
    ValueError
        If the rung is not registered for this campaign.
    BackfillEraError
        If the records do not prove the rung is terminally closed
        pre-machinery history: no ``leg-close.v1`` row attests this exact rung
        identity, or the rung has leg machinery of its own (a machinery-era
        ``leg`` close, or a ``leg``-kind generation live or closed). Refused
        before anything is written.
    PartitionProofError
        If any member cannot be proven from the recorded evidence.
    """

    from menagerie.crawler.rung_staging import (
        _lineage_member_ids,  # noqa: PLC2701 -- the one registry-verified membership reader
    )

    registration = _registered_rung(records_root, campaign_id, intake_snapshot_id)
    era_findings = backfill_era_refusals(
        records_root,
        campaign_id=campaign_id,
        intake_snapshot_id=intake_snapshot_id,
        intake_snapshot_sha256=registration.snapshot_sha256,
        phase=phase,
    )
    if era_findings:
        raise BackfillEraError(f"cannot backfill {intake_snapshot_id}: " + "; ".join(era_findings))
    members = sorted(_lineage_member_ids(records_root, registration))
    generation = _backfill_generation(
        records_root,
        campaign_id=campaign_id,
        intake_snapshot_id=intake_snapshot_id,
        intake_snapshot_sha256=registration.snapshot_sha256,
        phase=phase,
        minted_by=minted_by,
        minted_at=minted_at,
    )
    attributions = dict(operator_attributions or {})
    proof = derive_partition_proof(
        records_root,
        campaign_id=campaign_id,
        intake_snapshot_id=intake_snapshot_id,
        intake_snapshot_sha256=registration.snapshot_sha256,
        phase=phase,
        close_kind="backfill",
        generation_id=generation.generation_id,
        baselines=None,
        member_ids=members,
        operator_attributions=attributions,
    )
    write_partition_proof(records_root, proof)
    partition = _disclosure_arms(proof)
    from menagerie.crawler.rung_staging import introduced_member_ids

    partition["introduced"] = sorted(introduced_member_ids(records_root, registration))
    completion_status = (
        "complete-with-deferrals"
        if partition["withheld_here"] or partition.get("continuing")
        else "complete"
    )
    currency: dict[str, Any] = {
        "ledger_heads": len(partition["adjudicated_here"]) + partition["carried_count"],
        "dependency_current": len(partition["adjudicated_here"])
        + partition["carried_count"],
        "stale": 0,
        "stale_by_axis": {},
        "stale_reasons": {},
        "awards_stale_on_live_basis": [],
        "award_runs_to_non_runs": list(
            dict(proof["award_transition_census"])["runs_to_non_runs"]
        ),
        "award_runs_to_runs": list(
            dict(proof["award_transition_census"])["runs_to_runs"]
        ),
        "award_transition_instrumentation": dict(proof["award_transition_census"])[
            "instrumentation"
        ],
        "award_transition_windows": dict(proof["award_transition_census"])[
            "coverage_windows"
        ],
    }
    sources = census_identities(records_root)
    # Idempotency is derived from CONTENT, never from the mint timestamp
    # (OPS-10 (iv)): re-deriving unchanged books reproduces the same
    # generation, sealed proof, and source census, and THAT binding -- not a
    # retyped --minted-at -- decides whether an attestation already exists.
    for row in load_close_objects([close_object_path(records_root)]):
        if (
            row.get("schema_version") == CLOSE_OBJECT_SCHEMA_VERSION
            and row.get("close_kind") == "backfill"
            and str(row.get("campaign_id")) == campaign_id
            and str(row.get("intake_snapshot_id")) == intake_snapshot_id
            and str(row.get("intake_snapshot_sha256")) == registration.snapshot_sha256
            and row.get("phase") == phase
            and row.get("generation_id") == generation.generation_id
            and row.get("partition_proof_sha256") == proof["partition_proof_sha256"]
            and row.get("sources") == sources
        ):
            return dict(row)
    return append_close_object(
        close_object_path(records_root),
        build_close_object(
            campaign_id=campaign_id,
            intake_snapshot_id=intake_snapshot_id,
            intake_snapshot_sha256=registration.snapshot_sha256,
            phase=phase,
            close_kind="backfill",
            generation_id=generation.generation_id,
            leg_open_row_sha256=generation.row_sha256,
            sources=sources,
            partition_proof_sha256=proof["partition_proof_sha256"],
            predecessor_close_sha256=predecessor_close_sha256(records_root, campaign_id),
            run_id=f"backfill:{minted_by}",
            machine_id=f"backfill:{minted_by}",
            closed_at=minted_at,
            completion_status=completion_status,
            partition_arms=partition,
            currency_census=currency,
            authority_identities={},
            operator_attributions=attributions or None,
        ),
    )


def _parse_attribution(value: str) -> tuple[str, str]:
    stable_id, _, revision = value.partition("=")
    if not stable_id or not revision:
        raise argparse.ArgumentTypeError("generation authority must be stable_id=record_revision")
    return stable_id, revision


def _parse_recorded_under_authority(value: str) -> tuple[str, dict[str, str]]:
    stable_id, _, binding = value.partition("=")
    recorded_under, _, revision = binding.partition(":")
    if not stable_id or not recorded_under or not revision:
        raise argparse.ArgumentTypeError(
            "recorded-under authority must be STABLE_ID=SNAPSHOT_ID:REVISION "
            "(the NAMED non-chain snapshot the head is recorded-under, then "
            "the exact current record revision)"
        )
    return stable_id, {"recorded_under": recorded_under, "record_revision": revision}


def _utc_now() -> str:
    """Return the current UTC timestamp in the ledger's format."""

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _print_refusal(exc: Exception) -> None:
    """Print one typed migration refusal, member reasons as a table (BD-1).

    The operator's first contact with this tool must be a refusal that reads,
    never a stack trace whose message concatenates every unresolved member
    (OPS-10 (iii)).
    """

    member_reasons: Mapping[str, str] = dict(getattr(exc, "member_reasons", None) or {})
    if not member_reasons:
        print(f"backfill-close refused: {exc}", file=sys.stderr)
        return
    print(
        f"backfill-close refused: {len(member_reasons)} member(s) cannot be "
        "resolved by exact citation:",
        file=sys.stderr,
    )
    width = max(len(stable_id) for stable_id in member_reasons)
    for stable_id in sorted(member_reasons):
        print(f"  {stable_id.ljust(width)}  {member_reasons[stable_id]}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    """Build the migration-door argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--minted-by", required=True)
    parser.add_argument(
        "--minted-at",
        default=None,
        help=(
            "mint provenance timestamp; defaults to now. Idempotency is "
            "derived from content, never from this value"
        ),
    )
    parser.add_argument("--phase", default=None)
    parser.add_argument(
        "--generation-authority",
        action="append",
        type=_parse_attribution,
        default=[],
        metavar="STABLE_ID=REVISION",
        help="explicit operator authority for one pre-machinery head lacking "
        "its recorded-under field; must name the EXACT current revision",
    )
    parser.add_argument(
        "--recorded-under-authority",
        action="append",
        type=_parse_recorded_under_authority,
        default=[],
        metavar="STABLE_ID=SNAPSHOT_ID:REVISION",
        help=(
            "explicit operator authority for one head recorded-under a NAMED "
            "non-chain snapshot (re-adjudicated pilot history); binds both the "
            "named snapshot and the EXACT current revision, disclosed in the "
            "close, dead the moment either moves"
        ),
    )
    parser.add_argument(
        "--attest-v1-history",
        choices=[V1_HISTORY_ATTESTED_ERA],
        default=None,
        metavar="ERA",
        help=(
            "instead of minting a v2 close, append the leg-close.v1 HISTORY "
            f"attestation the era gate demands; ERA must be the literal "
            f"{V1_HISTORY_ATTESTED_ERA!r} (closed vocabulary). Requires "
            "--closed-at (the historical close time being attested). The row "
            "licenses nothing by itself"
        ),
    )
    parser.add_argument(
        "--closed-at",
        default=None,
        help=(
            "with --attest-v1-history: the HISTORICAL close time being "
            "attested (a claim about the past; it has no honest default)"
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI: the pre-machinery migration door (v1 attestation + v2 backfill mint)."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.attest_v1_history is not None:
            if args.closed_at is None:
                parser.error("--attest-v1-history requires --closed-at")
            row = mint_v1_history_attestation(
                args.records_root,
                campaign_id=args.campaign,
                intake_snapshot_id=args.snapshot_id,
                attested_era=args.attest_v1_history,
                attested_by=args.minted_by,
                closed_at=args.closed_at,
                phase=args.phase,
            )
            payload = {
                "close_sha256": row["close_sha256"],
                "schema_version": row["schema_version"],
                "intake_snapshot_id": row["intake_snapshot_id"],
                "intake_snapshot_sha256": row["intake_snapshot_sha256"],
                "closed_at": row["closed_at"],
            }
        else:
            close = backfill_close(
                args.records_root,
                campaign_id=args.campaign,
                intake_snapshot_id=args.snapshot_id,
                minted_by=args.minted_by,
                minted_at=args.minted_at if args.minted_at is not None else _utc_now(),
                phase=args.phase,
                operator_attributions={
                    **dict(args.generation_authority),
                    **dict(args.recorded_under_authority),
                },
            )
            payload = {
                "close_sha256": close["close_sha256"],
                "partition_proof_sha256": close["partition_proof_sha256"],
                "generation_id": close["generation_id"],
                "completion_status": close["completion_status"],
            }
    except ValueError as exc:
        # BackfillEraError, PartitionProofError, CloseObjectError, and
        # RungStagingError are all ValueError family: every typed migration
        # refusal prints and exits 2, never a traceback.
        _print_refusal(exc)
        return 2
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
