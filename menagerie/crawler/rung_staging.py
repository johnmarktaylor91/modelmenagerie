"""Durable in-repo rung staging: registered intake subsets with campaign lineage.

A "rung" is one ladder step of a campaign: a content-addressed intake snapshot
whose members are drawn from the campaign's bound full-partition snapshot. The
frozen four-campaign manifest (``campaigns.json``) owns only the four root
snapshots, so every ladder subset used to be an UNBOUND snapshot that silently
fell back to the default ``DriverConfig`` -- the wrong author model for any
campaign whose binding differs from the defaults (T4-T2).

This module makes rungs first-class and durable (T2-03):

* staged snapshot directories live beside the four roots under
  ``records/intake/<snapshot-id>/``, written with the same immutable layout the
  intake writer uses and validated through the real loader; and
* an append-only registry ``records/intake/registered-rungs.jsonl`` binds each
  rung snapshot to exactly one frozen campaign with explicit
  ``derived_from_snapshot_id`` lineage.

Staging a NEW rung is refused with a typed error until the intake-currency
carve-out (terminals under a PRIOR registered snapshot of the same campaign
stay current) exists in :mod:`menagerie.crawler.authority`. Without it, a new
active snapshot id projects every prior terminal dependency-stale and the
ladder cost goes quadratic (T4-T1). Registering an EXISTING already-run
snapshot is deliberately not CARVE-OUT gated (it is the durability copy
protecting history already recorded under that snapshot id), but it walks the
same close-object and award-authority gates as ``stage_rung``: the gates bind
the lineage and the evidence, never the entry point (2026-08-16 red-team), and
"already-run history" must be PROVEN from recorded heads, not claimed.

A rung composed by :mod:`menagerie.crawler.rung_stratification` publishes a
COMPOSITION ATTESTATION (B-03) naming the campaign, every member with its
stratum, and the snapshot identity those members compose to. Presented to
either staging door, it is re-derived from the stage itself and enforced, so a
stale or hand-edited snapshot sha refuses instead of staging (the S200-PREP
``e3745a1a`` hazard). The gate is disposition-negative: it can only refuse a
stage, never license one that the close-object, membership, or award gates
would otherwise have refused.

Rung membership is CUMULATIVE: a successor derives from the registry tip and
strictly widens it, so its members are everything the tip already proved plus
the members it introduces (which is what lets a close CARRY an inherited member
by digest to the ancestor close that adjudicated it). A composer therefore
carries the tip explicitly, as the disclosed :data:`CARRIED_TIP_STRATUM`
(C-2 / 2026-08-19). The door's widen rule is unchanged and stays strict; what
the door adds for a presented attestation is the one refusal a carry label can
otherwise launder -- a member claiming ``lineage-tip`` provenance that the
registry tip does not contain, or claiming it at all with no rung registered.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from menagerie.crawler.identity import (
    atomic_replace_bytes,
    canonical_json_bytes,
    durable_fsync,
    foreign_content_address_fields,
    fsync_directory,
    hash_bytes,
    row_content_address,
    stable_hash,
    stamp_row_digest,
)
from menagerie.crawler.close_reads import read_epoch
from menagerie.crawler.intake import IntakeSnapshot, load_intake_snapshot
from menagerie.crawler.models import JsonObject
from menagerie.crawler.partitioner import (
    CampaignBinding,
    CampaignPartitionError,
    load_campaign_bindings,
)

RUNG_REGISTRATION_SCHEMA_VERSION = "menagerie.crawler.rung-registration.v1"

#: Closed rung-class vocabulary (W16/S-2 consumer split). The class is DECLARED
#: at the staging door, bound into the registration's content address, and
#: replay-derived by every certificate consumer — never steered by a launch-time
#: operator flag. ``successor_rung`` is the composed next-rung launch shape (its
#: legs demand a ``rung_launch`` certificate); ``canary_seed`` is an R7 canary
#: seed composition and ``adopted_history`` an OPS-2 proven-history adoption
#: (both admit under ``canary_admission``). Widening this set is a reviewed
#: schema change that must land with its first consumer.
RUNG_CLASS_SUCCESSOR = "successor_rung"
RUNG_CLASS_CANARY_SEED = "canary_seed"
RUNG_CLASS_ADOPTED_HISTORY = "adopted_history"
RUNG_CLASSES = frozenset(
    {RUNG_CLASS_SUCCESSOR, RUNG_CLASS_CANARY_SEED, RUNG_CLASS_ADOPTED_HISTORY}
)

#: Append-once class refinement for PRE-VOCABULARY registrations (P5 round-2
#: R2-4 / SOL-01). The live 200a-era rows predate the rung-class vocabulary and
#: lawfully carry no class; the append-once registry can never rebind them, and
#: the idempotent restage deliberately writes nothing — so the staged successor
#: kept deriving ``canary_admission``. A refinement row binds ONE exact
#: unclassed registration (by snapshot identity AND registration row address)
#: to its declared class, at the campaign's registry tip only, exactly once.
RUNG_CLASS_REFINEMENT_SCHEMA_VERSION = "menagerie.crawler.rung-class-refinement.v1"

#: Schema of the composition attestation a stratified rung plan publishes and
#: both staging doors re-verify (B-03). v2 (P5 round-2 SOL-02) adds the
#: REQUIRED ``pd7`` disposition block: the PD-7 slop-family-routes side status
#: is digest-bound INTO the attestation body, so the staging doors consume the
#: adjudication itself instead of trusting a side file that can move or be
#: edited after the composer signed. v1 rows (no ``pd7`` key) remain loadable
#: for the committed history; a successor-rung stage refuses them.
RUNG_COMPOSITION_ATTESTATION_SCHEMA_VERSION_V1 = (
    "menagerie.crawler.rung-composition-attestation.v1"
)
RUNG_COMPOSITION_ATTESTATION_SCHEMA_VERSION = "menagerie.crawler.rung-composition-attestation.v2"

#: Closed PD-7 disposition vocabulary carried by a v2 composition attestation.
#: ``SATISFIED`` is the only status that stages a successor rung;
#: ``UNSATISFIED-BLOCKING`` refuses EVERY stage it reaches (a known-blocking
#: adjudication never stages anything); ``NOT-EVALUATED`` is the honest
#: pre-adjudication state and refuses exactly where ``SATISFIED`` is demanded.
PD7_DISPOSITION_SATISFIED = "SATISFIED"
PD7_DISPOSITION_BLOCKING = "UNSATISFIED-BLOCKING"
PD7_DISPOSITION_NOT_EVALUATED = "NOT-EVALUATED"
PD7_DISPOSITION_VOCABULARY = frozenset(
    {PD7_DISPOSITION_SATISFIED, PD7_DISPOSITION_BLOCKING, PD7_DISPOSITION_NOT_EVALUATED}
)

#: Report key carrying the attestation inside a stratified rung plan report, so
#: the operator hands ``stage_rung`` the SAME artifact the composer emitted
#: instead of a hand-copied digest.
COMPOSITION_ATTESTATION_REPORT_KEY = "composition_attestation"

#: Stratum name a composition uses for the lineage-tip members it CARRIES
#: forward (C-2). Defined here rather than beside the composer's own strata
#: because both sides need one spelling and only this direction of the import
#: exists: the composer imports this module at module scope, the door reaches
#: the composer's vocabulary lazily (:func:`_closed_stratum_vocabulary`).
CARRIED_TIP_STRATUM = "carried-tip"

#: Provenance strings a composition report discloses for its carried block: the
#: campaign's registry tip, or the root exemption case where no rung exists yet.
CARRIED_TIP_PROVENANCE = "lineage-tip"
NO_CARRIED_TIP_PROVENANCE = "no-registered-lineage-tip"

#: Module attribute on :mod:`menagerie.crawler.authority` that declares the
#: intake-currency carve-out (terminals under a PRIOR registered snapshot of
#: the same campaign stay current). The C-M1-KERNEL worker (K2) sets it to
#: ``True`` in the same change that lands the ``validate_currency`` carve-out;
#: until then staging a new rung is refused so no mixed-snapshot traffic can
#: exist for the carve-out to mishandle.
CURRENCY_CARVE_OUT_ATTRIBUTE = "PRIOR_SNAPSHOT_TERMINAL_CURRENCY_CARVE_OUT"

_SNAPSHOT_ID_PATTERN = re.compile(r"intake-[0-9a-f]{20}")
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


class RungStagingError(ValueError):
    """Raised when a rung cannot be staged, registered, or resolved."""


class UnboundIntakeSnapshotError(RungStagingError):
    """Raised when an intake snapshot has no campaign binding.

    Running, statusing, or checkpointing an unbound snapshot would silently
    adopt the default driver configuration -- the wrong author model and
    review checkpoint for any campaign whose binding differs (T4-T2) -- so
    every consumer refuses instead.
    """


class CompositionAttestationError(RungStagingError):
    """Raised when a rung's composition attestation does not describe the stage.

    B-03. The composer (:mod:`menagerie.crawler.rung_stratification`) publishes
    what it composed -- campaign, member/stratum pairs, and the snapshot
    identity those members compose to -- and both staging doors recompute all
    three from the thing actually being staged. A stale attestation (composed
    against an older records state, so its snapshot sha no longer describes
    those members), a hand-edited member list, and a hand-edited snapshot sha
    are the SAME failure to this gate, and all three refuse before any durable
    byte exists (the S200-PREP ``e3745a1a`` hazard: a snapshot sha carried
    forward by hand into a stage it no longer described).
    """


class CurrencyCarveOutMissingError(RungStagingError):
    """Raised when staging is attempted before the K2 authority carve-out.

    Staging a new rung changes the active snapshot id. Until
    ``validate_currency`` carves out terminals recorded under a PRIOR
    registered snapshot of the same campaign, that change projects every
    prior terminal dependency-stale and re-drives it (T4-T1).
    """


@dataclass(frozen=True)
class RungRegistration:
    """One durable registry row binding a rung snapshot to its campaign.

    Parameters
    ----------
    campaign_id:
        Frozen tier campaign that owns the rung.
    snapshot_id, snapshot_sha256:
        Content-addressed identity of the staged intake snapshot.
    derived_from_snapshot_id:
        Prior snapshot in this campaign's lineage: the campaign's root
        snapshot or an earlier registered rung.
    member_count:
        Number of intake items in the rung.
    label:
        Operator-facing rung name (``mini-10``, ``step-200``).
    registered_by, created_at:
        Registration provenance.
    row_sha256:
        Content address of the row itself.
    rung_class:
        Closed staging-door class declaration (:data:`RUNG_CLASSES`); absent
        only on pre-vocabulary rows, which lawfully carry no
        ``live_origin_custody`` licence.
    """

    campaign_id: str
    snapshot_id: str
    snapshot_sha256: str
    derived_from_snapshot_id: str
    member_count: int
    label: str
    registered_by: str
    created_at: str
    row_sha256: str
    licensed_by_close_sha256: Optional[str] = None
    license_kind: Optional[str] = None
    closes_chain_head: Optional[str] = None
    rung_class: Optional[str] = None
    chain: Optional[Mapping[str, Any]] = None

    def to_row(self) -> JsonObject:
        """Return the canonical persisted JSON row."""

        row: JsonObject = {
            "schema_version": RUNG_REGISTRATION_SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "snapshot_id": self.snapshot_id,
            "snapshot_sha256": self.snapshot_sha256,
            "derived_from_snapshot_id": self.derived_from_snapshot_id,
            "member_count": self.member_count,
            "label": self.label,
            "registered_by": self.registered_by,
            "created_at": self.created_at,
            "row_sha256": self.row_sha256,
        }
        if self.licensed_by_close_sha256 is not None:
            row["licensed_by_close_sha256"] = self.licensed_by_close_sha256
            row["license_kind"] = self.license_kind
            row["closes_chain_head"] = self.closes_chain_head
        if self.rung_class is not None:
            row["rung_class"] = self.rung_class
        if self.chain is not None:
            row["chain"] = dict(self.chain)
        return row


@dataclass(frozen=True)
class StagedRung:
    """Result of one staging or registration operation.

    Parameters
    ----------
    registration:
        Durable registry row binding the rung.
    snapshot_root:
        Rung snapshot directory inside the durable record store.
    snapshot_created:
        Whether this call materialized new snapshot bytes.
    registration_created:
        Whether this call appended a new registry row.
    """

    registration: RungRegistration
    snapshot_root: Path
    snapshot_created: bool
    registration_created: bool


@dataclass(frozen=True)
class RungCompositionAttestation:
    """What a rung composer attested about the rung it composed (B-03).

    The attestation is deliberately about the COMPOSITION, not about the
    operator's intent: it names the campaign, the exact member set with each
    member's stratum, and the ``(snapshot_id, snapshot_sha256)`` identity that
    member set composes to. Everything a staging door needs to prove that the
    thing being staged is the thing that was composed is therefore recomputable
    from the records state -- nothing here is taken on trust.

    Parameters
    ----------
    campaign_id:
        Frozen tier campaign the composition drew from.
    size, seed:
        Composer inputs, carried for operator diagnosis only; they steer no
        verification decision.
    member_strata:
        ``(stable_id, stratum)`` pairs sorted by stable id. Selection ORDER is
        deliberately excluded: the subset writer emits members in root-manifest
        order, so order is not part of what gets staged and comparing it would
        make the gate fail on a correct stage.
    snapshot_id, snapshot_sha256:
        Content-addressed identity ``member_strata`` composes to.
    composition_digest:
        Content address of the attestation body above.
    """

    campaign_id: str
    size: int
    seed: int
    member_strata: tuple[tuple[str, str], ...]
    snapshot_id: str
    snapshot_sha256: str
    composition_digest: str
    #: PD-7 disposition block (v2, SOL-02): ``{"status": <closed vocabulary>,
    #: "source_attestation_sha256": "sha256:..."}`` bound into
    #: ``composition_digest``; ``None`` on a pre-vocabulary v1 attestation.
    pd7_disposition: Optional[Mapping[str, str]] = None

    @property
    def member_ids(self) -> frozenset[str]:
        """Return the attested member set."""

        return frozenset(stable_id for stable_id, _stratum in self.member_strata)

    @property
    def carried_tip_member_ids(self) -> frozenset[str]:
        """Return the members this composition CLAIMS it carried from the tip.

        A claim, never a fact: the staging door recomputes the campaign's tip
        membership from the registry and refuses any claim the tip does not
        support (:func:`_verify_carried_tip_claim`).
        """

        return frozenset(
            stable_id for stable_id, stratum in self.member_strata if stratum == CARRIED_TIP_STRATUM
        )

    def strata_histogram(self) -> dict[str, int]:
        """Return the attested per-stratum member counts."""

        histogram: dict[str, int] = {}
        for _stable_id, stratum in self.member_strata:
            histogram[stratum] = histogram.get(stratum, 0) + 1
        return histogram

    def to_row(self) -> JsonObject:
        """Return the canonical persisted JSON attestation."""

        return {
            **_composition_attestation_body(
                campaign_id=self.campaign_id,
                size=self.size,
                seed=self.seed,
                member_strata=self.member_strata,
                snapshot_id=self.snapshot_id,
                snapshot_sha256=self.snapshot_sha256,
                pd7_disposition=self.pd7_disposition,
            ),
            "composition_digest": self.composition_digest,
        }


def _composition_attestation_body(
    *,
    campaign_id: str,
    size: int,
    seed: int,
    member_strata: Sequence[tuple[str, str]],
    snapshot_id: str,
    snapshot_sha256: str,
    pd7_disposition: Optional[Mapping[str, str]] = None,
) -> JsonObject:
    """Return the canonical digest preimage of one composition attestation.

    A ``pd7_disposition`` selects the v2 schema and rides INSIDE the digest
    preimage (SOL-02: the consumed adjudication is what the composer signed,
    never a side file); ``None`` reproduces the v1 body byte-for-byte so the
    committed v1 history keeps verifying.
    """

    body: JsonObject = {
        "schema_version": (
            RUNG_COMPOSITION_ATTESTATION_SCHEMA_VERSION
            if pd7_disposition is not None
            else RUNG_COMPOSITION_ATTESTATION_SCHEMA_VERSION_V1
        ),
        "campaign_id": campaign_id,
        "size": size,
        "seed": seed,
        "member_strata": [[stable_id, stratum] for stable_id, stratum in member_strata],
        "snapshot_id": snapshot_id,
        "snapshot_sha256": snapshot_sha256,
    }
    if pd7_disposition is not None:
        body["pd7"] = {
            "status": str(pd7_disposition["status"]),
            "source_attestation_sha256": str(pd7_disposition["source_attestation_sha256"]),
        }
    return body


def build_composition_attestation(
    *,
    campaign_id: str,
    size: int,
    seed: int,
    member_strata: Mapping[str, str],
    snapshot_id: str,
    snapshot_sha256: str,
    pd7_disposition: Optional[Mapping[str, str]] = None,
) -> RungCompositionAttestation:
    """Attest one composed rung: its members, their strata, and its identity.

    Parameters
    ----------
    campaign_id:
        Frozen tier campaign the composition drew from.
    size, seed:
        Composer inputs, carried as diagnostics.
    member_strata:
        ``stable_id -> stratum`` for every composed member.
    snapshot_id, snapshot_sha256:
        Identity the composed member set composes to.
    pd7_disposition:
        PD-7 slop-family-routes disposition the composer consumed (SOL-02):
        ``{"status": <PD7_DISPOSITION_VOCABULARY>, "source_attestation_sha256":
        "sha256:..."}``. Present selects the v2 schema and binds the block into
        ``composition_digest``; ``None`` emits a v1 attestation (which no
        successor-rung stage accepts).

    Returns
    -------
    RungCompositionAttestation
        Validated attestation carrying its content address.

    Raises
    ------
    CompositionAttestationError
        If any field is empty, malformed, or names an unknown stratum.
    """

    pairs = tuple(sorted((str(key), str(value)) for key, value in member_strata.items()))
    body = _composition_attestation_body(
        campaign_id=campaign_id,
        size=size,
        seed=seed,
        member_strata=pairs,
        snapshot_id=snapshot_id,
        snapshot_sha256=snapshot_sha256,
        pd7_disposition=pd7_disposition,
    )
    row = dict(body)
    row["composition_digest"] = stable_hash(body)
    return _composition_attestation_from_row(row, source="build")


def _closed_stratum_vocabulary() -> frozenset[str]:
    """Return the composer's closed stratum names.

    Read from :mod:`menagerie.crawler.rung_stratification` through a LAZY
    import: the stratifier imports this module at module scope, so the reverse
    edge only exists at call time. The vocabulary is closed on purpose -- an
    attestation naming a stratum no composer can produce is not an attestation
    a composer wrote.
    """

    from menagerie.crawler.rung_stratification import (  # noqa: PLC0415 -- lazy: reverse of the composer's own import
        LADDER_BULK_STRATUM,
        TAIL_STRATA,
    )

    return frozenset({LADDER_BULK_STRATUM, CARRIED_TIP_STRATUM, *TAIL_STRATA})


def _composition_attestation_from_row(
    row: Mapping[str, Any], *, source: str
) -> RungCompositionAttestation:
    """Validate one raw attestation object into a typed attestation.

    Parameters
    ----------
    row:
        Raw JSON attestation.
    source:
        Diagnostic origin (path or ``build``).

    Returns
    -------
    RungCompositionAttestation
        Typed validated attestation.

    Raises
    ------
    CompositionAttestationError
        If the object is malformed, names an unknown stratum, or its
        ``composition_digest`` does not match its own body.
    """

    schema_version = row.get("schema_version")
    if schema_version not in (
        RUNG_COMPOSITION_ATTESTATION_SCHEMA_VERSION,
        RUNG_COMPOSITION_ATTESTATION_SCHEMA_VERSION_V1,
    ):
        raise CompositionAttestationError(
            f"{source}: composition attestation schema must be "
            f"{RUNG_COMPOSITION_ATTESTATION_SCHEMA_VERSION} (or the pre-PD-7 "
            f"{RUNG_COMPOSITION_ATTESTATION_SCHEMA_VERSION_V1})"
        )
    pd7_disposition: Optional[dict[str, str]] = None
    if schema_version == RUNG_COMPOSITION_ATTESTATION_SCHEMA_VERSION:
        pd7_raw = row.get("pd7")
        if not isinstance(pd7_raw, Mapping):
            raise CompositionAttestationError(
                f"{source}: a v2 composition attestation must carry the pd7 "
                "disposition block (SOL-02); the composer consumes the PD-7 "
                "adjudication, a stage never trusts a side file"
            )
        if set(pd7_raw) != {"status", "source_attestation_sha256"}:
            raise CompositionAttestationError(
                f"{source}: pd7 block keys must be exactly status and "
                f"source_attestation_sha256, got {sorted(pd7_raw)}"
            )
        if pd7_raw.get("status") not in PD7_DISPOSITION_VOCABULARY:
            raise CompositionAttestationError(
                f"{source}: pd7 status {pd7_raw.get('status')!r} is outside the "
                f"closed vocabulary {sorted(PD7_DISPOSITION_VOCABULARY)}"
            )
        if (
            not isinstance(pd7_raw.get("source_attestation_sha256"), str)
            or _SHA256_PATTERN.fullmatch(pd7_raw["source_attestation_sha256"]) is None
        ):
            raise CompositionAttestationError(
                f"{source}: pd7 source_attestation_sha256 must be a sha256: digest "
                "of the adjudication artifact the composer consumed"
            )
        pd7_disposition = {
            "status": str(pd7_raw["status"]),
            "source_attestation_sha256": str(pd7_raw["source_attestation_sha256"]),
        }
    elif "pd7" in row:
        raise CompositionAttestationError(
            f"{source}: a v1 composition attestation may not carry a pd7 block; "
            "a PD-7-bearing attestation declares the v2 schema"
        )
    if not isinstance(row.get("campaign_id"), str) or not row["campaign_id"]:
        raise CompositionAttestationError(
            f"{source}: composition attestation campaign_id must be non-empty"
        )
    for field in ("size", "seed"):
        value = row.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            raise CompositionAttestationError(
                f"{source}: composition attestation {field} must be an integer"
            )
    if (
        not isinstance(row.get("snapshot_id"), str)
        or _SNAPSHOT_ID_PATTERN.fullmatch(row["snapshot_id"]) is None
    ):
        raise CompositionAttestationError(
            f"{source}: composition attestation snapshot_id is not a snapshot id"
        )
    if (
        not isinstance(row.get("snapshot_sha256"), str)
        or _SHA256_PATTERN.fullmatch(row["snapshot_sha256"]) is None
    ):
        raise CompositionAttestationError(
            f"{source}: composition attestation snapshot_sha256 is invalid"
        )
    raw_pairs = row.get("member_strata")
    if not isinstance(raw_pairs, Sequence) or isinstance(raw_pairs, (str, bytes)) or not raw_pairs:
        raise CompositionAttestationError(
            f"{source}: composition attestation member_strata must be a non-empty list"
        )
    vocabulary = _closed_stratum_vocabulary()
    pairs: list[tuple[str, str]] = []
    for entry in raw_pairs:
        if (
            not isinstance(entry, Sequence)
            or isinstance(entry, (str, bytes))
            or len(entry) != 2
            or not all(isinstance(value, str) and value for value in entry)
        ):
            raise CompositionAttestationError(
                f"{source}: composition attestation member_strata rows must be "
                "[stable_id, stratum] pairs"
            )
        stable_id, stratum = str(entry[0]), str(entry[1])
        if stratum not in vocabulary:
            raise CompositionAttestationError(
                f"{source}: composition attestation places {stable_id} in unknown "
                f"stratum {stratum!r}; the closed vocabulary is {sorted(vocabulary)}"
            )
        pairs.append((stable_id, stratum))
    if len({stable_id for stable_id, _ in pairs}) != len(pairs):
        raise CompositionAttestationError(f"{source}: composition attestation names a member twice")
    ordered = tuple(sorted(pairs))
    body = _composition_attestation_body(
        campaign_id=str(row["campaign_id"]),
        size=int(row["size"]),
        seed=int(row["seed"]),
        member_strata=ordered,
        snapshot_id=str(row["snapshot_id"]),
        snapshot_sha256=str(row["snapshot_sha256"]),
        pd7_disposition=pd7_disposition,
    )
    if row.get("composition_digest") != stable_hash(body):
        raise CompositionAttestationError(
            f"{source}: composition_digest does not match the attestation body -- "
            "the attestation was edited after the composer signed it, or its "
            "member_strata were re-ordered by hand"
        )
    return RungCompositionAttestation(
        campaign_id=str(row["campaign_id"]),
        size=int(row["size"]),
        seed=int(row["seed"]),
        member_strata=ordered,
        snapshot_id=str(row["snapshot_id"]),
        snapshot_sha256=str(row["snapshot_sha256"]),
        composition_digest=str(row["composition_digest"]),
        pd7_disposition=pd7_disposition,
    )


def load_composition_attestation(path: Path) -> RungCompositionAttestation:
    """Load one composition attestation from a composer artifact.

    Accepts either the attestation object itself or the full stratified rung
    plan report that embeds it under
    :data:`COMPOSITION_ATTESTATION_REPORT_KEY`, so the operator can hand the
    staging door the exact ``--out-report`` file the composer wrote.

    Parameters
    ----------
    path:
        JSON attestation or strata-report path.

    Returns
    -------
    RungCompositionAttestation
        Typed validated attestation.

    Raises
    ------
    CompositionAttestationError
        If the file is missing, unreadable, not JSON, carries no attestation,
        or the attestation is malformed.
    """

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CompositionAttestationError(
            f"cannot read composition attestation {path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise CompositionAttestationError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise CompositionAttestationError(f"{path} must contain a JSON object")
    if COMPOSITION_ATTESTATION_REPORT_KEY in raw:
        embedded = raw[COMPOSITION_ATTESTATION_REPORT_KEY]
        if not isinstance(embedded, Mapping):
            raise CompositionAttestationError(
                f"{path}: {COMPOSITION_ATTESTATION_REPORT_KEY} must be a JSON object"
            )
        return _composition_attestation_from_row(
            embedded, source=f"{path}:{COMPOSITION_ATTESTATION_REPORT_KEY}"
        )
    return _composition_attestation_from_row(raw, source=str(path))


def verify_composition_attestation(
    attestation: RungCompositionAttestation,
    *,
    campaign_id: str,
    member_ids: Sequence[str],
    snapshot_id: str,
    snapshot_sha256: str,
    label: str,
    rung_class: Optional[str] = None,
) -> None:
    """Refuse a stage the attestation does not describe (B-03).

    The attestation's own ``composition_digest`` is re-derived here rather than
    assumed: the decode path already checks it, but this gate must not depend on
    HOW its argument was constructed, or a hand-assembled dataclass would reach
    the comparisons below carrying a digest nothing ever verified.

    Then three independent facts are recomputed from the stage itself:

    #. the campaign -- an attestation for another campaign adjudicates nothing;
    #. the member SET -- a hand-edited or stale member list refuses; and
    #. the composed ``(snapshot_id, snapshot_sha256)`` -- the fact that fails
       when the attestation is STALE, i.e. the same members no longer compose to
       the identity that was attested because the records state moved under it.

    Parameters
    ----------
    attestation:
        Decoded composer attestation.
    campaign_id:
        Campaign the stage is binding to.
    member_ids:
        Members actually being staged.
    snapshot_id, snapshot_sha256:
        Identity the staged members actually compose to (the planned identity
        for :func:`stage_rung`, the verified adopted identity for
        :func:`register_existing_snapshot`).
    label:
        Rung label, for the diagnostic.
    rung_class:
        The stage's DECLARED rung class, threading the PD-7 consumption rule
        (SOL-02): a ``successor_rung`` stage demands a digest-bound
        ``pd7.status == SATISFIED``; ANY stage refuses a
        ``UNSATISFIED-BLOCKING`` disposition.

    Raises
    ------
    CompositionAttestationError
        If any of the three facts disagrees with the attestation.
    """

    _composition_attestation_from_row(attestation.to_row(), source=f"rung {label}")
    if attestation.campaign_id != campaign_id:
        raise CompositionAttestationError(
            f"rung {label!r}: the composition attestation was composed for "
            f"{attestation.campaign_id}, not {campaign_id}"
        )
    staged = frozenset(str(value) for value in member_ids)
    attested = attestation.member_ids
    if staged != attested:
        missing = sorted(attested - staged)
        extra = sorted(staged - attested)
        raise CompositionAttestationError(
            f"rung {label!r}: the staged member set is not the composed member "
            f"set ({len(attested)} attested, {len(staged)} staged; "
            f"{len(missing)} attested-but-absent {missing[:5]}, "
            f"{len(extra)} staged-but-unattested {extra[:5]}). Stage the member "
            "list the composer emitted, or re-compose"
        )
    if (attestation.snapshot_id, attestation.snapshot_sha256) != (snapshot_id, snapshot_sha256):
        raise CompositionAttestationError(
            f"rung {label!r}: these members compose to {snapshot_id} "
            f"({snapshot_sha256}), but the attestation names "
            f"{attestation.snapshot_id} ({attestation.snapshot_sha256}). The "
            "attestation is STALE or its snapshot identity was edited by hand: "
            "the records state it was composed against is not the state being "
            "staged. Re-compose against the current state -- a snapshot sha is "
            "never carried forward by hand"
        )
    _verify_pd7_disposition(attestation, label=label, rung_class=rung_class)


def _verify_pd7_disposition(
    attestation: Optional[RungCompositionAttestation],
    *,
    label: str,
    rung_class: Optional[str],
) -> None:
    """Consume the digest-bound PD-7 disposition at the staging door (SOL-02).

    A ``UNSATISFIED-BLOCKING`` disposition refuses EVERY stage it reaches; a
    ``successor_rung`` stage additionally demands a v2 attestation whose bound
    ``pd7.status`` is exactly ``SATISFIED`` — an absent attestation, a
    pre-vocabulary v1 attestation, and a ``NOT-EVALUATED`` status are the SAME
    missing-consumption failure. The doors call this with the attestation the
    operator actually presented, so a green side file the attestation does not
    bind licenses nothing.
    """

    pd7 = None if attestation is None else attestation.pd7_disposition
    pd7_status = None if pd7 is None else str(pd7.get("status"))
    if pd7_status == PD7_DISPOSITION_BLOCKING:
        raise CompositionAttestationError(
            f"rung {label!r}: the composition attestation binds PD-7 disposition "
            f"{PD7_DISPOSITION_BLOCKING}; a known-blocking slop-family-routes "
            "adjudication never stages anything"
        )
    if rung_class == RUNG_CLASS_SUCCESSOR and pd7_status != PD7_DISPOSITION_SATISFIED:
        raise CompositionAttestationError(
            f"rung {label!r}: a successor-rung stage demands a composition "
            f"attestation whose digest-bound pd7.status is "
            f"{PD7_DISPOSITION_SATISFIED!r} (found "
            f"{'no attestation' if attestation is None else 'no pd7 block' if pd7 is None else pd7_status!r}); "
            "B-03 consumes the PD-7 adjudication itself — a green side file the "
            "attestation does not bind licenses nothing (SOL-02)"
        )


def rung_registry_path(records_root: Path) -> Path:
    """Return the append-only rung registration ledger path.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.

    Returns
    -------
    pathlib.Path
        ``records/intake/registered-rungs.jsonl``.
    """

    return records_root / "intake" / "registered-rungs.jsonl"


def currency_carve_out_present() -> bool:
    """Return whether the authority intake-currency carve-out has landed.

    Returns
    -------
    bool
        ``True`` only when :data:`CURRENCY_CARVE_OUT_ATTRIBUTE` is declared
        ``True`` on :mod:`menagerie.crawler.authority`.
    """

    from menagerie.crawler import authority

    return getattr(authority, CURRENCY_CARVE_OUT_ATTRIBUTE, False) is True


def build_rung_registration(
    *,
    campaign_id: str,
    snapshot_id: str,
    snapshot_sha256: str,
    derived_from_snapshot_id: str,
    member_count: int,
    label: str,
    registered_by: str,
    created_at: str,
    licensed_by_close_sha256: Optional[str] = None,
    license_kind: Optional[str] = None,
    closes_chain_head: Optional[str] = None,
    rung_class: Optional[str] = None,
) -> RungRegistration:
    """Assemble one content-addressed registration row.

    Parameters
    ----------
    campaign_id, snapshot_id, snapshot_sha256, derived_from_snapshot_id:
        Binding identity and lineage.
    member_count:
        Number of intake items in the rung.
    label, registered_by, created_at:
        Operator provenance.
    rung_class:
        Closed staging-door class declaration (:data:`RUNG_CLASSES`); required
        whenever the row carries a ``live_origin_custody`` licence.

    Returns
    -------
    RungRegistration
        Validated row with its content address.

    Raises
    ------
    RungStagingError
        If any field is empty or malformed.
    """

    body: JsonObject = {
        "schema_version": RUNG_REGISTRATION_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "snapshot_id": snapshot_id,
        "snapshot_sha256": snapshot_sha256,
        "derived_from_snapshot_id": derived_from_snapshot_id,
        "member_count": member_count,
        "label": label,
        "registered_by": registered_by,
        "created_at": created_at,
    }
    if licensed_by_close_sha256 is not None:
        body.update(
            {
                "licensed_by_close_sha256": licensed_by_close_sha256,
                "license_kind": license_kind,
                "closes_chain_head": closes_chain_head,
            }
        )
    if rung_class is not None:
        body["rung_class"] = rung_class
    return _registration_from_row(
        stamp_row_digest(body), source="build"
    )


def _registration_from_row(row: Mapping[str, Any], *, source: str) -> RungRegistration:
    """Validate one raw registry row into a typed registration.

    Parameters
    ----------
    row:
        Raw JSON row.
    source:
        Diagnostic origin (path or ``build``).

    Returns
    -------
    RungRegistration
        Typed validated row.

    Raises
    ------
    RungStagingError
        If the row is malformed or its content address does not match.
    """

    if row.get("schema_version") != RUNG_REGISTRATION_SCHEMA_VERSION:
        raise RungStagingError(
            f"{source}: rung registration schema must be {RUNG_REGISTRATION_SCHEMA_VERSION}"
        )
    for field in ("campaign_id", "label", "registered_by", "created_at"):
        if not isinstance(row.get(field), str) or not row[field]:
            raise RungStagingError(f"{source}: rung registration {field} must be non-empty")
    for field in ("snapshot_id", "derived_from_snapshot_id"):
        if (
            not isinstance(row.get(field), str)
            or _SNAPSHOT_ID_PATTERN.fullmatch(row[field]) is None
        ):
            raise RungStagingError(f"{source}: rung registration {field} is not a snapshot id")
    if (
        not isinstance(row.get("snapshot_sha256"), str)
        or _SHA256_PATTERN.fullmatch(row["snapshot_sha256"]) is None
    ):
        raise RungStagingError(f"{source}: rung registration snapshot_sha256 is invalid")
    member_count = row.get("member_count")
    if not isinstance(member_count, int) or isinstance(member_count, bool) or member_count < 1:
        raise RungStagingError(f"{source}: rung registration member_count must be positive")
    licence_fields = (
        "licensed_by_close_sha256",
        "license_kind",
        "closes_chain_head",
    )
    present_licence_fields = [field for field in licence_fields if row.get(field) is not None]
    if present_licence_fields and len(present_licence_fields) != len(licence_fields):
        raise RungStagingError(f"{source}: rung registration has a partial licence binding")
    if present_licence_fields and row.get("license_kind") not in (
        "live_origin_custody",
        "historical_bootstrap",
    ):
        raise RungStagingError(f"{source}: rung registration has an unknown licence kind")
    rung_class = row.get("rung_class")
    if rung_class is not None and rung_class not in RUNG_CLASSES:
        raise RungStagingError(
            f"{source}: rung registration rung_class {rung_class!r} is outside the "
            f"closed vocabulary {sorted(RUNG_CLASSES)}"
        )
    if rung_class is None and row.get("license_kind") == "live_origin_custody":
        # The pre-vocabulary era contains no live-licensed rows (the registry
        # predates close-licenses-successor), so an unclassed live-licensed
        # row is forged or mis-staged, never legacy — fail closed.
        raise RungStagingError(
            f"{source}: live-licensed rung registration lacks a declared rung_class; "
            "every staging under a live close declares its class at the door"
        )
    foreign = foreign_content_address_fields(row, own="row_sha256")
    if foreign:
        raise RungStagingError(
            f"{source}: rung registration carries a foreign content-address field "
            f"{', '.join(foreign)}: a registration owns exactly one content address"
        )
    if row.get("row_sha256") != row_content_address(row):
        raise RungStagingError(f"{source}: rung registration row_sha256 does not match its body")
    return RungRegistration(
        campaign_id=str(row["campaign_id"]),
        snapshot_id=str(row["snapshot_id"]),
        snapshot_sha256=str(row["snapshot_sha256"]),
        derived_from_snapshot_id=str(row["derived_from_snapshot_id"]),
        member_count=member_count,
        label=str(row["label"]),
        registered_by=str(row["registered_by"]),
        created_at=str(row["created_at"]),
        row_sha256=str(row["row_sha256"]),
        licensed_by_close_sha256=(
            str(row["licensed_by_close_sha256"])
            if row.get("licensed_by_close_sha256") is not None
            else None
        ),
        license_kind=(str(row["license_kind"]) if row.get("license_kind") is not None else None),
        closes_chain_head=(
            str(row["closes_chain_head"]) if row.get("closes_chain_head") is not None else None
        ),
        rung_class=(str(row["rung_class"]) if row.get("rung_class") is not None else None),
        chain=(dict(row["chain"]) if isinstance(row.get("chain"), Mapping) else None),
    )


def _decode_registry_lines(data: bytes, path: Path) -> list[RungRegistration]:
    """Decode complete registry lines in byte order.

    Parameters
    ----------
    data:
        Raw ledger bytes.
    path:
        Ledger path for diagnostics.

    Returns
    -------
    list[RungRegistration]
        Every persisted registration in append order.

    Raises
    ------
    RungStagingError
        If any complete line is malformed.
    """

    rows: list[RungRegistration] = []
    for line_number, line in enumerate(data.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RungStagingError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise RungStagingError(f"{path}:{line_number} must contain a JSON object")
        rows.append(_registration_from_row(value, source=f"{path}:{line_number}"))
    return rows


def load_rung_registrations(records_root: Path) -> tuple[RungRegistration, ...]:
    """Load every durable rung registration in append order.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.

    Returns
    -------
    tuple[RungRegistration, ...]
        Registrations in ledger order; empty when no rung was ever staged.

    Raises
    ------
    RungStagingError
        If the ledger holds a malformed or duplicate-identity row.
    """

    path = rung_registry_path(records_root)
    if not path.is_file():
        return ()
    from menagerie.crawler.ledger_chain import LedgerChainError, decode_ledger, verified_chain

    # Round-2 T4 (scale): the decode and the chain verification are HELD, so the
    # eight close/staging consumers that ask for the registry during ONE
    # licensing decision verify each row once, not once per consumer.
    raw_rows = decode_ledger(path)
    try:
        verified_chain(path, scope="meta.registry")
    except LedgerChainError as exc:
        raise RungStagingError(str(exc)) from exc
    rows = [_registration_from_row(row, source=str(path)) for row in raw_rows]
    seen: dict[str, RungRegistration] = {}
    for row in rows:
        prior = seen.get(row.snapshot_id)
        if prior is not None:
            # I-TIP (rt3 residual, now a reader invariant): the two registry
            # writers never append a duplicate -- an idempotent restage returns
            # the prior row without writing -- so ANY duplicate snapshot id,
            # byte-identical included, is a third-writer signature. Tolerating
            # it would move ``registrations[-1]`` and silently rewind the
            # campaign's lineage tip past the close gate.
            raise RungStagingError(
                f"{path}: snapshot {row.snapshot_id} is registered twice -- the "
                "registry writers are append-once, so a duplicate row rewinds "
                "the lineage tip and refuses outright"
            )
        seen[row.snapshot_id] = row
    return tuple(rows)


def _binding_identity(
    registration: RungRegistration,
) -> tuple[str, str, str, int, Optional[str], Optional[str], Optional[str]]:
    """Return the fields that define one snapshot's binding, ignoring provenance."""

    return (
        registration.campaign_id,
        registration.snapshot_sha256,
        registration.derived_from_snapshot_id,
        registration.member_count,
        registration.licensed_by_close_sha256,
        registration.license_kind,
        registration.closes_chain_head,
    )


def _registered_prior_lineage(
    intake_root: Path, active_snapshot_id: str
) -> dict[str, tuple[str, Path]]:
    """Resolve the active snapshot's registered prior lineage with durable roots.

    The ONE lineage spelling shared by the currency half
    (:func:`registered_prior_intake_snapshots`) and the membership half
    (:func:`registered_prior_intake_members`), so the two carve-out inputs can
    never disagree about which snapshots the lineage contains: the campaign
    ROOT snapshot (at its manifest ``intake_path``) and every rung registered
    BEFORE the active one (at ``records/intake/<snapshot-id>``), for the SAME
    campaign only.

    Parameters
    ----------
    intake_root:
        Active intake snapshot directory (``records/intake/<snapshot-id>``).
    active_snapshot_id:
        Canonical active snapshot identity.

    Returns
    -------
    dict[str, tuple[str, pathlib.Path]]
        ``snapshot_id -> (snapshot_sha256, durable snapshot root)`` for the
        prior registered lineage; empty for an unregistered active snapshot
        or an unreadable registry (fail-closed: no carve-out granted).
    """

    records_root = intake_root.parent.parent
    if not rung_registry_path(records_root).is_file():
        return {}
    try:
        registrations = load_rung_registrations(records_root)
    except RungStagingError:
        # Fail-closed: an unreadable registry grants no carve-out.
        return {}
    active = next((row for row in registrations if row.snapshot_id == active_snapshot_id), None)
    if active is None:
        return {}
    prior: dict[str, tuple[str, Path]] = {}
    manifest_path = records_root / "campaigns.json"
    if manifest_path.is_file():
        try:
            bindings: Sequence[CampaignBinding] = load_campaign_bindings(manifest_path)
        except CampaignPartitionError:
            bindings = ()
        for binding in bindings:
            if binding.spec.campaign_id == active.campaign_id:
                prior[binding.intake_snapshot_id] = (
                    binding.intake_snapshot_sha256,
                    records_root / binding.intake_path,
                )
    for row in registrations:
        if row.snapshot_id == active_snapshot_id:
            break
        if row.campaign_id == active.campaign_id:
            prior[row.snapshot_id] = (
                row.snapshot_sha256,
                records_root / "intake" / row.snapshot_id,
            )
    prior.pop(active_snapshot_id, None)
    return prior


def registered_prior_intake_snapshots(intake_root: Path, active_snapshot_id: str) -> dict[str, str]:
    """Return the registered prior snapshots of the active snapshot's campaign.

    The lineage is read from the durable rung registry beside the intake
    snapshots (``registered-rungs.jsonl``) plus the frozen campaign manifest:
    the campaign ROOT snapshot and every rung registered BEFORE the active one,
    for the SAME campaign only. An unregistered active snapshot -- including
    every campaign root and every pre-staging snapshot -- yields an empty map,
    which keeps exact-pin semantics exactly as they were.

    This reader lives DRIVER-SIDE by design: :mod:`menagerie.crawler.authority`
    is a sealed worker-closure member whose static import graph must never
    reach this module (its graph reaches catalog-side ``menagerie.identity``).
    The driver calls this and hands the resulting facts to
    ``build_authority_context(prior_intake_snapshots=...)``.

    Parameters
    ----------
    intake_root:
        Active intake snapshot directory (``records/intake/<snapshot-id>``).
    active_snapshot_id:
        Canonical active snapshot identity.

    Returns
    -------
    dict[str, str]
        ``snapshot_id -> snapshot_sha256`` for the prior registered lineage.
    """

    return {
        snapshot_id: snapshot_sha256
        for snapshot_id, (snapshot_sha256, _root) in _registered_prior_lineage(
            intake_root, active_snapshot_id
        ).items()
    }


def registered_prior_intake_members(intake_root: Path, active_snapshot_id: str) -> frozenset[str]:
    """Return the member union of the active snapshot's registered prior lineage.

    The MEMBERSHIP half of the T4-T1 carve-out: the reducer's construction-time
    check needs to know which stable ids belong to REGISTERED PRIOR
    same-campaign snapshots so a SUBSET rung can open over its superset history
    as read-only history. Like :func:`registered_prior_intake_snapshots`, this
    reader lives DRIVER-SIDE; the driver hands the facts to
    ``build_authority_context(prior_intake_member_ids=...)`` and the reducer
    never reads the registry itself.

    Every lineage snapshot is re-verified before it licenses members: its
    durable bytes must load through the real content-addressed loader and
    reproduce the exact REGISTERED ``(snapshot_id, snapshot_sha256)`` identity.
    A missing, unreadable, or tampered snapshot contributes NO members
    (fail-closed: the reducer then refuses its revisions exactly as before).

    Parameters
    ----------
    intake_root:
        Active intake snapshot directory (``records/intake/<snapshot-id>``).
    active_snapshot_id:
        Canonical active snapshot identity.

    Returns
    -------
    frozenset[str]
        Union of member stable ids across the verified prior lineage; empty
        for an unregistered active snapshot (the exact-pin world unchanged).
    """

    members: set[str] = set()
    for snapshot_id, (snapshot_sha256, snapshot_root) in _registered_prior_lineage(
        intake_root, active_snapshot_id
    ).items():
        try:
            snapshot = load_intake_snapshot(snapshot_root)
        except (OSError, ValueError):
            # Fail-closed: bytes that cannot re-verify license no members.
            continue
        if snapshot.snapshot_id != snapshot_id or snapshot.snapshot_sha256 != snapshot_sha256:
            continue
        members.update(item.stable_id for item in snapshot.items)
    return frozenset(members)


def find_rung_registration(
    records_root: Path,
    snapshot_id: str,
    snapshot_sha256: str,
) -> Optional[RungRegistration]:
    """Find the registration owning one exact snapshot identity.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    snapshot_id, snapshot_sha256:
        Exact snapshot identity.

    Returns
    -------
    RungRegistration or None
        Owning registration, or ``None`` for an unregistered snapshot.
    """

    for registration in load_rung_registrations(records_root):
        if (
            registration.snapshot_id == snapshot_id
            and registration.snapshot_sha256 == snapshot_sha256
        ):
            return registration
    return None


def append_rung_registration(path: Path, registration: RungRegistration) -> StagedRung:
    """Append one registration durably and idempotently.

    Parameters
    ----------
    path:
        Append-only registry ledger path.
    registration:
        Complete typed registration row.

    Returns
    -------
    StagedRung
        Persisted registration; ``registration_created`` is ``False`` when an
        identical binding already exists.

    Raises
    ------
    RungStagingError
        If the snapshot id is already registered with a different binding.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        from menagerie.crawler.ledger_chain import decode_ledger, link_row, verify_rows

        existing_raw = decode_ledger(path, locked_handle=handle)
        verify_rows(existing_raw, scope="meta.registry")
        existing = [_registration_from_row(row, source=str(path)) for row in existing_raw]
        for prior in existing:
            if prior.snapshot_id != registration.snapshot_id:
                continue
            if _binding_identity(prior) == _binding_identity(registration):
                return StagedRung(
                    registration=prior,
                    snapshot_root=path.parent / prior.snapshot_id,
                    snapshot_created=False,
                    registration_created=False,
                )
            raise RungStagingError(
                f"snapshot {registration.snapshot_id} is already registered to "
                f"{prior.campaign_id} (lineage {prior.derived_from_snapshot_id}); "
                "registrations are append-only and never rebind"
            )
        linked = link_row(
            registration.to_row(), scope="meta.registry", existing_rows=existing_raw
        )
        persisted = _registration_from_row(linked, source="append")
        handle.seek(0, os.SEEK_END)
        handle.write(canonical_json_bytes(persisted.to_row()) + b"\n")
        handle.flush()
        durable_fsync(handle.fileno())
    fsync_directory(path.parent)
    return StagedRung(
        registration=persisted,
        snapshot_root=path.parent / persisted.snapshot_id,
        snapshot_created=False,
        registration_created=True,
    )


def rung_class_refinement_path(records_root: Path) -> Path:
    """Return the append-once rung-class refinement ledger path.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.

    Returns
    -------
    pathlib.Path
        ``records/intake/rung-class-refinements.jsonl``.
    """

    return records_root / "intake" / "rung-class-refinements.jsonl"


@dataclass(frozen=True)
class RungClassRefinement:
    """One append-once class refinement of a pre-vocabulary registration.

    Parameters
    ----------
    campaign_id, snapshot_id, snapshot_sha256:
        Exact identity of the registration being refined.
    registration_row_sha256:
        Content address of the EXACT registry row the refinement binds; a
        refinement never transfers to a different row for the same snapshot.
    rung_class:
        Declared class from the closed :data:`RUNG_CLASSES` vocabulary.
    refined_by, created_at:
        Operator provenance.
    row_sha256:
        Content address of this row itself.
    """

    campaign_id: str
    snapshot_id: str
    snapshot_sha256: str
    registration_row_sha256: str
    rung_class: str
    refined_by: str
    created_at: str
    row_sha256: str
    chain: Optional[Mapping[str, Any]] = None

    def to_row(self) -> JsonObject:
        """Return the canonical persisted JSON row."""

        row: JsonObject = {
            "schema_version": RUNG_CLASS_REFINEMENT_SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "snapshot_id": self.snapshot_id,
            "snapshot_sha256": self.snapshot_sha256,
            "registration_row_sha256": self.registration_row_sha256,
            "rung_class": self.rung_class,
            "refined_by": self.refined_by,
            "created_at": self.created_at,
            "row_sha256": self.row_sha256,
        }
        if self.chain is not None:
            row["chain"] = dict(self.chain)
        return row


def _refinement_from_row(row: Mapping[str, Any], *, source: str) -> RungClassRefinement:
    """Validate one raw refinement row into a typed refinement."""

    if row.get("schema_version") != RUNG_CLASS_REFINEMENT_SCHEMA_VERSION:
        raise RungStagingError(
            f"{source}: rung-class refinement schema must be "
            f"{RUNG_CLASS_REFINEMENT_SCHEMA_VERSION}"
        )
    for field in ("campaign_id", "refined_by", "created_at"):
        if not isinstance(row.get(field), str) or not row[field]:
            raise RungStagingError(f"{source}: rung-class refinement {field} must be non-empty")
    if (
        not isinstance(row.get("snapshot_id"), str)
        or _SNAPSHOT_ID_PATTERN.fullmatch(row["snapshot_id"]) is None
    ):
        raise RungStagingError(f"{source}: rung-class refinement snapshot_id is not a snapshot id")
    if (
        not isinstance(row.get("snapshot_sha256"), str)
        or _SHA256_PATTERN.fullmatch(row["snapshot_sha256"]) is None
    ):
        raise RungStagingError(f"{source}: rung-class refinement snapshot_sha256 is invalid")
    if (
        not isinstance(row.get("registration_row_sha256"), str)
        or _SHA256_PATTERN.fullmatch(row["registration_row_sha256"]) is None
    ):
        raise RungStagingError(
            f"{source}: rung-class refinement registration_row_sha256 is invalid"
        )
    if row.get("rung_class") not in RUNG_CLASSES:
        raise RungStagingError(
            f"{source}: rung-class refinement class {row.get('rung_class')!r} is outside "
            f"the closed vocabulary {sorted(RUNG_CLASSES)}"
        )
    foreign = foreign_content_address_fields(row, own="row_sha256")
    if foreign:
        raise RungStagingError(
            f"{source}: rung-class refinement carries a foreign content-address field "
            f"{', '.join(foreign)}"
        )
    if row.get("row_sha256") != row_content_address(row):
        raise RungStagingError(
            f"{source}: rung-class refinement row_sha256 does not match its body"
        )
    return RungClassRefinement(
        campaign_id=str(row["campaign_id"]),
        snapshot_id=str(row["snapshot_id"]),
        snapshot_sha256=str(row["snapshot_sha256"]),
        registration_row_sha256=str(row["registration_row_sha256"]),
        rung_class=str(row["rung_class"]),
        refined_by=str(row["refined_by"]),
        created_at=str(row["created_at"]),
        row_sha256=str(row["row_sha256"]),
        chain=(dict(row["chain"]) if isinstance(row.get("chain"), Mapping) else None),
    )


def load_rung_class_refinements(records_root: Path) -> tuple[RungClassRefinement, ...]:
    """Load every durable rung-class refinement, append-once verified.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.

    Returns
    -------
    tuple[RungClassRefinement, ...]
        Refinements in ledger order; empty when none was ever appended.

    Raises
    ------
    RungStagingError
        If the ledger holds a malformed row or a second refinement for one
        snapshot (append-once: a class refinement is decided exactly once).
    """

    path = rung_class_refinement_path(records_root)
    if not path.is_file():
        return ()
    from menagerie.crawler.ledger_chain import LedgerChainError, decode_ledger, verified_chain

    raw_rows = decode_ledger(path)
    try:
        verified_chain(path, scope="meta.class-refinements")
    except LedgerChainError as exc:
        raise RungStagingError(str(exc)) from exc
    rows = [_refinement_from_row(row, source=str(path)) for row in raw_rows]
    seen: set[str] = set()
    for row in rows:
        if row.snapshot_id in seen:
            raise RungStagingError(
                f"{path}: snapshot {row.snapshot_id} carries two class refinements -- "
                "a refinement is append-once; a second row rewrites the class an "
                "earlier reviewed decision already declared and refuses outright"
            )
        seen.add(row.snapshot_id)
    return tuple(rows)


def effective_rung_class(
    records_root: Path, registration: RungRegistration
) -> Optional[str]:
    """Return one registration's governing class: declared, else refined.

    A class DECLARED at the staging door always wins (it is bound into the
    registration's content address). A pre-vocabulary unclassed row is
    governed by its append-once refinement when one exists; a refinement whose
    exact registration binding does not match the live row refuses typed
    rather than silently reading as unclassed (fail closed, never fail open).

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    registration:
        The registry row whose class is being derived.

    Returns
    -------
    str or None
        The governing class, or ``None`` for an unrefined pre-vocabulary row.
    """

    if registration.rung_class is not None:
        return registration.rung_class
    for refinement in load_rung_class_refinements(records_root):
        if refinement.snapshot_id != registration.snapshot_id:
            continue
        if (
            refinement.snapshot_sha256 != registration.snapshot_sha256
            or refinement.registration_row_sha256 != registration.row_sha256
        ):
            raise RungStagingError(
                f"rung-class refinement {refinement.row_sha256} names snapshot "
                f"{refinement.snapshot_id} but binds registration row "
                f"{refinement.registration_row_sha256}, which is not the live "
                f"registry row {registration.row_sha256}; a refinement never "
                "transfers to a row it did not bind"
            )
        return refinement.rung_class
    return None


def append_rung_class_refinement(
    records_root: Path,
    *,
    snapshot_id: str,
    snapshot_sha256: str,
    rung_class: str,
    refined_by: str,
    created_at: str,
) -> tuple[RungClassRefinement, bool]:
    """Durably class ONE pre-vocabulary registration, append-once (R2-4/SOL-01).

    The refinement door is deliberately narrow:

    * the target registration must exist with this EXACT snapshot identity;
    * its ``rung_class`` must be absent (a class declared at the staging door
      is immutable — refinement exists only for pre-vocabulary rows);
    * the registration must be its campaign's current registry TIP — the rung
      the next leg consumes. Reclassifying mid-lineage history would rewrite
      what an already-closed leg was licensed as; and
    * exactly one refinement may ever exist per snapshot: an identical
      re-append returns idempotently, a conflicting class refuses.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    snapshot_id, snapshot_sha256:
        Exact identity of the registration being refined.
    rung_class:
        Declared class from :data:`RUNG_CLASSES`.
    refined_by, created_at:
        Operator provenance.

    Returns
    -------
    tuple[RungClassRefinement, bool]
        The governing refinement row and whether this call appended it.

    Raises
    ------
    RungStagingError
        On any narrowness violation above.
    """

    if rung_class not in RUNG_CLASSES:
        raise RungStagingError(
            f"rung_class {rung_class!r} is outside the closed vocabulary {sorted(RUNG_CLASSES)}"
        )
    registration = find_rung_registration(records_root, snapshot_id, snapshot_sha256)
    if registration is None:
        raise RungStagingError(
            f"no registration exists for snapshot {snapshot_id} ({snapshot_sha256}); "
            "a refinement classes an existing registered rung, never a future one"
        )
    if registration.rung_class is not None:
        raise RungStagingError(
            f"snapshot {snapshot_id} declared rung_class {registration.rung_class!r} at "
            "its staging door; a declared class is immutable and never refined"
        )
    campaign_rows = [
        row
        for row in load_rung_registrations(records_root)
        if row.campaign_id == registration.campaign_id
    ]
    if campaign_rows and campaign_rows[-1].snapshot_id != snapshot_id:
        raise RungStagingError(
            f"snapshot {snapshot_id} is not campaign {registration.campaign_id}'s registry "
            f"tip ({campaign_rows[-1].snapshot_id} is); only the rung the next leg "
            "consumes may be classed by refinement — mid-lineage history stays as it "
            "was licensed"
        )
    existing = next(
        (
            row
            for row in load_rung_class_refinements(records_root)
            if row.snapshot_id == snapshot_id
        ),
        None,
    )
    if existing is not None:
        if (
            existing.rung_class == rung_class
            and existing.snapshot_sha256 == snapshot_sha256
            and existing.registration_row_sha256 == registration.row_sha256
        ):
            return existing, False
        raise RungStagingError(
            f"snapshot {snapshot_id} already carries the refinement "
            f"{existing.rung_class!r} ({existing.row_sha256}); refinements are "
            "append-once and never rebind"
        )
    body: JsonObject = {
        "schema_version": RUNG_CLASS_REFINEMENT_SCHEMA_VERSION,
        "campaign_id": registration.campaign_id,
        "snapshot_id": snapshot_id,
        "snapshot_sha256": snapshot_sha256,
        "registration_row_sha256": registration.row_sha256,
        "rung_class": rung_class,
        "refined_by": refined_by,
        "created_at": created_at,
    }
    candidate = _refinement_from_row(stamp_row_digest(body), source="build")
    path = rung_class_refinement_path(records_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        from menagerie.crawler.ledger_chain import decode_ledger, link_row, verify_rows

        existing_raw = decode_ledger(path, locked_handle=handle)
        verify_rows(existing_raw, scope="meta.class-refinements")
        for prior_raw in existing_raw:
            prior = _refinement_from_row(prior_raw, source=str(path))
            if prior.snapshot_id != snapshot_id:
                continue
            if (
                prior.rung_class == rung_class
                and prior.snapshot_sha256 == snapshot_sha256
                and prior.registration_row_sha256 == registration.row_sha256
            ):
                return prior, False
            raise RungStagingError(
                f"snapshot {snapshot_id} already carries the refinement "
                f"{prior.rung_class!r} ({prior.row_sha256}); refinements are "
                "append-once and never rebind"
            )
        linked = link_row(
            candidate.to_row(), scope="meta.class-refinements", existing_rows=existing_raw
        )
        persisted = _refinement_from_row(linked, source="append")
        handle.seek(0, os.SEEK_END)
        handle.write(canonical_json_bytes(persisted.to_row()) + b"\n")
        handle.flush()
        durable_fsync(handle.fileno())
    fsync_directory(path.parent)
    return persisted, True


def _refuse_conflicting_class_claim(
    records_root: Path,
    registration: RungRegistration,
    declared_rung_class: Optional[str],
) -> None:
    """Refuse an idempotent restage that would silently drop its class claim.

    R2-4/SOL-01: the append-once registry returns the prior row on an
    idempotent restage, so a ``--rung-class`` declared at that restage used to
    be ACCEPTED AND DISCARDED — the operator believed the successor class
    landed while every certificate consumer kept deriving the pre-launch kind.
    A declared class that the existing registration does not already carry now
    refuses loudly and names the one lawful door (the append-once refinement).
    """

    if declared_rung_class is None:
        return
    effective = effective_rung_class(records_root, registration)
    if effective == declared_rung_class:
        return
    raise RungStagingError(
        f"snapshot {registration.snapshot_id} is already registered with rung_class "
        f"{effective!r}; an idempotent restage never adopts the declared class "
        f"{declared_rung_class!r} (the registry is append-once). Class a "
        "pre-vocabulary registration through the append-once refinement door: "
        "python -m menagerie.crawler.tools.stage_rung refine-class "
        f"--snapshot-id {registration.snapshot_id} --rung-class {declared_rung_class}"
    )


def _campaign_root_binding(records_root: Path, campaign_id: str) -> CampaignBinding:
    """Resolve one frozen campaign's manifest root binding.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    campaign_id:
        Requested frozen tier campaign.

    Returns
    -------
    CampaignBinding
        Manifest binding for the campaign's full-partition snapshot.

    Raises
    ------
    RungStagingError
        If the manifest is absent or the campaign is not one of the frozen four.
    """

    manifest_path = records_root / "campaigns.json"
    if not manifest_path.is_file():
        raise RungStagingError(
            f"no campaign manifest at {manifest_path}; rungs can only be staged "
            "inside a partitioned campaign repository"
        )
    bindings = load_campaign_bindings(manifest_path)
    for binding in bindings:
        if binding.spec.campaign_id == campaign_id:
            return binding
    known = ", ".join(binding.spec.campaign_id for binding in bindings)
    raise RungStagingError(
        f"unknown campaign {campaign_id!r}; the frozen manifest binds only: {known}"
    )


def resolve_verified_root(
    records_root: Path, campaign_id: str
) -> tuple[CampaignBinding, IntakeSnapshot]:
    """Resolve one campaign's root binding and VERIFY it against the bytes.

    The manifest's root binding used to be declared-but-never-verified on the
    composition and staging paths (rt4 F5): both loaded
    ``records_root / binding.intake_path`` and never compared the loaded
    ``(snapshot_id, snapshot_sha256)`` to the frozen binding, so a root
    re-emitted in place under its old directory name -- a catalog epoch --
    silently rebased the whole ladder while every attestation kept copying the
    frozen (wrong) identity out of the manifest. This is the ONE resolver every
    composition and staging path must use: it returns the binding only after

    * the durable root snapshot loads through the content-addressed loader AND
      reproduces the bound ``(intake_snapshot_id, intake_snapshot_sha256)``;
    * the campaign partition bytes exist and hash to the bound
      ``partition_sha256`` (composition reads those bytes next, so absence is
      a refusal here, not a later ``OSError``).

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    campaign_id:
        Requested frozen tier campaign.

    Returns
    -------
    tuple[CampaignBinding, IntakeSnapshot]
        The verified binding and the loaded, verified root snapshot.

    Raises
    ------
    RungStagingError
        If the manifest is absent, the campaign is unknown, the root snapshot
        does not load or is not the bound identity, or the partition bytes are
        absent or not the bound digest.
    """

    binding = _campaign_root_binding(records_root, campaign_id)
    snapshot_root = records_root / binding.intake_path
    try:
        snapshot = load_intake_snapshot(snapshot_root)
    except (OSError, ValueError) as exc:
        raise RungStagingError(
            f"cannot verify {campaign_id}'s root binding: the bound root snapshot "
            f"at {snapshot_root} does not load ({exc})"
        ) from exc
    if (
        snapshot.snapshot_id != binding.intake_snapshot_id
        or snapshot.snapshot_sha256 != binding.intake_snapshot_sha256
    ):
        raise RungStagingError(
            f"{campaign_id}'s root snapshot at {snapshot_root} is not the frozen "
            f"binding: the manifest binds {binding.intake_snapshot_id} / "
            f"{binding.intake_snapshot_sha256} but the loaded bytes are "
            f"{snapshot.snapshot_id} / {snapshot.snapshot_sha256}. A root "
            "re-emitted in place (a catalog epoch) must be re-bound through the "
            "manifest, never silently composed under the frozen identity."
        )
    partition_path = records_root / binding.partition_path
    try:
        partition_bytes = partition_path.read_bytes()
    except OSError as exc:
        raise RungStagingError(
            f"cannot verify {campaign_id}'s root binding: the bound partition "
            f"{partition_path} is unreadable ({exc}); composition and staging "
            "read those bytes, so their digest must verify first"
        ) from exc
    observed_partition = hash_bytes(partition_bytes)
    if observed_partition != binding.partition_sha256:
        raise RungStagingError(
            f"{campaign_id}'s partition bytes at {partition_path} hash to "
            f"{observed_partition}, not the frozen binding's "
            f"{binding.partition_sha256}: the scheduling order is not the one "
            "the manifest froze"
        )
    return binding, snapshot


def registration_binding(
    records_root: Path,
    registration: RungRegistration,
) -> CampaignBinding:
    """Project one registered rung as a campaign binding.

    The campaign spec (author model, checker model, review checkpoint) comes
    from the frozen manifest root binding; only the intake identity and row
    count are the rung's own.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    registration:
        Durable rung registration.

    Returns
    -------
    CampaignBinding
        Binding carrying the rung snapshot identity under its campaign spec.
    """

    root = _campaign_root_binding(records_root, registration.campaign_id)
    return CampaignBinding(
        spec=root.spec,
        row_count=registration.member_count,
        partition_path=root.partition_path,
        partition_sha256=root.partition_sha256,
        intake_path=f"intake/{registration.snapshot_id}",
        intake_snapshot_id=registration.snapshot_id,
        intake_snapshot_sha256=registration.snapshot_sha256,
        review_cohort_size=root.review_cohort_size,
    )


def _require_currency_carve_out(campaign_id: str) -> None:
    """Refuse staging while the K2 authority carve-out is absent.

    Parameters
    ----------
    campaign_id:
        Campaign being staged, for the diagnostic.

    Raises
    ------
    CurrencyCarveOutMissingError
        If :mod:`menagerie.crawler.authority` does not declare the carve-out.
    """

    if currency_carve_out_present():
        return
    raise CurrencyCarveOutMissingError(
        f"cannot stage a new {campaign_id} rung: menagerie.crawler.authority does "
        f"not declare {CURRENCY_CARVE_OUT_ATTRIBUTE}. Without the intake-currency "
        "carve-out (terminals recorded under a PRIOR registered snapshot of the "
        "same campaign stay current), activating a new snapshot id projects every "
        "prior terminal dependency-stale and re-drives it. Land the C-M1-KERNEL "
        "(K2) authority carve-out first; registering an EXISTING already-run "
        "snapshot for durability does not require it."
    )


def _lineage_member_ids(records_root: Path, registration: RungRegistration) -> frozenset[str]:
    """Return the durable membership of one registered rung, or refuse.

    The close gate's coverage arm is recomputed against the members the
    lineage snapshot's OWN bytes contain, re-verified through the
    content-addressed loader against the registered identity. Bytes that
    cannot be loaded, or that no longer reproduce the registered identity,
    prove nothing and refuse (fail-closed).

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    registration:
        Durable registration of the lineage rung.

    Returns
    -------
    frozenset[str]
        Stable ids of the lineage snapshot's members.

    Raises
    ------
    RungStagingError
        If the durable snapshot is absent, unreadable, or not the registered
        identity.
    """

    snapshot_root = records_root / "intake" / registration.snapshot_id
    try:
        snapshot = load_intake_snapshot(snapshot_root)
    except (OSError, ValueError) as exc:
        raise RungStagingError(
            f"the lineage rung {registration.snapshot_id} has no loadable durable "
            f"snapshot at {snapshot_root}: its close cannot be checked against a "
            f"membership nothing can read ({exc})"
        ) from exc
    if (
        snapshot.snapshot_id != registration.snapshot_id
        or snapshot.snapshot_sha256 != registration.snapshot_sha256
    ):
        raise RungStagingError(
            f"the durable snapshot at {snapshot_root} is not the registered "
            f"identity {registration.snapshot_id} / {registration.snapshot_sha256}"
        )
    return frozenset(item.stable_id for item in snapshot.items)


def introduced_member_ids(
    records_root: Path, registration: RungRegistration
) -> frozenset[str]:
    """Return members no registered strict ancestor of this rung contained."""

    root_binding, _root_snapshot = resolve_verified_root(
        records_root, registration.campaign_id
    )
    registrations = {
        row.snapshot_id: row
        for row in load_rung_registrations(records_root)
        if row.campaign_id == registration.campaign_id
    }
    ancestor_members: set[str] = set()
    cursor = registration.derived_from_snapshot_id
    seen: set[str] = set()
    while cursor != root_binding.intake_snapshot_id and cursor in registrations:
        if cursor in seen:
            raise RungStagingError(f"registered rung lineage cycles at {cursor}")
        seen.add(cursor)
        ancestor = registrations[cursor]
        ancestor_members.update(_lineage_member_ids(records_root, ancestor))
        cursor = ancestor.derived_from_snapshot_id
    return _lineage_member_ids(records_root, registration) - ancestor_members


def lineage_tip_membership(
    records_root: Path, campaign_id: str
) -> Optional[tuple[RungRegistration, frozenset[str]]]:
    """Return one campaign's registry tip and its durable membership.

    The tip a successor must widen, in the door's own terms, exposed for the
    composer that has to CARRY it: same registry-tip rule (the campaign's last
    registration), same membership authority (:func:`_lineage_member_ids`, which
    re-verifies the tip snapshot's bytes against its registered identity), so the
    set a composition carries cannot differ from the set the door enforces.
    Unverifiable bytes refuse rather than silently shrinking the carried set.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    campaign_id:
        Campaign whose registry tip is wanted.

    Returns
    -------
    tuple[RungRegistration, frozenset[str]] or None
        The tip and its members, or ``None`` when the campaign has no
        registered rung (the root exemption case).

    Raises
    ------
    RungStagingError
        If the tip's durable snapshot is absent, unreadable, or not the
        registered identity.
    """

    owned = [row for row in load_rung_registrations(records_root) if row.campaign_id == campaign_id]
    if not owned:
        return None
    tip = owned[-1]
    return tip, _lineage_member_ids(records_root, tip)


def _verify_carried_tip_claim(
    attestation: Optional[RungCompositionAttestation],
    tip_members: Optional[frozenset[str]],
    *,
    label: str,
    tip_description: str,
) -> None:
    """Refuse a ``carried-tip`` claim the campaign's registry tip does not support.

    Rung membership is cumulative, so a composition carries the tip forward and
    labels those members :data:`CARRIED_TIP_STRATUM` (C-2). That label is inside
    the composition digest, which makes it evidence -- and evidence a producer
    writes is exactly what may never be trusted. The door therefore recomputes
    the tip's membership from the registry and refuses the one direction the
    label could launder: a member CLAIMING carry provenance it does not have,
    which would present operator-facing disclosure ("inherited, already
    adjudicated") for a model the tip never contained.

    The other direction needs no rule: a tip member labelled by some other
    stratum is staged either way, because the widen rule independently forces
    every tip member into the successor's membership.

    Parameters
    ----------
    attestation:
        Presented composition attestation, or ``None`` (nothing claimed).
    tip_members:
        The registry tip's recomputed membership, or ``None`` when the campaign
        has no registered rung.
    label:
        Rung label, for the diagnostic.
    tip_description:
        How to name the tip in the diagnostic.

    Raises
    ------
    CompositionAttestationError
        If the attestation claims carry provenance for a non-tip member, or
        claims it at all while the campaign has no registered lineage tip.
    """

    if attestation is None:
        return
    claimed = attestation.carried_tip_member_ids
    if not claimed:
        return
    if tip_members is None:
        raise CompositionAttestationError(
            f"rung {label!r}: the composition claims {len(claimed)} member(s) were "
            f"carried from the lineage tip ({sorted(claimed)[:5]}), but this campaign "
            "has no registered lineage tip to carry -- a first rung inherits nothing"
        )
    unfounded = sorted(claimed - tip_members)
    if unfounded:
        raise CompositionAttestationError(
            f"rung {label!r}: the composition claims {len(unfounded)} member(s) were "
            f"carried from the lineage tip {tip_description}, which does not contain "
            f"them: {unfounded[:5]}. Carry provenance is recomputed from the "
            "registry, never read from the attestation -- re-compose against the "
            "current tip"
        )


def _licensed_close_chain_head(
    records_root: Path, licensing_close: Optional[Mapping[str, Any]]
) -> Optional[str]:
    """Return the exact Track A close-chain anchor for a licensing close."""

    if licensing_close is None:
        return None
    from menagerie.crawler.ledger_chain import close_chain_head

    return close_chain_head(records_root, str(licensing_close["close_sha256"]))


def _require_closed_lineage(
    records_root: Path,
    root_binding: CampaignBinding,
    lineage: str,
    *,
    label: str,
    staged_snapshot_id: str,
    staged_member_ids: Iterable[str],
    composition_attestation: Optional[RungCompositionAttestation] = None,
) -> Optional[JsonObject]:
    """Refuse a new rung until the campaign's live rung has an immutable close.

    BD-1's staging half (T1-sol-02). A rung boundary is crossed on a durable,
    content-addressed close attestation and on nothing else: at the step-100
    halt the newest driver JSON was an INTERRUPTION record while the boundary
    gate exited 1, and every downstream decision was made from a hand-tallied
    spreadsheet of raw heads. The gate re-verifies the attested ledger bytes
    against the ledgers as they are now, so a close that no longer describes
    the state being staged from must be re-attested.

    The gated lineage comes from the STATE, never from the argument
    (2026-08-16 red-team round 2, break 1). The root exemption is sound only
    while the campaign has ZERO registered rungs -- a root never ran as a rung,
    so nothing can have closed it, and demanding a close there would make the
    first rung of every campaign unstageable. Once ANY rung is registered, the
    campaign's registry tip is the lineage, and a ``derived_from`` naming
    anything else (the root included) is an operator claim this door does not
    adjudicate: the exemption written for rung 1 was reusable at rung N simply
    by naming the root, which skipped the gate before any re-verification ran.

    The close itself is then bound to that tip by campaign, by registered
    digest, by the tip's real membership, and by the heads the models ledger
    actually holds (breaks 2 and 3), so matching an id string licenses nothing.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    root_binding:
        Manifest root binding for the campaign being staged.
    lineage:
        Resolved ``derived_from_snapshot_id``.
    label:
        Diagnostic name of the rung being staged.
    staged_snapshot_id:
        Identity of the rung being staged. An ALREADY REGISTERED identity is an
        idempotent restage: it crosses no boundary, so it demands no close.
    composition_attestation:
        Presented composition attestation, whose ``carried-tip`` claims are
        adjudicated here against the tip this function already recomputes
        (:func:`_verify_carried_tip_claim`). A restage crosses no boundary and
        re-adjudicates nothing, exactly as with the close itself.

    Raises
    ------
    CloseObjectError
        If the campaign's registry tip has no licensing close object.
    RungStagingError
        If the named lineage is not the campaign's registry tip, the tip's
        durable snapshot cannot be re-verified, or the attestation claims carry
        provenance the tip does not support.
    """

    registrations = [
        registration
        for registration in load_rung_registrations(records_root)
        if registration.campaign_id == root_binding.spec.campaign_id
    ]
    if any(registration.snapshot_id == staged_snapshot_id for registration in registrations):
        # Idempotent restage of an already registered identity: the append is a
        # no-op (or refuses as a contradiction), so no boundary is crossed.
        return None
    if not registrations:
        # The campaign's FIRST rung: its only possible lineage is the root, and
        # a root never ran as a rung. It also inherits nothing, so a carry claim
        # here is unfounded by construction.
        _verify_carried_tip_claim(
            composition_attestation, None, label=label, tip_description="(none)"
        )
        return None
    tip = registrations[-1]
    if lineage != tip.snapshot_id:
        raise RungStagingError(
            f"cannot stage {label!r} from {lineage}: "
            f"{root_binding.spec.campaign_id}'s registered lineage tip is "
            f"{tip.snapshot_id} ({tip.label!r}). A rung boundary is crossed on the "
            "state's own lineage, never on a lineage the operator names -- "
            "re-deriving from the campaign root or an older rung routes around the "
            "close gate the tip must satisfy. Derive from the tip (closing it "
            "first), or retire it in its own reviewed lane."
        )
    # The tip's membership is recomputed from its own re-verified bytes, the
    # same authority :func:`lineage_tip_membership` hands the composer, and
    # deliberately AFTER the lineage-mismatch refusal above: naming the wrong
    # lineage is an operator error to report as such, not as unreadable bytes.
    tip_members = _lineage_member_ids(records_root, tip)
    _verify_carried_tip_claim(
        composition_attestation,
        tip_members,
        label=label,
        tip_description=f"{tip.snapshot_id} ({tip.label!r})",
    )
    staged_members = frozenset(str(value) for value in staged_member_ids)
    if not tip_members < staged_members:
        raise RungStagingError(
            f"cannot stage {label!r}: a successor must strictly widen the lineage tip "
            f"({len(tip_members)} -> {len(staged_members)} members). Re-open a "
            "generation on the tip for retries, or retire it in its reviewed lane."
        )
    from menagerie.crawler.close_object import (  # noqa: PLC0415 -- lazy: keeps staging import-light
        require_closed_lineage,
    )

    return require_closed_lineage(
        records_root,
        intake_snapshot_id=tip.snapshot_id,
        campaign_id=tip.campaign_id,
        intake_snapshot_sha256=tip.snapshot_sha256,
        lineage_member_ids=tip_members,
        introduced_member_ids=introduced_member_ids(records_root, tip),
        label=label,
    )


def _resolve_lineage(
    records_root: Path,
    root_binding: CampaignBinding,
    derived_from_snapshot_id: Optional[str],
    *,
    staged_snapshot_id: Optional[str] = None,
) -> str:
    """Resolve and validate one rung's ``derived_from_snapshot_id``.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    root_binding:
        Manifest root binding for the campaign being staged.
    derived_from_snapshot_id:
        Explicit prior snapshot, or ``None`` for the campaign's lineage tip.
    staged_snapshot_id:
        Identity of the rung being staged; a rung never derives from itself,
        so a restage of the lineage tip resolves to the tip's own lineage.

    Returns
    -------
    str
        Validated prior snapshot id: the campaign root or a registered rung.

    Raises
    ------
    RungStagingError
        If the explicit lineage names a snapshot this campaign does not own.
    """

    campaign_registrations = [
        registration
        for registration in load_rung_registrations(records_root)
        if registration.campaign_id == root_binding.spec.campaign_id
    ]
    if derived_from_snapshot_id is None:
        for registration in campaign_registrations:
            if registration.snapshot_id == staged_snapshot_id:
                return registration.derived_from_snapshot_id
        if campaign_registrations:
            return campaign_registrations[-1].snapshot_id
        return root_binding.intake_snapshot_id
    known = {root_binding.intake_snapshot_id} | {
        registration.snapshot_id for registration in campaign_registrations
    }
    if derived_from_snapshot_id not in known:
        raise RungStagingError(
            f"derived_from_snapshot_id {derived_from_snapshot_id!r} is neither "
            f"{root_binding.spec.campaign_id}'s root snapshot "
            f"{root_binding.intake_snapshot_id} nor one of its "
            f"{len(campaign_registrations)} registered rungs"
        )
    return derived_from_snapshot_id


def partition_member_stable_ids(
    records_root: Path,
    root_binding: CampaignBinding,
    root_snapshot: IntakeSnapshot,
) -> tuple[str, ...]:
    """Return every campaign partition row as a stable id, in scheduling order.

    Partition order is the campaign's frozen scheduling order (for c1-mech the
    stratified review cohort comes first). This is the ONE reader of that
    order shared by the ladder-prefix ``--count`` shape and the stratified
    rung composer, so the two can never disagree about what "next partition
    rows" means.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    root_binding:
        Manifest root binding naming the partition file.
    root_snapshot:
        Campaign root snapshot supplying stable ids.

    Returns
    -------
    tuple[str, ...]
        Stable ids of every partition row, in partition order.

    Raises
    ------
    RungStagingError
        If the partition cannot be read or a row has no intake item.
    """

    partition_path = records_root / root_binding.partition_path
    by_key = {item.natural_key: item.stable_id for item in root_snapshot.items}
    selected: list[str] = []
    try:
        lines = partition_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RungStagingError(f"cannot read campaign partition {partition_path}: {exc}") from exc
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["name"]), str(row["zoo"]), str(row.get("variant", "")))
        stable_id = by_key.get(key)
        if stable_id is None:
            raise RungStagingError(
                f"partition row {key!r} has no item in the campaign root snapshot"
            )
        selected.append(stable_id)
    return tuple(selected)


def _partition_prefix_ids(
    records_root: Path,
    root_binding: CampaignBinding,
    root_snapshot: IntakeSnapshot,
    count: int,
) -> tuple[str, ...]:
    """Select the first ``count`` partition rows as stable ids.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    root_binding:
        Manifest root binding naming the partition file.
    root_snapshot:
        Campaign root snapshot supplying stable ids.
    count:
        Number of leading partition rows to select.

    Returns
    -------
    tuple[str, ...]
        Stable ids of the partition-prefix members.

    Raises
    ------
    RungStagingError
        If the partition is shorter than ``count`` or a row has no intake item.
    """

    members = partition_member_stable_ids(records_root, root_binding, root_snapshot)
    if len(members) < count:
        raise RungStagingError(
            f"campaign partition holds {len(members)} rows; cannot stage a {count}-model rung"
        )
    return members[:count]


def _plan_subset_snapshot(
    root_snapshot_dir: Path,
    stable_ids: Sequence[str],
) -> tuple[str, str, bytes, bytes, tuple[str, ...]]:
    """Compute a subset snapshot's identity and bytes WITHOUT writing anything.

    Planning is separated from writing so every staging gate runs before the
    first durable byte exists (2026-08-16 red-team break 2: the subset was
    written before the close gate, so a refused stage left a ready-made orphan
    snapshot under ``records/intake/`` for the ungated register door to adopt).

    Parameters
    ----------
    root_snapshot_dir:
        Verified campaign root snapshot directory.
    stable_ids:
        Members to keep.

    Returns
    -------
    tuple[str, str, bytes, bytes, tuple[str, ...]]
        Snapshot id, snapshot sha256, manifest bytes, items bytes, and source
        file names.
    """

    manifest = json.loads((root_snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    wanted = set(stable_ids)
    items = [item for item in manifest["items"] if item["stable_id"] in wanted]
    basis = {
        "format": manifest["format"],
        "sources": manifest["sources"],
        "stable_ids": manifest["stable_ids"],
        "items": items,
    }
    snapshot_sha256 = stable_hash(basis)
    snapshot_id = f"intake-{snapshot_sha256.removeprefix('sha256:')[:20]}"
    out_manifest = {
        **basis,
        "snapshot_id": snapshot_id,
        "snapshot_sha256": snapshot_sha256,
        "item_count": len(items),
    }
    manifest_bytes = canonical_json_bytes(out_manifest) + b"\n"
    items_bytes = b"".join(canonical_json_bytes(item) + b"\n" for item in items)
    return snapshot_id, snapshot_sha256, manifest_bytes, items_bytes, tuple(manifest["sources"])


def planned_subset_identity(
    records_root: Path,
    campaign_id: str,
    stable_ids: Sequence[str],
) -> tuple[str, str]:
    """Return the identity one member set composes to, writing nothing.

    The ONE spelling of "what would staging these members produce", shared by
    the composer's attestation (B-03) and the staging door that re-verifies it,
    so the two can never disagree about the identity under discussion.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    campaign_id:
        Frozen tier campaign supplying the root snapshot.
    stable_ids:
        Members to compose.

    Returns
    -------
    tuple[str, str]
        ``(snapshot_id, snapshot_sha256)`` of the composed subset.

    Raises
    ------
    RungStagingError
        If the campaign is unknown or its root snapshot is unreadable.
    """

    root_binding, _root_snapshot = resolve_verified_root(records_root, campaign_id)
    snapshot_id, snapshot_sha256, _, _, _ = _plan_subset_snapshot(
        records_root / root_binding.intake_path, stable_ids
    )
    return snapshot_id, snapshot_sha256


def _write_subset_snapshot(
    root_snapshot_dir: Path,
    stable_ids: Sequence[str],
    intake_root: Path,
) -> tuple[IntakeSnapshot, bool]:
    """Materialize an immutable subset snapshot from one verified root.

    The subset keeps the root's exact source bytes and stable-ids digest and
    subsets its items in manifest order, so the result is byte-stable for a
    given member set and validates through the real loader. Callers holding
    gates must run them BEFORE this writes (see :func:`_plan_subset_snapshot`).

    Parameters
    ----------
    root_snapshot_dir:
        Verified campaign root snapshot directory.
    stable_ids:
        Members to keep.
    intake_root:
        Durable intake store receiving the subset snapshot.

    Returns
    -------
    tuple[IntakeSnapshot, bool]
        Loaded verified subset snapshot and whether new bytes were written.

    Raises
    ------
    RungStagingError
        If an existing directory holds different bytes for the same identity.
    """

    snapshot_id, _snapshot_sha256, manifest_bytes, items_bytes, source_names = (
        _plan_subset_snapshot(root_snapshot_dir, stable_ids)
    )
    snapshot_root = intake_root / snapshot_id
    manifest_path = snapshot_root / "manifest.json"
    created = True
    if manifest_path.exists():
        if manifest_path.read_bytes() != manifest_bytes:
            raise RungStagingError(f"snapshot identity collision at {snapshot_root}")
        if (snapshot_root / "items.jsonl").read_bytes() != items_bytes:
            raise RungStagingError(f"snapshot items changed at {snapshot_root}")
        created = False
    else:
        for name in source_names:
            atomic_replace_bytes(
                snapshot_root / "sources" / name,
                (root_snapshot_dir / "sources" / name).read_bytes(),
            )
        atomic_replace_bytes(snapshot_root / "items.jsonl", items_bytes)
        atomic_replace_bytes(manifest_path, manifest_bytes)
    return load_intake_snapshot(snapshot_root), created


def _validate_membership(
    stable_ids: Sequence[str],
    root_snapshot: IntakeSnapshot,
    campaign_id: str,
) -> tuple[str, ...]:
    """Validate rung members against the campaign root membership.

    Parameters
    ----------
    stable_ids:
        Requested members in request order.
    root_snapshot:
        Campaign root snapshot defining the membership universe.
    campaign_id:
        Campaign for diagnostics.

    Returns
    -------
    tuple[str, ...]
        De-duplicated validated members.

    Raises
    ------
    RungStagingError
        If the selection is empty or names ids outside the campaign.
    """

    unique = tuple(dict.fromkeys(str(value) for value in stable_ids))
    if not unique:
        raise RungStagingError("a rung needs at least one stable id")
    known = {item.stable_id for item in root_snapshot.items}
    missing = sorted(set(unique) - known)
    if missing:
        raise RungStagingError(
            f"{len(missing)} stable ids are not members of {campaign_id}: {missing[:5]}"
        )
    return unique


def _models_ledger_paths(records_root: Path) -> tuple[Path, ...]:
    """Return EVERY models-ledger layout present, failing CLOSED when none is.

    Both supported layouts exist in the tree -- the sharded production layout
    (``models/current-shard.jsonl``) and the flat driver-rig layout
    (``models.jsonl``) -- and both are consulted. Preferring whichever exists
    FIRST turned the absent-shard refusal's own remedy into the next escape
    (2026-08-16 red-team round 2, break 4): following "materialize the models
    ledger first -- an EMPTY ledger file is the explicit statement that no
    awards exist" in a flat-layout root put a zero-byte shard in front of the
    ledger holding the awards, and the AW-2 door went silent over real
    operator property. An ABSENT ledger is still not the absence of awards; it
    is the absence of proof.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Every existing models ledger path, sharded layout first.

    Raises
    ------
    RungStagingError
        If no models ledger exists in either layout.
    """

    from menagerie.crawler.leg_generation import (  # noqa: PLC0415 -- lazy: keeps staging import-light
        LegGenerationError,
        current_model_heads,
        models_source_paths,
    )

    present = models_source_paths(records_root)
    if present:
        return present
    try:
        current_model_heads(records_root, require_ledger=True)
    except LegGenerationError as exc:
        raise RungStagingError(str(exc)) from exc
    raise RungStagingError(f"no models ledger exists under {records_root} (AW-2)")


def _current_model_heads(records_root: Path) -> Mapping[str, Mapping[str, Any]]:
    """Return the raw current model heads below one records root.

    Every present layout contributes its heads (the union), and a member whose
    two layouts disagree about its current revision is AMBIGUOUS authority:
    there is no ledger the door can call THE ledger, and silently picking one
    is how an award-holding head disappears. Ambiguity refuses.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.

    Returns
    -------
    Mapping[str, Mapping[str, Any]]
        Stable id to current raw head, unioned across layouts.

    Raises
    ------
    RungStagingError
        If no models ledger exists (:func:`_models_ledger_paths`), or two
        layouts disagree about a member's current head.
    """

    from menagerie.crawler.leg_generation import (  # noqa: PLC0415 -- lazy: keeps staging import-light
        LegGenerationError,
        current_model_heads,
    )

    try:
        return current_model_heads(records_root, require_ledger=True)
    except LegGenerationError as exc:
        raise RungStagingError(str(exc)) from exc


def _recorded_snapshot_of(head: Mapping[str, Any]) -> Optional[str]:
    """Return the intake snapshot one raw head was recorded under, if declared."""

    vector = head.get("dependency_vector")
    if not isinstance(vector, Mapping):
        return None
    recorded = vector.get("intake_snapshot_id")
    return recorded if isinstance(recorded, str) else None


def _require_recorded_history(
    records_root: Path,
    snapshot: IntakeSnapshot,
    member_ids: Sequence[str],
) -> None:
    """Prove an adopted snapshot is already-run history, or refuse.

    ``historical_only`` was a caller-supplied boolean (and the shipped CLI's
    silent default), so nothing distinguished an already-run snapshot from a
    freshly composed one and the AW-2 runs-holder gate was skippable at will
    (2026-08-16 red-team break 3). History is now EVIDENCE: every member must
    hold a current raw head recorded under the exact snapshot being adopted
    (``dependency_vector.intake_snapshot_id``). A snapshot proven to be its
    members' own recorded run may carry their awards -- they are that run's
    output, not an unauthorized staging of operator property.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    snapshot:
        Verified snapshot being adopted.
    member_ids:
        Validated snapshot members.

    Raises
    ------
    RungStagingError
        If any member holds no head, or a head recorded elsewhere.
    """

    # I7 (2026-08-16 tri-lab contract): the register door's historical-adoption
    # proof and the close gate's coverage authority are the SAME proof type. A
    # snapshot with a resident v2 close whose sealed proof accounts EVERY
    # adopted member is proven history -- this is what lets the pilot's
    # step-100 shape (33 runs + deferred + review-helds) re-adopt, where the
    # per-head rule below refuses on the first parked member (a parked member
    # has no head by definition).
    from menagerie.crawler.close_object import (  # noqa: PLC0415 -- lazy: keeps staging import-light
        CLOSE_OBJECT_SCHEMA_VERSION,
        close_object_path,
        find_close_object,
        load_close_objects,
    )
    from menagerie.crawler.partition_proof import (  # noqa: PLC0415
        PartitionProofError,
        load_partition_proof,
    )

    close = find_close_object(
        load_close_objects([close_object_path(records_root)]),
        intake_snapshot_id=snapshot.snapshot_id,
    )
    if close is not None and close.get("schema_version") == CLOSE_OBJECT_SCHEMA_VERSION:
        try:
            proof = load_partition_proof(records_root, str(close["partition_proof_sha256"]))
        except PartitionProofError:
            proof = None
        if proof is not None and set(dict(proof.get("members") or {})) >= {
            str(stable_id) for stable_id in member_ids
        }:
            return

    heads = _current_model_heads(records_root)
    unproven: list[str] = []
    for stable_id in member_ids:
        head = heads.get(str(stable_id))
        recorded = None if head is None else _recorded_snapshot_of(head)
        if recorded != snapshot.snapshot_id:
            unproven.append(
                f"{stable_id} ("
                + ("no recorded head" if head is None else f"head recorded under {recorded}")
                + ")"
            )
    if unproven:
        raise RungStagingError(
            f"{len(unproven)} member(s) do not prove {snapshot.snapshot_id} is "
            "already-run history (a historical adoption requires every member's "
            "current head to be recorded under that exact snapshot): "
            + "; ".join(unproven[:5])
            + ". Registering a freshly composed snapshot for NEW work must pass "
            "historical_only=False / --new-work and satisfy the same "
            "award-authority guard as stage_rung."
        )


def _runs_award_demand(
    heads: Mapping[str, Mapping[str, Any]], member_ids: Sequence[str]
) -> dict[str, str]:
    """Return ``stable_id -> exact current revision`` for every runs-award member.

    ONE predicate for "which members hold a runs award": the AW-2 refusal below
    and the composition-time DISCLOSURE
    (:func:`required_award_authorities`) both read it, so the authority list an
    operator is told to pass can never drift from the list the door demands.

    Parameters
    ----------
    heads:
        Raw current model heads by stable id.
    member_ids:
        Members under consideration.

    Returns
    -------
    dict[str, str]
        Award-holding members and the exact revision an authority must name.
    """

    demanded: dict[str, str] = {}
    for stable_id in member_ids:
        head = heads.get(str(stable_id))
        if head is None:
            continue
        status = head.get("status")
        if not isinstance(status, Mapping) or status.get("kind") != "runs":
            continue
        demanded[str(stable_id)] = str(head.get("record_revision"))
    return demanded


def required_award_authorities(records_root: Path, member_ids: Sequence[str]) -> dict[str, str]:
    """Return the AW-2 authorities a stage of these members will demand.

    DISCLOSURE ONLY, and deliberately not fail-closed: an absent models ledger
    reads here as "no award evidence to show", because this function steers no
    disposition -- :func:`_refuse_unauthorized_runs_holders` re-derives the same
    demand on the staging path with ``require_ledger=True``, where no ledger
    refuses outright. A composer calls this so its report can name every
    ``--award-authority stable_id=revision`` the door will ask for (the carried
    lineage tip's converted models above all), instead of leaving the operator
    to hand-tally them from the ledger.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    member_ids:
        Members a stage would contain.

    Returns
    -------
    dict[str, str]
        Award-holding members and the exact revision each authority must name.

    Raises
    ------
    RungStagingError
        If two models layouts disagree about a member's current head, which is
        ambiguous authority on every path that reads it.
    """

    from menagerie.crawler.leg_generation import (  # noqa: PLC0415 -- lazy: keeps staging import-light
        LegGenerationError,
        current_model_heads,
    )

    try:
        heads = current_model_heads(records_root, require_ledger=False)
    except LegGenerationError as exc:
        raise RungStagingError(str(exc)) from exc
    return _runs_award_demand(heads, member_ids)


def _refuse_unauthorized_runs_holders(
    records_root: Path,
    member_ids: Sequence[str],
    award_authorities: Optional[Mapping[str, str]],
) -> None:
    """Refuse staging any ``runs`` holder without an exact award authority (AW-2).

    The staging tripwire: pool composition excludes runs holders through the
    stratifier, but hand-built overlays, explicit stable-id lists, and count
    prefixes reached registration unchecked -- the only award guard left fired
    at TERMINAL APPEND, after the author and execution spend (each of the ten
    close-state award park rows is a full end-to-end re-author the guard could
    only refuse at the last step). Every member holding a raw ``runs`` head
    must be named by an explicit operator authority binding the EXACT current
    revision; a head that moved between composition and registration therefore
    refuses too. An absent models ledger refuses outright
    (:func:`_models_ledger_paths`): no ledger, no proof, no staging.

    The demand covers every staged member, which since C-2 includes the members
    a cumulative successor CARRIES from its lineage tip: a settled award is
    operator property whether the rung earned it or inherited it, so carrying
    one without its authority refuses exactly like staging one fresh.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    member_ids:
        Snapshot members about to be registered.
    award_authorities:
        Operator-supplied ``stable_id -> exact raw runs record_revision``
        acknowledgments.

    Raises
    ------
    RungStagingError
        If any member holds a runs head the authorities do not exactly name,
        or no models ledger exists at all.
    """

    authorities = dict(award_authorities or {})
    unauthorized = [
        f"{stable_id} (runs head {revision})"
        for stable_id, revision in _runs_award_demand(
            _current_model_heads(records_root), member_ids
        ).items()
        if authorities.get(stable_id) != revision
    ]
    if unauthorized:
        raise RungStagingError(
            f"{len(unauthorized)} member(s) hold runs awards and are operator "
            "property: a snapshot may not stage them without an explicit "
            "award authority naming each exact current revision "
            "(award_authorities / --award-authority stable_id=revision): "
            + "; ".join(unauthorized[:5])
        )


def _environment_registry_for_staging(target: str) -> Any:
    """Load the LIVE committed environment registry for the compose door (EN-2).

    Imported lazily: staging is records-side and only this gate needs the
    environment machinery.
    """

    from menagerie.crawler.envs import DEFAULT_ENVS_ROOT, load_environment_registry

    return load_environment_registry(DEFAULT_ENVS_ROOT, target=target)


def refuse_lockless_member_routes(
    members: Sequence[str],
    snapshot: IntakeSnapshot,
    *,
    registry: Any,
    allow_lockless: Sequence[str] = (),
) -> None:
    """Refuse any member whose derived route lands on a lockless non-exempt intent.

    EN-2: the route is derived with THE driver derivation
    (``constants.framework_from_markers`` -> ``routing.route_model``), so the
    compose door and the admission door can never disagree. ``allow_lockless``
    is validated against the LIVE registry: an entry naming an intent that is
    actually locked refuses as stale (an exemption is operator word about the
    present, never a tool-written field to be trusted later).

    Parameters
    ----------
    members:
        Stable IDs being composed.
    snapshot:
        Intake snapshot carrying the members' immutable routing hints.
    registry:
        Loaded environment registry (the LIVE lock statuses).
    allow_lockless:
        Explicit operator exemption: intents that may admit members while
        their lock family is absent.

    Raises
    ------
    RungStagingError
        If an exemption is stale or any member routes to a lockless
        non-exempt intent.
    """

    from menagerie.crawler.constants import framework_from_markers
    from menagerie.crawler.routing import route_model

    intents = registry.intents
    for name in allow_lockless:
        intent = intents.get(name)
        if intent is None:
            raise RungStagingError(
                f"allow-lockless names unknown environment intent {name!r}"
            )
        if intent.lock.status == "locked":
            raise RungStagingError(
                f"allow-lockless names intent {name!r}, which is LOCKED in the live "
                "registry; the exemption is stale -- exemptions are re-derived "
                "against the live registry, never trusted from a recorded field"
            )
    allowed = frozenset(allow_lockless)
    wanted = frozenset(members)
    violations: list[str] = []
    for item in snapshot.items:
        if item.stable_id not in wanted:
            continue
        route = route_model(
            item.to_model_requirements(framework_from_markers(item.zoo, item.name))
        )
        intent = intents.get(route.intent)
        lock_status = getattr(getattr(intent, "lock", None), "status", None)
        if lock_status != "locked" and route.intent not in allowed:
            violations.append(
                f"{item.stable_id} ({item.name!r}, zoo {item.zoo!r}) routes to "
                f"{route.intent!r} whose lock family is "
                f"{'absent' if intent is None else str(lock_status)}"
            )
    if violations:
        raise RungStagingError(
            "lockless-route refusal (EN-2): no member may compose onto an intent "
            "without a committed lock family; pass allow-lockless for a "
            "deliberately lockless intent or fix the routing: "
            + "; ".join(violations)
        )


def stage_rung(
    records_root: Path,
    *,
    campaign_id: str,
    stable_ids: Optional[Sequence[str]] = None,
    count: Optional[int] = None,
    derived_from_snapshot_id: Optional[str] = None,
    label: str,
    registered_by: str,
    created_at: str,
    award_authorities: Optional[Mapping[str, str]] = None,
    composition_attestation: Optional[RungCompositionAttestation] = None,
    target: str = "osx-arm64",
    allow_lockless: Sequence[str] = (),
    rung_class: Optional[str] = None,
) -> StagedRung:
    """Stage the next rung of one campaign into the durable record store.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    campaign_id:
        Frozen tier campaign the rung belongs to; every member must be in its
        bound partition.
    stable_ids:
        Explicit members (the mini-rung shape).
    count:
        Partition-prefix size (the step-200 shape). Exactly one of
        ``stable_ids`` and ``count`` must be provided.
    derived_from_snapshot_id:
        Explicit lineage; defaults to the campaign's latest registered rung,
        or its root snapshot when no rung exists yet.
    label, registered_by, created_at:
        Registration provenance.
    award_authorities:
        Operator ``stable_id -> exact raw runs record_revision`` authorities
        (AW-2).
    composition_attestation:
        What the composer attested about this rung (B-03). When present it is
        ENFORCED against the stage: the campaign, the member set, and the
        identity these members compose to must all match, or staging refuses
        before any durable byte exists.
    target:
        Exact conda-lock platform whose committed lock families gate the
        EN-2 lockless-route refusal.
    allow_lockless:
        Explicit operator exemption: intents that may compose members while
        their lock family is absent. Validated against the LIVE registry; a
        stale entry naming a locked intent refuses.
    rung_class:
        Closed staging-door class declaration (:data:`RUNG_CLASSES`): what the
        composed rung IS (``successor_rung``, ``canary_seed``). REQUIRED when
        the resolved lineage close is live (the row would carry a
        ``live_origin_custody`` licence); the certificate consumers replay
        their expected certificate kind from it (W16/S-2), so it can never be
        defaulted or steered by a launch-time flag.

    Returns
    -------
    StagedRung
        Durable snapshot plus its registration.

    Raises
    ------
    CurrencyCarveOutMissingError
        If the authority intake-currency carve-out has not landed.
    CompositionAttestationError
        If the attestation does not describe this stage.
    RungStagingError
        If the campaign, membership, lineage, or selection is invalid.
    """

    # ONE staging decision is ONE close-read epoch (round-3 B1). Everything the
    # licensing gate below consults -- the close ledger, the registry, the Track A
    # chain of all 15 census sources, the models projections -- is proven from its
    # W32 (F-O23 rider): staging is where the frozen tier campaign — hence
    # the author-model tier — is first bound, so the toolchain attestation is
    # re-derived HERE. A CLI pin bump without a storm-envelope re-attestation
    # or a tier change without a fuse-table re-derivation refuses the stage.
    assert_toolchain_attestation_current(campaign_id)
    # bytes once for this decision instead of once per prior close per source,
    # which is the O(#closes) x O(campaign-history) BYTE cost the ladder paid.
    with read_epoch():
        return _stage_rung(
            records_root,
            campaign_id=campaign_id,
            stable_ids=stable_ids,
            count=count,
            derived_from_snapshot_id=derived_from_snapshot_id,
            label=label,
            registered_by=registered_by,
            created_at=created_at,
            award_authorities=award_authorities,
            composition_attestation=composition_attestation,
            target=target,
            allow_lockless=allow_lockless,
            rung_class=rung_class,
        )


def assert_toolchain_attestation_current(campaign_id: str) -> None:
    """Refuse staging over a stale toolchain attestation (W32 rider).

    Raises
    ------
    menagerie.crawler.toolchain_pins.ToolchainPinError
        Naming every mismatch and demanding a re-attestation, when the
        committed toolchain attestation does not re-derive on this tree for
        the CLI pins or any tier's fuse table (including this campaign's).
    """

    from menagerie.crawler.toolchain_pins import (
        ToolchainPinError,
        verify_toolchain_attestation,
    )

    mismatches = verify_toolchain_attestation()
    if mismatches:
        raise ToolchainPinError(
            f"staging for {campaign_id!r} refuses: the committed toolchain "
            "attestation does not re-derive on this tree — re-attest the storm "
            "envelope/fuse tables in the same reviewed change as the pin or "
            "tier move: " + "; ".join(mismatches)
        )


def _stage_rung(
    records_root: Path,
    *,
    campaign_id: str,
    stable_ids: Optional[Sequence[str]] = None,
    count: Optional[int] = None,
    derived_from_snapshot_id: Optional[str] = None,
    label: str,
    registered_by: str,
    created_at: str,
    award_authorities: Optional[Mapping[str, str]] = None,
    composition_attestation: Optional[RungCompositionAttestation] = None,
    target: str = "osx-arm64",
    allow_lockless: Sequence[str] = (),
    rung_class: Optional[str] = None,
) -> StagedRung:
    """Stage one rung inside the caller's open close-read epoch."""

    if rung_class is not None and rung_class not in RUNG_CLASSES:
        raise RungStagingError(
            f"rung_class {rung_class!r} is outside the closed vocabulary {sorted(RUNG_CLASSES)}"
        )
    _require_currency_carve_out(campaign_id)
    root_binding, root_snapshot = resolve_verified_root(records_root, campaign_id)
    if (stable_ids is None) == (count is None):
        raise RungStagingError("select rung members with exactly one of stable_ids or count")
    if count is not None:
        if count < 1:
            raise RungStagingError("count must be positive")
        selected = _partition_prefix_ids(records_root, root_binding, root_snapshot, count)
    else:
        selected = tuple(str(value) for value in stable_ids or ())
    members = _validate_membership(selected, root_snapshot, campaign_id)
    # EN-2: no member may compose onto an intent without a committed lock
    # family unless the operator exempted that intent explicitly. Refusing at
    # the COMPOSE door means the defect never reaches a paid author session
    # (pilot m4714 died at a lockless env turn after a full author spend).
    refuse_lockless_member_routes(
        members,
        root_snapshot,
        registry=_environment_registry_for_staging(target),
        allow_lockless=allow_lockless,
    )
    _refuse_unauthorized_runs_holders(records_root, members, award_authorities)
    # Identity is PLANNED before anything durable exists: every gate below
    # must refuse with zero snapshot bytes on disk, or a refused stage leaves
    # an orphan for the register door to adopt (red-team break 2).
    planned_snapshot_id, planned_snapshot_sha256, _, _, _ = _plan_subset_snapshot(
        records_root / root_binding.intake_path, members
    )
    if composition_attestation is not None:
        verify_composition_attestation(
            composition_attestation,
            campaign_id=campaign_id,
            member_ids=members,
            snapshot_id=planned_snapshot_id,
            snapshot_sha256=planned_snapshot_sha256,
            label=label,
            rung_class=rung_class,
        )
    else:
        # SOL-02: a successor-rung stage cannot skip the PD-7 consumption by
        # simply presenting no attestation; pre-launch classes keep B-03's
        # disposition-negative optionality.
        _verify_pd7_disposition(None, label=label, rung_class=rung_class)
    lineage = _resolve_lineage(
        records_root,
        root_binding,
        derived_from_snapshot_id,
        staged_snapshot_id=planned_snapshot_id,
    )
    licensing_close = _require_closed_lineage(
        records_root,
        root_binding,
        lineage,
        label=label,
        staged_snapshot_id=planned_snapshot_id,
        staged_member_ids=members,
        composition_attestation=composition_attestation,
    )
    # W16/S-2: a live-licensed rung must declare what it IS before any durable
    # byte exists — the certificate consumers replay their expected kind from
    # this declaration, and a refusal here leaves no orphan for the register
    # door to adopt.
    if (
        licensing_close is not None
        and licensing_close.get("close_kind") != "backfill"
        and rung_class is None
    ):
        raise RungStagingError(
            f"staging {label!r} under a live licensing close requires a declared "
            f"rung_class from {sorted(RUNG_CLASSES)}"
        )
    snapshot, snapshot_created = _write_subset_snapshot(
        records_root / root_binding.intake_path,
        members,
        records_root / "intake",
    )
    existing_registration = next(
        (
            row
            for row in load_rung_registrations(records_root)
            if row.snapshot_id == snapshot.snapshot_id
        ),
        None,
    )
    if existing_registration is not None:
        # R2-4/SOL-01: never silently discard a declared class on the
        # idempotent path — the restage either agrees with the governing
        # class or refuses toward the refinement door.
        _refuse_conflicting_class_claim(records_root, existing_registration, rung_class)
        return StagedRung(
            registration=existing_registration,
            snapshot_root=records_root / "intake" / snapshot.snapshot_id,
            snapshot_created=snapshot_created,
            registration_created=False,
        )
    registration = build_rung_registration(
        campaign_id=campaign_id,
        snapshot_id=snapshot.snapshot_id,
        snapshot_sha256=snapshot.snapshot_sha256,
        derived_from_snapshot_id=lineage,
        member_count=len(snapshot.items),
        label=label,
        registered_by=registered_by,
        created_at=created_at,
        licensed_by_close_sha256=(
            None if licensing_close is None else str(licensing_close["close_sha256"])
        ),
        license_kind=(
            None
            if licensing_close is None
            else "historical_bootstrap"
            if licensing_close.get("close_kind") == "backfill"
            else "live_origin_custody"
        ),
        closes_chain_head=_licensed_close_chain_head(records_root, licensing_close),
        rung_class=rung_class,
    )
    appended = append_rung_registration(rung_registry_path(records_root), registration)
    return StagedRung(
        registration=appended.registration,
        snapshot_root=snapshot.root,
        snapshot_created=snapshot_created,
        registration_created=appended.registration_created,
    )


def register_existing_snapshot(
    records_root: Path,
    snapshot_root: Path,
    *,
    campaign_id: str,
    derived_from_snapshot_id: Optional[str] = None,
    label: str,
    registered_by: str,
    created_at: str,
    historical_only: bool = True,
    award_authorities: Optional[Mapping[str, str]] = None,
    composition_attestation: Optional[RungCompositionAttestation] = None,
    rung_class: Optional[str] = None,
) -> StagedRung:
    """Adopt an existing rung snapshot into the durable record store.

    This is the OPS-2 contract: an already-run snapshot living only in
    disposable scratch is copied append-only under ``records/intake/`` and
    bound to its campaign. It is deliberately not gated on the currency
    carve-out -- it protects history that already exists -- but it walks the
    SAME lineage and award gates as :func:`stage_rung` (2026-08-16 red-team
    break 1: the ungated register door adopted exactly the states stage_rung
    refused):

    * the close-object gate binds the LINEAGE, on every path;
    * with the default ``historical_only=True`` the claim of history must be
      PROVEN from evidence (:func:`_require_recorded_history`: every member's
      current head recorded under this exact snapshot) -- a proven historical
      snapshot may carry its own run's awards;
    * a snapshot intended for NEW work must pass ``historical_only=False``
      (``--new-work``) and satisfy the same AW-2 award-authority guard as
      :func:`stage_rung`; and
    * a presented composition attestation is enforced identically (B-03),
      against the ADOPTED snapshot's own verified identity -- the door binds the
      evidence, never the entry point.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    snapshot_root:
        Existing snapshot directory (inside or outside the record store).
    campaign_id:
        Frozen tier campaign that owns the snapshot's members.
    derived_from_snapshot_id:
        Explicit lineage; defaults as in :func:`stage_rung`.
    label, registered_by, created_at:
        Registration provenance.
    historical_only:
        Whether this adoption claims to be already-run history; the claim is
        verified, never trusted.
    award_authorities:
        Operator ``stable_id -> exact raw runs record_revision`` authorities
        for the new-work path.
    composition_attestation:
        What the composer attested about this rung (B-03), enforced against the
        adopted snapshot's verified identity and member set when present.
    rung_class:
        Closed staging-door class declaration (:data:`RUNG_CLASSES`). A
        proven-history adoption IS ``adopted_history`` — the door derives it
        from its own recorded-history proof, and an explicit conflicting claim
        refuses. A new-work adoption under a live licensing close must declare
        its class explicitly, exactly as :func:`stage_rung`.

    Returns
    -------
    StagedRung
        Durable snapshot plus its registration.

    Raises
    ------
    CompositionAttestationError
        If the attestation does not describe the adopted snapshot.
    RungStagingError
        If the snapshot's members are not all members of the campaign, the
        historical claim is unproven, a new-work runs holder lacks an exact
        award authority, no models ledger exists, or the durable store already
        holds different bytes for its identity.
    CloseObjectError
        If the registered lineage rung has no licensing close object.
    """

    root_binding, root_snapshot = resolve_verified_root(records_root, campaign_id)
    snapshot = load_intake_snapshot(snapshot_root)
    adopted_members = _validate_membership(
        tuple(item.stable_id for item in snapshot.items), root_snapshot, campaign_id
    )
    if composition_attestation is not None:
        # The adopted snapshot's identity is the one the content-addressed
        # loader just re-verified from its own bytes, so a hand-edited sha in
        # the attestation cannot agree with it.
        verify_composition_attestation(
            composition_attestation,
            campaign_id=campaign_id,
            member_ids=adopted_members,
            snapshot_id=snapshot.snapshot_id,
            snapshot_sha256=snapshot.snapshot_sha256,
            label=label,
            rung_class=rung_class,
        )
    else:
        # SOL-02: identical to stage_rung — the register door never bypasses
        # the successor-rung PD-7 consumption by omitting the attestation.
        _verify_pd7_disposition(None, label=label, rung_class=rung_class)
    if rung_class is not None and rung_class not in RUNG_CLASSES:
        raise RungStagingError(
            f"rung_class {rung_class!r} is outside the closed vocabulary {sorted(RUNG_CLASSES)}"
        )
    # The operator's EXPLICIT claim, before the door derives its own class on
    # the proven-history path: only an explicit claim can conflict with an
    # existing registration on the idempotent path (R2-4/SOL-01).
    declared_rung_class = rung_class
    if historical_only:
        if rung_class not in (None, RUNG_CLASS_ADOPTED_HISTORY):
            raise RungStagingError(
                f"a proven-history adoption is {RUNG_CLASS_ADOPTED_HISTORY!r}; the "
                f"conflicting claim {rung_class!r} refuses"
            )
        _require_recorded_history(records_root, snapshot, adopted_members)
        # Derived from the door's own recorded-history proof, never trusted.
        rung_class = RUNG_CLASS_ADOPTED_HISTORY
    else:
        _refuse_unauthorized_runs_holders(records_root, adopted_members, award_authorities)
    lineage = _resolve_lineage(
        records_root,
        root_binding,
        derived_from_snapshot_id,
        staged_snapshot_id=snapshot.snapshot_id,
    )
    licensing_close = _require_closed_lineage(
        records_root,
        root_binding,
        lineage,
        label=label,
        staged_snapshot_id=snapshot.snapshot_id,
        staged_member_ids=adopted_members,
        composition_attestation=composition_attestation,
    )
    # W16/S-2: identical to stage_rung — a new-work adoption under a live
    # close must declare its class before any durable byte exists.
    if (
        licensing_close is not None
        and licensing_close.get("close_kind") != "backfill"
        and rung_class is None
    ):
        raise RungStagingError(
            f"adopting {label!r} for new work under a live licensing close requires "
            f"a declared rung_class from {sorted(RUNG_CLASSES)}"
        )
    durable_root = records_root / "intake" / snapshot.snapshot_id
    copied = False
    if durable_root.resolve() != snapshot_root.resolve():
        # H1 F2: the manifest is written LAST, preserving the load-bearing
        # "manifest present => members present" invariant the intake sweeps
        # rely on (`_write_subset_snapshot` is the sibling convention). The
        # historical sorted order wrote manifest.json BEFORE sources/, so a
        # crash mid-copy left a durable dir that CLAIMED to be a snapshot but
        # failed to load -- bricking every requeue-grant issuance and the
        # canonical checkpoint campaign-wide (the B-2 blast shape). A crash
        # under this ordering leaves a manifest-less partial dir instead: not
        # a claim, skipped by the sweeps, and healed by re-running register
        # (existing identical bytes are tolerated below).
        source_files = sorted(
            path.relative_to(snapshot_root) for path in snapshot_root.rglob("*") if path.is_file()
        )
        source_files.sort(key=lambda relative: relative.name == "manifest.json")
        for relative in source_files:
            data = (snapshot_root / relative).read_bytes()
            target = durable_root / relative
            if target.exists():
                if target.read_bytes() != data:
                    raise RungStagingError(
                        f"durable snapshot {durable_root} holds different bytes for {relative}"
                    )
                continue
            atomic_replace_bytes(target, data)
            copied = True
        load_intake_snapshot(durable_root)
    existing_registration = next(
        (
            row
            for row in load_rung_registrations(records_root)
            if row.snapshot_id == snapshot.snapshot_id
        ),
        None,
    )
    if existing_registration is not None:
        # R2-4/SOL-01: the operator's explicit class claim is never silently
        # discarded on the idempotent path (the door-derived adopted_history
        # of a proven-history re-adoption is not a claim and stays tolerant).
        _refuse_conflicting_class_claim(records_root, existing_registration, declared_rung_class)
        return StagedRung(
            registration=existing_registration,
            snapshot_root=durable_root,
            snapshot_created=copied,
            registration_created=False,
        )
    registration = build_rung_registration(
        campaign_id=campaign_id,
        snapshot_id=snapshot.snapshot_id,
        snapshot_sha256=snapshot.snapshot_sha256,
        derived_from_snapshot_id=lineage,
        member_count=len(snapshot.items),
        label=label,
        registered_by=registered_by,
        created_at=created_at,
        licensed_by_close_sha256=(
            None if licensing_close is None else str(licensing_close["close_sha256"])
        ),
        license_kind=(
            None
            if licensing_close is None
            else "historical_bootstrap"
            if licensing_close.get("close_kind") == "backfill"
            else "live_origin_custody"
        ),
        closes_chain_head=_licensed_close_chain_head(records_root, licensing_close),
        rung_class=rung_class,
    )
    appended = append_rung_registration(rung_registry_path(records_root), registration)
    return StagedRung(
        registration=appended.registration,
        snapshot_root=durable_root,
        snapshot_created=copied,
        registration_created=appended.registration_created,
    )
