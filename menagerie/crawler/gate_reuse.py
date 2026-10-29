"""Gate-identity verdict reuse -- RETIRED live, retained as validation machinery.

**RETIRED (BL-10 h, 2026-08-17).** Semantic verdict reuse recorded ZERO lifetime
hits across the pilot's 344 gate rows (extending the chartered 0/286 and W4's
own 0/395 census) while adding four hot-path hooks and a second verdict
provenance. Its key is also not semantically closed: it pins the author-result
schema and dispatcher but not ``checker_dispatch.py``/``metadata.py``/``gates.py``
derivation drift, the checker CLI build, or batch composition -- so its first
ever hit could replay a rejection under a semantic world its key does not
identify and charge it against a model's bounded repair cap. Per the tri-lab
round-1 adjudication the live feature is DISABLED: every entry point below
answers "no reuse" unconditionally (:data:`GATE_VERDICT_REUSE_RETIRED`), so a
session is always bought live. What remains supported:

* **Transport-level completed-result adoption** is NOT this module and is NOT
  retired: a frozen checker request replayed through one provider-weather storm
  (``driver_admission._frozen_checker_weather_request``) reuses exact request
  bytes, never a verdict.
* The marked-replay READ machinery (``verdict_source``/``reused_from`` and the
  authenticated-claim scan) stays importable: historical ledgers must keep
  validating, and a forged replay marking must keep failing closed.

Re-arming reuse is a deliberate reviewed change: flip the constant only with a
key that digests the complete normalized request, validator/routing closure,
an immutable checker build identity, and batch context -- and only after a
shadow census demonstrates a material hit rate.

The original design rationale follows, retained for the read machinery.

The pilot campaign spent ~15 hours of checker wall verifying 3.4 minutes of
model execution, and its single biggest waste class was re-adjudication of
UNCHANGED evidence: the same item pack, checker prompt, checker model, and
result contract re-bought a fresh session in a later work generation because
gate history is (correctly) consulted per lineage while the adjudication it
already paid for is generation-independent.

This module owns the reuse decision. It is deliberately fail-closed:

* The FULL adjudication identity must match -- the exact evidence payload
  identity (every dependent hash the dispatched item pack binds, via the same
  :func:`~menagerie.crawler.driver_models._gate_item_matches_proposal` matcher
  the live routing replays), the exact checker ``provider``/``model``/
  ``version``/``prompt_sha256``, the CURRENT shipped author-result schema and
  dispatcher component identities, the current gate schema version, and the
  same gate kind. Any mismatch, an unreadable or self-hash-broken prior row,
  or a checker lane that does not advertise its adjudication identity buys a
  full live session. Nothing is ever fuzzy-matched.
* RECENCY IS RESOLVED BEFORE ELIGIBILITY, per model and per gate kind. The
  checker's LATEST same-kind row containing the model decides, whatever it
  says. Every ineligibility -- payload drift, an acceptance, a checker
  identity this lane cannot claim, a retired gate schema, a torn self-hash --
  is a live session, NEVER a fall-back into the older history that row
  superseded. Scoping identity BEFORE recency was a stale-verdict side door:
  a row failing the identity match was skipped and the scan walked past the
  newest adjudication into an older one, resurrecting a rejection the checker
  had since reversed (and charging it against the model's bounded repair cap).
  Ledgers interleave identities as a matter of course -- a rotated then
  reverted checker prompt or model, and host shards merged by
  ``merge_host_shards`` from hosts pinning different checker models.
* A same-kind row that CLAIMS this kind but cannot be AUTHENTICATED (broken
  ``result_envelope_sha256`` self-hash, or unreadable ``items``) fails the
  whole reuse question closed rather than being scanned past. Such a row may
  itself be a later adjudication whose write tore or whose marking was
  stripped, and when its items are unreadable there is no way to know whether
  it even concerns this model.
* A replayed verdict is HONESTLY MARKED: the synthesized row carries
  ``verdict_source: "reused"`` plus a ``reused_from`` binding naming the prior
  row's ``gate_id``, ``gate_identity``, and ``result_envelope_sha256``, and
  the row's own ``result_envelope_sha256`` self-hash covers both fields, so a
  replay can never later masquerade as a live adjudication.
* No proof rule is weakened. A synthesized metadata/fidelity replay row rides
  the exact same normalization, binding, routing, schema-validation, and
  write-door paths as a live checker result, and a terminal replay re-runs the
  full :func:`~menagerie.crawler.gates.validate_terminal_disposition_gate`
  reference check against the CURRENT typed author result.

The three lanes replay differently, and the difference is a PROOF-RULE
consequence, not convenience:

* **Metadata and fidelity** reuse applies ONLY to non-accepted prior verdicts.
  An ACCEPTED matching adjudication never reaches dispatch at all -- the
  existing gate-history skip consumes it for free upstream -- so an accepted
  item arriving here is either the rung-9 write-refusing poisoned-gate class
  or a repair that reverted to previously accepted bytes, and BOTH must buy a
  live session (fail-closed; appending a duplicate of an accepted item's
  bytes would also break the exact-membership anti-substitution rule below).
  The rejected replay appends a new, marked generation row: the bounded
  repair accounting counts rejected rows per lineage, and rejected items are
  never published, so duplicated rejected bytes prove nothing anyone resolves
  uniqueness over.
* **Terminal dispositions** replay the PRIOR ROW ITSELF: the append is
  byte-idempotent (no new row), because the accepted item's publication
  authorization resolves the item to its UNIQUE owning v3 ledger row
  (``resolve_exact_gate_item_membership``), and a second row carrying the
  same item bytes would honestly break that anti-substitution tripwire. The
  reuse is durably visible on the operational ledger instead
  (``checker-verdict-reused`` with the replayed row's identity), and the gate
  ledger keeps exactly one adjudication claim -- which is also the truth.

A reused REJECTED verdict stays re-purchasable through the existing bounded
repair and grant lanes: the cached rejection routes to repair exactly like a
live one, and a repair that actually changed the author result moves the
payload identity, so the next round buys a live session by construction.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.author_dispatch import ProposedAuthorResult
from menagerie.crawler.checker_dispatch import GATE_ID_PREFIX, component_identity
from menagerie.crawler.constants import (
    AUTHOR_DISPATCHER_COMPONENT,
    AUTHOR_RESULT_SCHEMA_COMPONENT,
    GATE_SCHEMA_VERSION_V3,
    METADATA_BATCH_MAX,
    GateKind,
    fidelity_item_accepted,
    metadata_item_accepted,
)
from menagerie.crawler.driver_contracts import AuthorArtifact, DriverIntegrationError
from menagerie.crawler.gates import GateRoutingError, validate_terminal_disposition_gate
from menagerie.crawler.identity import is_shadow_rehearsal_record, stable_hash
from menagerie.crawler.models import JsonObject

GATE_VERDICT_SOURCE_FIELD = "verdict_source"
GATE_VERDICT_SOURCE_REUSED = "reused"
GATE_REUSED_FROM_FIELD = "reused_from"

#: BL-10 (h): live semantic verdict reuse is RETIRED (0 hits in 344 lifetime
#: gate rows). While True, every reuse entry point answers "buy a live session"
#: unconditionally and can never mint a verdict from history. The read-side
#: machinery for HISTORICAL marked replays stays armed regardless.
GATE_VERDICT_REUSE_RETIRED = True

#: Ledger-assigned fields excluded from every gate self-hash, spelled once here
#: exactly as ``_normalize_gate_generation`` and ``load_current_gate_proof``
#: spell them.
_SELF_HASH_EXCLUDED = frozenset({"result_envelope_sha256", "payload_sha256", "ledger_seq"})


@dataclass(frozen=True)
class CheckerAdjudicationIdentity:
    """The exact checker identity a lane would buy a session under.

    The LANE owns model selection (the command lane pins per-gate-kind models;
    test lanes stamp the configured one), so reuse never guesses: a lane that
    cannot name this identity gets no reuse at all, and a recorded verdict is
    only replayed when its row was sealed under the byte-identical checker
    provider, model, version channel, and prompt.
    """

    provider: str
    model: str
    version: str
    prompt_sha256: str


def lane_confirmation_identity(
    checker: Any, gate_kind: str, config: Any
) -> Optional[CheckerAdjudicationIdentity]:
    """Ask one checker lane for its cross-model K=2 CONFIRMATION identity.

    Seat swap (2026-09-01): a lane that seats a confirmation partner declares
    it through ``confirmation_identity``; a lane without one (test lanes, the
    pre-seat protocol) confirms under its own adjudication identity, which is
    the historical same-identity K=2. The conflict ledger's pairing rule --
    not this lookup -- is the authority on whether the recorded pair is lawful.
    """

    declare = getattr(checker, "confirmation_identity", None)
    if callable(declare):
        identity = declare(gate_kind, config)
        return identity if isinstance(identity, CheckerAdjudicationIdentity) else None
    return lane_adjudication_identity(checker, gate_kind, config)


def lane_adjudication_identity(
    checker: Any, gate_kind: str, config: Any
) -> Optional[CheckerAdjudicationIdentity]:
    """Ask one checker lane for its adjudication identity, or ``None``.

    Parameters
    ----------
    checker:
        The dependency-injected checker lane.
    gate_kind:
        ``metadata_batch``, ``fidelity``, or ``terminal_disposition``.
    config:
        Driver configuration handed to the lane's dispatch methods.

    Returns
    -------
    CheckerAdjudicationIdentity | None
        The lane's declared identity, or ``None`` when the lane does not
        advertise one (reuse is then disabled, fail-closed) or the probe
        itself fails.
    """

    probe = getattr(checker, "adjudication_identity", None)
    if probe is None:
        return None
    try:
        identity = probe(gate_kind, config)
    except Exception:  # noqa: BLE001 -- an unprobeable lane disables reuse, never the run
        return None
    return identity if isinstance(identity, CheckerAdjudicationIdentity) else None


def plan_metadata_gate_reuse(
    artifacts: Sequence[AuthorArtifact],
    persisted: Sequence[Mapping[str, Any]],
    *,
    checker: Optional[CheckerAdjudicationIdentity],
) -> tuple[tuple[tuple[AuthorArtifact, ...], ...], tuple[AuthorArtifact, ...]]:
    """Partition pending metadata work into reusable groups and live work.

    Members whose latest matching adjudication lives in the SAME prior row
    group together (one reuse row per prior row, so ``reused_from`` stays a
    single honest binding); everything else batches live at the calibrated
    cap. Extracting reusable members before batching also keeps live batches
    packed with work that actually needs a session.

    Parameters
    ----------
    artifacts:
        Pending metadata artifacts, in scheduling order.
    persisted:
        Durable gate rows in ledger order.
    checker:
        Lane adjudication identity, or ``None`` to disable reuse entirely.

    Returns
    -------
    tuple[tuple[tuple[AuthorArtifact, ...], ...], tuple[AuthorArtifact, ...]]
        Reuse groups (each fully resolvable by :func:`reusable_metadata_gate`)
        and the live remainder.
    """

    if GATE_VERDICT_REUSE_RETIRED or checker is None:
        return (), tuple(artifacts)
    groups: dict[tuple[str, str], list[AuthorArtifact]] = {}
    live: list[AuthorArtifact] = []
    for artifact in artifacts:
        match = _latest_reuse_match(
            artifact, persisted, kind=GateKind.METADATA_BATCH.value, checker=checker
        )
        if match is None:
            live.append(artifact)
            continue
        gate, _item = match
        key = (str(gate.get("gate_id")), str(gate.get("result_envelope_sha256")))
        groups.setdefault(key, []).append(artifact)
    grouped = tuple(
        tuple(members[offset : offset + METADATA_BATCH_MAX])
        for members in groups.values()
        for offset in range(0, len(members), METADATA_BATCH_MAX)
    )
    return grouped, tuple(live)


def reusable_metadata_gate(
    artifacts: Sequence[AuthorArtifact],
    persisted: Sequence[Mapping[str, Any]],
    *,
    checker: Optional[CheckerAdjudicationIdentity],
) -> Optional[JsonObject]:
    """Return one replayed metadata gate for the batch, or ``None`` for live.

    Every batch member must resolve its latest full-identity match in the SAME
    prior gate row; any member without one -- or any identity component
    mismatch anywhere -- refuses the whole batch to a live session.
    """

    return _reusable_gate(artifacts, persisted, kind=GateKind.METADATA_BATCH.value, checker=checker)


def reusable_fidelity_gate(
    artifact: AuthorArtifact,
    persisted: Sequence[Mapping[str, Any]],
    *,
    checker: Optional[CheckerAdjudicationIdentity],
) -> Optional[JsonObject]:
    """Return one replayed per-model fidelity gate, or ``None`` for live."""

    return _reusable_gate((artifact,), persisted, kind=GateKind.FIDELITY.value, checker=checker)


def reusable_terminal_gate(
    artifact: AuthorArtifact,
    persisted: Sequence[Mapping[str, Any]],
    *,
    checker: Optional[CheckerAdjudicationIdentity],
) -> Optional[JsonObject]:
    """Return one replayed terminal-disposition gate, or ``None`` for live.

    Beyond the shared gate-level identity, the prior item must bind the exact
    CURRENT typed author result (result id and sha256, source-manifest,
    evidence, and license identities, work and lineage bindings), and the
    synthesized row must replay the full terminal reference check against the
    current artifact -- the same validation a live gate faces after append.
    Any failure anywhere refuses to a live session.
    """

    if GATE_VERDICT_REUSE_RETIRED or checker is None:
        return None
    result = artifact.author_result
    if isinstance(result, ProposedAuthorResult):
        return None
    binding = result.binding
    for gate in reversed(list(persisted)):
        claim = _authenticated_kind_claim(gate, kind=GateKind.TERMINAL_DISPOSITION.value)
        if claim is None:
            continue
        if claim is False:
            return None
        items = gate.get("items")
        assert isinstance(items, list)  # guaranteed by the authenticated claim
        mine = [item for item in items if item.get("stable_id") == binding.stable_id]
        if not mine:
            continue
        if len(mine) != 1:
            # One row adjudicating this model twice is not a contract this code
            # knows how to read, and picking either copy would be a silent
            # substitution choice. Fail closed.
            return None
        item = mine[0]
        # This row is the checker's LAST terminal word on this model. It either
        # replays under the exact current adjudication identity AND result
        # binding, or nothing recorded is current -- an older row is by
        # definition superseded, so every mismatch below refuses to a live
        # session instead of falling back.
        if not _gate_adjudication_matches(
            gate, kind=GateKind.TERMINAL_DISPOSITION.value, checker=checker
        ):
            return None
        terminal = item.get("terminal_disposition")
        if not isinstance(terminal, Mapping):
            return None
        exact = (
            (item, "work_id", binding.work_id),
            (item, "campaign_root_work_id", binding.campaign_id),
            (terminal, "author_result_id", binding.result_id),
            (terminal, "author_result_sha256", binding.result_sha256),
            (terminal, "source_manifest_identity", binding.source_manifest_identity),
            (terminal, "evidence_identity", result.evidence_identity),
            (terminal, "license_identity", result.license_identity),
        )
        if any(owner.get(field) != value for owner, field, value in exact):
            return None
        # The replay IS the prior row's logical payload: re-appending it is a
        # byte-idempotent no-op, so the ledger keeps exactly one adjudication
        # claim and the accepted item's exact-membership resolution stays
        # unique. A second row with the same item bytes would break that
        # anti-substitution tripwire, so no marked duplicate is ever minted
        # for terminals; the reuse is recorded on the operational ledger.
        candidate: JsonObject = {
            key: deepcopy(value)
            for key, value in gate.items()
            if key not in {"ledger_seq", "payload_sha256"}
        }
        # Replay the exact reference validation a live terminal gate faces.
        # The pack is machine-re-derived from frozen bytes, so a source tree
        # that no longer re-derives (or any stale reference) refuses reuse
        # here rather than crashing the routing after append.
        try:
            from menagerie.crawler.driver_models import _terminal_checker_item

            pack = _terminal_checker_item(artifact)
            validate_terminal_disposition_gate(
                {**candidate, "ledger_seq": 1, "payload_sha256": "sha256:" + "0" * 64},
                result,
                source_manifest=artifact.source_manifest,
                evidence_pack=pack["evidence_pack"],
                license_identity=str(pack["license_identity"]),
            )
        except (GateRoutingError, DriverIntegrationError, ValueError, KeyError, TypeError):
            return None
        return candidate
    return None


def _reusable_gate(
    artifacts: Sequence[AuthorArtifact],
    persisted: Sequence[Mapping[str, Any]],
    *,
    kind: str,
    checker: Optional[CheckerAdjudicationIdentity],
) -> Optional[JsonObject]:
    """Resolve one whole-batch replay from a single prior row, fail-closed."""

    if GATE_VERDICT_REUSE_RETIRED or checker is None or not artifacts:
        return None
    owner_key: Optional[tuple[str, str]] = None
    owner_gate: Optional[Mapping[str, Any]] = None
    items: list[JsonObject] = []
    for artifact in artifacts:
        match = _latest_reuse_match(artifact, persisted, kind=kind, checker=checker)
        if match is None:
            return None
        gate, item = match
        key = (str(gate.get("gate_id")), str(gate.get("result_envelope_sha256")))
        if owner_key is None:
            owner_key, owner_gate = key, gate
        elif key != owner_key:
            return None
        replayed = deepcopy(dict(item))
        # The ONLY rebound fields are the work-generation bindings the current
        # scheduling assigned; every adjudicated field is the prior row's,
        # verbatim. For a byte-identical proposal these are equal anyway (the
        # proposal embeds its own work binding), so the rebind is coherence,
        # not correction.
        from menagerie.crawler.driver_models import _artifact_lineage

        replayed["work_id"] = artifact.proposal["work_id"]
        replayed["campaign_root_work_id"] = _artifact_lineage(artifact)
        items.append(replayed)
    assert owner_gate is not None
    return _build_reused_gate(owner_gate, items, gate_kind=kind)


def _latest_reuse_match(
    artifact: AuthorArtifact,
    persisted: Sequence[Mapping[str, Any]],
    *,
    kind: str,
    checker: CheckerAdjudicationIdentity,
) -> Optional[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Return the LATEST prior (gate, item) eligible to serve as a verdict source.

    RECENCY IS RESOLVED FIRST, THEN ELIGIBILITY. The latest same-kind row
    containing this model is located by kind and stable ID alone -- deliberately
    NOT filtered by checker identity, schema version, or payload -- and that one
    row decides. Any ineligibility is a live session; there is never a fall-back
    into older history the deciding row superseded. See the module docstring for
    the stale-verdict side door this ordering closes.
    """

    try:
        proposal = artifact.proposal
    except DriverIntegrationError:
        return None
    from menagerie.crawler.driver_models import _gate_item_matches_proposal

    stable_id = str(proposal.get("stable_id"))
    for gate in reversed(list(persisted)):
        claim = _authenticated_kind_claim(gate, kind=kind)
        if claim is None:
            continue
        if claim is False:
            return None
        items = gate.get("items")
        assert isinstance(items, list)  # guaranteed by the authenticated claim
        mine = [item for item in items if item.get("stable_id") == stable_id]
        if not mine:
            continue
        if len(mine) != 1:
            # One row adjudicating this model twice is not a contract this code
            # knows how to read, and picking either copy would be a silent
            # substitution choice. Fail closed.
            return None
        if is_shadow_rehearsal_record(gate):
            # W44 SHADOW-EXCLUSION: this row IS the latest adjudication of the
            # model, and it was rendered inside a shadow rehearsal (R6b). A
            # rehearsal verdict never fires reuse, and walking past it into
            # older real history would resurrect a verdict the rehearsal may
            # have contradicted -- so it decides LIVE, exactly like every
            # other ineligibility of the deciding row. Keyed on the record's
            # own marker (self-hash-covered); independent of retirement state
            # and of any future DX-59 re-arm.
            return None
        if gate.get(GATE_VERDICT_SOURCE_FIELD) == GATE_VERDICT_SOURCE_REUSED:
            # A marked replay is TRANSPARENT for recency and never a verdict
            # SOURCE. It carries no adjudication of its own -- it is a copy of
            # the row its ``reused_from`` names, and that row was already proven
            # the latest word when this replay was minted. Sourcing a replay
            # would chain ``reused_from`` bindings away from the session
            # actually bought. A genuinely newer LIVE row still shadows it,
            # because a live row is never marked, and a FORGED marking cannot
            # buy anything: the row had to authenticate above to get here, and
            # the self-hash covers this field.
            continue
        # This row IS the checker's latest adjudication of this model in this
        # lane, so every ineligibility below is a live session, never a
        # fall-back to something it superseded.
        if not _gate_adjudication_matches(gate, kind=kind, checker=checker):
            return None
        item = mine[0]
        if not _gate_item_matches_proposal(item, proposal, kind):
            return None
        if _item_accepted(item, kind):
            # An ACCEPTED matching adjudication never legitimately reaches
            # dispatch (the gate-history skip already consumes it, free),
            # so this is either the rung-9 write-refusing poisoned-gate
            # class or a repair that reverted to previously accepted bytes
            # -- both buy a live session, fail-closed. Replaying would
            # also duplicate accepted item bytes on the ledger, which the
            # exact-membership anti-substitution rule refuses at
            # publication.
            return None
        return gate, item
    return None


def _item_accepted(item: Mapping[str, Any], kind: str) -> bool:
    """Return whether one prior gate item is an accepted verdict for its kind.

    Delegates to the shared acceptance kernels so the reuse eligibility
    tripwire and the write door can never answer differently about the same
    ledger bytes: an item this returns False for is REPLAYABLE (and a metadata
    replay appends a marked row), so if routing would publish the same item the
    replay mints a duplicate over accepted bytes.
    """

    if kind == GateKind.FIDELITY.value:
        return fidelity_item_accepted(item)
    return metadata_item_accepted(item)


def _authenticated_kind_claim(gate: Any, *, kind: str) -> Optional[bool]:
    """Classify one persisted row for the recency scan of one gate kind.

    Returns
    -------
    bool | None
        ``None`` when the row legibly is NOT part of this kind's adjudication
        stream (not a mapping, or another gate kind) -- the scan skips it.
        ``True`` when the row claims this kind AND authenticates: its
        result-envelope self-hash replays and its items are a readable list of
        objects, so the caller may trust the item list well enough to ask
        whether this model is in it. ``False`` when the row claims this kind
        but cannot be authenticated -- the caller must fail the whole reuse
        question closed. Such a row may BE a later adjudication whose write
        tore or whose replay marking someone stripped, and when its items are
        unreadable there is no way to know whether it concerns this model at
        all, so scanning past it would replay superseded history.

    Deliberately does NOT check ``schema_version``: a retired-schema row that
    contains this model is the row that DECIDES, and refusing it belongs to
    :func:`_gate_adjudication_matches` (a live session) rather than here (which
    would skip past it into older history).
    """

    if not isinstance(gate, Mapping):
        return None
    if gate.get("gate_kind") != kind:
        return None
    try:
        recomputed = stable_hash(
            {key: value for key, value in gate.items() if key not in _SELF_HASH_EXCLUDED}
        )
    except (TypeError, ValueError):
        return False
    if gate.get("result_envelope_sha256") != recomputed:
        return False
    items = gate.get("items")
    if not isinstance(items, list) or not all(isinstance(item, Mapping) for item in items):
        return False
    return True


def _gate_adjudication_matches(
    gate: Mapping[str, Any], *, kind: str, checker: CheckerAdjudicationIdentity
) -> bool:
    """Check every gate-level adjudication-identity component, fail-closed.

    The prior row must be current-schema, of the same kind, sealed under the
    exact expected checker identity AND the CURRENTLY shipped author-result
    schema and dispatcher component bytes, and must replay its own
    result-envelope self-hash -- a tampered or torn row is unreadable history
    and never a verdict source.
    """

    if not isinstance(gate, Mapping):
        return False
    if gate.get("schema_version") != GATE_SCHEMA_VERSION_V3 or gate.get("gate_kind") != kind:
        return False
    checker_block = gate.get("checker")
    if not isinstance(checker_block, Mapping):
        return False
    if (
        checker_block.get("provider") != checker.provider
        or checker_block.get("model") != checker.model
        or checker_block.get("version") != checker.version
        or checker_block.get("prompt_sha256") != checker.prompt_sha256
    ):
        return False
    if gate.get("author_result_schema_identity") != component_identity(
        AUTHOR_RESULT_SCHEMA_COMPONENT
    ) or gate.get("dispatcher_identity") != component_identity(AUTHOR_DISPATCHER_COMPONENT):
        return False
    try:
        recomputed = stable_hash(
            {key: value for key, value in gate.items() if key not in _SELF_HASH_EXCLUDED}
        )
    except (TypeError, ValueError):
        return False
    return gate.get("result_envelope_sha256") == recomputed


def _build_reused_gate(
    prior: Mapping[str, Any], items: list[JsonObject], *, gate_kind: str
) -> JsonObject:
    """Synthesize one honestly marked replay row from a matched prior gate.

    The checker block is the ORIGINAL adjudication's identity and wall timing,
    copied verbatim: it describes the session whose verdict this row replays.
    ``verdict_source``/``reused_from`` are covered by the row's own
    result-envelope self-hash, so stripping the marking breaks the row.
    """

    reused_from = {
        "gate_id": str(prior["gate_id"]),
        "gate_identity": str(prior["gate_identity"]),
        "result_envelope_sha256": str(prior["result_envelope_sha256"]),
    }
    seed = stable_hash(
        {
            "reused_from": reused_from,
            "stable_ids": [str(item.get("stable_id")) for item in items],
            "lineages": [str(item.get("campaign_root_work_id")) for item in items],
        }
    )
    raw: JsonObject = {
        "schema_version": GATE_SCHEMA_VERSION_V3,
        "gate_id": GATE_ID_PREFIX + seed.removeprefix("sha256:")[:32],
        "gate_kind": gate_kind,
        "batch_size": len(items),
        "gate_round": 1,
        "gate_identity": seed,
        "checker": deepcopy(dict(prior["checker"])),
        "items": items,
        "author_result_schema_identity": prior["author_result_schema_identity"],
        "dispatcher_identity": prior["dispatcher_identity"],
        GATE_VERDICT_SOURCE_FIELD: GATE_VERDICT_SOURCE_REUSED,
        GATE_REUSED_FROM_FIELD: reused_from,
    }
    raw["result_envelope_sha256"] = stable_hash(
        {key: value for key, value in raw.items() if key not in _SELF_HASH_EXCLUDED}
    )
    return raw
