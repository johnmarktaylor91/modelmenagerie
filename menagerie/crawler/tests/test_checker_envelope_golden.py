"""Golden checker transcripts through the PRODUCTION validator and routing.

The corpus under ``golden_checker_transcripts/`` is REAL recorded traffic --
exact ``request.json``/``result.json`` pairs the production dispatch froze and
gpt-5.6-terra (metadata, terminal) / gpt-5.6-sol (the pinned fidelity lane)
actually answered, each tied to its durable pilot gates-ledger row. Nothing
here is built by ``make_gate`` or any other test builder: this module is the
one place the suite proves the production envelope loop against checker output
a model REALLY emitted, which the pilot's rigs structurally could not
(2026-08-04: a binding defect was fixed by changing the mock).

Two positive replays run per corpus entry:

* the DURABLE replay strips every machine-owned field and re-runs the exact
  production publication sequence (``apply_machine_owned_gate_fields`` ->
  ``validate_checker_result_mapping``), proving the current tree still parses,
  stamps, and validates the real checker-authored bytes;
* the VERBATIM replay feeds the recorded pair unchanged; the ONLY tolerated
  refusal is component-identity drift (the shipped dispatcher/schema component
  bytes moved since capture), named exactly. Anything else failing on recorded
  traffic is a validator or capture regression.

Re-record the corpus with ``capture_golden_checker_transcripts.py`` whenever
the checker rotates.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.checker_dispatch import (
    DETERMINISTIC_GATE_SCAFFOLD_FIELDS,
    CheckerDispatchError,
    CheckerPartialResultError,
    apply_machine_owned_gate_fields,
    compute_result_envelope_sha256,
    validate_checker_result_mapping,
)
from menagerie.crawler.constants import (
    GATE_MAX_REPAIRS,
    GATE_SCHEMA_VERSION_V3,
    METADATA_BATCH_MAX,
    GateRoute,
    metadata_item_accepted,
)
from menagerie.crawler.driver_models import (
    _chargeable_metadata_repair_count,
    _metadata_repair_count,
    _terminal_metadata_gate,
)
from menagerie.crawler.gates import (
    emit_gate_records,
    route_fidelity_gate,
    route_metadata_gate,
)
from menagerie.crawler.identity import stable_hash
from menagerie.crawler.schema import validate_payload

pytestmark = pytest.mark.smoke

CORPUS_DIR = Path(__file__).parent / "golden_checker_transcripts"
CORPUS_ENTRIES = (
    "metadata_accept",
    "metadata_refusal",
    "fidelity_accept",
    "fidelity_refusal",
    "terminal_accept",
    "terminal_refusal",
    # Richest-of-class entry: the capture tool's smallest-of-role policy is
    # structurally blind to disputed integrity (see the tool's docstring).
    "terminal_disputed",
    # An adverse RUNG check (the reason kernel's identity-mismatch arm).
    "metadata_rung_adverse",
    # Additive round-3 accept exemplar (2026-08-29): metadata_accept's m10470
    # replays as a deterministic non-accept under the lineage-sufficiency rule,
    # so the metadata accept path needed a fresh witness; this pair (m10949)
    # machine-attests every excerpt valid and asks a question no other entry
    # asks. metadata_accept stays committed as the recorded adjudication and
    # deterministic-abstention measurement it still is.
    "metadata_accept_lineage_grounded",
)
# Multi-item shape classes -- per-item route independence
# (``metadata_mixed_batch``) and cap-width heterogeneity with a disputed item
# (``metadata_mixed_disputed``) -- exist only when a production batch can carry
# more than one item. At the calibrated width-1 production config
# (constants.py, Opus adjudication 5.3 rule 1) no recorded session can ever
# populate them, so their entries are retired from the committed corpus; the
# capture tool keeps their class predicates, and a future width increase
# re-records them and re-arms this tuple in the same reviewed change.
if METADATA_BATCH_MAX > 1:
    CORPUS_ENTRIES = (*CORPUS_ENTRIES, "metadata_mixed_disputed", "metadata_mixed_batch")
#: Fields the production wrapper owns and stamps; everything else in a result
#: is checker-authored content the golden replay must preserve byte-for-byte.
MACHINE_OWNED_RESULT_FIELDS = (
    *DETERMINISTIC_GATE_SCAFFOLD_FIELDS,
    "checker",
    "result_envelope_sha256",
)
COMPONENT_IDENTITY_FIELDS = ("author_result_schema_identity", "dispatcher_identity")


def _entry(name: str) -> dict[str, Any]:
    return json.loads((CORPUS_DIR / f"{name}.json").read_text())


def _restamp(result: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    """Re-run the production publication sequence over the checker-authored bytes."""

    raw = {
        key: deepcopy(value)
        for key, value in result.items()
        if key not in MACHINE_OWNED_RESULT_FIELDS
    }
    return apply_machine_owned_gate_fields(
        raw,
        request,
        started_at=str(result["checker"]["started_at"]),
        finished_at=str(result["checker"]["finished_at"]),
    )


@pytest.mark.parametrize("name", CORPUS_ENTRIES)
def test_golden_transcript_replays_the_production_publication_path(name: str) -> None:
    """Real checker-authored bytes stamp and validate cleanly on the current tree."""

    entry = _entry(name)
    restamped = _restamp(entry["result"], entry["request"])
    validated = validate_checker_result_mapping(restamped, entry["request"])
    # The checker-authored content -- the entire adjudication -- is preserved
    # verbatim; only the machine scaffold was re-derived by the current tree.
    assert validated["items"] == entry["result"]["items"]
    assert validated["gate_kind"] == entry["result"]["gate_kind"]
    assert validated["checker"] == entry["result"]["checker"]
    for field in ("gate_id", "gate_identity", "batch_size", "gate_round", "schema_version"):
        assert validated[field] == entry["result"][field], field


@pytest.mark.parametrize("name", CORPUS_ENTRIES)
def test_golden_transcript_validates_verbatim_or_names_component_drift(name: str) -> None:
    """The recorded pair validates as-is; ONLY component drift may refuse it.

    ``author_result_schema_identity``/``dispatcher_identity`` are derived from
    the currently shipped component bytes, so a tree that has legitimately
    moved those components refuses the recorded scaffold -- named exactly. Any
    OTHER refusal of recorded traffic is a regression; re-record the corpus on
    rotation rather than loosening this.
    """

    entry = _entry(name)
    try:
        validated = validate_checker_result_mapping(entry["result"], entry["request"])
    except CheckerDispatchError as exc:
        message = str(exc)
        assert "is not the machine-owned value for its envelope" in message, message
        assert any(field in message for field in COMPONENT_IDENTITY_FIELDS), message
    else:
        assert validated == entry["result"]


@pytest.mark.parametrize("name", CORPUS_ENTRIES)
def test_golden_self_hashes_replay_and_ledger_rows_stay_valid(name: str) -> None:
    """Transcript and ledger self-hashes replay under the current digest."""

    entry = _entry(name)
    result = entry["result"]
    assert result["result_envelope_sha256"] == compute_result_envelope_sha256(result)
    row = entry["ledger_row"]
    assert row is not None, "corpus entries must tie back to the durable gates ledger"
    recomputed = stable_hash(
        {
            key: value
            for key, value in row.items()
            if key not in {"result_envelope_sha256", "payload_sha256", "ledger_seq"}
        }
    )
    assert row["result_envelope_sha256"] == recomputed
    validate_payload(row, GATE_SCHEMA_VERSION_V3)
    assert emit_gate_records(row)


def test_golden_metadata_routes_follow_the_recorded_verdicts() -> None:
    """Production routing over the recorded rows lands where the pilot did."""

    accept_row = _entry("metadata_accept")["ledger_row"]
    for decision in route_metadata_gate(accept_row, {}, max_repairs=2):
        assert decision.route is GateRoute.ACCEPT
        assert decision.canonical_write_allowed is True

    refusal = _entry("metadata_refusal")
    adverse = {
        str(item["stable_id"])
        for item in refusal["result"]["items"]
        if item["verdict"] != "accurate"
    }
    assert adverse, "the refusal transcript must carry a recorded rejection"
    for decision in route_metadata_gate(refusal["ledger_row"], {}, max_repairs=2):
        if decision.stable_id in adverse:
            assert decision.route is not GateRoute.ACCEPT
            assert decision.canonical_write_allowed is False
            assert decision.required_repairs


@pytest.mark.skipif(
    METADATA_BATCH_MAX < 2,
    reason=(
        "per-item route independence inside one session has no consumer at the "
        "calibrated width-1 production config: the envelope builder refuses a "
        "second item, so no recorded multi-item session can exist to pin"
    ),
)
def test_golden_cap_width_batch_routes_per_item_not_per_batch() -> None:
    """One real cap-width session, five items, five independently scored routes.

    A batched lane's whole job is per-item adjudication, and this is the only
    place the suite proves it over traffic a checker really emitted at the
    token-calibrated cap: batch-level collapse in either direction is a
    catastrophe (publishing four rejections, or re-buying an accepted model's
    session forever). Both directions are pinned by requiring the accepted and
    the refused sets to be non-empty AND every route to agree item-by-item with
    the shared acceptance kernel.
    """

    entry = _entry("metadata_mixed_disputed")
    row = entry["ledger_row"]
    assert row["batch_size"] == METADATA_BATCH_MAX, "this golden pins the cap-width shape"

    # Spelled from the recorded bytes, deliberately NOT via
    # ``metadata_item_accepted``: an oracle that calls the kernel under test
    # moves with it, and would pass just as happily on the day routing and the
    # kernel disagree again.
    expected = {
        str(item["stable_id"]): (
            item["verdict"] == "accurate"
            and item["integrity"]["verdict"] == "accurate"
            and item["rung_check"]["verdict"] == "accurate"
        )
        for item in entry["result"]["items"]
    }
    assert expected == {
        str(item["stable_id"]): metadata_item_accepted(item) for item in entry["result"]["items"]
    }, "the shared acceptance kernel disagrees with the recorded component verdicts"
    assert any(expected.values()) and not all(expected.values()), (
        f"a vacuous golden proves nothing about per-item routing: {expected}"
    )

    routed = {}
    for decision in route_metadata_gate(row, {}, max_repairs=2):
        accepted = decision.route is GateRoute.ACCEPT
        routed[decision.stable_id] = accepted
        # Publication authorization and repair demand must track the SAME
        # per-item predicate -- the defect class where routing accepted on
        # verdict alone published items every other authority scored a rejection.
        assert decision.canonical_write_allowed is accepted, decision.stable_id
        assert bool(decision.required_repairs) is not accepted, decision.stable_id
    assert routed == expected


def test_golden_disputed_integrity_is_never_published_whatever_the_verdict() -> None:
    """Real items the checker could not re-attest are refused publication.

    ``integrity`` is the checker's answer to "could I re-derive this evidence at
    all", and it is orthogonal to the content verdict: an item can be judged
    accurate over excerpt locators that no longer resolve. The write door is the
    one site that authorizes publication, so a disputed item reaching ACCEPT
    there publishes evidence the adjudication itself disputed.
    """

    checked = 0
    for name in CORPUS_ENTRIES:
        entry = _entry(name)
        if entry["kind"] != "metadata_batch" or entry["ledger_row"] is None:
            continue
        disputed = {
            str(item["stable_id"])
            for item in entry["result"]["items"]
            if item["integrity"]["verdict"] != "accurate"
        }
        for decision in route_metadata_gate(entry["ledger_row"], {}, max_repairs=2):
            if decision.stable_id in disputed:
                assert decision.route is not GateRoute.ACCEPT, decision.stable_id
                assert decision.canonical_write_allowed is False, decision.stable_id
                checked += 1
    assert checked, "the corpus must carry at least one real disputed-integrity metadata item"


def test_golden_terminal_disputed_integrity_rejects_in_lockstep() -> None:
    """The terminal lane's disputed-evidence arm, as really recorded.

    A terminal advisory decides whether a model leaves the campaign permanently,
    so a disputed-evidence terminal must restate its refusal in the typed
    disposition rather than riding the content verdict alone.
    """

    entry = _entry("terminal_disputed")
    item = entry["result"]["items"][0]
    assert item["integrity"]["verdict"] != "accurate"
    assert item["verdict"] != "accurate"
    assert item["terminal_disposition"]["verdict"] == "rejected"
    envelope_item = entry["request"]["items"][0]
    terminal = item["terminal_disposition"]
    assert terminal["author_result_id"] == envelope_item["author_result"]["result_id"]
    assert terminal["author_result_sha256"] == envelope_item["author_result"]["result_sha256"]


def test_corpus_still_covers_the_shapes_a_smallest_of_role_recapture_drops() -> None:
    """Coverage floor: the corpus keeps its width, heterogeneity, and disputes.

    Re-recording selects fresh pairs from whatever campaign it is pointed at, and
    the classes added here are exactly the ones a size-minimizing policy discards
    first. Without this floor the corpus silently reverts to all-uniform,
    all-attestable, near-single-item traffic -- which is the state in which the
    split metadata acceptance predicate survived every golden test.
    """

    entries = {name: _entry(name) for name in CORPUS_ENTRIES}
    items = {name: entry["result"]["items"] for name, entry in entries.items()}

    assert any(
        len(batch) == METADATA_BATCH_MAX
        for name, batch in items.items()
        if entries[name]["kind"] == "metadata_batch"
    ), "no cap-width metadata batch left in the corpus"
    if METADATA_BATCH_MAX > 1:
        # A single-item session cannot spread verdicts, so this floor binds
        # only while the production width admits multi-item sessions.
        assert any(len({item["verdict"] for item in batch}) > 1 for batch in items.values()), (
            "no heterogeneous-verdict session left in the corpus"
        )
    for lane in ("metadata_batch", "terminal_disposition"):
        assert any(
            item["integrity"]["verdict"] != "accurate"
            for name, batch in items.items()
            for item in batch
            if entries[name]["kind"] == lane
        ), f"no disputed-integrity transcript left on the {lane} lane"


def test_golden_fidelity_routes_follow_the_recorded_verdicts() -> None:
    """The recorded sol fidelity verdicts route through the production gate."""

    for name, accepted in (("fidelity_accept", True), ("fidelity_refusal", False)):
        entry = _entry(name)
        proposal = entry["request"]["items"][0]["proposal"]
        decision = route_fidelity_gate(entry["ledger_row"], proposal)
        assert decision.accepted_for_fidelity is accepted
        assert decision.route is (GateRoute.ACCEPT if accepted else GateRoute.BLOCK_FIDELITY)


def test_golden_terminal_transcripts_carry_lockstep_recorded_dispositions() -> None:
    """Recorded terminal dispositions restate their verdicts in lockstep."""

    for name, disposition in (("terminal_accept", "accepted"), ("terminal_refusal", "rejected")):
        entry = _entry(name)
        item = entry["result"]["items"][0]
        assert item["terminal_disposition"]["verdict"] == disposition
        # Era pin: the corpus re-record after the all-sol rotation (AG-4/W2)
        # converges the terminal lane on the pinned adjudication authority.
        assert entry["checker_model"] == "gpt-5.6-sol"
        # The recommendation identities the checker ruled over are the machine
        # pack's, copied exactly -- the binding a fabricated ruling cannot fake.
        envelope_item = entry["request"]["items"][0]
        terminal = item["terminal_disposition"]
        assert terminal["author_result_id"] == envelope_item["author_result"]["result_id"]
        assert terminal["author_result_sha256"] == envelope_item["author_result"]["result_sha256"]
        assert terminal["evidence_identity"] == envelope_item["evidence_pack"]["evidence_identity"]
        assert terminal["license_identity"] == envelope_item["license_identity"]


# ---------------------------------------------------------------------------
# Armed-tripwire goldens: real bytes, minimally corrupted, MUST refuse.
# These are the tests that die if the validator is killed or hollowed out.
# ---------------------------------------------------------------------------


def test_tampered_golden_reasoning_breaks_the_result_self_hash() -> None:
    """Rewriting recorded free-text reasoning refuses on the self-hash alone.

    Free text trips no semantic cross-check, so ONLY the result-envelope hash
    stands between a silently edited adjudication and publication. The tamper
    lands on the RESTAMPED result -- the current tree's own scaffold, exactly
    like the durable replay -- so component-identity drift of the recorded
    scaffold (a shipped dispatcher/schema component legitimately moved since
    capture) can never preempt the self-hash and mask this tripwire; the
    verbatim-drift path keeps its own named test above.
    """

    entry = _entry("metadata_accept")
    tampered = _restamp(entry["result"], entry["request"])
    tampered["items"][0]["field_checks"][0]["reason"] = "silently rewritten adjudication"
    with pytest.raises(CheckerDispatchError, match="result_envelope_sha256"):
        validate_checker_result_mapping(tampered, entry["request"])


def test_resealed_fabricated_verdict_contradicts_its_own_checks() -> None:
    """A rejection fabricated over accurate checks refuses even fully resealed."""

    entry = _entry("metadata_accept")
    fabricated = _restamp(entry["result"], entry["request"])
    fabricated["items"][0]["verdict"] = "inaccurate"
    fabricated["result_envelope_sha256"] = compute_result_envelope_sha256(fabricated)
    with pytest.raises(CheckerDispatchError):
        validate_checker_result_mapping(fabricated, entry["request"])


def test_truncated_golden_result_is_refused_as_partial() -> None:
    """Dropping one item from the recorded batch refuses as partial.

    At a multi-item production width the truncation leaves a short batch; at
    width 1 it leaves an EMPTY result. Both are the same defect class -- a
    checker answer covering fewer items than its envelope asked -- and both
    must refuse rather than validate.
    """

    entry = _entry("metadata_accept")
    assert entry["result"]["batch_size"] == METADATA_BATCH_MAX
    truncated = deepcopy(entry["result"])
    truncated["items"] = truncated["items"][:-1]
    with pytest.raises((CheckerPartialResultError, CheckerDispatchError)) as excinfo:
        validate_checker_result_mapping(truncated, entry["request"])
    # Preferred is the typed partial; schema-first refusal is still a refusal,
    # but silently ACCEPTING a truncated batch is the whole-leg killer.
    assert excinfo.type in (CheckerPartialResultError, CheckerDispatchError)


def test_golden_result_under_a_foreign_envelope_is_refused() -> None:
    """A real result presented under a different real envelope never validates."""

    accept = _entry("metadata_accept")
    refusal = _entry("metadata_refusal")
    with pytest.raises(CheckerDispatchError):
        validate_checker_result_mapping(accept["result"], refusal["request"])


def test_recorded_component_drift_transcript_is_refused_by_name() -> None:
    """A real pair sealed under retired component bytes refuses exactly.

    This corpus entry is a genuine pilot transcript whose recorded
    ``dispatcher_identity`` no longer matches the shipped component -- the
    validator must keep naming that drift instead of validating stale identity.
    """

    entry = _entry("metadata_component_drift")
    with pytest.raises(CheckerDispatchError, match="machine-owned value for its envelope"):
        validate_checker_result_mapping(entry["result"], entry["request"])


def _reseal_ledger_row(row: dict[str, Any]) -> dict[str, Any]:
    """Recompute a golden ledger row's result hash after an adversarial edit."""

    row["result_envelope_sha256"] = stable_hash(
        {
            key: value
            for key, value in row.items()
            if key not in {"result_envelope_sha256", "payload_sha256", "ledger_seq"}
        }
    )
    return row


@pytest.mark.skipif(
    METADATA_BATCH_MAX < 2,
    reason=(
        "a session mixing accepted with non-accepted items cannot exist at the "
        "calibrated width-1 production config; the class re-records and this "
        "test re-arms with any future width increase"
    ),
)
def test_golden_mixed_batch_routes_each_item_independently() -> None:
    """One recorded session carrying accept + reject + cannot-verify routes per item.

    The pilot's dominant adverse class is ``cannot-verify`` (612 of 852 adverse
    field checks); this real three-item batch pins that an accepted sibling's
    write stays allowed while the inaccurate and cannot-verify siblings each
    requeue with repairs -- no verdict bleeds across items in one session.
    """

    entry = _entry("metadata_mixed_batch")
    verdicts = {str(item["stable_id"]): str(item["verdict"]) for item in entry["result"]["items"]}
    assert set(verdicts.values()) == {"accurate", "inaccurate", "cannot-verify"}
    decisions = {
        decision.stable_id: decision
        for decision in route_metadata_gate(entry["ledger_row"], {}, max_repairs=2)
    }
    for stable_id, verdict in verdicts.items():
        decision = decisions[stable_id]
        if verdict == "accurate":
            assert decision.route is GateRoute.ACCEPT
            assert decision.canonical_write_allowed is True
        else:
            assert decision.route is GateRoute.REQUEUE_NEXT_BATCH
            assert decision.canonical_write_allowed is False
            assert decision.required_repairs


def test_golden_rung_adverse_item_derives_the_kernel_reason() -> None:
    """A real rung-adverse item derives its reason from the ONE shared kernel.

    The recorded item carries ``verdict: inaccurate``, ``integrity:
    inaccurate``, ``rung_check: cannot-verify`` -- the exact shape behind the
    pilot's recorded ``identity-mismatch`` terminals (m4066 class). The shared
    kernel's precedence (an integrity dispute on a determined round is an
    identity dispute, never a merits cap) must keep deriving that reason over
    the real bytes.
    """

    from menagerie.crawler.constants import metadata_gate_failure_reason

    entry = _entry("metadata_rung_adverse")
    adverse = [
        item
        for item in entry["result"]["items"]
        if item["verdict"] != "accurate" or item.get("rung_check", {}).get("verdict") != "accurate"
    ]
    assert adverse, "the rung-adverse transcript must carry an adverse item"
    item = adverse[0]
    assert item["integrity"]["verdict"] != "accurate"
    assert metadata_gate_failure_reason(item) == "identity-mismatch"
    decisions = {
        decision.stable_id: decision
        for decision in route_metadata_gate(entry["ledger_row"], {}, max_repairs=2)
    }
    assert decisions[str(item["stable_id"])].route is GateRoute.REQUEUE_NEXT_BATCH


def test_golden_metadata_router_refuses_acceptance_over_degraded_integrity() -> None:
    """A self-consistent row hash cannot launder a contradictory acceptance.

    This is a real ACCEPTED metadata ledger row with only the checker-authored
    integrity verdict changed, then honestly resealed -- an adversarial but
    self-hash-valid persisted payload. The router may not mint acceptance from
    the favorable top-level/rung fields while ignoring the item's contradictory
    integrity verdict.

    SYN-B adjudication: lab c2c asserted this as a ``GateRoutingError`` raised
    by re-running the dispatch-door precedence check at the routing door; lab
    c2b asserted it as a NON-ACCEPTANCE from the shared acceptance kernel. The
    merge took c2b's mechanism (see ``SYN_B_DOCKET.md``, C2 ruling 3), so the
    assertion below states the security property both mechanisms exist for --
    no acceptance, no canonical write, and a repair route -- rather than the
    mechanism c2c chose. Nothing is weakened: publication is refused either way,
    and ``test_gate_acceptance_parity`` covers the same cell exhaustively.
    """

    row = deepcopy(_entry("metadata_accept")["ledger_row"])
    row["items"][0]["integrity"]["verdict"] = "inaccurate"
    _reseal_ledger_row(row)
    decision = route_metadata_gate(row, {}, max_repairs=2)[0]
    assert decision.route is not GateRoute.ACCEPT
    assert decision.canonical_write_allowed is False
    assert metadata_item_accepted(row["items"][0]) is False


@pytest.mark.parametrize("name", CORPUS_ENTRIES)
def test_golden_transcript_refuses_a_resealed_top_level_verdict_flip(name: str) -> None:
    """Every real lane rejects a top-level verdict contradicting its components."""

    entry = _entry(name)
    fabricated = _restamp(entry["result"], entry["request"])
    current = fabricated["items"][0]["verdict"]
    fabricated["items"][0]["verdict"] = "inaccurate" if current == "accurate" else "accurate"
    fabricated["result_envelope_sha256"] = compute_result_envelope_sha256(fabricated)
    with pytest.raises(CheckerDispatchError, match="contradict|less severe"):
        validate_checker_result_mapping(fabricated, entry["request"])


def test_pre_prompt1_real_refusal_copies_charge_each_answered_repair_round() -> None:
    """Copied real checker payloads cannot reset or evade repair accounting.

    ``metadata_refusal`` is a recorded pre-PROMPT1 checker payload. Replaying
    its exact item as three durable author repair generations models the costly
    historical behavior PROMPT1 was designed to stop: the author copies the
    payload without addressing either disclosed complaint. Raw generation and
    chargeable-cap accounting must both advance; copied prose or gate IDs must
    not make an answered complaint look newly disclosed.
    """

    original = _entry("metadata_refusal")["ledger_row"]
    gates: list[dict[str, Any]] = []
    for gate_round in range(1, GATE_MAX_REPAIRS + 2):
        copied = deepcopy(original)
        copied["gate_id"] = f"gate-prompt1-copy-{gate_round}"
        copied["gate_identity"] = stable_hash(
            {"pre_prompt1_copy": original["gate_identity"], "round": gate_round}
        )
        copied["gate_round"] = gate_round
        copied["ledger_seq"] = gate_round
        gates.append(_reseal_ledger_row(copied))

    item = original["items"][0]
    stable_id = str(item["stable_id"])
    campaign_id = str(item["campaign_root_work_id"])

    assert _metadata_repair_count(gates, stable_id, campaign_id) == 3
    assert _chargeable_metadata_repair_count(gates, stable_id, campaign_id) == 3
    terminal = _terminal_metadata_gate(gates, stable_id, campaign_id, max_repairs=GATE_MAX_REPAIRS)
    assert terminal is not None
    assert terminal["gate_id"] == gates[-1]["gate_id"]


def test_corpus_provenance_is_recorded_traffic_not_a_builder() -> None:
    """Every corpus entry names its frozen source pair and its ledger row."""

    for name in CORPUS_ENTRIES:
        entry = _entry(name)
        assert entry["source"]["request"].endswith("request.json")
        assert entry["source"]["result"].endswith("result.json")
        assert entry["ledger_gate_id"], name
        assert entry["request"]["checker"]["model"] == entry["result"]["checker"]["model"]
    # Era pin: the width-1 re-harvest (2026-08-29) retired the last terra pair
    # (the cap-width mixed_disputed golden, which cannot re-seal at width 1);
    # every committed entry now records the pinned sol adjudication authority.
    models = {_entry(name)["checker_model"] for name in CORPUS_ENTRIES}
    assert models == {"gpt-5.6-sol"}
