"""Rung-7 checker-calibration pins: prompt clauses and their machine mirrors.

The rung-7 adjudication (dual blind cross-lab grading of the 8 frozen
``terminal_disposition`` gate verdicts) found every disposition correct on the
merits and exactly three prompt-clarity gaps, closed by three additive clauses
in ``codex_accuracy_checker_v2.txt``:

1. the rejected/cannot-verify boundary on a terminal item (a resolved pack
   whose excerpts do not entail the arm's own predicate is an OVERSTATED-support
   rejection; cannot-verify is reserved for evidence that could not be resolved
   or bound) -- the frozen rung-7 ledger split 7 rejected / 1 cannot-verify on
   materially identical wall-exhaustion claims because this boundary was
   unstated;
2. the typed inapplicable ``rung_check`` form on a terminal item -- the eight
   frozen sessions improvised four different shapes for a block nothing
   terminal-side consumes;
3. the exact R4 ``search-attested:`` / ``search-cannot-verify:`` correlation,
   which ``metadata._validate_rung_and_search_attestation`` already enforces
   mechanically but the prompt only half-stated.

This module pins each clause into the shipped prompt bytes (so the calibration
cannot silently drift back out) and pins the two machine facts the clauses
mirror: terminal routing is ``rung_check``-inert, and the R4 correlation stated
in prose is byte-for-byte the rule the metadata validator enforces. None of
these pins loosens anything: every probe that failed before still fails.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from menagerie.crawler.gates import GateRoutingError, validate_terminal_disposition_gate
from menagerie.crawler.metadata import (
    MetadataValidationError,
    _validate_rung_and_search_attestation,
)
from menagerie.crawler.schema import validate_payload
from menagerie.crawler.tests.conftest import HASH
from menagerie.crawler.tests.test_slice_d_checker_gates import _blocked_terminal_fixture

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "codex_accuracy_checker_v2.txt"

#: The exact typed inapplicable ``rung_check`` the prompt prescribes for a
#: ``terminal_disposition`` item. A terminal advisory arm selects no
#: implementation rung, so this block is canonical filler; the tests below
#: prove the deterministic terminal route never reads it.
TYPED_INAPPLICABLE_RUNG_CHECK = {
    "selected_rung": "R5_SKIP",
    "highest_applicable": "R5_SKIP",
    "verdict": "accurate",
    "findings": ["not-applicable: terminal advisory arm; no implementation rung selected"],
}

#: rung_check shapes the eight frozen rung-7 checker sessions actually emitted
#: on terminal items (improvised, because no canonical form was stated).
_RUNG7_IMPROVISED_RUNG_CHECKS = (
    {
        "selected_rung": "R1_LIBRARY",
        "highest_applicable": "R1_LIBRARY",
        "verdict": "accurate",
        "findings": ["an available library implementation route exists"],
    },
    {
        "selected_rung": "R5_SKIP",
        "highest_applicable": "R5_SKIP",
        "verdict": "cannot-verify",
        "findings": ["no independently evidenced skip rationale was supplied"],
    },
)


def _prompt_text() -> str:
    """Return the shipped checker prompt bytes as text."""

    return _PROMPT_PATH.read_text(encoding="utf-8")


@pytest.mark.smoke
def test_prompt_states_the_terminal_rejected_cannot_verify_boundary() -> None:
    """The rejected/cannot-verify boundary clause ships in the prompt bytes.

    Rung 7 split 7 rejected / 1 cannot-verify (m5445) over materially identical
    author-process wall claims; the boundary was checker-session jitter until
    the prompt stated it. These literals are the load-bearing phrases.
    """

    text = _prompt_text()
    assert "OVERSTATES its support and the disposition is rejected" in text
    assert "never source-establishable" in text
    assert "Reserve cannot-verify for evidence you could not" in text


@pytest.mark.smoke
def test_prompt_states_the_verdict_bound_terminal_source_ids_semantics() -> None:
    """The verdict-bound source_ids/evidence_ids clause ships in the prompt bytes.

    Rung 10 leg 1 killed four models on a spec-internal contradiction: the
    gate-v3 prose said ``source_ids`` is the full checked set while
    ``gates._validate_terminal_rejection_subset`` (the deliberate rung-8 subset
    binding) demands exactly the sources the cited evidence resolves to. The
    subset binding is the contract -- the full set is already hash-bound twice
    on the same item -- so the prompt now states the ONE semantics, including
    the single empty-citation state (cannot-verify over an unresolved pack)
    that was previously unsatisfiable. These literals are the load-bearing
    phrases; the machine mirrors live in test_slice_d_checker_gates.
    """

    text = _prompt_text()
    unwrapped = " ".join(text.split())
    assert "`source_ids` and `evidence_ids` are VERDICT-BOUND" in unwrapped
    assert "copy `evidence_pack.declared_evidence_ids` into `evidence_ids`" in unwrapped
    assert "`evidence_pack.checked_source_ids` into `source_ids`" in unwrapped
    assert "resolve it row by row from the literal excerpt table" in unwrapped
    assert "Never copy `checked_source_ids` into a non-accepted `source_ids`" in unwrapped
    assert "carries `evidence_ids: []` and `source_ids: []`" in unwrapped
    assert "whenever any excerpt was resolved, an empty citation is refused" in unwrapped


@pytest.mark.smoke
def test_the_verdict_bound_source_ids_clause_is_the_machine_rule() -> None:
    """Both directions of the stated source_ids rule are what the router enforces.

    A non-accepted ``source_ids`` equal to the full checked set is refused; the
    literal-resolution set is bound. The prompt clause and the validator must
    agree or checker sessions get mechanically discarded for obeying the prose
    -- exactly how rung-10 leg 1 burned m3671, m4256, m4257, and m538.
    """

    manifest = ("source-1", "source-2", "source-3")
    gate, result, source_manifest, evidence_pack = _blocked_terminal_fixture(
        manifest, ("evidence-1",)
    )
    gate["items"][0]["terminal_disposition"].update(
        {
            "verdict": "rejected",
            "evidence_ids": ["evidence-1"],
            "findings": ["the excerpt does not establish the claimed blocker"],
        }
    )

    # The stale prose reading -- the full checked set -- is refused.
    probe = deepcopy(gate)
    probe["items"][0]["terminal_disposition"]["source_ids"] = list(manifest)
    with pytest.raises(
        GateRoutingError,
        match=r"^terminal disposition source IDs do not match adjudicated evidence$",
    ):
        validate_terminal_disposition_gate(
            probe,
            result,
            source_manifest=source_manifest,
            evidence_pack=evidence_pack,
            license_identity=HASH,
        )

    # The prompt-stated reading -- exactly the cited excerpts' sources -- binds.
    gate["items"][0]["terminal_disposition"]["source_ids"] = ["source-1"]
    decision = validate_terminal_disposition_gate(
        gate,
        result,
        source_manifest=source_manifest,
        evidence_pack=evidence_pack,
        license_identity=HASH,
    )
    assert decision.accepted is False
    assert decision.source_ids == ("source-1",)


@pytest.mark.smoke
def test_prompt_keys_r1_inventory_adjudication_on_the_machine_stamp() -> None:
    """The R1 inventory clause is keyed on the coordinator's inventory_mode stamp.

    The 7e4a7e93 repair closed the POSITIVE absent-inventory assertion arm and
    left the negative/meta arm open: m10759's rung rationale quotes its own
    dispatch envelope's `inventory_disclosed: false`, which some same-epoch
    draws read as an unverifiable inventory assertion and others as a machine
    scope fact (the width-1 fidelity flip). The rule-2 closure makes the
    disclosure mode a MACHINE STAMP (`inventory_mode`, coordinator-derived
    from the manifest's `env-routed-inventory` row) and states the negative
    arm explicitly: restating non-disclosure is never an inventory fact
    needing frozen basis, while a specific package IS/IS-NOT claim always is.
    These literals are the load-bearing phrases.
    """

    text = _prompt_text()
    unwrapped = " ".join(text.split())
    assert "THE DISCLOSURE MODE IS MACHINE-STAMPED" in unwrapped
    assert "Never re-derive the mode from prose" in unwrapped
    assert "the row's absence is a MACHINE SCOPE FACT, never a defect and never evidence" in unwrapped
    assert "merely restates the machine stamp, asserts no inventory fact" in unwrapped
    assert (
        "Judge R1 source-ladder applicability solely from the pinned maintained "
        "public-library bytes and exact recipe the pack carries" in unwrapped
    )
    assert "SPECIFIC package or version IS or IS NOT installed" in unwrapped
    assert "name the asserting claim in the finding" in unwrapped
    assert (
        "Never demand a frozen inventory that was never disclosed; never accept an "
        "inventory assertion that has none" in unwrapped
    )
    # The prior epoch's underdetermined hinge is gone: what the justification
    # "rests on" was a per-draw reading choice on the negative/meta arm.
    assert "decide by what the proposal's own rung justification RESTS ON" not in unwrapped


@pytest.mark.smoke
def test_prompt_withdraws_locator_adjudication_from_model_judgment() -> None:
    """Excerpt-locator adjudication is machine-owned; the prompt says so.

    m10470 se2 asserted 21 mechanically false line claims about frozen bytes
    (the section heading's line reported as the excerpt's line) and
    mass-rejected a clean pack; 7,031 mechanical checks across 402 item-draws
    found ZERO real mislocations and exactly one draw that ever claimed
    otherwise. The only positional adjudication in the pipeline was model
    eyeball arithmetic. The prompt now consumes the coordinator-frozen
    attestation and forbids re-deriving or disputing it, with the CRLF/LF
    equivalence stated (the m10759 terminal integrity flip) and the scope
    bounded to evidence excerpts (m8189's true code-locator catch keeps its
    home in the owning check's findings).
    """

    text = _prompt_text()
    unwrapped = " ".join(text.split())
    assert "EXCERPT-LOCATOR AND EXCERPT-EQUIVALENCE ADJUDICATION IS MACHINE-OWNED" in unwrapped
    assert "`locator_attestation` block" in unwrapped
    assert "Never re-derive, recount, or dispute a line number" in unwrapped
    assert (
        "`integrity.locator_failures` and `integrity.excerpt_discrepancies` are exactly []"
        in unwrapped
    )
    assert "disputes machine arithmetic and your entire result is refused typed" in unwrapped
    assert "locator-attestation:<evidence_id>:<status>" in unwrapped
    assert "CRLF vs LF" in unwrapped
    assert "scoped to EVIDENCE EXCERPTS only" in unwrapped
    assert "`hash_mismatches` remains yours" in unwrapped
    # The withdrawn instruction is gone: it delegated a machine-decidable,
    # publication-steering authority to model line-counting.
    assert "exists verbatim at its locator" not in unwrapped


@pytest.mark.smoke
def test_prompt_closes_the_match_minor_drift_precedence() -> None:
    """A declared source-unspecified deviation is minor-drift, never match.

    As previously written, a pack declaring a nonmaterial source-unspecified
    init choice satisfied BOTH definitions literally (match = no sourced
    material item deviates; minor-drift = a declared nonmaterial choice
    differs) -- an unclosed vocabulary overlap sitting in a verdict enum,
    measured flipping on m10759 (match <-> minor-drift, acceptance-neutral
    but zero-bar-relevant on the full projection).
    """

    text = _prompt_text()
    unwrapped = " ".join(text.split())
    assert "the proposal declares NO deviation of any kind" in unwrapped
    assert (
        "any declared nonmaterial source-unspecified deviation makes the verdict "
        "minor-drift, never match, so no pack satisfies both definitions" in unwrapped
    )


@pytest.mark.smoke
def test_prompt_states_the_unattestable_insufficiency_and_row_scope_rules() -> None:
    """m2247's round-3 cell: unattestable scoring + constituent-scope rung.

    The component-drift adverse golden (10 excerpts citing sources the
    manifest never froze) split twice: integrity cannot-verify <-> inaccurate
    (is an unattestable citation fabricated or unverifiable?) and rung_check
    accurate <-> inaccurate (does the row denote the PointTransformerConv
    layer or the PointTransformerConv-net composite the frozen
    impl-net-classification source names?). Both closures key on machine
    vocabulary: the closed attestation status drives a closed integrity rule
    (absence of basis is insufficiency; inaccurate needs positive
    contradiction), poisoning is per-claim so valid-row claims stay judged on
    the merits, and the adjudicated architecture is the row's own scope so a
    constituent recipe can never make the rung accurate.
    """

    text = _prompt_text()
    unwrapped = " ".join(text.split())
    assert (
        "UNATTESTABLE ROWS ARE INSUFFICIENCY, NEVER FABRICATION, AND THEY POISON ONLY "
        "THE CLAIMS THAT CITE THEM" in unwrapped
    )
    assert "the machine cannot refute what it holds no bytes for, and neither can you" in unwrapped
    assert (
        "an item whose only integrity defects are unattestable rows has "
        "`integrity.verdict` cannot-verify" in unwrapped
    )
    assert "integrity inaccurate requires a POSITIVE contradiction" in unwrapped
    assert "never blanket-degraded because unattestable rows exist elsewhere in the pack" in unwrapped
    assert "THE ADJUDICATED ARCHITECTURE IS THE ROW'S, AT THE ROW'S OWN SCOPE" in unwrapped
    assert (
        "a constituent's own library availability never substitutes for the composite's"
        in unwrapped
    )
    assert (
        "genuinely undetermined between a layer and a network, rung_check is "
        "cannot-verify, never accurate" in unwrapped
    )


@pytest.mark.smoke
def test_prompt_states_the_r1_revision_binding_rule() -> None:
    """R1 applicability never bridges an evidence-vs-pinned revision gap.

    m3353's round-3 cell (rung accurate <-> cannot-verify, 1-2): the recipe
    pins torchaudio at the disclosed installed 2.11.0 while every
    architecture-grounding excerpt is frozen from the v2.8.0 tag. Some draws
    read the R1 selection as applicable (the package is installed and the
    constructor is evidenced somewhere), others as unverifiable at the pinned
    revision. The closure is monotone-safe: package presence alone never
    re-attests bytes across a revision gap (assumed API continuity is the
    banned likely-equivalence), so the cell lands on the conservative
    deterministic cannot-verify -- an ordered-abstention pole, disclosed
    insufficiency, never a contradiction and never a bad publication.
    """

    text = _prompt_text()
    unwrapped = " ".join(text.split())
    assert "R1 APPLICABILITY IS REVISION-BOUND" in unwrapped
    assert "AT THE REVISION THEY WERE FROZEN FROM, and only there" in unwrapped
    assert (
        "package presence alone never re-attests the constructor or its builder values "
        "across the revision gap" in unwrapped
    )
    assert "bridging the gap by assumed API continuity is exactly the banned likely-equivalence" in unwrapped
    assert (
        "Unless the pack carries frozen bytes at the pinned revision itself covering the "
        "same material claims, rung_check is cannot-verify" in unwrapped
    )
    # Bounded in both directions: never inaccurate by itself, and same-revision
    # packs are untouched.
    assert "never makes rung_check inaccurate by itself" in unwrapped
    assert "evidence and recipe pinned to the SAME revision are untouched by this rule" in unwrapped


@pytest.mark.smoke
def test_prompt_states_the_lineage_sufficiency_rule() -> None:
    """Architecture-class membership alone never proves a historical antecedent.

    m10470 se3's residual pole: an excerpt classifying T5 as "an
    encoder-decoder transformer" read by some draws as establishing
    lineage/predecessors and by others as insufficient. The closure is
    monotone-safe: it can only turn that coin flip into a deterministic
    cannot-verify (an ordered-abstention pole, disclosed false-reject at
    worst, never a bad publication).
    """

    text = _prompt_text()
    unwrapped = " ".join(text.split())
    assert "LINEAGE/PREDECESSOR SUFFICIENCY" in unwrapped
    assert "architecture-class membership alone never proves a historical antecedent" in unwrapped
    assert "naming the antecedent work in an explicit derivation relation" in unwrapped
    assert "Without such an excerpt, that field check is cannot-verify, never accurate" in unwrapped


@pytest.mark.smoke
def test_prompt_states_the_typed_inapplicable_terminal_rung_check() -> None:
    """The canonical terminal rung_check form ships verbatim in the prompt."""

    text = _prompt_text()
    assert "selected_rung and highest_applicable both R5_SKIP, verdict accurate" in text
    assert TYPED_INAPPLICABLE_RUNG_CHECK["findings"][0] in text


@pytest.mark.smoke
def test_prompt_states_the_exact_r4_search_marker_correlation() -> None:
    """The R4 marker/verdict correlation clause ships in the prompt bytes."""

    text = _prompt_text()
    assert "search_report.links_checked" in text
    assert "each link spelled byte-identically, the set complete" in text
    assert "carries at least one `search-cannot-verify:<reason>`" in text
    assert "All three markers are R4-only" in text


@pytest.mark.smoke
def test_prompt_states_the_r4_attestation_totality_and_receipt_route() -> None:
    """The prompt instructs total per-link attestation, never a spot-check.

    ``metadata._validate_rung_and_search_attestation`` has always demanded an
    attestation for EVERY ``links_checked`` entry before an accurate R4 rung
    verdict, yet the shipped prompt opened with "spot-check the highest-risk
    searches" -- a checker obeying that sentence was mechanically refused on
    every R4, a guaranteed rejection loop. The machine rule is the contract
    (an accurate verdict over a sampled subset blesses unexamined members of a
    closed negative-proof set); these pins hold the prompt to it, including
    the receipt-bound route that makes honest coverage COMPLETABLE for typed
    ``{url, disposition}`` links no checker could ever re-fetch.
    """

    text = _prompt_text()
    unwrapped = " ".join(text.split())
    assert "re-attest EVERY link" in unwrapped
    assert "never a sample" in unwrapped
    assert "spot-check the highest-risk" not in unwrapped
    assert "`search-receipt-attested:<exact checked link>`" in unwrapped
    assert "cannot be re-fetched by design" in unwrapped
    assert "Never swap the routes" in unwrapped


@pytest.mark.smoke
def test_terminal_routing_is_rung_check_inert() -> None:
    """One terminal gate routes identically under every rung_check shape.

    ``validate_terminal_disposition_gate`` keys on the terminal_disposition
    block, the exact identities, and item integrity -- never on ``rung_check``.
    Standardizing the typed inapplicable form therefore cannot drift any
    terminal verdict: the decision under the canonical form equals the decision
    under both improvised rung-7 shapes, field for field.
    """

    gate, result, source_manifest, evidence_pack = _blocked_terminal_fixture(
        ("source-1", "source-2", "source-3"), ("evidence-1",)
    )

    decisions = []
    for rung_check in (TYPED_INAPPLICABLE_RUNG_CHECK, *_RUNG7_IMPROVISED_RUNG_CHECKS):
        probe = deepcopy(gate)
        probe["items"][0]["rung_check"] = deepcopy(rung_check)
        validate_payload(probe)
        decisions.append(
            validate_terminal_disposition_gate(
                probe,
                result,
                source_manifest=source_manifest,
                evidence_pack=evidence_pack,
                license_identity=HASH,
            )
        )

    first = decisions[0]
    assert first.accepted is True
    assert first.predicate == "blocked-prerequisite"
    for other in decisions[1:]:
        assert other == first


def _r4_probe(
    *,
    verdict: str,
    findings: list[str],
    links: list[object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Build minimal facts and gate-item mappings for the R4 correlation rule."""

    facts = {
        "source_resolution": {
            "rung": "R4_REIMPLEMENT",
            "search_report": {"links_checked": links},
        }
    }
    gate_item = {
        "rung_check": {
            "selected_rung": "R4_REIMPLEMENT",
            "highest_applicable": "R4_REIMPLEMENT",
            "verdict": verdict,
            "findings": findings,
        }
    }
    return facts, gate_item


@pytest.mark.smoke
def test_the_prompt_r4_correlation_is_the_machine_rule() -> None:
    """Every branch of the stated R4 correlation is what the validator enforces.

    The prompt clause and ``_validate_rung_and_search_attestation`` must agree
    or checker sessions get mechanically discarded for obeying the prose. Each
    probe below is one clause of the stated correlation.
    """

    links = ["https://example.org/search-a", "https://example.org/search-b"]

    # accurate + complete byte-identical attestation + no cannot-verify: passes.
    facts, item = _r4_probe(
        verdict="accurate",
        findings=[f"search-attested:{link}" for link in links],
        links=links,
    )
    _validate_rung_and_search_attestation(facts, item)

    # accurate with one link unattested: refused, set-complete is mandatory.
    facts, item = _r4_probe(
        verdict="accurate",
        findings=[f"search-attested:{links[0]}"],
        links=links,
    )
    with pytest.raises(MetadataValidationError, match="lacks re-executed search attestations"):
        _validate_rung_and_search_attestation(facts, item)

    # accurate alongside any search-cannot-verify marker: refused.
    facts, item = _r4_probe(
        verdict="accurate",
        findings=[f"search-attested:{link}" for link in links]
        + ["search-cannot-verify:rate-limited"],
        links=links,
    )
    with pytest.raises(MetadataValidationError, match="cannot carry search-cannot-verify"):
        _validate_rung_and_search_attestation(facts, item)

    # every non-accurate R4 rung verdict must carry the typed marker -- both
    # cannot-verify and inaccurate (including inaccurate-for-found-code).
    for verdict in ("cannot-verify", "inaccurate"):
        facts, item = _r4_probe(verdict=verdict, findings=[], links=links)
        with pytest.raises(MetadataValidationError, match="requires typed search-cannot-verify"):
            _validate_rung_and_search_attestation(facts, item)

        facts, item = _r4_probe(
            verdict=verdict,
            findings=["search-cannot-verify:usable code found at the first link"],
            links=links,
        )
        _validate_rung_and_search_attestation(facts, item)


@pytest.mark.smoke
def test_the_receipt_bound_attestation_route_is_the_machine_rule() -> None:
    """Typed disposition links attest by broker receipt, never by re-fetch.

    W25 made ``links_checked`` carry typed ``{url, disposition}`` entries for
    broker-proven non-byte outcomes. Before this rule those entries were
    mechanically un-attestable (the old set arithmetic stringified the dict),
    so every honest R4 that reported one was forced non-accurate -- the death
    loop's second half. The receipt route completes honest coverage without
    weakening anything: re-fetchable links still demand ``search-attested:``,
    receipt-bound links demand ``search-receipt-attested:``, and the routes
    never substitute for each other in either direction.
    """

    plain = "https://example.org/search-a"
    typed = {"url": "https://example.org/policy-blocked", "disposition": "unfetchable-by-policy"}
    links: list[object] = [plain, typed]
    complete = [
        f"search-attested:{plain}",
        f"search-receipt-attested:{typed['url']}",
    ]

    # accurate + route-correct complete attestation: passes.
    facts, item = _r4_probe(verdict="accurate", findings=complete, links=links)
    _validate_rung_and_search_attestation(facts, item)

    # accurate with the typed link unattested: refused.
    facts, item = _r4_probe(verdict="accurate", findings=[f"search-attested:{plain}"], links=links)
    with pytest.raises(MetadataValidationError, match="lacks receipt-bound search attestations"):
        _validate_rung_and_search_attestation(facts, item)

    # a re-fetch marker on the receipt-bound link claims bytes the broker
    # never yielded: refused whatever the verdict.
    facts, item = _r4_probe(
        verdict="accurate",
        findings=[f"search-attested:{plain}", f"search-attested:{typed['url']}"],
        links=links,
    )
    with pytest.raises(MetadataValidationError, match="wrong route for its link class"):
        _validate_rung_and_search_attestation(facts, item)

    # a receipt marker on the plain link would let a checker skip the
    # re-execution: refused even beside a cannot-verify marker.
    facts, item = _r4_probe(
        verdict="cannot-verify",
        findings=[
            f"search-receipt-attested:{plain}",
            "search-cannot-verify:route confusion",
        ],
        links=links,
    )
    with pytest.raises(MetadataValidationError, match="wrong route for its link class"):
        _validate_rung_and_search_attestation(facts, item)

    # a malformed links_checked entry fails closed rather than silently
    # dropping out of the required-attestation set.
    facts, item = _r4_probe(
        verdict="accurate",
        findings=complete,
        links=[plain, {"url": typed["url"], "disposition": "unfetchable-by-policy", "extra": 1}],
    )
    with pytest.raises(MetadataValidationError, match="links_checked entry is malformed"):
        _validate_rung_and_search_attestation(facts, item)
