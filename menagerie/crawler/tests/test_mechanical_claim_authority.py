"""W-8: production mechanical refutation at the write door (2026-08-29).

The rule-2 evidence: the one root-caused acceptance flip was a draw asserting
21 mechanically false claims about frozen bytes, and it MASS-REJECTED a clean
pack -- a bad reject is invisible because it looks exactly like the gate
working. The locator domain got its machine authority in the rule-2 build;
these tests pin the write-door extension to the remaining machine-decidable
claim classes (the closed registry on ``CheckerMechanicalClaimError``):

- ``hash-claim-binding``: every hash-mismatch entry is the closed structured
  claim ``{"digest_key", "detail"?}`` whose ``digest_key`` is EXACTLY a digest
  the envelope's machine-frozen ``verified_hashes`` binds (P5-SOL-06: the
  prose-era substring test admitted incidental-bound-key and superstring
  laundering; semantic mismatch adjudication over staged bytes stays
  checker-owned);
- ``locator-citation-license``: with non-valid attestation rows disclosed,
  every locator/excerpt entry is the closed structured claim
  ``{"evidence_id", "status", "claim_kind", "detail"?}`` alleging its OWN
  present non-valid row with the row's exact status and the array-matched
  claim_kind (P5-SOL-06: the prose-era rule validated only the cited license,
  so a false allegation about a VALID row passed by citing an unrelated
  non-valid row);
- ``r4-search-marker-correlation``: search-attestation markers must correlate
  with the frozen ``links_checked`` closed set, route-matched, and an accurate
  R4 rung check must attest EVERY link (the negative proof is over exactly
  that closed set);
- the era gate: everything binds through the LIVE envelope's machine
  attestation stamp, which recorded envelopes never carry;
- the driver routing: a refuted claim parks as ``checker-verdict-conflict``
  (never a member verdict, never a re-roll, never a model-blaming terminal).
"""

from __future__ import annotations

import inspect
import json

import pytest

from menagerie.crawler.checker_dispatch import (
    CheckerDeviationAuthorityError,
    _validate_item_deviation_authority,
    CheckerLocatorAuthorityError,
    CheckerMechanicalClaimError,
    _validate_item_mechanical_claims,
    validate_checker_result_mapping,
)
from menagerie.crawler.evidence import (
    LOCATOR_ATTESTATION_UNATTESTABLE,
    LOCATOR_ATTESTATION_VALID,
    LOCATOR_AUTHORITY_TEXT_LINE,
)

pytestmark = pytest.mark.smoke


def _attestation(rows: list[dict]) -> dict:
    return {
        "authority": LOCATOR_AUTHORITY_TEXT_LINE,
        "rows": rows,
        "all_valid": all(row.get("status") == LOCATOR_ATTESTATION_VALID for row in rows),
    }


def _envelope_item(**overrides: object) -> dict:
    item = {
        "locator_attestation": _attestation(
            [{"evidence_id": "ev-readme", "status": LOCATOR_ATTESTATION_VALID}]
        ),
        "verified_hashes": {
            "proposal": "a" * 64,
            "source_manifest": "b" * 64,
            "evidence": "c" * 64,
            "code": None,
            "source_to_code_map": "d" * 64,
            "family_template": None,
        },
        "proposal": {
            "proposed_facts": {
                "source_resolution": {
                    "rung": "R4_REIMPLEMENT",
                    "search_report": {
                        "links_checked": [
                            "https://example.org/a",
                            {
                                "url": "https://example.org/b",
                                "disposition": "unfetchable-by-policy",
                            },
                        ]
                    },
                }
            }
        },
    }
    item.update(overrides)
    return item


def _result_item(**overrides: object) -> dict:
    item: dict = {
        "integrity": {
            "verdict": "accurate",
            "hash_mismatches": [],
            "locator_failures": [],
            "excerpt_discrepancies": [],
        },
        "rung_check": {
            "verdict": "accurate",
            "findings": [
                "search-attested:https://example.org/a re-executed, no usable code",
                "search-receipt-attested:https://example.org/b broker receipt matches",
            ],
        },
    }
    item.update(overrides)
    return item


def _hash_claim(digest_key: str, detail: str | None = None) -> str:
    """One wire-form (JSON-serialized) structured hash-mismatch claim."""

    claim: dict = {"digest_key": digest_key}
    if detail is not None:
        claim["detail"] = detail
    return json.dumps(claim)


def _locator_claim(evidence_id: str, status: str, claim_kind: str, **extra: object) -> str:
    """One wire-form (JSON-serialized) structured locator/excerpt claim."""

    return json.dumps(
        {"evidence_id": evidence_id, "status": status, "claim_kind": claim_kind, **extra}
    )


# ---------------------------------------------------------------------------
# Era gate: recorded envelopes (no machine attestation) keep legacy semantics.
# ---------------------------------------------------------------------------


def test_legacy_envelope_is_never_reinterpreted() -> None:
    wild = _result_item(
        integrity={
            "verdict": "inaccurate",
            "hash_mismatches": ["a digest the pack never bound"],
            "locator_failures": ["uncited line claim"],
            "excerpt_discrepancies": [],
        },
        rung_check={"verdict": "accurate", "findings": ["search-attested:https://nowhere"]},
    )
    _validate_item_mechanical_claims(wild, {}, stable_id="legacy")
    _validate_item_mechanical_claims(
        wild, {"verified_hashes": {"proposal": "a" * 64}}, stable_id="legacy"
    )


# ---------------------------------------------------------------------------
# hash-claim-binding
# ---------------------------------------------------------------------------


def _hash_mismatch_result(*entries: object) -> dict:
    return _result_item(
        integrity={
            "verdict": "inaccurate",
            "hash_mismatches": list(entries),
            "locator_failures": [],
            "excerpt_discrepancies": [],
        }
    )


def test_hash_claim_naming_a_bound_digest_stays_checker_owned() -> None:
    result = _hash_mismatch_result(
        _hash_claim("source_manifest", "digest does not match the staged bytes")
    )
    _validate_item_mechanical_claims(result, _envelope_item(), stable_id="m1")


def test_hash_claim_decoded_mapping_form_is_adjudicated_identically() -> None:
    """The closed object itself (pre-decoded) binds exactly like its wire form."""

    good = _hash_mismatch_result({"digest_key": "source_manifest"})
    _validate_item_mechanical_claims(good, _envelope_item(), stable_id="m1")
    bad = _hash_mismatch_result({"digest_key": "weights_checkpoint"})
    with pytest.raises(CheckerMechanicalClaimError, match="unbound digest"):
        _validate_item_mechanical_claims(bad, _envelope_item(), stable_id="m1")


def test_hash_claim_about_an_unbound_digest_is_refuted() -> None:
    result = _hash_mismatch_result(_hash_claim("weights_checkpoint", "digest differs"))
    with pytest.raises(CheckerMechanicalClaimError, match="unbound digest"):
        _validate_item_mechanical_claims(result, _envelope_item(), stable_id="m1")


def test_hash_claim_naming_only_a_null_bound_key_is_refuted() -> None:
    # "code" is bound to null in the envelope: a mismatch claim about a
    # digest the pack deliberately does not bind disputes nothing.
    result = _hash_mismatch_result(_hash_claim("code", "digest differs from the frozen tree"))
    with pytest.raises(CheckerMechanicalClaimError, match="unbound digest"):
        _validate_item_mechanical_claims(result, _envelope_item(), stable_id="m1")


def test_prose_hash_entry_with_an_incidental_bound_key_is_refuted() -> None:
    """The P5-SOL-06 probe: prose alleging an UNBOUND weights digest passed the
    prose-era substring test because its text contained the incidental bound
    key "evidence". Free prose is now a typed refusal regardless of content."""

    result = _hash_mismatch_result(
        "the weights checkpoint digest differs; see the evidence bundle"
    )
    with pytest.raises(CheckerMechanicalClaimError, match="closed structured claim object"):
        _validate_item_mechanical_claims(result, _envelope_item(), stable_id="m1")


def test_superstring_digest_key_is_refuted_as_non_exact() -> None:
    """The P5-SOL-06 probe: "not_source_manifest_weights" passed the prose-era
    substring test via bound key "source_manifest". Membership is now exact."""

    result = _hash_mismatch_result(_hash_claim("not_source_manifest_weights"))
    with pytest.raises(CheckerMechanicalClaimError, match="unbound digest"):
        _validate_item_mechanical_claims(result, _envelope_item(), stable_id="m1")


def test_hash_claim_with_unknown_extra_key_is_refuted() -> None:
    entry = json.dumps({"digest_key": "source_manifest", "severity": "high"})
    with pytest.raises(CheckerMechanicalClaimError, match="unknown key"):
        _validate_item_mechanical_claims(
            _hash_mismatch_result(entry), _envelope_item(), stable_id="m1"
        )


def test_hash_claim_missing_digest_key_is_refuted() -> None:
    entry = json.dumps({"detail": "a digest differs somewhere"})
    with pytest.raises(CheckerMechanicalClaimError, match="required string field 'digest_key'"):
        _validate_item_mechanical_claims(
            _hash_mismatch_result(entry), _envelope_item(), stable_id="m1"
        )


def test_hash_claim_json_non_object_is_refuted() -> None:
    for entry in (json.dumps(["digest_key", "source_manifest"]), json.dumps("source_manifest")):
        with pytest.raises(CheckerMechanicalClaimError, match="closed structured claim object"):
            _validate_item_mechanical_claims(
                _hash_mismatch_result(entry), _envelope_item(), stable_id="m1"
            )


def test_hash_claim_non_string_detail_is_refuted() -> None:
    entry = json.dumps({"digest_key": "source_manifest", "detail": 7})
    with pytest.raises(CheckerMechanicalClaimError, match="'detail'"):
        _validate_item_mechanical_claims(
            _hash_mismatch_result(entry), _envelope_item(), stable_id="m1"
        )


def test_mixed_hash_rows_refuse_on_the_prose_member() -> None:
    """One valid structured entry never launders a prose sibling through."""

    result = _hash_mismatch_result(
        _hash_claim("source_manifest", "digest differs"),
        "and the weights checkpoint also looks off",
    )
    with pytest.raises(CheckerMechanicalClaimError, match="closed structured claim object"):
        _validate_item_mechanical_claims(result, _envelope_item(), stable_id="m1")


# ---------------------------------------------------------------------------
# locator-citation-license (the open branch the all-valid authority leaves)
# ---------------------------------------------------------------------------


def _drifted_envelope() -> dict:
    return _envelope_item(
        locator_attestation=_attestation(
            [
                {"evidence_id": "ev-readme", "status": LOCATOR_ATTESTATION_VALID},
                {"evidence_id": "ev-drift", "status": LOCATOR_ATTESTATION_UNATTESTABLE},
            ]
        )
    )


def _locator_result(
    *, locator_failures: list[object] | None = None, excerpt_discrepancies: list[object] | None = None
) -> dict:
    return _result_item(
        integrity={
            "verdict": "inaccurate",
            "hash_mismatches": [],
            "locator_failures": list(locator_failures or []),
            "excerpt_discrepancies": list(excerpt_discrepancies or []),
        }
    )


def test_structured_allegation_of_a_non_valid_row_passes() -> None:
    result = _locator_result(
        excerpt_discrepancies=[
            _locator_claim(
                "ev-drift",
                LOCATOR_ATTESTATION_UNATTESTABLE,
                "excerpt-discrepancy",
                detail="cites a source the manifest never froze",
            )
        ]
    )
    _validate_item_mechanical_claims(result, _drifted_envelope(), stable_id="m2247")


def test_structured_allegation_passes_in_the_locator_failures_array_too() -> None:
    result = _locator_result(
        locator_failures=[
            _locator_claim("ev-drift", LOCATOR_ATTESTATION_UNATTESTABLE, "locator-failure")
        ]
    )
    _validate_item_mechanical_claims(result, _drifted_envelope(), stable_id="m2247")


def test_prose_locator_finding_is_refuted_when_rows_are_disclosed() -> None:
    result = _locator_result(
        locator_failures=["the family excerpt starts at line 19 rather than 21"]
    )
    with pytest.raises(CheckerMechanicalClaimError, match="closed structured claim object"):
        _validate_item_mechanical_claims(result, _drifted_envelope(), stable_id="m10470")


def test_mixed_locator_rows_refuse_on_the_prose_member() -> None:
    """One correctly structured allegation never launders a prose sibling."""

    result = _locator_result(
        locator_failures=[
            _locator_claim("ev-drift", LOCATOR_ATTESTATION_UNATTESTABLE, "locator-failure"),
            "and the readme excerpt is also off by two lines",
        ]
    )
    with pytest.raises(CheckerMechanicalClaimError, match="closed structured claim object"):
        _validate_item_mechanical_claims(result, _drifted_envelope(), stable_id="m10470")


def test_false_allegation_about_a_valid_row_is_refuted_despite_a_non_valid_sibling() -> None:
    """The P5-SOL-06 probe: a false ev-readme allegation used to pass by citing
    the unrelated non-valid ev-drift row. The alleged row IS the entry's own
    evidence_id, so the VALID ev-readme row refuses even while ev-drift is
    genuinely unattestable."""

    for status in (LOCATOR_ATTESTATION_UNATTESTABLE, "invalid", LOCATOR_ATTESTATION_VALID):
        result = _locator_result(
            locator_failures=[_locator_claim("ev-readme", status, "locator-failure")]
        )
        with pytest.raises(CheckerMechanicalClaimError, match="valid or does not record"):
            _validate_item_mechanical_claims(result, _drifted_envelope(), stable_id="m1")


def test_allegation_about_an_absent_row_is_refuted() -> None:
    result = _locator_result(
        excerpt_discrepancies=[
            _locator_claim("ev-ghost", LOCATOR_ATTESTATION_UNATTESTABLE, "excerpt-discrepancy")
        ]
    )
    with pytest.raises(CheckerMechanicalClaimError, match="valid or does not record"):
        _validate_item_mechanical_claims(result, _drifted_envelope(), stable_id="m1")


def test_allegation_with_the_wrong_status_is_refuted() -> None:
    result = _locator_result(
        locator_failures=[_locator_claim("ev-drift", "invalid", "locator-failure")]
    )
    with pytest.raises(CheckerMechanicalClaimError, match="contradicting machine"):
        _validate_item_mechanical_claims(result, _drifted_envelope(), stable_id="m1")


def test_claim_kind_must_match_the_array_the_entry_sits_in() -> None:
    swapped_in_failures = _locator_result(
        locator_failures=[
            _locator_claim("ev-drift", LOCATOR_ATTESTATION_UNATTESTABLE, "excerpt-discrepancy")
        ]
    )
    with pytest.raises(CheckerMechanicalClaimError, match="must carry exactly 'locator-failure'"):
        _validate_item_mechanical_claims(swapped_in_failures, _drifted_envelope(), stable_id="m1")
    swapped_in_discrepancies = _locator_result(
        excerpt_discrepancies=[
            _locator_claim("ev-drift", LOCATOR_ATTESTATION_UNATTESTABLE, "locator-failure")
        ]
    )
    with pytest.raises(
        CheckerMechanicalClaimError, match="must carry exactly 'excerpt-discrepancy'"
    ):
        _validate_item_mechanical_claims(
            swapped_in_discrepancies, _drifted_envelope(), stable_id="m1"
        )


def test_locator_claim_with_unknown_extra_key_is_refuted() -> None:
    result = _locator_result(
        locator_failures=[
            _locator_claim(
                "ev-drift", LOCATOR_ATTESTATION_UNATTESTABLE, "locator-failure", severity="high"
            )
        ]
    )
    with pytest.raises(CheckerMechanicalClaimError, match="unknown key"):
        _validate_item_mechanical_claims(result, _drifted_envelope(), stable_id="m1")


def test_locator_claim_missing_a_required_field_is_refuted() -> None:
    entry = json.dumps({"evidence_id": "ev-drift", "status": LOCATOR_ATTESTATION_UNATTESTABLE})
    result = _locator_result(locator_failures=[entry])
    with pytest.raises(CheckerMechanicalClaimError, match="required string field 'claim_kind'"):
        _validate_item_mechanical_claims(result, _drifted_envelope(), stable_id="m1")


def test_all_valid_attestation_leaves_citation_arm_to_the_locator_authority() -> None:
    # With every row valid the arrays are the locator authority's business;
    # this validator's citation arm must not double-adjudicate them.
    result = _result_item()
    _validate_item_mechanical_claims(result, _envelope_item(), stable_id="m1")


# ---------------------------------------------------------------------------
# r4-search-marker-correlation
# ---------------------------------------------------------------------------


def test_route_matched_full_coverage_accurate_r4_passes() -> None:
    _validate_item_mechanical_claims(_result_item(), _envelope_item(), stable_id="m1359")


def test_marker_outside_the_closed_set_is_refuted() -> None:
    result = _result_item(
        rung_check={
            "verdict": "accurate",
            "findings": [
                "search-attested:https://example.org/a ok",
                "search-receipt-attested:https://example.org/b ok",
                "search-attested:https://example.org/ghost also checked",
            ],
        }
    )
    with pytest.raises(CheckerMechanicalClaimError, match="not in the frozen"):
        _validate_item_mechanical_claims(result, _envelope_item(), stable_id="m1359")


def test_swapped_attestation_route_is_mechanically_refused() -> None:
    result = _result_item(
        rung_check={
            "verdict": "accurate",
            "findings": [
                "search-receipt-attested:https://example.org/a receipt says so",
                "search-receipt-attested:https://example.org/b ok",
            ],
        }
    )
    with pytest.raises(CheckerMechanicalClaimError, match="swapped"):
        _validate_item_mechanical_claims(result, _envelope_item(), stable_id="m1359")


def test_accurate_r4_over_a_spot_checked_subset_is_refuted() -> None:
    result = _result_item(
        rung_check={
            "verdict": "accurate",
            "findings": ["search-attested:https://example.org/a re-executed"],
        }
    )
    with pytest.raises(CheckerMechanicalClaimError, match="negative proof"):
        _validate_item_mechanical_claims(result, _envelope_item(), stable_id="m1359")


def test_non_accurate_r4_rung_check_needs_no_coverage() -> None:
    result = _result_item(
        rung_check={
            "verdict": "cannot-verify",
            "findings": ["search-attested:https://example.org/a re-executed; b unreachable"],
        }
    )
    _validate_item_mechanical_claims(result, _envelope_item(), stable_id="m1359")


def test_non_r4_rung_needs_no_coverage_but_markers_still_correlate() -> None:
    envelope = _envelope_item(
        proposal={
            "proposed_facts": {
                "source_resolution": {
                    "rung": "R1_LIBRARY",
                    "search_report": {"links_checked": ["https://example.org/a"]},
                }
            }
        }
    )
    _validate_item_mechanical_claims(
        _result_item(rung_check={"verdict": "accurate", "findings": ["no markers"]}),
        envelope,
        stable_id="m1",
    )
    bad = _result_item(
        rung_check={
            "verdict": "accurate",
            "findings": ["search-receipt-attested:https://example.org/a"],
        }
    )
    with pytest.raises(CheckerMechanicalClaimError, match="swapped"):
        _validate_item_mechanical_claims(bad, envelope, stable_id="m1")


def test_marker_with_no_frozen_search_report_is_refuted() -> None:
    envelope = _envelope_item(proposal={"proposed_facts": {"source_resolution": {}}})
    result = _result_item(
        rung_check={
            "verdict": "accurate",
            "findings": ["search-attested:https://example.org/a"],
        }
    )
    with pytest.raises(CheckerMechanicalClaimError, match="not in the frozen"):
        _validate_item_mechanical_claims(result, envelope, stable_id="m1")


# ---------------------------------------------------------------------------
# Routing: the typed hierarchy and the write-door wiring.
# ---------------------------------------------------------------------------


def test_locator_authority_error_is_a_mechanical_claim_error() -> None:
    """The driver's single ``except CheckerMechanicalClaimError`` arm covers both."""

    assert issubclass(CheckerLocatorAuthorityError, CheckerMechanicalClaimError)
    error = CheckerMechanicalClaimError("refuted", stable_ids=["m1", 2])
    assert error.stable_ids == ("m1", "2")


def test_write_door_invokes_the_mechanical_claim_validator() -> None:
    source = inspect.getsource(validate_checker_result_mapping)
    assert "_validate_item_mechanical_claims(result_item, expected" in source


def test_driver_arms_park_refuted_claims_as_verdict_conflict() -> None:
    """Every lane routes a machine-refuted claim to the AG-4 conflict park."""

    from menagerie.crawler import driver, driver_admission

    admission = inspect.getsource(driver_admission)
    front = inspect.getsource(driver)
    # metadata batch + fidelity dispatch + the H-5 confirmation buy (admission);
    # terminal front half + terminal confirmation (driver).
    for source, expected_arms in ((admission, 3), (front, 2)):
        arms = source.count("except CheckerMechanicalClaimError as exc:")
        assert arms == expected_arms
    for source in (admission, front):
        for chunk in source.split("except CheckerMechanicalClaimError as exc:")[1:]:
            head = chunk.split("except ", 1)[0]
            assert 'failure_class="checker-verdict-conflict"' in head
            assert 'failure_site="mechanical-refutation"' in head


@pytest.mark.smoke
def test_deviation_authority_refusal_is_a_mechanical_claim_defect() -> None:
    """Composition seam (2026-08-29 whole-stack review): the declaration-anchor
    dispute must travel W-8's typed hierarchy, so the production fidelity lane
    and the H-5 confirmation arm catch it as ``CheckerMechanicalClaimError``
    and park ``checker-verdict-conflict`` — never the generic re-askable
    checker-dispatch path."""

    assert issubclass(CheckerDeviationAuthorityError, CheckerMechanicalClaimError)
    result = _result_item(
        fidelity={"verdict": "match", "deviations": []},
    )
    envelope = _envelope_item(
        declared_deviation_basis={
            "schema_version": "declared-deviation-basis-v1",
            "any_declared": True,
            "fidelity_deviation_entries": 1,
            "source_unspecified_choices": [],
        }
    )
    with pytest.raises(CheckerMechanicalClaimError) as caught:
        _validate_item_deviation_authority(result, envelope, stable_id="m1")
    assert isinstance(caught.value, CheckerDeviationAuthorityError)
    assert caught.value.stable_ids == ("m1",)
