"""The mechanical excerpt-locator authority (rule-2 resolution, 2026-08-29).

The B1-XV2 matrix's flagship "contested" cell was a draw asserting 21
mechanically false line claims about frozen bytes (m10470 se2: the section
heading's line reported as the excerpt's line), and nothing in the pipeline
was looking -- ``_validate_locator`` checked ``bytes:`` locators positionally
but read every textual locator as whitespace-folded membership, so the only
positional adjudication anywhere was model eyeball arithmetic. These tests pin
the withdrawal of that authority from model judgment:

- the closed three-form locator grammar (fable A-2);
- strict 1-based inclusive universal-newline positional adjudication for
  line-form locators, zero tolerance, under the documented whitespace-run
  equivalence (Sol H-2.1) -- the m10470 mislocation fixture refuses and the
  m10759 CRLF-vs-LF fixture passes;
- fail-closed parse refusal (an ambiguous positional claim is never demoted
  to membership);
- the typed per-excerpt attestation rows the coordinator freezes into checker
  envelopes (valid / invalid / unattestable, disclosed rather than guessed);
- the era gate: the strict authority binds through the LIVE envelope's
  machine stamp only; the legacy default keeps membership semantics so
  append-only history is never reinterpreted (9 real mislocations exist in
  historical pilot artifacts and must keep rehydrating at their own basis).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from menagerie.crawler.evidence import (
    LOCATOR_ATTESTATION_INVALID,
    LOCATOR_ATTESTATION_UNATTESTABLE,
    LOCATOR_ATTESTATION_VALID,
    LOCATOR_AUTHORITY_LEGACY_MEMBERSHIP,
    LOCATOR_AUTHORITY_TEXT_LINE,
    LOCATOR_FORM_BYTES,
    LOCATOR_FORM_LINE,
    LOCATOR_FORM_MEMBERSHIP,
    TEXT_LINE_LOCATOR_BASIS,
    TEXT_LINE_LOCATOR_EQUIVALENCE,
    TEXT_LINE_LOCATOR_SCHEMA,
    EvidenceValidationError,
    _validate_locator,
    attest_excerpt_locators,
    classify_locator,
    excerpt_in_line_window,
    parse_line_locator,
    split_universal_lines,
)
from menagerie.crawler.identity import hash_bytes

#: The m10470 shape: a markdown doc whose section heading sits two lines above
#: the excerpt. se2 reported the heading's line (1 here, 19 in the frozen
#: bytes) instead of the declared content line (3 here, 21 there) -- wrong
#: under every numbering convention, refuted by the bytes.
_T5_DOC = b"# T5\n\n[T5](https://example.org/1910.10683) is a encoder-decoder transformer.\nMore prose.\n"


@pytest.mark.smoke
def test_locator_grammar_is_closed_by_enumeration() -> None:
    """Exactly three forms; only two of them carry a positional claim."""

    assert classify_locator("bytes:10-20") == LOCATOR_FORM_BYTES
    assert classify_locator("t5.md line 21") == LOCATOR_FORM_LINE
    assert classify_locator("modeling_t5.py lines 784-808") == LOCATOR_FORM_LINE
    assert classify_locator("line 60 (author affiliation block)") == LOCATOR_FORM_LINE
    # No digits after "line" = no positional claim = membership semantics,
    # which is CORRECT for anchor/element forms, not a gap (fable 2.3).
    assert classify_locator("the command line interface section") == LOCATOR_FORM_MEMBERSHIP
    assert classify_locator("meta tag og:title") == LOCATOR_FORM_MEMBERSHIP
    assert classify_locator("div#abstract") == LOCATOR_FORM_MEMBERSHIP


@pytest.mark.smoke
def test_line_locator_parse_is_fail_closed() -> None:
    """One claim parses; ambiguity and degeneracy refuse, never guess."""

    assert parse_line_locator("t5.md line 21") == (21, 21)
    assert parse_line_locator("jmlr.html lines 167-169") == (167, 169)
    assert parse_line_locator("line 60 (author affiliation block)") == (60, 60)
    # Two positional claims in one locator: ambiguous, refused.
    assert parse_line_locator("lines 12-13 and line 16") is None
    # Degenerate ranges refuse.
    assert parse_line_locator("line 0") is None
    assert parse_line_locator("lines 9-3") is None


@pytest.mark.smoke
def test_universal_newline_splitting_basis() -> None:
    """CRLF, CR, LF all split; final unterminated line counts; U+2028 never splits."""

    assert split_universal_lines(b"a\r\nb\rc\nd") == ["a", "b", "c", "d"]
    assert split_universal_lines(b"one\ntwo\n") == ["one", "two", ""]
    # BOM stays inside line 1; U+2028 LINE SEPARATOR is not a line boundary
    # under the declared basis (str.splitlines would split it -- that is why
    # the basis is spelled explicitly).
    lines = split_universal_lines("﻿x y\nz".encode("utf-8"))
    assert lines == ["﻿x y", "z"]


@pytest.mark.smoke
def test_strict_line_adjudication_refuses_the_m10470_mislocation() -> None:
    """The heading's line number is a locator failure with zero tolerance."""

    excerpt = b"[T5](https://example.org/1910.10683) is a encoder-decoder transformer."
    # Declared correctly (line 3): valid.
    assert excerpt_in_line_window(excerpt, _T5_DOC, 3, 3)
    _validate_locator(
        "ex-doc-family", "t5.md line 3", excerpt, _T5_DOC, authority=LOCATOR_AUTHORITY_TEXT_LINE
    )
    # Declared at the heading (line 1): refuted -- under EVERY convention.
    assert not excerpt_in_line_window(excerpt, _T5_DOC, 1, 1)
    with pytest.raises(EvidenceValidationError, match="not inside its declared line window"):
        _validate_locator(
            "ex-doc-family",
            "t5.md line 1",
            excerpt,
            _T5_DOC,
            authority=LOCATOR_AUTHORITY_TEXT_LINE,
        )
    # Adjacent-line slack does not exist: one line off is off.
    with pytest.raises(EvidenceValidationError, match="not inside its declared line window"):
        _validate_locator(
            "ex-doc-family",
            "t5.md line 2",
            excerpt,
            _T5_DOC,
            authority=LOCATOR_AUTHORITY_TEXT_LINE,
        )


@pytest.mark.smoke
def test_strict_line_adjudication_keeps_the_documented_equivalence() -> None:
    """m10759's CRLF/LF flavor difference is INSIDE the fold, never a failure.

    The terminal_disputed se2 draw silently imposed byte identity on a
    whitespace-codepoint difference the validator has always treated as
    equivalent; position and text equivalence are ONE authority (Sol 4.1).
    """

    source = b"torch>=2.1\r\nnumpy>=1.24\r\nrequests\r\n"
    excerpt = b"torch>=2.1\nnumpy>=1.24"
    assert excerpt_in_line_window(excerpt, source, 1, 2)
    _validate_locator(
        "ev-requirements-base",
        "requirements.txt lines 1-2",
        excerpt,
        source,
        authority=LOCATOR_AUTHORITY_TEXT_LINE,
    )
    # And the window is still exact: the same excerpt declared one line short
    # of its real extent is refused.
    assert not excerpt_in_line_window(excerpt, source, 1, 1)


@pytest.mark.smoke
def test_strict_authority_refuses_unparseable_positional_claims() -> None:
    """Ambiguity refuses outright; it is never demoted to membership."""

    with pytest.raises(EvidenceValidationError, match="ambiguous or degenerate line claim"):
        _validate_locator(
            "ev-x",
            "adapter.py lines 12-13 and line 16",
            b"return pit_xs_distilled(False)",
            b"import x\n\nreturn pit_xs_distilled(False)\n",
            authority=LOCATOR_AUTHORITY_TEXT_LINE,
        )
    # A range past the end of the file refutes the claim.
    with pytest.raises(EvidenceValidationError, match="not inside its declared line window"):
        _validate_locator(
            "ev-y", "f.md line 99", b"prose", b"prose\n", authority=LOCATOR_AUTHORITY_TEXT_LINE
        )


@pytest.mark.smoke
def test_legacy_authority_is_byte_identical_membership_semantics() -> None:
    """History is never reinterpreted: the default era ignores line numbers.

    Nine real mislocations exist in historical pilot author artifacts
    (m9678/m9653/m8635); their recorded results must keep rehydrating at the
    membership basis they were admitted under.
    """

    excerpt = b"[T5](https://example.org/1910.10683) is a encoder-decoder transformer."
    # Mislocated line claim, text present: legacy membership ACCEPTS.
    _validate_locator("ex-doc-family", "t5.md line 1", excerpt, _T5_DOC)
    _validate_locator(
        "ex-doc-family",
        "t5.md line 1",
        excerpt,
        _T5_DOC,
        authority=LOCATOR_AUTHORITY_LEGACY_MEMBERSHIP,
    )
    # Absent text still refuses in every era.
    with pytest.raises(EvidenceValidationError, match="not verbatim"):
        _validate_locator("ex-doc-family", "t5.md line 1", b"fabricated words", _T5_DOC)
    # bytes: locators were always strict and stay strict in every era.
    with pytest.raises(EvidenceValidationError, match="does not exist verbatim"):
        _validate_locator("ex-b", "bytes:0-4", b"WRONG", _T5_DOC)
    _validate_locator(
        "ex-b", "bytes:0-4", b"# T5", _T5_DOC, authority=LOCATOR_AUTHORITY_TEXT_LINE
    )


def _manifest_for(tmp_path: Path, name: str, content: bytes) -> dict:
    cas = tmp_path / f"{name}.bin"
    cas.write_bytes(content)
    return {
        "source_id": name,
        "content_sha256": hash_bytes(content),
        "cas_path": str(cas),
    }


@pytest.mark.smoke
def test_attestation_rows_are_typed_disclosed_and_closed(tmp_path: Path) -> None:
    """Valid, invalid, and unattestable rows all disclose; none ever guesses."""

    doc_row = _manifest_for(tmp_path, "doc", _T5_DOC)
    manifest = {"sources": [doc_row]}
    excerpt_text = "[T5](https://example.org/1910.10683) is a encoder-decoder transformer."
    excerpts = [
        {
            "evidence_id": "ex-ok",
            "source_id": "doc",
            "locator": "t5.md line 3",
            "text": excerpt_text,
        },
        {
            "evidence_id": "ex-mislocated",
            "source_id": "doc",
            "locator": "t5.md line 1",
            "text": excerpt_text,
        },
        {
            "evidence_id": "ex-membership",
            "source_id": "doc",
            "locator": "introduction paragraph",
            "text": "More prose.",
        },
        {
            "evidence_id": "ex-unknown-source",
            "source_id": "paper-abs",
            "locator": "line 12",
            "text": "anything",
        },
    ]
    rows = attest_excerpt_locators(excerpts, manifest)
    by_id = {row["evidence_id"]: row for row in rows}
    assert [row["schema"] for row in rows] == [TEXT_LINE_LOCATOR_SCHEMA] * 4

    ok = by_id["ex-ok"]
    assert ok["status"] == LOCATOR_ATTESTATION_VALID
    assert ok["form"] == LOCATOR_FORM_LINE
    assert (ok["start_line"], ok["end_line"]) == (3, 3)
    assert ok["basis"] == TEXT_LINE_LOCATOR_BASIS
    assert ok["equivalence"] == TEXT_LINE_LOCATOR_EQUIVALENCE
    assert ok["source_content_sha256"] == doc_row["content_sha256"]
    assert ok["excerpt_text_sha256"] == hash_bytes(excerpt_text.encode("utf-8"))

    bad = by_id["ex-mislocated"]
    assert bad["status"] == LOCATOR_ATTESTATION_INVALID
    assert "not inside declared line window 1-1" in bad["detail"]

    member = by_id["ex-membership"]
    assert member["status"] == LOCATOR_ATTESTATION_VALID
    assert member["form"] == LOCATOR_FORM_MEMBERSHIP
    assert member["basis"] is None

    unknown = by_id["ex-unknown-source"]
    assert unknown["status"] == LOCATOR_ATTESTATION_UNATTESTABLE
    assert "absent from the frozen manifest" in unknown["detail"]

    # Rows are pure JSON (they ride inside a hash-bound envelope).
    json.dumps(rows)


@pytest.mark.smoke
def test_attestation_discloses_missing_or_tampered_cas(tmp_path: Path) -> None:
    """Unreadable or hash-broken bytes are unattestable, never silently valid."""

    row = _manifest_for(tmp_path, "doc", _T5_DOC)
    Path(row["cas_path"]).write_bytes(b"tampered")
    rows = attest_excerpt_locators(
        [{"evidence_id": "ex", "source_id": "doc", "locator": "line 3", "text": "T5"}],
        {"sources": [row]},
    )
    assert rows[0]["status"] == LOCATOR_ATTESTATION_UNATTESTABLE
    assert "unavailable or no longer hash-bound" in rows[0]["detail"]


@pytest.mark.smoke
def test_checker_envelope_stamps_are_machine_owned(tmp_path: Path) -> None:
    """Attestation + inventory_mode stamp unconditionally; conflicts refuse."""

    from menagerie.crawler.checker_dispatch import (
        INVENTORY_MODE_DISCLOSED,
        INVENTORY_MODE_NOT_DISCLOSED,
        CheckerDispatchError,
        CheckerItemContractError,
        _derive_inventory_mode,
        _stamp_machine_item_fields,
    )

    assert (
        _derive_inventory_mode(
            {"sources": [{"source_kind": "routed-environment-inventory-v1"}]}
        )
        == INVENTORY_MODE_DISCLOSED
    )
    assert _derive_inventory_mode({"sources": [{"source_kind": "fetched"}]}) == (
        INVENTORY_MODE_NOT_DISCLOSED
    )
    assert _derive_inventory_mode(None) == INVENTORY_MODE_NOT_DISCLOSED

    doc_row = _manifest_for(tmp_path, "doc", _T5_DOC)
    manifest = {"sources": [doc_row]}
    excerpt = {
        "evidence_id": "ex-ok",
        "source_id": "doc",
        "locator": "t5.md line 3",
        "text": "[T5](https://example.org/1910.10683) is a encoder-decoder transformer.",
    }
    item: dict = {"stable_id": "m1"}
    _stamp_machine_item_fields(
        item,
        stable_id="m1",
        excerpts=[excerpt],
        source_manifest=manifest,
        inventory_mode=INVENTORY_MODE_NOT_DISCLOSED,
    )
    assert item["locator_attestation"]["authority"] == LOCATOR_AUTHORITY_TEXT_LINE
    assert item["locator_attestation"]["all_valid"] is True
    assert item["inventory_mode"] == INVENTORY_MODE_NOT_DISCLOSED

    # A caller-supplied conflicting stamp is refused, never overwritten.
    forged = {"stable_id": "m1", "inventory_mode": INVENTORY_MODE_DISCLOSED}
    with pytest.raises(CheckerDispatchError, match="machine-owned and cannot be authored"):
        _stamp_machine_item_fields(
            forged,
            stable_id="m1",
            excerpts=[excerpt],
            source_manifest=manifest,
            inventory_mode=INVENTORY_MODE_NOT_DISCLOSED,
        )
    forged_attestation = {"stable_id": "m1", "locator_attestation": {"authority": "x"}}
    with pytest.raises(CheckerDispatchError, match="machine-owned and cannot be authored"):
        _stamp_machine_item_fields(
            forged_attestation,
            stable_id="m1",
            excerpts=[excerpt],
            source_manifest=manifest,
            inventory_mode=None,
        )

    # A machine-refuted (mislocated) excerpt refuses the envelope pre-spend,
    # as an author-side pack defect.
    mislocated = dict(excerpt, locator="t5.md line 1")
    with pytest.raises(CheckerItemContractError, match="machine locator authority"):
        _stamp_machine_item_fields(
            {"stable_id": "m1"},
            stable_id="m1",
            excerpts=[mislocated],
            source_manifest=manifest,
            inventory_mode=None,
        )

    # An unattestable excerpt (unknown source) DISCLOSES and still builds:
    # adverse packs are exactly what the checker must adjudicate.
    drifted = dict(excerpt, evidence_id="ex-drift", source_id="paper-abs")
    disclosed: dict = {"stable_id": "m1"}
    _stamp_machine_item_fields(
        disclosed,
        stable_id="m1",
        excerpts=[excerpt, drifted],
        source_manifest=manifest,
        inventory_mode=None,
    )
    assert disclosed["locator_attestation"]["all_valid"] is False
    statuses = [row["status"] for row in disclosed["locator_attestation"]["rows"]]
    assert statuses == [LOCATOR_ATTESTATION_VALID, LOCATOR_ATTESTATION_UNATTESTABLE]


@pytest.mark.smoke
def test_checker_locator_claims_are_adjudicated_against_the_authority() -> None:
    """Disputing a fully machine-verified item is a typed checker defect."""

    from menagerie.crawler.checker_dispatch import (
        CheckerLocatorAuthorityError,
        _validate_item_locator_authority,
    )

    valid_row = {"status": LOCATOR_ATTESTATION_VALID}
    attested = {
        "locator_attestation": {
            "authority": LOCATOR_AUTHORITY_TEXT_LINE,
            "rows": [valid_row, dict(valid_row)],
            "all_valid": True,
        }
    }
    clean = {"integrity": {"locator_failures": [], "excerpt_discrepancies": []}}
    _validate_item_locator_authority(clean, attested, stable_id="m10470")

    # The m10470 se2 shape: line claims against machine-verified excerpts.
    fabricating = {
        "integrity": {
            "locator_failures": ["family excerpt starts at frozen line 19 rather than 21"],
            "excerpt_discrepancies": [],
        }
    }
    with pytest.raises(CheckerLocatorAuthorityError, match="machine-owned"):
        _validate_item_locator_authority(fabricating, attested, stable_id="m10470")

    # The m10759 terminal se2 shape: byte identity imposed on the documented
    # whitespace equivalence.
    crlf_disputing = {
        "integrity": {
            "locator_failures": [],
            "excerpt_discrepancies": ["ev-requirements-base uses LF where the source has CRLF"],
        }
    }
    with pytest.raises(CheckerLocatorAuthorityError, match="machine-owned"):
        _validate_item_locator_authority(crlf_disputing, attested, stable_id="m10759")

    # hash_mismatches stays checker-owned: it never trips this rule.
    hashes_only = {
        "integrity": {
            "locator_failures": [],
            "excerpt_discrepancies": [],
            "hash_mismatches": ["proposal sha256 mismatch"],
        }
    }
    _validate_item_locator_authority(hashes_only, attested, stable_id="m10470")

    # Unattestable rows license integrity findings (the component-drift shape).
    partially = {
        "locator_attestation": {
            "authority": LOCATOR_AUTHORITY_TEXT_LINE,
            "rows": [valid_row, {"status": LOCATOR_ATTESTATION_UNATTESTABLE}],
            "all_valid": False,
        }
    }
    _validate_item_locator_authority(fabricating, partially, stable_id="m2247")

    # Legacy envelopes (no attestation) keep historical semantics untouched.
    _validate_item_locator_authority(fabricating, {}, stable_id="legacy")


@pytest.mark.smoke
def test_live_author_envelope_stamps_the_strict_authority() -> None:
    """The dispatch entry stamps the era; the recorded entry omits it."""

    import inspect

    from menagerie.crawler import author_dispatch

    live = inspect.getsource(author_dispatch.build_author_envelope)
    assert "locator_authority=LOCATOR_AUTHORITY_TEXT_LINE" in live
    recorded = inspect.getsource(author_dispatch.build_recorded_result_envelope)
    assert "locator_authority" not in recorded
    assert LOCATOR_AUTHORITY_TEXT_LINE == "text-line-locator-v1"
