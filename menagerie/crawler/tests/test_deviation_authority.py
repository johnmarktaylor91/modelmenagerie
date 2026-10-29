"""The machine match/minor-drift declaration anchor (rule-2 round 3, cell 1).

The post-closure matrix's surviving fidelity_accept cell was m10759 flipping
match <-> minor-drift (2-1) because the precedence clause's ANCHOR FACT --
"does the proposal declare a deviation?" -- was split across two pack
vocabularies: ``fidelity.deviations`` was ``[]`` while
``implementation.declared_choices`` carried author-chosen entries, and draws
chose per-read which field owned the declaration. These tests pin the
withdrawal of that reading choice from model judgment:

- the closed derivation (every ``fidelity.deviations`` row plus every
  ``declared_choices`` row whose ``source_status`` is author-chosen;
  source-specified rows restate the source and never count);
- the stamp is machine-owned: coordinator-derived at the envelope boundary,
  fidelity items only, a caller-supplied conflicting value refused;
- the adjudication: ``match`` against a nonempty stamped anchor is a typed
  ``CheckerDeviationAuthorityError`` discarding the whole result, never a
  member verdict, while every non-match verdict stays the checker's judgment;
- the committed m10759 golden itself derives a nonempty anchor (the real
  bytes that measured the flip), so the closure is grounded, not synthetic.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from menagerie.crawler.checker_dispatch import (
    DECLARED_CHOICE_AUTHOR_CHOSEN,
    DEVIATION_BASIS_AUTHORITY,
    CheckerDeviationAuthorityError,
    CheckerDispatchError,
    _derive_declared_deviation_basis,
    _stamp_machine_item_fields,
    _validate_item_deviation_authority,
)

_GOLDEN_FIDELITY_ACCEPT = (
    Path(__file__).resolve().parent / "golden_checker_transcripts" / "fidelity_accept.json"
)


@pytest.mark.smoke
def test_declared_deviation_basis_is_the_closed_two_vocabulary_union() -> None:
    """fidelity.deviations rows + author-chosen declared_choices rows, nothing else."""

    facts = {
        "fidelity": {"deviations": []},
        "implementation": {
            "declared_choices": [
                {"field": "library_recipe.kwargs.verbose", "source_status": "author-chosen", "material": False},
                {"field": "library_recipe.kwargs.cfg", "source_status": "source-specified", "material": True},
            ]
        },
    }
    basis = _derive_declared_deviation_basis(facts)
    assert basis["authority"] == DEVIATION_BASIS_AUTHORITY
    assert basis["fidelity_deviation_entries"] == 0
    assert basis["any_declared"] is True
    assert basis["source_unspecified_choices"] == [
        {
            "field": "library_recipe.kwargs.verbose",
            "source_status": DECLARED_CHOICE_AUTHOR_CHOSEN,
            "author_material_flag": False,
        }
    ]

    # A fidelity.deviations row alone also sets the anchor.
    with_deviation = _derive_declared_deviation_basis(
        {"fidelity": {"deviations": ["stride replaced"]}, "implementation": {"declared_choices": []}}
    )
    assert with_deviation["fidelity_deviation_entries"] == 1
    assert with_deviation["any_declared"] is True

    # Nothing declared anywhere: the declaration half of match is satisfied.
    clean = _derive_declared_deviation_basis(
        {
            "fidelity": {"deviations": []},
            "implementation": {
                "declared_choices": [
                    {"field": "initialization", "source_status": "source-specified", "material": True}
                ]
            },
        }
    )
    assert clean["any_declared"] is False
    assert clean["source_unspecified_choices"] == []

    # Fail-closed over malformed shapes: no anchor is ever guessed.
    assert _derive_declared_deviation_basis(None)["any_declared"] is False
    assert _derive_declared_deviation_basis({})["any_declared"] is False


@pytest.mark.smoke
def test_the_committed_m10759_golden_derives_a_nonempty_anchor() -> None:
    """The real flip-measuring bytes: deviations [] beside author-chosen rows.

    Under the pre-closure prompt, reading ``fidelity.deviations: []`` as the
    whole declaration licensed match (the se3 draw) while reading
    ``declared_choices`` licensed minor-drift (se1/se2). The machine anchor
    settles it: the committed pack declares two author-chosen rows, so match
    is impossible and the deterministic landing is the acceptance-preserving
    minor-drift.
    """

    entry = json.loads(_GOLDEN_FIDELITY_ACCEPT.read_text(encoding="utf-8"))
    item = entry["request"]["items"][0]
    assert item["stable_id"] == "m10759"
    facts = item["proposal"]["proposed_facts"]
    assert facts["fidelity"]["deviations"] == []
    basis = _derive_declared_deviation_basis(facts)
    assert basis["any_declared"] is True
    assert basis["fidelity_deviation_entries"] == 0
    assert sorted(row["field"] for row in basis["source_unspecified_choices"]) == [
        "library_recipe.kwargs.verbose",
        "library_recipe.version",
    ]


@pytest.mark.smoke
def test_deviation_basis_stamp_is_machine_owned() -> None:
    """Stamping is unconditional on fidelity items; conflicts refuse."""

    facts = {
        "fidelity": {"deviations": []},
        "implementation": {
            "declared_choices": [
                {"field": "x", "source_status": "author-chosen", "material": False}
            ]
        },
    }
    basis = _derive_declared_deviation_basis(facts)
    item = {"stable_id": "m1"}
    _stamp_machine_item_fields(
        item,
        stable_id="m1",
        excerpts=[],
        source_manifest={"sources": []},
        inventory_mode=None,
        deviation_basis=basis,
    )
    assert item["declared_deviation_basis"] == basis

    forged = {"stable_id": "m1", "declared_deviation_basis": {"any_declared": False}}
    with pytest.raises(CheckerDispatchError, match="machine-owned"):
        _stamp_machine_item_fields(
            forged,
            stable_id="m1",
            excerpts=[],
            source_manifest={"sources": []},
            inventory_mode=None,
            deviation_basis=basis,
        )

    # Metadata/terminal items pass None and are never stamped.
    unstamped = {"stable_id": "m2"}
    _stamp_machine_item_fields(
        unstamped,
        stable_id="m2",
        excerpts=[],
        source_manifest={"sources": []},
        inventory_mode=None,
        deviation_basis=None,
    )
    assert "declared_deviation_basis" not in unstamped


@pytest.mark.smoke
def test_envelope_builder_threads_the_stamp_to_fidelity_items_only() -> None:
    """The anchor is a fidelity-verdict question; other kinds never carry it."""

    from menagerie.crawler import checker_dispatch

    source = inspect.getsource(checker_dispatch._build_envelope)
    assert "_derive_declared_deviation_basis(facts)" in source
    assert "if gate_kind is GateKind.FIDELITY" in source


@pytest.mark.smoke
def test_match_against_a_nonempty_anchor_is_a_typed_checker_defect() -> None:
    """The m10759 se3 shape refuses typed; every non-match verdict is untouched."""

    stamped = {
        "declared_deviation_basis": {
            "authority": DEVIATION_BASIS_AUTHORITY,
            "fidelity_deviation_entries": 0,
            "source_unspecified_choices": [
                {
                    "field": "library_recipe.kwargs.verbose",
                    "source_status": DECLARED_CHOICE_AUTHOR_CHOSEN,
                    "author_material_flag": False,
                }
            ],
            "any_declared": True,
        }
    }
    with pytest.raises(CheckerDeviationAuthorityError, match="machine-owned") as caught:
        _validate_item_deviation_authority(
            {"fidelity": {"verdict": "match"}}, stamped, stable_id="m10759"
        )
    assert caught.value.stable_ids == ("m10759",)

    # WHICH non-match verdict applies remains the checker's judgment.
    for verdict in ("minor-drift", "major-drift", "slop", "cannot-verify"):
        _validate_item_deviation_authority(
            {"fidelity": {"verdict": verdict}}, stamped, stable_id="m10759"
        )

    # An empty anchor keeps match available.
    empty = {
        "declared_deviation_basis": {
            "authority": DEVIATION_BASIS_AUTHORITY,
            "fidelity_deviation_entries": 0,
            "source_unspecified_choices": [],
            "any_declared": False,
        }
    }
    _validate_item_deviation_authority({"fidelity": {"verdict": "match"}}, empty, stable_id="m1")

    # Legacy envelopes (no stamp) keep historical semantics untouched.
    _validate_item_deviation_authority({"fidelity": {"verdict": "match"}}, {}, stable_id="old")

    # Metadata results carry no authored fidelity block; the rule is inert.
    _validate_item_deviation_authority({}, stamped, stable_id="m3")


@pytest.mark.smoke
def test_prompt_keys_the_declaration_anchor_on_the_machine_stamp() -> None:
    """The consumption clause ships in the prompt bytes (round-3 epoch)."""

    prompt = (
        Path(__file__).resolve().parents[1] / "prompts" / "codex_accuracy_checker_v2.txt"
    ).read_text(encoding="utf-8")
    unwrapped = " ".join(prompt.split())
    assert "THE DECLARATION ANCHOR IS MACHINE-STAMPED" in unwrapped
    assert "`declared_deviation_basis`" in unwrapped
    assert (
        "every `fidelity.deviations` row plus every `implementation.declared_choices` row "
        "whose source_status is author-chosen" in unwrapped
    )
    assert "never decide which pack field \"owns\" deviation declarations" in unwrapped
    assert "match is impossible and a match verdict disputes machine arithmetic" in unwrapped
    assert "the AUTHOR'S build-disclosure flag, never your fidelity judgment" in unwrapped
    # The closed precedence the stamp anchors stays pinned beside it.
    assert "the proposal declares NO deviation of any kind" in unwrapped
