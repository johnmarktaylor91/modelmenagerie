"""The machine weights-convention anchor (rule-2 round 4, the last cell).

The round-3 matrix's ONE surviving same-epoch disagreement was
fidelity_refusal/m3353 flipping cannot-verify <-> major-drift (2-1) on an
underdetermined contract question: is the catalog's UNIVERSAL weights-not-used
convention (pretrained loading disabled, random construction,
``licenses.weights.status`` ``not-used``) a nonmaterial declared choice on a
source row whose bundle is DEFINED BY a pretrained checkpoint (WAV2VEC2_BASE
fixes ``wav2vec2_fairseq_base_ls960.pth``), or a material sourced
contradiction? One draw charged the checkpoint as a sourced material item the
random-init recipe contradicts -- a reading that would reject EVERY R1 recipe
of every pretrained bundle in the roster. These tests pin the withdrawal of
that reading choice from model judgment:

- the closed derivation (``implementation.initialization.policy`` ``random``
  AND ``implementation.initialization.pretrained_disabled`` true AND
  ``licenses.weights.status`` ``not-used``; anything else fails closed);
- the stamp is machine-owned: coordinator-derived at the envelope boundary,
  fidelity items only, a caller-supplied conflicting value refused;
- the committed m3353 golden itself derives a declared convention (the real
  bytes that measured the flip), so the closure is grounded, not synthetic;
- the prompt clause keys on the stamp and is bounded both ways
  (checkpoint-fixed architectural facts stay adjudicable; an undeclared
  convention keeps historical semantics).
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from menagerie.crawler.checker_dispatch import (
    INITIALIZATION_POLICY_RANDOM,
    WEIGHTS_CONVENTION_AUTHORITY,
    WEIGHTS_LICENSE_NOT_USED,
    CheckerDispatchError,
    _derive_weights_convention_basis,
    _stamp_machine_item_fields,
)

_GOLDEN_FIDELITY_REFUSAL = (
    Path(__file__).resolve().parent / "golden_checker_transcripts" / "fidelity_refusal.json"
)


def _convention_facts() -> dict:
    """The catalog's three const declarations, minimally."""

    return {
        "implementation": {
            "initialization": {"policy": "random", "pretrained_disabled": True}
        },
        "licenses": {"weights": {"status": "not-used"}},
    }


@pytest.mark.smoke
def test_weights_convention_basis_is_the_closed_three_const_conjunction() -> None:
    """policy random AND pretrained_disabled true AND weights not-used; else false."""

    basis = _derive_weights_convention_basis(_convention_facts())
    assert basis == {
        "authority": WEIGHTS_CONVENTION_AUTHORITY,
        "initialization_policy": INITIALIZATION_POLICY_RANDOM,
        "pretrained_disabled": True,
        "weights_license_status": WEIGHTS_LICENSE_NOT_USED,
        "convention_declared": True,
    }

    # Each conjunct failing independently fails the anchor closed.
    other_policy = _convention_facts()
    other_policy["implementation"]["initialization"]["policy"] = "checkpoint"
    assert _derive_weights_convention_basis(other_policy)["convention_declared"] is False

    not_disabled = _convention_facts()
    not_disabled["implementation"]["initialization"]["pretrained_disabled"] = False
    assert _derive_weights_convention_basis(not_disabled)["convention_declared"] is False

    weights_used = _convention_facts()
    weights_used["licenses"]["weights"]["status"] = "mit"
    used = _derive_weights_convention_basis(weights_used)
    assert used["convention_declared"] is False
    assert used["weights_license_status"] == "mit"

    # Fail-closed over malformed/absent shapes: no convention is ever guessed.
    assert _derive_weights_convention_basis(None)["convention_declared"] is False
    assert _derive_weights_convention_basis({})["convention_declared"] is False
    assert (
        _derive_weights_convention_basis({"implementation": {"initialization": None}})[
            "convention_declared"
        ]
        is False
    )
    truthy_not_bool = _convention_facts()
    truthy_not_bool["implementation"]["initialization"]["pretrained_disabled"] = "yes"
    frozen = _derive_weights_convention_basis(truthy_not_bool)
    assert frozen["convention_declared"] is False
    assert frozen["pretrained_disabled"] is None


@pytest.mark.smoke
def test_the_committed_m3353_golden_declares_the_convention() -> None:
    """The real flip-measuring bytes: the WAV2VEC2_BASE pack's const fields.

    Under the round-3 epoch, one draw read the bundle's defining checkpoint
    (``wav2vec2_fairseq_base_ls960.pth``) as a sourced material item the
    random-init recipe contradicts (major-drift) while the others held the
    convention out of fidelity scope (cannot-verify on the revision gap). The
    machine anchor settles WHAT the pack declares: the committed golden's
    const fields declare the convention, so the checkpoint question is out of
    fidelity scope and the deterministic landing is the conservative
    cannot-verify the revision-bound rule already forces.
    """

    entry = json.loads(_GOLDEN_FIDELITY_REFUSAL.read_text(encoding="utf-8"))
    item = entry["request"]["items"][0]
    assert item["stable_id"] == "m3353"
    facts = item["proposal"]["proposed_facts"]
    initialization = facts["implementation"]["initialization"]
    assert initialization["policy"] == "random"
    assert initialization["pretrained_disabled"] is True
    assert facts["licenses"]["weights"]["status"] == "not-used"
    basis = _derive_weights_convention_basis(facts)
    assert basis["convention_declared"] is True
    assert basis["authority"] == WEIGHTS_CONVENTION_AUTHORITY


@pytest.mark.smoke
def test_weights_convention_stamp_is_machine_owned() -> None:
    """Stamping is unconditional on fidelity items; conflicts refuse."""

    basis = _derive_weights_convention_basis(_convention_facts())
    item = {"stable_id": "m1"}
    _stamp_machine_item_fields(
        item,
        stable_id="m1",
        excerpts=[],
        source_manifest={"sources": []},
        inventory_mode=None,
        weights_convention=basis,
    )
    assert item["weights_convention"] == basis

    forged = {"stable_id": "m1", "weights_convention": {"convention_declared": False}}
    with pytest.raises(CheckerDispatchError, match="machine-owned"):
        _stamp_machine_item_fields(
            forged,
            stable_id="m1",
            excerpts=[],
            source_manifest={"sources": []},
            inventory_mode=None,
            weights_convention=basis,
        )

    # Metadata/terminal items pass None and are never stamped.
    unstamped = {"stable_id": "m2"}
    _stamp_machine_item_fields(
        unstamped,
        stable_id="m2",
        excerpts=[],
        source_manifest={"sources": []},
        inventory_mode=None,
        weights_convention=None,
    )
    assert "weights_convention" not in unstamped


@pytest.mark.smoke
def test_envelope_builder_threads_the_stamp_to_fidelity_items_only() -> None:
    """The anchor is a fidelity-verdict question; other kinds never carry it."""

    from menagerie.crawler import checker_dispatch

    source = inspect.getsource(checker_dispatch._build_envelope)
    assert "_derive_weights_convention_basis(facts)" in source
    assert "if gate_kind is GateKind.FIDELITY" in source


@pytest.mark.smoke
def test_prompt_scopes_fidelity_off_the_weights_convention() -> None:
    """The consumption clause ships in the prompt bytes (round-4 epoch)."""

    prompt = (
        Path(__file__).resolve().parents[1] / "prompts" / "codex_accuracy_checker_v2.txt"
    ).read_text(encoding="utf-8")
    unwrapped = " ".join(prompt.split())
    assert (
        "ARCHITECTURE FIDELITY EXCLUDES THE CATALOG'S UNIVERSAL WEIGHTS CONVENTION, "
        "AND THE CONVENTION IS MACHINE-STAMPED" in unwrapped
    )
    assert "`weights_convention`" in unwrapped
    assert "a checkpoint is a WEIGHTS fact" in unwrapped
    assert (
        "never major-drift, never by itself minor-drift, and never a reason to abstain"
        in unwrapped
    )
    # Bounded both ways: checkpoint-fixed architecture stays adjudicable, and an
    # undeclared convention keeps historical semantics.
    assert (
        "architectural facts a checkpoint's own definition fixes (its dimensions, its "
        "layer count, its head count) remain adjudicable sourced items" in unwrapped
    )
    assert "sourced material initialization SCHEME the code contradicts remains adjudicable" in unwrapped
    assert (
        "When `convention_declared` is false, no exclusion applies and historical semantics "
        "are untouched." in unwrapped
    )
    # The three const declarations the stamp derives from stay pinned beside it.
    assert (
        '`implementation.initialization.policy` "random" with `pretrained_disabled` true, '
        'and `licenses.weights.status` "not-used"' in unwrapped
    )
