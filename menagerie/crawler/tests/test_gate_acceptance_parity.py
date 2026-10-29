"""One metadata acceptance predicate, or the write door mints a wrong verdict.

The pilot's defect class 4 was gate-verdict derivation split across call sites
that then DISAGREED. The same split reappeared in the acceptance predicate: nine
production sites decide "is this metadata gate item accepted", and eight of them
require ``integrity.verdict == "accurate"``:

* ``authority._chargeable_metadata_rejection_count`` (the bounded repair census)
* ``authority._derive_gate_failure`` (the terminal-proof kernel's ``_accepted``)
* ``authority._terminal_gate`` (item-level portion)
* ``gate_reuse._item_accepted`` (the reuse eligibility tripwire)
* ``driver_models._metadata_item_accepted``
* ``driver_models`` family-representative usability
* ``driver`` accepted-metadata projection
* ``artifact_transactions`` write-door acceptance
* ``reducer`` canonical acceptance

The ninth is :func:`menagerie.crawler.gates.route_metadata_gate` -- the WRITE
DOOR, the one site whose ``canonical_write_allowed`` actually authorizes
publication -- and it accepted on ``verdict`` plus ``rung_check`` alone.

The shape that separates them is real and named by the codebase itself:
``constants.metadata_gate_failure_reason`` carries a dedicated arm for a
content-``accurate`` round whose integrity is disputed (``identity-mismatch``),
and ``checker_dispatch.metadata_complaint_fields`` counts a non-accurate
``integrity`` block as a first-class complaint key. Both describe an item the
write door was nonetheless publishing.

Two consequences, both of which these tests pin:

1. **Publication of evidence-disputed metadata.** An item whose claims read
   accurate but whose excerpt locators drifted or whose source hashes mismatched
   (the pilot ledger holds 28 locator failures, 9 hash mismatches and 3 excerpt
   discrepancies; PROMPT1 ranks the class 14th of 15) would be published, while
   every downstream authority scored the same row a rejection.
2. **Anti-substitution violation through reuse.** ``gate_reuse`` replays only
   NON-accepted priors, and a metadata replay appends a NEW marked row. Under
   the split predicate an integrity-disputed item was simultaneously replayable
   (reuse: not accepted) and publishable (routing: accepted), so reuse would
   mint a SECOND ledger row carrying accepted-item bytes -- precisely what
   ``resolve_exact_gate_item_membership`` refuses, and what ``gate_reuse``'s own
   docstring promises never to mint.

Nothing here weakens a check: the fix tightens the write door to agree with the
eight authorities that already required integrity, and every test below asserts
a REFUSAL that the pre-fix code granted.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import Any, Mapping

import pytest

from menagerie.crawler.authority import _metadata_complaint_fields
from menagerie.crawler.constants import (
    AccuracyVerdict,
    GateRoute,
    metadata_gate_failure_reason,
    metadata_item_accepted,
)
from menagerie.crawler.gate_reuse import _item_accepted
from menagerie.crawler.gates import route_metadata_gate
from menagerie.crawler.tests.conftest import make_gate

pytestmark = pytest.mark.smoke

#: The three closed accuracy verdicts, spelled as the ledger spells them.
VERDICTS = ("accurate", "cannot-verify", "inaccurate")


def _one_item_gate(
    *,
    verdict: str = "accurate",
    integrity_verdict: str = "accurate",
    rung_verdict: str = "accurate",
) -> dict[str, Any]:
    """Build a schema-valid one-item metadata gate with the given verdicts.

    Only the three component verdicts move; every other field stays exactly as
    the shared builder produces it, so a routing difference can only come from
    the verdicts under test.
    """

    gate = make_gate(["m_parity"])
    item = gate["items"][0]
    item["verdict"] = verdict
    item["integrity"]["verdict"] = integrity_verdict
    item["rung_check"]["verdict"] = rung_verdict
    if integrity_verdict != "accurate":
        # A disputed integrity block names what it disputed; an empty finding
        # list under a non-accurate verdict is a different (checker-contract)
        # defect and is not what this module is about.
        item["integrity"]["hash_mismatches"] = ["source-1:sha256 mismatch"]
    if rung_verdict != "accurate":
        item["rung_check"]["highest_applicable"] = "R3_PORT"
        item["rung_check"]["findings"] = ["selected rung is not the highest applicable"]
    if verdict != "accurate":
        # Keep the top-level verdict coherent with a scoped check, the way the
        # checker-dispatch max-severity lockstep requires of a live result.
        item["field_checks"][0]["verdict"] = verdict
        item["field_checks"][0]["required_repair"] = "ground this claim"
        item["required_repairs"] = ["ground this claim"]
    return gate


def _route(gate: Mapping[str, Any]) -> Any:
    """Route one single-item metadata gate with a fresh repair allowance."""

    return route_metadata_gate(gate, {}, max_repairs=2)[0]


# ---------------------------------------------------------------------------
# 1. The write door must refuse to publish evidence-disputed metadata.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("integrity_verdict", ("inaccurate", "cannot-verify"))
def test_routing_refuses_publication_when_integrity_is_disputed(
    integrity_verdict: str,
) -> None:
    """Content-accurate claims over disputed evidence never reach canonical write.

    The checker scored every claim accurate and the rung correct, but disputed
    the evidence itself. Eight downstream authorities call that a rejection; the
    write door must not be the one site that publishes it.
    """

    decision = _route(_one_item_gate(integrity_verdict=integrity_verdict))

    assert decision.canonical_write_allowed is False
    assert decision.route is GateRoute.REQUEUE_NEXT_BATCH
    assert decision.repair_count == 1


@pytest.mark.parametrize("integrity_verdict", ("inaccurate", "cannot-verify"))
def test_disputed_integrity_terminalizes_under_its_own_kernel_arm(
    integrity_verdict: str,
) -> None:
    """The reason kernel's integrity arm becomes reachable, not decorative.

    ``metadata_gate_failure_reason`` has carried an ``identity-mismatch`` arm for
    a content-accurate round with disputed integrity since the kernel was
    unified. While routing accepted that shape the arm was unreachable from the
    metadata lane; it must now be the honest reason a repeat offender terminals
    on.
    """

    gate = _one_item_gate(integrity_verdict=integrity_verdict)
    item = gate["items"][0]

    assert metadata_item_accepted(item) is False
    assert metadata_gate_failure_reason(item) == "identity-mismatch"
    # And the complaint is disclosed under the bare component name, so a repeat
    # of the same dispute is chargeable against the bounded repair cap.
    assert "integrity" in _metadata_complaint_fields(item)


def test_routing_still_publishes_a_fully_accepted_item() -> None:
    """The tightening is exact: an all-accurate item is unaffected.

    Guards against a vacuous suite -- if this failed, the refusals above would
    prove nothing but a broken builder.
    """

    decision = _route(_one_item_gate())

    assert decision.canonical_write_allowed is True
    assert decision.route is GateRoute.ACCEPT
    assert decision.verdict is AccuracyVerdict.ACCURATE


# ---------------------------------------------------------------------------
# 2. Total parity across every authority, over the closed verdict product.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("verdict", "integrity_verdict", "rung_verdict"),
    tuple(product(VERDICTS, VERDICTS, VERDICTS)),
)
def test_every_authority_agrees_on_acceptance_over_the_closed_product(
    verdict: str, integrity_verdict: str, rung_verdict: str
) -> None:
    """Routing, reuse eligibility, and the shared kernel must never disagree.

    Exhaustive over the 27 cells of (verdict x integrity x rung). A single
    disagreeing cell is a wrong verdict waiting for a checker to emit it: the
    write door and the reuse tripwire would be answering different questions
    about the same ledger bytes.
    """

    gate = _one_item_gate(
        verdict=verdict,
        integrity_verdict=integrity_verdict,
        rung_verdict=rung_verdict,
    )
    item = gate["items"][0]
    expected = (
        verdict == "accurate" and integrity_verdict == "accurate" and (rung_verdict == "accurate")
    )

    assert metadata_item_accepted(item) is expected
    assert _item_accepted(item, "metadata_batch") is expected
    assert _route(gate).canonical_write_allowed is expected
    # A rejected cell always has a reason and at least one disclosed complaint;
    # an accepted cell has neither.
    assert bool(_metadata_complaint_fields(item)) is not expected
    if not expected:
        assert metadata_gate_failure_reason(item)


# ---------------------------------------------------------------------------
# 3. The anti-substitution consequence: replayable XOR publishable.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("verdict", "integrity_verdict", "rung_verdict"),
    tuple(product(VERDICTS, VERDICTS, VERDICTS)),
)
def test_no_item_is_both_reuse_replayable_and_publishable(
    verdict: str, integrity_verdict: str, rung_verdict: str
) -> None:
    """An item may never be both replayable and publishable.

    ``gate_reuse`` replays a prior only when it is NOT accepted, and a metadata
    replay APPENDS a new marked row. If routing would publish that same item,
    the replay mints a second ledger row carrying accepted-item bytes, which
    ``resolve_exact_gate_item_membership`` refuses at publication -- the
    width-1 shape that stopped a campaign at close. The two predicates must
    therefore be exact complements, cell by cell.
    """

    gate = _one_item_gate(
        verdict=verdict,
        integrity_verdict=integrity_verdict,
        rung_verdict=rung_verdict,
    )
    replayable = not _item_accepted(deepcopy(gate["items"][0]), "metadata_batch")
    publishable = _route(gate).canonical_write_allowed

    assert not (replayable and publishable), (
        "reuse would mint a duplicate marked row for an item the write door "
        "publishes: exact-membership anti-substitution violation"
    )
