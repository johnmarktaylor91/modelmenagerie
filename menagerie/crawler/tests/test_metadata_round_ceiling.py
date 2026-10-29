"""BL-10 (b): union-of-disclosures chargeability + the absolute round ceiling.

Audit B-1 measured the walkable complaint-disclosure rule on the pilot ledger:
lineage m8650 ran 10 rounds and 9 rejections against a cap of 2, charging
exactly 1, because chargeability compared each rejection against only the
IMMEDIATELY PRECEDING round -- a checker alternating disjoint complaint sets
``{A}, {B}, {A}, {B}`` was never chargeable after round one. Two fixes, both
pinned here:

* chargeability compares against the UNION of every prior round's disclosures,
  so a claim grants a free round at most once per lineage, ever;
* an ABSOLUTE per-lineage round ceiling parks a still-unaccepted lineage as
  ``checker-disclosure-debt`` on the checker-lane valve (human review, never a
  model-blaming ``failed:*`` terminal), and the park is decided BEFORE another
  session is bought.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pytest

import menagerie.crawler.author_spend as author_spend_module
from menagerie.crawler.checker_dispatch import (
    chargeable_fidelity_rejections,
    chargeable_metadata_rejections,
    disclosed_metadata_complaint_union,
    metadata_rejection_chargeable,
)
from menagerie.crawler.constants import METADATA_GATE_ROUND_CEILING
from menagerie.crawler.driver_contracts import CheckerOutcome, DriverConfig
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.tests.test_checker_lane_failure_scoping import _valve_rows
from menagerie.crawler.tests.test_slice_f_driver import (
    ScriptedChecker,
    _driver,
    _paths,
    _snapshot,
)

pytestmark = pytest.mark.smoke


def _adverse_item(complaints: Sequence[str]) -> dict[str, Any]:
    """Build one rejected metadata item disclosing exactly ``complaints``.

    Integrity and rung_check stay accurate so the complaint set is exactly the
    named field checks; one accurate filler check keeps ``field_checks``
    non-empty when the round discloses nothing.
    """

    checks = [
        {"field": field, "verdict": "inaccurate", "reason": "wrong", "required_repair": "fix"}
        for field in complaints
    ]
    checks.append({"field": "filler", "verdict": "accurate", "reason": "ok"})
    return {
        "verdict": "inaccurate",
        "integrity": {"verdict": "accurate"},
        "rung_check": {"verdict": "accurate"},
        "field_checks": checks,
    }


def _accepted_item() -> dict[str, Any]:
    return {
        "verdict": "accurate",
        "integrity": {"verdict": "accurate"},
        "rung_check": {"verdict": "accurate"},
        "field_checks": [{"field": "filler", "verdict": "accurate", "reason": "ok"}],
    }


def _history(*items: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return [({"gate_id": f"gate-{index}"}, item) for index, item in enumerate(items, start=1)]


# ---------------------------------------------------------------------------
# The union rule: disjoint alternation is chargeable, fresh claims stay free.
# ---------------------------------------------------------------------------


def test_disjoint_alternation_is_chargeable_under_the_union_rule() -> None:
    """The m8650 walk -- {A},{B},{A},{B} -- charges 3, not 1.

    Round 1 is the lineage's first (always chargeable); round 2 discloses a
    fresh claim (free, first disclosure); rounds 3 and 4 re-raise claims the
    UNION already disclosed, so both charge. Under the pre-fix one-round rule
    only round 1 charged and the checker could spin forever.
    """

    history = _history(
        _adverse_item(["a"]),
        _adverse_item(["b"]),
        _adverse_item(["a"]),
        _adverse_item(["b"]),
    )
    assert len(chargeable_metadata_rejections(history)) == 3


def test_a_genuinely_fresh_claim_each_round_charges_only_round_one() -> None:
    """Late disclosure still grants the rounds the checker owes -- once per claim."""

    history = _history(_adverse_item(["a"]), _adverse_item(["b"]), _adverse_item(["c"]))
    assert len(chargeable_metadata_rejections(history)) == 1


def test_rejection_after_an_accepting_first_round_is_not_chargeable() -> None:
    """An accepted round disclosed nothing, so the follow-up re-raises nothing."""

    history = _history(_accepted_item(), _adverse_item(["a"]))
    assert len(chargeable_metadata_rejections(history)) == 0
    assert metadata_rejection_chargeable(_adverse_item(["a"]), frozenset()) is False
    assert metadata_rejection_chargeable(_adverse_item(["a"]), None) is True


def test_union_helper_distinguishes_no_history_from_empty_disclosure() -> None:
    assert disclosed_metadata_complaint_union([]) is None
    assert disclosed_metadata_complaint_union(_history(_accepted_item())) == frozenset()
    assert disclosed_metadata_complaint_union(
        _history(_adverse_item(["a"]), _adverse_item(["b"]))
    ) == frozenset({"a", "b"})


def test_fidelity_lane_shares_the_union_rule() -> None:
    """The fidelity twin cannot be walked by disjoint alternation either.

    Fidelity rejections alternate between the two rejecting components: a rung
    complaint, then a drift complaint, then each again. Under the one-round
    rule only round 1 charged; under the union rule rounds 3 and 4 re-raise
    already-disclosed complaints and charge.
    """

    def rung_rejected() -> dict[str, Any]:
        return {
            "verdict": "inaccurate",
            "integrity": {"verdict": "accurate"},
            "rung_check": {"verdict": "inaccurate"},
            "fidelity": {"verdict": "match"},
            "field_checks": [{"field": "filler", "verdict": "accurate", "reason": "ok"}],
        }

    def drift_rejected() -> dict[str, Any]:
        return {
            "verdict": "inaccurate",
            "integrity": {"verdict": "accurate"},
            "rung_check": {"verdict": "accurate"},
            "fidelity": {"verdict": "major-drift"},
            "field_checks": [{"field": "filler", "verdict": "accurate", "reason": "ok"}],
        }

    history = _history(rung_rejected(), drift_rejected(), rung_rejected(), drift_rejected())
    assert len(chargeable_fidelity_rejections(history)) == 3


# ---------------------------------------------------------------------------
# The absolute ceiling: parks as checker-disclosure-debt, never failed:*.
# ---------------------------------------------------------------------------


class ChurningChecker(ScriptedChecker):
    """Reject every round with a FRESH complaint set: never chargeable, ever.

    Models the m8650 shape the ceiling exists for -- a checker whose every
    rejection is a first disclosure, so the chargeable cap can never terminal
    the lineage and only the absolute ceiling bounds it.
    """

    def check_metadata(
        self, artifacts: Any, work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        outcome = super().check_metadata(artifacts, work_root, config)
        assert outcome.gate is not None
        for item in outcome.gate["items"]:
            item["verdict"] = "inaccurate"
            fresh = f"claim_{self.metadata_calls}"
            item["required_repairs"] = [f"repair {fresh}"]
            item["field_checks"][0]["field"] = fresh
            item["field_checks"][0]["verdict"] = "inaccurate"
            item["field_checks"][0]["reason"] = f"{fresh} is unsupported"
            item["field_checks"][0]["required_repair"] = f"repair {fresh}"
        return outcome


def test_churning_lineage_parks_as_disclosure_debt_at_the_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A never-chargeable churn is bounded by the ceiling and parks, blame-free.

    The checker discloses a brand-new complaint every round; before the ceiling
    it could hold the leg open forever (the pilot's m8650 ran 10 rounds this
    way). The lineage must stop at exactly ``METADATA_GATE_ROUND_CEILING``
    sessions, park on the checker-lane valve as ``checker-disclosure-debt``,
    and never mint a ``failed:*`` model record.
    """

    snapshot = _snapshot(tmp_path, count=1)
    monkeypatch.setattr(
        author_spend_module, "AUTHOR_SESSIONS_MAX", METADATA_GATE_ROUND_CEILING + 1
    )
    stable_id = snapshot.items[0].stable_id
    checker = ChurningChecker()
    result = _driver(tmp_path, snapshot, checker=checker).run()

    assert result.status == "complete-with-deferrals"
    assert checker.metadata_calls == METADATA_GATE_ROUND_CEILING, (
        "the ceiling must stop the lineage before buying another session"
    )
    rows = _valve_rows(tmp_path, snapshot)
    assert len(rows) == 1
    row = rows[0]
    assert row["stable_id"] == stable_id
    assert row["checker_lane"]["failure_class"] == "checker-disclosure-debt"
    # The trail records the batch sizes the rounds actually rode.
    assert row["checker_lane"]["attempted_batch_sizes"] == [1] * METADATA_GATE_ROUND_CEILING
    paths = _paths(tmp_path, snapshot)
    models = list(scan_jsonl(paths.ledgers.models)) if paths.ledgers.models.exists() else []
    assert all(not str(record["status"]["code"]).startswith("failed:") for record in models), (
        "disclosure debt must never be minted as a model-blaming terminal"
    )


def test_disclosure_debt_parks_are_exempt_from_the_lane_streak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Six ceiling parks are verdict-shaped facts, not lane weather.

    The park-streak breaker exists for a lane answering NOBODY; disclosure-debt
    parks each carry a full adjudication history, so a cluster of them must not
    abort the leg.
    """

    from menagerie.crawler.constants import CHECKER_LANE_PARK_STREAK_LIMIT

    count = CHECKER_LANE_PARK_STREAK_LIMIT + 1
    monkeypatch.setattr(
        author_spend_module, "AUTHOR_SESSIONS_MAX", METADATA_GATE_ROUND_CEILING + 1
    )
    snapshot = _snapshot(tmp_path, count=count)
    checker = ChurningChecker()
    result = _driver(tmp_path, snapshot, checker=checker).run()

    assert result.status == "complete-with-deferrals"
    rows = _valve_rows(tmp_path, snapshot)
    assert len(rows) == count
    assert {row["checker_lane"]["failure_class"] for row in rows} == {"checker-disclosure-debt"}


def test_ceiling_is_meaningfully_above_the_chargeable_cap() -> None:
    """The ceiling backs the cap up; it must never preempt honest repair rounds."""

    assert METADATA_GATE_ROUND_CEILING >= 1 + 2 + 1  # initial + max_repairs + margin
