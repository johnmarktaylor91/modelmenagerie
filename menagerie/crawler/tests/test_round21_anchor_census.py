"""Anchor-occurs-exactly-once census for the round21 deliberate reversions.

On 2026-08-16 six of the 29 reversion cases (D01, D02, D08, D16, D22, D27) had
mutation anchors silently killed by ordinary source drift — D22's anchor was
deleted by a workflow repair with nothing watching — and the runner then died
on D01 with a traceback before producing any attestation. The reversion suite
is the proof that the release gates BITE; a reversion that can no longer be
applied is a gate whose teeth fell out without anyone noticing.

This census runs on every smoke pass: every textual anchor of every case must
occur EXACTLY ONCE in the source it patches, and every non-textual mutation
target must exist. Any source edit that moves an anchor must update the
reversion case in the same reviewed change.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from menagerie.crawler.tools import round21_reversions

_REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.smoke
def test_every_reversion_anchor_occurs_exactly_once() -> None:
    failures = round21_reversions.verify_anchor_census(_REPO_ROOT)
    assert failures == [], (
        "round21 reversion anchors drifted; update the reversion case(s) in "
        "this same change so the release proof keeps biting:\n- " + "\n- ".join(failures)
    )


@pytest.mark.smoke
def test_census_covers_every_registered_case() -> None:
    """The census walks the same case table the runner executes — all 29."""
    case_ids = [case.reversion_id for case in round21_reversions._cases()]
    assert case_ids == list(round21_reversions._REVERSIONS)
