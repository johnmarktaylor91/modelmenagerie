"""A checked BLOCKED verdict terminalizes under its own honest reason code.

Pilot ``m1777`` (leg 7, 2026-08-20): the author returned a checked BLOCKED
recommendation at stage ``environment`` with reason ``missing-runtime-dependency``
(prerequisite ``runtime-dependency-minkowskiengine``), the terminal checker
adjudicated it ACCURATE end to end -- and the record terminalized as
``failed:author`` / ``malformed-result``, asserting an authoring defect that never
happened. ``driver._blocked_terminal`` is a TOTAL crossing from the author's open
checker-adjudicated reason vocabulary onto the closed record vocabulary, and its
only fallback collapsed every unrecordable-but-truthful reason onto that false
label.

Pinned here (updated for AL-6, JMT-signed 2026-08-24 -- the full routing-split
battery lives in ``test_blocked_prerequisite_terminals.py``):

* an off-vocabulary blocked reason at a recordable stage keeps the author's
  own named stage and NEVER collapses onto ``failed:author``; new mints spell
  the stage-suffixed ``blocked-prerequisite:<stage>`` member;
* the m1777 shape (runtime-dependency prerequisite) lands on the AL-6
  ``failed:environment-adjacent`` routing status with the blocking stage
  preserved in the reason suffix;
* an exactly recordable reason still maps verbatim (the crossing tightened,
  nothing widened);
* the bare legacy ``blocked-prerequisite`` member stays read-valid in every
  blocked-arm stage, in both the constants vocabulary and the attempt schema
  enum, so historical witness rows stay representable;
* the tier and access arms are untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from menagerie.crawler.author_dispatch import AuthorResultBinding, BlockedRecommendation
from menagerie.crawler.constants import (
    ACCESS_BLOCKED_REASON_CODE,
    ACCESS_BLOCKED_STATUS_CODE,
    BLOCKED_PREREQUISITE_REASON_CODE,
    FAILURE_REASON_CODES,
)
from menagerie.crawler.driver import CrawlerDriver
from menagerie.crawler.driver_contracts import DriverConfig
from menagerie.crawler.tests.conftest import HASH

pytestmark = pytest.mark.smoke

_CREATED_AT = "2026-08-20T19:01:17Z"

#: The closed stage enum of the author-result blocked arm (author-result-v4).
_BLOCKED_ARM_STAGES = ("source", "fetch", "author", "evidence", "environment", "policy", "runner")


def _blocked(
    stage: str,
    reason_code: str,
    prerequisite_ids: tuple[str, ...] = ("runtime-dependency-minkowskiengine",),
) -> BlockedRecommendation:
    """Build one minimal checked BLOCKED recommendation for the mapping seam."""

    raw_fields = {
        "result_id": "result-blocked",
        "result_sha256": HASH,
        "stable_id": "m1777",
        "work_id": "work-m1777",
        "campaign_id": "work-m1777",
        "author_identity": HASH,
        "prompt_identity": HASH,
        "dispatcher_identity": HASH,
        "source_manifest_identity": HASH,
        "intake_snapshot_id": "intake-1",
        "intake_snapshot_sha256": HASH,
        "intake_item_sha256": HASH,
        "created_at": _CREATED_AT,
    }
    binding = AuthorResultBinding(
        raw_result={**raw_fields, "kind": "BLOCKED", "payload": {}}, **raw_fields
    )
    return BlockedRecommendation(
        binding=binding,
        stage=stage,
        reason_code=reason_code,
        prerequisite_ids=prerequisite_ids,
        evidence_ids=("ev-blocked-1",),
        evidence_identity=HASH,
        license_identity=HASH,
        recommendation_sha256=HASH,
    )


def _map(
    stage: str,
    reason_code: str,
    prerequisite_ids: tuple[str, ...] = ("runtime-dependency-minkowskiengine",),
) -> tuple[str, str | None]:
    """Run the REAL total mapping with the default campaign config."""

    host = cast(CrawlerDriver, SimpleNamespace(config=DriverConfig()))
    return CrawlerDriver._blocked_terminal(host, _blocked(stage, reason_code, prerequisite_ids))


def test_m1777_shape_terminalizes_under_the_al6_adjacency_routing() -> None:
    """The live mislabel: stage environment, missing-runtime-dependency. AL-6
    routes the runtime-dependency prerequisite class onto the adjacency status
    with the blocking stage preserved in the reason suffix."""

    status_code, reason_code = _map("environment", "missing-runtime-dependency")
    assert status_code == "failed:environment-adjacent"
    assert reason_code == f"{BLOCKED_PREREQUISITE_REASON_CODE}:environment"


@pytest.mark.parametrize("stage", _BLOCKED_ARM_STAGES)
def test_every_blocked_arm_stage_keeps_its_own_naming(stage: str) -> None:
    """No off-vocabulary reason at any blocked-arm stage collapses onto author.

    A neutral (non-runtime-dep, non-CUDA) prerequisite pins the plain arm: the
    stage survives on the status code and the new-mint reason names it too.
    """

    status_code, reason_code = _map(
        stage, "some-reason-the-record-cannot-express", ("upstream-weights-release",)
    )
    assert status_code == f"failed:{stage}"
    assert reason_code == f"{BLOCKED_PREREQUISITE_REASON_CODE}:{stage}"
    # The pair is record-valid, so the witness attempt row is representable.
    assert reason_code in FAILURE_REASON_CODES[stage]


def test_exactly_recordable_reasons_still_map_verbatim() -> None:
    """The tightened fallback changes nothing for already-recordable reasons."""

    assert _map("source", "missing-material-source") == (
        "failed:source",
        "missing-material-source",
    )
    assert _map("environment", "solve-failed") == ("failed:environment", "solve-failed")
    assert _map("author", "needs-higher-tier") == ("failed:author", "needs-higher-tier")
    assert _map("source", ACCESS_BLOCKED_REASON_CODE) == (ACCESS_BLOCKED_STATUS_CODE, None)


def test_blocked_prerequisite_is_schema_representable_everywhere() -> None:
    """Vocabulary and schema stay in lockstep for every blocked-arm stage."""

    for stage in _BLOCKED_ARM_STAGES:
        assert BLOCKED_PREREQUISITE_REASON_CODE in FAILURE_REASON_CODES[stage], stage
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "attempt-common.schema.json"
    enum = json.loads(schema_path.read_text(encoding="utf-8"))["$defs"]["reason_code"]["enum"]
    assert BLOCKED_PREREQUISITE_REASON_CODE in enum
    # And of no stage OUTSIDE the blocked arm's schema enum: the member exists
    # exactly where a checked BLOCKED verdict can land, nowhere else.
    for stage, codes in FAILURE_REASON_CODES.items():
        if stage not in _BLOCKED_ARM_STAGES:
            assert BLOCKED_PREREQUISITE_REASON_CODE not in codes, stage
