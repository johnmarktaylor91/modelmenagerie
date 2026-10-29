"""Schema-vs-constants failure-reason vocabulary cross-pin (DX-15 / RE-17).

The failure-reason vocabulary lives in TWO authorities that must never drift:
``attempt-common.schema.json`` (what a record may carry) and
``constants.FAILURE_REASON_CODES`` (what the reducer accepts per stage). Both
prior incidents were one-sided registrations:

* legs 31-33: a python-side reason missing from the schema crashed the driver
  (fixed ``d899bf8d``);
* BL-12: schema-side reasons missing from ``FAILURE_REASON_CODES`` made the
  reducer refuse every worker attempt carrying them, quarantining the model as
  a machinery fault on every leg (m12646/m13258/m13279).

This census pins the two authorities EQUAL, in both directions, for every
pair, so the next one-sided registration fails in the same change that
introduces it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from menagerie.crawler.constants import FAILURE_REASON_CODES

_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "schemas" / "attempt-common.schema.json"
)


def _schema_defs() -> dict[str, object]:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))["$defs"]


@pytest.mark.smoke
def test_stage_vocabulary_is_identical_in_schema_and_constants() -> None:
    """The schema stage enum and the FAILURE_REASON_CODES keys are one set."""

    schema_stages = set(_schema_defs()["stage"]["enum"])
    constant_stages = set(FAILURE_REASON_CODES)
    assert schema_stages - constant_stages == set(), (
        "stages only the schema knows (the reducer would refuse every attempt "
        f"carrying them): {sorted(schema_stages - constant_stages)}"
    )
    assert constant_stages - schema_stages == set(), (
        "stages only the constants know (the schema would make them "
        f"unrecordable): {sorted(constant_stages - schema_stages)}"
    )


@pytest.mark.smoke
def test_reason_vocabulary_is_identical_in_schema_and_constants() -> None:
    """The schema reason_code enum equals the union of per-stage reason sets.

    A reason present only in the schema is the BL-12 shape: the record layer
    accepts it, the reducer refuses it, and the model quarantines as a
    machinery fault every leg. A reason present only in the constants is the
    legs-31-33 shape: the classifier can mint it and the record layer makes it
    unrecordable. Both directions must be empty.
    """

    schema_reasons = set(_schema_defs()["reason_code"]["enum"])
    constant_reasons = set().union(*FAILURE_REASON_CODES.values())
    assert schema_reasons - constant_reasons == set(), (
        "reasons only the schema knows (reducer-refused if ever minted): "
        f"{sorted(schema_reasons - constant_reasons)}"
    )
    assert constant_reasons - schema_reasons == set(), (
        "reasons only the constants know (unrecordable if ever minted): "
        f"{sorted(constant_reasons - schema_reasons)}"
    )


@pytest.mark.smoke
def test_every_emittable_stage_reason_pair_is_reducer_appendable() -> None:
    """Every (stage, reason) literal the driver can emit is stage-valid.

    ``test_reducer_state_hygiene`` scans the emitting modules for literals;
    this pin closes the remaining gap for the one pair that scan missed
    (``runner``/``worker-unreapable``, driver_receipts.py D-state arm): the
    named pairs here are the host/machinery arms whose refusal at the reducer
    is precisely the quarantine-loop recurrence.
    """

    for stage, reason in (
        ("runner", "worker-unreapable"),
        ("runner", "internal-error"),
        ("policy", "sandbox-unavailable-v1"),
        ("import", "import-exception"),
    ):
        assert reason in FAILURE_REASON_CODES[stage], (
            f"driver-emittable pair ({stage!r}, {reason!r}) is not "
            "reducer-appendable: the first live emission quarantine-loops"
        )
