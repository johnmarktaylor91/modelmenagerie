"""Terminal-owner census: every failure the worker area can mint names an owner.

BL-12(e) / catalog C-16: host-transient faults used to mint permanent model
terminals through three lanes, and telemetry failures wore the
``checkpoint-read`` label. The closed authority is
:data:`~menagerie.crawler.driver_receipts.WORKER_FAILURE_REASON_OWNERS`; this
census walks the classifier source for every reason literal the area can
publish and refuses:

* an OWNER-LESS reason (not in the map), and
* a host-caused reason that could reach a model terminal (host and machinery
  owners are in :data:`NEVER_MODEL_TERMINAL_OWNERS`, which the terminal site
  consults before ``_terminalize``).
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator

import pytest

from menagerie.crawler.driver_receipts import (
    NEVER_MODEL_TERMINAL_OWNERS,
    WORKER_FAILURE_REASON_OWNERS,
    _failure_owner,
)

_CRAWLER_ROOT = Path(__file__).resolve().parents[1]

#: Every actor an owner may name. A value outside this set is a vocabulary
#: error, not a new category.
_CLOSED_OWNER_VOCABULARY = frozenset({"model", "author", "checker", "host", "machinery"})


def _classifier_reason_literals() -> Iterator[tuple[str, int, str]]:
    """Yield every reason-code literal minted by the worker-area classifier.

    Walks ``driver_receipts._supervised_failure`` plus the forward-lane
    exception triage in ``_forward_and_reduce`` for string literals passed as
    the reason position of ``_attempt_error_fields`` and the
    ``(stage, reason)`` tuples assigned in the blanket handler.
    """

    path = _CRAWLER_ROOT / "driver_receipts.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_attempt_error_fields"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            yield ("_attempt_error_fields", node.lineno, node.args[1].value)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Tuple):
            targets = node.targets
            if (
                len(targets) == 1
                and isinstance(targets[0], ast.Tuple)
                and len(targets[0].elts) == 2
                and all(isinstance(elt, ast.Name) for elt in targets[0].elts)
                and [elt.id for elt in targets[0].elts] == ["stage", "reason"]  # type: ignore[attr-defined]
                and len(node.value.elts) == 2
                and isinstance(node.value.elts[1], ast.Constant)
                and isinstance(node.value.elts[1].value, str)
            ):
                yield ("stage-reason-assignment", node.lineno, node.value.elts[1].value)


@pytest.mark.smoke
def test_every_minted_reason_names_a_closed_owner() -> None:
    """CENSUS: an owner-less reason literal in the classifier is a recurrence.

    The mapping in the policy_reasons/classifier tables is scanned from
    source, so ADDING a new classification arm without extending the owner
    map fails here in the same change.
    """

    minted = {reason for _site, _line, reason in _classifier_reason_literals()}
    assert minted, "census found no classifier reason literals -- scanner broken"
    ownerless = sorted(
        reason for reason in minted if reason not in WORKER_FAILURE_REASON_OWNERS
    )
    assert ownerless == [], f"owner-less worker-area reasons: {ownerless}"


@pytest.mark.smoke
def test_owner_vocabulary_is_closed() -> None:
    """Every owner value is one of the five closed actors."""

    assert set(WORKER_FAILURE_REASON_OWNERS.values()) <= _CLOSED_OWNER_VOCABULARY
    assert NEVER_MODEL_TERMINAL_OWNERS <= _CLOSED_OWNER_VOCABULARY


@pytest.mark.smoke
def test_host_caused_reasons_can_never_read_as_model_terminals() -> None:
    """Every host-owned reason is in the never-terminalize set's jurisdiction."""

    host_reasons = sorted(
        reason
        for reason, owner in WORKER_FAILURE_REASON_OWNERS.items()
        if owner in NEVER_MODEL_TERMINAL_OWNERS
    )
    # The three lanes the synthesis names, plus the new typed host faults.
    for required in (
        "sandbox-unavailable-v1",
        "telemetry-integrity",
        "worker-unreapable",
        "attestation-mismatch",
        "unclassified-supervised-failure",
        "internal-error",
        "ledger-corruption",
    ):
        assert required in host_reasons, f"{required} must be park-routed, never a terminal"


@pytest.mark.smoke
def test_failure_owner_prefers_the_attempt_stamp_over_the_map() -> None:
    """A per-attempt owner stamp (infra triage) overrides the per-reason default."""

    stamped = {
        "reason_code": "internal-error",
        "details": {"owner": "host"},
    }
    assert _failure_owner(stamped) == "host"
    unstamped = {"reason_code": "internal-error", "details": {}}
    assert _failure_owner(unstamped) == "machinery"
    unknown = {"reason_code": "never-heard-of-it", "details": {}}
    assert _failure_owner(unknown) is None


@pytest.mark.smoke
def test_owner_map_matches_the_attempt_schema_vocabulary() -> None:
    """Every owned reason is representable in the attempt schema enum."""

    import json

    schema = json.loads(
        (_CRAWLER_ROOT / "schemas" / "attempt-common.schema.json").read_text(encoding="utf-8")
    )
    enum = set(schema["$defs"]["reason_code"]["enum"])
    unrepresentable = sorted(set(WORKER_FAILURE_REASON_OWNERS) - enum)
    assert unrepresentable == [], f"owned reasons missing from attempt schema: {unrepresentable}"
