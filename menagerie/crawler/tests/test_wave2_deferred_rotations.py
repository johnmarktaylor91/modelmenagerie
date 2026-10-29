"""Wave-2 rotation-batch DEFERRED-ROTATION pins (batch assembly, AB-30 fold).

PLAN_v6 SS4.1 names four rotation-bearing items as Wave-2 members that were NOT
dispatched to any Wave-2 build lane and therefore did NOT merge in the sealed
rotation batch: W3's third-mode leg-config calibration enum, D-3's settlement
local-outcome widening, DX-63's credential-mode operational events, and DX-65's
source-drift records. The pre-batch mixed-era declaration registry
(``mixed_era/wave2_batch_declarations.json``) carried seam rows for them, built
from the Wave-1 design notes; the batch assembly removed those rows because the
rotations provably do not exist at the sealed tree -- a registry obligation with
no rotation to cover is not dischargeable and must not be faked.

These pins are the removal's tripwire: each asserts the NON-ROTATION at this
tree. Landing any of these items later is a closed-vocabulary / identity-axis
rotation and per the AT wave law must ride a NEW sealed rotation batch; that
change must delete its pin here AND add the seam row + mixed-era matrix to the
declaration registry in the same reviewed change. A pin going red outside such
a change means a rotation slipped in unmanaged -- exactly what W43 exists to
refuse.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest

from menagerie.crawler.checker_calibration import evaluate_checker_calibration_mode
from menagerie.crawler.constants import OperationalEventKind
from menagerie.crawler.paid_session_accounting import LOCAL_OUTCOMES

pytestmark = pytest.mark.smoke

_SCHEMAS_ROOT = Path(__file__).resolve().parents[1] / "schemas"


def test_w3_third_mode_calibration_enum_did_not_ride_this_batch() -> None:
    """W3's three-mode calibration enum is absent: the XOR arm is still two-mode.

    The W2 lane's ``evaluate_checker_calibration_mode`` docstring states the
    extension contract for W3's third mode (a shadow-rehearsal disjunct that is
    never receiptless). Until that lands in its own sealed rotation batch, the
    arm's signature carries no shadow-rehearsal input -- this pin forces the W3
    change to re-enter mixed-era registry review when it flips.
    """

    parameters = inspect.signature(evaluate_checker_calibration_mode).parameters
    assert not any("shadow" in name for name in parameters), (
        "W3's third-mode input appeared on the calibration XOR arm: the W3 "
        "rotation must ride a new sealed rotation batch with its own mixed-era "
        "registry row + matrix, deleting this pin in the same reviewed change"
    )


def test_d3_settlement_local_outcome_widening_did_not_ride_this_batch() -> None:
    """D-3's failed-stale-boot outcome is absent and unknown stays lawful.

    Prior-era settlement rows with ``local_outcome="unknown"`` (the 11 charged
    leg-start crash rows) remain lawful in every era; the D-3 widening must
    re-enter through a sealed rotation batch + registry row, deleting this pin.
    """

    assert "failed-stale-boot" not in LOCAL_OUTCOMES
    assert "unknown" in LOCAL_OUTCOMES, (
        "unknown must STAY a chargeable, lawful local outcome for the genuinely "
        "unanswerable -- prior-era rows depend on it"
    )


def test_dx63_credential_mode_events_did_not_ride_this_batch() -> None:
    """DX-63's credential-transition event kind is absent from the closed enum."""

    assert not any("credential" in kind.value for kind in OperationalEventKind), (
        "a credential-mode operational event kind appeared: DX-63 must ride a "
        "new sealed rotation batch with its own mixed-era registry row"
    )


def test_dx65_source_drift_records_did_not_ride_this_batch() -> None:
    """DX-65's per-(member, source_id) drift record class is unborn at this tree."""

    assert importlib.util.find_spec("menagerie.crawler.source_drift") is None
    drift_schemas = [
        path.name for path in _SCHEMAS_ROOT.glob("*.json") if "drift" in path.name
    ]
    assert not drift_schemas, (
        f"source-drift schemas appeared ({drift_schemas}): DX-65 must ride a new "
        "sealed rotation batch with its own mixed-era registry row"
    )
