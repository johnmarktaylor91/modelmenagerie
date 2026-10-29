"""Backend-parity gaps classify as machinery, never model verdicts (DX-14 / RE-15).

The worker refuses typed (``BackendProtocolUnimplementedError``) when a
framework's seed protocol, inference context, input conversion, or dtype
normalization is unimplemented. Pre-fix, those refusals classified as
``import/import-exception`` or ``forward/mode-run`` -- model verdicts for a
machinery capability gap (the m7960 shape; predicted to bite on the first
non-torch framework in 200b).

The honest terminal is ``deferred:needs-backend-parity``; its machine-derived
proof arm does not exist yet (reported to the plan), so the INTERIM
classification is machinery-owned -- routed to the park lane by
``NEVER_MODEL_TERMINAL_OWNERS``, with the intended disposition disclosed on
the attempt.
"""

from __future__ import annotations

from typing import Any

import pytest

from menagerie.crawler.backend import (
    BackendProtocolUnimplementedError,
    Framework,
    LaneStatus,
    framework_capability,
)
from menagerie.crawler.constants import BACKEND_PARITY_DEFERRAL_STATUS_CODE
from menagerie.crawler.driver_receipts import (
    NEVER_MODEL_TERMINAL_OWNERS,
    _failure_owner,
    _supervised_failure,
)
from menagerie.crawler.routing import backend_parity_deferral
from menagerie.crawler.tests.conftest import make_supervised_worker_result_v3
from menagerie.crawler.worker import _assert_award_comparable_dtypes
from menagerie.crawler.worker_supervisor import SupervisorObservation

pytestmark = pytest.mark.smoke

_GAP_TYPE = "menagerie.crawler.backend.BackendProtocolUnimplementedError"


def _clean_policy() -> dict[str, Any]:
    return {
        "network_attempted": False,
        "socket_targets": [],
        "checkpoint_or_weight_read_attempted": False,
        "checkpoint_paths": [],
        "write_outside_scratch_attempted": False,
        "write_paths": [],
        "credentials_present": False,
        "torchlens_import_attempted": False,
        "cache_read_attempted": False,
    }


def _failed_observation() -> SupervisorObservation:
    return SupervisorObservation(
        argv=("python", "-m", "menagerie.crawler.worker"),
        cwd="/tmp/worker",
        exit_code=1,
        signal_number=None,
        wall_seconds=1.0,
        cpu_seconds=0.5,
        peak_rss_bytes=1,
        timed_out=False,
        rss_exceeded=False,
        stdout_sha256="sha256:" + "0" * 64,
        stdout_bytes=0,
        stdout_tail="",
        stderr_sha256="sha256:" + "0" * 64,
        stderr_bytes=0,
        stderr_tail="",
        stdout_path="/tmp/worker/stdout",
        stderr_path="/tmp/worker/stderr",
    )


def _classified(receipt: dict[str, Any], mode_receipt: dict[str, Any]) -> dict[str, Any]:
    result = make_supervised_worker_result_v3(_failed_observation(), dict(receipt))
    return _supervised_failure(
        result,
        receipt,
        mode_receipt,
        _clean_policy(),
        worker_result_present=True,
    )


def _gap_error(message: str) -> dict[str, Any]:
    return {
        "reason_code": None,
        "exception_type": _GAP_TYPE,
        "message": message,
        "traceback": "Traceback ... BackendProtocolUnimplementedError",
        "errno": None,
    }


def test_seed_protocol_gap_is_machinery_owned_not_an_import_verdict() -> None:
    """The jax first-member shape: seed refusal before the constructor started."""

    receipt = {
        "constructor_started": False,
        "constructor_completed": False,
        "input_completed": False,
        "per_mode": {},
        "error": _gap_error("jax declares seed protocol 'jax.random.PRNGKey-explicit' ..."),
    }
    fields = _classified(receipt, {})
    assert (fields["stage"], fields["reason_code"]) == ("runner", "internal-error")
    assert fields["details"]["owner"] == "machinery"
    assert fields["details"]["backend_protocol_unimplemented"] is True
    assert fields["details"]["intended_disposition"] == BACKEND_PARITY_DEFERRAL_STATUS_CODE
    assert _failure_owner(fields) in NEVER_MODEL_TERMINAL_OWNERS


def test_mode_context_gap_is_machinery_owned_not_a_forward_verdict() -> None:
    """An inference-context/dtype-conformance refusal is not forward/mode-run."""

    mode_error = _gap_error("tensorflow declares inference context 'x' ...")
    receipt = {
        "constructor_started": True,
        "constructor_completed": True,
        "input_completed": True,
        "per_mode": {"eval": {"error": mode_error}},
        "error": None,
    }
    fields = _classified(receipt, {"error": mode_error})
    assert (fields["stage"], fields["reason_code"]) == ("runner", "internal-error")
    assert _failure_owner(fields) in NEVER_MODEL_TERMINAL_OWNERS


def test_ordinary_mode_failure_keeps_the_model_verdict() -> None:
    """Direction pin: a real forward explosion still charges the model."""

    mode_error = {
        "exception_type": "builtins.RuntimeError",
        "message": "mode forward exploded",
        "traceback": "Traceback ... RuntimeError",
    }
    receipt = {
        "constructor_started": True,
        "constructor_completed": True,
        "input_completed": True,
        "per_mode": {"eval": {"error": mode_error}},
        "error": None,
    }
    fields = _classified(receipt, {"error": mode_error})
    assert (fields["stage"], fields["reason_code"]) == ("forward", "mode-run")
    assert _failure_owner(fields) == "model"


def test_non_torch_award_requires_closed_vocabulary_dtypes() -> None:
    """DX-14 detail (award row 40): the normalizer's closed map gates non-torch."""

    unmappable = {
        "tree": {"leaf": 0},
        "leaves": [
            {"path": "output", "kind": "tensor", "dtype": "custom-quant-4x", "device": None}
        ],
    }
    mappable = {
        "tree": {"leaf": 0},
        "leaves": [{"path": "output", "kind": "tensor", "dtype": "float32", "device": None}],
    }
    with pytest.raises(BackendProtocolUnimplementedError, match="closed normalizer"):
        _assert_award_comparable_dtypes(unmappable, "jax")
    _assert_award_comparable_dtypes(mappable, "jax")
    # Torch and the NumPy materialization targets are exempt by design.
    _assert_award_comparable_dtypes(unmappable, "pytorch")
    _assert_award_comparable_dtypes(unmappable, "numpy")
    # Non-tensor leaves never participate.
    _assert_award_comparable_dtypes(
        {"tree": {"leaf": 0}, "leaves": [{"path": "output", "kind": "python", "dtype": None}]},
        "jax",
    )


def test_every_no_lane_framework_mints_the_parity_deferral() -> None:
    """M-15's pin over every non-EXECUTABLE Framework member, both directions."""

    for framework in Framework:
        capability = framework_capability(framework)
        if capability.lane_status is LaneStatus.EXECUTABLE:
            with pytest.raises(ValueError, match="fabricated proof"):
                backend_parity_deferral("m_pin", framework, detail="pin")
            continue
        assert capability.no_lane_reason
        decision = backend_parity_deferral("m_pin", framework, detail="pin")
        assert decision.status == BACKEND_PARITY_DEFERRAL_STATUS_CODE
