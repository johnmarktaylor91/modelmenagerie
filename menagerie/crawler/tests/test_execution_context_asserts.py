"""In-worker execution-identity cheap asserts (DX-70 / UU-24).

``output_value_sha256`` is thread-count- and device-dependent. These asserts
refuse BEFORE a digest can mint under an unattested context:

* thread pin: torch's live intra-op thread count must equal the
  ``OMP_NUM_THREADS`` pin (the locked determinism gate) -- a drifted context is
  a HOST fact, classified machinery-side, never a model verdict;
* device pin (MPS-leakage arm, m11584 class): a recipe that self-selects
  ``mps``/``cuda`` against a pinned request refuses in the constructor phase
  as a MODEL defect.
"""

from __future__ import annotations

from typing import Any

import pytest

from menagerie.crawler.driver_receipts import (
    NEVER_MODEL_TERMINAL_OWNERS,
    _execution_context_violation_error,
    _failure_owner,
    _supervised_failure,
)
from menagerie.crawler.tests.conftest import make_supervised_worker_result_v3
from menagerie.crawler.worker import (
    ExecutionContextViolationError,
    _assert_model_device,
    _assert_thread_context,
)
from menagerie.crawler.worker_supervisor import SupervisorObservation

pytestmark = pytest.mark.smoke

_VIOLATION_TYPE = "menagerie.crawler.worker.ExecutionContextViolationError"


def test_thread_context_assert_refuses_a_drifted_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live thread count off the OMP pin refuses before any digest mints."""

    torch = pytest.importorskip("torch")
    live = torch.get_num_threads()
    monkeypatch.setenv("OMP_NUM_THREADS", str(live))
    _assert_thread_context("pytorch")
    monkeypatch.setenv("OMP_NUM_THREADS", str(live + 7))
    with pytest.raises(ExecutionContextViolationError, match="thread-count-dependent"):
        _assert_thread_context("pytorch")
    monkeypatch.setenv("OMP_NUM_THREADS", "not-a-number")
    with pytest.raises(ExecutionContextViolationError, match="unparseable"):
        _assert_thread_context("pytorch")
    monkeypatch.delenv("OMP_NUM_THREADS")
    _assert_thread_context("pytorch")  # unpinned contexts (unit tests) pass
    monkeypatch.setenv("OMP_NUM_THREADS", str(live + 7))
    _assert_thread_context("numpy")  # materialization targets are exempt


class _Request:
    """Minimal request stand-in for the device assert."""

    framework = "pytorch"
    device = "cpu"


def test_model_device_assert_refuses_self_selecting_recipes() -> None:
    """m11584 class: parameters off the pinned device refuse at the constructor."""

    torch = pytest.importorskip("torch")
    on_device = torch.nn.Linear(2, 2)
    _assert_model_device(on_device, _Request())
    off_device = torch.nn.Linear(2, 2).to("meta")
    with pytest.raises(ValueError, match="self-selected device"):
        _assert_model_device(off_device, _Request())
    # Non-module constructions and non-torch frameworks are out of scope.
    _assert_model_device(object(), _Request())

    class _JaxRequest:
        framework = "jax"
        device = "cpu"

    _assert_model_device(off_device, _JaxRequest())


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


def test_context_violation_classifies_as_host_never_model() -> None:
    """The parent routes the typed violation to the park lane."""

    receipt = {
        "constructor_started": False,
        "constructor_completed": False,
        "input_completed": False,
        "per_mode": {},
        "error": {
            "reason_code": None,
            "exception_type": _VIOLATION_TYPE,
            "message": "torch intra-op thread count is 8 but OMP_NUM_THREADS pins 1",
            "traceback": "Traceback ... ExecutionContextViolationError",
            "errno": None,
        },
    }
    result = make_supervised_worker_result_v3(_failed_observation(), dict(receipt))
    fields = _supervised_failure(result, receipt, {}, _clean_policy(), worker_result_present=True)
    assert (fields["stage"], fields["reason_code"]) == ("runner", "internal-error")
    assert fields["details"]["owner"] == "host"
    assert fields["details"]["execution_context_violation"] is True
    assert _failure_owner(fields) in NEVER_MODEL_TERMINAL_OWNERS
    assert _execution_context_violation_error(receipt, {}) is receipt["error"]
    assert _execution_context_violation_error({"error": None}, {}) is None
