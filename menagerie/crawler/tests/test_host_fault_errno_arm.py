"""Errno-typed host-fault arm for pre-receipt worker aborts (DX-13 / RE-14).

A worker that aborts before observing anything writes the
``worker-aborted-before-receipt`` diagnostic; the classifier used to charge
every one of them to the MODEL as ``import/import-exception``. When the
abort's typed errno names OS resource weather (ENOSPC/EROFS/ENODEV/EIO),
thirty minutes of disk pressure would permanently burn whichever models were
scheduled in the window. The arm flips exactly the closed errno set to a
host-owned park route; model exceptions keep their ownership -- RED is pinned
in BOTH directions.
"""

from __future__ import annotations

import errno
from typing import Any

import pytest

from menagerie.crawler.driver_receipts import (
    NEVER_MODEL_TERMINAL_OWNERS,
    _failure_owner,
    _host_resource_abort_errno,
    _supervised_failure,
)
from menagerie.crawler.tests.conftest import make_supervised_worker_result_v3
from menagerie.crawler.worker import _aborted_before_receipt_diagnostic
from menagerie.crawler.worker_supervisor import SupervisorObservation

pytestmark = pytest.mark.smoke


def _clean_policy() -> dict[str, Any]:
    """Policy observation with every tripwire silent."""

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
    """Parent observation for a worker that exited nonzero without a signal."""

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


def _abort_receipt(error: dict[str, Any]) -> dict[str, Any]:
    """Receipt shape of the worker's aborted-before-receipt diagnostic."""

    return {
        "constructor_started": False,
        "constructor_completed": False,
        "input_completed": False,
        "per_mode": {},
        "error": error,
    }


def _classified(error: dict[str, Any]) -> dict[str, Any]:
    receipt = _abort_receipt(error)
    result = make_supervised_worker_result_v3(_failed_observation(), dict(receipt))
    return _supervised_failure(
        result,
        receipt,
        {},
        _clean_policy(),
        worker_result_present=True,
    )


@pytest.mark.parametrize("code", sorted({errno.ENOSPC, errno.EROFS, errno.ENODEV, errno.EIO}))
def test_os_resource_abort_parks_as_host(code: int) -> None:
    """RED direction one: an OS-resource errno abort is a HOST park, never a verdict."""

    fields = _classified(
        {
            "reason_code": "worker-aborted-before-receipt",
            "exception_type": "builtins.OSError",
            "message": f"[Errno {code}] resource fault while staging scratch",
            "traceback": "Traceback ... OSError",
            "errno": code,
        }
    )
    assert (fields["stage"], fields["reason_code"]) == ("runner", "internal-error")
    assert fields["details"]["owner"] == "host"
    assert fields["details"]["errno"] == code
    assert _failure_owner(fields) in NEVER_MODEL_TERMINAL_OWNERS


def test_model_import_abort_keeps_model_ownership() -> None:
    """RED direction two: a model ImportError abort still charges the model."""

    fields = _classified(
        {
            "reason_code": "worker-aborted-before-receipt",
            "exception_type": "builtins.ModuleNotFoundError",
            "message": "No module named 'special_ops'",
            "traceback": "Traceback ... ModuleNotFoundError",
            "errno": None,
        }
    )
    assert (fields["stage"], fields["reason_code"]) == ("import", "import-exception")
    assert _failure_owner(fields) == "model"


def test_non_resource_errno_keeps_model_ownership() -> None:
    """EPERM and friends stay model-owned: the closed set is exactly four errnos."""

    fields = _classified(
        {
            "reason_code": "worker-aborted-before-receipt",
            "exception_type": "builtins.PermissionError",
            "message": "[Errno 1] Operation not permitted: '/etc/shadow'",
            "traceback": "Traceback ... PermissionError",
            "errno": errno.EPERM,
        }
    )
    assert (fields["stage"], fields["reason_code"]) == ("import", "import-exception")
    assert _failure_owner(fields) == "model"


def test_arm_never_fires_outside_the_abort_diagnostic() -> None:
    """A post-constructor failure carrying an errno is NOT the abort shape."""

    assert (
        _host_resource_abort_errno({"reason_code": None, "errno": errno.ENOSPC, "message": "x"})
        is None
    )
    assert _host_resource_abort_errno(None) is None
    assert (
        _host_resource_abort_errno({"reason_code": "worker-aborted-before-receipt", "errno": "28"})
        is None
    )


class _FakeRequest:
    """Minimal WorkerRequest stand-in for the diagnostic builder."""

    stable_id = "m_test"
    source_identity = "sha256:" + "0" * 64
    recipe_revision = "unbound"
    execution_identity = "sha256:" + "1" * 64
    seed = 0
    input_seed = 0
    mode = None
    device = "cpu"
    framework = "pytorch"


def test_worker_diagnostic_records_the_typed_errno() -> None:
    """The worker-side diagnostic carries the abort's errno as an integer."""

    os_error = OSError(errno.ENOSPC, "No space left on device")
    diagnostic = _aborted_before_receipt_diagnostic(_FakeRequest(), os_error)
    assert diagnostic["error"]["errno"] == errno.ENOSPC
    plain = _aborted_before_receipt_diagnostic(_FakeRequest(), ValueError("model defect"))
    assert plain["error"]["errno"] is None
