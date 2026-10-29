"""RED pin: a worker's own designed failure shape is never attestation weather.

Incident (2026-08-23, step-200 legs 31-36): m12646/m13258/m13279 fail their
forwards deterministically (``IndexError: index out of range in self``), so
``worker.main`` exits ``1`` with a complete failure receipt and -- by the
success-only v3 protocol -- no raw award receipt and no completion file. The
supervisor minted ``missing-or-mismatched-v3-attestation`` plus a
``protocol-violation`` partial for exactly that designed shape, the driver
published ``runner/attestation-mismatch`` (machinery weather, no model
verdict), and the scheduler re-quarantined the same three models every leg.
The stderr OpenMP lines (``reclaimed provably stale LLVM OpenMP registration
file``, ``OMP: Warning #179``) appear on successful award workers too and were
a red herring.

These tests pin the closed ``_worker_declared_failure_v3`` shape and its
wiring: the worker-declared-failure lane must reach the driver's
receipt-evidence arms (``forward/mode-run``, model-owned) while every other
raw-receipt-less shape keeps the fail-closed attestation-mismatch path.
"""

from __future__ import annotations

import inspect
from typing import Any, Optional

import pytest

from menagerie.crawler import worker_supervisor
from menagerie.crawler.worker_supervisor import (
    SupervisorObservation,
    _worker_declared_failure_v3,
)


def _observation(**overrides: Any) -> SupervisorObservation:
    """Return one designed-failure-exit supervisor observation with overrides."""

    base: dict[str, Any] = {
        "argv": ("python",),
        "cwd": "/tmp",
        "exit_code": 1,
        "signal_number": None,
        "wall_seconds": 1.0,
        "cpu_seconds": 1.0,
        "peak_rss_bytes": 1,
        "timed_out": False,
        "rss_exceeded": False,
        "stdout_sha256": "sha256:0",
        "stdout_bytes": 0,
        "stdout_tail": "",
        "stderr_sha256": "sha256:0",
        "stderr_bytes": 0,
        "stderr_tail": "",
        "stdout_path": "/tmp/stdout.log",
        "stderr_path": "/tmp/stderr.log",
    }
    base.update(overrides)
    return SupervisorObservation(**base)


def _failure_receipt(**overrides: Any) -> dict[str, Any]:
    """Return the incident's honest failure receipt shape (m12646, minimized)."""

    receipt: dict[str, Any] = {
        "raw_award_receipt": None,
        "raw_award_receipt_sha256": None,
        "diagnostic": {
            "error": None,
            "per_mode": {
                "train": {
                    "constructor_started": True,
                    "constructor_completed": True,
                    "input_completed": True,
                    "forward_started": True,
                    "forward_completed": False,
                    "error": {
                        "exception_type": "builtins.IndexError",
                        "message": "index out of range in self",
                        "traceback": "Traceback (most recent call last): ...",
                    },
                }
            },
        },
    }
    receipt.update(overrides)
    return receipt


def _declared(
    receipt: Optional[dict[str, Any]],
    *,
    receipt_error: Optional[str] = None,
    completion: Optional[dict[str, Any]] = None,
    **observation_overrides: Any,
) -> bool:
    """Evaluate the closed shape with incident defaults and overrides."""

    return _worker_declared_failure_v3(
        _observation(**observation_overrides),
        receipt,
        receipt_error,
        completion,
    )


@pytest.mark.smoke
def test_incident_shape_is_a_worker_declared_failure() -> None:
    """The exact m12646 leg-36 shape settles as the worker's own failure."""

    assert _declared(_failure_receipt()) is True


@pytest.mark.smoke
def test_global_diagnostic_error_also_declares_failure() -> None:
    """A structured receipt-global error is failure evidence on its own."""

    receipt = _failure_receipt()
    receipt["diagnostic"] = {
        "error": {"exception_type": "builtins.RuntimeError", "message": "boom"},
        "per_mode": {},
    }
    assert _declared(receipt) is True


@pytest.mark.smoke
def test_global_error_beside_a_completed_forward_is_a_contradiction() -> None:
    """DX-12/RE-13: a global failure claim over completed work never declares.

    A worker asserting "I failed" while its own per-mode evidence asserts a
    forward COMPLETED is contradicting itself; the contradictory receipt stays
    on the fail-closed machinery protocol-park path instead of reaching the
    model-owned receipt-evidence arms. Both directions pinned: the same global
    error with only INCOMPLETE forwards still declares honestly.
    """

    contradictory = _failure_receipt()
    contradictory["diagnostic"] = {
        "error": {"exception_type": "builtins.RuntimeError", "message": "boom"},
        "per_mode": {
            "train": {"forward_completed": True, "error": None},
            "eval": {"forward_completed": False, "error": None},
        },
    }
    assert _declared(contradictory) is False

    honest = _failure_receipt()
    honest["diagnostic"] = {
        "error": {"exception_type": "builtins.RuntimeError", "message": "boom"},
        "per_mode": {
            "train": {"forward_completed": False, "error": None},
            "eval": {"forward_completed": False, "error": None},
        },
    }
    assert _declared(honest) is True


@pytest.mark.smoke
def test_contradictory_receipt_at_a_kill_boundary_never_declares() -> None:
    """DX-12/RE-13 matrix: the contradiction composes with every kill arm.

    A worker killed at the publication boundary (signal, timeout, RSS
    watermark) with a contradictory receipt on disk must keep the fail-closed
    path on BOTH grounds -- the observation defection and the contradiction.
    """

    contradictory = _failure_receipt()
    contradictory["diagnostic"] = {
        "error": {"exception_type": "builtins.RuntimeError", "message": "boom"},
        "per_mode": {"train": {"forward_completed": True, "error": None}},
    }
    assert _declared(contradictory) is False
    assert _declared(contradictory, signal_number=9) is False
    assert _declared(contradictory, timed_out=True) is False
    assert _declared(contradictory, rss_exceeded=True) is False
    assert _declared(contradictory, exit_code=0) is False


@pytest.mark.smoke
def test_every_single_defection_keeps_the_fail_closed_path() -> None:
    """Each one-field defection from the closed shape refuses the lane."""

    assert _declared(None) is False
    assert _declared(_failure_receipt(), receipt_error="invalid-receipt:x") is False
    assert _declared(_failure_receipt(), completion={"completion_line": "x"}) is False
    assert _declared(_failure_receipt(raw_award_receipt={"present": True})) is False
    assert _declared(_failure_receipt(raw_award_receipt_sha256="sha256:1")) is False
    assert _declared(_failure_receipt(), exit_code=0) is False
    assert _declared(_failure_receipt(), exit_code=126) is False
    assert _declared(_failure_receipt(), exit_code=None) is False
    assert _declared(_failure_receipt(), signal_number=9) is False
    assert _declared(_failure_receipt(), timed_out=True) is False
    assert _declared(_failure_receipt(), rss_exceeded=True) is False
    assert _declared(_failure_receipt(diagnostic=None)) is False
    assert _declared(_failure_receipt(diagnostic={"error": None, "per_mode": {}})) is False


@pytest.mark.smoke
def test_success_shaped_receipt_without_attestation_stays_mismatch() -> None:
    """A completed-forward claim with no completion file is NOT this lane.

    A worker claiming success (``forward_completed`` true, no structured
    error) that produced no attestation is exactly the unclean-association
    case the mismatch path exists for; the failure lane must never bless it,
    even when a stray error mapping rides next to the success claim.
    """

    receipt = _failure_receipt()
    mode = receipt["diagnostic"]["per_mode"]["train"]
    mode["forward_completed"] = True
    mode["error"] = None
    assert _declared(receipt) is False
    mode["error"] = {"exception_type": "builtins.ValueError", "message": "x"}
    assert _declared(receipt) is False


@pytest.mark.smoke
def test_supervise_worker_wires_the_declared_failure_lane() -> None:
    """The v3 arm consults the lane for BOTH the mismatch mint and the partial.

    Pre-fix, ``supervise_worker`` minted ``missing-or-mismatched-v3-attestation``
    and a ``protocol-violation`` partial for every raw-receipt-less result;
    this pin fails on any regression that drops either guard.
    """

    source = inspect.getsource(worker_supervisor.supervise_worker)
    assert "_worker_declared_failure_v3(" in source
    assert "if receipt_error is None and not worker_declared_failure:" in source
    assert "if raw_receipt is None and not worker_declared_failure:" in source
