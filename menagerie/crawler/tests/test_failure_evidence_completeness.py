"""Failure-evidence completeness: no published failure without captured evidence.

BL-12(f) / AREA_WORKER opus W-7 + fable W-F7 / catalog C-17: 17 of the pilot's
45 published ``forward/mode-run`` failures carried zero exception evidence while
the parent's own ``unattested_partial`` said ``protocol-violation``, and a
``SystemExit`` from model code skipped the receipt entirely. The fixes pinned
here:

* the worker records ``BaseException`` (``SystemExit`` from argparse inside a
  constructor, ``absl.app``-style exits) in the receipt instead of unwinding
  past it;
* ``run_worker`` writes a last-resort receipt when capture machinery itself
  aborts, so "worker aborted before receipt" is a recorded state, not an
  absence;
* ``_supervised_failure`` publishes an explicit ``runner/attestation-mismatch``
  arm when the worker result is present but its attestation is unclean, and the
  final fall-through can never mint an evidence-free ``forward/mode-run`` -- a
  residue with no worker error is ``runner/unclassified-supervised-failure``.

The census test at the bottom is the permanent completeness tripwire: every
failure classification the worker area can publish must carry its evidence or
be a parent-observed fact whose observation IS the evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import pytest

import menagerie.crawler.worker as worker_module
from menagerie.crawler.constants import RunMode
from menagerie.crawler.driver_receipts import _supervised_failure
from menagerie.crawler.identity import compute_recipe_revision, hash_bytes
from menagerie.crawler.standard_inputs import InputSpec
from menagerie.crawler.worker import WorkerRequest, run_worker
from menagerie.crawler.worker_supervisor import (
    SupervisedResult,
    SupervisorObservation,
)


def _adapter_revision(path: Path, source_identity: str = "unbound") -> str:
    """Return the exact typed-adapter revision expected by a worker request."""

    return compute_recipe_revision(
        {"recipe_type": "typed-adapter", "path": path.name},
        source_identity,
        adapter_bytes=path.read_bytes(),
    )


def _request_for(adapter: Path, tmp_path: Path, stable_id: str) -> WorkerRequest:
    """Return one direct worker request for an adapter under ``tmp_path``."""

    return WorkerRequest(
        stable_id=stable_id,
        recipe={
            "kind": "typed-adapter",
            "path": str(adapter),
            "adapter_sha256": hash_bytes(adapter.read_bytes()),
        },
        modality="unknown",
        input_spec=InputSpec((1, 4), "float32"),
        scratch_root=tmp_path / "scratch",
        receipt_path=tmp_path / "result" / "receipt.json",
        meaningful_modes=(RunMode.EVAL,),
        recipe_revision=_adapter_revision(adapter),
    )


@pytest.mark.smoke
def test_systemexit_in_constructor_still_writes_receipt_with_evidence(
    tmp_path: Path,
) -> None:
    """``sys.exit()`` inside a model constructor leaves a full evidence receipt."""

    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        """from __future__ import annotations
import sys
import torch

def build_model() -> object:
    sys.exit(3)

def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    return ((torch.zeros(1, 4, device=device),), {})
""",
        encoding="utf-8",
    )
    request = _request_for(adapter, tmp_path, "m_sysexit_ctor")
    receipt = run_worker(request)
    assert request.receipt_path.exists()
    error = receipt["error"]
    assert error is not None
    assert error["exception_type"] == "builtins.SystemExit"
    assert error["traceback"]
    assert receipt["constructor_started"] is True
    assert receipt["constructor_completed"] is False


@pytest.mark.smoke
def test_systemexit_in_forward_records_mode_error(tmp_path: Path) -> None:
    """``SystemExit`` raised by ``forward`` lands in the per-mode error, typed."""

    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        """from __future__ import annotations
import sys
import torch

class Exiting(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        sys.exit(7)

def build_model() -> object:
    return Exiting()

def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    return ((torch.zeros(1, 4, device=device),), {})
""",
        encoding="utf-8",
    )
    request = _request_for(adapter, tmp_path, "m_sysexit_forward")
    receipt = run_worker(request)
    assert request.receipt_path.exists()
    mode_receipt = receipt["per_mode"]["eval"]
    assert mode_receipt["forward_completed"] is False
    assert mode_receipt["error"]["exception_type"] == "builtins.SystemExit"
    assert mode_receipt["error"]["traceback"]


@pytest.mark.smoke
def test_capture_machinery_abort_writes_last_resort_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A TL-side abort before the receipt still leaves a recorded envelope."""

    adapter = tmp_path / "adapter.py"
    adapter.write_text("def build_model():\n    return None\n", encoding="utf-8")
    request = WorkerRequest(
        stable_id="m_aborted",
        recipe={
            "kind": "typed-adapter",
            "path": str(adapter),
            "adapter_sha256": hash_bytes(adapter.read_bytes()),
        },
        modality="unknown",
        input_spec=InputSpec((1, 4), "float32"),
        scratch_root=tmp_path / "scratch",
        receipt_path=tmp_path / "result" / "receipt.json",
        protocol_version="menagerie.crawler.worker-request.v3",
        meaningful_modes=(RunMode.EVAL,),
        mode=RunMode.EVAL,
        recipe_revision=_adapter_revision(adapter),
    )

    def exploding_execute(_request: WorkerRequest) -> tuple[dict[str, Any], Any]:
        raise SystemExit(9)

    monkeypatch.setattr(worker_module, "_execute", exploding_execute)
    with pytest.raises(SystemExit):
        run_worker(request)
    assert request.receipt_path.exists()
    persisted = json.loads(request.receipt_path.read_text(encoding="utf-8"))
    diagnostic = persisted["diagnostic"]
    assert diagnostic["error"]["reason_code"] == "worker-aborted-before-receipt"
    assert diagnostic["error"]["exception_type"] == "builtins.SystemExit"
    assert persisted["raw_award_receipt"] is None


def _observation(**overrides: Any) -> SupervisorObservation:
    """Return one clean supervisor observation with optional overrides."""

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


def _classify(
    *,
    receipt_error: Optional[str],
    receipt: Mapping[str, Any],
    mode_receipt: Mapping[str, Any],
    worker_result_present: bool,
    unattested_partial: Optional[Mapping[str, Any]] = None,
    observation: Optional[SupervisorObservation] = None,
) -> Mapping[str, Any]:
    """Run one supervised-failure classification with a minimal result.

    This helper is this module's ONLY ``SupervisedResult`` construction site;
    the anti-substitution inventory pins that fact so synthetic result shapes
    stay auditable in one place per module.
    """

    result = SupervisedResult(
        observation=observation if observation is not None else _observation(),
        worker_receipt=dict(receipt) if worker_result_present else None,
        receipt_error=receipt_error,
        unattested_partial=dict(unattested_partial) if unattested_partial else None,
    )
    return _supervised_failure(
        result,
        receipt,
        mode_receipt,
        receipt.get("policy_observation", {}),
        worker_result_present=worker_result_present,
    )


@pytest.mark.smoke
def test_attestation_mismatch_is_published_typed_not_as_forward_failure() -> None:
    """Worker result present + unclean attestation is ``runner/attestation-mismatch``.

    The pre-fix fall-through published ``forward/mode-run`` with
    ``mode_error: None`` -- a model verdict with zero evidence, contradicting
    the parent's own ``unattested_partial``.
    """

    receipt: dict[str, Any] = {
        "constructor_started": True,
        "constructor_completed": True,
        "input_completed": True,
        "error": None,
        "policy_observation": {},
    }
    failure = _classify(
        receipt_error="missing-or-mismatched-v3-attestation",
        receipt=receipt,
        mode_receipt={"error": None},
        worker_result_present=True,
        unattested_partial={
            "state": "unattested-partial",
            "stage": "forward",
            "reason_code": "protocol-violation",
            "diagnostic_sha256": None,
        },
    )
    assert failure["stage"] == "runner"
    assert failure["reason_code"] == "attestation-mismatch"
    assert failure["details"]["receipt_error"] == "missing-or-mismatched-v3-attestation"


@pytest.mark.smoke
def test_forward_mode_run_requires_worker_error_evidence() -> None:
    """The final branch publishes ``forward/mode-run`` only WITH worker evidence."""

    receipt: dict[str, Any] = {
        "constructor_started": True,
        "constructor_completed": True,
        "input_completed": True,
        "error": None,
        "policy_observation": {},
    }
    failure = _classify(
        receipt_error=None,
        receipt=receipt,
        mode_receipt={"error": None},
        worker_result_present=True,
    )
    assert failure["stage"] == "runner"
    assert failure["reason_code"] == "unclassified-supervised-failure"
    assert failure["details"]["inspected"]


@pytest.mark.smoke
def test_forward_mode_run_with_evidence_still_publishes_forward() -> None:
    """A real per-mode worker error keeps its honest ``forward/mode-run`` verdict."""

    mode_error = {
        "exception_type": "builtins.ValueError",
        "message": "bad shape",
        "traceback": "Traceback (most recent call last): ...",
    }
    receipt: dict[str, Any] = {
        "constructor_started": True,
        "constructor_completed": True,
        "input_completed": True,
        "error": None,
        "policy_observation": {},
    }
    failure = _classify(
        receipt_error=None,
        receipt=receipt,
        mode_receipt={"error": mode_error},
        worker_result_present=True,
    )
    assert failure["stage"] == "forward"
    assert failure["reason_code"] == "mode-run"
    assert failure["exception_type"] == "builtins.ValueError"
    assert failure["traceback"]


#: Parent-observed classifications whose observation IS the evidence: the fact
#: was measured by the supervisor, not claimed by the worker, so no Python
#: traceback can exist for it.
_PARENT_FACT_REASONS = frozenset(
    {
        "timeout",
        "rss-cap",
        "signal",
        "missing-receipt",
        "protocol-violation",
        "sandbox-unavailable-v1",
        "attestation-mismatch",
        "unclassified-supervised-failure",
        "telemetry-integrity",
    }
)


@pytest.mark.smoke
def test_failure_evidence_completeness_census() -> None:
    """CENSUS: every classification carries worker evidence or is a parent fact.

    A published failure with no exception evidence AND a reason code outside
    the parent-observed-fact set is exactly the C-17 recurrence; this census
    fails on any new classification arm that reintroduces one.
    """

    worker_error = {
        "exception_type": "builtins.RuntimeError",
        "message": "boom",
        "traceback": "Traceback (most recent call last): ...",
    }
    scenarios: list[Mapping[str, Any]] = []
    clean_receipt: dict[str, Any] = {
        "constructor_started": True,
        "constructor_completed": True,
        "input_completed": True,
        "error": None,
        "policy_observation": {},
    }
    # Parent-fact arms.
    scenarios.append(
        _classify(
            receipt_error=None,
            receipt={},
            mode_receipt={},
            worker_result_present=False,
            observation=_observation(timed_out=True),
        )
    )
    scenarios.append(
        _classify(
            receipt_error=None,
            receipt={},
            mode_receipt={},
            worker_result_present=False,
            observation=_observation(rss_exceeded=True),
        )
    )
    scenarios.append(
        _classify(
            receipt_error=None,
            receipt={},
            mode_receipt={},
            worker_result_present=False,
            observation=_observation(exit_code=None, signal_number=9),
        )
    )
    scenarios.append(
        _classify(
            receipt_error="missing-receipt",
            receipt={},
            mode_receipt={},
            worker_result_present=False,
        )
    )
    # Worker-evidence arms: each stage failure carries the worker error.
    for flags in (
        {"constructor_started": False},
        {"constructor_started": True, "constructor_completed": False},
        {
            "constructor_started": True,
            "constructor_completed": True,
            "input_completed": False,
        },
    ):
        receipt = {**clean_receipt, **flags, "error": worker_error}
        scenarios.append(
            _classify(
                receipt_error=None,
                receipt=receipt,
                mode_receipt={},
                worker_result_present=True,
            )
        )
    scenarios.append(
        _classify(
            receipt_error=None,
            receipt=clean_receipt,
            mode_receipt={"error": worker_error},
            worker_result_present=True,
        )
    )
    # Machinery arms.
    scenarios.append(
        _classify(
            receipt_error="missing-or-mismatched-v3-attestation",
            receipt=clean_receipt,
            mode_receipt={"error": None},
            worker_result_present=True,
        )
    )
    scenarios.append(
        _classify(
            receipt_error=None,
            receipt=clean_receipt,
            mode_receipt={"error": None},
            worker_result_present=True,
        )
    )
    evidence_free = [
        failure
        for failure in scenarios
        if failure["exception_type"] is None
        and failure["traceback"] is None
        and failure["reason_code"] not in _PARENT_FACT_REASONS
    ]
    assert evidence_free == []
