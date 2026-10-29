"""Pre-model-code worker deaths are host/machinery parks (DX-2 / RE-2).

200a charged workers that never reached model code to the MODEL: an exit-126
exec refusal read ``missing-receipt``, a crawler-runtime import death (the
StrEnum/py3.10 bootstrap class) read ``import/import-exception``, and every
missing or protocol-violating receipt was model-owned regardless of whether
any model code provably ran. The classification now keys on the evidence:

* exit 126 with no constructor evidence -> ``runner/worker-bootstrap-failed``
  (host);
* a crawler-frame ImportError before the constructor ->
  ``runner/worker-runtime-unavailable`` (host);
* ``missing-receipt``/``protocol-violation`` are machinery-owned UNLESS the
  receipt proves model code ran (``constructor_started``) -- anti-forgery
  holds (a receipt-suppressing model can still never mint success), and the
  suppressed-AFTER-constructor residual is owned by B3-XQ-1.

RED is pinned in BOTH directions: model-owned import failures and
post-constructor protocol violations keep charging the model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pytest

from menagerie.crawler.constants import FAILURE_REASON_CODES
from menagerie.crawler.driver_receipts import (
    NEVER_MODEL_TERMINAL_OWNERS,
    WORKER_FAILURE_REASON_OWNERS,
    _crawler_runtime_import_failure,
    _failure_owner,
    _supervised_failure,
)
from menagerie.crawler.terminal_ownership import pre_model_code_runner_head_findings
from menagerie.crawler.tests.conftest import make_supervised_worker_result_v3
from menagerie.crawler.worker_supervisor import SupervisorObservation

pytestmark = pytest.mark.smoke

_CRAWLER_FRAME_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "/routed-env/lib/python3.10/site-packages/menagerie/crawler/worker.py",'
    " line 12, in <module>\n"
    "    from enum import StrEnum\n"
    "ImportError: cannot import name 'StrEnum' from 'enum'"
)

_RECIPE_FRAME_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "/routed-env/lib/python3.10/site-packages/menagerie/crawler/worker.py",'
    " line 210, in _load_recipe\n"
    "    module = importlib.import_module(recipe)\n"
    '  File "/work/m123/model_dir/recipe.py", line 3, in <module>\n'
    "    import special_ops\n"
    "ModuleNotFoundError: No module named 'special_ops'"
)


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


def _observation(exit_code: int) -> SupervisorObservation:
    """Parent observation for a worker that exited without a signal."""

    return SupervisorObservation(
        argv=("python", "-m", "menagerie.crawler.worker"),
        cwd="/tmp/worker",
        exit_code=exit_code,
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


def _classified(
    *,
    exit_code: int = 1,
    receipt: Optional[dict[str, Any]] = None,
    receipt_error: Optional[str] = None,
    worker_result_present: bool = True,
) -> dict[str, Any]:
    """Classify one synthetic supervised failure."""

    effective_receipt = receipt if receipt is not None else {}
    result = make_supervised_worker_result_v3(
        _observation(exit_code),
        dict(effective_receipt),
        receipt_error=receipt_error,
    )
    return _supervised_failure(
        result,
        effective_receipt,
        {},
        _clean_policy(),
        worker_result_present=worker_result_present,
    )


def test_exit_126_before_model_code_is_a_host_bootstrap_park() -> None:
    """The exec layer refusing the worker is host weather, never a verdict."""

    fields = _classified(
        exit_code=126,
        receipt={},
        receipt_error="missing-receipt",
        worker_result_present=False,
    )
    assert (fields["stage"], fields["reason_code"]) == ("runner", "worker-bootstrap-failed")
    assert fields["details"]["owner"] == "host"
    assert _failure_owner(fields) in NEVER_MODEL_TERMINAL_OWNERS


def test_exit_126_after_constructor_keeps_receipt_evidence_classification() -> None:
    """A model exiting 126 after its constructor ran is NOT a bootstrap death."""

    fields = _classified(
        exit_code=126,
        receipt={
            "constructor_started": True,
            "constructor_completed": False,
            "error": {"type": "RuntimeError", "message": "constructor blew", "traceback": "..."},
        },
    )
    assert (fields["stage"], fields["reason_code"]) == ("constructor", "exception")
    assert _failure_owner(fields) == "model"


def test_crawler_frame_import_death_is_a_host_runtime_park() -> None:
    """The StrEnum/py3.10 class: our runtime died before any model code."""

    fields = _classified(
        receipt={
            "constructor_started": False,
            "error": {
                "type": "ImportError",
                "message": "cannot import name 'StrEnum' from 'enum'",
                "traceback": _CRAWLER_FRAME_TRACEBACK,
            },
        },
    )
    assert (fields["stage"], fields["reason_code"]) == ("runner", "worker-runtime-unavailable")
    assert fields["details"]["owner"] == "host"
    assert "menagerie/crawler/worker.py" in fields["details"]["crawler_frame"]
    assert _failure_owner(fields) in NEVER_MODEL_TERMINAL_OWNERS


def test_recipe_import_death_keeps_model_ownership() -> None:
    """An import that died in the RECIPE's own frame still charges the model."""

    fields = _classified(
        receipt={
            "constructor_started": False,
            "error": {
                "type": "ModuleNotFoundError",
                "message": "No module named 'special_ops'",
                "traceback": _RECIPE_FRAME_TRACEBACK,
            },
        },
    )
    assert (fields["stage"], fields["reason_code"]) == ("import", "import-exception")
    assert _failure_owner(fields) == "model"


def test_runtime_discriminator_is_import_only() -> None:
    """Only import machinery can die pre-model-code; other types never match."""

    assert (
        _crawler_runtime_import_failure(
            {
                "type": "RuntimeError",
                "message": "boom",
                "traceback": _CRAWLER_FRAME_TRACEBACK,
            }
        )
        is None
    )
    assert _crawler_runtime_import_failure(None) is None
    assert (
        _crawler_runtime_import_failure(
            {"type": "ImportError", "message": "x", "traceback": None}
        )
        is None
    )


@pytest.mark.parametrize(
    ("receipt_error", "reason"),
    [
        ("missing-receipt", "missing-receipt"),
        ("worker receipt failed schema validation", "protocol-violation"),
    ],
)
def test_receiptless_death_without_constructor_evidence_is_machinery(
    receipt_error: str, reason: str
) -> None:
    """No proof model code ran -> the delivery fault is the harness's own."""

    fields = _classified(
        receipt={},
        receipt_error=receipt_error,
        worker_result_present=False,
    )
    assert (fields["stage"], fields["reason_code"]) == ("runner", reason)
    assert fields["details"]["owner"] == "machinery"
    assert fields["details"]["constructor_started"] is False
    assert _failure_owner(fields) in NEVER_MODEL_TERMINAL_OWNERS


def test_receipt_proving_model_code_ran_keeps_model_ownership() -> None:
    """constructor_started=True keeps the anti-forgery direction: model-owned."""

    fields = _classified(
        receipt={"constructor_started": True},
        receipt_error="worker receipt failed schema validation",
        worker_result_present=False,
    )
    assert (fields["stage"], fields["reason_code"]) == ("runner", "protocol-violation")
    assert fields["details"]["constructor_started"] is True
    assert "owner" not in fields["details"]
    assert _failure_owner(fields) == "model"


def test_new_reason_codes_ride_the_closed_vocabulary() -> None:
    """Schema + constants + owner map extend together (DX-15 cross-pin rides CI)."""

    for reason in ("worker-bootstrap-failed", "worker-runtime-unavailable"):
        assert reason in FAILURE_REASON_CODES["runner"]
        assert WORKER_FAILURE_REASON_OWNERS[reason] == "host"


def _head_row(stable_id: str, reason: str, attempt_ids: list[str]) -> dict[str, Any]:
    return {
        "stable_id": stable_id,
        "record_revision": f"rev-{stable_id}",
        "status": {
            "kind": "failed",
            "code": "failed:runner",
            "reason_code": reason,
            "attempt_ids": attempt_ids,
        },
    }


def _attempt_row(attempt_id: str, actor: str, constructor_started: bool) -> dict[str, Any]:
    return {
        "attempt_id": attempt_id,
        "actor": actor,
        "worker_receipt": {"constructor_started": constructor_started},
    }


def test_close_clause_flags_constructor_free_runner_heads(tmp_path: Path) -> None:
    """DX-2 ledger clause: pinned in both directions, driver-lane heads exempt."""

    records_root = tmp_path / "records"
    models = records_root / "models" / "current-shard.jsonl"
    attempts = records_root / "attempts" / "local.jsonl"
    models.parent.mkdir(parents=True)
    attempts.parent.mkdir(parents=True)
    heads = [
        # Offender: worker attempts exist, none reached the constructor.
        _head_row("m-dead", "missing-receipt", ["a-dead-1", "a-dead-2"]),
        # Exempt: one worker attempt proves model code ran.
        _head_row("m-ran", "protocol-violation", ["a-ran-1", "a-ran-2"]),
        # Exempt: driver-lane adjudication only, no worker was ever spawned.
        _head_row("m-driver", "protocol-violation", ["a-driver-1"]),
    ]
    rows = [
        _attempt_row("a-dead-1", "worker", False),
        _attempt_row("a-dead-2", "worker", False),
        _attempt_row("a-ran-1", "worker", False),
        _attempt_row("a-ran-2", "worker", True),
        _attempt_row("a-driver-1", "driver", False),
    ]
    with models.open("w") as handle:
        for row in heads:
            handle.write(json.dumps(row) + "\n")
    with attempts.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    findings = pre_model_code_runner_head_findings(
        records_root,
        baselines={"models.shard": {"rows": 0}, "models.flat": {"rows": 0}},
    )
    assert [finding["stable_id"] for finding in findings] == ["m-dead"]
    assert findings[0]["worker_attempts"] == 2
    # Window scoping: a floor past the offender leaves the close clean.
    assert (
        pre_model_code_runner_head_findings(
            records_root,
            baselines={"models.shard": {"rows": 1}, "models.flat": {"rows": 0}},
        )
        == ()
    )
