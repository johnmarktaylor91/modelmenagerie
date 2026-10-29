"""B2 (step-100 triage): graph-model inputs and the per-mode receipt pipeline.

m538 (DGL-TAGConv): the worker's input-signature builder hashed every
``python``-kind leaf with canonical-JSON ``stable_hash``, so a DGLGraph input
raised ``TypeError: Object of type DGLGraph is not JSON serializable`` at
``worker._input_signature`` -- AFTER the forward itself may have succeeded --
and destroyed the whole mode receipt. The published diagnostic then carried
``$.error.details.mode_error: None`` because the failure classifier only read
the (never-built) per-mode receipt's error and dropped the receipt's global
error where the real traceback lived.

The stub below is deliberately dependency-free: it reproduces the exact
non-JSON-serializable leaf shape without adding dgl to the test environment.
"""

from __future__ import annotations

from typing import Any

import pytest

from menagerie.crawler.constants import RunMode
from menagerie.crawler.identity import stable_hash, stable_leaf_hash
from menagerie.crawler.metadata import input_signature_matches_contract
from menagerie.crawler.worker import _input_signature, _mode_receipt
from menagerie.crawler.driver_receipts import _supervised_failure
from menagerie.crawler.tests.conftest import make_supervised_worker_result_v3
from menagerie.crawler.worker_supervisor import SupervisorObservation

pytestmark = pytest.mark.smoke


class _DGLGraphLike:
    """Minimal non-JSON-serializable graph-object stand-in (no dgl dependency)."""

    def __init__(self, n_nodes: int = 3) -> None:
        self.n_nodes = n_nodes
        self.edges = tuple((index, index + 1) for index in range(n_nodes - 1))


class _OtherGraphLike:
    """A second foreign type proving the fallback digest is type-tagged."""


def test_stable_leaf_hash_falls_back_to_type_tagged_digest() -> None:
    """JSON leaves keep stable_hash digests; foreign objects get typed digests."""

    assert stable_leaf_hash({"a": 1}) == stable_hash({"a": 1})
    assert stable_leaf_hash("text") == stable_hash("text")

    first = stable_leaf_hash(_DGLGraphLike(3))
    second = stable_leaf_hash(_DGLGraphLike(7))
    # Documented determinism bound: type-tagged, content-blind.
    assert first == second
    assert first.startswith("sha256:")
    assert stable_leaf_hash(_OtherGraphLike()) != first
    # Non-finite floats (ValueError under allow_nan=False) also fall back
    # instead of destroying the receipt.
    assert stable_leaf_hash(float("nan")) == stable_leaf_hash(float("inf"))


def test_input_signature_survives_graph_object_leaves() -> None:
    """The m538 crash shape: a graph-object leaf no longer raises TypeError."""

    graph = _DGLGraphLike()
    signature = _input_signature((graph,), {"graph": _DGLGraphLike()})
    leaves = {leaf["path"]: leaf for leaf in signature["leaves"]}
    assert set(leaves) == {"input.args[0]", "input.kwargs.graph"}
    for leaf in leaves.values():
        assert leaf["kind"] == "python"
        assert str(leaf["value_sha256"]).startswith("sha256:")
    # Two instances of the same foreign type produce one deterministic digest.
    assert leaves["input.args[0]"]["value_sha256"] == leaves["input.kwargs.graph"]["value_sha256"]


def test_mode_receipt_completes_for_graph_input_forward() -> None:
    """A graph-input forward earns a complete per-mode receipt, not a crash."""

    class _GraphModel:
        def eval(self) -> None:
            return None

        def forward(self, graph: Any) -> int:
            return graph.n_nodes

    receipt, output = _mode_receipt(
        _GraphModel(),
        (_DGLGraphLike(5),),
        {},
        "constructed",
        None,
        "graph stub",
        RunMode.EVAL,
        "numpy",
        0.01,
    )
    assert output == 5
    assert receipt["error"] is None
    assert receipt["forward_completed"] is True
    assert receipt["input_signature"]["leaves"][0]["value_sha256"].startswith("sha256:")


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


def test_supervised_failure_propagates_global_receipt_error_into_mode_error() -> None:
    """The published diagnostic carries the receipt's global error verbatim.

    m538's exact shape: constructor and input completed, per_mode empty because
    the failure escaped while BUILDING the mode receipt, and the only traceback
    recorded in the receipt's global error. The pre-fix classifier published
    ``mode_error: None`` and no traceback.
    """

    global_error = {
        "reason_code": None,
        "exception_type": "builtins.TypeError",
        "message": "Object of type DGLGraph is not JSON serializable",
        "traceback": "Traceback (most recent call last): ... TypeError: ...",
    }
    receipt = {
        "constructor_started": True,
        "constructor_completed": True,
        "input_completed": True,
        "per_mode": {},
        "error": global_error,
    }
    result = make_supervised_worker_result_v3(_failed_observation(), dict(receipt))

    fields = _supervised_failure(
        result,
        receipt,
        {},
        _clean_policy(),
        worker_result_present=True,
    )

    assert fields["stage"] == "forward"
    assert fields["reason_code"] == "mode-run"
    assert fields["details"]["mode_error"] == global_error
    assert fields["message"] == "Object of type DGLGraph is not JSON serializable"
    assert fields["traceback"] == global_error["traceback"]


def test_supervised_failure_prefers_the_per_mode_error_when_present() -> None:
    """A real per-mode forward error stays the primary published evidence."""

    mode_error = {
        "exception_type": "builtins.RuntimeError",
        "message": "mode forward exploded",
        "traceback": "Traceback ... RuntimeError: mode forward exploded",
    }
    receipt = {
        "constructor_started": True,
        "constructor_completed": True,
        "input_completed": True,
        "per_mode": {"eval": {"error": mode_error}},
        "error": {"message": "outer bookkeeping error", "traceback": "outer"},
    }
    result = make_supervised_worker_result_v3(_failed_observation(), dict(receipt))

    fields = _supervised_failure(
        result,
        receipt,
        {"error": mode_error},
        _clean_policy(),
        worker_result_present=True,
    )

    assert fields["details"]["mode_error"] == mode_error
    assert fields["message"] == "mode forward exploded"


def _signature(kind: str) -> dict[str, Any]:
    """Observed one-leaf worker input signature of the given leaf kind."""

    leaf: dict[str, Any] = {"path": "input.args[0]", "kind": kind}
    if kind == "tensor":
        leaf.update({"shape": [1, 5], "dtype": "float32", "device": "cpu"})
    else:
        leaf["value_sha256"] = stable_leaf_hash(_DGLGraphLike())
    return {"tree": {"args": {"tuple": [{"leaf": 0}]}, "kwargs": {}}, "leaves": [leaf]}


def _graph_contract(
    *, kind: str = "constructed", distribution: str = "constructor"
) -> dict[str, Any]:
    """One-leaf input contract with a constructed graph leaf (schema-required shape)."""

    return {
        "args": [
            {
                "path": "args[0]",
                "kind": kind,
                "shape": [5, 4],
                "dtype": "float32",
                "distribution": distribution,
            }
        ],
        "kwargs": [],
        "non_tensor_values": [],
    }


def test_constructed_graph_leaves_match_python_observations() -> None:
    """The award-path seam behind m538: a constructor leaf is never a tensor.

    A declarative-constructor contract leaf carries the schema-required
    ``shape`` yet materializes as a python-kind object, so requiring a tensor
    observation refused every constructor-distribution graph family at award
    verification even once the receipt crash was fixed.
    """

    assert input_signature_matches_contract(_signature("python"), _graph_contract())


def test_constructed_graph_leaf_observed_as_tensor_still_refuses() -> None:
    """Direction 2: the carve-out accepts exactly python-kind observations."""

    assert not input_signature_matches_contract(_signature("tensor"), _graph_contract())


def test_plain_tensor_contract_leaves_keep_the_tensor_requirement() -> None:
    """Direction 2: an ordinary tensor leaf observed as python still refuses."""

    contract = _graph_contract(kind="tensor", distribution="normal")
    assert not input_signature_matches_contract(_signature("python"), contract)
    signature = _signature("tensor")
    signature["leaves"][0]["shape"] = [5, 4]
    assert input_signature_matches_contract(signature, contract)
