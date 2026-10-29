"""DX-9/RE-9: mode disagreement is resolved by evidence, never by repair burn.

200a resolved every worker-vs-proposal mode disagreement by author repair --
0/14 repairs succeeded, several burning their whole cap into a false
``failed:runner`` verdict on the batch-1 unsatisfiability contradiction. Three
arms replace that:

* the WORKER resolves the disagreement itself: when its structural heuristic
  demands a mode the gated proposal lacks, it runs BOTH modes in-process on
  identically seeded fresh builds and lets the captured outputs adjudicate
  (the previously dead ``captured_outputs`` arm of ``detect_meaningful_modes``,
  now authoritative in both directions -- the C.12-adopted mode-contract
  change);
* the DRIVER refuses a structurally unsatisfiable demanded revision typed
  (worker demands train, the proposal validator refuses every batch-1
  rank>=3 train declaration) before any author spend;
* repair-budget exhaustion publishes as the AUTHOR stage, never runner
  (pinned in test_slice_f_driver's cap tests).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from menagerie.crawler.constants import RunMode
from menagerie.crawler.identity import compute_recipe_revision, hash_bytes
from menagerie.crawler.modes import detect_meaningful_modes
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.worker import WorkerRequest, run_worker

_IDENTITY_ADAPTER = """from __future__ import annotations
import torch

class ModeReader(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.training:
            pass
        return value * 2

def build_model() -> object:
    return ModeReader()

def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    return ((torch.ones(1, 4, device=device),), {})
"""

_DIVERGENT_ADAPTER = """from __future__ import annotations
import torch

class ModeShifter(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.training:
            return value + 3
        return value * 2

def build_model() -> object:
    return ModeShifter()

def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    return ((torch.ones(1, 4, device=device),), {})
"""


def _request(tmp_path: Path, adapter_source: str) -> WorkerRequest:
    """Build a single-mode eval request over a heuristic-positive adapter."""

    adapter = tmp_path / "adapter.py"
    adapter.write_text(adapter_source, encoding="utf-8")
    return WorkerRequest(
        stable_id="m_mode_evidence",
        recipe={
            "kind": "typed-adapter",
            "path": str(adapter),
            "adapter_sha256": hash_bytes(adapter.read_bytes()),
        },
        modality=None,
        input_spec=None,
        scratch_root=tmp_path / "scratch",
        receipt_path=tmp_path / "result" / "receipt.json",
        recipe_revision=compute_recipe_revision(
            {"recipe_type": "typed-adapter", "path": adapter.name},
            "unbound",
            adapter_bytes=adapter.read_bytes(),
        ),
        meaningful_modes=(RunMode.EVAL,),
        mode=RunMode.EVAL,
    )


@pytest.mark.smoke
def test_captured_equivalence_is_authoritative_over_the_heuristic() -> None:
    """Observed no-divergence outputs beat the structural mode heuristic."""

    model = torch.nn.Sequential(torch.nn.BatchNorm1d(4))
    # No evidence: the structural heuristic fails closed to both modes.
    assert detect_meaningful_modes(model) == (RunMode.TRAIN, RunMode.EVAL)
    same = torch.ones(2, 4)
    assert detect_meaningful_modes(
        model, captured_outputs={"train": same, "eval": same.clone()}
    ) == (RunMode.EVAL,)
    # Observed divergence keeps both modes, exactly as before.
    assert detect_meaningful_modes(
        model, captured_outputs={"train": same, "eval": same + 1}
    ) == (RunMode.TRAIN, RunMode.EVAL)


def test_worker_probe_resolves_a_heuristic_expansion_in_process(tmp_path: Path) -> None:
    """The worker runs both modes itself and reports NO expansion when equal.

    The adapter's forward reads ``self.training`` (heuristic: both modes) but
    is mode-insensitive; before DX-9 this receipt reported detected
    ``[train, eval]`` over a declared ``[eval]`` and bought an author-repair
    burn. The in-process probe (~one extra forward) settles it.
    """

    receipt = run_worker(_request(tmp_path, _IDENTITY_ADAPTER))

    assert receipt["declared_meaningful_modes"] == ["eval"]
    assert receipt["detected_meaningful_modes"] == ["eval"]
    assert receipt["train_eval_divergence"] == "none"
    assert receipt["divergence_evidence"] is not None
    assert receipt["error"] is None


def test_worker_probe_confirms_a_real_expansion(tmp_path: Path) -> None:
    """A genuinely mode-sensitive model keeps its evidenced expansion report."""

    receipt = run_worker(_request(tmp_path, _DIVERGENT_ADAPTER))

    assert receipt["declared_meaningful_modes"] == ["eval"]
    assert receipt["detected_meaningful_modes"] == ["train", "eval"]
    assert receipt["train_eval_divergence"] != "none"


def test_batch_one_unsatisfiable_expansion_refuses_typed_without_spend(
    tmp_path: Path,
) -> None:
    """The m-batch-1 contradiction refuses at the seam with zero author spend.

    The worker demands train; the proposal validator refuses EVERY train
    declaration over a batch-1 rank>=3 contract, so the demanded revision can
    never validate. 200a burned the full repair cap (0/14 successes) into
    ``failed:runner``; the contradiction now terminalizes as a typed
    input-contract refusal naming the conflict, with no repair generation
    dispatched.
    """

    from menagerie.crawler.tests.test_slice_f_driver import (
        DetectedModeRepairAuthor,
        ForwardScript,
        ScriptedForward,
        _driver,
        _paths,
        _refresh_proposal_identities,
        _rebind_fake_author_result,
        _snapshot,
    )

    class BatchOneAuthor(DetectedModeRepairAuthor):
        """Eval-only proposal whose probe contract is the batch-1 class."""

        def author(self, item, work_root, config, context):  # type: ignore[no-untyped-def]
            artifact = super().author(item, work_root, config, context)
            facts = artifact.proposal["proposed_facts"]
            facts["input_contract"]["args"][0]["shape"] = [1, 3, 8, 8]
            _refresh_proposal_identities(
                artifact.proposal,
                checker_model=config.checker_model,
                checker_version=config.checker_version,
            )
            return _rebind_fake_author_result(artifact)

    snapshot = _snapshot(tmp_path, count=1)
    stable_id = snapshot.items[0].stable_id
    author = BatchOneAuthor()
    result = _driver(
        tmp_path,
        snapshot,
        author=author,
        forward=ScriptedForward(ForwardScript(expanded_id=stable_id)),
        run_repair_max=2,
    ).run()

    assert result.status == "complete"
    paths = _paths(tmp_path, snapshot)
    model = scan_jsonl(paths.ledgers.models)[-1]
    assert model["status"]["code"] == "failed:input"
    assert model["status"]["reason_code"] == "contract-invalid"
    # Zero author-repair spend: only the initial proposal was authored, no
    # repair generation was requested or charged.
    assert author.calls == {stable_id: 1}
    repair_root = paths.work_root / stable_id / "repair"
    assert list(repair_root.glob("run-modes-generation-*.json")) == []
    assert list(repair_root.glob("run-modes-charge-*.json")) == []
    # The typed contradiction rides the attempt evidence.
    unsatisfiable = [
        attempt
        for attempt in scan_jsonl(paths.ledgers.attempts)
        if isinstance(attempt.get("error"), dict)
        and attempt["error"].get("details", {}).get("route") == "mode-contract-unsatisfiable"
    ]
    assert len(unsatisfiable) == 1
    detail = json.dumps(model["status"].get("detail"))
    assert "batch dimension 1" in detail or model["status"].get("detail") is None
