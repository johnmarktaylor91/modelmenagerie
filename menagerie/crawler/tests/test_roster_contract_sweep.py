"""Roster-contract sweep: declared inputs materialize offline or are disclosed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from menagerie.crawler.standard_inputs import InputSpec
from menagerie.crawler.tools.roster_contract_sweep import main, sweep
from menagerie.crawler.worker import WorkerRequest, _materialize_declarative_call

_BERT_CONTRACT = {
    "args": [
        {
            "kind": "tensor",
            "path": "args[0]",
            "shape": [1, 128],
            "dtype": "int64",
            "distribution": "zeros",
        }
    ],
    "kwargs": [
        {
            "kind": "tensor",
            "path": "kwargs.attention_mask",
            "shape": [1, 128],
            "dtype": "int64",
            "distribution": "ones",
        }
    ],
    "non_tensor_values": [],
}


@pytest.mark.smoke
def test_bert_zeros_ones_contract_materializes_exactly(tmp_path: Path) -> None:
    """THE NAMED TRIPWIRE: the exact zeros/ones BERT shape through the REAL seam."""

    request = WorkerRequest(
        stable_id="m-bert",
        recipe={},
        modality="language",
        input_spec=None,
        scratch_root=tmp_path,
        receipt_path=tmp_path / "receipt.json",
        input_contract=_BERT_CONTRACT,
    )
    args, kwargs, input_kind, _asset, _note = _materialize_declarative_call(request)
    (input_ids,) = args
    attention_mask = kwargs["attention_mask"]
    assert isinstance(input_ids, torch.Tensor) and isinstance(attention_mask, torch.Tensor)
    assert tuple(input_ids.shape) == (1, 128) and input_ids.dtype == torch.int64
    assert tuple(attention_mask.shape) == (1, 128) and attention_mask.dtype == torch.int64
    assert torch.equal(input_ids, torch.zeros(1, 128, dtype=torch.int64))
    assert torch.equal(attention_mask, torch.ones(1, 128, dtype=torch.int64))
    # NOTE (finding, runner-exec territory): worker.py aggregates MIXED leaf
    # kinds (standard-zeros + standard-ones here) into the provenance label
    # "random-fallback" although nothing here is random — the VALUES above are
    # the contract's substance; the label is pinned as-is so the fix (an
    # honest mixed-kind label) shows up as a deliberate change.
    assert input_kind == "random-fallback"


@pytest.mark.smoke
def test_sweep_discloses_unhonorable_and_contractless(tmp_path: Path) -> None:
    """Honored, refused, and contractless members are each accounted exactly."""

    rows: list[dict[str, object]] = [
        {"stable_id": "m-ok", "modality": "vision", "input_spec": {"shape": [1, 3, 8, 8]}},
        {"stable_id": "m-rich", "modality": "language", "input_contract": _BERT_CONTRACT},
        {"stable_id": "m-bad", "modality": "vision", "input_spec": {"shape": [0, 3]}},
        {"stable_id": "m-none"},
    ]
    report = sweep(rows, tmp_path)
    assert report["honored"] == ["m-ok", "m-rich"]
    assert report["contractless"] == ["m-none"]
    assert [entry["stable_id"] for entry in report["refused"]] == ["m-bad"]
    assert report["status"] == "unhonorable-disclosed"
    assert "positive" in report["refused"][0]["reason"]


@pytest.mark.smoke
def test_cli_refuses_unhonorable_when_asked(tmp_path: Path) -> None:
    """--refuse-unhonorable turns the disclosure into a staging preflight arm."""

    members = tmp_path / "members.jsonl"
    members.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"stable_id": "m-ok", "modality": "vision", "input_spec": {"shape": [2, 3]}},
                {"stable_id": "m-bad", "input_spec": {"shape": "not-a-shape"}},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    assert (
        main(
            [
                "--members-jsonl",
                str(members),
                "--scratch-root",
                str(tmp_path / "scratch"),
                "--refuse-unhonorable",
            ]
        )
        == 2
    )
    assert main(["--members-jsonl", str(members), "--scratch-root", str(tmp_path / "scratch")]) == 0


@pytest.mark.smoke
def test_plain_spec_path_matches_production_normalization() -> None:
    """The sweep normalizes specs with the same InputSpec authority production uses."""

    spec = InputSpec.from_value({"shape": [1, 128], "dtype": "torch.int64"})
    assert spec.dtype == "int64"
    assert spec.shape == (1, 128)
