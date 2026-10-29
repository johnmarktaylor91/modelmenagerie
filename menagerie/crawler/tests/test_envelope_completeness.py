"""AG-9: envelope completeness is asserted BEFORE a checker session is bought.

200a evidence (m8590 seq 296, m8635, m11578, m5445, m4334 x2): seven
``cannot-verify`` complaints were pure machinery faults -- the declared
``allowed_read_roots`` excluded or no longer contained the bytes the checker
was asked to recompute -- yet each burned a MODEL repair round, and one
contributed to an in-rung terminal (m8590). These tests pin the fix shape:

* an item whose author ``model_dir`` does not dereference refuses at envelope
  build with the typed :class:`CheckerEnvelopeIncompleteError` (the
  m8590-seq-296 path: the work-dir root the checker needed was not readable);
* an item whose frozen ``source-cas`` sibling is missing refuses the same way
  (the m4334 path: no recomputable preimage behind the declared digests);
* the refusal is PRE-PURCHASE and produces no envelope, no session, and no
  durable gate row -- and a round that never becomes a durable gate can never
  charge (the complaint-disclosure doctrine's existing boundary), so an
  envelope fault is structurally non-chargeable;
* a complete item still builds -- nothing is weakened, the probe only refuses
  requests that were guaranteed to die as machinery cannot-verify.

The probe sits in ``_build_envelope``, the one boundary every real checker
request passes through, so all three gate kinds are covered by construction.
"""

from __future__ import annotations

import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.constants import METADATA_BATCH_MAX, GateKind

from menagerie.crawler.checker_dispatch import (
    CheckerDispatchError,
    CheckerEnvelopeIncompleteError,
    build_metadata_vet_envelope,
)
from menagerie.crawler.tests.conftest import (
    _model_facts,
    bind_checker_item_verified_hashes,
    make_gate,
    make_model,
    materialize_checker_model_dir,
)

pytestmark = pytest.mark.smoke


def _request_item(item: dict[str, Any]) -> dict[str, Any]:
    """Build one minimal identity-bound checker request item with real roots."""

    request: dict[str, Any] = {
        "work_id": item["work_id"],
        "campaign_root_work_id": item["campaign_root_work_id"],
        "stable_id": item["stable_id"],
        "family_representative_id": item["family_representative_id"],
        "fidelity_identity": item["fidelity_identity"],
        "vet_identity": item["vet_identity"],
        "verified_hashes": deepcopy(item["verified_hashes"]),
        "proposal": {
            "description": "scoped test proposal",
            "proposed_facts": _model_facts(make_model(item["stable_id"], accepted=True)),
        },
        "source_manifest": {"sources": []},
        "evidence": {"excerpts": []},
        "model_dir": materialize_checker_model_dir(item["stable_id"]),
    }
    # The machine digest attestation recomputes every bound key at build; the
    # shared gate fixture's placeholder digests describe no real author round.
    return bind_checker_item_verified_hashes(request, gate_kind=GateKind.METADATA_BATCH)


def _build(items: list[dict[str, Any]], tmp_path: Path) -> dict[str, Any]:
    return build_metadata_vet_envelope(
        items,
        gate_round=1,
        output_path=tmp_path / "result.json",
        checker_model="codex",
        checker_version="test",
        request_nonce="nonce-1",
    )


def test_missing_model_dir_root_refuses_before_purchase(tmp_path: Path) -> None:
    """The m8590-seq-296 path: an unreadable work-dir root refuses typed."""

    gate = make_gate([f"m_{index}" for index in range(METADATA_BATCH_MAX)])
    items = [_request_item(item) for item in gate["items"]]
    items[-1]["model_dir"] = str(tmp_path / "gone" / "author" / "model")
    with pytest.raises(CheckerEnvelopeIncompleteError) as excinfo:
        _build(items, tmp_path)
    assert excinfo.value.stable_ids == (items[-1]["stable_id"],)
    assert any("gone" in path for path in excinfo.value.unreadable_paths)
    # The refusal is a checker-LANE contract fault, so the driver's existing
    # lane handling parks it; it is never spelled as a model complaint.
    assert isinstance(excinfo.value, CheckerDispatchError)


def test_missing_source_cas_sibling_refuses_before_purchase(tmp_path: Path) -> None:
    """The m4334 path: digests without a recomputable preimage root refuse."""

    gate = make_gate([f"m_cas_{index}" for index in range(METADATA_BATCH_MAX - 1)] + ["m_solo"])
    items = [_request_item(item) for item in gate["items"]]
    model_dir = Path(items[-1]["model_dir"])
    shutil.rmtree(model_dir.parent / "source-cas")
    try:
        with pytest.raises(CheckerEnvelopeIncompleteError) as excinfo:
            _build(items, tmp_path)
    finally:
        (model_dir.parent / "source-cas").mkdir()
    assert excinfo.value.stable_ids == ("m_solo",)
    assert any(path.endswith("source-cas") for path in excinfo.value.unreadable_paths)


def test_complete_items_still_build(tmp_path: Path) -> None:
    """Nothing is weakened: readable roots build exactly as before."""

    gate = make_gate([f"m_{index}" for index in range(METADATA_BATCH_MAX)])
    items = [_request_item(item) for item in gate["items"]]
    envelope = _build(items, tmp_path)
    declared_roots = envelope["allowed_read_roots"]
    for item in items:
        cas_root = str(Path(item["model_dir"]).parent / "source-cas")
        assert cas_root in declared_roots


def test_missing_declaration_keeps_its_established_refusal(tmp_path: Path) -> None:
    """An item with NO model_dir still refuses on the older declaration contract."""

    gate = make_gate([f"m_und_{index}" for index in range(METADATA_BATCH_MAX - 1)] + ["m_undeclared"])
    items = [_request_item(item) for item in gate["items"]]
    items[-1].pop("model_dir")
    with pytest.raises(CheckerDispatchError, match="model_dir"):
        _build(items, tmp_path)
