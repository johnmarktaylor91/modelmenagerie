"""D-51: gate-citation era pinning is validated BEFORE checker purchase.

The m6484 class: a checker adjudicating cited facts against a source manifest
that was re-frozen or rotated underneath the citation renders a wrong-era
verdict, and nothing named the mismatch. These tests pin the pre-spend join:

* an item whose artifact binds source-manifest identity X while the envelope
  ships a manifest whose recomputed identity is Y refuses at envelope build
  -- before any session is bought -- with a typed error naming BOTH
  identities (the rotated-fact rig's named diff);
* a matching era builds exactly as before;
* the rotation direction is exercised with real row rotation: re-freezing a
  source row moves the recomputed identity and the stale binding refuses.

Result-time joins (``validate_terminal_disposition_gate``'s identity equality
and membership checks, ``metadata``'s reference resolution) continue to govern
the append side; this pin closes the purchase side.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.constants import METADATA_BATCH_MAX, GateKind

from menagerie.crawler.checker_dispatch import (
    CheckerCitationJoinError,
    CheckerDigestAttestationError,
    build_metadata_vet_envelope,
    validate_citation_source_join,
)
from menagerie.crawler.identity import stable_hash
from menagerie.crawler.tests.conftest import (
    _model_facts,
    bind_checker_item_verified_hashes,
    make_gate,
    make_model,
    materialize_checker_model_dir,
)

pytestmark = pytest.mark.smoke

_ROW = {
    "source_id": "source-1",
    "url": "https://example.com/model",
    "revision": "a" * 40,
    "expected_sha256": "sha256:" + "a" * 64,
}


def _request_item(item: dict[str, Any], *, manifest: dict[str, Any]) -> dict[str, Any]:
    """Build one identity-bound checker request item shipping ``manifest``."""

    proposal = {
        "description": "scoped test proposal",
        "proposed_facts": _model_facts(make_model(item["stable_id"], accepted=True)),
        "source_manifest_identity": stable_hash(manifest["sources"]),
    }
    request: dict[str, Any] = {
        "work_id": item["work_id"],
        "campaign_root_work_id": item["campaign_root_work_id"],
        "stable_id": item["stable_id"],
        "family_representative_id": item["family_representative_id"],
        "fidelity_identity": item["fidelity_identity"],
        "vet_identity": item["vet_identity"],
        "verified_hashes": deepcopy(item["verified_hashes"]),
        "proposal": proposal,
        "source_manifest": deepcopy(manifest),
        "evidence": {"excerpts": []},
        "model_dir": materialize_checker_model_dir(item["stable_id"]),
    }
    # ``verified_hashes.source_manifest`` binds the SHIPPED rows' identity (the
    # same value the proposal's ``source_manifest_identity`` carries); the
    # machine digest attestation recomputes it from the rows at build.
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


def test_matching_manifest_era_builds(tmp_path: Path) -> None:
    """A shipped manifest whose identity equals the binding passes the pin."""

    manifest = {"sources": [dict(_ROW)]}
    gate = make_gate([f"m_{index}" for index in range(METADATA_BATCH_MAX)])
    items = [_request_item(item, manifest=manifest) for item in gate["items"]]
    envelope = _build(items, tmp_path)
    assert len(envelope["items"]) == METADATA_BATCH_MAX


def test_rotated_manifest_refuses_pre_spend_naming_both_identities(tmp_path: Path) -> None:
    """The m6484 fixture: a re-frozen source row moves the era; the stale
    binding refuses before any session, naming cited AND current identities."""

    manifest = {"sources": [dict(_ROW)]}
    gate = make_gate([f"m_{index}" for index in range(METADATA_BATCH_MAX)])
    items = [_request_item(item, manifest=manifest) for item in gate["items"]]
    stale_identity = items[-1]["proposal"]["source_manifest_identity"]
    assert items[-1]["verified_hashes"]["source_manifest"] == stale_identity
    # The rotation: the same source_id re-frozen with different bytes.
    items[-1]["source_manifest"]["sources"][0]["expected_sha256"] = "sha256:" + "b" * 64
    rotated_identity = stable_hash(items[-1]["source_manifest"]["sources"])
    # A real pack binds ``verified_hashes.source_manifest`` to the same stale
    # identity, so the per-item machine digest attestation (D8) now refutes the
    # rotation one step upstream of the D-51 era pin -- still pre-spend, still
    # item-scoped, still naming the cited AND current identities. The era-pin
    # kernel itself is exercised directly below.
    with pytest.raises(CheckerDigestAttestationError) as excinfo:
        _build(items, tmp_path)
    assert excinfo.value.stable_ids == (items[-1]["stable_id"],)
    assert excinfo.value.mismatched_keys == ("source_manifest",)
    assert stale_identity in str(excinfo.value) and rotated_identity in str(excinfo.value)


def test_join_kernel_accepts_explicit_manifest_identity_field() -> None:
    """A manifest carrying its own identity field is compared on that field."""

    manifest = {"manifest_sha256": "sha256:" + "c" * 64, "sources": [dict(_ROW)]}
    validate_citation_source_join(
        stable_id="m_explicit",
        source_manifest=manifest,
        bound_identity="sha256:" + "c" * 64,
    )
    with pytest.raises(CheckerCitationJoinError):
        validate_citation_source_join(
            stable_id="m_explicit",
            source_manifest=manifest,
            bound_identity="sha256:" + "d" * 64,
        )
