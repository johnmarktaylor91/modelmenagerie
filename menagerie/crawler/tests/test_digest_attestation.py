"""The machine digest attestation (F51 checker-lifecycle D8 / DECIDED_PLAN item 20).

The receipt2 live-epoch probe (2026-09-01) measured the one class where a checker
model MISSED a real integrity defect: the rung-adverse golden's bound
``verified_hashes.code`` does not match the staged ``model_dir`` bytes, sol caught
it 3/3, Fable 1/3, and no tripwire sees a miss (a miss makes no claim). The
acceptrate replay then found the same class on 9 of 81 packs. Digest
verification is machine-decidable and was delegated to model shell arithmetic
in both lanes. These tests pin its withdrawal from model judgment -- the third
instance of the locator-attestation idiom:

- the coordinator recomputes EVERY non-null ``verified_hashes`` key against the
  frozen bytes it binds, at the one boundary every checker request passes
  through, and freezes a typed per-item ``digest_attestation`` block with a
  closed status vocabulary (valid / mismatch / unresolvable) and a closed
  per-key recompute basis;
- a ``mismatch`` refuses the envelope BEFORE any session is bought, as an
  item-scoped author-artifact fact (the bound digest is the frozen pack's own
  claim about its bytes; a re-ask is provably futile);
- ``unresolvable`` (the machine cannot locate the staged bytes) discloses and
  dispatch proceeds -- insufficiency, never fabrication, exactly like an
  unattestable locator row;
- the stamp is machine-owned (a caller-supplied conflicting block is refused);
- on the result side a ``hash_mismatches`` entry naming a ``valid`` row disputes
  machine arithmetic and discards the whole result typed; citing a non-valid
  row is licensed; a legacy envelope without the stamp keeps historical
  semantics untouched;
- the wrapper's dynamic prompt region names the authority only on attested
  envelopes, and the frozen prompt bytes (the ``checker_prompt`` identity) do
  not move.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.checker_dispatch import (
    DIGEST_ATTESTATION_AUTHORITY,
    DIGEST_ATTESTATION_MISMATCH,
    DIGEST_ATTESTATION_STATUSES,
    DIGEST_ATTESTATION_UNRESOLVABLE,
    DIGEST_ATTESTATION_VALID,
    DIGEST_BASIS_BY_KEY,
    CheckerDigestAttestationError,
    CheckerDigestAuthorityError,
    CheckerDispatchError,
    CheckerItemContractError,
    _derive_digest_attestation,
    _validate_item_digest_authority,
    build_metadata_vet_envelope,
)
from menagerie.crawler.constants import GateKind
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.tests.conftest import (
    HASH,
    _model_facts,
    bind_checker_item_verified_hashes,
    make_gate,
    make_model,
    materialize_checker_model_dir,
)

_ADAPTER = b"import torch\n\n\ndef build_model():\n    return torch.nn.Identity()\n"
_PIT = b"class PiT:\n    pass\n"


def _typed_item(stable_id: str) -> dict[str, Any]:
    """One R2-shaped metadata item whose staged code closure really exists.

    Mirrors the real m8189 pack shape: a two-row ``code_manifest`` (entry
    point plus one vendored module) under a real ``model_dir``, every digest
    derived from the bytes it binds -- never placeholder-filled.
    """

    gate = make_gate([stable_id])
    expected = gate["items"][0]
    model_dir = Path(materialize_checker_model_dir(stable_id))
    (model_dir / "pit_adapter.py").write_bytes(_ADAPTER)
    (model_dir / "pit.py").write_bytes(_PIT)
    facts = _model_facts(make_model(stable_id, accepted=True))
    facts["source_resolution"]["rung"] = "R2_VENDOR"
    manifest = [
        {"path": "pit.py", "sha256": hash_bytes(_PIT)},
        {"path": "pit_adapter.py", "sha256": hash_bytes(_ADAPTER)},
    ]
    facts["implementation"].update(
        {
            "code_path": "pit_adapter.py",
            "code_sha256": hash_bytes(_ADAPTER),
            "code_manifest": manifest,
            "recipe_type": "adapter",
        }
    )
    item: dict[str, Any] = {
        "work_id": expected["work_id"],
        "campaign_root_work_id": expected["campaign_root_work_id"],
        "stable_id": stable_id,
        "family_representative_id": expected["family_representative_id"],
        "fidelity_identity": None,
        "vet_identity": expected["vet_identity"],
        "verified_hashes": {
            "proposal": HASH,
            "source_manifest": HASH,
            "evidence": HASH,
            "code": HASH,
            "code_manifest": HASH,
            "source_to_code_map": HASH,
            "family_template": None,
        },
        "proposal": {"description": "digest attestation fixture", "proposed_facts": facts},
        "source_manifest": {"sources": []},
        "model_dir": str(model_dir),
    }
    bind_checker_item_verified_hashes(item, gate_kind=GateKind.METADATA_BATCH)
    return item


def _envelope(item: dict[str, Any], tmp_path: Path, nonce: str) -> dict[str, Any]:
    return build_metadata_vet_envelope(
        [item],
        gate_round=1,
        output_path=tmp_path / nonce / "result.json",
        checker_model="codex",
        checker_version="test",
        request_nonce=nonce,
        final_tail=True,
    )


@pytest.mark.smoke
def test_attestation_rows_are_typed_derived_and_closed(tmp_path: Path) -> None:
    """Every non-null key gets one row; null keys are disclosed; basis is closed."""

    item = _typed_item("m_digest_ok")
    envelope = _envelope(item, tmp_path, "digest-ok")
    block = envelope["items"][0]["digest_attestation"]
    assert block["authority"] == DIGEST_ATTESTATION_AUTHORITY
    assert block["all_valid"] is True
    assert block["unbound_keys"] == ["family_template"]
    rows = {row["digest_key"]: row for row in block["rows"]}
    assert sorted(rows) == sorted(
        key for key, digest in item["verified_hashes"].items() if digest is not None
    )
    for key, row in rows.items():
        assert row["status"] == DIGEST_ATTESTATION_VALID
        assert row["bound"] == item["verified_hashes"][key] == row["recomputed"]
        assert row["basis"] == DIGEST_BASIS_BY_KEY[GateKind.METADATA_BATCH][key]
        assert set(row) == {"digest_key", "basis", "bound", "recomputed", "status", "detail"}
    assert DIGEST_ATTESTATION_STATUSES == frozenset(
        {DIGEST_ATTESTATION_VALID, DIGEST_ATTESTATION_MISMATCH, DIGEST_ATTESTATION_UNRESOLVABLE}
    )
    # The envelope-wide era stamp travels beside the locator one.
    assert envelope["digest_authority"] == DIGEST_ATTESTATION_AUTHORITY


@pytest.mark.smoke
def test_the_m8189_shape_refuses_pre_spend_naming_both_code_keys(tmp_path: Path) -> None:
    """Staged entry-point drift is a mismatch on ``code`` AND its manifest row.

    The real rung-adverse pack: ``verified_hashes.code`` (and the matching
    ``code_manifest`` row) bind bytes the staged ``pit_adapter.py`` no longer
    has. That is a coordinator fact about the frozen pack, adjudicated before
    any session exists -- the checker never gets to miss it.
    """

    item = _typed_item("m_digest_drift")
    (Path(item["model_dir"]) / "pit_adapter.py").write_bytes(_ADAPTER + b"# drifted\n")
    block = _derive_digest_attestation(item, GateKind.METADATA_BATCH)
    rows = {row["digest_key"]: row for row in block["rows"]}
    assert rows["code"]["status"] == DIGEST_ATTESTATION_MISMATCH
    assert rows["code"]["recomputed"] == hash_bytes(_ADAPTER + b"# drifted\n")
    assert rows["code_manifest"]["status"] == DIGEST_ATTESTATION_MISMATCH
    assert "pit_adapter.py" in str(rows["code_manifest"]["detail"])
    assert rows["proposal"]["status"] == DIGEST_ATTESTATION_VALID
    assert block["all_valid"] is False

    with pytest.raises(CheckerDigestAttestationError) as excinfo:
        _envelope(item, tmp_path, "digest-drift")
    error = excinfo.value
    assert isinstance(error, CheckerItemContractError)
    assert error.author_side is True
    assert error.stable_ids == ("m_digest_drift",)
    assert error.mismatched_keys == ("code", "code_manifest")
    assert "before any session is bought" in str(error)


@pytest.mark.smoke
def test_forged_in_envelope_digest_is_a_mismatch_too(tmp_path: Path) -> None:
    """A placeholder ``proposal`` digest is refuted by canonical recompute."""

    item = _typed_item("m_digest_forged")
    item["verified_hashes"]["proposal"] = HASH
    with pytest.raises(CheckerDigestAttestationError) as excinfo:
        _envelope(item, tmp_path, "digest-forged")
    assert excinfo.value.mismatched_keys == ("proposal",)


@pytest.mark.smoke
def test_missing_staged_bytes_disclose_unresolvable_and_dispatch_proceeds(
    tmp_path: Path,
) -> None:
    """No bytes to compare is insufficiency, never a positive contradiction."""

    item = _typed_item("m_digest_missing")
    (Path(item["model_dir"]) / "pit.py").unlink()
    envelope = _envelope(item, tmp_path, "digest-missing")
    block = envelope["items"][0]["digest_attestation"]
    rows = {row["digest_key"]: row for row in block["rows"]}
    assert rows["code"]["status"] == DIGEST_ATTESTATION_VALID
    assert rows["code_manifest"]["status"] == DIGEST_ATTESTATION_UNRESOLVABLE
    assert rows["code_manifest"]["recomputed"] is None
    assert "pit.py" in str(rows["code_manifest"]["detail"])
    assert block["all_valid"] is False

    # A missing entry point is unresolvable on ``code`` as well.
    item = _typed_item("m_digest_missing_entry")
    (Path(item["model_dir"]) / "pit_adapter.py").unlink()
    block = _derive_digest_attestation(item, GateKind.METADATA_BATCH)
    rows = {row["digest_key"]: row for row in block["rows"]}
    assert rows["code"]["status"] == DIGEST_ATTESTATION_UNRESOLVABLE


@pytest.mark.smoke
def test_digest_stamp_is_machine_owned(tmp_path: Path) -> None:
    """A caller-supplied conflicting block is refused, never overwritten."""

    item = _typed_item("m_digest_forged_stamp")
    item["digest_attestation"] = {"authority": DIGEST_ATTESTATION_AUTHORITY, "rows": []}
    with pytest.raises(CheckerDispatchError, match="machine-owned and cannot be authored"):
        _envelope(item, tmp_path, "digest-forged-stamp")
    # The exact machine derivation supplied back is accepted (idempotent re-seal).
    item["digest_attestation"] = _derive_digest_attestation(item, GateKind.METADATA_BATCH)
    envelope = _envelope(item, tmp_path, "digest-reseal")
    assert envelope["items"][0]["digest_attestation"] == item["digest_attestation"]


@pytest.mark.smoke
def test_terminal_item_recipes_follow_the_terminal_producer() -> None:
    """Terminal packs bind the author result, the citation table, and source ids."""

    sources = [{"source_id": "s1", "url": "https://example.org/s1", "content_sha256": HASH}]
    preimage = [{"evidence_id": "ev-1", "source_id": "s1", "supports": ["blocked-prerequisite"]}]
    author_result = {"result_id": "r1", "kind": "BLOCKED", "payload": {"arm": "BLOCKED"}}
    author_result["result_sha256"] = stable_hash(author_result)
    item = {
        "stable_id": "m_td",
        "verified_hashes": {
            "proposal": author_result["result_sha256"],
            "source_manifest": stable_hash(sources),
            "evidence": stable_hash(preimage),
            "code": None,
            "source_to_code_map": stable_hash(["s1"]),
            "family_template": None,
        },
        "author_result": author_result,
        "source_manifest": {"manifest_sha256": stable_hash(sources), "sources": sources},
        "evidence_pack": {"identity_preimage": preimage, "checked_source_ids": ["s1"]},
        "model_dir": materialize_checker_model_dir("m_td"),
    }
    block = _derive_digest_attestation(item, GateKind.TERMINAL_DISPOSITION)
    assert block["all_valid"] is True
    assert block["unbound_keys"] == ["code", "family_template"]
    bases = {row["digest_key"]: row["basis"] for row in block["rows"]}
    assert bases == {key: DIGEST_BASIS_BY_KEY[GateKind.TERMINAL_DISPOSITION][key] for key in bases}
    # Dropping a checked source id is a mismatch on the source map, nothing else.
    drifted = deepcopy(item)
    drifted["evidence_pack"]["checked_source_ids"] = []
    statuses = {
        row["digest_key"]: row["status"]
        for row in _derive_digest_attestation(drifted, GateKind.TERMINAL_DISPOSITION)["rows"]
    }
    assert statuses["source_to_code_map"] == DIGEST_ATTESTATION_MISMATCH
    assert all(
        status == DIGEST_ATTESTATION_VALID
        for key, status in statuses.items()
        if key != "source_to_code_map"
    )


def _result_item(hash_mismatches: list[Any]) -> dict[str, Any]:
    return {
        "stable_id": "m1",
        "integrity": {
            "verdict": "accurate",
            "hash_mismatches": hash_mismatches,
            "excerpt_discrepancies": [],
            "locator_failures": [],
        },
    }


def _attested_envelope_item(statuses: dict[str, str]) -> dict[str, Any]:
    rows = [
        {
            "digest_key": key,
            "basis": "staged-entry-point-bytes-v1",
            "bound": HASH,
            "recomputed": None if status == DIGEST_ATTESTATION_UNRESOLVABLE else HASH,
            "status": status,
            "detail": None,
        }
        for key, status in statuses.items()
    ]
    return {
        "stable_id": "m1",
        "verified_hashes": {key: HASH for key in statuses},
        "digest_attestation": {
            "authority": DIGEST_ATTESTATION_AUTHORITY,
            "rows": rows,
            "unbound_keys": [],
            "all_valid": all(status == DIGEST_ATTESTATION_VALID for status in statuses.values()),
        },
    }


@pytest.mark.smoke
def test_checker_hash_claim_against_a_valid_row_is_refuted() -> None:
    """Disputing a machine-verified digest is a typed checker defect."""

    envelope_item = _attested_envelope_item({"code": DIGEST_ATTESTATION_VALID})
    claim = json.dumps({"digest_key": "code", "detail": "shasum disagrees"})
    with pytest.raises(CheckerDigestAuthorityError, match="machine-owned"):
        _validate_item_digest_authority(_result_item([claim]), envelope_item, stable_id="m1")
    # The decoded-mapping form is adjudicated identically.
    with pytest.raises(CheckerDigestAuthorityError):
        _validate_item_digest_authority(
            _result_item([{"digest_key": "code"}]), envelope_item, stable_id="m1"
        )
    # An empty array on an all-valid item is the only conformant shape.
    _validate_item_digest_authority(_result_item([]), envelope_item, stable_id="m1")


@pytest.mark.smoke
def test_checker_hash_claim_citing_an_unresolvable_row_is_licensed() -> None:
    """A non-valid row is exactly what the checker may cite."""

    envelope_item = _attested_envelope_item(
        {"code": DIGEST_ATTESTATION_UNRESOLVABLE, "proposal": DIGEST_ATTESTATION_VALID}
    )
    _validate_item_digest_authority(
        _result_item([json.dumps({"digest_key": "code"})]), envelope_item, stable_id="m1"
    )
    with pytest.raises(CheckerDigestAuthorityError):
        _validate_item_digest_authority(
            _result_item([json.dumps({"digest_key": "proposal"})]), envelope_item, stable_id="m1"
        )


@pytest.mark.smoke
def test_legacy_envelope_keeps_hash_mismatches_checker_owned() -> None:
    """No stamp, no authority: recorded history is never reinterpreted."""

    legacy = {"stable_id": "m1", "verified_hashes": {"code": HASH}}
    _validate_item_digest_authority(
        _result_item([json.dumps({"digest_key": "code"})]), legacy, stable_id="m1"
    )


@pytest.mark.smoke
def test_wrapper_prompt_names_the_authority_only_on_attested_envelopes(tmp_path: Path) -> None:
    """The dynamic region cites the block; the frozen prompt identity does not move."""

    from menagerie.crawler.operator_checker import PROMPT_PATH, _build_prompt

    frozen_sha = hash_bytes(PROMPT_PATH.read_bytes())
    item = _typed_item("m_digest_prompt")
    envelope = _envelope(item, tmp_path, "digest-prompt")
    request_path = tmp_path / "digest-prompt" / "request.json"
    attested = _build_prompt(envelope, request_path)
    assert "DIGEST VERIFICATION IS MACHINE-OWNED ON THIS ENVELOPE" in attested
    assert DIGEST_ATTESTATION_AUTHORITY in attested
    assert attested.startswith(PROMPT_PATH.read_text(encoding="utf-8"))

    legacy = deepcopy(envelope)
    for legacy_item in legacy["items"]:
        legacy_item.pop("digest_attestation")
    legacy.pop("digest_authority")
    legacy.pop("envelope_sha256")
    legacy = {**legacy, "envelope_sha256": stable_hash(legacy)}
    assert "DIGEST VERIFICATION IS MACHINE-OWNED" not in _build_prompt(legacy, request_path)
    assert hash_bytes(PROMPT_PATH.read_bytes()) == frozen_sha
