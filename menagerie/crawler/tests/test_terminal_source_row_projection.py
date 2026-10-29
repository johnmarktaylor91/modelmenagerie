"""B3 (step-100 triage): terminal bookkeeping source rows vs the model.v3 schema.

m_145cbbe58c341e16748e (ERNIE 2.0): the terminal assembler retained the frozen
source-manifest rows for a BLOCKED verdict, but only two of the three
machine-written row shapes had a projection to the closed public ``source``
shape. The routed-environment-inventory row (``source_kind =
routed-environment-inventory-v1``) passed through with its wide internal
custody fields (``basis``, ``fetched_bytes_len``, ``resolved_export_sha256``,
``retrieval_status``, ``routed_environment_intent``, ``source_kind``) and
without ``byte_count`` -- ``menagerie.crawler.model.v3`` rejected the record
with 9 violations, terminal bookkeeping refused, and the disposition jammed
into human review.

model.v3 already defined the closed target projection (``kind:
environment-inventory``, URN url, ``revision_kind:
routed-environment-inventory-sha256``); the writer never produced it. This
module is the round-trip CI fixture the triage prescribed: a terminal
disposition assembled over ALL THREE machine row shapes must validate against
model.v3 exactly as the ledger write path validates it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.author_dispatch import (
    AuthorResultBinding,
    BlockedRecommendation,
)
from menagerie.crawler.constants import (
    MODEL_SCHEMA_VERSION_V3,
    EnvironmentPhase,
)
from menagerie.crawler.driver_contracts import AuthorArtifact, DriverConfig, WorkItem

# Importing the driver installs ``driver_models``' dependency table; without it
# the terminal assembler raises. Depend on it explicitly rather than on
# conftest import order, so this module is runnable on its own.
from menagerie.crawler import driver as _driver  # noqa: F401
from menagerie.crawler.driver_models import _assemble_terminal_model
from menagerie.crawler.environment_inventory import (
    ROUTED_INVENTORY_SOURCE_ID,
    ROUTED_INVENTORY_SOURCE_KIND,
)
from menagerie.crawler.artifact_transactions import ArtifactAuthorityRecord
from menagerie.crawler.authority import (
    DependencyState,
    DependencyVector,
    dependency_vector_projection,
    derive_family_authority,
    family_authority_projection,
)
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.intake import IntakeItem
from menagerie.crawler.routing import IntentRoute
from menagerie.crawler.schema import validate_payload
from menagerie.crawler.tests.conftest import HASH, make_authority_context

pytestmark = pytest.mark.smoke

_CREATED_AT = "2026-08-12T00:00:00Z"
_STABLE_ID = "m_145cbbe58c341e1674aa"

_DISCOVERY_BYTES = b'{"arm":"blocked","conclusion":"prerequisite unproven"}'
_PAPER_BYTES = b"ERNIE 2.0: a continual pre-training framework for language understanding\n"
_INVENTORY_BYTES = (
    b'{"record_version":"menagerie.crawler.routed-environment-inventory.v1",'
    b'"intent":"core","packages":[{"name":"torch","version":"2.4.1"}]}'
)


def _work_item() -> WorkItem:
    """Return the routed work item behind the terminal record."""

    intake = IntakeItem(
        stable_id=_STABLE_ID,
        name="ERNIE 2.0",
        zoo="paddle",
        variant="base",
        discovery_source="crawl_roster",
        legacy_row_sha256=HASH,
        preserved_legacy_flags=(),
        variant_scope="standalone",
        family_representative_id=_STABLE_ID,
    )
    return WorkItem(
        intake=intake,
        route=IntentRoute(stable_id=_STABLE_ID, intent="core", phase=EnvironmentPhase.PYTORCH),
    )


def _search_evidence() -> dict[str, Any]:
    """Return one bounded stage-1 search summary."""

    return {
        "queries": ["ERNIE 2.0 continual pre-training reference implementation"],
        "places": ["github.com/PaddlePaddle/ERNIE"],
        "candidate_links": [{"url": "https://github.com/PaddlePaddle/ERNIE"}],
        "languages": ["en", "zh"],
        "conclusion": "Source located; a claimed prerequisite remained unproven.",
    }


def _source_manifest(tmp_path: Path) -> dict[str, Any]:
    """Freeze a manifest carrying ALL THREE machine-written row shapes.

    The ERNIE terminal failed exactly because the retained set was wider than
    the projected set: discovery-evidence and broker rows projected, the
    routed-inventory row leaked its internal shape. All three ride together
    here so the round-trip cannot pass while ANY machine shape lacks its
    projection.
    """

    cas_root = tmp_path / "source-cas"
    cas_root.mkdir(parents=True, exist_ok=True)
    discovery_digest = hash_bytes(_DISCOVERY_BYTES)
    paper_digest = hash_bytes(_PAPER_BYTES)
    inventory_digest = hash_bytes(_INVENTORY_BYTES)
    for digest, content in (
        (discovery_digest, _DISCOVERY_BYTES),
        (paper_digest, _PAPER_BYTES),
        (inventory_digest, _INVENTORY_BYTES),
    ):
        (cas_root / f"{digest.removeprefix('sha256:')}.source").write_bytes(content)
    discovery_row: dict[str, Any] = {
        "source_id": f"discovery-evidence-{discovery_digest.removeprefix('sha256:')[:16]}",
        "url": f"urn:menagerie:source-discovery:{discovery_digest.removeprefix('sha256:')}",
        "revision": discovery_digest,
        "content_sha256": discovery_digest,
        "fetched_bytes_len": len(_DISCOVERY_BYTES),
        "retrieval_status": "machine-derived",
        "media_type": "application/json",
        "cas_path": str(cas_root / f"{discovery_digest.removeprefix('sha256:')}.source"),
        "source_kind": "discovery-evidence-v1",
        "discovery_arm": "blocked",
        "search_evidence": _search_evidence(),
        "retained_vague_text": None,
        "candidate_probe_receipts": None,
        "candidate_probes": [
            {
                "identifier_kind": None,
                "identifier": None,
                "locator": "https://github.com/PaddlePaddle/ERNIE",
                "attempted_at": _CREATED_AT,
                "probe_outcome": "reachable",
                "http_status": 200,
                "author_claimed_class": "access-barrier",
            }
        ],
    }
    broker_row: dict[str, Any] = {
        "source_id": "broker-ernie-repo",
        "broker_role": "implementation",
        "url": "https://github.com/PaddlePaddle/ERNIE",
        "final_url": "https://github.com/PaddlePaddle/ERNIE",
        "revision": paper_digest,
        "content_sha256": paper_digest,
        "fetched_bytes_len": len(_PAPER_BYTES),
        "retrieval_status": "fetched",
        "media_type": "text/html",
        "requested_repo": True,
        "basis": "stage-1 candidate link dereference",
        "cas_path": str(cas_root / f"{paper_digest.removeprefix('sha256:')}.source"),
    }
    # The exact wide shape environment_inventory.routed_inventory_source_row
    # mints (ERNIE's sources[17]): every unexpected-property name from the gate
    # diagnostic rides here, and byte_count is spelled fetched_bytes_len.
    inventory_row: dict[str, Any] = {
        "source_id": ROUTED_INVENTORY_SOURCE_ID,
        "url": (
            f"urn:menagerie:routed-environment-inventory:{inventory_digest.removeprefix('sha256:')}"
        ),
        "revision": inventory_digest,
        "content_sha256": inventory_digest,
        "fetched_bytes_len": len(_INVENTORY_BYTES),
        "retrieval_status": "machine-derived",
        "retrieved_at": _CREATED_AT,
        "media_type": "application/json",
        "cas_path": str(cas_root / f"{inventory_digest.removeprefix('sha256:')}.source"),
        "source_kind": ROUTED_INVENTORY_SOURCE_KIND,
        "routed_environment_intent": "core",
        "resolved_export_sha256": HASH,
        "basis": (
            "Machine-materialized routed-environment inventory: the exact "
            "resolved package export the R1_LIBRARY availability judgment for "
            "this model is adjudicated against."
        ),
    }
    rows = [discovery_row, broker_row, inventory_row]
    return {"manifest_sha256": stable_hash(rows), "sources": rows}


def _blocked_artifact(tmp_path: Path) -> AuthorArtifact:
    """Stage one BLOCKED terminal artifact retaining every manifest row."""

    raw_fields = {
        "result_id": "result-blocked",
        "result_sha256": HASH,
        "stable_id": _STABLE_ID,
        "work_id": f"work-{_STABLE_ID}",
        "campaign_id": f"work-{_STABLE_ID}",
        "author_identity": HASH,
        "prompt_identity": HASH,
        "dispatcher_identity": HASH,
        "source_manifest_identity": HASH,
        "intake_snapshot_id": "intake-1",
        "intake_snapshot_sha256": HASH,
        "intake_item_sha256": HASH,
        "created_at": _CREATED_AT,
    }
    result = BlockedRecommendation(
        binding=AuthorResultBinding(
            raw_result={**raw_fields, "kind": "BLOCKED", "payload": {}}, **raw_fields
        ),
        stage="source",
        reason_code="missing-material-source",
        prerequisite_ids=("prereq-weights",),
        evidence_ids=("ev-blocked-1",),
        evidence_identity=HASH,
        license_identity=HASH,
        recommendation_sha256=HASH,
        research_summary=_search_evidence(),
    )
    return AuthorArtifact(
        author_result=result,
        source_manifest=_source_manifest(tmp_path),
        model_dir=tmp_path / "model",
    )


def _assemble(tmp_path: Path) -> dict[str, Any]:
    """Run the real terminal assembler over the three-shape manifest."""

    return _assemble_terminal_model(
        _work_item(),
        _blocked_artifact(tmp_path),
        "failed:source",
        "missing-material-source",
        None,
        [],
        [],
        DriverConfig(),
        _CREATED_AT,
        human_review=False,
        root_cause_fingerprint=HASH,
        terminal_gate_obtained=True,
    )


def _reducer_envelope(model: dict[str, Any]) -> dict[str, Any]:
    """Stamp the reducer-owned envelope fields with their REAL producers.

    ``prepare_model`` derives these four fields from ledger authority; the
    drift under test lives entirely in the assembler's authored/observed body,
    so the envelope uses the same producer classes with their explicit
    not-applicable states rather than a hand-written imitation of their shape.
    """

    not_applicable = DependencyState.NOT_APPLICABLE
    vector = DependencyVector(
        intake_snapshot_id=not_applicable,
        intake_snapshot_sha256=not_applicable,
        intake_item_sha256=not_applicable,
        author_result_schema_identity=not_applicable,
        author_dispatcher_identity=not_applicable,
        author_prompt_identity=not_applicable,
        checker_prompt_identity=not_applicable,
        terminal_rule_identity=not_applicable,
        status_proof_identity=not_applicable,
        source_manifest_identity=not_applicable,
        proposal_identity=not_applicable,
        author_result_identity=not_applicable,
        checker_gate_identity=not_applicable,
        recipe_revision=not_applicable,
        runner_identity=not_applicable,
        award_closure_identity=not_applicable,
        environment_generation=not_applicable,
        accepted_attempt_ids=(),
        artifact_transaction_id=not_applicable,
        artifact_claim_ids=(),
        representative_revision=not_applicable,
        publication_policy_identity=not_applicable,
    )
    family = derive_family_authority(make_authority_context([_STABLE_ID]), _STABLE_ID)
    return {
        **model,
        "record_seq": 1,
        "record_revision": HASH,
        "artifact_authority": ArtifactAuthorityRecord.not_applicable().to_payload(),
        "dependency_vector": dependency_vector_projection(vector),
        "family_authority": family_authority_projection(family),
    }


def test_terminal_disposition_round_trips_model_v3_validation(tmp_path: Path) -> None:
    """The assembled terminal record passes the exact ledger-write validation.

    Pre-fix, ``validate_payload`` -- the very check the driver's terminal
    bookkeeping runs -- raised ``menagerie.crawler.model.v3 validation failed``
    with the ERNIE unexpected-property set for ``source_resolution.sources``,
    and the disposition jammed into human review.
    """

    record = _reducer_envelope(_assemble(tmp_path))
    validate_payload(record, MODEL_SCHEMA_VERSION_V3)


def test_routed_inventory_row_projects_to_the_closed_public_shape(tmp_path: Path) -> None:
    """The inventory row is recorded via its schema-anticipated projection."""

    model = _assemble(tmp_path)
    sources = model["source_resolution"]["sources"]
    by_id = {str(row["source_id"]): row for row in sources}
    assert set(by_id) == {
        ROUTED_INVENTORY_SOURCE_ID,
        "broker-ernie-repo",
        f"discovery-evidence-{hash_bytes(_DISCOVERY_BYTES).removeprefix('sha256:')[:16]}",
    }
    inventory = by_id[ROUTED_INVENTORY_SOURCE_ID]
    assert inventory["kind"] == "environment-inventory"
    assert inventory["revision_kind"] == "routed-environment-inventory-sha256"
    assert inventory["byte_count"] == len(_INVENTORY_BYTES)
    assert inventory["retrieved_at"] == _CREATED_AT
    assert inventory["url"].startswith("urn:menagerie:routed-environment-inventory:")
    # None of the wide internal custody fields may reach the public record.
    leaked = {
        "basis",
        "fetched_bytes_len",
        "resolved_export_sha256",
        "retrieval_status",
        "routed_environment_intent",
        "source_kind",
        "cas_path",
    } & set(inventory)
    assert not leaked, leaked

    broker = by_id["broker-ernie-repo"]
    assert broker["kind"] == "repository"
    assert broker["byte_count"] == len(_PAPER_BYTES)
    assert "broker_role" not in broker and "basis" not in broker
